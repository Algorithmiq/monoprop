// Copyright 2026 Algorithmiq
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <variant>

#include "TestUtilities.h"

#include "monoprop/MPFunctions.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/graph/MPGraphLayers.h"

using namespace test_utils;
using namespace monoprop;

namespace {

constexpr size_t kNumModes = 8;

// Mirrors the streaming provider the pare functional uses: fold the operator's inverted index,
// truncated to each layer's scaled_count.
template <size_t NumModes>
auto recompute_cos(const monoprop::detail::InvertedIndex<NumModes> &inverted_index, const LayerTraversal &layer)
    -> CosMask {
    Monomial<NumModes> gen{};
    const auto &gw = layer.generator_words();
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    const auto combined = monoprop::detail::make_fold_cache<NumModes>(inverted_index,
                                                                      gen,
                                                                      layer.scaled_count(),
                                                                      monoprop::Basis::Majorana);
    return monoprop::detail::fold_to_cos_mask<NumModes>(combined);
}

} // namespace

// The streaming pare sweep must engage pruned_cos on exactly the layers whose cos loses an index.
BOOST_AUTO_TEST_CASE(pare_graph_emits_expected_layer_kinds) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};

    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const auto &graph = sim.graph();
    const auto &inverted_index = sim.mp_op().inverted_index();
    const VecD state = sim.mp_op().materialize_state();
    BOOST_REQUIRE(state.size() > 0);

    // materialize_state() hands back a caller-owned vector and caches nothing on the operator, so
    // state_coeffs stays empty and the sparse entry count must equal the dense vector's nonzero count.
    BOOST_CHECK(sim.mp_op().state_coeffs.empty());
    const auto sparse = sim.mp_op().sparse_state();
    BOOST_CHECK_EQUAL(sparse.rows.size(),
                      static_cast<size_t>(std::ranges::count_if(state, [](double c) { return c != 0.0; })));

    // Single-rank, mark_replayed_d_targets force-keeps every cosine index, so a real threshold prunes
    // nothing (real pruning is a multi-rank effect, covered by mpi_pare). To reach the prune path
    // deterministically, inject a synthetic cos index one past the real index space into layer 0 and
    // leave it out of the keep-set: only that layer may end up with a stored cos.
    const size_t marked_layer = 0;
    const size_t synth_index = state.size();
    const size_t local_index_count = state.size() + 1;
    const size_t synth_base = (synth_index >> 6) << 6;
    const uint64_t synth_bit = uint64_t{1} << (synth_index & 63U);

    auto provider = [&](size_t i) -> CosMask {
        CosMask cos = recompute_cos<kNumModes>(inverted_index, graph.get_layer_traversal(i));
        if (i == marked_layer) {
            bool merged = false;
            for (auto &b : cos.blocks) {
                if (b.first == synth_base) {
                    b.second |= synth_bit;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                cos.blocks.emplace_back(synth_base, synth_bit);
            }
            ++cos.total_count;
        }
        return cos;
    };

    VecZ seed;
    seed.reserve(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        seed.push_back(i);
    }

    auto pared = pare_graph(graph, seed, local_index_count, Picture::Heisenberg, MPI_COMM_SELF, provider);
    BOOST_REQUIRE_EQUAL(pared.layers(), graph.layers());

    size_t pruned_count = 0;
    for (size_t i = 0; i < pared.layers(); ++i) {
        if (const CosMask *pruned = pared.get_layer(i).pruned_cos(); pruned != nullptr) {
            ++pruned_count;
            // The synthetic index is the only one outside the keep-set, so a stored cos must be the
            // recomputed one minus exactly that index. `<=` here would also pass on a sweep that
            // dropped real indices.
            BOOST_CHECK_EQUAL(pruned->total_count + 1, provider(i).total_count);

            // Same block-mask walk graph_data() uses to turn a stored cos back into indices: the
            // decoded set must be the recomputed one with only synth_index missing, so a stored cos
            // is never read as the unpruned fold.
            const auto decode = [](const CosMask &cos) {
                VecZ inds;
                inds.reserve(cos.total_count);
                for (const auto &[base, bits] : cos.blocks) {
                    monoprop::detail::for_each_cos_index(base, bits, [&](size_t idx) { inds.push_back(idx); });
                }
                return inds;
            };
            const auto kept = decode(*pruned);
            auto expected = decode(provider(i));
            std::erase(expected, synth_index);
            BOOST_CHECK_EQUAL(kept.size(), pruned->total_count);
            BOOST_TEST(kept == expected, boost::test_tools::per_element());
        }
    }
    // Exactly the marked layer is pruned: every other layer's cos lies entirely inside the keep-set,
    // and a preserved layer stores nothing (its cos is recomputed at replay).
    BOOST_CHECK_EQUAL(pruned_count, 1u);
    BOOST_TEST(pared.get_layer(marked_layer).pruned_cos() != static_cast<const CosMask *>(nullptr));
}

BOOST_AUTO_TEST_CASE(pare_graph_energy_matches_unpared) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};

    auto sim_full = build_simulator<kNumModes>(data, cfg);
    sim_full.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto ev_full = sim_full.expectation_value_functional(std::nullopt);
    const double e_full = ev_full(data.parameters);

    // The pared and unpared replays reduce in an unpinned accumulation order, so they differ by a
    // few ULP (~1e-18 here) run-to-run: exact == is the wrong assertion. The tolerance is far tighter
    // than any real pruning effect but comfortably above that reorder noise.
    auto sim_tiny = build_simulator<kNumModes>(data, cfg);
    sim_tiny.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto ev_tiny = sim_tiny.expectation_value_functional(std::optional<double>{1e-12});
    const double e_tiny = ev_tiny(data.parameters);
    BOOST_CHECK_SMALL(std::abs(e_full - e_tiny), 1e-12);

    auto sim_real = build_simulator<kNumModes>(data, cfg);
    sim_real.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto ev_real = sim_real.expectation_value_functional(std::optional<double>{1e-10});
    const double e_real = ev_real(data.parameters);
    BOOST_CHECK_SMALL(std::abs(e_real - data.actual_expval), 1e-9);
}
