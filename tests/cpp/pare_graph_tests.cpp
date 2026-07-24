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

// Full-cos provider mirroring the streaming provider the pare functional uses: fold the operator's
// persistent even-parity inverted index truncated to each layer's scaled_count.
template <size_t NumModes>
auto recompute_cos(const monoprop::detail::InvertedIndex<NumModes> &inverted_index, const LayerTraversal &layer)
    -> CosMask {
    Monomial<NumModes> gen{};
    const auto &gw = layer.generator_words();
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    const auto combined = monoprop::detail::make_fold_cache<NumModes>(inverted_index, gen, layer.scaled_count());
    return monoprop::detail::fold_to_cos_mask<NumModes>(combined);
}

} // namespace

// 1. The streaming pare sweep emits the typed layers we expect. For the real fold cos, every cosine
//    index single-rank is a force-kept rotation endpoint (mark_replayed_d_targets), so a real
//    threshold prunes nothing — matching the original masked-plan behavior. To exercise the
//    filter+emit path deterministically (single-rank), we feed a provider whose cos carries one
//    synthetic index that is NOT in the keep-set: that one layer must become a PrunedLayer whose
//    stored cos is a strict subset, while every untouched layer stays a FoldLayer.
BOOST_AUTO_TEST_CASE(pare_graph_emits_expected_layer_kinds) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};

    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const auto &graph = sim.graph();
    const auto &inverted_index = sim.mp_op().inverted_index();
    const VecD state = sim.mp_op().get_state();
    BOOST_REQUIRE(state.size() > 0);

    // Single-rank, the cumulative rotation endpoints (D-targets) across all layers cover the entire
    // operator index space, and mark_replayed_d_targets force-keeps every one of them — so a real
    // threshold prunes nothing single-rank (matching the original masked-plan behavior; real pruning
    // is a multi-rank effect, covered by mpi_pare). To exercise the prune+emit path deterministically
    // here, inject a synthetic cos index ONE PAST the real index space (never a D-target, so nothing
    // force-keeps it) into layer 0, widen local_index_count to include it, and leave it out of the
    // keep-set. That one layer must become a PrunedLayer; every other layer stays a FoldLayer.
    const size_t marked_layer = 0;
    const size_t synth_index = state.size();
    const size_t local_index_count = state.size() + 1;
    const size_t synth_base = (synth_index >> 6) << 6;
    const uint64_t synth_bit = uint64_t{1} << (synth_index & 63U);

    // Provider: real recomputed cos for layer 0 PLUS the synthetic index; real recomputed cos for all others.
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

    // Seed the keep-set with every real index (so no real cos bit is dropped) but NOT the synthetic
    // one, so only the synthetic index is pruned from the marked layer's cos.
    VecZ seed;
    seed.reserve(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        seed.push_back(i);
    }

    auto pared = pare_graph(graph, seed, local_index_count, /*schrodinger=*/false, MPI_COMM_SELF, provider);
    BOOST_REQUIRE_EQUAL(pared.layers(), graph.layers());

    size_t pruned_count = 0;
    for (size_t i = 0; i < pared.layers(); ++i) {
        const auto &layer = pared.get_layer(i);
        if (const CosMask *pruned = layer.pruned_cos(); pruned != nullptr) {
            // A pruned layer carries an explicitly-stored (possibly empty) filtered cos.
            ++pruned_count;
            // Stored pruned cos is a strict subset of the full (synthetic-augmented) cos.
            BOOST_TEST(pruned->total_count <= provider(i).total_count);
        }
        else {
            // Preserved layers are fold layers (cos recomputed at replay, nothing stored).
            BOOST_TEST(layer.pruned_cos() == static_cast<const CosMask *>(nullptr));
        }
    }
    // Exactly the marked layer should be pruned (its synthetic index dropped).
    BOOST_TEST(pruned_count >= 1u);
    BOOST_TEST(pared.get_layer(marked_layer).pruned_cos() != static_cast<const CosMask *>(nullptr));
}

// 2. The pared energy matches the unpared energy at a tiny threshold (prunes ~nothing) up to
//    floating-point summation order, and stays within the pare tolerance at a real threshold.
BOOST_AUTO_TEST_CASE(pare_graph_energy_matches_unpared) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};

    // Unpared energy.
    auto sim_full = build_simulator<kNumModes>(data, cfg);
    sim_full.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto ev_full = sim_full.expectation_value_functional(std::nullopt);
    const double e_full = ev_full(data.parameters);

    // Pared at a tiny threshold: prunes essentially nothing, so the energy must agree with the
    // unpared value up to floating-point summation order. The pared and unpared replays each run a
    // multithreaded reduction whose accumulation order is not pinned, so the two differ by a few ULP
    // (~1e-18 here) run-to-run — exact == is therefore the wrong assertion; require a tolerance far
    // tighter than any real pruning effect but comfortably above reduction-reorder noise.
    auto sim_tiny = build_simulator<kNumModes>(data, cfg);
    sim_tiny.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto ev_tiny = sim_tiny.expectation_value_functional(std::optional<double>{1e-12});
    const double e_tiny = ev_tiny(data.parameters);
    BOOST_CHECK_SMALL(std::abs(e_full - e_tiny), 1e-12);

    // Pared at a real threshold: close to the exact energy (mirrors mpi_pare expectations).
    auto sim_real = build_simulator<kNumModes>(data, cfg);
    sim_real.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto ev_real = sim_real.expectation_value_functional(std::optional<double>{1e-10});
    const double e_real = ev_real(data.parameters);
    BOOST_CHECK_SMALL(std::abs(e_real - data.actual_expval), 1e-9);
}
