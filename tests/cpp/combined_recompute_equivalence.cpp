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

// Guardrail: the two ways a layer's cosine set is applied must agree bit-for-bit.
//   - CACHE path:     make_fold_cache + scale_cos_cached / accumulate_cos_cached
//   - RECOMPUTE path: make_lazy_fold   + scale_cos_lazy / accumulate_cos_lazy
// The functional switches between them on a memory budget (recompute_cache_budget_bytes) and relies on
// them being bit-identical. This pins that claim directly on every layer of a real propagated
// operator, so a refactor of the shared word-scan cannot silently diverge the two paths.

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <cstring>
#include <vector>

#include "TestUtilities.h"

#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/graph/MPGraphLayers.h"

using namespace test_utils;
using namespace monoprop;

namespace {

constexpr size_t kNumModes = 8;

template <size_t NumModes>
auto generator_of(const Layer &layer) -> MajoranaSet<NumModes> {
    MajoranaSet<NumModes> gen{};
    const auto &gw = layer.generator_words();
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    return gen;
}

} // namespace

// scale: coeff[i] *= cos over the layer's cosine index set. A pure per-index scatter, so the cache
// and recompute paths must produce byte-identical arrays regardless of thread scheduling.
BOOST_AUTO_TEST_CASE(combined_scale_cache_equals_recompute) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const auto &inverted_index = sim.mp_op().inverted_index();
    const auto &graph = sim.graph();
    const size_t n = sim.mp_op().get_state().size();
    BOOST_REQUIRE(n > 0);

    // Distinct, non-degenerate coefficients so a missed/extra index shows up.
    std::vector<double> baseline(n);
    for (size_t i = 0; i < n; ++i) {
        baseline[i] = 1.0 + static_cast<double>(i) * 1e-3;
    }
    const double cos_val = 0.6234;

    size_t odd_layers = 0;
    for (size_t li = 0; li < graph.layers(); ++li) {
        const auto &layer = graph.get_layer(li);
        if (layer.generator_words().empty()) {
            continue;
        }
        const auto gen = generator_of<kNumModes>(layer);
        if (gen.count() % 2 != 0) {
            ++odd_layers;
        }

        auto prepared = monoprop::detail::make_fold_cache<kNumModes>(inverted_index, gen, layer.scaled_count());
        auto recipe = monoprop::detail::make_lazy_fold<kNumModes>(inverted_index, gen, layer.scaled_count());

        std::vector<double> a = baseline;
        std::vector<double> b = baseline;
        monoprop::detail::scale_cos_cached<kNumModes>(prepared, a.data(), cos_val);
        monoprop::detail::scale_cos_lazy<kNumModes>(inverted_index, recipe, b.data(), cos_val);

        BOOST_TEST_INFO("layer " << li);
        BOOST_TEST(std::memcmp(a.data(), b.data(), n * sizeof(double)) == 0);
    }
    // The fixture must actually exercise the odd-|G| parity correction, or the guardrail is hollow.
    BOOST_TEST(odd_layers > 0u);
}

// accumulate: reads state*ham into an energy term and mutates state/ham per index. The array
// mutations are per-index (order-independent) so must be byte-identical; the returned reduction is
// summed in a possibly different order, so compare it within a tight fp tolerance.
BOOST_AUTO_TEST_CASE(combined_accumulate_cache_equals_recompute) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const auto &inverted_index = sim.mp_op().inverted_index();
    const auto &graph = sim.graph();
    const size_t n = sim.mp_op().get_state().size();
    BOOST_REQUIRE(n > 0);

    std::vector<double> state0(n);
    std::vector<double> ham0(n);
    for (size_t i = 0; i < n; ++i) {
        state0[i] = 0.5 + static_cast<double>(i) * 1e-3;
        ham0[i] = 1.0 - static_cast<double>(i) * 7e-4;
    }
    const double cos_val = 0.6234;
    const double sec_val = 0.4157;

    for (size_t li = 0; li < graph.layers(); ++li) {
        const auto &layer = graph.get_layer(li);
        if (layer.generator_words().empty()) {
            continue;
        }
        const auto gen = generator_of<kNumModes>(layer);
        auto prepared = monoprop::detail::make_fold_cache<kNumModes>(inverted_index, gen, layer.scaled_count());
        auto recipe = monoprop::detail::make_lazy_fold<kNumModes>(inverted_index, gen, layer.scaled_count());

        std::vector<double> sa = state0, ha = ham0;
        std::vector<double> sb = state0, hb = ham0;
        const double ea =
            monoprop::detail::accumulate_cos_cached<kNumModes>(prepared, sa.data(), ha.data(), cos_val, sec_val);
        const double eb = monoprop::detail::accumulate_cos_lazy<kNumModes>(
            inverted_index, recipe, sb.data(), hb.data(), cos_val, sec_val);

        BOOST_TEST_INFO("layer " << li);
        BOOST_TEST(std::memcmp(sa.data(), sb.data(), n * sizeof(double)) == 0);
        BOOST_TEST_INFO("layer " << li);
        BOOST_TEST(std::memcmp(ha.data(), hb.data(), n * sizeof(double)) == 0);
        BOOST_CHECK_SMALL(std::abs(ea - eb), 1e-9 * (1.0 + std::abs(ea)));
    }
}
