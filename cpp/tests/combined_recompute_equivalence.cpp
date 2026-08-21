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

// The live recompute path (make_lazy_fold + scale_cos_lazy / accumulate_cos_lazy) must agree
// bit-for-bit with the materialised-fold oracle (make_fold_cache + the scale_cos_cached /
// accumulate_cos_cached replays below), on every layer of a real propagated operator.

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
// These oracles cover the Majorana fold only; the Pauli J(G) fold generator has no equivalence test yet.
constexpr auto kBasis = Basis::Majorana;

template <size_t NumModes>
auto generator_of(const LayerTraversal &layer) -> Monomial<NumModes> {
    Monomial<NumModes> gen{};
    const auto &gw = layer.generator_words();
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    return gen;
}

// Reference oracle (test-only): replay a materialised FoldCache buffer, which the live path never does.
template <size_t NumModes>
void scale_cos_cached(const monoprop::detail::FoldCache<NumModes> &p, double *coeff, double cos_val) {
    const size_t mask_words = p.fold.mask_words;
    for (size_t wi = 0; wi < mask_words; ++wi) {
        monoprop::detail::for_each_cos_index(wi * 64, monoprop::detail::fold_word<NumModes>(p, wi), [&](size_t i) {
            coeff[i] *= cos_val;
        });
    }
}

template <size_t NumModes>
double accumulate_cos_cached(const monoprop::detail::FoldCache<NumModes> &p,
                             double *state,
                             double *ham,
                             double cos_val,
                             double sec_val) {
    const size_t mask_words = p.fold.mask_words;
    double loc = 0.0;
    for (size_t wi = 0; wi < mask_words; ++wi) {
        monoprop::detail::for_each_cos_index(wi * 64, monoprop::detail::fold_word<NumModes>(p, wi), [&](size_t i) {
            loc += state[i] * ham[i];
            ham[i] *= sec_val;
            state[i] *= cos_val;
        });
    }
    return loc;
}

} // namespace

// scale: coeff[i] *= cos over the layer's cosine index set — a pure per-index scatter, so the two
// paths must produce byte-identical arrays.
BOOST_AUTO_TEST_CASE(combined_scale_cache_equals_recompute) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const auto &inverted_index = sim.mp_op().inverted_index();
    const auto &graph = sim.graph();
    const size_t n = sim.mp_op().size();
    BOOST_REQUIRE(n > 0);

    // Distinct, non-degenerate coefficients so a missed/extra index shows up.
    std::vector<double> baseline(n);
    for (size_t i = 0; i < n; ++i) {
        baseline[i] = 1.0 + static_cast<double>(i) * 1e-3;
    }
    const double cos_val = 0.6234;

    size_t odd_layers = 0;
    for (size_t li = 0; li < graph.layers(); ++li) {
        const auto layer = graph.get_layer_traversal(li);
        if (layer.generator_words().empty()) {
            continue;
        }
        const auto gen = generator_of<kNumModes>(layer);
        if (gen.count() % 2 != 0) {
            ++odd_layers;
        }

        auto prepared = monoprop::detail::make_fold_cache<kNumModes>(inverted_index, gen, layer.scaled_count(), kBasis);
        auto recipe = monoprop::detail::make_lazy_fold<kNumModes>(inverted_index, gen, layer.scaled_count(), kBasis);

        std::vector<double> a = baseline;
        std::vector<double> b = baseline;
        scale_cos_cached<kNumModes>(prepared, a.data(), cos_val);
        monoprop::detail::scale_cos_lazy<kNumModes>(inverted_index, recipe, b.data(), cos_val);

        BOOST_TEST_INFO("layer " << li);
        BOOST_TEST(std::memcmp(a.data(), b.data(), n * sizeof(double)) == 0);
    }
    // The fixture must actually exercise the odd-|G| parity correction, or the guardrail is hollow.
    BOOST_TEST(odd_layers > 0u);
}

// accumulate: the per-index state/ham mutations must be byte-identical; the returned reduction may be
// summed in a different order, so it is compared within a tight fp tolerance.
BOOST_AUTO_TEST_CASE(combined_accumulate_cache_equals_recompute) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const auto &inverted_index = sim.mp_op().inverted_index();
    const auto &graph = sim.graph();
    const size_t n = sim.mp_op().size();
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
        const auto layer = graph.get_layer_traversal(li);
        if (layer.generator_words().empty()) {
            continue;
        }
        const auto gen = generator_of<kNumModes>(layer);
        auto prepared = monoprop::detail::make_fold_cache<kNumModes>(inverted_index, gen, layer.scaled_count(), kBasis);
        auto recipe = monoprop::detail::make_lazy_fold<kNumModes>(inverted_index, gen, layer.scaled_count(), kBasis);

        std::vector<double> sa = state0, ha = ham0;
        std::vector<double> sb = state0, hb = ham0;
        const double ea = accumulate_cos_cached<kNumModes>(prepared, sa.data(), ha.data(), cos_val, sec_val);
        const double eb = monoprop::detail::accumulate_cos_lazy<kNumModes>(inverted_index,
                                                                           recipe,
                                                                           sb.data(),
                                                                           hb.data(),
                                                                           nullptr,
                                                                           cos_val,
                                                                           sec_val);

        BOOST_TEST_INFO("layer " << li);
        BOOST_TEST(std::memcmp(sa.data(), sb.data(), n * sizeof(double)) == 0);
        BOOST_TEST_INFO("layer " << li);
        BOOST_TEST(std::memcmp(ha.data(), hb.data(), n * sizeof(double)) == 0);
        BOOST_CHECK_SMALL(std::abs(ea - eb), 1e-9 * (1.0 + std::abs(ea)));
    }
}

// Lives here because it re-runs the same recompute machinery exercised above.
BOOST_FIXTURE_TEST_CASE(snapshot_invariance_repeated_evaluation, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    auto fn = sim.expectation_value_functional();
    const double e1 = fn(data.parameters);
    const double e2 = fn(data.parameters);

    BOOST_CHECK_SMALL(e1 - e2, 1e-13);
    BOOST_TEST_MESSAGE("snapshot_invariance energy=" << e1);
}

// Lifetime contract: a LazyFold outlives the index it was built from (build_cos_callbacks retains one
// per layer in a functional's closure, and a later build_graph rebuilds InvertedIndex::row_parity_),
// so it must hold no pointer into that buffer. Pins both halves — that the buffer really does move
// under growth, and that a fold built before the growth still folds like a FoldCache built after it.
BOOST_AUTO_TEST_CASE(lazy_fold_survives_operator_growth) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<kNumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    // Find an odd-|G| layer: row_parity_ is only consulted for those (Pauli and even |G| never touch it).
    const auto &graph = sim.graph();
    size_t odd_layer = graph.layers();
    for (size_t li = 0; li < graph.layers(); ++li) {
        const auto layer = graph.get_layer_traversal(li);
        if (!layer.generator_words().empty() && generator_of<kNumModes>(layer).count() % 2 != 0) {
            odd_layer = li;
            break;
        }
    }
    BOOST_REQUIRE(odd_layer < graph.layers());

    const auto layer = graph.get_layer_traversal(odd_layer);
    const auto gen = generator_of<kNumModes>(layer);
    const auto scaled_count = layer.scaled_count();

    const uint64_t *before = sim.mp_op().inverted_index().row_parity_words();
    BOOST_REQUIRE(before != nullptr);
    auto recipe = monoprop::detail::make_lazy_fold<kNumModes>(sim.mp_op().inverted_index(), gen, scaled_count, kBasis);

    // Grow the operator, forcing the index and its row parity onto fresh storage.
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const uint64_t *after = sim.mp_op().inverted_index().row_parity_words();
    BOOST_REQUIRE(after != nullptr);
    BOOST_TEST(before != after); // a pointer cached in the fold would now dangle

    const size_t n = sim.mp_op().size();
    std::vector<double> baseline(n);
    for (size_t i = 0; i < n; ++i) {
        baseline[i] = 1.0 + static_cast<double>(i) * 1e-3;
    }
    const double cos_val = 0.6234;

    auto prepared =
        monoprop::detail::make_fold_cache<kNumModes>(sim.mp_op().inverted_index(), gen, scaled_count, kBasis);
    std::vector<double> expected = baseline;
    std::vector<double> actual = baseline;
    scale_cos_cached<kNumModes>(prepared, expected.data(), cos_val);
    monoprop::detail::scale_cos_lazy<kNumModes>(sim.mp_op().inverted_index(), recipe, actual.data(), cos_val);

    BOOST_TEST(std::memcmp(expected.data(), actual.data(), n * sizeof(double)) == 0);
}
