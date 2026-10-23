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
#include <complex>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/mpi/MPICompat.h"

// evolved_operator_coefficients() probes every shard's operator index with caller-supplied keys instead of
// enumerating it. Oracle throughout: evolved_operator_terms(params, 0.0), which enumerates, at the same shard count.
// Every case runs at the launch's T; cpp/tests/CMakeLists.txt also reruns the suite at T = 2 and 4.

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 8;

auto majorana_sim(const CaseData &data) -> MonomialPropagator<kNumModes> {
    return MonomialPropagator<kNumModes>(data.hamiltonian,
                                         kCutoff,
                                         data.initial_state,
                                         std::nullopt,
                                         MPI_COMM_SELF,
                                         std::nullopt,
                                         std::nullopt,
                                         CutoffType::Length,
                                         std::nullopt,
                                         kNumModes,
                                         Basis::Majorana);
}

// A propagator whose graph is built and ready to contract.
auto built_sim(const CaseData &data) -> MonomialPropagator<kNumModes> {
    auto sim = majorana_sim(data);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    return sim;
}

// A Schrodinger-picture propagator: the state's identity amplitude is an ordinary index row there,
// so the empty key resolves through the probe rather than through core_term().
auto built_schrodinger_sim(const CaseData &data) -> MonomialPropagator<kNumModes> {
    auto sim = MonomialPropagator<kNumModes>(data.hamiltonian,
                                             kCutoff,
                                             data.initial_state,
                                             std::optional<unsigned int>{kCutoff},
                                             MPI_COMM_SELF,
                                             std::nullopt,
                                             std::nullopt,
                                             CutoffType::Length,
                                             std::nullopt,
                                             kNumModes,
                                             Basis::Majorana);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    return sim;
}

auto keys_of(const std::vector<std::pair<VecZ, std::complex<double>>> &terms) -> std::vector<VecZ> {
    std::vector<VecZ> keys;
    keys.reserve(terms.size());
    for (const auto &[indices, coeff] : terms) {
        keys.push_back(indices);
    }
    return keys;
}

} // namespace

// Querying every enumerated term must reproduce the enumerated coefficients exactly: same
// contraction, same decode, only the index access differs.
BOOST_AUTO_TEST_CASE(evolved_coefficients_coefficients_match_enumerated_terms) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_sim(data);

    const auto terms = sim.evolved_operator_terms(data.parameters, 0.0);
    BOOST_REQUIRE(!terms.empty());

    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, keys_of(terms));
    BOOST_REQUIRE_EQUAL(coeffs.size(), terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST_CONTEXT("term " << i) {
            BOOST_TEST(coeffs[i].real() == terms[i].second.real());
            BOOST_TEST(coeffs[i].imag() == terms[i].second.imag());
        }
    }
}

// A monomial the operator does not carry reads back as exactly 0, not as noise and not as a throw.
BOOST_AUTO_TEST_CASE(evolved_coefficients_absent_term_reads_back_zero) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_sim(data);

    const auto terms = sim.evolved_operator_terms(data.parameters, 0.0);
    std::map<VecZ, std::complex<double>> present(terms.begin(), terms.end());

    // Search the weight-2 monomials for one the evolved operator misses; the cutoff makes some
    // absent for any realistic case, but assert we actually found one rather than trusting it.
    std::optional<VecZ> absent;
    for (size_t i = 0; i < 2 * kNumModes && !absent; ++i) {
        for (size_t j = i + 1; j < 2 * kNumModes; ++j) {
            const VecZ candidate{i, j};
            if (!present.contains(candidate)) {
                absent = candidate;
                break;
            }
        }
    }
    BOOST_REQUIRE_MESSAGE(absent.has_value(), "no absent weight-2 monomial to probe");

    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, {*absent});
    BOOST_REQUIRE_EQUAL(coeffs.size(), 1u);
    BOOST_TEST(coeffs[0].real() == 0.0);
    BOOST_TEST(coeffs[0].imag() == 0.0);
}

// The result is positional: out[q] belongs to keys[q] whatever order the keys arrive in.
BOOST_AUTO_TEST_CASE(evolved_coefficients_query_order_is_preserved) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_sim(data);

    const auto terms = sim.evolved_operator_terms(data.parameters, 0.0);
    BOOST_REQUIRE(terms.size() > 4);

    std::vector<size_t> order(terms.size());
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(1234);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<VecZ> keys;
    keys.reserve(order.size());
    for (const auto &i : order) {
        keys.push_back(terms[i].first);
    }

    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, keys);
    BOOST_REQUIRE_EQUAL(coeffs.size(), order.size());
    for (size_t q = 0; q < order.size(); ++q) {
        BOOST_TEST_CONTEXT("query " << q << " -> term " << order[q]) {
            BOOST_TEST(coeffs[q].real() == terms[order[q]].second.real());
            BOOST_TEST(coeffs[q].imag() == terms[order[q]].second.imag());
        }
    }
}

// The keys are user input, so the checked encode is mandatory: an out-of-range slot must throw
// rather than write past the monomial (indices_to_bitset is noexcept and would corrupt memory).
BOOST_AUTO_TEST_CASE(evolved_coefficients_out_of_range_slot_index_throws) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_sim(data);

    const VecZ too_large{2 * kNumModes};
    BOOST_CHECK_THROW(sim.evolved_operator_coefficients(data.parameters, {too_large}), AlgebraIndexOutOfRange);
}

// An empty query list is a no-op, not a degenerate probe.
BOOST_AUTO_TEST_CASE(evolved_coefficients_empty_query_list_returns_empty) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_sim(data);

    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, {});
    BOOST_TEST(coeffs.empty());
}

// evolved_operator_terms() excludes the core term and the binder re-adds it under the empty key, so
// the empty query has to resolve to core_term() for the two APIs to agree term for term. The
// fixture Hamiltonian has no identity term, so inject one -- otherwise core_term() is 0 and the
// check cannot tell the core term apart from the absent-term answer.
BOOST_AUTO_TEST_CASE(evolved_coefficients_empty_term_resolves_to_core_term) {
    auto data = load_case_data<kNumModes>("random_exact.msgpack");
    data.hamiltonian[VecZ{}] = std::complex{0.75, 0.0};
    auto sim = built_sim(data);

    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, {VecZ{}});
    BOOST_REQUIRE_EQUAL(coeffs.size(), 1u);
    BOOST_TEST(coeffs[0].real() == 0.75);
    BOOST_TEST(coeffs[0].imag() == 0.0);
    BOOST_TEST(coeffs[0].real() == sim.core_term());

    // The identity is not an index row in the Heisenberg picture, so the enumeration never sees it
    // and the two APIs would otherwise disagree on exactly this key.
    const auto terms = sim.evolved_operator_terms(data.parameters, 0.0);
    BOOST_TEST(std::none_of(terms.begin(), terms.end(), [](const auto &term) { return term.first.empty(); }));
}

// The Schrodinger picture is the motivating case: reading a handful of amplitudes out of an evolved
// state. There is no core term to divert the identity there -- it is an ordinary index row carrying
// the state's identity amplitude -- so the empty query resolves through the probe like any other.
BOOST_AUTO_TEST_CASE(evolved_coefficients_schrodinger_coefficients_match_enumerated_terms) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_schrodinger_sim(data);

    const auto terms = sim.evolved_operator_terms(data.parameters, 0.0);
    BOOST_REQUIRE(!terms.empty());

    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, keys_of(terms));
    BOOST_REQUIRE_EQUAL(coeffs.size(), terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST_CONTEXT("term " << i) {
            BOOST_TEST(coeffs[i].real() == terms[i].second.real());
            BOOST_TEST(coeffs[i].imag() == terms[i].second.imag());
        }
    }

    // The enumeration above covers the identity only because the state carries it; pin that, so a
    // regression that started diverting it to the core term would be caught here.
    const auto identity = std::find_if(terms.begin(), terms.end(), [](const auto &term) { return term.first.empty(); });
    BOOST_REQUIRE_MESSAGE(identity != terms.end(), "the evolved state has no identity amplitude to pin");
    const auto probed = sim.evolved_operator_coefficients(data.parameters, {VecZ{}});
    BOOST_TEST(probed.at(0).real() == identity->second.real());
    BOOST_TEST(probed.at(0).imag() == identity->second.imag());
}

// The query is a list, not a set: a repeated key is answered once per occurrence. Worth pinning because the probe
// visits every shard, so a key counted twice would show up here first.
BOOST_AUTO_TEST_CASE(evolved_coefficients_repeated_key_is_answered_once_per_occurrence) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = built_sim(data);
    const auto terms = sim.evolved_operator_terms(data.parameters, 0.0);
    BOOST_REQUIRE(!terms.empty());
    const auto &key = terms.front().first;
    const auto expected = terms.front().second;
    const auto coeffs = sim.evolved_operator_coefficients(data.parameters, {key, VecZ{}, key});
    BOOST_REQUIRE_EQUAL(coeffs.size(), 3u);
    BOOST_TEST(coeffs[0].real() == expected.real());
    BOOST_TEST(coeffs[0].imag() == expected.imag());
    BOOST_TEST(coeffs[2] == coeffs[0]);
}
