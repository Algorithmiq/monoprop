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

#include <complex>
#include <optional>
#include <vector>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// term_expectation_values() reads every initial term's value off one adjoint replay. Oracles: the
// expectation value itself, which the values must reassemble, and a re-weight to a single term. LiH,
// since random_exact's values are all 0 and so cannot tell a wrong replay from a right one.

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = LihFixture::n_modes;
constexpr auto kCase = "lih_fermionic_spin_exact.msgpack";
constexpr unsigned int kFullCutoff = 2 * kNumModes;
// Truncating but keeping rotations, so a layer mixes rotation pairs with cosine-only rows whose partner
// was cut.
constexpr unsigned int kTruncatedCutoff = 4;

// The case's circuit as a stored graph, ready to query.
auto built_sim(const CaseData &data,
               unsigned int cutoff,
               size_t partitions = 1,
               std::optional<unsigned int> schrodinger_cutoff = std::nullopt) -> MonomialPropagator<kNumModes> {
    auto sim = MonomialPropagator<kNumModes>(data.hamiltonian,
                                             cutoff,
                                             data.initial_state,
                                             schrodinger_cutoff,
                                             MPI_COMM_SELF,
                                             std::nullopt,
                                             std::nullopt,
                                             CutoffType::Length,
                                             std::nullopt,
                                             kNumModes,
                                             Basis::Majorana,
                                             partitions);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    return sim;
}

// `op`'s terms in its iteration order, which reassembled() pairs the values with.
auto keys_of(const OperatorDict &op) -> std::vector<VecZ> {
    std::vector<VecZ> keys;
    for (const auto &[indices, coeff] : op) {
        keys.push_back(indices);
    }
    return keys;
}

// Σ coeff·value over `op`: the expectation value the values claim to decompose.
auto reassembled(const OperatorDict &op, const std::vector<std::complex<double>> &values) -> std::complex<double> {
    std::complex<double> total = 0.0;
    size_t q = 0;
    for (const auto &[indices, coeff] : op) {
        total += coeff * values[q++];
    }
    return total;
}

} // namespace

// The values decompose the expectation value over the initial operator, including the core term,
// which reads 1. Linearity makes this exact for the truncated graph too.
BOOST_AUTO_TEST_CASE(values_reassemble_the_expectation_value) {
    auto data = load_case_data<kNumModes>(kCase);
    data.hamiltonian[VecZ{}] = std::complex{0.75, 0.0};
    for (const auto cutoff : {kFullCutoff, kTruncatedCutoff}) {
        auto sim = built_sim(data, cutoff);
        const auto values = sim.term_expectation_values(data.parameters, keys_of(data.hamiltonian));
        const auto total = reassembled(data.hamiltonian, values);
        BOOST_TEST_CONTEXT("cutoff=" << cutoff) {
            BOOST_TEST(near(total.real(), sim.expectation_value(data.parameters)));
            BOOST_TEST(near(total.imag(), 0.0));
        }
    }
}

// Each value is what expectation_value() reports once the initial operator is that term alone.
BOOST_AUTO_TEST_CASE(each_value_matches_a_single_term_reweight) {
    const auto data = load_case_data<kNumModes>(kCase);
    auto sim = built_sim(data, kTruncatedCutoff);
    const auto keys = keys_of(data.hamiltonian);
    const auto values = sim.term_expectation_values(data.parameters, keys);

    size_t q = 0;
    for (const auto &[indices, coeff] : data.hamiltonian) {
        sim.update_initial_operator({{indices, coeff}});
        // A re-weight that omits the empty term keeps the core term, so it is taken back off.
        const double core = indices.empty() ? 0.0 : sim.core_term();
        const auto single = sim.expectation_value(data.parameters) - core;
        const auto expected = coeff * values[q];
        BOOST_TEST_CONTEXT("term " << q) {
            BOOST_TEST(near(single, expected.real()));
            BOOST_TEST(near(expected.imag(), 0.0));
        }
        ++q;
    }
}

// A row the graph created mid-circuit reads the same value expectation_value() gives once the initial
// operator is that row alone: the two share one contract, which is not that row's ⟨P⟩.
BOOST_AUTO_TEST_CASE(mid_circuit_row_matches_a_single_term_reweight) {
    const auto data = load_case_data<kNumModes>(kCase);
    auto sim = built_sim(data, kTruncatedCutoff);
    // A Hermitian coefficient for the row: its decoded evolved coefficient.
    std::optional<std::pair<VecZ, std::complex<double>>> created;
    for (const auto &term : sim.evolved_operator_terms(data.parameters, 1e-8)) {
        if (!term.first.empty() && !data.hamiltonian.contains(term.first)) {
            created = term;
            break;
        }
    }
    BOOST_REQUIRE(created.has_value());
    const auto &[indices, coeff] = *created;

    const auto expected = coeff * sim.term_expectation_values(data.parameters, {indices})[0];
    sim.update_initial_operator({{indices, coeff}});
    const auto single = sim.expectation_value(data.parameters) - sim.core_term();
    BOOST_TEST(near(single, expected.real()));
    BOOST_TEST(near(expected.imag(), 0.0));
}

// Partitions hash-split the rows, so each key resolves in one partition and the gather must not
// double-count it.
BOOST_AUTO_TEST_CASE(partitioned_values_match_single_partition) {
    const auto data = load_case_data<kNumModes>(kCase);
    const auto keys = keys_of(data.hamiltonian);
    const auto expected = built_sim(data, kTruncatedCutoff).term_expectation_values(data.parameters, keys);

    for (const size_t partitions : {size_t{2}, size_t{4}}) {
        auto sim = built_sim(data, kTruncatedCutoff, partitions);
        const auto values = sim.term_expectation_values(data.parameters, keys);
        BOOST_REQUIRE_EQUAL(values.size(), expected.size());
        for (size_t q = 0; q < expected.size(); ++q) {
            BOOST_TEST_CONTEXT("partitions=" << partitions << " query " << q) {
                BOOST_TEST(near(values[q].real(), expected[q].real()));
                BOOST_TEST(near(values[q].imag(), expected[q].imag()));
            }
        }
    }
}

// In Schrodinger the adjoint of the state is the evolved state itself, so the values are the
// conjugated state coefficients: the value is per unit of a monomial, the coefficient per unit of its
// adjoint. Untruncated, both pictures agree.
BOOST_AUTO_TEST_CASE(schrodinger_values_are_the_evolved_state_and_match_heisenberg) {
    const auto data = load_case_data<kNumModes>(kCase);
    const auto keys = keys_of(data.hamiltonian);
    auto schrodinger = built_sim(data, kFullCutoff, 1, kFullCutoff);
    const auto values = schrodinger.term_expectation_values(data.parameters, keys);
    const auto state = schrodinger.evolved_operator_coefficients(data.parameters, keys);
    const auto heisenberg = built_sim(data, kFullCutoff).term_expectation_values(data.parameters, keys);

    for (size_t q = 0; q < keys.size(); ++q) {
        BOOST_TEST_CONTEXT("query " << q) {
            BOOST_TEST(near(values[q].real(), state[q].real(), 1e-11));
            BOOST_TEST(near(values[q].imag(), -state[q].imag(), 1e-11));
            BOOST_TEST(near(values[q].real(), heisenberg[q].real()));
            BOOST_TEST(near(values[q].imag(), heisenberg[q].imag()));
        }
    }
}

// Heisenberg has no row, and so no value, for a monomial outside the operator. Past the cutoff and
// unpaired, so the fully-paired exception cannot have kept it. Partitioned, every partition throws.
BOOST_AUTO_TEST_CASE(term_values_heisenberg_absent_term_throws) {
    const auto data = load_case_data<kNumModes>(kCase);
    const VecZ absent{0, 2, 4, 6, 8, 10, 12, 14};
    for (const size_t partitions : {size_t{1}, size_t{2}}) {
        auto sim = built_sim(data, kTruncatedCutoff, partitions);
        BOOST_TEST_CONTEXT("partitions=" << partitions) {
            BOOST_CHECK_THROW(sim.term_expectation_values(data.parameters, {absent}),
                              monoprop::detail::OperatorTermNotFound);
        }
    }
}

// An in-place contraction drops the gates' adjoint, which the values need, so the query must refuse
// rather than replay what is left. Partitioned, the facade refuses before fanning out.
BOOST_AUTO_TEST_CASE(term_values_heisenberg_after_inplace_contraction_throws) {
    const auto data = load_case_data<kNumModes>(kCase);
    for (const size_t partitions : {size_t{1}, size_t{2}}) {
        auto sim = built_sim(data, kTruncatedCutoff, partitions);
        (void)sim.contract_partially(data.parameters, true);
        BOOST_TEST_CONTEXT("partitions=" << partitions) {
            BOOST_CHECK_THROW(sim.term_expectation_values({}, keys_of(data.hamiltonian)), GraphStateConflict);
        }
    }
}

// The keys are user input: an out-of-range slot must throw rather than write past the monomial.
BOOST_AUTO_TEST_CASE(term_values_out_of_range_slot_index_throws) {
    const auto data = load_case_data<kNumModes>(kCase);
    auto sim = built_sim(data, kTruncatedCutoff);
    BOOST_CHECK_THROW(sim.term_expectation_values(data.parameters, {VecZ{2 * kNumModes}}), AlgebraIndexOutOfRange);
}
