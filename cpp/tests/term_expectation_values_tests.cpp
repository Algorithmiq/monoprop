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

// Tests for term_expectation_values(), which returns each term's expectation value. The main checks: the
// values reassemble expectation_value(), and each matches expectation_value() with that term alone. Uses
// LiH because random_exact's values are all 0, so a wrong replay would pass there too.

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = LihFixture::n_modes;
constexpr auto kCase = "lih_fermionic_spin_exact.msgpack";
constexpr unsigned int kFullCutoff = 2 * kNumModes;
// Low enough to truncate but high enough to keep rotations, so the replay meets both rotated row pairs
// and cosine-only rows whose partner was cut.
constexpr unsigned int kTruncatedCutoff = 4;

// A propagator with the case's circuit built as a graph, ready to query.
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

// `op`'s terms, in the iteration order reassembled() relies on to pair them with values.
auto keys_of(const OperatorDict &op) -> std::vector<VecZ> {
    std::vector<VecZ> keys;
    for (const auto &[indices, coeff] : op) {
        keys.push_back(indices);
    }
    return keys;
}

// Σ coeff·value over `op`, which should equal expectation_value().
auto reassembled(const OperatorDict &op, const std::vector<std::complex<double>> &values) -> std::complex<double> {
    std::complex<double> total = 0.0;
    size_t q = 0;
    for (const auto &[indices, coeff] : op) {
        total += coeff * values[q++];
    }
    return total;
}

} // namespace

// The initial coefficients times the values sum to expectation_value(), with and without truncation; exact
// either way, since expectation_value() is linear in the coefficients. The operator includes an identity
// term, so this also checks that it returns 1.
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
        // update_initial_operator() keeps the core term unless the new operator has its own, so subtract it.
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

// For a row created mid-circuit, the value is the derivative of expectation_value() with respect to the row's
// coefficient, as documented: it matches expectation_value() with that row alone. It is not the row's ⟨P⟩.
BOOST_AUTO_TEST_CASE(mid_circuit_row_matches_a_single_term_reweight) {
    const auto data = load_case_data<kNumModes>(kCase);
    auto sim = built_sim(data, kTruncatedCutoff);
    // Re-weight with the row's own evolved coefficient, which is guaranteed to be Hermitian-encodable.
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

// Partitioned values match the single-partition ones. Partitions hash-split the rows, so each key is found
// in exactly one partition and the gather must not double-count it.
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

// Untruncated, Schrodinger and Heisenberg give the same values. In Schrodinger they are also the evolved
// state's coefficients, complex-conjugated: a coefficient multiplies M in the state, while ⟨M⟩ picks up the
// coefficient of M's adjoint.
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

// Heisenberg throws for a term absent from the evolved operator. The term is past the cutoff and not fully
// paired (fully-paired terms are exempt from truncation), so it cannot be present. With partitions, every
// partition throws right after a collective; this case exposed the poison race fixed in run_on_all().
BOOST_AUTO_TEST_CASE(heisenberg_absent_term_throws) {
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

// Heisenberg throws once an in-place contraction has absorbed the gates into the operator, since the values
// would miss those gates. With partitions, the facade throws before fanning out.
BOOST_AUTO_TEST_CASE(heisenberg_after_inplace_contraction_throws) {
    const auto data = load_case_data<kNumModes>(kCase);
    for (const size_t partitions : {size_t{1}, size_t{2}}) {
        auto sim = built_sim(data, kTruncatedCutoff, partitions);
        (void)sim.contract_partially(data.parameters, true);
        BOOST_TEST_CONTEXT("partitions=" << partitions) {
            BOOST_CHECK_THROW(sim.term_expectation_values({}, keys_of(data.hamiltonian)), GraphStateConflict);
        }
    }
}

// An out-of-range index throws rather than writing past the monomial, since keys are user input.
BOOST_AUTO_TEST_CASE(out_of_range_slot_index_throws) {
    const auto data = load_case_data<kNumModes>(kCase);
    auto sim = built_sim(data, kTruncatedCutoff);
    BOOST_CHECK_THROW(sim.term_expectation_values(data.parameters, {VecZ{2 * kNumModes}}), AlgebraIndexOutOfRange);
}
