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

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "AlgebraReference.h" // fermionic_to_binary_operator, get_multiplicative_phase (test-only)
#include "TestUtilities.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/Utilities.h"

using namespace monoprop;

namespace utf = boost::unit_test;
namespace bdata = utf::data;

constexpr int NumQubits = 4;
static std::vector<VecZ> ds_input_indices_to_bitset_test = {
    {0, 1, 2, 3}, // Full indices set
    {},           // No indices set, empty majorana
    {5},          // Single index set
    {4, 7}        // Two indices set
};
static std::vector<Monomial<NumQubits>> ds_output_indices_to_bitset_test = {

    {0b11110000}, // Full indices set
    {0b00000000}, // No indices set, empty majorana
    {0b00000100}, // Single index set
    {0b00001001}  // Two indices set

};

BOOST_DATA_TEST_CASE(indices_to_bitset_test,
                     bdata::make(ds_input_indices_to_bitset_test) ^ ds_output_indices_to_bitset_test,
                     input_indices,
                     expected_bitset) {
    auto bitset = indices_to_bitset<NumQubits>(input_indices);
    BOOST_CHECK(bitset == expected_bitset);
}

static std::vector<Monomial<NumQubits>> ds_input_bitset_to_indices_test = {
    0b00000000, // fully paired
    0b00011000, // 2 slots
    0b10101010  // 4 slots
};
static std::vector<int> ds_cutoff_values = {
    4,
    2,
    2 // below the last case's slot count
};
static std::vector<bool> ds_expected_length_results = {
    true,
    true,
    false // length_cutoff keeps iff fully paired or slot count <= cutoff
};

BOOST_DATA_TEST_CASE(length_cutoff_test,
                     bdata::make(ds_input_bitset_to_indices_test) ^ bdata::make(ds_cutoff_values)
                         ^ ds_expected_length_results,
                     input_bitset,
                     cutoff,
                     expected_result) {
    auto result = length_cutoff<NumQubits>(input_bitset, cutoff);
    BOOST_CHECK(result == expected_result);
}

BOOST_AUTO_TEST_CASE(test_fermionic_to_binary_operator_empty) {
    std::vector<VecZ> empty_operator;
    auto result = fermionic_to_binary_operator<NumQubits>(empty_operator);
    BOOST_CHECK(result.empty());
}

BOOST_AUTO_TEST_CASE(test_fermionic_to_binary_operator_single_term) {
    std::vector<VecZ> single_term_operator = {{0, 1, 2}};
    auto result = fermionic_to_binary_operator<NumQubits>(single_term_operator);
    BOOST_CHECK(result.size() == 1);
    BOOST_CHECK(result[0] == 0b11100000);
}

BOOST_AUTO_TEST_CASE(test_fermionic_to_binary_operator_multiple_terms) {
    std::vector<VecZ> multi_term_operator = {{0, 1}, {2, 3}};
    auto result = fermionic_to_binary_operator<NumQubits>(multi_term_operator);
    BOOST_CHECK(result.size() == 2);
    BOOST_CHECK(result[0] == 0b11000000);
    BOOST_CHECK(result[1] == 0b00110000);
}

constexpr size_t NumQubits2 = 2;
static std::vector<std::pair<Monomial<NumQubits2>, int>> ds_get_multiplicative_phase = {{{0b0001}, 0},
                                                                                        {{0b0101}, -1},
                                                                                        {{0b1001}, 1}};

BOOST_DATA_TEST_CASE(get_multiplicative_phase_test, bdata::make(ds_get_multiplicative_phase), test_pair) {
    auto [majorana_set, expected_phase] = test_pair;
    VecZ gen_vec = {0, 1};
    auto gen_bitset = indices_to_bitset<NumQubits2>(gen_vec);
    auto mono_count = majorana_set.count();
    auto gen_count = gen_bitset.count();
    auto overlap = (majorana_set & gen_bitset).count();
    auto result = get_multiplicative_phase<NumQubits2>(majorana_set, gen_bitset, mono_count, gen_count, overlap);
    BOOST_CHECK(result == expected_phase);
}

struct IS_FULLY_PAIRED_TEST_CASE {
    VecZ inds;
    MonomialList<NumQubits2> op_terms;
    VecZ expected_result;
    std::string test_name;

    friend std::ostream& operator<<(std::ostream& os, const IS_FULLY_PAIRED_TEST_CASE& aTestCase) {
        return os << aTestCase.test_name;
    }
};

static std::vector<IS_FULLY_PAIRED_TEST_CASE> ds_is_fully_paired_test = {
    {{0, 1, 2, 3}, {0b0001, 0b0010, 0b0100, 0b1000}, {}, "Nothing is paired"},
    {{0, 1, 2, 3}, {0b0000, 0b0011, 0b1100, 0b1111}, {0, 1, 2, 3}, "Everything is paired"},
    {{0, 1, 2, 3, 4, 5, 6}, {0b0001, 0b0011, 0b1000, 0b0101, 0b1100, 0b0110, 0b1110}, {1, 4}, "Partially paired"}};

BOOST_DATA_TEST_CASE(is_fully_paired_test, bdata::make(ds_is_fully_paired_test), test_case) {
    auto result = is_fully_paired<NumQubits2>(test_case.inds, test_case.op_terms);
    BOOST_CHECK(std::is_permutation(result.cbegin(), result.cend(), test_case.expected_result.cbegin()));
}

BOOST_AUTO_TEST_CASE(bit_flipping_utilities) {
    auto val1 = even_bits<10, LSb0>();
    auto val2 = odd_bits<10, LSb0>();
    auto val3 = even_bits<10, MSb0>();
    auto val4 = odd_bits<10, MSb0>();
    BOOST_TEST(val1 == 0b0101010101);
    BOOST_TEST(val2 == 0b1010101010);
    BOOST_TEST(val3 == 0b1010101010);
    BOOST_TEST(val4 == 0b0101010101);
}

// The evaluation functional carries the reference state sparsely, so every EvalState operation has to
// agree with what the equivalent dense vector would have produced -- exactly, not approximately.

namespace {

// A reference state shaped like a real one: unit +-1 phases on a sparse ascending row set, and a
// deliberately awkward operator (mixed magnitudes, exact zeros, a subnormal) to catch any reordering.
constexpr size_t kStateLength = 37;
const std::vector<TermIndex> kRows = {0, 1, 5, 12, 13, 14, 30, 36};
const VecD kVals = {1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0};

auto make_dense(size_t length, const std::vector<TermIndex>& rows, const VecD& vals) -> VecD {
    VecD dense(length, 0.0);
    for (size_t k = 0; k < rows.size(); ++k) {
        dense[static_cast<size_t>(rows[k])] = vals[k];
    }
    return dense;
}

auto make_op(size_t length) -> VecD {
    VecD op(length, 0.0);
    for (size_t i = 0; i < length; ++i) {
        // Spread over ~11 orders of magnitude (2^-23 .. 2^13) with alternating signs, so a dropped or
        // reordered term shows up in the low bits rather than cancelling.
        op[i] = ((i % 3 == 0) ? -1.0 : 1.0) * std::ldexp(1.0 + static_cast<double>(i) / 7.0, static_cast<int>(i) - 23);
    }
    op[5] = 0.0;
    op[13] = std::numeric_limits<double>::denorm_min();
    return op;
}

} // namespace

// dot() over the sparse rows is bit-identical to the dense inner product, hence CHECK_EQUAL.
BOOST_AUTO_TEST_CASE(eval_state_sparse_dot_is_bit_identical_to_dense) {
    const auto op = make_op(kStateLength);
    const auto dense = make_dense(kStateLength, kRows, kVals);

    const auto sparse = EvalState::sparse(kStateLength, kRows, kVals);
    BOOST_CHECK_EQUAL(sparse.length(), kStateLength);
    BOOST_CHECK_EQUAL(sparse.dot(op), inner_product(dense, op));
    BOOST_CHECK_EQUAL(EvalState::dense(dense).dot(op), inner_product(dense, op));

    // No scored rows at all (an operator with no fully-paired terms) is a legal state, not a shortcut:
    // the caller still has to reach its allreduce.
    const auto empty = EvalState::sparse(kStateLength, {}, {});
    BOOST_CHECK_EQUAL(empty.length(), kStateLength);
    BOOST_CHECK_EQUAL(empty.dot(op), 0.0);

    // An operator longer than the state is fine (dot spans the state); shorter is a hard error.
    VecD longer = op;
    longer.push_back(1.0);
    BOOST_CHECK_EQUAL(sparse.dot(longer), sparse.dot(op));
    const VecD shorter(kStateLength - 1, 1.0);
    BOOST_CHECK_THROW(sparse.dot(shorter), std::invalid_argument);
}

// scatter_into() must assign: its only caller hands it thread-local scratch holding a previous, longer
// state, which resize-and-scatter would leak through.
BOOST_AUTO_TEST_CASE(eval_state_scatter_into_overwrites_a_dirty_buffer) {
    const auto dense = make_dense(kStateLength, kRows, kVals);
    const auto sparse = EvalState::sparse(kStateLength, kRows, kVals);

    VecD out;
    sparse.scatter_into(out);
    BOOST_CHECK(out == dense);

    VecD dirty(kStateLength * 2, 7.5);
    sparse.scatter_into(dirty);
    BOOST_CHECK_EQUAL(dirty.size(), kStateLength);
    BOOST_CHECK(dirty == dense);

    // Pre-dirtied at exactly the right length: the size check alone must not let the scatter be skipped.
    VecD same_length(kStateLength, 7.5);
    sparse.scatter_into(same_length);
    BOOST_CHECK(same_length == dense);

    sparse.scatter_into(same_length);
    BOOST_CHECK(same_length == dense);
    VecD from_dense(3, -1.0);
    EvalState::dense(dense).scatter_into(from_dense);
    BOOST_CHECK(from_dense == dense);
}

// indices_above() is the paring keep-set. It must match the dense scan for every threshold -- including
// a negative one, where |0.0| > threshold keeps even the unscored rows.
BOOST_AUTO_TEST_CASE(eval_state_indices_above_matches_the_dense_scan) {
    const auto dense = make_dense(kStateLength, kRows, kVals);
    const auto sparse = EvalState::sparse(kStateLength, kRows, kVals);

    const std::vector<double> thresholds =
        {-1.0, -0.0, 0.0, 1e-12, 0.5, std::nextafter(1.0, 0.0), 1.0, 2.0, std::numeric_limits<double>::quiet_NaN()};
    for (const auto t : thresholds) {
        BOOST_TEST_CONTEXT("threshold = " << t) {
            const auto expected = indices_above(dense, t);
            BOOST_CHECK(sparse.indices_above(t) == expected);
            BOOST_CHECK(EvalState::dense(dense).indices_above(t) == expected);
        }
    }

    // Spot-check the two ends rather than trusting the dense oracle alone.
    BOOST_CHECK_EQUAL(sparse.indices_above(0.0).size(), kRows.size());
    BOOST_CHECK_EQUAL(sparse.indices_above(-1.0).size(), kStateLength);
    BOOST_CHECK(sparse.indices_above(1.0).empty());
    BOOST_CHECK(sparse.indices_above(std::numeric_limits<double>::quiet_NaN()).empty());
}

// End-to-end sparse-vs-dense equivalence, with no synthetic operator involved: the gradient path takes
// its value from the dense inner_product over its back-evolution buffer, the energy path from the sparse
// dot over the very same forward-evolved operator, so the two must agree bit-exactly -- on the exact
// graph and on the pared one, in both pictures.
BOOST_AUTO_TEST_CASE(sparse_energy_matches_the_dense_gradient_value_bit_exactly) {
    constexpr size_t kNumModes = 8;
    const auto data = test_utils::load_case_data<kNumModes>("random_exact.msgpack");

    for (const auto schrodinger_cutoff : {std::optional<unsigned int>{}, std::optional<unsigned int>{4}}) {
        BOOST_TEST_CONTEXT("schrodinger_cutoff = " << (schrodinger_cutoff ? "4" : "none")) {
            test_utils::SimulatorConfig cfg{.schrodinger_cutoff = schrodinger_cutoff, .comm = MPI_COMM_SELF};
            auto sim = test_utils::build_simulator<kNumModes>(data, cfg);
            sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

            BOOST_CHECK_EQUAL(sim.expectation_value(data.parameters),
                              sim.expectation_value_and_gradient(data.parameters).first);

            // Paring drives the keep-set off the sparse scores in the Heisenberg picture, so this also
            // pins EvalState::indices_above against the dense scan through the real functional.
            const std::optional<double> threshold{1e-10};
            BOOST_CHECK_EQUAL(sim.expectation_value_functional(threshold)(data.parameters),
                              sim.expectation_value_and_gradient_functional(threshold)(data.parameters).first);
        }
    }
}

// End-to-end counterpart of the scatter_into contract: the gradient's dense state lives in thread-local
// scratch shared by every functional on the thread, so two propagators of different operator sizes
// interleaving gradient calls must each keep reproducing their isolated value exactly.
BOOST_AUTO_TEST_CASE(interleaved_gradients_do_not_share_scratch_state) {
    constexpr size_t kNumModes = 8;
    const auto data = test_utils::load_case_data<kNumModes>("random_exact.msgpack");

    auto build = [&data](unsigned int cutoff) {
        auto sim =
            MonomialPropagator<kNumModes>(data.hamiltonian, cutoff, data.initial_state, std::nullopt, MPI_COMM_SELF);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        return sim;
    };

    auto wide = build(2 * kNumModes);
    auto narrow = build(4);
    BOOST_REQUIRE(wide.mp_op().size() != narrow.mp_op().size());

    auto grad_wide = wide.expectation_value_and_gradient_functional();
    auto grad_narrow = narrow.expectation_value_and_gradient_functional();

    const auto [ref_wide_value, ref_wide_grad] = grad_wide(data.parameters);
    const auto [ref_narrow_value, ref_narrow_grad] = grad_narrow(data.parameters);
    for (int round = 0; round < 2; ++round) {
        BOOST_TEST_CONTEXT("round " << round) {
            const auto [wide_value, wide_grad] = grad_wide(data.parameters);
            BOOST_CHECK_EQUAL(wide_value, ref_wide_value);
            BOOST_CHECK(wide_grad == ref_wide_grad);

            const auto [narrow_value, narrow_grad] = grad_narrow(data.parameters);
            BOOST_CHECK_EQUAL(narrow_value, ref_narrow_value);
            BOOST_CHECK(narrow_grad == ref_narrow_grad);
        }
    }

    // The sparse energy path leaves no dense state behind on the Heisenberg operator.
    BOOST_CHECK_EQUAL(wide.expectation_value_functional()(data.parameters), ref_wide_value);
    BOOST_CHECK(wide.mp_op().state_coeffs.empty());
    BOOST_CHECK(narrow.mp_op().state_coeffs.empty());
}

BOOST_AUTO_TEST_CASE(eval_state_sparse_rejects_inconsistent_inputs) {
    BOOST_CHECK_THROW(EvalState::sparse(kStateLength, kRows, VecD{1.0}), std::invalid_argument);
    const std::vector<TermIndex> out_of_range = {0, static_cast<TermIndex>(kStateLength)};
    BOOST_CHECK_THROW(EvalState::sparse(kStateLength, out_of_range, VecD{1.0, 1.0}), std::invalid_argument);
    BOOST_CHECK_NO_THROW(EvalState::sparse(0, {}, {}));
}
