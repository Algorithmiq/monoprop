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

#include "TestUtilities.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/Utilities.h"

using namespace monoprop;

namespace utf = boost::unit_test;
namespace bdata = utf::data;

constexpr int NumQubits = 4;
// Test cases for indices_to_bitset
static std::vector<VecZ> ds_input_indices_to_bitset_test = {
    {0, 1, 2, 3}, // Full indices set
    {},           // No indices set, empty majorana
    {5},          // Single index set
    {4, 7}        // Two indices set
};
static std::vector<MajoranaSet<NumQubits>> ds_output_indices_to_bitset_test = {

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

// Test cases for length_cutoff
static std::vector<MajoranaSet<NumQubits>> ds_input_bitset_to_indices_test = {
    0b00000000, // No indices set, empty majorana
    0b00011000, // Single index set
    0b10101010  // Two indices set
};
static std::vector<int> ds_cutoff_values = {
    4, // Cutoff larger than any pairing distance
    2, // Cutoff smaller than pairing distance
    2  // Cutoff equal to pairing distance
};
static std::vector<bool> ds_expected_length_results = {
    true, // No indices set, should be paired
    true, // Single pair within cutoff, should be paired
    false // Multiple pairs exceeding cutoff, should not be paired
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
static std::vector<std::pair<MajoranaSet<NumQubits2>, int>> ds_get_multiplicative_phase = {{{0b0001}, 0},
                                                                                           {{0b0101}, -1},
                                                                                           {{0b1001}, 1}};

BOOST_DATA_TEST_CASE(get_multiplicative_phase_test, bdata::make(ds_get_multiplicative_phase), test_pair) {
    auto [majorana_set, expected_phase] = test_pair;
    VecZ gen_vec = {0, 1};
    auto gen_bitset = indices_to_bitset<NumQubits2>(gen_vec);
    auto maj_count = majorana_set.count();
    auto gen_count = gen_bitset.count();
    auto overlap = (majorana_set & gen_bitset).count();
    auto result = get_multiplicative_phase<NumQubits2>(majorana_set, gen_bitset, maj_count, gen_count, overlap);
    BOOST_CHECK(result == expected_phase);
}

struct IS_FULLY_PAIRED_TEST_CASE {
    VecZ inds;
    MajoranaVector<NumQubits2> op_terms;
    VecZ expected_result;
    std::string test_name;

    friend std::ostream& operator<<(std::ostream& os, const IS_FULLY_PAIRED_TEST_CASE& aTestCase) {
        return os << aTestCase.test_name;
    }
};

static std::vector<IS_FULLY_PAIRED_TEST_CASE> ds_is_fully_paired_test = {
    // Nothing is paired
    {{0, 1, 2, 3}, {0b0001, 0b0010, 0b0100, 0b1000}, {}, "Nothing is paired"},
    // Everything is paired
    {{0, 1, 2, 3}, {0b0000, 0b0011, 0b1100, 0b1111}, {0, 1, 2, 3}, "Everything is paired"},
    // Partially paired
    {{0, 1, 2, 3, 4, 5, 6}, {0b0001, 0b0011, 0b1000, 0b0101, 0b1100, 0b0110, 0b1110}, {1, 4}, "Partially paired"}};

BOOST_DATA_TEST_CASE(is_fully_paired_test, bdata::make(ds_is_fully_paired_test), test_case) {
    auto result = is_fully_paired<NumQubits2>(test_case.inds, test_case.op_terms);
    BOOST_CHECK(std::is_permutation(result.cbegin(), result.cend(), test_case.expected_result.cbegin()));
}

// even_bits/odd_bits from Utilities.h under both bit orderings (formerly utilities.cpp).
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
