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

#include "monoprop/Evolution.h"
#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/Utilities.h"

using namespace monoprop;
namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(bit_flipping_utilities) {
    auto val1 = even_bits<10, LSb0>();
    auto val2 = odd_bits<10, LSb0>();
    auto val3 = even_bits<10, MSb0>();
    auto val4 = odd_bits<10, MSb0>();
    BOOST_TEST(val1 == 0b0101010101);
    BOOST_TEST(val2 == 0b1010101010);
    BOOST_TEST(val3 == 0b1010101010);
    BOOST_TEST(val4 == 0b0101010101);
};

BOOST_AUTO_TEST_CASE(single_rank_below_sin_skip_preserves_existing_cycle) {
    constexpr size_t NumModes = 2;

    MPOperator<NumModes> mp_op;
    const auto first = indices_to_bitset<NumModes>(VecZ{1});
    const auto second = indices_to_bitset<NumModes>(VecZ{0, 1});
    mp_op.op = {first, second};
    mp_op.indexing.reset(1);
    mp_op.indexing.emplace(first, 0);
    mp_op.indexing.emplace(second, 1);

    const auto gen = indices_to_bitset<NumModes>(VecZ{0});
    const VecD coeffs = {1e-12, 1e-12};
    const auto cutoff_fn = [](const MajoranaSet<NumModes>&) { return true; };

    const auto result =
        evolve_maj<NumModes>(mp_op, gen, cutoff_fn, 1e-6, std::cref(coeffs), std::nullopt, 1e-4, false, MPI_COMM_SELF);

    BOOST_REQUIRE_EQUAL(result.cycles.size(), 1UL);
    BOOST_REQUIRE_EQUAL(result.phases.size(), 1UL);
    BOOST_REQUIRE_EQUAL(result.cycles[0].size(), 1UL);
    BOOST_CHECK(result.half_cycles[0].empty());
    BOOST_CHECK(result.half_op[0].empty());
    BOOST_CHECK(result.cos_inds.empty());
    BOOST_REQUIRE(result.compressed_cos_data.has_value());
    BOOST_CHECK_EQUAL(result.compressed_cos_data->total_count, 0UL);

    const auto cycle = result.cycles[0][0];
    BOOST_CHECK((cycle.first == 0UL && cycle.second == 1UL) || (cycle.first == 1UL && cycle.second == 0UL));
}
