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

