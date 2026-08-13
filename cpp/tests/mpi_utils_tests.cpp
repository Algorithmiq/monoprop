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

// The pure MPIUtils.h primitives (term->owner mapping, wire word packing), driven without a comm.

#include <boost/test/unit_test.hpp>

#include <random>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/mpi/MPIUtils.h"

using namespace monoprop;

BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_range_and_hash_mod) {
    constexpr size_t N = 32;
    std::mt19937_64 rng(0x9E3779B9ULL);
    std::uniform_int_distribution<size_t> slot(0, 2 * N - 1);
    for (int trial = 0; trial < 500; ++trial) {
        VecZ inds;
        for (int k = 0; k < 4; ++k) {
            inds.push_back(slot(rng));
        }
        const auto mono = indices_to_bitset(inds, 2 * N);
        for (size_t n_ranks : {size_t{1}, size_t{2}, size_t{3}, size_t{7}}) {
            const size_t r = find_rank(mono, n_ranks);
            BOOST_TEST(r < n_ranks);
            BOOST_TEST(r == monomial_hash(mono) % n_ranks);
            BOOST_TEST(r == find_rank(mono, n_ranks)); // deterministic
        }
    }
}

// n_ranks == 0 is degenerate: owner is rank 0, not a modulo by zero.
BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_zero_ranks) {
    constexpr size_t N = 32;
    const auto mono = indices_to_bitset(VecZ{0, 3, 5}, 2 * N);
    BOOST_TEST(find_rank(mono, 0) == 0U);
}

BOOST_AUTO_TEST_CASE(mpi_utils_monomial_words_roundtrip) {
    constexpr size_t N = 96; // 2N = 192 bits -> 3 words
    const auto a = indices_to_bitset(VecZ{0, 1, 100, 191}, 2 * N);
    const auto b = indices_to_bitset(VecZ{5}, 2 * N);
    const auto c = indices_to_bitset(VecZ{}, 2 * N);

    // The record width comes off the monomial now, not a kWords<N> constant.
    const size_t kW = a.num_words();
    VecZ buf;
    mpi_detail::append_monomial_words(a, buf);
    mpi_detail::append_monomial_words(b, buf);
    mpi_detail::append_monomial_words(c, buf);
    BOOST_REQUIRE(buf.size() == 3 * kW);

    BOOST_TEST((mpi_detail::read_monomial_from_words(buf, 0, 2 * N) == a));
    BOOST_TEST((mpi_detail::read_monomial_from_words(buf, kW, 2 * N) == b));
    BOOST_TEST((mpi_detail::read_monomial_from_words(buf, 2 * kW, 2 * N) == c));

    constexpr size_t M = 32;
    const auto d = indices_to_bitset(VecZ{2, 40, 63}, 2 * M);
    VecZ sbuf;
    mpi_detail::append_monomial_words(d, sbuf);
    BOOST_REQUIRE(sbuf.size() == d.num_words());
    BOOST_TEST((mpi_detail::read_monomial_from_words(sbuf, 0, 2 * M) == d));
}
