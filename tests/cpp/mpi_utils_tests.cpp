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

// Unit coverage of the pure MPI helper primitives in MPIUtils.h: the deterministic term->owner
// mapping (find_rank) and the Majorana word (de)serialization used to pack terms onto the wire.
// These need no MPI runtime — they are exercised here directly rather than only through the
// distributed suites.

#include <boost/test/unit_test.hpp>

#include <random>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/mpi/MPIUtils.h"

using namespace monoprop;

// find_rank must return a value in [0, n_ranks), agree with hash % n_ranks, and be deterministic.
BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_range_and_hash_mod) {
    constexpr size_t N = 32;
    std::mt19937_64 rng(0x9E3779B9ULL);
    std::uniform_int_distribution<size_t> slot(0, 2 * N - 1);
    for (int trial = 0; trial < 500; ++trial) {
        VecZ inds;
        for (int k = 0; k < 4; ++k) {
            inds.push_back(slot(rng));
        }
        const auto maj = indices_to_bitset<N>(inds);
        for (size_t n_ranks : {size_t{1}, size_t{2}, size_t{3}, size_t{7}}) {
            const size_t r = find_rank<N>(maj, n_ranks);
            BOOST_TEST(r < n_ranks);
            BOOST_TEST(r == monomial_hash<N>(maj) % n_ranks); // matches the documented formula
            BOOST_TEST(r == find_rank<N>(maj, n_ranks));       // deterministic
        }
    }
}

// n_ranks == 0 is the documented degenerate case: owner is rank 0 (no modulo by zero).
BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_zero_ranks) {
    constexpr size_t N = 32;
    const auto maj = indices_to_bitset<N>(VecZ{0, 3, 5});
    BOOST_TEST(find_rank<N>(maj, 0) == 0U);
}

// append_majorana_words / read_majorana_from_words round-trip several packed records at their
// offsets, single-word (N=32) and multi-word (N=96) alike.
BOOST_AUTO_TEST_CASE(mpi_utils_majorana_words_roundtrip) {
    constexpr size_t N = 96; // 2N = 192 bits -> 3 words
    const auto a = indices_to_bitset<N>(VecZ{0, 1, 100, 191});
    const auto b = indices_to_bitset<N>(VecZ{5});
    const auto c = indices_to_bitset<N>(VecZ{});

    VecZ buf;
    mpi_detail::append_majorana_words<N>(a, buf);
    mpi_detail::append_majorana_words<N>(b, buf);
    mpi_detail::append_majorana_words<N>(c, buf);
    BOOST_REQUIRE(buf.size() == 3 * mpi_detail::kWords<N>);

    BOOST_TEST((mpi_detail::read_majorana_from_words<N>(buf, 0) == a));
    BOOST_TEST((mpi_detail::read_majorana_from_words<N>(buf, mpi_detail::kWords<N>) == b));
    BOOST_TEST((mpi_detail::read_majorana_from_words<N>(buf, 2 * mpi_detail::kWords<N>) == c));

    // Single-word path.
    constexpr size_t M = 32;
    const auto d = indices_to_bitset<M>(VecZ{2, 40, 63});
    VecZ sbuf;
    mpi_detail::append_majorana_words<M>(d, sbuf);
    BOOST_REQUIRE(sbuf.size() == mpi_detail::kWords<M>);
    BOOST_TEST((mpi_detail::read_majorana_from_words<M>(sbuf, 0) == d));
}
