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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/mpi/MPIUtils.h"

using namespace monoprop;

TEST_CASE("mpi_utils_find_rank_range_and_hash_mod") {
    constexpr size_t N = 32;
    std::mt19937_64 rng(0x9E3779B9ULL);
    std::uniform_int_distribution<size_t> slot(0, 2 * N - 1);
    for (int trial = 0; trial < 500; ++trial) {
        VecZ inds;
        for (int k = 0; k < 4; ++k) {
            inds.push_back(slot(rng));
        }
        const auto mono = indices_to_bitset<N>(inds);
        for (size_t n_ranks : {size_t{1}, size_t{2}, size_t{3}, size_t{7}}) {
            const size_t r = find_rank<N>(mono, n_ranks);
            CHECK(r < n_ranks);
            CHECK(r == monomial_hash<N>(mono) % n_ranks);
            CHECK(r == find_rank<N>(mono, n_ranks)); // deterministic
        }
    }
}

// n_ranks == 0 is degenerate: owner is rank 0, not a modulo by zero.
TEST_CASE("mpi_utils_find_rank_zero_ranks") {
    constexpr size_t N = 32;
    const auto mono = indices_to_bitset<N>(VecZ{0, 3, 5});
    CHECK(find_rank<N>(mono, 0) == 0U);
}

TEST_CASE("mpi_utils_monomial_words_roundtrip") {
    constexpr size_t N = 96; // 2N = 192 bits -> 3 words
    const auto a = indices_to_bitset<N>(VecZ{0, 1, 100, 191});
    const auto b = indices_to_bitset<N>(VecZ{5});
    const auto c = indices_to_bitset<N>(VecZ{});

    VecZ buf;
    mpi_detail::append_monomial_words<N>(a, buf);
    mpi_detail::append_monomial_words<N>(b, buf);
    mpi_detail::append_monomial_words<N>(c, buf);
    REQUIRE(buf.size() == 3 * mpi_detail::kWords<N>);

    CHECK((mpi_detail::read_monomial_from_words<N>(buf, 0) == a));
    CHECK((mpi_detail::read_monomial_from_words<N>(buf, mpi_detail::kWords<N>) == b));
    CHECK((mpi_detail::read_monomial_from_words<N>(buf, 2 * mpi_detail::kWords<N>) == c));

    constexpr size_t M = 32;
    const auto d = indices_to_bitset<M>(VecZ{2, 40, 63});
    VecZ sbuf;
    mpi_detail::append_monomial_words<M>(d, sbuf);
    REQUIRE(sbuf.size() == mpi_detail::kWords<M>);
    CHECK((mpi_detail::read_monomial_from_words<M>(sbuf, 0) == d));
}
