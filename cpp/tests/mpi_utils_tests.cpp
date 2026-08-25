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

// The pure MPIUtils.h primitives (term->owner mapping, wire word packing), driven without a comm --
// plus the routing agreement between find_rank and the scan, a property of neither call site alone.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;

// find_rank is splitmix over the dense words modulo the rank count, and nothing else, so the oracle
// is asserted unconditionally rather than as one of several permitted hashes.
BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_range_and_hash_mod) {
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
            BOOST_TEST(r == monomial_hash<N>(mono) % n_ranks);
            BOOST_TEST(r < n_ranks);
            BOOST_TEST(r == find_rank<N>(mono, n_ranks)); // deterministic
        }
    }
}

// n_ranks == 0 is degenerate: owner is rank 0, not a modulo by zero.
BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_zero_ranks) {
    constexpr size_t N = 32;
    const auto mono = indices_to_bitset<N>(VecZ{0, 3, 5});
    BOOST_TEST(find_rank<N>(mono, 0) == 0U);
}

BOOST_AUTO_TEST_CASE(mpi_utils_monomial_words_roundtrip) {
    constexpr size_t N = 96; // 2N = 192 bits -> 3 words
    const auto a = indices_to_bitset<N>(VecZ{0, 1, 100, 191});
    const auto b = indices_to_bitset<N>(VecZ{5});
    const auto c = indices_to_bitset<N>(VecZ{});

    VecZ buf;
    mpi_detail::append_monomial_words<N>(a, buf);
    mpi_detail::append_monomial_words<N>(b, buf);
    mpi_detail::append_monomial_words<N>(c, buf);
    BOOST_REQUIRE(buf.size() == 3 * mpi_detail::kWords<N>);

    BOOST_TEST((mpi_detail::read_monomial_from_words<N>(buf, 0) == a));
    BOOST_TEST((mpi_detail::read_monomial_from_words<N>(buf, mpi_detail::kWords<N>) == b));
    BOOST_TEST((mpi_detail::read_monomial_from_words<N>(buf, 2 * mpi_detail::kWords<N>) == c));

    constexpr size_t M = 32;
    const auto d = indices_to_bitset<M>(VecZ{2, 40, 63});
    VecZ sbuf;
    mpi_detail::append_monomial_words<M>(d, sbuf);
    BOOST_REQUIRE(sbuf.size() == mpi_detail::kWords<M>);
    BOOST_TEST((mpi_detail::read_monomial_from_words<M>(sbuf, 0) == d));
}

namespace {

template <size_t N>
auto draw_well_formed(std::mt19937_64 &rng, size_t logical, size_t weight) -> Monomial<N> {
    VecZ idx;
    std::uniform_int_distribution<size_t> dist(0, (2 * logical) - 1);
    while (idx.size() < weight) {
        const size_t v = dist(rng);
        if (std::find(idx.begin(), idx.end(), v) == idx.end()) {
            idx.push_back(v);
        }
    }
    return indices_to_bitset<N>(idx);
}

auto build_op(const std::vector<Monomial<32>> &terms) -> detail::MPOperator<32> {
    detail::MPOperator<32> op;
    op.basis = Basis::Majorana;
    detail::insert_absent_terms<32>(
        op,
        terms.size(),
        [&](size_t k) -> const Monomial<32> & { return terms[k]; },
        [&](size_t k, size_t base) { assign_row<32>(*op.store, base + k, terms[k]); });
    return op;
}

auto check_bucket_ownership(const std::vector<VecZ> &buckets, size_t ranks, size_t &checked) -> void {
    // Every offset comes from the codec's walk: the record is VARIABLE WIDTH, so a hardcoded stride
    // would compare a monomial decoded at the wrong offset against the wrong rank.
    using QC = detail::QueryCodec<32>;
    const detail::QueryLayout layout{/*fused=*/false};
    for (size_t r = 0; r < buckets.size(); ++r) {
        size_t off = 0;
        while (off < buckets[r].size()) {
            Monomial<32> mono;
            int phase = 0;
            QC::read_mono(buckets[r], off, mono, phase);
            BOOST_REQUIRE_EQUAL(find_rank<32>(mono, ranks), r);
            off = QC::next_off(buckets[r], layout, off);
            ++checked;
        }
        BOOST_REQUIRE_EQUAL(off, buckets[r].size());
    }
}

// The self-owned bucket is staged as positions, not encoded, so it is invisible to the walk above --
// without this the r == my_rank arm of the routing decision goes unchecked.
auto check_self_ownership(const detail::SelfQueryStage<32> &stage, size_t ranks, size_t my_rank, size_t &checked)
    -> void {
    for (size_t q = 0; q < stage.size(); ++q) {
        Monomial<32> mono;
        for (size_t j = 0; j < stage.k_of[q]; ++j) {
            mono.set(static_cast<size_t>(stage.pos_flat[stage.pos_off[q] + j]));
        }
        BOOST_REQUIRE_EQUAL(find_rank<32>(mono, ranks), my_rank);
        ++checked;
    }
}

} // namespace

// The scan hashes the partner it just built; find_rank hashes what the resolve side decoded off the
// wire. Nothing downstream notices if they diverge -- the term simply exists twice.
BOOST_AUTO_TEST_CASE(mpi_utils_scan_routing_agrees_with_find_rank) {
    constexpr size_t kN = 32;
    constexpr size_t kLogical = 30;
    std::mt19937_64 rng(0xB0B1E5U);

    std::vector<Monomial<kN>> terms;
    for (size_t i = 0; i < 2000; ++i) {
        terms.push_back(draw_well_formed<kN>(rng, kLogical, 1 + (rng() % 6)));
    }
    auto op = build_op(terms);
    const Monomial<kN> gen = draw_well_formed<kN>(rng, kLogical, 4);
    VecD coeffs(op.store->size(), 1.0);

    const CutoffFn<kN> fn = detail::LengthCutoff<kN>{10, kLogical};
    const detail::CutoffEvaluator<kN> eval(fn);
    const auto cut = detail::build_majorana_evolution_cutoff_state(std::nullopt,
                                                                   std::cref(coeffs),
                                                                   std::nullopt,
                                                                   std::optional<double>{0.3});

    size_t checked = 0;
    size_t self_checked = 0;
    for (const size_t ranks : {2U, 4U, 8U}) {
        const auto res = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>>(op,
                                                                                 gen,
                                                                                 eval,
                                                                                 cut,
                                                                                 coeffs,
                                                                                 std::nullopt,
                                                                                 ranks,
                                                                                 0,
                                                                                 false,
                                                                                 nullptr,
                                                                                 1.0);
        BOOST_REQUIRE_EQUAL(res.leader_queries.size(), ranks);
        // The scan routes a self-owned partner to the stage, so bucket 0 must be empty here.
        BOOST_REQUIRE(res.leader_queries[0].empty());
        BOOST_REQUIRE(res.follower_queries[0].empty());
        check_bucket_ownership(res.leader_queries, ranks, checked);
        check_bucket_ownership(res.follower_queries, ranks, checked);
        check_self_ownership(res.leader_self, ranks, /*my_rank=*/0, self_checked);
        check_self_ownership(res.follower_self, ranks, /*my_rank=*/0, self_checked);
    }
    // Without this the loop above passes trivially if the scan emitted nothing. The floor is on the SUM
    // because that is what is invariant across the split: the encoded counter alone fell to 797 of 1161
    // when the self-owned partners moved into the stage, with nothing going unchecked. Each arm still
    // carries its own floor -- a routing bug sending everything one way leaves the sum intact -- and the
    // message prints the measured 797/364 so those can be re-grounded rather than guessed.
    BOOST_TEST_MESSAGE("encoded=" << checked << " staged=" << self_checked);
    BOOST_TEST(checked + self_checked > 1000U);
    BOOST_TEST(checked > 500U);
    BOOST_TEST(self_checked > 200U);
}
