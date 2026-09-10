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
#include <bit>
#include <cstddef>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;

// Under the splitmix router find_rank is the dense words modulo the rank count and nothing else, so the
// oracle is asserted unconditionally rather than as one of several permitted hashes. The router is
// constructed explicitly: there is no rank-count overload to reach it by accident.
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
            const auto router = routing::Router::splitmix(n_ranks);
            const size_t r = find_rank<N>(mono, router);
            BOOST_TEST(r == monomial_hash<N>(mono) % n_ranks);
            BOOST_TEST(r < n_ranks);
            BOOST_TEST(r == find_rank<N>(mono, router)); // deterministic
        }
    }
}

// A zero-slot world is degenerate: Router clamps it to one slot, so the owner is rank 0 rather than a
// modulo by zero.
BOOST_AUTO_TEST_CASE(mpi_utils_find_rank_zero_ranks) {
    constexpr size_t N = 32;
    const auto mono = indices_to_bitset<N>(VecZ{0, 3, 5});
    BOOST_TEST(find_rank<N>(mono, routing::Router::splitmix(0)) == 0U);
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
    detail::insert_absent_terms<32>(op, terms.size(), [&](size_t k, size_t base) {
        assign_row<32>(*op.store, base + k, terms[k]);
    });
    return op;
}

auto check_bucket_ownership(const std::vector<VecZ> &buckets, const routing::Router &router, size_t &checked) -> void {
    // Every offset comes from the record walk: widths vary, so a hardcoded stride would compare a
    // monomial decoded at the wrong offset against the wrong rank.
    using QW = detail::QueryWire<32>;
    const detail::QueryForm form = detail::QueryForm::Plain;
    for (size_t r = 0; r < buckets.size(); ++r) {
        size_t off = 0;
        while (off < buckets[r].size()) {
            const size_t k = QW::k_at(buckets[r], off);
            std::vector<QW::PosT> pos(k);
            (void)QW::read_positions(buckets[r], off, pos);
            Monomial<32> mono;
            for (size_t j = 0; j < k; ++j) {
                mono.set(static_cast<size_t>(pos[j]));
            }
            BOOST_REQUIRE_EQUAL(find_rank<32>(mono, router), r);
            off = QW::next_off(buckets[r], form, off);
            ++checked;
        }
        BOOST_REQUIRE_EQUAL(off, buckets[r].size());
    }
}

// The self-owned bucket is staged as positions, not encoded, so it is invisible to the walk above --
// without this the r == my_rank arm of the routing decision goes unchecked.
auto check_self_ownership(const detail::SelfQueryStage<32> &stage,
                          const routing::Router &router,
                          size_t my_rank,
                          size_t &checked) -> void {
    for (size_t q = 0; q < stage.size(); ++q) {
        Monomial<32> mono;
        for (size_t j = 0; j < stage.k_of[q]; ++j) {
            mono.set(static_cast<size_t>(stage.pos_flat[stage.pos_off[q] + j]));
        }
        BOOST_REQUIRE_EQUAL(find_rank<32>(mono, router), my_rank);
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
    const Monomial<kN> gen = draw_well_formed<kN>(rng, kLogical, 4);

    const CutoffFn<kN> fn = detail::LengthCutoff<kN>{10, kLogical};
    const detail::CutoffEvaluator<kN> eval(fn);

    // Each rank is handed exactly the terms it owns, as MonomialPropagator seeds it: the emit path
    // routes by rank(M) ^ rank_shift(G), which is only the owner of M^G when M really is local.
    //
    // The partner count is a property of the operator and the gate, not of where the partners live, so
    // the sum over the ranks is the same for every router; routing only moves a partner between ranks
    // and between the encoded (cross-rank) and staged (self-owned) sides. Pinning that invariance is
    // stronger than a floor: a routing bug that drops partners moves the total, and one that misroutes
    // them moves the split.
    size_t routers = 0;
    std::optional<size_t> first_total;
    for (const size_t ranks : {2U, 4U, 8U}) {
        // BOTH routers, because the agreement is a property of the pair and not of either hash: the
        // scan routes off the partner's fingerprint (Router::dest_from_fingerprint) or the dense hash,
        // and find_rank calls Router::dest, so a divergence introduced by one of them shows up here
        // whichever routing the geometry resolves to.
        for (const bool linear : {false, true}) {
            const auto router = routing::Router::for_modes<kN>(ranks, /*partitions=*/1, linear);
            BOOST_REQUIRE_EQUAL(router.is_linear(), linear);
            const size_t shift = router.rank_shift<kN>(gen);
            // Per router, not summed over them: the floors are what stops the loop passing on an empty
            // scan, and a sum lets one router carry the other.
            size_t checked = 0;
            size_t self_checked = 0;
            for (size_t my_rank = 0; my_rank < ranks; ++my_rank) {
                std::vector<Monomial<kN>> owned;
                for (const auto &t : terms) {
                    if (find_rank<kN>(t, router) == my_rank) {
                        owned.push_back(t);
                    }
                }
                BOOST_REQUIRE(!owned.empty());
                auto op = build_op(owned);
                VecD coeffs(op.store->size(), 1.0);
                const auto cut = detail::build_majorana_evolution_cutoff_state(std::nullopt,
                                                                               std::cref(coeffs),
                                                                               std::nullopt,
                                                                               std::optional<double>{0.3});
                detail::GateScratch<kN> scratch;
                const auto res = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>, /*CaptureValues=*/false>(
                    op,
                    gen,
                    eval,
                    cut,
                    coeffs,
                    std::nullopt,
                    /*over_cutoff_possible=*/false,
                    my_rank,
                    router,
                    scratch,
                    nullptr,
                    1.0);
                // The fold found anticommuting rows: they are what the join's row side is built from.
                BOOST_REQUIRE(!scratch.nz.empty());
                BOOST_REQUIRE_EQUAL(res.queries.size(), ranks);
                BOOST_REQUIRE_EQUAL(res.sent.size(), ranks);
                // The scan routes a self-owned partner to the stage, so my own bucket must be empty,
                // and the stage's parallel record list is the one for the self slot.
                BOOST_REQUIRE(res.queries[my_rank].empty());
                BOOST_REQUIRE_EQUAL(res.sent[my_rank].size(), res.self.size());
                // One record per sending source, so the per-slot source lists account for every record
                // and are ascending in row: the absence pass and the graph sink's out lists rely on that.
                for (size_t r = 0; r < ranks; ++r) {
                    std::vector<TermIndex> rows;
                    for (const auto &rec : res.sent[r]) {
                        rows.push_back(rec.row);
                    }
                    BOOST_REQUIRE(std::is_sorted(rows.begin(), rows.end()));
                    BOOST_REQUIRE((std::adjacent_find(rows.begin(), rows.end()) == rows.end()));
                    if (r != my_rank) {
                        BOOST_REQUIRE_EQUAL(
                            detail::QueryWire<kN>::count_queries(res.queries[r], detail::QueryForm::Plain),
                            rows.size());
                    }
                    for (const auto row : rows) {
                        BOOST_REQUIRE(static_cast<size_t>(row) < op.store->size());
                    }
                }
                check_bucket_ownership(res.queries, router, checked);
                check_self_ownership(res.self, router, my_rank, self_checked);
            }
            BOOST_TEST_MESSAGE("ranks=" << ranks << " linear=" << router.is_linear() << " shift=" << shift
                                        << " encoded=" << checked << " staged=" << self_checked);
            const size_t total = checked + self_checked;
            if (first_total.has_value()) {
                BOOST_TEST(total == *first_total); // routing moves partners, it does not create or lose them
            }
            else {
                first_total = total;
            }
            // The floors sit below the observed minimum of each arm and only catch a scan that emitted
            // nothing; the invariance check above is the sharp one.
            BOOST_TEST(total > 300U);
            if (!router.is_linear()) {
                // Splitmix re-hashes the partner, so both sides are populated at every rank count.
                BOOST_TEST(checked > 150U);
                BOOST_TEST(self_checked > 40U);
            }
            else if (shift != 0) {
                // Fanout 1: every partner leaves for my_rank ^ shift, so nothing stays self-owned.
                BOOST_TEST(self_checked == 0U);
            }
            else {
                BOOST_TEST(checked == 0U); // a shift of zero keeps every partner on its own rank
            }
            ++routers;
        }
    }
    BOOST_TEST(routers == 6U); // the floors above are per router, so the router count is part of them
}

// A Schrodinger propagator seeds a slot by walking the whole paired basis and keeping find_rank ==
// my_slot, on every slot independently. Two properties make that a partition of the basis: the kept
// sets are disjoint and covering, and each slot's kept sequence is a SUBSEQUENCE of the one global
// walk, so a term's row index is fixed by its owning slot alone. Neither call site can check this.
BOOST_AUTO_TEST_CASE(mpi_utils_paired_enumeration_partitions_by_find_rank) {
    constexpr size_t kN = 32;
    constexpr size_t kLogical = 9;
    constexpr size_t kMaxPairs = 4;

    MonomialList<kN> full;
    for_each_paired_monomial<kN>(kMaxPairs, kLogical, [&](const Monomial<kN> &mono) { full.push_back(mono); });
    BOOST_REQUIRE_EQUAL(full.size(), paired_op_size(kMaxPairs, kLogical));

    for (const size_t slots : {size_t{1}, size_t{2}, size_t{3}, size_t{8}, size_t{128}}) {
        // Both routers where the geometry admits both: the partition property is a property of the walk
        // and the owner function, so it must not depend on which owner function is in force.
        for (const bool linear : {false, true}) {
            if (linear && !std::has_single_bit(slots)) {
                continue; // linear routing refuses a non-power-of-two world (routing_tests pins the throw)
            }
            const auto router = routing::Router::for_modes<kN>(slots, /*partitions=*/1, linear);
            size_t total = 0;
            for (size_t slot = 0; slot < slots; ++slot) {
                MonomialList<kN> kept;
                for_each_paired_monomial<kN>(kMaxPairs, kLogical, [&](const Monomial<kN> &mono) {
                    if (find_rank<kN>(mono, router) == slot) {
                        kept.push_back(mono);
                    }
                });
                total += kept.size();

                // kept[r] == full[j_r] with j_0 < j_1 < ... : row r's monomial follows from the global order.
                size_t cursor = 0;
                for (const auto &mono : kept) {
                    while (cursor < full.size() && !(full[cursor] == mono)) {
                        ++cursor;
                    }
                    BOOST_REQUIRE_LT(cursor, full.size());
                    ++cursor;
                }
            }
            // The walk is duplicate-free (pinned in majorana_cutoff_tests), so equal counts mean the slots'
            // kept sets are disjoint AND cover every term exactly once.
            BOOST_TEST_CONTEXT("slots=" << slots << " linear=" << linear) {
                BOOST_CHECK_EQUAL(total, full.size());
            }
        }
    }
}

// One environment across the world resolves one router, so the agreement check must pass. The value of
// the case is under the MPI variants (2 ranks): the digests are built from the mode, the seed and the
// partition count and reduced with two allreduces, so a rank that never entered them would hang here
// rather than return -- which is the failure this check exists to convert into an exception.
BOOST_AUTO_TEST_CASE(mpi_utils_routing_agreement_holds_across_the_world) {
    const monoprop::mpi::Comm comm{MPI_COMM_WORLD};
    BOOST_CHECK_NO_THROW(check_routing_agreement(comm));
    // MPI_COMM_SELF is a world of one: the check returns before any collective, so it is safe to call
    // on a communicator the other ranks are not in.
    BOOST_CHECK_NO_THROW(check_routing_agreement(monoprop::mpi::Comm{MPI_COMM_SELF}));
}
