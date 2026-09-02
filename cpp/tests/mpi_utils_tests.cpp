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
#include "monoprop/detail/mpi/Comm.h"
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

// A zero-rank world is degenerate: Router clamps it to one slot, so the owner is rank 0 rather than a
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

auto check_bucket_ownership(const mpi::WindowVec<VecZ> &buckets, const routing::Router &router, size_t &checked)
    -> void {
    // Every offset comes from the record walk: widths vary, so a hardcoded stride would compare a monomial
    // decoded at the wrong offset against the wrong rank. The bucket index is re-based, so the rank
    // compared against is window.slot(k) -- a wrong window base shows up here.
    using QW = detail::QueryWire<32>;
    const detail::QueryForm form = detail::QueryForm::Plain;
    const mpi::SlotWindow w = buckets.window();
    for (size_t k = 0; k < w.count; ++k) {
        const mpi::WindowIndex wi{k};
        const VecZ &bucket = buckets[wi];
        size_t off = 0;
        while (off < bucket.size()) {
            const size_t kq = QW::k_at(bucket, off);
            std::vector<QW::PosT> pos(kq);
            (void)QW::read_positions(bucket, off, pos);
            Monomial<32> mono;
            for (size_t j = 0; j < kq; ++j) {
                mono.set(static_cast<size_t>(pos[j]));
            }
            BOOST_REQUIRE_EQUAL(find_rank<32>(mono, router), w.slot(wi));
            off = QW::next_off(bucket, form, off);
            ++checked;
        }
        BOOST_REQUIRE_EQUAL(off, bucket.size());
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

    // Each rank is handed exactly the terms it owns, as MonomialPropagator seeds it: the emit path routes
    // by rank(M) ^ rank_shift(G), which is only the owner of M^G when M really is local.
    //
    // The partner count is a property of the operator and the gate, not of where the partners live, so
    // the sum over the ranks is the same for every router; routing only moves a partner between ranks and
    // between the encoded (cross-rank) and staged (self-owned) sides. Pinning that invariance is stronger
    // than a floor: a routing bug that drops partners moves the total, and one that misroutes them moves
    // the split.
    size_t routers = 0;
    std::optional<size_t> first_total;
    for (const size_t ranks : {2U, 4U, 8U}) {
        // BOTH routers, because the agreement is a property of the pair and not of either hash: the scan
        // routes off the partner's fingerprint (Router::dest_from_fingerprint) or the dense hash, and
        // find_rank calls Router::dest, so a divergence introduced by one of them shows up here whichever
        // routing the geometry resolves to.
        for (const bool linear : {false, true}) {
            const auto router = routing::Router::for_modes<kN>(ranks, /*partitions=*/1, linear);
            const size_t shift = router.rank_shift<kN>(gen);
            const mpi::PeerPlan plan{.sparse = router.is_linear(), .shift = static_cast<int>(shift)};
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
                const mpi::SlotWindow window = plan.window(my_rank, ranks, /*parts=*/1);
                VecD coeffs(op.store->size(), 1.0);
                const auto cut = detail::build_majorana_evolution_cutoff_state(std::nullopt,
                                                                               std::cref(coeffs),
                                                                               std::nullopt,
                                                                               std::optional<double>{0.3});
                detail::GateScratch<kN> scratch;
                const auto res = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>>(op,
                                                                                         gen,
                                                                                         eval,
                                                                                         cut,
                                                                                         coeffs,
                                                                                         std::nullopt,
                                                                                         /*over_cutoff_possible=*/false,
                                                                                         window,
                                                                                         my_rank,
                                                                                         router,
                                                                                         scratch,
                                                                                         false,
                                                                                         nullptr,
                                                                                         1.0);
                // Every anticommuting row is in the table, emitted or not.
                BOOST_REQUIRE(scratch.anti.size() > 0U);
                BOOST_REQUIRE_EQUAL(res.queries.size(), window.count);
                BOOST_REQUIRE_EQUAL(res.sent.size(), window.count);
                if (window.contains(my_rank)) {
                    // The scan routes a self-owned partner to the stage, so my own bucket must be empty,
                    // and the stage's parallel ordinal list is the one for the self slot.
                    BOOST_REQUIRE(res.queries.at_slot(my_rank).empty());
                    BOOST_REQUIRE_EQUAL(res.sent.at_slot(my_rank).size(), res.self.size());
                }
                else {
                    // A non-zero shift puts self outside the window entirely, so nothing may be staged.
                    BOOST_REQUIRE_EQUAL(res.self.size(), 0U);
                }
                // One record per emitting source, so the per-slot ordinal lists account for every record
                // and are ascending: the absence pass and the graph sink's out lists rely on that.
                for (size_t k = 0; k < window.count; ++k) {
                    const mpi::WindowIndex wi{k};
                    const auto &ords = res.sent[wi];
                    BOOST_REQUIRE(std::is_sorted(ords.begin(), ords.end()));
                    BOOST_REQUIRE((std::adjacent_find(ords.begin(), ords.end()) == ords.end()));
                    if (window.slot(wi) != my_rank) {
                        BOOST_REQUIRE_EQUAL(
                            detail::QueryWire<kN>::count_queries(res.queries[wi], detail::QueryForm::Plain),
                            ords.size());
                    }
                    for (const auto ord : ords) {
                        BOOST_REQUIRE(ord < scratch.anti.size());
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
            // Measured, all six routers: total 387 every time, with the splitmix split running from
            // 212/175 at R=2 to 343/44 at R=8 as more partners fall cross-rank, and the linear arm all
            // encoded (shift 1, 3, 7). The floors sit below the observed minimum of each arm and only
            // catch a scan that emitted nothing.
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
