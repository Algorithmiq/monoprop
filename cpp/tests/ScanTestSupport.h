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

#pragma once

// Test-only support for the threaded bitmap scan: a synthetic operator spanning several fold blocks per rank,
// scan scenarios over windows, routers, pivots, bases and cutoffs, and an exact comparison of two scan
// results stream by stream. Every comparison is on the caller, after the scan has joined.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "KernelTestSupport.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace scan_test {

using namespace monoprop;

inline constexpr size_t kModes = 32; // 64 bits, so a monomial is one word
using Mono = Monomial<kModes>;
using Op = detail::MPOperator<kModes>;
using Result = detail::FusedScanResult<kModes>;

// Column tiers of the synthetic operator, by bit index (the generator's pivot is its lowest set bit):
inline constexpr size_t kEmptyEnd = 4;   // bits [0, 4) are never set: empty fold columns
inline constexpr size_t kSparseEnd = 12; // bits [4, 12) are set with probability 1/256: sparse columns
// bits [12, 64) are set with probability 1/8: dense columns (the promotion threshold is 1/64).

// Three full fold blocks and a one-word fourth, whose last word holds 37 rows: budgets 2-4 split it into
// 2-4 ranges, and the last range carries the global tail.
inline constexpr size_t kRowsPerRank = (3 * detail::kColumnBlockWords * 64) + 37;

inline auto draw_term(kernel_test::SplitMix &rng) -> uint64_t {
    uint64_t w = 0;
    for (size_t b = kEmptyEnd; b < 64; ++b) {
        const bool set = b < kSparseEnd ? rng.below(256) == 0 : rng.below(8) == 0;
        w |= static_cast<uint64_t>(set) << b;
    }
    return w;
}

inline auto mono_of(uint64_t word) -> Mono {
    Mono m{};
    for (uint64_t x = word; x != 0; x &= x - 1) {
        m.set(static_cast<size_t>(std::countr_zero(x)));
    }
    return m;
}

inline auto mono_of_bits(std::initializer_list<size_t> bits) -> Mono {
    Mono m{};
    for (const auto b : bits) {
        m.set(b);
    }
    return m;
}

inline auto build_op(const std::vector<Mono> &terms, Basis basis) -> Op {
    Op op;
    op.basis = basis;
    detail::insert_absent_terms<kModes>(
        op,
        terms.size(),
        [&](size_t k) -> const Mono & { return terms[k]; },
        [&](size_t k, size_t base) { assign_row<kModes>(*op.store, base + k, terms[k]); });
    return op;
}

// Distinct terms this rank owns under `router`, `rows` of them, in draw order. Owned-only, as a propagator
// seeds each rank: the scan routes by rank(M) ^ rank_shift(G), which names the owner only for a local M.
inline auto owned_terms(const routing::Router &router, size_t my_rank, size_t rows, uint64_t seed)
    -> std::vector<Mono> {
    kernel_test::SplitMix rng{seed};
    std::unordered_set<uint64_t> seen;
    std::vector<Mono> terms;
    terms.reserve(rows);
    while (terms.size() < rows) {
        const uint64_t w = draw_term(rng);
        if (w == 0 || !seen.insert(w).second) {
            continue;
        }
        const Mono m = mono_of(w);
        if (router.flat_world() == 1 || find_rank<kModes>(m, router) == my_rank) {
            terms.push_back(m);
        }
    }
    return terms;
}

// Signed coefficients in (-1, 1), deterministic.
inline auto coefficients(size_t n, uint64_t seed) -> VecD {
    kernel_test::SplitMix rng{seed};
    VecD c(n);
    for (auto &v : c) {
        v = (static_cast<double>(rng.below(1U << 30)) / static_cast<double>(1U << 29)) - 1.0;
    }
    return c;
}

// One scan geometry: a router, the rank the scan runs on, and that rank's operator.
struct Geometry {
    const char *label;
    size_t ranks;
    size_t parts;
    bool linear;
    size_t my_rank;
};

inline auto router_of(const Geometry &g) -> routing::Router {
    return routing::Router::for_modes<kModes>(g.ranks, g.parts, g.linear);
}

// Operators are expensive to build at this size, so each (geometry, basis) is built once per process.
inline auto geometry_op(const Geometry &g, Basis basis) -> const Op & {
    static std::map<std::tuple<size_t, size_t, bool, size_t, int>, Op> cache;
    const auto key = std::tuple{g.ranks, g.parts, g.linear, g.my_rank, static_cast<int>(basis)};
    auto it = cache.find(key);
    if (it == cache.end()) {
        const auto router = router_of(g);
        auto op = build_op(owned_terms(router, g.my_rank, kRowsPerRank, 0x5CA11ULL + g.my_rank), basis);
        (void)op.inverted_index();
        it = cache.emplace(key, std::move(op)).first;
    }
    return it->second;
}

enum class Atol {
    none,
    lower_equal, // lower atol equal to one term's |sin|·|c|: that term is dropped (<=)
    upper_equal, // upper atol equal to one term's |sin|·|c| under a tight cutoff: that term is rescued (>=)
    drop_all,    // lower atol above every |sin|·|c|: the scan traverses its words but emits nothing
};

// Everything one fused_find_and_collect call takes besides the operator and the budget.
struct Scenario {
    const char *label;
    Basis basis = Basis::Majorana;
    Mono gen{};
    CutoffFn<kModes> cutoff = detail::LengthCutoff<kModes>{2 * kModes};
    std::optional<size_t> cap = std::nullopt; // only_rotate_len_k
    Atol atol = Atol::none;
    bool capture = false;
    bool fused_scale = false; // requires capture and no cap
    double param = 0.3;
};

// A scan's outputs together with the coefficients it may have scaled.
struct Outcome {
    Result result;
    VecD coeffs;
    mpi::SlotWindow window;
    size_t my_rank = 0;
};

inline constexpr size_t kAtolTerm = 1234; // the term whose |sin|·|c| an atol scenario lands on exactly

// Runs `s` on `op`, which must be the operator of geometry `g` (or one built the same way).
template <class Observer = detail::NoRangeObserver>
auto run_scan_on(const Op &op,
                 const Geometry &g,
                 const Scenario &s,
                 detail::parallel::Options options,
                 const Observer &observer = {}) -> Outcome {
    const auto router = router_of(g);
    const size_t shift = router.rank_shift<kModes>(s.gen);
    const mpi::PeerPlan plan{.sparse = router.is_linear(), .shift = static_cast<int>(shift)};
    Outcome out;
    out.window = plan.window(g.my_rank, router.ranks(), router.partitions());
    out.my_rank = g.my_rank;
    out.coeffs = coefficients(op.size(), 0xC0EFFULL + g.my_rank);
    const double abs_sin = std::abs(std::sin(2 * s.param));
    std::optional<double> lower;
    std::optional<double> upper;
    if (s.atol == Atol::lower_equal) {
        lower = abs_sin * std::abs(out.coeffs[kAtolTerm]);
    }
    if (s.atol == Atol::upper_equal) {
        upper = abs_sin * std::abs(out.coeffs[kAtolTerm]);
    }
    if (s.atol == Atol::drop_all) {
        lower = 2.0;
    }
    const auto cut_st =
        detail::build_majorana_evolution_cutoff_state(lower, std::cref(out.coeffs), upper, std::optional{s.param});
    const detail::CutoffEvaluator<kModes> eval(s.cutoff);
    double *const sweep = s.fused_scale ? out.coeffs.data() : nullptr;
    const double cos_val = std::cos(2 * s.param);
    out.result = with_algebra<kModes>(s.basis, [&]<typename A>() {
        return detail::fused_find_and_collect<kModes, A>(op,
                                                         s.gen,
                                                         eval,
                                                         cut_st,
                                                         out.coeffs,
                                                         s.cap,
                                                         out.window,
                                                         g.my_rank,
                                                         router,
                                                         shift,
                                                         s.capture,
                                                         sweep,
                                                         cos_val,
                                                         options,
                                                         observer);
    });
    return out;
}

template <class Observer = detail::NoRangeObserver>
auto run_scan(const Geometry &g, const Scenario &s, detail::parallel::Options options, const Observer &observer = {})
    -> Outcome {
    return run_scan_on(geometry_op(g, s.basis), g, s, options, observer);
}

// A fresh operator for `g`: its inverted index and row-parity cache have never been built.
inline auto cold_op(const Geometry &g, Basis basis) -> Op {
    return build_op(owned_terms(router_of(g), g.my_rank, kRowsPerRank, 0x5CA11ULL + g.my_rank), basis);
}

inline auto bitwise_equal(std::span<const double> a, std::span<const double> b) -> bool {
    // memcmp must not see an empty vector's null data pointer, even for zero bytes.
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// The concatenated cosine set, as LayerBuildEngine consumes it.
inline auto concatenated_cos(const Result &r) -> CosMask {
    CosMask all;
    for (const auto &m : r.cos_blocks) {
        all.total_count += m.total_count;
        all.blocks.insert(all.blocks.end(), m.blocks.begin(), m.blocks.end());
    }
    return all;
}

// Total queries a result emitted, over every stream.
inline auto emitted(const Result &r) -> size_t {
    size_t n = 0;
    for (const auto &v : r.leader_src) {
        n += v.size();
    }
    for (const auto &v : r.follower_src) {
        n += v.size();
    }
    return n;
}

// The self stage's logical content only: never its capacity tail.
inline auto check_same_stage(const detail::SelfQueryStage<kModes> &a,
                             const detail::SelfQueryStage<kModes> &b,
                             const char *which) -> void {
    BOOST_TEST_CONTEXT(which) {
        BOOST_TEST_REQUIRE(a.size() == b.size());
        BOOST_TEST_REQUIRE(a.positions() == b.positions());
        for (size_t q = 0; q < a.size(); ++q) {
            if (a.pos_off[q] != b.pos_off[q] || a.k_of[q] != b.k_of[q] || a.phase_of[q] != b.phase_of[q]) {
                BOOST_TEST_REQUIRE(false, "self query " << q << " differs");
            }
        }
        BOOST_TEST(std::equal(a.pos_flat.begin(),
                              a.pos_flat.begin() + static_cast<std::ptrdiff_t>(a.positions()),
                              b.pos_flat.begin()));
    }
}

// Checks one result's own invariants: every populated stream carries the scan's window, value streams are
// empty without capture, self wire stays empty, and the self stages align with the self slot's sources.
inline auto check_shape(const Outcome &o, bool capture) -> void {
    const Result &r = o.result;
    BOOST_TEST((r.leader_queries.window() == o.window));
    BOOST_TEST((r.follower_queries.window() == o.window));
    BOOST_TEST((r.leader_src.window() == o.window));
    BOOST_TEST((r.follower_src.window() == o.window));
    if (capture) {
        BOOST_TEST((r.leader_val.window() == o.window));
        BOOST_TEST((r.follower_val.window() == o.window));
    }
    else {
        BOOST_TEST(r.leader_val.size() == 0U);
        BOOST_TEST(r.follower_val.size() == 0U);
    }
    if (o.window.contains(o.my_rank)) {
        BOOST_TEST(r.leader_queries.at_slot(o.my_rank).empty());
        BOOST_TEST(r.follower_queries.at_slot(o.my_rank).empty());
        BOOST_TEST(r.leader_self.size() == r.leader_src.at_slot(o.my_rank).size());
        BOOST_TEST(r.follower_self.size() == r.follower_src.at_slot(o.my_rank).size());
        if (capture) {
            BOOST_TEST(r.leader_self.size() == r.leader_val.at_slot(o.my_rank).size());
            BOOST_TEST(r.follower_self.size() == r.follower_val.at_slot(o.my_rank).size());
        }
    }
    else {
        BOOST_TEST(r.leader_self.size() == 0U);
        BOOST_TEST(r.follower_self.size() == 0U);
    }
    for (const auto wi : o.window.indices()) {
        if (o.window.slot(wi) != o.my_rank) { // a remote slot: one wire record, at least one word, per source
            BOOST_TEST(r.leader_queries[wi].size() >= r.leader_src[wi].size());
            BOOST_TEST(r.follower_queries[wi].size() >= r.follower_src[wi].size());
        }
        if (capture) {
            BOOST_TEST(r.leader_val[wi].size() == r.leader_src[wi].size());
            BOOST_TEST(r.follower_val[wi].size() == r.follower_src[wi].size());
        }
    }
    // Ascending and disjoint across the concatenation, which LayerBuildEngine relies on.
    const auto cos = concatenated_cos(r);
    size_t bits = 0;
    for (size_t k = 0; k < cos.blocks.size(); ++k) {
        bits += static_cast<size_t>(std::popcount(cos.blocks[k].second));
        if (k != 0 && cos.blocks[k - 1].first >= cos.blocks[k].first) {
            BOOST_TEST_REQUIRE(false, "cosine blocks not ascending at " << k);
        }
    }
    BOOST_TEST(bits == cos.total_count);
}

// Exact equality of two scans of the same inputs: every stream of every window slot, both self stages,
// the concatenated cosine set, and the (possibly scaled) coefficients.
inline auto check_same_scan(const Outcome &a, const Outcome &b, bool capture) -> void {
    BOOST_TEST_REQUIRE((a.window == b.window));
    check_shape(a, capture);
    check_shape(b, capture);
    for (const auto wi : a.window.indices()) {
        BOOST_TEST_CONTEXT("slot " << a.window.slot(wi)) {
            BOOST_TEST((a.result.leader_queries[wi] == b.result.leader_queries[wi]));
            BOOST_TEST((a.result.follower_queries[wi] == b.result.follower_queries[wi]));
            BOOST_TEST((a.result.leader_src[wi] == b.result.leader_src[wi]));
            BOOST_TEST((a.result.follower_src[wi] == b.result.follower_src[wi]));
            if (capture) {
                BOOST_TEST(bitwise_equal(a.result.leader_val[wi], b.result.leader_val[wi]));
                BOOST_TEST(bitwise_equal(a.result.follower_val[wi], b.result.follower_val[wi]));
            }
        }
    }
    check_same_stage(a.result.leader_self, b.result.leader_self, "leader_self");
    check_same_stage(a.result.follower_self, b.result.follower_self, "follower_self");
    const auto ca = concatenated_cos(a.result);
    const auto cb = concatenated_cos(b.result);
    BOOST_TEST(ca.total_count == cb.total_count);
    BOOST_TEST((ca.blocks == cb.blocks));
    BOOST_TEST(bitwise_equal(a.coeffs, b.coeffs));
}

// FNV-1a over a result's logical content, for comparing one build's scan with another's.
inline auto digest(const Outcome &o, bool capture) -> uint64_t {
    uint64_t h = 1469598103934665603ULL;
    const auto mix = [&h](uint64_t v) {
        for (int k = 0; k < 8; ++k) {
            h = (h ^ ((v >> (8 * k)) & 0xFFU)) * 1099511628211ULL;
        }
    };
    const auto mix_d = [&mix](double d) { mix(std::bit_cast<uint64_t>(d)); };
    const Result &r = o.result;
    for (const auto wi : o.window.indices()) {
        mix(0xABCDEFULL);
        for (const auto v : r.leader_queries[wi]) {
            mix(v);
        }
        for (const auto v : r.follower_queries[wi]) {
            mix(v);
        }
        for (const auto v : r.leader_src[wi]) {
            mix(v);
        }
        for (const auto v : r.follower_src[wi]) {
            mix(v);
        }
        if (capture) {
            for (const auto v : r.leader_val[wi]) {
                mix_d(v);
            }
            for (const auto v : r.follower_val[wi]) {
                mix_d(v);
            }
        }
    }
    for (const auto *stage : {&r.leader_self, &r.follower_self}) {
        mix(stage->size());
        for (size_t q = 0; q < stage->size(); ++q) {
            mix(stage->pos_off[q]);
            mix(stage->k_of[q]);
            mix(static_cast<uint64_t>(static_cast<int64_t>(stage->phase_of[q])));
        }
        for (size_t p = 0; p < stage->positions(); ++p) {
            mix(stage->pos_flat[p]);
        }
    }
    const auto cos = concatenated_cos(r);
    mix(cos.total_count);
    for (const auto &[base, bits] : cos.blocks) {
        mix(base);
        mix(bits);
    }
    for (const auto c : o.coeffs) {
        mix_d(c);
    }
    return h;
}

enum class Expect {
    output,        // the scan emits queries from several ranges
    empty_scanned, // the scan traverses its words but emits nothing
    not_scanned,   // the caller returns before any traversal (no fold columns, or all of them empty)
};

// Runs `s` at budget 1 and at budgets 2-4 and requires identical results. Checks on the caller, after each
// scan joined, that the threaded runs split the words into min(budget, fold blocks) ranges that ran inside
// the scan's own region, and that the serial run stayed on the caller in one range.
inline auto check_threaded_matches_serial(const Geometry &g, const Scenario &s, Expect expect) -> void {
    using detail::KernelRange;
    BOOST_TEST_CONTEXT(g.label << " / " << s.label) {
        const size_t words = geometry_op(g, s.basis).inverted_index().words();
        const size_t blocks = detail::logical_ranges(words, detail::kColumnBlockWords);
        const bool safe = detail::CutoffEvaluator<kModes>(s.cutoff).parallel_safe();
        const size_t scanned = expect == Expect::not_scanned ? 0U : 1U;

        kernel_test::RangeLog serial_log;
        const auto serial = run_scan(g, s, {.threads = 1}, kernel_test::RecordingObserver{&serial_log});
        kernel_test::check_serial(serial_log[KernelRange::scan], scanned, "serial scan");
        if (expect == Expect::output) {
            BOOST_TEST_REQUIRE(emitted(serial.result) > 0U);
            // Sources in at least two of the four budget-4 ranges, so the merge really joins pieces.
            std::set<size_t> ranges_hit;
            const size_t k = std::min<size_t>(4, blocks);
            for (const auto *streams : {&serial.result.leader_src, &serial.result.follower_src}) {
                for (const auto &v : *streams) {
                    for (const auto src : v) {
                        for (size_t r = 0; r < k; ++r) {
                            const auto [lo, hi] = detail::scan_range_words(r, k, words);
                            if (src >= lo * 64 && src < hi * 64) {
                                ranges_hit.insert(r);
                            }
                        }
                    }
                }
            }
            BOOST_TEST(ranges_hit.size() >= 2U);
        }
        else {
            BOOST_TEST(emitted(serial.result) == 0U);
        }
        for (const int threads : {2, 3, 4}) {
            BOOST_TEST_CONTEXT("budget " << threads) {
                kernel_test::RangeLog log;
                const auto threaded = run_scan(g, s, {.threads = threads}, kernel_test::RecordingObserver{&log});
                check_same_scan(serial, threaded, s.capture);
                const size_t expected = scanned == 0 ? 0 : (safe ? std::min<size_t>(threads, blocks) : 1);
                if (expected >= 2) {
                    kernel_test::check_participation(log[KernelRange::scan], expected, "threaded scan");
                }
                else {
                    kernel_test::check_serial(log[KernelRange::scan], expected, "serial fallback scan");
                }
            }
        }
    }
}

// A generator of the requested pivot tier and parity whose rank shift under `router` is (non)zero, found by
// a deterministic search, so a scenario does not depend on the routing basis the build happens to draw.
inline auto find_generator(const routing::Router &router,
                           size_t pivot_lo,
                           size_t pivot_hi,
                           size_t weight,
                           bool want_zero_shift,
                           uint64_t seed) -> Mono {
    kernel_test::SplitMix rng{seed};
    for (int attempt = 0; attempt < 100000; ++attempt) {
        Mono g{};
        g.set(pivot_lo + rng.below(pivot_hi - pivot_lo));
        const size_t pivot = g.find_first();
        while (g.count() < weight) {
            const size_t b = pivot + 1 + rng.below(64 - pivot - 1);
            g.set(b);
        }
        if ((router.rank_shift<kModes>(g) == 0) == want_zero_shift) {
            return g;
        }
    }
    BOOST_FAIL("no generator with the requested shift");
    return {};
}

} // namespace scan_test
