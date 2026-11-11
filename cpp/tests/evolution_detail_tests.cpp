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

// MatchedEpochSet, CutoffContext and the self-resolve mark guard driven directly, not through build_layer.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "ScanTestSupport.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

using namespace monoprop;
using monoprop::detail::CutoffContext;
using monoprop::detail::MatchedEpochSet;

namespace {

// resolve_range_ touches only wants_values and self_hit, so the engine drives without the cross-rank sink surface.
struct RecordingSink {
    static constexpr bool wants_values = false;
    using Response = TermIndex;                  // named by the engine's exchange phases, which this case never calls
    std::vector<std::pair<size_t, size_t>> hits; // (src, found)
    auto self_hit(size_t src, size_t found, int /*phase*/, double /*v_src*/) -> void { hits.emplace_back(src, found); }
};

// append_term writes a row only; find_batch needs the hash index, which insert_absent_terms populates.
auto indexed_op(const std::vector<Monomial<8>> &terms) -> detail::MPOperator<8> {
    detail::MPOperator<8> op;
    detail::insert_absent_terms<8>(
        op,
        terms.size(),
        [&](size_t k) -> const Monomial<8> & { return terms[k]; },
        [&](size_t k, size_t base) { assign_row<8>(*op.store, base + k, terms[k]); });
    return op;
}

} // namespace

BOOST_AUTO_TEST_CASE(matched_epoch_begin_gate_clears_all) {
    MatchedEpochSet set;
    set.begin_gate(5);
    set.mark(2);
    set.mark(4);
    BOOST_TEST(set.is_marked(2));
    BOOST_TEST(set.is_marked(4));
    BOOST_TEST(!set.is_marked(0));

    set.begin_gate(5);
    BOOST_TEST(!set.is_marked(2));
    BOOST_TEST(!set.is_marked(4));
    set.mark(0);
    BOOST_TEST(set.is_marked(0));
    BOOST_TEST(!set.is_marked(2));
}

// Growing the operator only appends to the tail; old slots stay cleared and new slots are usable.
BOOST_AUTO_TEST_CASE(matched_epoch_tail_grow) {
    MatchedEpochSet set;
    set.begin_gate(4);
    set.mark(3);
    BOOST_TEST(set.is_marked(3));

    set.begin_gate(8);
    BOOST_TEST(!set.is_marked(3));
    set.mark(7);
    BOOST_TEST(set.is_marked(7));
    BOOST_TEST(!set.is_marked(3));
}

// Reaches the wrap by assigning cur_, which pins the branch and the counter restart but not the fill.
BOOST_AUTO_TEST_CASE(matched_epoch_stamp_wrap_resets) {
    MatchedEpochSet set;
    set.begin_gate(4); // allocate the backing array
    // Force the counter to the wrap boundary; a stale slot still equals the pre-wrap counter.
    set.cur_ = std::numeric_limits<MatchedEpochSet::Stamp>::max();
    set.mark(1);
    BOOST_TEST(set.is_marked(1));

    set.begin_gate(4); // triggers the fill(0) + cur_ = 0 -> ++cur_ = 1 reset
    BOOST_TEST(set.cur_ == 1U);
    BOOST_TEST(!set.is_marked(1));
    set.mark(2);
    BOOST_TEST(set.is_marked(2));
}

// Reaches the wrap by counting gates, with the mark at epoch 1 so a missing fill would alias onto it.
BOOST_AUTO_TEST_CASE(matched_epoch_stamp_wrap_reached_by_gate_count) {
    constexpr auto kMaxStamp = std::numeric_limits<MatchedEpochSet::Stamp>::max();
    constexpr size_t kPeriod = static_cast<size_t>(kMaxStamp);

    MatchedEpochSet set;
    set.begin_gate(4);
    BOOST_REQUIRE(set.cur_ == MatchedEpochSet::Stamp{1});
    set.mark(1);
    BOOST_TEST(set.is_marked(1));

    // One increment per gate, folded into a single assertion rather than 65534 of them.
    bool one_epoch_per_gate = true;
    for (size_t k = 2; k <= kPeriod; ++k) {
        set.begin_gate(4);
        one_epoch_per_gate = one_epoch_per_gate && (static_cast<size_t>(set.cur_) == k);
    }
    BOOST_TEST(one_epoch_per_gate);
    BOOST_TEST(set.cur_ == kMaxStamp); // boundary reached by counting, not by assignment

    // The wrap: cur_ returns to 1, the surviving mark's own stamp, so a false is_marked(1) is the fill.
    set.begin_gate(4);
    BOOST_TEST(set.cur_ == MatchedEpochSet::Stamp{1});
    BOOST_TEST(!set.is_marked(1));
    set.mark(2);
    BOOST_TEST(set.is_marked(2));
    BOOST_TEST(!set.is_marked(1));
}

// A self-resolve hit whose index the store only grew into after construction is a real hit -- it must reach
// the sink -- but it is outside the matched set, whose array is sized to combined_size.
BOOST_AUTO_TEST_CASE(self_resolve_mark_bounded_by_combined_size) {
    std::vector<Monomial<8>> terms;
    for (size_t i = 0; i < 6; ++i) {
        terms.push_back(indices_to_bitset<8>({i, i + 8}));
    }
    detail::MPOperator<8> op = indexed_op(terms);
    const size_t combined_size = 4; // rows 4 and 5 stand for terms this layer inserted after construction

    MatchedEpochSet matched;
    // Pre-grown past combined_size on purpose: an unguarded mark then lands in an observable slot rather
    // than past the end of epoch_, where it would be silent undefined behaviour.
    matched.begin_gate(op.size());

    detail::LayerBuildEngine<8, RecordingSink> eng(op,
                                                   /*R_=*/1,
                                                   /*my_rank_=*/0,
                                                   matched,
                                                   combined_size,
                                                   RecordingSink{},
                                                   mpi::SlotWindow{.base = 0, .count = 1});
    // The self leg is staged as positions, never encoded, so this feeds the stage the scan would fill.
    using Eng = detail::LayerBuildEngine<8, RecordingSink>;
    const auto stage_self = [&eng](const Monomial<8> &m, int phase) {
        std::vector<Eng::RowPosT> pos;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            pos.push_back(static_cast<Eng::RowPosT>(b));
        }
        eng.self_stage_.push(pos, phase);
    };
    stage_self(terms[1], 1);
    stage_self(terms[5], -1);
    eng.src_idx_r.at_slot(0) = {0, 2};

    eng.resolve_self_queries(/*is_leader_pass=*/true);

    // Both keys are in the store, so both resolve as hits and neither may be deferred as a miss.
    BOOST_TEST(eng.deferred_self_misses.empty());
    BOOST_TEST_REQUIRE(eng.sink.hits.size() == 2U);
    BOOST_TEST(eng.sink.hits[0].second == 1U);
    BOOST_TEST(eng.sink.hits[1].second == 5U);
    BOOST_TEST(matched.is_marked(1));
    BOOST_TEST(!matched.is_marked(4));
    BOOST_TEST(!matched.is_marked(5));
}

BOOST_AUTO_TEST_CASE(cutoff_context_abs_coeff_for) {
    const VecD coeffs{-3.0, 2.0, 0.0};

    CutoffContext off; // use_coeff_checks defaults false
    BOOST_TEST(off.abs_coeff_for(0, coeffs) == 0.0);

    CutoffContext on;
    on.use_coeff_checks = true;
    BOOST_TEST(on.abs_coeff_for(0, coeffs) == 3.0);
    BOOST_TEST(on.abs_coeff_for(1, coeffs) == 2.0);
    BOOST_TEST(on.abs_coeff_for(3, coeffs) == 0.0); // out of range -> 0
}

// is_above_upper is the rescue predicate: enabled AND |sin|·|coeff| >= upper_atol (inclusive).
BOOST_AUTO_TEST_CASE(cutoff_context_is_above_upper) {
    CutoffContext ctx;
    ctx.abs_sin_val = 0.5;

    ctx.check_upper_atol = false;
    BOOST_TEST(!ctx.is_above_upper(100.0));

    ctx.check_upper_atol = true;
    ctx.upper_atol_value = 1.0;
    BOOST_TEST(ctx.is_above_upper(2.0)); // 0.5*2.0 == 1.0 -> boundary inclusive
    BOOST_TEST(ctx.is_above_upper(4.0));
    BOOST_TEST(!ctx.is_above_upper(1.0));
}

// is_below_sin is the lower-atol drop predicate: enabled AND |sin|·|coeff| <= atol (inclusive).
BOOST_AUTO_TEST_CASE(cutoff_context_is_below_sin) {
    CutoffContext ctx;
    ctx.abs_sin_val = 2.0;

    ctx.check_atol = false;
    BOOST_TEST(!ctx.is_below_sin(0.0));

    ctx.check_atol = true;
    ctx.atol_value = 1.0;
    BOOST_TEST(ctx.is_below_sin(0.5)); // 2.0*0.5 == 1.0 -> boundary inclusive
    BOOST_TEST(ctx.is_below_sin(0.1));
    BOOST_TEST(!ctx.is_below_sin(1.0));
}

// --- threaded bitmap scan (fused_find_and_collect over logical word ranges) ------------------------------------

// Ranges are contiguous runs of whole fold blocks, nonempty, ascending with the range ID, covering every
// word, and depend only on the word count and the budget.
BOOST_AUTO_TEST_CASE(openmp_scan_ranges_partition_words_in_order) {
    constexpr size_t kBlock = detail::kColumnBlockWords;
    for (const size_t words :
         {size_t{1}, kBlock - 1, kBlock, kBlock + 1, (3 * kBlock) + 1, 7 * kBlock, size_t{100003}}) {
        for (const int threads : {1, 2, 3, 4, 7, 96}) {
            const size_t ranges = detail::scan_ranges(words, {.threads = threads});
            BOOST_TEST_CONTEXT("words=" << words << " threads=" << threads) {
                BOOST_TEST(ranges == std::min<size_t>(threads, detail::logical_ranges(words, kBlock)));
                size_t next = 0;
                for (size_t r = 0; r < ranges; ++r) {
                    const auto [lo, hi] = detail::scan_range_words(r, ranges, words);
                    BOOST_TEST(lo == next);
                    BOOST_TEST(hi > lo);
                    BOOST_TEST((lo % kBlock) == 0U);
                    next = hi;
                }
                BOOST_TEST(next == words);
            }
        }
    }
}

// Majorana scans at budgets 2-4 against budget 1, byte for byte, across windows (one rank; linear routing
// with zero and non-zero shifts, including a two-slot window at a non-zero base; dense splitmix at four and
// three ranks), dense/sparse/empty pivots, odd generators with empty fold columns, length caps, structural
// cutoffs (including cutoff 0), lower/upper atol boundaries, scans that emit nothing, value capture and the
// fused cos sweep. only_rotate_len_k must be positive (0 is rejected before any scan).
BOOST_AUTO_TEST_CASE(openmp_scan_ranges_match_serial_majorana) {
    using scan_test::Atol;
    using scan_test::Expect;
    using scan_test::Scenario;
    const scan_test::Geometry solo{"one rank", 1, 1, false, 0};
    const auto dense_even = scan_test::mono_of_bits({14, 20, 33, 47});
    const auto sparse_even = scan_test::mono_of_bits({5, 40});
    const auto dense_odd = scan_test::mono_of_bits({13, 29, 52});
    const auto empty_odd = scan_test::mono_of_bits({0, 1, 2});
    const auto empty_even = scan_test::mono_of_bits({0, 3});
    const std::vector<std::pair<Scenario, Expect>> single{
        {{.label = "dense pivot", .gen = dense_even}, Expect::output},
        {{.label = "sparse pivot", .gen = sparse_even}, Expect::output},
        {{.label = "odd generator", .gen = dense_odd}, Expect::output},
        {{.label = "odd generator, empty fold columns", .gen = empty_odd}, Expect::output},
        {{.label = "even generator, empty fold columns", .gen = empty_even}, Expect::not_scanned},
        {{.label = "lower atol above every term", .gen = dense_even, .atol = Atol::drop_all}, Expect::empty_scanned},
        {{.label = "lower atol above every term, fused",
          .gen = dense_even,
          .atol = Atol::drop_all,
          .capture = true,
          .fused_scale = true},
         Expect::empty_scanned},
        {{.label = "length cutoff 0", .gen = dense_even, .cutoff = detail::LengthCutoff<32>{0}}, Expect::empty_scanned},
        {{.label = "length cutoff 0, upper atol rescue",
          .gen = dense_odd,
          .cutoff = detail::LengthCutoff<32>{0},
          .atol = Atol::upper_equal,
          .capture = true},
         Expect::output},
        {{.label = "cap 6", .gen = dense_even, .cap = 6}, Expect::output},
        {{.label = "cap 6, capture", .gen = dense_odd, .cap = 6, .capture = true}, Expect::output},
        {{.label = "length cutoff 5", .gen = dense_even, .cutoff = detail::LengthCutoff<32>{5}}, Expect::output},
        {{.label = "support cutoff 4", .gen = sparse_even, .cutoff = detail::SupportCutoff<32>{4}}, Expect::output},
        {{.label = "lower atol at a term", .gen = dense_even, .atol = Atol::lower_equal}, Expect::output},
        {{.label = "lower atol at a term, capture", .gen = dense_odd, .atol = Atol::lower_equal, .capture = true},
         Expect::output},
        {{.label = "upper atol rescue",
          .gen = dense_even,
          .cutoff = detail::LengthCutoff<32>{2},
          .atol = Atol::upper_equal},
         Expect::output},
        {{.label = "capture", .gen = sparse_even, .capture = true}, Expect::output},
        {{.label = "fused cos sweep", .gen = dense_even, .capture = true, .fused_scale = true}, Expect::output},
        {{.label = "fused cos sweep, odd, lower atol",
          .gen = dense_odd,
          .atol = Atol::lower_equal,
          .capture = true,
          .fused_scale = true},
         Expect::output},
    };
    for (const auto &[s, e] : single) {
        scan_test::check_threaded_matches_serial(solo, s, e);
    }

    const std::vector<scan_test::Geometry> multi{
        {"linear R=4, zero shift", 4, 1, true, 2},
        {"linear R=4, non-zero shift", 4, 1, true, 0},
        {"linear R=2 S=2, non-zero shift", 2, 2, true, 1},
        {"linear R=2 S=2, zero shift", 2, 2, true, 1},
        {"splitmix R=4", 4, 1, false, 3},
        {"splitmix R=3", 3, 1, false, 1},
    };
    for (size_t gi = 0; gi < multi.size(); ++gi) {
        const auto &g = multi[gi];
        const auto router = scan_test::router_of(g);
        const bool want_zero = gi == 0 || gi == 3 || !g.linear;
        const auto dense = g.linear ? scan_test::find_generator(router, 12, 30, 4, want_zero, 11 + gi) : dense_even;
        const auto odd = g.linear ? scan_test::find_generator(router, 12, 30, 3, want_zero, 23 + gi) : dense_odd;
        const auto sparse = g.linear ? scan_test::find_generator(router, 4, 12, 2, want_zero, 37 + gi) : sparse_even;
        if (g.linear) {
            BOOST_TEST_REQUIRE((router.rank_shift<32>(dense) == 0U) == want_zero);
        }
        const std::vector<Scenario> cases{
            {.label = "dense pivot", .gen = dense},
            {.label = "sparse pivot, capture", .gen = sparse, .capture = true},
            {.label = "odd generator, lower atol", .gen = odd, .atol = Atol::lower_equal},
            {.label = "odd generator, empty fold columns", .gen = empty_odd},
            {.label = "fused cos sweep", .gen = dense, .capture = true, .fused_scale = true},
            {.label = "cap 6, capture", .gen = odd, .cap = 6, .capture = true},
        };
        for (const auto &s : cases) {
            scan_test::check_threaded_matches_serial(g, s, Expect::output);
        }
    }
}

// A threaded scan as the first user of a fresh operator: the inverted index and the odd-generator row-parity
// cache are both cold, so building either lazily inside a worker would race (and TSan reports it). The scan
// must prepare both on the caller and still match a serial scan of an identical operator.
BOOST_AUTO_TEST_CASE(openmp_scan_prepares_cold_caches_on_the_caller) {
    const scan_test::Geometry solo{"one rank", 1, 1, false, 0};
    for (const auto &gen : {scan_test::mono_of_bits({13, 29, 52}), scan_test::mono_of_bits({0, 1, 2})}) {
        const scan_test::Scenario s{.label = "odd, cold", .gen = gen, .capture = true};
        const auto threaded_op = scan_test::cold_op(solo, Basis::Majorana);
        kernel_test::RangeLog log;
        const auto threaded =
            scan_test::run_scan_on(threaded_op, solo, s, {.threads = 4}, kernel_test::RecordingObserver{&log});
        BOOST_TEST(log[detail::KernelRange::scan].worker.size() == 4U);
        const auto serial_op = scan_test::cold_op(solo, Basis::Majorana);
        const auto serial = scan_test::run_scan_on(serial_op, solo, s, {.threads = 1});
        scan_test::check_same_scan(serial, threaded, true);
    }
}

// Worker participation in the scan's own range bodies, at the budget of the ranges' count.
BOOST_AUTO_TEST_CASE(openmp_scan_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    const scan_test::Geometry solo{"one rank", 1, 1, false, 0};
    const scan_test::Scenario s{.label = "dense pivot", .gen = scan_test::mono_of_bits({14, 20, 33, 47})};
    kernel_test::RangeLog log;
    (void)scan_test::run_scan(solo, s, {.threads = 4}, kernel_test::RecordingObserver{&log});
    kernel_test::check_participation(log[detail::KernelRange::scan], 4, "scan");
}

// A worker failure inside one scan range is joined, then rethrown on the caller; the other ranges are not
// merged into anything the caller can observe.
BOOST_AUTO_TEST_CASE(openmp_scan_worker_failure_joins_before_rethrow) {
    const scan_test::Geometry solo{"one rank", 1, 1, false, 0};
    for (const bool fused : {false, true}) {
        const scan_test::Scenario s{.label = "failure",
                                    .gen = scan_test::mono_of_bits({14, 20, 33, 47}),
                                    .capture = fused,
                                    .fused_scale = fused};
        for (const size_t failing : {size_t{0}, size_t{2}}) {
            kernel_test::RangeLog log;
            BOOST_CHECK_THROW(
                (void)scan_test::run_scan(solo,
                                          s,
                                          {.threads = 3},
                                          kernel_test::RecordingObserver{&log, detail::KernelRange::scan, failing}),
                std::runtime_error);
            const auto &slots = log[detail::KernelRange::scan];
            BOOST_TEST(slots.prepares == 1U);
            BOOST_TEST(slots.visits.size() == 3U);
            BOOST_TEST(slots.visits[failing] == 1);
        }
    }
}
