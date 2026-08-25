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

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
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

constexpr size_t kNumBits = 2 * 8;

// resolve_range_ touches only wants_values and self_hit, so the engine drives without the cross-rank sink surface.
struct RecordingSink {
    static constexpr bool wants_values = false;
    std::vector<std::pair<size_t, size_t>> hits; // (src, found)
    auto self_hit(size_t src, size_t found, int /*phase*/, double /*v_src*/) -> void { hits.emplace_back(src, found); }
};

// append_term writes a row only; find_batch needs the hash index, which insert_absent_terms populates.
auto indexed_op(const MonomialList &terms) -> detail::MPOperator {
    detail::MPOperator op(kNumBits);
    op.with_store([&](auto &rows) {
        detail::insert_absent_terms(
            op,
            rows,
            terms.size(),
            [&](size_t k) -> const Bitset & { return terms[k]; },
            [&](size_t k, size_t base) { assign_row(rows, base + k, terms[k]); });
    });
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
    MonomialList terms;
    for (size_t i = 0; i < 6; ++i) {
        terms.push_back(indices_to_bitset({i, i + 8}, kNumBits));
    }
    detail::MPOperator op = indexed_op(terms);
    const size_t combined_size = 4; // rows 4 and 5 stand for terms this layer inserted after construction

    MatchedEpochSet matched;
    // Pre-grown past combined_size on purpose: an unguarded mark then lands in an observable slot rather
    // than past the end of epoch_, where it would be silent undefined behaviour.
    matched.begin_gate(op.size());

    op.with_store([&](auto &store) {
        const size_t capacity = 0; // no generator here, so no sparse record to size
        detail::LayerBuildEngine<RecordingSink, std::remove_reference_t<decltype(store)>> eng(
            op,
            store,
            mpi::Comm{},
            /*R_=*/1,
            /*my_rank_=*/0,
            matched,
            combined_size,
            detail::query_payload_words_for(store, capacity),
            capacity,
            RecordingSink{});
        eng.queries_r[0] = detail::query_buffer();
        detail::query_push(eng.queries_r[0], terms[1], 1);
        detail::query_push(eng.queries_r[0], terms[5], -1);
        eng.src_idx_r[0] = {0, 2};

        eng.resolve_self_queries(/*is_leader_pass=*/true);

        // Both keys are in the store, so both resolve as hits and neither may be deferred as a miss.
        BOOST_TEST(eng.deferred_self_misses.empty());
        BOOST_TEST_REQUIRE(eng.sink.hits.size() == 2U);
        BOOST_TEST(eng.sink.hits[0].second == 1U);
        BOOST_TEST(eng.sink.hits[1].second == 5U);
        BOOST_TEST(matched.is_marked(1));
        BOOST_TEST(!matched.is_marked(4));
        BOOST_TEST(!matched.is_marked(5));
    });
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
