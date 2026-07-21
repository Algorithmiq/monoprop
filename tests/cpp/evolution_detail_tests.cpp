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

// Unit coverage of two small, load-bearing build-time helpers that are otherwise only exercised
// deep inside build_layer: MatchedEpochSet (the O(1)-clear follower-mark set) and CutoffContext
// (the atol / upper-atol gating predicates). Both are pure and stateful-in-isolation, so a direct
// test pins their contract without spinning up a propagator. The query/value codecs in the same
// headers are covered by fused_query_codec_tests.cpp and not duplicated here.

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <limits>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"

using namespace monoprop;
using monoprop::detail::CutoffContext;
using monoprop::detail::MatchedEpochSet;

// begin_gate is an O(1) clear: a mark set in one gate must not survive into the next.
BOOST_AUTO_TEST_CASE(matched_epoch_begin_gate_clears_all) {
    MatchedEpochSet set;
    set.begin_gate(5);
    set.mark(2);
    set.mark(4);
    BOOST_TEST(set.is_marked(2));
    BOOST_TEST(set.is_marked(4));
    BOOST_TEST(!set.is_marked(0));

    set.begin_gate(5); // one counter bump -> every prior mark clears
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

    set.begin_gate(8); // grew from 4 to 8 slots
    BOOST_TEST(!set.is_marked(3)); // old mark cleared by the epoch bump
    set.mark(7);                   // new tail slot works
    BOOST_TEST(set.is_marked(7));
    BOOST_TEST(!set.is_marked(3));
}

// When the epoch counter saturates uint32_t, begin_gate zero-fills and restarts so marks stay correct.
BOOST_AUTO_TEST_CASE(matched_epoch_u32_wrap_resets) {
    MatchedEpochSet set;
    set.begin_gate(4); // allocate the backing array
    // Force the counter to the wrap boundary; a stale slot still equals the pre-wrap counter.
    set.cur_ = std::numeric_limits<uint32_t>::max();
    set.mark(1);
    BOOST_TEST(set.is_marked(1));

    set.begin_gate(4); // triggers the fill(0) + cur_ = 0 -> ++cur_ = 1 reset
    BOOST_TEST(set.cur_ == 1U);
    BOOST_TEST(!set.is_marked(1)); // the stale UINT32_MAX slot must not read as marked
    set.mark(2);
    BOOST_TEST(set.is_marked(2));
}

// abs_coeff_for gates on use_coeff_checks and bounds the index.
BOOST_AUTO_TEST_CASE(cutoff_context_abs_coeff_for) {
    const VecD coeffs{-3.0, 2.0, 0.0};

    CutoffContext off; // use_coeff_checks defaults false
    BOOST_TEST(off.abs_coeff_for(0, coeffs) == 0.0);

    CutoffContext on;
    on.use_coeff_checks = true;
    BOOST_TEST(on.abs_coeff_for(0, coeffs) == 3.0); // |−3|
    BOOST_TEST(on.abs_coeff_for(1, coeffs) == 2.0);
    BOOST_TEST(on.abs_coeff_for(3, coeffs) == 0.0); // out of range -> 0
}

// is_above_upper is the rescue predicate: enabled AND |sin|·|coeff| >= upper_atol (inclusive).
BOOST_AUTO_TEST_CASE(cutoff_context_is_above_upper) {
    CutoffContext ctx;
    ctx.abs_sin_val = 0.5;

    ctx.check_upper_atol = false;
    BOOST_TEST(!ctx.is_above_upper(100.0)); // disabled -> never rescues

    ctx.check_upper_atol = true;
    ctx.upper_atol_value = 1.0;
    BOOST_TEST(ctx.is_above_upper(2.0));  // 0.5*2.0 == 1.0 -> boundary inclusive
    BOOST_TEST(ctx.is_above_upper(4.0));  // 0.5*4.0 == 2.0 >= 1.0
    BOOST_TEST(!ctx.is_above_upper(1.0)); // 0.5*1.0 == 0.5 < 1.0
}

// is_below_sin is the lower-atol drop predicate: enabled AND |sin|·|coeff| <= atol (inclusive).
BOOST_AUTO_TEST_CASE(cutoff_context_is_below_sin) {
    CutoffContext ctx;
    ctx.abs_sin_val = 2.0;

    ctx.check_atol = false;
    BOOST_TEST(!ctx.is_below_sin(0.0)); // disabled -> never drops

    ctx.check_atol = true;
    ctx.atol_value = 1.0;
    BOOST_TEST(ctx.is_below_sin(0.5));  // 2.0*0.5 == 1.0 -> boundary inclusive
    BOOST_TEST(ctx.is_below_sin(0.1));  // 2.0*0.1 == 0.2 <= 1.0
    BOOST_TEST(!ctx.is_below_sin(1.0)); // 2.0*1.0 == 2.0 > 1.0
}
