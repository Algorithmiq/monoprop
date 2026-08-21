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

// MatchedEpochSet and CutoffContext driven directly, not through build_layer.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <limits>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"

using namespace monoprop;
using monoprop::detail::CutoffContext;
using monoprop::detail::MatchedEpochSet;

TEST_CASE("matched_epoch_begin_gate_clears_all") {
    MatchedEpochSet set;
    set.begin_gate(5);
    set.mark(2);
    set.mark(4);
    CHECK(set.is_marked(2));
    CHECK(set.is_marked(4));
    CHECK(!set.is_marked(0));

    set.begin_gate(5);
    CHECK(!set.is_marked(2));
    CHECK(!set.is_marked(4));
    set.mark(0);
    CHECK(set.is_marked(0));
    CHECK(!set.is_marked(2));
}

// Growing the operator only appends to the tail; old slots stay cleared and new slots are usable.
TEST_CASE("matched_epoch_tail_grow") {
    MatchedEpochSet set;
    set.begin_gate(4);
    set.mark(3);
    CHECK(set.is_marked(3));

    set.begin_gate(8);
    CHECK(!set.is_marked(3));
    set.mark(7);
    CHECK(set.is_marked(7));
    CHECK(!set.is_marked(3));
}

// When the epoch counter saturates uint32_t, begin_gate zero-fills and restarts so marks stay correct.
TEST_CASE("matched_epoch_u32_wrap_resets") {
    MatchedEpochSet set;
    set.begin_gate(4); // allocate the backing array
    // Force the counter to the wrap boundary; a stale slot still equals the pre-wrap counter.
    set.cur_ = std::numeric_limits<uint32_t>::max();
    set.mark(1);
    CHECK(set.is_marked(1));

    set.begin_gate(4); // triggers the fill(0) + cur_ = 0 -> ++cur_ = 1 reset
    CHECK(set.cur_ == 1U);
    CHECK(!set.is_marked(1));
    set.mark(2);
    CHECK(set.is_marked(2));
}

TEST_CASE("cutoff_context_abs_coeff_for") {
    const VecD coeffs{-3.0, 2.0, 0.0};

    CutoffContext off; // use_coeff_checks defaults false
    CHECK(off.abs_coeff_for(0, coeffs) == 0.0);

    CutoffContext on;
    on.use_coeff_checks = true;
    CHECK(on.abs_coeff_for(0, coeffs) == 3.0);
    CHECK(on.abs_coeff_for(1, coeffs) == 2.0);
    CHECK(on.abs_coeff_for(3, coeffs) == 0.0); // out of range -> 0
}

// is_above_upper is the rescue predicate: enabled AND |sin|·|coeff| >= upper_atol (inclusive).
TEST_CASE("cutoff_context_is_above_upper") {
    CutoffContext ctx;
    ctx.abs_sin_val = 0.5;

    ctx.check_upper_atol = false;
    CHECK(!ctx.is_above_upper(100.0));

    ctx.check_upper_atol = true;
    ctx.upper_atol_value = 1.0;
    CHECK(ctx.is_above_upper(2.0)); // 0.5*2.0 == 1.0 -> boundary inclusive
    CHECK(ctx.is_above_upper(4.0));
    CHECK(!ctx.is_above_upper(1.0));
}

// is_below_sin is the lower-atol drop predicate: enabled AND |sin|·|coeff| <= atol (inclusive).
TEST_CASE("cutoff_context_is_below_sin") {
    CutoffContext ctx;
    ctx.abs_sin_val = 2.0;

    ctx.check_atol = false;
    CHECK(!ctx.is_below_sin(0.0));

    ctx.check_atol = true;
    ctx.atol_value = 1.0;
    CHECK(ctx.is_below_sin(0.5)); // 2.0*0.5 == 1.0 -> boundary inclusive
    CHECK(ctx.is_below_sin(0.1));
    CHECK(!ctx.is_below_sin(1.0));
}
