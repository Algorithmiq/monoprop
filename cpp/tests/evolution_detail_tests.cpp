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

// The self-resolve pass over the per-gate AntiTable and CutoffContext, driven directly, not through build_layer.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/AntiTable.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

using namespace monoprop;
using monoprop::detail::AntiTable;
using monoprop::detail::CutoffContext;

namespace {

// resolve_range_ touches only wants_values and self_hit, so the engine drives without the cross-rank sink surface.
struct RecordingSink {
    static constexpr bool wants_values = false;
    std::vector<std::pair<size_t, size_t>> hits; // (src, found)
    auto self_hit(size_t src, size_t found, int /*phase*/, double /*v_src*/) -> void { hits.emplace_back(src, found); }
};

auto indexed_op(const std::vector<Monomial<8>> &terms) -> detail::MPOperator<8> {
    detail::MPOperator<8> op;
    detail::insert_absent_terms<8>(op, terms.size(), [&](size_t k, size_t base) {
        assign_row<8>(*op.store, base + k, terms[k]);
    });
    return op;
}

// The partner table the scan would build for a gate that finds rows [0, n) anticommuting.
auto table_over_rows(detail::GateScratch<8> &scratch, const detail::MPOperator<8> &op, size_t n) -> void {
    const uint64_t *labels = routing::linear_basis<16>().data();
    scratch.anti.begin(n);
    for (size_t i = 0; i < n; ++i) {
        const auto src = op.store->row_positions(i);
        scratch.anti.add(static_cast<TermIndex>(i),
                         routing::fingerprint_positions(labels, src.pos.data(), src.pos.size()));
    }
}

} // namespace

// Self-resolve against the gate's partner table: a query whose term is in the table is a hit and, on the
// leader pass, marks that term (the follower pass then skips it); a query whose term is absent is a
// deferred miss. Rows the table does not hold -- terms this layer inserted -- are misses by construction.
BOOST_AUTO_TEST_CASE(self_resolve_hits_mark_only_on_the_leader_pass) {
    std::vector<Monomial<8>> terms;
    for (size_t i = 0; i < 6; ++i) {
        terms.push_back(indices_to_bitset<8>({i, i + 8}));
    }
    detail::MPOperator<8> op = indexed_op(terms);
    const size_t combined_size = 6;

    detail::GateScratch<8> scratch;
    table_over_rows(scratch, op, combined_size);

    detail::LayerBuildEngine<8, RecordingSink> eng(op,
                                                   mpi::Comm{},
                                                   /*R_=*/1,
                                                   /*my_rank_=*/0,
                                                   scratch,
                                                   combined_size,
                                                   RecordingSink{});
    // The self leg is staged as positions, never encoded, so this feeds the stage the scan would fill.
    using Eng = detail::LayerBuildEngine<8, RecordingSink>;
    const uint64_t *labels = routing::linear_basis<16>().data();
    const auto stage_self = [&](const Monomial<8> &m, int phase) {
        std::vector<Eng::RowPosT> pos;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            pos.push_back(static_cast<Eng::RowPosT>(b));
        }
        eng.self_stage_.push(pos, phase, routing::fingerprint_positions(labels, pos.data(), pos.size()));
    };
    stage_self(terms[1], 1);
    stage_self(terms[5], -1);
    stage_self(indices_to_bitset<8>({2, 3}), 1); // absent: a miss
    eng.src_idx_r.at_slot(0) = {0, 2, 4};

    eng.resolve_self_queries(/*is_leader_pass=*/true);

    BOOST_TEST_REQUIRE(eng.sink.hits.size() == 2U);
    BOOST_TEST(eng.sink.hits[0].second == 1U);
    BOOST_TEST(eng.sink.hits[1].second == 5U);
    BOOST_TEST_REQUIRE(eng.deferred_self_misses.size() == 1U);
    BOOST_TEST(eng.deferred_self_misses[0].src == 4U);
    BOOST_TEST(scratch.anti.is_marked_row(1));
    BOOST_TEST(scratch.anti.is_marked_row(5));
    BOOST_TEST(!scratch.anti.is_marked_row(0));
    BOOST_TEST(!scratch.anti.is_marked_row(4));

    // The follower pass skips a marked source and resolves the rest; nothing is marked on it.
    eng.self_stage_.clear();
    stage_self(terms[2], 1);
    stage_self(terms[3], 1);
    eng.src_idx_r.at_slot(0) = {1, 0}; // source 1 was matched by a leader → dropped; source 0 was not
    eng.sink.hits.clear();
    eng.resolve_self_queries(/*is_leader_pass=*/false);
    BOOST_TEST_REQUIRE(eng.sink.hits.size() == 1U);
    BOOST_TEST(eng.sink.hits[0].first == 0U);
    BOOST_TEST(eng.sink.hits[0].second == 3U);
    BOOST_TEST(!scratch.anti.is_marked_row(3));
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
