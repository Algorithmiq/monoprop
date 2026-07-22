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

// White-box unit tests for the graph-encoding packing/layout math
// (src/monoprop/detail/graph_encoding/*). These are pure/free functions, so the tests
// construct their inputs directly and check them against hand-computed oracles. The
// distributed round-trip / bit-packing paths are covered by large_cosine_storage_tests.cpp;
// this file targets the pieces that file leaves uncovered: the CosineWordBuilder coalescer,
// the checked_* overflow throws, build_layer_exchange_layout_impl, the int8 phase read, and
// both arms of the D-from-B derivation.

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"

using namespace monoprop;

// ── CosineWordBuilder: coalesce ascending indices/words into (base, mask) blocks ──────────────

BOOST_AUTO_TEST_CASE(graph_encoding_word_builder_push_index_coalesces_within_word) {
    CosineWordBuilder b;
    b.push_index(0);
    b.push_index(1);
    b.push_index(3);   // same 64-bit word (base 0)
    b.push_index(64);  // crosses into the next word -> flushes word 0
    b.push_index(197); // word base 192, bit 5
    const CosMask cos = b.finish();

    BOOST_REQUIRE_EQUAL(cos.blocks.size(), 3U);
    BOOST_CHECK_EQUAL(cos.blocks[0].first, 0U);
    BOOST_CHECK_EQUAL(cos.blocks[0].second, 0b1011ULL);
    BOOST_CHECK_EQUAL(cos.blocks[1].first, 64U);
    BOOST_CHECK_EQUAL(cos.blocks[1].second, 0b1ULL);
    BOOST_CHECK_EQUAL(cos.blocks[2].first, 192U);
    BOOST_CHECK_EQUAL(cos.blocks[2].second, uint64_t{1} << 5);
    BOOST_CHECK_EQUAL(cos.total_count, 5U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_word_builder_push_word_skips_zero_and_counts_bits) {
    CosineWordBuilder b;
    b.push_word(0, 0b101ULL); // 2 bits
    b.push_word(64, 0ULL);    // zero word: no-op, no block emitted
    b.push_word(128, 0xFULL); // 4 bits
    const CosMask cos = b.finish();

    BOOST_REQUIRE_EQUAL(cos.blocks.size(), 2U);
    BOOST_CHECK_EQUAL(cos.blocks[0].first, 0U);
    BOOST_CHECK_EQUAL(cos.blocks[1].first, 128U);
    BOOST_CHECK_EQUAL(cos.total_count, 2U + 4U);
    BOOST_CHECK_EQUAL(cos.span_count(), 2U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_word_builder_finish_flushes_pending_and_empty_is_empty) {
    // A pending word (never followed by a word-crossing push) must still be flushed by finish().
    CosineWordBuilder pending;
    pending.push_index(5);
    const CosMask cos = pending.finish();
    BOOST_REQUIRE_EQUAL(cos.blocks.size(), 1U);
    BOOST_CHECK_EQUAL(cos.blocks[0].second, uint64_t{1} << 5);

    // Nothing pushed -> empty CosMask.
    CosineWordBuilder empty;
    const CosMask none = empty.finish();
    BOOST_CHECK(none.empty());
    BOOST_CHECK_EQUAL(none.total_count, 0U);
}

// ── checked_* overflow guards ─────────────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(graph_encoding_checked_term_index_boundary) {
    // At the TermIndex ceiling it round-trips; above it throws (narrow build only — under the
    // wide build 2^32 is well within range, so it must NOT throw there).
    const size_t ceiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    BOOST_CHECK_EQUAL(detail::checked_term_index(ceiling, "term"), std::numeric_limits<TermIndex>::max());
#if !defined(monoprop_WIDE_TERM_INDEX)
    BOOST_CHECK_THROW(detail::checked_term_index(ceiling + 1, "term"), std::overflow_error);
#else
    // 2^32 is representable under the wide index: no throw.
    const size_t above_u32 = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
    BOOST_CHECK_EQUAL(detail::checked_term_index(above_u32, "term"), static_cast<TermIndex>(above_u32));
#endif
}

BOOST_AUTO_TEST_CASE(graph_encoding_checked_packed_phase_bounds) {
    BOOST_CHECK_EQUAL(detail::checked_packed_phase(127, "phase"), 127);
    BOOST_CHECK_EQUAL(detail::checked_packed_phase(-128, "phase"), -128);
    BOOST_CHECK_THROW(detail::checked_packed_phase(128, "phase"), std::overflow_error);
    BOOST_CHECK_THROW(detail::checked_packed_phase(-129, "phase"), std::overflow_error);
}

// ── PackedPhaseStorage allocation + int8 read path ─────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(graph_encoding_make_packed_phase_storage_modes_and_zero) {
    // count == 0 yields an empty storage in either mode.
    BOOST_CHECK(detail::make_packed_phase_storage(0, /*binary=*/true).empty());
    BOOST_CHECK(detail::make_packed_phase_storage(0, /*binary=*/false).empty());

    // Binary mode packs 64 phases per word; int8 mode is one byte per phase.
    const auto binary = detail::make_packed_phase_storage(130, /*binary=*/true);
    BOOST_CHECK(binary.uses_binary_phases);
    BOOST_CHECK_EQUAL(binary.phase_words.size(), 3U); // ceil(130/64)
    BOOST_CHECK(binary.phase_values.empty());

    const auto wide = detail::make_packed_phase_storage(130, /*binary=*/false);
    BOOST_CHECK(!wide.uses_binary_phases);
    BOOST_CHECK_EQUAL(wide.phase_values.size(), 130U);
    BOOST_CHECK(wide.phase_words.empty());
}

BOOST_AUTO_TEST_CASE(graph_encoding_packed_phase_at_reads_int8_values) {
    // A non-binary (int8) storage must read back the stored value through packed_phase_at.
    // (build_packed_cross_rank_storage below produces int8 storage when any phase is non-binary;
    //  here we exercise the reader directly so the int8 branch of packed_phase_at is covered.)
    auto storage = detail::make_packed_phase_storage(3, /*binary=*/false);
    storage.phase_values[0] = 5;
    storage.phase_values[1] = -7;
    storage.phase_values[2] = 1;
    BOOST_CHECK_EQUAL(detail::packed_phase_at(storage, 0), 5);
    BOOST_CHECK_EQUAL(detail::packed_phase_at(storage, 1), -7);
    BOOST_CHECK_EQUAL(detail::packed_phase_at(storage, 2), 1);
}

// ── build_layer_exchange_layout_impl: counts*scale, prefix-sum displacements ────────────────────

BOOST_AUTO_TEST_CASE(graph_encoding_exchange_layout_scale_and_displacements) {
    struct RangeLike {
        size_t sin_send_count;
    };
    const std::vector<RangeLike> ranges = {{3}, {0}, {5}};

    const auto s1 = detail::build_layer_exchange_layout_impl(ranges, /*scale=*/1);
    BOOST_CHECK((s1.counts == std::vector<int>{3, 0, 5}));
    BOOST_CHECK((s1.displs == std::vector<int>{0, 3, 3})); // prefix sum: 0, 0+3, 3+0
    BOOST_CHECK_EQUAL(s1.total_count, 8U);

    const auto s2 = detail::build_layer_exchange_layout_impl(ranges, /*scale=*/2);
    BOOST_CHECK((s2.counts == std::vector<int>{6, 0, 10}));
    BOOST_CHECK((s2.displs == std::vector<int>{0, 6, 6}));
    BOOST_CHECK_EQUAL(s2.total_count, 16U);

    BOOST_CHECK_GT(detail::layer_exchange_layout_storage_bytes(s1), 0U);
}

// ── D-from-B derivation: exercise BOTH arms of cross_rank_sin_recv_index ─────────────────────────

BOOST_AUTO_TEST_CASE(graph_encoding_d_from_b_derivation_both_arms) {
    // Lay out B = [in(P=2)] ++ [out(Q=3)] = [10,11 | 20,21,22]. D = [out(Q)] ++ [in(P)].
    // sin_recv_count = P + Q = 5, in_count = P = 2, so out_count Q = 3.
    //   idx < Q  -> D[idx] = B[P + idx]        (the out block)
    //   idx >= Q -> D[idx] = B[idx - Q]        (the in block)
    std::vector<CrossRankPartnerData> data(1);
    auto &p = data[0];
    for (size_t v : {10U, 11U, 20U, 21U, 22U}) {
        p.sin_send_indices.push_back(v);
    }
    for (size_t k = 0; k < 5; ++k) {
        p.sin_recv_entries.push_back({0, 1}); // phases only; D indices are derived, not stored
    }
    p.in_count = 2;

    const auto storage = detail::build_packed_cross_rank_storage(std::move(data));

    // out arm (idx < Q): B[P+idx]
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_index(storage, 0, 0), 20U);
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_index(storage, 0, 1), 21U);
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_index(storage, 0, 2), 22U);
    // in arm (idx >= Q): B[idx-Q]
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_index(storage, 0, 3), 10U);
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_index(storage, 0, 4), 11U);
    // send side reads B verbatim
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_send_index(storage, 0, 0), 10U);
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_send_index(storage, 0, 4), 22U);
}
