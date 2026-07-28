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

// White-box tests for the pure packing/layout functions in
// src/monoprop/detail/graph_encoding/*, checked against hand-computed oracles.

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"

using namespace monoprop;

BOOST_AUTO_TEST_CASE(graph_encoding_word_builder_push_index_coalesces_within_word) {
    CosineWordBuilder b;
    b.push_index(0);
    b.push_index(1);
    b.push_index(3);
    b.push_index(64); // crosses into the next word -> flushes word 0
    b.push_index(197);
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
    b.push_word(0, 0b101ULL);
    b.push_word(64, 0ULL); // zero word: no-op, no block emitted
    b.push_word(128, 0xFULL);
    const CosMask cos = b.finish();

    BOOST_REQUIRE_EQUAL(cos.blocks.size(), 2U);
    BOOST_CHECK_EQUAL(cos.blocks[0].first, 0U);
    BOOST_CHECK_EQUAL(cos.blocks[1].first, 128U);
    BOOST_CHECK_EQUAL(cos.total_count, 2U + 4U);
    BOOST_CHECK_EQUAL(cos.span_count(), 2U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_word_builder_finish_flushes_pending_and_empty_is_empty) {
    CosineWordBuilder pending;
    pending.push_index(5);
    const CosMask cos = pending.finish();
    BOOST_REQUIRE_EQUAL(cos.blocks.size(), 1U);
    BOOST_CHECK_EQUAL(cos.blocks[0].second, uint64_t{1} << 5);

    CosineWordBuilder empty;
    const CosMask none = empty.finish();
    BOOST_CHECK(none.blocks.empty());
    BOOST_CHECK_EQUAL(none.total_count, 0U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_checked_term_index_boundary) {
    // At the TermIndex ceiling it round-trips; above it throws only in the narrow build.
    const size_t ceiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    BOOST_CHECK_EQUAL(detail::checked_term_index(ceiling, "term"), std::numeric_limits<TermIndex>::max());
#if !defined(monoprop_WIDE_TERM_INDEX)
    BOOST_CHECK_THROW(detail::checked_term_index(ceiling + 1, "term"), std::overflow_error);
#else
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

BOOST_AUTO_TEST_CASE(graph_encoding_make_packed_phase_storage_modes_and_zero) {
    BOOST_CHECK(detail::make_packed_phase_storage(0, /*binary=*/true).empty());
    BOOST_CHECK(detail::make_packed_phase_storage(0, /*binary=*/false).empty());

    // Binary mode packs 64 phases per word; int8 mode is one byte per phase.
    const auto binary = detail::make_packed_phase_storage(130, /*binary=*/true);
    BOOST_CHECK(binary.uses_binary_phases);
    BOOST_CHECK_EQUAL(binary.phase_words.size(), 3U);
    BOOST_CHECK(binary.phase_values.empty());

    const auto wide = detail::make_packed_phase_storage(130, /*binary=*/false);
    BOOST_CHECK(!wide.uses_binary_phases);
    BOOST_CHECK_EQUAL(wide.phase_values.size(), 130U);
    BOOST_CHECK(wide.phase_words.empty());
}

BOOST_AUTO_TEST_CASE(graph_encoding_packed_phase_at_reads_int8_values) {
    auto storage = detail::make_packed_phase_storage(3, /*binary=*/false);
    storage.phase_values[0] = 5;
    storage.phase_values[1] = -7;
    storage.phase_values[2] = 1;
    BOOST_CHECK_EQUAL(detail::packed_phase_at(storage, 0), 5);
    BOOST_CHECK_EQUAL(detail::packed_phase_at(storage, 1), -7);
    BOOST_CHECK_EQUAL(detail::packed_phase_at(storage, 2), 1);
}

BOOST_AUTO_TEST_CASE(graph_encoding_exchange_layout_scale_and_displacements) {
    const std::vector<size_t> send_counts = {3, 0, 5};

    const auto s1 = detail::build_layer_exchange_layout(send_counts, /*scale=*/1);
    BOOST_CHECK((s1.counts == std::vector<int>{3, 0, 5}));
    BOOST_CHECK((s1.displs == std::vector<int>{0, 3, 3})); // prefix sum: 0, 0+3, 3+0
    BOOST_CHECK_EQUAL(s1.total_count, 8U);

    const auto s2 = detail::build_layer_exchange_layout(send_counts, /*scale=*/2);
    BOOST_CHECK((s2.counts == std::vector<int>{6, 0, 10}));
    BOOST_CHECK((s2.displs == std::vector<int>{0, 6, 6}));
    BOOST_CHECK_EQUAL(s2.total_count, 16U);

    BOOST_CHECK_GT(detail::layer_exchange_layout_storage_bytes(s1), 0U);
}

// Production only builds scale=1; the 2x layout reaches MPI through this accessor, which is
// unreachable at comm size 1, so the default non-MPI suite would otherwise never touch it.

BOOST_AUTO_TEST_CASE(graph_encoding_derivative_exchange_layout_is_twice_the_evolution_layout) {
    LayerCore core;
    core.evolution_exchange_layout = detail::build_layer_exchange_layout({3, 0, 5}, /*scale=*/1);

    const auto &derivative = core.derivative_exchange_layout();
    BOOST_CHECK((derivative.counts == std::vector<int>{6, 0, 10}));
    BOOST_CHECK((derivative.displs == std::vector<int>{0, 6, 6}));
    BOOST_CHECK_EQUAL(derivative.total_count, 16U);

    // Cached: the second read returns the same object, so eval-time MPI holds a stable pointer.
    BOOST_CHECK_EQUAL(&core.derivative_exchange_layout(), &derivative);

    // Reset drops the cache (relabel copies cores and must not inherit eval-time state).
    core.reset_derivative_exchange_layout();
    BOOST_CHECK_EQUAL(core.derivative_exchange_layout().total_count, 16U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_derivative_exchange_layout_overflow_throws) {
    // A count that fits int at 1x but not at 2x. build_layer_storage_unified runs this derivation
    // eagerly, so the throw lands in build_graph and not inside the gradient collective window.
    const size_t just_over_half = static_cast<size_t>(std::numeric_limits<int>::max()) / 2 + 1;

    LayerCore core;
    core.evolution_exchange_layout = detail::build_layer_exchange_layout({just_over_half}, 1);
    BOOST_CHECK_THROW(detail::build_derivative_exchange_layout(core.evolution_exchange_layout), std::overflow_error);
}

BOOST_AUTO_TEST_CASE(graph_encoding_d_from_b_derivation_both_arms) {
    // B = [in(P=2)] ++ [out(Q=3)] = [10,11 | 20,21,22]; D = [out] ++ [in], derived from B and in_count.
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
