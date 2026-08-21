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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"

using namespace monoprop;

TEST_CASE("graph_encoding_word_builder_push_index_coalesces_within_word") {
    CosineWordBuilder b;
    b.push_index(0);
    b.push_index(1);
    b.push_index(3);
    b.push_index(64); // crosses into the next word -> flushes word 0
    b.push_index(197);
    const CosMask cos = b.finish();

    REQUIRE((cos.blocks.size()) == (3U));
    CHECK((cos.blocks[0].first) == (0U));
    CHECK((cos.blocks[0].second) == (0b1011ULL));
    CHECK((cos.blocks[1].first) == (64U));
    CHECK((cos.blocks[1].second) == (0b1ULL));
    CHECK((cos.blocks[2].first) == (192U));
    CHECK((cos.blocks[2].second) == (uint64_t{1} << 5));
    CHECK((cos.total_count) == (5U));
}

TEST_CASE("graph_encoding_word_builder_push_word_skips_zero_and_counts_bits") {
    CosineWordBuilder b;
    b.push_word(0, 0b101ULL);
    b.push_word(64, 0ULL); // zero word: no-op, no block emitted
    b.push_word(128, 0xFULL);
    const CosMask cos = b.finish();

    REQUIRE((cos.blocks.size()) == (2U));
    CHECK((cos.blocks[0].first) == (0U));
    CHECK((cos.blocks[1].first) == (128U));
    CHECK((cos.total_count) == (2U + 4U));
    CHECK((cos.span_count()) == (2U));
}

TEST_CASE("graph_encoding_word_builder_finish_flushes_pending_and_empty_is_empty") {
    CosineWordBuilder pending;
    pending.push_index(5);
    const CosMask cos = pending.finish();
    REQUIRE((cos.blocks.size()) == (1U));
    CHECK((cos.blocks[0].second) == (uint64_t{1} << 5));

    CosineWordBuilder empty;
    const CosMask none = empty.finish();
    CHECK(none.blocks.empty());
    CHECK((none.total_count) == (0U));
}

TEST_CASE("graph_encoding_checked_term_index_boundary") {
    // At the TermIndex ceiling it round-trips; above it throws only in the narrow build.
    const size_t ceiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    CHECK((detail::checked_term_index(ceiling, "term")) == (std::numeric_limits<TermIndex>::max()));
#if !defined(monoprop_WIDE_TERM_INDEX)
    CHECK_THROWS_AS(detail::checked_term_index(ceiling + 1, "term"), std::overflow_error);
#else
    const size_t above_u32 = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
    CHECK((detail::checked_term_index(above_u32, "term")) == (static_cast<TermIndex>(above_u32)));
#endif
}

TEST_CASE("graph_encoding_checked_packed_phase_bounds") {
    CHECK((detail::checked_packed_phase(127, "phase")) == (127));
    CHECK((detail::checked_packed_phase(-128, "phase")) == (-128));
    CHECK_THROWS_AS(detail::checked_packed_phase(128, "phase"), std::overflow_error);
    CHECK_THROWS_AS(detail::checked_packed_phase(-129, "phase"), std::overflow_error);
}

TEST_CASE("graph_encoding_make_packed_phase_storage_modes_and_zero") {
    CHECK(detail::make_packed_phase_storage(0, /*binary=*/true).empty());
    CHECK(detail::make_packed_phase_storage(0, /*binary=*/false).empty());

    // Binary mode packs 64 phases per word; int8 mode is one byte per phase.
    const auto binary = detail::make_packed_phase_storage(130, /*binary=*/true);
    CHECK(binary.uses_binary_phases);
    CHECK((binary.phase_words.size()) == (3U));
    CHECK(binary.phase_values.empty());

    const auto wide = detail::make_packed_phase_storage(130, /*binary=*/false);
    CHECK(!wide.uses_binary_phases);
    CHECK((wide.phase_values.size()) == (130U));
    CHECK(wide.phase_words.empty());
}

TEST_CASE("graph_encoding_packed_phase_at_reads_int8_values") {
    auto storage = detail::make_packed_phase_storage(3, /*binary=*/false);
    storage.phase_values[0] = 5;
    storage.phase_values[1] = -7;
    storage.phase_values[2] = 1;
    CHECK((detail::packed_phase_at(storage, 0)) == (5));
    CHECK((detail::packed_phase_at(storage, 1)) == (-7));
    CHECK((detail::packed_phase_at(storage, 2)) == (1));
}

TEST_CASE("graph_encoding_exchange_layout_scale_and_displacements") {
    const std::vector<size_t> send_counts = {3, 0, 5};

    const auto s1 = detail::build_layer_exchange_layout(send_counts, /*scale=*/1);
    CHECK((s1.counts == std::vector<int>{3, 0, 5}));
    CHECK((s1.displs == std::vector<int>{0, 3, 3})); // prefix sum: 0, 0+3, 3+0
    CHECK((s1.total_count) == (8U));

    const auto s2 = detail::build_layer_exchange_layout(send_counts, /*scale=*/2);
    CHECK((s2.counts == std::vector<int>{6, 0, 10}));
    CHECK((s2.displs == std::vector<int>{0, 6, 6}));
    CHECK((s2.total_count) == (16U));

    CHECK((detail::layer_exchange_layout_storage_bytes(s1)) > (0U));
}

// Production only builds scale=1; the 2x layout reaches MPI through this accessor, which is
// unreachable at comm size 1, so the default non-MPI suite would otherwise never touch it.

TEST_CASE("graph_encoding_derivative_exchange_layout_is_twice_the_evolution_layout") {
    LayerCore core;
    core.evolution_exchange_layout = detail::build_layer_exchange_layout({3, 0, 5}, /*scale=*/1);

    const auto &derivative = core.derivative_exchange_layout();
    CHECK((derivative.counts == std::vector<int>{6, 0, 10}));
    CHECK((derivative.displs == std::vector<int>{0, 6, 6}));
    CHECK((derivative.total_count) == (16U));

    // Cached: the second read returns the same object, so eval-time MPI holds a stable pointer.
    CHECK((&core.derivative_exchange_layout()) == (&derivative));

    // Reset drops the cache (relabel copies cores and must not inherit eval-time state).
    core.reset_derivative_exchange_layout();
    CHECK((core.derivative_exchange_layout().total_count) == (16U));
}

TEST_CASE("graph_encoding_derivative_exchange_layout_overflow_throws") {
    // A count that fits int at 1x but not at 2x. build_layer_storage_unified runs this derivation
    // eagerly, so the throw lands in build_graph and not inside the gradient collective window.
    const size_t just_over_half = static_cast<size_t>(std::numeric_limits<int>::max()) / 2 + 1;

    LayerCore core;
    core.evolution_exchange_layout = detail::build_layer_exchange_layout({just_over_half}, 1);
    CHECK_THROWS_AS(detail::build_derivative_exchange_layout(core.evolution_exchange_layout), std::overflow_error);
}

TEST_CASE("graph_encoding_d_from_b_derivation_both_arms") {
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
    CHECK((detail::cross_rank_sin_recv_index(storage, 0, 0)) == (20U));
    CHECK((detail::cross_rank_sin_recv_index(storage, 0, 1)) == (21U));
    CHECK((detail::cross_rank_sin_recv_index(storage, 0, 2)) == (22U));
    // in arm (idx >= Q): B[idx-Q]
    CHECK((detail::cross_rank_sin_recv_index(storage, 0, 3)) == (10U));
    CHECK((detail::cross_rank_sin_recv_index(storage, 0, 4)) == (11U));
    // send side reads B verbatim
    CHECK((detail::cross_rank_sin_send_index(storage, 0, 0)) == (10U));
    CHECK((detail::cross_rank_sin_send_index(storage, 0, 4)) == (22U));
}
