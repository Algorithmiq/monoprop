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
#include <stdexcept>
#include <vector>

#include "ExchangeLayoutOracle.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"

using namespace monoprop;

using exchange_layout_oracle::build_layer_exchange_layout;

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

    const auto s1 = build_layer_exchange_layout(send_counts, /*scale=*/1);
    BOOST_CHECK((s1.counts == std::vector<int>{3, 0, 5}));
    BOOST_CHECK((s1.displs == std::vector<int>{0, 3, 3})); // prefix sum: 0, 0+3, 3+0
    BOOST_CHECK_EQUAL(s1.total_count, 8U);

    const auto s2 = build_layer_exchange_layout(send_counts, /*scale=*/2);
    BOOST_CHECK((s2.counts == std::vector<int>{6, 0, 10}));
    BOOST_CHECK((s2.displs == std::vector<int>{0, 6, 6}));
    BOOST_CHECK_EQUAL(s2.total_count, 16U);
}

// The claim under test is EQUIVALENCE: what the exchange derives at the call site must equal,
// elementwise, the oracle in ExchangeLayoutOracle.h rather than hand-written literals.
namespace {

auto slot_partners(const std::vector<size_t> &sin_send_counts) -> std::vector<CrossRankPartnerData> {
    std::vector<CrossRankPartnerData> data(sin_send_counts.size());
    for (size_t r = 0; r < sin_send_counts.size(); ++r) {
        for (size_t k = 0; k < sin_send_counts[r]; ++k) {
            data[r].sin_send_indices.push_back(k);
            data[r].sin_recv_entries.push_back({k, 1});
        }
    }
    return data;
}

} // namespace

BOOST_AUTO_TEST_CASE(graph_encoding_derived_layout_matches_the_layout_it_replaces) {
    const std::vector<size_t> counts{3, 0, 5, 2};
    const auto storage = detail::build_packed_cross_rank_storage(slot_partners(counts));

    for (size_t my_rank = 0; my_rank < counts.size(); ++my_rank) {
        // The self slot is excluded from the transfer and handled locally.
        std::vector<size_t> expected_counts = counts;
        expected_counts[my_rank] = 0;

        for (const int scale : {1, 2}) {
            const auto reference = build_layer_exchange_layout(expected_counts, scale);
            LayerExchangeLayout derived;
            detail::derive_exchange_layout(storage, my_rank, scale, derived);

            BOOST_CHECK(derived.counts == reference.counts);
            BOOST_CHECK(derived.displs == reference.displs);
            BOOST_CHECK_EQUAL(derived.total_count, reference.total_count);
        }
    }
}

BOOST_AUTO_TEST_CASE(graph_encoding_derived_layout_reuses_its_scratch) {
    // Reused across layers: a stale tail would be read by MPI as a real count for an unused slot.
    const auto wide = detail::build_packed_cross_rank_storage(slot_partners({1, 2, 3, 4}));
    const auto narrow = detail::build_packed_cross_rank_storage(slot_partners({7, 7}));

    LayerExchangeLayout scratch;
    detail::derive_exchange_layout(wide, /*my_rank=*/0, 1, scratch);
    BOOST_CHECK_EQUAL(scratch.counts.size(), 4U);
    detail::derive_exchange_layout(narrow, /*my_rank=*/0, 1, scratch);
    BOOST_CHECK_EQUAL(scratch.counts.size(), 2U);
    BOOST_CHECK((scratch.counts == std::vector<int>{0, 7}));
    BOOST_CHECK_EQUAL(scratch.total_count, 7U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_a_zero_traffic_slot_still_gets_a_valid_displacement) {
    // Empty slots hide prefix-sum off-by-ones: count 0, but the displacement must still advance.
    const auto storage = detail::build_packed_cross_rank_storage(slot_partners({0, 4, 0, 0, 6}));
    LayerExchangeLayout derived;
    detail::derive_exchange_layout(storage, /*my_rank=*/3, 1, derived);

    BOOST_CHECK((derived.counts == std::vector<int>{0, 4, 0, 0, 6}));
    BOOST_CHECK((derived.displs == std::vector<int>{0, 0, 4, 4, 4}));
    BOOST_CHECK_EQUAL(derived.total_count, 10U);
    for (size_t r = 1; r < derived.displs.size(); ++r) {
        BOOST_CHECK_GE(derived.displs[r], derived.displs[r - 1]);
    }
}

BOOST_AUTO_TEST_CASE(graph_encoding_derivative_exchange_layout_overflow_throws) {
    // A count that fits int at 1x but not at 2x. build_layer_storage_unified derives the 2x layout
    // eagerly, so the throw lands in build_graph rather than inside the collective window, where the
    // peers of a rank that threw are already blocked against a size nobody will send.
    // Declared, not materialised: 2^30 real endpoints cost a 16 GB runner its whole VM.
    const size_t just_over_half = static_cast<size_t>(std::numeric_limits<int>::max()) / 2 + 1;
    PackedCrossRankStorage storage;
    storage.world_size = 2;
    storage.occupied.push_back(CrossRankOccupiedSlot{.slot = 0, .sin_send_count = just_over_half, .in_count = 0});

    LayerExchangeLayout derived;
    BOOST_CHECK_NO_THROW(detail::derive_exchange_layout(storage, /*my_rank=*/1, 1, derived));
    BOOST_CHECK_THROW(detail::derive_exchange_layout(storage, /*my_rank=*/1, 2, derived), std::overflow_error);
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

// The accounting split behind graph_memory_breakdown(): the slot-record cost is set by the size of
// the world and does not move when the traffic through it does.

BOOST_AUTO_TEST_CASE(graph_encoding_occupied_slots_counts_only_slots_carrying_traffic) {
    // Zeros at the front, in the interior and at the back -- the three places a scan loses count.
    const auto storage = detail::build_packed_cross_rank_storage(slot_partners({0, 3, 0, 0, 7, 0}));

    BOOST_CHECK_EQUAL(storage.rank_count(), 6U);
    BOOST_CHECK_EQUAL(detail::cross_rank_occupied_slots(storage), 2U);
    BOOST_CHECK_EQUAL(detail::cross_rank_endpoint_count(storage), 10U);
    // The ceiling this instrument exists to expose: an occupied slot holds at least one endpoint.
    BOOST_CHECK_LE(detail::cross_rank_occupied_slots(storage), detail::cross_rank_endpoint_count(storage));
}

BOOST_AUTO_TEST_CASE(graph_encoding_slot_record_bytes_track_the_traffic_not_the_world) {
    // Same traffic, four times the world: the record array is a function of what is sent.
    const auto narrow = detail::build_packed_cross_rank_storage(slot_partners({5, 0, 0, 0}));
    const auto wide =
        detail::build_packed_cross_rank_storage(slot_partners({5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    BOOST_CHECK_EQUAL(narrow.rank_count(), 4U);
    BOOST_CHECK_EQUAL(wide.rank_count(), 16U);
    BOOST_CHECK_EQUAL(detail::cross_rank_endpoint_count(narrow), detail::cross_rank_endpoint_count(wide));
    BOOST_CHECK_EQUAL(detail::cross_rank_occupied_slots(narrow), detail::cross_rank_occupied_slots(wide));
    // The world quadrupled and the record array did not move at all.
    BOOST_CHECK_EQUAL(detail::cross_rank_slot_record_bytes(narrow), detail::cross_rank_slot_record_bytes(wide));
    BOOST_CHECK_EQUAL(detail::cross_rank_slot_record_bytes(wide), 1U * sizeof(CrossRankOccupiedSlot));
    // And the slot records are a slice of cross_rank_bytes, not an addition to it.
    BOOST_CHECK_LT(detail::cross_rank_slot_record_bytes(wide), detail::cross_rank_storage_bytes(wide));
}

BOOST_AUTO_TEST_CASE(graph_encoding_a_layer_retains_no_exchange_layout) {
    // The point of the change: a built layer holds the slot records and nothing else sized by P.
    // Neither side of the exchange is retained -- not the send layout, and not a transpose of it.
    const auto core = detail::build_layer_storage_unified(slot_partners({3, 0, 5}), /*my_rank=*/1);

    // Not stored, but recoverable from the slot records alone: 3 + 5, my_rank's own slot excluded.
    LayerExchangeLayout derived;
    detail::derive_exchange_layout(core->cross_rank, /*my_rank=*/1, /*scale=*/1, derived);
    BOOST_CHECK_EQUAL(derived.total_count, 8U);

    // A LayerCore is held L x P times across a job, so pin its size against the members it should
    // have: three vectors plus the scalars. A new P-sized member costs once per layer per partition.
    BOOST_CHECK_LE(sizeof(LayerCore), 256U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_skewed_endpoint_counts_are_refused) {
    // The packed record keeps one count and one offset for both B and D. Nothing in the TYPE prevents
    // a skew, and unchecked it would not throw: Q would be mis-derived and a wrong-but-valid endpoint
    // read. Refusing at the choke point makes the assumption a precondition.
    std::vector<CrossRankPartnerData> data(1);
    data[0].sin_send_indices.push_back(1);
    data[0].sin_send_indices.push_back(2);
    data[0].sin_recv_entries.push_back({1, 1}); // one D against two B
    data[0].in_count = 1;

    BOOST_CHECK_THROW(detail::build_packed_cross_rank_storage(data), std::logic_error);
}

BOOST_AUTO_TEST_CASE(graph_encoding_an_in_block_past_the_endpoint_list_is_refused) {
    // in_count is a boundary INSIDE B and the out-block size is an unsigned subtraction, so one past
    // the end wraps instead of going negative. Refused where the record is built, so the reader can
    // subtract without branching.
    std::vector<CrossRankPartnerData> data(1);
    data[0].sin_send_indices.push_back(1);
    data[0].sin_send_indices.push_back(2);
    data[0].sin_recv_entries.push_back({1, 1});
    data[0].sin_recv_entries.push_back({2, 1});
    data[0].in_count = 3; // three inside two

    BOOST_CHECK_THROW(detail::build_packed_cross_rank_storage(data), std::logic_error);

    // The boundary itself is legal: in_count == B.size() is an all-in slot with an empty out-block.
    data[0].in_count = 2;
    BOOST_CHECK_NO_THROW(detail::build_packed_cross_rank_storage(data));
}

BOOST_AUTO_TEST_CASE(graph_encoding_occupied_sweep_matches_the_dense_sweep_it_replaces) {
    // Zeros at the front, interior and back, and a slot whose in_count splits the B list, so the
    // derived offset has somewhere to go wrong.
    auto data = slot_partners({0, 3, 0, 0, 7, 0});
    data[1].in_count = 1;
    data[4].in_count = 4;
    const auto storage = detail::build_packed_cross_rank_storage(data);

    // The offsets are the prefix over ALL slots, to which the empty ones contributed zero.
    struct Expected {
        size_t slot, offset, count, in_count;
    };
    const std::vector<Expected> expected{{1, 0, 3, 1}, {4, 3, 7, 4}};

    std::vector<Expected> seen;
    detail::for_each_occupied_slot(storage, [&](size_t slot, const detail::CrossRankSlotView &view) {
        seen.push_back({slot, view.phase_offset, view.sin_send_count, view.in_count});
    });

    BOOST_REQUIRE_EQUAL(seen.size(), expected.size());
    for (size_t k = 0; k < expected.size(); ++k) {
        BOOST_CHECK_EQUAL(seen[k].slot, expected[k].slot);
        BOOST_CHECK_EQUAL(seen[k].offset, expected[k].offset);
        BOOST_CHECK_EQUAL(seen[k].count, expected[k].count);
        BOOST_CHECK_EQUAL(seen[k].in_count, expected[k].in_count);
    }

    // The general single-slot resolver must agree with the sweep, including on an absent slot.
    for (const auto &e : expected) {
        const auto view = detail::cross_rank_slot(storage, e.slot);
        BOOST_CHECK_EQUAL(view.phase_offset, e.offset);
        BOOST_CHECK_EQUAL(view.sin_send_count, e.count);
    }
    BOOST_CHECK_EQUAL(detail::cross_rank_slot(storage, 0).sin_send_count, 0U);
    BOOST_CHECK_EQUAL(detail::cross_rank_slot(storage, 5).sin_send_count, 0U);
    BOOST_CHECK_EQUAL(storage.sin_send_size(3), 0U);
    BOOST_CHECK_EQUAL(storage.sin_send_size(4), 7U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_self_slot_is_resolved_without_a_search) {
    auto storage = detail::build_packed_cross_rank_storage(slot_partners({0, 3, 0, 7, 0}));

    detail::resolve_self_slot(storage, 3);
    BOOST_CHECK_EQUAL(storage.self_offset, 3U); // slot 1's three endpoints precede it
    const auto self = detail::cross_rank_self_slot(storage);
    BOOST_CHECK_EQUAL(self.sin_send_count, 7U);
    BOOST_CHECK_EQUAL(self.phase_offset, 3U);

    // A rank whose own slot carries nothing must resolve to an empty view, not to a neighbour's.
    detail::resolve_self_slot(storage, 2);
    BOOST_CHECK_EQUAL(storage.self_pos, kNoSelfSlot);
    BOOST_CHECK_EQUAL(detail::cross_rank_self_slot(storage).sin_send_count, 0U);
}

BOOST_AUTO_TEST_CASE(graph_encoding_a_layer_with_no_cross_rank_traffic_stores_no_slots) {
    const auto storage = detail::build_packed_cross_rank_storage(slot_partners({0, 0, 0, 0, 0, 0, 0, 0}));

    BOOST_CHECK_EQUAL(storage.rank_count(), 8U); // the world is still eight wide
    BOOST_CHECK_EQUAL(detail::cross_rank_occupied_slots(storage), 0U);
    BOOST_CHECK_EQUAL(detail::cross_rank_slot_record_bytes(storage), 0U);
}
