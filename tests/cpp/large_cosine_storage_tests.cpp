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

#include <boost/test/unit_test.hpp>

#include <limits>
#include <memory>

#include "monoprop/Evolution.h"
#include "monoprop/MPGraph.h"

using namespace monoprop;

BOOST_AUTO_TEST_CASE(compressed_cosine_data_supports_indices_above_u32) {
    const size_t base = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 5;
    const size_t expected_chunk_base = detail::cosine_chunk_base(base);
    const uint16_t expected_offset0 = detail::cosine_chunk_offset(base);
    const uint16_t expected_offset1 = detail::cosine_chunk_offset(base + 10);
    const VecZ indices = {base, base + 1, base + 2, base + 10};

    const auto data = detail::build_compressed_cosine_data(indices);
    BOOST_CHECK_EQUAL(data.total_count, indices.size());
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 2UL);
    BOOST_CHECK(data.has_wide_starts());
    BOOST_CHECK_EQUAL(data.chunk_bases[0], expected_chunk_base);
    BOOST_CHECK_EQUAL(data.chunk_span_starts[0], 0UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], expected_offset0);
    BOOST_CHECK_EQUAL(data.span_counts[0], 3U);
    BOOST_CHECK_EQUAL(detail::cosine_subspan_start(data, 0, 0), base);
    BOOST_CHECK_EQUAL(data.span_offsets[1], expected_offset1);
    BOOST_CHECK_EQUAL(data.span_counts[1], 1U);
    BOOST_CHECK_EQUAL(detail::cosine_subspan_start(data, 0, 1), base + 10);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), indices.begin(), indices.end());
}

BOOST_AUTO_TEST_CASE(compressed_cosine_data_supports_very_large_indices_with_chunk_bases) {
    if (std::numeric_limits<size_t>::digits <= 48) {
        BOOST_TEST_MESSAGE("Skipping large-index chunk-base test on platforms with <= 48 size_t bits.");
        return;
    }

    const size_t base = (size_t{1} << 48) + 33;
    const size_t expected_chunk_base = detail::cosine_chunk_base(base);
    const VecZ indices = {base, base + 1, base + 5};

    const auto data = detail::build_compressed_cosine_data(indices);
    BOOST_CHECK(data.has_wide_starts());
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 2UL);
    BOOST_CHECK_EQUAL(data.chunk_bases[0], expected_chunk_base);
    BOOST_CHECK_EQUAL(data.chunk_span_starts[0], 0UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], detail::cosine_chunk_offset(base));
    BOOST_CHECK_EQUAL(data.span_counts[0], 2U);
    BOOST_CHECK_EQUAL(detail::cosine_subspan_start(data, 0, 0), base);
    BOOST_CHECK_EQUAL(data.span_offsets[1], detail::cosine_chunk_offset(base + 5));
    BOOST_CHECK_EQUAL(data.span_counts[1], 1U);
    BOOST_CHECK_EQUAL(detail::cosine_subspan_start(data, 0, 1), base + 5);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), indices.begin(), indices.end());
}

BOOST_AUTO_TEST_CASE(compressed_cosine_data_reduces_dense_wide_span_storage_bytes) {
    const size_t base = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 9;
    VecZ indices;
    indices.reserve(1024);
    for (size_t offset = 0; offset < 2048; offset += 2) {
        indices.push_back(base + offset);
    }

    auto data = detail::build_compressed_cosine_data(indices);
    detail::shrink_compressed_cosine_data(data);

    BOOST_CHECK(data.has_wide_starts());
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);

    const size_t compact_bytes = detail::compressed_cosine_data_storage_bytes(data);
    const size_t legacy_wide_bytes = data.span_count() * (sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t));

    BOOST_CHECK_LT(compact_bytes, legacy_wide_bytes);
}

BOOST_AUTO_TEST_CASE(compressed_cosine_data_splits_runs_at_chunk_boundaries) {
    const size_t base = detail::kCosineChunkSize - 2;
    const VecZ indices = {base, base + 1, base + 2, base + 3};

    const auto data = detail::build_compressed_cosine_data(indices);

    BOOST_CHECK_EQUAL(data.chunk_count(), 2UL);
    BOOST_CHECK_EQUAL(data.span_count(), 2UL);
    BOOST_CHECK_EQUAL(data.chunk_bases[0], 0UL);
    BOOST_CHECK_EQUAL(data.chunk_span_starts[0], 0UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], detail::kCosineChunkSize - 2);
    BOOST_CHECK_EQUAL(data.span_counts[0], 2U);
    BOOST_CHECK_EQUAL(data.chunk_bases[1], detail::kCosineChunkSize);
    BOOST_CHECK_EQUAL(data.chunk_span_starts[1], 1UL);
    BOOST_CHECK_EQUAL(data.span_offsets[1], 0U);
    BOOST_CHECK_EQUAL(data.span_counts[1], 2U);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), indices.begin(), indices.end());
}

BOOST_AUTO_TEST_CASE(compressed_cosine_data_splits_long_runs) {
    const size_t base = 17;
    const size_t run_length = detail::kMaxCosineSpanLength + 5;

    VecZ indices(run_length);
    for (size_t offset = 0; offset < run_length; ++offset) {
        indices[offset] = base + offset;
    }

    const auto data = detail::build_compressed_cosine_data(indices);
    BOOST_CHECK_EQUAL(data.total_count, run_length);
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 2UL);
    BOOST_CHECK_EQUAL(data.chunk_bases[0], 0UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], base);
    BOOST_CHECK_EQUAL(data.span_counts[0], detail::kMaxCosineSpanLength);
    BOOST_CHECK_EQUAL(data.span_offsets[1], base + detail::kMaxCosineSpanLength);
    BOOST_CHECK_EQUAL(data.span_counts[1], 5U);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), indices.begin(), indices.end());
}

BOOST_AUTO_TEST_CASE(compressed_cosine_data_preserves_unsorted_input_order) {
    const VecZ indices = {42, 7, 8, 9, 1000, 1001};

    const auto data = detail::build_compressed_cosine_data(indices);
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 3UL);
    BOOST_CHECK_EQUAL(data.chunk_bases[0], 0UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], 42U);
    BOOST_CHECK_EQUAL(data.span_counts[0], 1U);
    BOOST_CHECK_EQUAL(data.span_offsets[1], 7U);
    BOOST_CHECK_EQUAL(data.span_counts[1], 3U);
    BOOST_CHECK_EQUAL(data.span_offsets[2], 1000U);
    BOOST_CHECK_EQUAL(data.span_counts[2], 2U);
    const auto expanded = detail::expand_compressed_cosine_data(data);

    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), indices.begin(), indices.end());
}

BOOST_AUTO_TEST_CASE(compressed_cosine_data_blocks_preserve_order_and_merge_runs) {
    const std::vector<VecZ> blocks = {
        {7, 8},
        {9, 10},
        {42},
        {43, 100},
    };
    const VecZ expected = {7, 8, 9, 10, 42, 43, 100};

    CompressedCosineData data;
    detail::reserve_compressed_cosine_data(data, expected.size());
    for (const auto &block : blocks) {
        detail::append_compressed_cosine_data(data, detail::build_compressed_cosine_data(block));
    }

    BOOST_CHECK_EQUAL(data.total_count, expected.size());
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 3UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], 7U);
    BOOST_CHECK_EQUAL(data.span_counts[0], 4U);
    BOOST_CHECK_EQUAL(data.span_offsets[1], 42U);
    BOOST_CHECK_EQUAL(data.span_counts[1], 2U);
    BOOST_CHECK_EQUAL(data.span_offsets[2], 100U);
    BOOST_CHECK_EQUAL(data.span_counts[2], 1U);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(appending_cosine_indices_in_segments_merges_contiguous_runs) {
    CompressedCosineData data;
    detail::PendingIndexRun pending_run;
    const VecZ first = {7, 8};
    const VecZ second = {9, 10, 42, 43};
    const VecZ expected = {7, 8, 9, 10, 42, 43};

    detail::append_cosine_indices(data, std::span<const size_t>{first.data(), first.size()}, pending_run);
    detail::append_cosine_indices(data, std::span<const size_t>{second.data(), second.size()}, pending_run);
    detail::finish_pending_cosine_run(data, pending_run);
    data.total_count = expected.size();

    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 2UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], 7U);
    BOOST_CHECK_EQUAL(data.span_counts[0], 4U);
    BOOST_CHECK_EQUAL(data.span_offsets[1], 42U);
    BOOST_CHECK_EQUAL(data.span_counts[1], 2U);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(filtered_compressed_cosine_data_supports_indices_above_u32) {
    const size_t base = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 11;
    const VecZ indices = {base, base + 1, base + 2, base + 6};
    const VecZ excluded = {base + 1, base + 6};
    const VecZ expected = {base, base + 2};

    const auto data = detail::build_filtered_compressed_cosine_data(indices, excluded);
    BOOST_CHECK_EQUAL(data.total_count, expected.size());
    BOOST_CHECK(data.has_wide_starts());
    BOOST_CHECK_EQUAL(data.chunk_count(), 1UL);
    BOOST_CHECK_EQUAL(data.span_count(), 2UL);
    BOOST_CHECK_EQUAL(data.span_offsets[0], detail::cosine_chunk_offset(base));
    BOOST_CHECK_EQUAL(data.span_counts[0], 1U);
    BOOST_CHECK_EQUAL(data.span_offsets[1], detail::cosine_chunk_offset(base + 2));
    BOOST_CHECK_EQUAL(data.span_counts[1], 1U);

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(filtered_compressed_cosine_data_preserves_input_order) {
    const VecZ indices = {42, 7, 8, 9, 1000, 1001};
    const VecZ excluded = {8, 1000};
    const VecZ expected = {42, 7, 9, 1001};

    const auto data = detail::build_filtered_compressed_cosine_data(indices, excluded);
    BOOST_CHECK_EQUAL(data.total_count, expected.size());

    const auto expanded = detail::expand_compressed_cosine_data(data);
    BOOST_CHECK_EQUAL_COLLECTIONS(expanded.begin(), expanded.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(layer_execution_plan_supports_counts_above_u32) {
    const size_t large_count = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 9;
    const size_t large_index = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 17;

    auto layer_cos_data = detail::build_compressed_cosine_data(VecZ{large_index});
    layer_cos_data.total_count = large_count;

    std::vector<CrossRankCycles> cross_rank(2);
    cross_rank[1].out_indices.resize(8);
    cross_rank[1].out_phases.resize(8);
    cross_rank[1].in_indices.resize(12);
    cross_rank[1].in_phases.resize(12);
    for (size_t idx = 0; idx < 8; ++idx) {
        cross_rank[1].out_indices[idx] = 100 + idx;
        cross_rank[1].out_phases[idx] = idx % 2 == 0 ? 1 : -1;
    }
    for (size_t idx = 0; idx < 12; ++idx) {
        cross_rank[1].in_indices[idx] = 200 + idx;
        cross_rank[1].in_phases[idx] = idx % 2 == 0 ? -1 : 1;
    }

    auto storage = detail::build_layer_storage(std::move(layer_cos_data), {}, std::move(cross_rank));
    Layer layer{storage};
    BOOST_CHECK_EQUAL(layer.num_cos_inds(), large_count);

    auto execution_storage = std::make_shared<detail::ExecutionPlanStorage>();
    auto plan_cos_data = detail::build_compressed_cosine_data(VecZ{large_index});
    plan_cos_data.total_count = large_count;
    execution_storage->cos_data_blocks.push_back(std::move(plan_cos_data));
    execution_storage->cross_rank_out_position_blocks.push_back(detail::build_compressed_position_data({7}));
    execution_storage->cross_rank_in_position_blocks.push_back(detail::build_compressed_position_data({11}));

    std::vector<detail::CrossRankMaskRange> ranges(2);
    ranges[1] = detail::CrossRankMaskRange{0, 1, 0, 1};

    LayerExecutionPlan plan{
        storage,
        execution_storage,
        false,
        0,
        true,
        0,
        false,
        0,
        std::move(ranges),
    };

    const auto traversal = plan.traversal();
    BOOST_CHECK_EQUAL(traversal.num_cos_inds(), large_count);
    BOOST_CHECK_EQUAL(traversal.cross_rank_out_index(1, 0), 107UL);
    BOOST_CHECK_EQUAL(traversal.cross_rank_out_phase(1, 0), -1);
    BOOST_CHECK_EQUAL(traversal.cross_rank_in_index(1, 0), 211UL);
    BOOST_CHECK_EQUAL(traversal.cross_rank_in_phase(1, 0), 1);
}

BOOST_AUTO_TEST_CASE(compressed_position_data_merges_contiguous_runs) {
    const auto packed = detail::build_compressed_position_data({1, 2, 3, 17, 18, 65535});
    BOOST_CHECK_EQUAL(packed.total_count, 6UL);
    BOOST_CHECK_EQUAL(packed.span_count(), 3UL);
    BOOST_CHECK_EQUAL(detail::compressed_position_at(packed, 0), 1U);
    BOOST_CHECK_EQUAL(detail::compressed_position_at(packed, 2), 3U);
    BOOST_CHECK_EQUAL(detail::compressed_position_at(packed, 3), 17U);
    BOOST_CHECK_EQUAL(detail::compressed_position_at(packed, 5), 65535U);
}

BOOST_AUTO_TEST_CASE(compressed_position_data_reduces_dense_storage_bytes) {
    std::vector<uint32_t> positions(1024);
    for (uint32_t idx = 0; idx < positions.size(); ++idx) {
        positions[idx] = idx;
    }

    auto packed = detail::build_compressed_position_data(std::move(positions));
    detail::shrink_compressed_position_data(packed);

    BOOST_CHECK_EQUAL(packed.total_count, 1024UL);
    BOOST_CHECK_EQUAL(packed.span_count(), 1UL);
    BOOST_CHECK_LT(detail::compressed_position_data_storage_bytes(packed), 1024UL * sizeof(uint32_t));
}

BOOST_AUTO_TEST_CASE(packed_local_cycle_storage_reduces_narrow_cycle_bytes) {
    std::vector<LocalCycle> local_cycles;
    local_cycles.reserve(128);
    for (size_t idx = 0; idx < 128; ++idx) {
        local_cycles.push_back(LocalCycle{idx, idx + 3, idx % 2 == 0 ? 1 : -1});
    }

    const auto storage = detail::build_packed_local_cycle_storage(local_cycles);

    BOOST_CHECK(!storage.uses_wide_indices);
    BOOST_CHECK(storage.phases.uses_binary_phases);
    BOOST_CHECK_EQUAL(storage.compact_pairs.size(), 128UL);
    BOOST_CHECK_EQUAL(detail::local_cycle_src(storage, 7), 7UL);
    BOOST_CHECK_EQUAL(detail::local_cycle_tgt(storage, 7), 10UL);
    BOOST_CHECK_EQUAL(detail::local_cycle_phase(storage, 7), -1);
    constexpr size_t legacy_packed_local_cycle_bytes =
        ((sizeof(uint32_t) * 2 + sizeof(int8_t) + alignof(uint32_t) - 1) / alignof(uint32_t)) * alignof(uint32_t);
    BOOST_CHECK_LT(detail::local_cycle_storage_bytes(storage), local_cycles.size() * legacy_packed_local_cycle_bytes);

    for (size_t idx = 0; idx < local_cycles.size(); ++idx) {
        BOOST_CHECK_EQUAL(detail::local_cycle_src(storage, idx), local_cycles[idx].src);
        BOOST_CHECK_EQUAL(detail::local_cycle_tgt(storage, idx), local_cycles[idx].tgt);
        BOOST_CHECK_EQUAL(detail::local_cycle_phase(storage, idx), local_cycles[idx].phase);
    }
}

BOOST_AUTO_TEST_CASE(packed_cross_rank_storage_bit_packs_binary_phases) {
    std::vector<CrossRankCycles> binary_cross_rank(2);
    std::vector<CrossRankCycles> wide_phase_cross_rank(2);

    binary_cross_rank[1].out_indices.resize(128);
    binary_cross_rank[1].out_phases.resize(128);
    binary_cross_rank[1].in_indices.resize(128);
    binary_cross_rank[1].in_phases.resize(128);
    wide_phase_cross_rank[1] = binary_cross_rank[1];

    for (size_t idx = 0; idx < 128; ++idx) {
        const int phase = idx % 2 == 0 ? 1 : -1;
        binary_cross_rank[1].out_indices[idx] = idx + 5;
        binary_cross_rank[1].out_phases[idx] = phase;
        binary_cross_rank[1].in_indices[idx] = idx + 1005;
        binary_cross_rank[1].in_phases[idx] = -phase;
        wide_phase_cross_rank[1].out_indices[idx] = idx + 5;
        wide_phase_cross_rank[1].out_phases[idx] = phase;
        wide_phase_cross_rank[1].in_indices[idx] = idx + 1005;
        wide_phase_cross_rank[1].in_phases[idx] = -phase;
    }
    wide_phase_cross_rank[1].out_phases[0] = 2;

    const auto binary_storage = detail::build_packed_cross_rank_storage(binary_cross_rank);
    const auto wide_phase_storage = detail::build_packed_cross_rank_storage(wide_phase_cross_rank);

    BOOST_CHECK(binary_storage.out_phases.uses_binary_phases);
    BOOST_CHECK(binary_storage.in_phases.uses_binary_phases);
    BOOST_CHECK(!wide_phase_storage.out_phases.uses_binary_phases);
    BOOST_CHECK_EQUAL(detail::cross_rank_out_phase(binary_storage, 1, 1), -1);
    BOOST_CHECK_EQUAL(detail::cross_rank_in_phase(binary_storage, 1, 1), 1);
    BOOST_CHECK_LT(detail::cross_rank_storage_bytes(binary_storage),
                   detail::cross_rank_storage_bytes(wide_phase_storage));
}
