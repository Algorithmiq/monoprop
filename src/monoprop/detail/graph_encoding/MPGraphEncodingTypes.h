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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop {

struct LayerExchangeLayout final {
    std::vector<int> counts;
    std::vector<int> displs;
    size_t total_count = 0;
};

struct CosineSpan final {
    size_t start = 0;
    uint16_t count = 0;
};

struct StoredPositionSpan final {
    size_t logical_start = 0;
    uint32_t position_start = 0;
    uint16_t count = 0;
};

struct CompressedCosineData final {
    size_t total_count = 0;
    std::vector<size_t> chunk_bases;
    std::vector<size_t> chunk_span_starts;
    std::vector<uint16_t> span_offsets;
    std::vector<uint8_t> span_counts;
    bool has_wide_start_values = false;

    auto chunk_count() const -> size_t { return chunk_bases.size(); }
    auto span_count() const -> size_t { return span_offsets.size(); }
    auto empty() const -> bool { return span_offsets.empty(); }
    auto has_wide_starts() const -> bool { return has_wide_start_values; }
    auto reset() -> void {
        total_count = 0;
        chunk_bases.clear();
        chunk_span_starts.clear();
        span_offsets.clear();
        span_counts.clear();
        has_wide_start_values = false;
    }
};

struct CompressedPositionData final {
    size_t total_count = 0;
    std::vector<StoredPositionSpan> spans;

    auto span_count() const -> size_t { return spans.size(); }
    auto empty() const -> bool { return total_count == 0; }
    auto reset() -> void {
        total_count = 0;
        spans.clear();
    }
};

struct PackedPhaseStorage final {
    bool uses_binary_phases = false;
    size_t total_count = 0;
    std::vector<uint64_t> phase_words;
    std::vector<int8_t> phase_values;

    auto size() const -> size_t { return total_count; }
    auto empty() const -> bool { return total_count == 0; }
};

struct PackedLocalCycleStorage final {
    bool uses_wide_indices = false;
    std::vector<uint64_t> compact_pairs;
    std::vector<size_t> wide_src_indices;
    std::vector<size_t> wide_tgt_indices;
    PackedPhaseStorage phases;

    auto size() const -> size_t { return uses_wide_indices ? wide_src_indices.size() : compact_pairs.size(); }
};

struct CrossRankStorageRange final {
    size_t out_offset = 0;
    size_t out_count = 0;
    size_t in_offset = 0;
    size_t in_count = 0;
};

struct PackedCrossRankStorage final {
    bool out_indices_wide = false;
    bool in_indices_wide = false;
    std::vector<CrossRankStorageRange> ranges;
    std::vector<uint32_t> out_indices;
    std::vector<size_t> wide_out_indices;
    PackedPhaseStorage out_phases;
    std::vector<uint32_t> in_indices;
    std::vector<size_t> wide_in_indices;
    PackedPhaseStorage in_phases;

    auto rank_count() const -> size_t { return ranges.size(); }
    auto out_size(size_t rank) const -> size_t { return ranges[rank].out_count; }
    auto in_size(size_t rank) const -> size_t { return ranges[rank].in_count; }
    auto empty() const -> bool { return out_phases.empty() && in_phases.empty(); }
};

struct LayerStorage final {
    CompressedCosineData cos_data;
    PackedLocalCycleStorage local_cycles;
    PackedCrossRankStorage cross_rank;
    LayerExchangeLayout evolution_exchange_layout;
};

} // namespace monoprop
