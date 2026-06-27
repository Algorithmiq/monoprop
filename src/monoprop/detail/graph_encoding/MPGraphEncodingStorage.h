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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingCompression.h"

namespace monoprop::detail {

inline auto checked_mpi_int(size_t value, const char *what) -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the MPI int limit {}.", what, value, std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

inline auto checked_packed_phase(int value, const char *what) -> int8_t {
    if (value < static_cast<int>(std::numeric_limits<int8_t>::min())
        || value > static_cast<int>(std::numeric_limits<int8_t>::max())) {
        throw std::overflow_error(std::format("{} {} exceeds the 8-bit phase limit.", what, value));
    }
    return static_cast<int8_t>(value);
}

inline constexpr size_t kPackedPhaseWordBits = std::numeric_limits<uint64_t>::digits;

inline auto packed_phase_word_count(size_t count) -> size_t {
    return count == 0 ? 0 : (count + kPackedPhaseWordBits - 1) / kPackedPhaseWordBits;
}

inline auto packed_phase_word_index(size_t idx) -> size_t {
    return idx / kPackedPhaseWordBits;
}

inline auto packed_phase_bit_mask(size_t idx) -> uint64_t {
    return uint64_t{1} << (idx % kPackedPhaseWordBits);
}

inline auto is_binary_phase(int value) -> bool {
    return value == -1 || value == 1;
}

inline auto make_packed_phase_storage(size_t count, bool use_binary_phases) -> PackedPhaseStorage {
    PackedPhaseStorage storage;
    storage.uses_binary_phases = use_binary_phases;
    storage.total_count = count;
    if (use_binary_phases) {
        storage.phase_words.assign(packed_phase_word_count(count), 0);
    }
    else {
        storage.phase_values.resize(count);
    }
    return storage;
}

inline auto set_packed_phase(PackedPhaseStorage &storage, size_t idx, int value, const char *what) -> void {
    if (storage.uses_binary_phases) {
        if (!is_binary_phase(value)) {
            throw std::overflow_error(std::format("{} {} is not representable as a binary packed phase.", what, value));
        }
        if (value < 0) {
            storage.phase_words[packed_phase_word_index(idx)] |= packed_phase_bit_mask(idx);
        }
        else {
            storage.phase_words[packed_phase_word_index(idx)] &= ~packed_phase_bit_mask(idx);
        }
        return;
    }

    storage.phase_values[idx] = checked_packed_phase(value, what);
}

inline auto packed_phase_at(const PackedPhaseStorage &storage, size_t idx) -> int {
    if (storage.uses_binary_phases) {
        return (storage.phase_words[packed_phase_word_index(idx)] & packed_phase_bit_mask(idx)) != 0 ? -1 : 1;
    }
    return static_cast<int>(storage.phase_values[idx]);
}

inline auto packed_phase_storage_bytes(const PackedPhaseStorage &storage) -> size_t {
    return storage.uses_binary_phases ? storage.phase_words.capacity() * sizeof(uint64_t)
                                      : storage.phase_values.capacity() * sizeof(int8_t);
}

inline auto pack_local_cycle_pair(size_t src, size_t tgt) -> uint64_t {
    return (static_cast<uint64_t>(checked_packed_index(src, "Local cycle source index")) << 32)
           | static_cast<uint64_t>(checked_packed_index(tgt, "Local cycle target index"));
}

inline auto packed_local_cycle_src(uint64_t pair) -> size_t {
    return static_cast<size_t>(pair >> 32);
}

inline auto packed_local_cycle_tgt(uint64_t pair) -> size_t {
    return static_cast<size_t>(static_cast<uint32_t>(pair));
}

inline auto build_packed_local_cycle_storage(std::vector<LocalCycle> local_cycles) -> PackedLocalCycleStorage {
    PackedLocalCycleStorage storage;
    const auto max_packed_index = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    bool uses_wide_indices = false;
    bool uses_binary_phases = true;
    for (const auto &cycle : local_cycles) {
        uses_wide_indices = uses_wide_indices || cycle.src > max_packed_index || cycle.tgt > max_packed_index;
        uses_binary_phases = uses_binary_phases && is_binary_phase(cycle.phase);
        if (uses_wide_indices && !uses_binary_phases) {
            break;
        }
    }

    storage.uses_wide_indices = uses_wide_indices;
    storage.phases = make_packed_phase_storage(local_cycles.size(), uses_binary_phases);

    if (uses_wide_indices) {
        storage.wide_src_indices.resize(local_cycles.size());
        storage.wide_tgt_indices.resize(local_cycles.size());
        for (size_t idx = 0; idx < local_cycles.size(); ++idx) {
            const auto &cycle = local_cycles[idx];
            storage.wide_src_indices[idx] = cycle.src;
            storage.wide_tgt_indices[idx] = cycle.tgt;
            set_packed_phase(storage.phases, idx, cycle.phase, "Local cycle phase");
        }
        return storage;
    }

    storage.compact_pairs.resize(local_cycles.size());
    for (size_t idx = 0; idx < local_cycles.size(); ++idx) {
        const auto &cycle = local_cycles[idx];
        storage.compact_pairs[idx] = pack_local_cycle_pair(cycle.src, cycle.tgt);
        set_packed_phase(storage.phases, idx, cycle.phase, "Local cycle phase");
    }
    return storage;
}

inline auto local_cycle_src(const PackedLocalCycleStorage &storage, size_t idx) -> size_t {
    return storage.uses_wide_indices ? storage.wide_src_indices[idx]
                                     : packed_local_cycle_src(storage.compact_pairs[idx]);
}

inline auto local_cycle_tgt(const PackedLocalCycleStorage &storage, size_t idx) -> size_t {
    return storage.uses_wide_indices ? storage.wide_tgt_indices[idx]
                                     : packed_local_cycle_tgt(storage.compact_pairs[idx]);
}

inline auto local_cycle_phase(const PackedLocalCycleStorage &storage, size_t idx) -> int {
    return packed_phase_at(storage.phases, idx);
}

inline auto build_packed_cross_rank_storage(std::vector<CrossRankCycles> cross_rank) -> PackedCrossRankStorage {
    PackedCrossRankStorage storage;
    storage.ranges.resize(cross_rank.size());

    size_t total_out = 0;
    size_t total_in = 0;
    bool out_uses_binary_phases = true;
    bool in_uses_binary_phases = true;
    for (const auto &cycles : cross_rank) {
        total_out += cycles.out_size();
        total_in += cycles.in_size();
        storage.out_indices_wide = storage.out_indices_wide
                                   || std::any_of(cycles.out_indices.begin(), cycles.out_indices.end(), [](size_t idx) {
                                          return idx > static_cast<size_t>(std::numeric_limits<uint32_t>::max());
                                      });
        storage.in_indices_wide =
            storage.in_indices_wide || std::any_of(cycles.in_indices.begin(), cycles.in_indices.end(), [](size_t idx) {
                return idx > static_cast<size_t>(std::numeric_limits<uint32_t>::max());
            });
        out_uses_binary_phases =
            out_uses_binary_phases && std::all_of(cycles.out_phases.begin(), cycles.out_phases.end(), [](int phase) {
                return is_binary_phase(phase);
            });
        in_uses_binary_phases =
            in_uses_binary_phases && std::all_of(cycles.in_phases.begin(), cycles.in_phases.end(), [](int phase) {
                return is_binary_phase(phase);
            });
    }

    if (storage.out_indices_wide) {
        storage.wide_out_indices.resize(total_out);
    }
    else {
        storage.out_indices.resize(total_out);
    }
    storage.out_phases = make_packed_phase_storage(total_out, out_uses_binary_phases);

    if (storage.in_indices_wide) {
        storage.wide_in_indices.resize(total_in);
    }
    else {
        storage.in_indices.resize(total_in);
    }
    storage.in_phases = make_packed_phase_storage(total_in, in_uses_binary_phases);

    size_t out_offset = 0;
    size_t in_offset = 0;
    for (size_t rank = 0; rank < cross_rank.size(); ++rank) {
        const auto &cycles = cross_rank[rank];
        auto &range = storage.ranges[rank];
        range.out_offset = out_offset;
        range.out_count = cycles.out_size();
        range.in_offset = in_offset;
        range.in_count = cycles.in_size();

        for (size_t idx = 0; idx < cycles.out_size(); ++idx) {
            if (storage.out_indices_wide) {
                storage.wide_out_indices[out_offset + idx] = cycles.out_indices[idx];
            }
            else {
                storage.out_indices[out_offset + idx] =
                    checked_packed_index(cycles.out_indices[idx], "Cross-rank outgoing index");
            }
            set_packed_phase(storage.out_phases, out_offset + idx, cycles.out_phases[idx], "Cross-rank outgoing phase");
        }

        for (size_t idx = 0; idx < cycles.in_size(); ++idx) {
            if (storage.in_indices_wide) {
                storage.wide_in_indices[in_offset + idx] = cycles.in_indices[idx];
            }
            else {
                storage.in_indices[in_offset + idx] =
                    checked_packed_index(cycles.in_indices[idx], "Cross-rank incoming index");
            }
            set_packed_phase(storage.in_phases, in_offset + idx, cycles.in_phases[idx], "Cross-rank incoming phase");
        }

        out_offset += cycles.out_size();
        in_offset += cycles.in_size();
    }

    return storage;
}

inline auto cross_rank_out_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const size_t offset = storage.ranges[rank].out_offset + idx;
    return storage.out_indices_wide ? storage.wide_out_indices[offset]
                                    : static_cast<size_t>(storage.out_indices[offset]);
}

inline auto cross_rank_out_phase(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> int {
    return packed_phase_at(storage.out_phases, storage.ranges[rank].out_offset + idx);
}

inline auto cross_rank_in_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const size_t offset = storage.ranges[rank].in_offset + idx;
    return storage.in_indices_wide ? storage.wide_in_indices[offset] : static_cast<size_t>(storage.in_indices[offset]);
}

inline auto cross_rank_in_phase(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> int {
    return packed_phase_at(storage.in_phases, storage.ranges[rank].in_offset + idx);
}

inline auto local_cycle_storage_bytes(const PackedLocalCycleStorage &storage) -> size_t {
    size_t bytes = packed_phase_storage_bytes(storage.phases);
    if (storage.uses_wide_indices) {
        bytes += storage.wide_src_indices.capacity() * sizeof(size_t);
        bytes += storage.wide_tgt_indices.capacity() * sizeof(size_t);
        return bytes;
    }

    return bytes + storage.compact_pairs.capacity() * sizeof(uint64_t);
}

inline auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t {
    size_t bytes = storage.ranges.capacity() * sizeof(CrossRankStorageRange)
                   + packed_phase_storage_bytes(storage.out_phases) + packed_phase_storage_bytes(storage.in_phases);
    bytes += storage.out_indices_wide ? storage.wide_out_indices.capacity() * sizeof(size_t)
                                      : storage.out_indices.capacity() * sizeof(uint32_t);
    bytes += storage.in_indices_wide ? storage.wide_in_indices.capacity() * sizeof(size_t)
                                     : storage.in_indices.capacity() * sizeof(uint32_t);
    return bytes;
}

inline auto layer_exchange_layout_storage_bytes(const LayerExchangeLayout &layout) -> size_t {
    return layout.counts.capacity() * sizeof(int) + layout.displs.capacity() * sizeof(int);
}

template <typename CrossRankRange>
inline auto build_layer_exchange_layout_impl(const std::vector<CrossRankRange> &cross_rank, int scale)
    -> LayerExchangeLayout {
    LayerExchangeLayout layout;
    layout.counts.resize(cross_rank.size());
    layout.displs.resize(cross_rank.size());

    size_t total_count = 0;
    for (size_t rank = 0; rank < cross_rank.size(); ++rank) {
        const auto &cr = cross_rank[rank];
        const size_t count = static_cast<size_t>(scale) * (cr.out_size() + cr.in_size());
        layout.counts[rank] = checked_mpi_int(count, "Layer exchange count");
        total_count += count;
    }

    if (!layout.displs.empty()) {
        size_t displacement = 0;
        for (size_t rank = 0; rank < cross_rank.size(); ++rank) {
            layout.displs[rank] = checked_mpi_int(displacement, "Layer exchange displacement");
            displacement += static_cast<size_t>(layout.counts[rank]);
        }
    }

    layout.total_count = total_count;
    return layout;
}

inline auto build_layer_exchange_layout(const std::vector<CrossRankCycles> &cross_rank, int scale)
    -> LayerExchangeLayout {
    return build_layer_exchange_layout_impl(cross_rank, scale);
}

struct CrossRankMaskRange final {
    size_t out_offset = 0;
    size_t out_count = 0;
    size_t in_offset = 0;
    size_t in_count = 0;

    bool empty() const { return out_count == 0 && in_count == 0; }
    size_t out_size() const { return out_count; }
    size_t in_size() const { return in_count; }
};

struct ExecutionPlanStorage final {
    std::vector<CompressedCosineData> cos_data_blocks;
    std::vector<CompressedPositionData> local_cycle_position_blocks;
    std::vector<CompressedPositionData> cross_rank_out_position_blocks;
    std::vector<CompressedPositionData> cross_rank_in_position_blocks;
};

inline auto build_layer_exchange_layout(const std::vector<CrossRankMaskRange> &cross_rank, int scale)
    -> LayerExchangeLayout {
    return build_layer_exchange_layout_impl(cross_rank, scale);
}

inline auto build_layer_storage(CompressedCosineData cos_data,
                                std::vector<LocalCycle> local_cycs,
                                std::vector<CrossRankCycles> cross_rank) -> std::shared_ptr<LayerStorage> {
    auto storage = std::make_shared<LayerStorage>();
    storage->cos_data = std::move(cos_data);
    storage->evolution_exchange_layout = build_layer_exchange_layout(cross_rank, 1);
    storage->local_cycles = build_packed_local_cycle_storage(std::move(local_cycs));
    storage->cross_rank = build_packed_cross_rank_storage(std::move(cross_rank));
    return storage;
}

inline auto build_layer_storage(VecZ cos_inds,
                                std::vector<LocalCycle> local_cycs,
                                std::vector<CrossRankCycles> cross_rank) -> std::shared_ptr<LayerStorage> {
    return build_layer_storage(build_compressed_cosine_data(cos_inds), std::move(local_cycs), std::move(cross_rank));
}

} // namespace monoprop::detail
