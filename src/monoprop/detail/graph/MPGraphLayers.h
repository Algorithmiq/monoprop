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

#include <memory>
#include <utility>
#include <vector>

#include "monoprop/MPGraphEncoding.h"

namespace monoprop {

// Runtime code talks to both full layers and masked execution plans through the
// same logical traversal surface. The storage stays compressed; traversal only
// remaps logical positions back to stored positions when a masked plan is used.
struct LayerTraversal final {
    LayerTraversal() = default;

    explicit LayerTraversal(const LayerStorage &storage)
        : storage_(&storage),
          cos_data_(&storage.cos_data),
          evolution_exchange_layout_(&storage.evolution_exchange_layout) {}

    LayerTraversal(const LayerStorage &storage,
                   const detail::ExecutionPlanStorage *execution_storage,
                   bool use_original_cos_data,
                   size_t cos_data_index,
                   bool use_original_local_cycles,
                   size_t local_cycle_position_block_index,
                   bool use_original_cross_rank,
                   size_t cross_rank_position_block_index,
                   const std::vector<detail::CrossRankMaskRange> &cross_rank_ranges,
                   const LayerExchangeLayout &evolution_exchange_layout)
        : storage_(&storage),
          cos_data_(use_original_cos_data ? &storage.cos_data : &execution_storage->cos_data_blocks[cos_data_index]),
          local_cycle_positions_(
              use_original_local_cycles
                  ? nullptr
                  : &execution_storage->local_cycle_position_blocks[local_cycle_position_block_index]),
          cross_rank_out_positions_(
              use_original_cross_rank
                  ? nullptr
                  : &execution_storage->cross_rank_out_position_blocks[cross_rank_position_block_index]),
          cross_rank_in_positions_(
              use_original_cross_rank
                  ? nullptr
                  : &execution_storage->cross_rank_in_position_blocks[cross_rank_position_block_index]),
          cross_rank_ranges_(use_original_cross_rank ? nullptr : &cross_rank_ranges),
          evolution_exchange_layout_(use_original_cross_rank ? &storage.evolution_exchange_layout
                                                             : &evolution_exchange_layout) {}

    auto cos_data() const -> const CompressedCosineData & { return *cos_data_; }
    auto num_cos_inds() const -> size_t { return cos_data_->total_count; }
    auto cos_span_count() const -> size_t { return cos_data_->span_count(); }

    template <typename Func>
    auto for_each_cos_span(Func &&func) const -> void {
        detail::for_each_cosine_span(*cos_data_, std::forward<Func>(func));
    }

    auto local_cycle_count() const -> size_t {
        return local_cycle_positions_ == nullptr ? storage_->local_cycles.size() : local_cycle_positions_->total_count;
    }

    auto local_cycle_src(size_t idx) const -> size_t {
        return detail::local_cycle_src(storage_->local_cycles, local_cycle_position(idx));
    }

    auto local_cycle_tgt(size_t idx) const -> size_t {
        return detail::local_cycle_tgt(storage_->local_cycles, local_cycle_position(idx));
    }

    auto local_cycle_phase(size_t idx) const -> int {
        return detail::local_cycle_phase(storage_->local_cycles, local_cycle_position(idx));
    }

    template <typename Func>
    auto for_each_local_cycle_range(size_t begin, size_t end, Func &&func) const -> void {
        if (begin == end) {
            return;
        }
        if (local_cycle_positions_ == nullptr) {
            for (size_t idx = begin; idx < end; ++idx) {
                func(idx,
                     detail::local_cycle_src(storage_->local_cycles, idx),
                     detail::local_cycle_tgt(storage_->local_cycles, idx),
                     detail::local_cycle_phase(storage_->local_cycles, idx));
            }
            return;
        }

        detail::for_each_compressed_position_range(
            *local_cycle_positions_,
            begin,
            end,
            [this, &func](size_t logical_start, size_t position_start, size_t count) {
                for (size_t offset = 0; offset < count; ++offset) {
                    const size_t logical_idx = logical_start + offset;
                    const size_t position = position_start + offset;
                    func(logical_idx,
                         detail::local_cycle_src(storage_->local_cycles, position),
                         detail::local_cycle_tgt(storage_->local_cycles, position),
                         detail::local_cycle_phase(storage_->local_cycles, position));
                }
            });
    }

    auto cross_rank_rank_count() const -> size_t { return storage_->cross_rank.rank_count(); }

    auto cross_rank_out_size(size_t rank) const -> size_t {
        return cross_rank_ranges_ == nullptr ? storage_->cross_rank.out_size(rank)
                                             : (*cross_rank_ranges_)[rank].out_size();
    }

    auto cross_rank_in_size(size_t rank) const -> size_t {
        return cross_rank_ranges_ == nullptr ? storage_->cross_rank.in_size(rank)
                                             : (*cross_rank_ranges_)[rank].in_size();
    }

    auto cross_rank_out_index(size_t rank, size_t idx) const -> size_t {
        return detail::cross_rank_out_index(storage_->cross_rank, rank, cross_rank_out_position(rank, idx));
    }

    auto cross_rank_out_phase(size_t rank, size_t idx) const -> int {
        return detail::cross_rank_out_phase(storage_->cross_rank, rank, cross_rank_out_position(rank, idx));
    }

    auto cross_rank_in_index(size_t rank, size_t idx) const -> size_t {
        return detail::cross_rank_in_index(storage_->cross_rank, rank, cross_rank_in_position(rank, idx));
    }

    auto cross_rank_in_phase(size_t rank, size_t idx) const -> int {
        return detail::cross_rank_in_phase(storage_->cross_rank, rank, cross_rank_in_position(rank, idx));
    }

    template <typename Func>
    auto for_each_cross_rank_out_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        if (begin == end) {
            return;
        }
        if (cross_rank_ranges_ == nullptr) {
            for (size_t idx = begin; idx < end; ++idx) {
                func(idx,
                     detail::cross_rank_out_index(storage_->cross_rank, rank, idx),
                     detail::cross_rank_out_phase(storage_->cross_rank, rank, idx));
            }
            return;
        }

        const auto &range = (*cross_rank_ranges_)[rank];
        detail::for_each_compressed_position_range(
            *cross_rank_out_positions_,
            range.out_offset + begin,
            range.out_offset + end,
            [this, &func, &range, rank](size_t logical_start, size_t position_start, size_t count) {
                for (size_t offset = 0; offset < count; ++offset) {
                    const size_t logical_idx = logical_start - range.out_offset + offset;
                    const size_t position = position_start + offset;
                    func(logical_idx,
                         detail::cross_rank_out_index(storage_->cross_rank, rank, position),
                         detail::cross_rank_out_phase(storage_->cross_rank, rank, position));
                }
            });
    }

    template <typename Func>
    auto for_each_cross_rank_in_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        if (begin == end) {
            return;
        }
        if (cross_rank_ranges_ == nullptr) {
            for (size_t idx = begin; idx < end; ++idx) {
                func(idx,
                     detail::cross_rank_in_index(storage_->cross_rank, rank, idx),
                     detail::cross_rank_in_phase(storage_->cross_rank, rank, idx));
            }
            return;
        }

        const auto &range = (*cross_rank_ranges_)[rank];
        detail::for_each_compressed_position_range(
            *cross_rank_in_positions_,
            range.in_offset + begin,
            range.in_offset + end,
            [this, &func, &range, rank](size_t logical_start, size_t position_start, size_t count) {
                for (size_t offset = 0; offset < count; ++offset) {
                    const size_t logical_idx = logical_start - range.in_offset + offset;
                    const size_t position = position_start + offset;
                    func(logical_idx,
                         detail::cross_rank_in_index(storage_->cross_rank, rank, position),
                         detail::cross_rank_in_phase(storage_->cross_rank, rank, position));
                }
            });
    }

    auto evolution_exchange_layout() const -> const LayerExchangeLayout & { return *evolution_exchange_layout_; }

private:
    auto local_cycle_position(size_t idx) const -> size_t {
        return local_cycle_positions_ == nullptr ? idx : detail::compressed_position_at(*local_cycle_positions_, idx);
    }

    auto cross_rank_out_position(size_t rank, size_t idx) const -> size_t {
        if (cross_rank_ranges_ == nullptr) {
            return idx;
        }
        const auto &range = (*cross_rank_ranges_)[rank];
        return detail::compressed_position_at(*cross_rank_out_positions_, range.out_offset + idx);
    }

    auto cross_rank_in_position(size_t rank, size_t idx) const -> size_t {
        if (cross_rank_ranges_ == nullptr) {
            return idx;
        }
        const auto &range = (*cross_rank_ranges_)[rank];
        return detail::compressed_position_at(*cross_rank_in_positions_, range.in_offset + idx);
    }

    const LayerStorage *storage_ = nullptr;
    const CompressedCosineData *cos_data_ = nullptr;
    const CompressedPositionData *local_cycle_positions_ = nullptr;
    const CompressedPositionData *cross_rank_out_positions_ = nullptr;
    const CompressedPositionData *cross_rank_in_positions_ = nullptr;
    const std::vector<detail::CrossRankMaskRange> *cross_rank_ranges_ = nullptr;
    const LayerExchangeLayout *evolution_exchange_layout_ = nullptr;
};

struct Layer final {
    Layer() : storage_(std::make_shared<LayerStorage>()) {}

    Layer(VecZ cos_inds, std::vector<LocalCycle> local_cycs, std::vector<CrossRankCycles> cross_rank)
        : storage_(detail::build_layer_storage(std::move(cos_inds), std::move(local_cycs), std::move(cross_rank))) {}

    Layer(CompressedCosineData cos_data, std::vector<LocalCycle> local_cycs, std::vector<CrossRankCycles> cross_rank)
        : storage_(detail::build_layer_storage(std::move(cos_data), std::move(local_cycs), std::move(cross_rank))) {}

    explicit Layer(std::shared_ptr<LayerStorage> storage) : storage_(std::move(storage)) {}

    auto traversal() const -> LayerTraversal { return LayerTraversal(*storage_); }

    auto cos_data() const -> const CompressedCosineData & { return storage_->cos_data; }
    auto cos_spans() const -> std::vector<CosineSpan> {
        return detail::materialize_stored_cosine_spans(storage_->cos_data);
    }
    auto has_wide_cos_spans() const -> bool { return storage_->cos_data.has_wide_starts(); }
    auto num_cos_inds() const -> size_t { return storage_->cos_data.total_count; }
    auto cos_span_count() const -> size_t { return storage_->cos_data.span_count(); }
    auto materialize_cos_inds() const -> VecZ { return detail::expand_compressed_cosine_data(storage_->cos_data); }

    template <typename Func>
    auto for_each_cos_span(Func &&func) const -> void {
        detail::for_each_cosine_span(storage_->cos_data, std::forward<Func>(func));
    }

    template <typename Func>
    auto for_each_cos_index(Func &&func) const -> void {
        for_each_cos_span([&func](const CosineSpan &span) {
            size_t idx = span.start;
            const size_t end = idx + static_cast<size_t>(span.count);
            for (; idx < end; ++idx) {
                func(idx);
            }
        });
    }

    auto local_cycle_count() const -> size_t { return storage_->local_cycles.size(); }
    auto local_cycle_src(size_t idx) const -> size_t { return detail::local_cycle_src(storage_->local_cycles, idx); }
    auto local_cycle_tgt(size_t idx) const -> size_t { return detail::local_cycle_tgt(storage_->local_cycles, idx); }
    auto local_cycle_phase(size_t idx) const -> int { return detail::local_cycle_phase(storage_->local_cycles, idx); }

    template <typename Func>
    auto for_each_local_cycle_range(size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx,
                 detail::local_cycle_src(storage_->local_cycles, idx),
                 detail::local_cycle_tgt(storage_->local_cycles, idx),
                 detail::local_cycle_phase(storage_->local_cycles, idx));
        }
    }

    auto cross_rank_rank_count() const -> size_t { return storage_->cross_rank.rank_count(); }
    auto cross_rank_out_size(size_t rank) const -> size_t { return storage_->cross_rank.out_size(rank); }
    auto cross_rank_in_size(size_t rank) const -> size_t { return storage_->cross_rank.in_size(rank); }
    auto cross_rank_out_index(size_t rank, size_t idx) const -> size_t {
        return detail::cross_rank_out_index(storage_->cross_rank, rank, idx);
    }
    auto cross_rank_out_phase(size_t rank, size_t idx) const -> int {
        return detail::cross_rank_out_phase(storage_->cross_rank, rank, idx);
    }
    auto cross_rank_in_index(size_t rank, size_t idx) const -> size_t {
        return detail::cross_rank_in_index(storage_->cross_rank, rank, idx);
    }
    auto cross_rank_in_phase(size_t rank, size_t idx) const -> int {
        return detail::cross_rank_in_phase(storage_->cross_rank, rank, idx);
    }

    template <typename Func>
    auto for_each_cross_rank_out_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx,
                 detail::cross_rank_out_index(storage_->cross_rank, rank, idx),
                 detail::cross_rank_out_phase(storage_->cross_rank, rank, idx));
        }
    }

    template <typename Func>
    auto for_each_cross_rank_in_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx,
                 detail::cross_rank_in_index(storage_->cross_rank, rank, idx),
                 detail::cross_rank_in_phase(storage_->cross_rank, rank, idx));
        }
    }

    auto evolution_exchange_layout() const -> const LayerExchangeLayout & {
        return storage_->evolution_exchange_layout;
    }
    auto shared_storage() const -> std::shared_ptr<LayerStorage> { return storage_; }

    auto empty() const -> bool {
        if (num_cos_inds() != 0 || local_cycle_count() != 0) {
            return false;
        }
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            if (cross_rank_out_size(rank) != 0 || cross_rank_in_size(rank) != 0) {
                return false;
            }
        }
        return true;
    }

    auto total_cycles() const -> size_t {
        size_t count = local_cycle_count();
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            count += cross_rank_in_size(rank);
        }
        return count;
    }

private:
    std::shared_ptr<LayerStorage> storage_;
};

} // namespace monoprop
