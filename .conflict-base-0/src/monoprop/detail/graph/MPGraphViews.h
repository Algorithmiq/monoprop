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

#include <format>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/detail/graph/MPGraphLayers.h"

namespace monoprop {

struct LayerExecutionPlan final {
    LayerExecutionPlan() = default;

    explicit LayerExecutionPlan(std::shared_ptr<LayerStorage> storage)
        : storage_(std::move(storage)),
          use_original_cos_data_(true),
          use_original_local_cycles_(true),
          use_original_cross_rank_(true) {}

    LayerExecutionPlan(std::shared_ptr<LayerStorage> storage,
                       std::shared_ptr<detail::ExecutionPlanStorage> execution_storage,
                       bool use_original_cos_data,
                       size_t cos_data_index,
                       bool use_original_local_cycles,
                       size_t local_cycle_position_block_index,
                       bool use_original_cross_rank,
                       size_t cross_rank_position_block_index,
                       std::vector<detail::CrossRankMaskRange> cross_rank_ranges)
        : storage_(std::move(storage)),
          execution_storage_(std::move(execution_storage)),
          use_original_cos_data_(use_original_cos_data),
          use_original_local_cycles_(use_original_local_cycles),
          use_original_cross_rank_(use_original_cross_rank),
          cos_data_index_(cos_data_index),
          local_cycle_position_block_index_(local_cycle_position_block_index),
          cross_rank_position_block_index_(cross_rank_position_block_index),
          cross_rank_ranges_(std::move(cross_rank_ranges)) {
        if (!use_original_cross_rank_) {
            evolution_exchange_layout_ = detail::build_layer_exchange_layout(cross_rank_ranges_, 1);
        }
    }

    auto traversal() const -> LayerTraversal {
        return LayerTraversal(
            *storage_,
            execution_storage_.get(),
            use_original_cos_data_,
            cos_data_index_,
            use_original_local_cycles_,
            local_cycle_position_block_index_,
            use_original_cross_rank_,
            cross_rank_position_block_index_,
            cross_rank_ranges_,
            use_original_cross_rank_ ? storage_->evolution_exchange_layout : evolution_exchange_layout_);
    }

    auto shared_storage() const -> std::shared_ptr<const LayerStorage> { return storage_; }
    auto shared_execution_storage() const -> std::shared_ptr<const detail::ExecutionPlanStorage> {
        return execution_storage_;
    }

private:
    std::shared_ptr<LayerStorage> storage_;
    std::shared_ptr<detail::ExecutionPlanStorage> execution_storage_;
    bool use_original_cos_data_ = false;
    bool use_original_local_cycles_ = false;
    bool use_original_cross_rank_ = false;
    size_t cos_data_index_ = 0;
    size_t local_cycle_position_block_index_ = 0;
    size_t cross_rank_position_block_index_ = 0;
    std::vector<detail::CrossRankMaskRange> cross_rank_ranges_;
    LayerExchangeLayout evolution_exchange_layout_;
};

struct GraphMemoryBreakdown final {
    size_t layer_descriptor_bytes = 0;
    size_t layer_storage_object_bytes = 0;
    size_t cos_data_bytes = 0;
    size_t local_cycle_bytes = 0;
    size_t cross_rank_bytes = 0;
    size_t exchange_layout_bytes = 0;
    size_t execution_plan_overhead_bytes = 0;
    size_t execution_plan_cos_data_bytes = 0;
    size_t execution_plan_local_cycle_position_bytes = 0;
    size_t execution_plan_cross_rank_position_bytes = 0;
    size_t execution_plan_bytes = 0;

    auto total_bytes() const -> size_t {
        return layer_descriptor_bytes + layer_storage_object_bytes + cos_data_bytes + local_cycle_bytes
               + cross_rank_bytes + exchange_layout_bytes + execution_plan_bytes;
    }
};

class MPExecutionPlan {
public:
    MPExecutionPlan() = default;

    MPExecutionPlan(bool schrodinger, std::vector<LayerExecutionPlan> layers)
        : schrodinger_(schrodinger),
          layers_(std::move(layers)) {}

    auto layers() const -> size_t { return layers_.size(); }
    auto is_schrodinger() const -> bool { return schrodinger_; }
    auto storage_memory_usage() const -> GraphMemoryBreakdown;

    auto get_layer(size_t layer_idx) const -> const LayerExecutionPlan & {
        if (layer_idx >= layers_.size()) {
            throw std::out_of_range(std::format("Layer {} is out of range (layers={})", layer_idx, layers_.size()));
        }
        return layers_[layer_idx];
    }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

private:
    bool schrodinger_ = false;
    std::vector<LayerExecutionPlan> layers_;
};

class MPGraphView {
public:
    MPGraphView() = default;

    MPGraphView(const std::vector<Layer> &layers, size_t base, size_t count, bool reverse)
        : layers_(&layers),
          base_(base),
          count_(count),
          reverse_(reverse) {}

    auto layers() const -> size_t { return count_; }

    auto get_layer(size_t layer_idx) const -> const Layer & { return (*layers_)[checked_layer_offset(layer_idx)]; }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

private:
    auto checked_layer_offset(size_t layer_idx) const -> size_t {
        if (layer_idx >= count_) {
            throw std::out_of_range(std::format("Layer {} is out of range (layers={})", layer_idx, count_));
        }

        return base_ + (reverse_ ? count_ - 1 - layer_idx : layer_idx);
    }

    const std::vector<Layer> *layers_ = nullptr;
    size_t base_ = 0;
    size_t count_ = 0;
    bool reverse_ = false;
};

} // namespace monoprop
