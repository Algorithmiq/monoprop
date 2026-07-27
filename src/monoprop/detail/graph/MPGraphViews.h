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
#include <stdexcept>
#include <utility>
#include <vector>

#include <format>
#include "monoprop/detail/print_compat.h"

#include "monoprop/detail/graph/MPGraphLayers.h"

namespace monoprop {

// Per-rank breakdown of graph memory, in bytes; the fields sum to total_bytes().
struct GraphMemoryBreakdown final {
    size_t layer_descriptor_bytes = 0;
    size_t layer_storage_object_bytes = 0;
    size_t cos_data_bytes = 0;
    size_t cross_rank_bytes = 0;
    size_t exchange_layout_bytes = 0;

    auto total_bytes() const -> size_t {
        return layer_descriptor_bytes + layer_storage_object_bytes + cos_data_bytes + cross_rank_bytes
               + exchange_layout_bytes;
    }

    // Field-wise sum, so a sharded propagator can aggregate its per-shard graph breakdowns.
    auto operator+=(const GraphMemoryBreakdown &o) -> GraphMemoryBreakdown & {
        layer_descriptor_bytes += o.layer_descriptor_bytes;
        layer_storage_object_bytes += o.layer_storage_object_bytes;
        cos_data_bytes += o.cos_data_bytes;
        cross_rank_bytes += o.cross_rank_bytes;
        exchange_layout_bytes += o.exchange_layout_bytes;
        return *this;
    }
};

// Windowed, optionally-reversed read-only view over a graph's layer vector. `reverse` traverses the window
// newest-first (Schrödinger replay order). Non-owning — the layer vector must outlive the view.
class MPGraphView {
public:
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
