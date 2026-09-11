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
#include <print>

#include "monoprop/detail/graph/MPGraphLayers.h"

namespace monoprop {

// A layer index at or past the end of the graph window being indexed.
class LayerIndexOutOfRange : public std::out_of_range {
public:
    using std::out_of_range::out_of_range;
};

// One rank's own graph memory only.
struct GraphMemoryBreakdown final {
    size_t layer_descriptor_bytes = 0;
    size_t layer_storage_object_bytes = 0;
    size_t cos_data_bytes = 0;
    size_t cross_rank_bytes = 0;
    size_t exchange_layout_bytes = 0;

    // Diagnostics, outside total_bytes(): counts, or a subset of a byte field, not additions to it.
    size_t slot_record_bytes = 0;
    size_t layer_cores = 0;
    size_t slot_records = 0; // slot_records / layer_cores is the flat world P
    size_t occupied_slots = 0;
    size_t cross_rank_endpoints = 0;

    auto total_bytes() const -> size_t {
        return layer_descriptor_bytes + layer_storage_object_bytes + cos_data_bytes + cross_rank_bytes
               + exchange_layout_bytes;
    }

    // Lets a partitioned propagator aggregate its per-partition graph breakdowns.
    auto operator+=(const GraphMemoryBreakdown &o) -> GraphMemoryBreakdown & {
        layer_descriptor_bytes += o.layer_descriptor_bytes;
        layer_storage_object_bytes += o.layer_storage_object_bytes;
        cos_data_bytes += o.cos_data_bytes;
        cross_rank_bytes += o.cross_rank_bytes;
        exchange_layout_bytes += o.exchange_layout_bytes;
        slot_record_bytes += o.slot_record_bytes;
        layer_cores += o.layer_cores;
        slot_records += o.slot_records;
        occupied_slots += o.occupied_slots;
        cross_rank_endpoints += o.cross_rank_endpoints;
        return *this;
    }
};

// The layer-window vocabulary shared by MPGraph and its views: the deriving class supplies get_layer(),
// this supplies everything derivable from it. Deducing this rather than CRTP, so a deriving class need not
// name itself as a template argument, and the mixin stays an empty base.
struct LayerWindow {
    template <typename Self>
    auto get_layer_traversal(this const Self &self, size_t layer_idx) -> LayerTraversal {
        return self.get_layer(layer_idx).traversal();
    }
};

// `reverse` traverses the window newest-first (Schrödinger replay order). Non-owning — the layer vector
// must outlive the view.
class MPGraphView : public LayerWindow {
public:
    MPGraphView(const std::vector<Layer> &layers, size_t base, size_t count, bool reverse)
        : layers_(&layers),
          base_(base),
          count_(count),
          reverse_(reverse) {}

    auto layers() const -> size_t { return count_; }

    auto get_layer(size_t layer_idx) const -> const Layer & { return (*layers_)[checked_layer_offset(layer_idx)]; }

private:
    auto checked_layer_offset(size_t layer_idx) const -> size_t {
        if (layer_idx >= count_) {
            throw LayerIndexOutOfRange(std::format("Layer {} is out of range (layers={})", layer_idx, count_));
        }

        return base_ + (reverse_ ? count_ - 1 - layer_idx : layer_idx);
    }

    const std::vector<Layer> *layers_ = nullptr;
    size_t base_ = 0;
    size_t count_ = 0;
    bool reverse_ = false;
};

} // namespace monoprop
