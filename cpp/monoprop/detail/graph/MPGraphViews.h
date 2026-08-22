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

    // Diagnostics, deliberately EXCLUDED from total_bytes(): each is either a count, a subset of a
    // field above, or memory total_bytes() has never counted. Folding any of them in would silently
    // redefine graph_memory_bytes() mid-flight, so an A/B against an older build would compare two
    // different quantities.
    //
    // The point of the split: a per-layer array indexed by rank is sized by the FLAT world
    // (mpi::size on a Hybrid comm is ranks x partitions), so it costs O(P) per layer per
    // partition and O(P^2) across the job. slot_record_bytes is that part; the endpoint count below
    // is the part that scales with terms actually crossing, which is real work.
    size_t slot_record_bytes = 0; // one record per STORED world slot -- occupied only, once sparse
    size_t layer_cores = 0;       // distinct LayerCores walked (shared cores counted once)
    size_t slot_records = 0;      // the flat world P per core, so slot_records / layer_cores == P
    size_t occupied_slots = 0;    // slots carrying any traffic: occupancy = occupied_slots / slot_records
    // Cross-rank endpoints -- the traffic itself, and the ceiling on occupied_slots, since an
    // occupied slot holds at least one endpoint. Unlike slot_records it does not depend on P, so the
    // two together say how much of the slot array is information and how much is reserved-and-empty.
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

// `reverse` traverses the window newest-first (Schrödinger replay order). Non-owning — the layer vector
// must outlive the view.
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
