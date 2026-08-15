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

    // Diagnostics, deliberately EXCLUDED from total_bytes(): the first three are either a
    // subset of a field above or memory that total_bytes() has never counted, and folding
    // them in would silently redefine graph_memory_bytes() mid-flight, so an A/B against an
    // older build would compare two different quantities. The rest are counts, not bytes.
    //
    // The point of the split: a per-layer array indexed by rank is sized by the FLAT world
    // (mpi::size on a Hybrid comm is ranks x partitions), so it costs O(P) per layer per
    // partition and O(P^2) across the job. slot_bytes is that part; traffic_bytes is the
    // part that scales with terms actually crossing, which is real work.
    size_t slot_record_bytes = 0;      // cross_rank ranges[]: one record per world slot, occupied or not
    size_t recv_cache_bytes = 0;       // evolution layout's resolve_recv transpose cache -- never in total_bytes()
    size_t derivative_layout_bytes = 0; // the lazily retained 2x layout AND its own recv cache -- likewise
    size_t layer_cores = 0;            // distinct LayerCores walked (shared cores counted once)
    size_t slot_records = 0;           // sum over cores of ranges.size(); divide by layer_cores to recover P
    size_t occupied_slots = 0;         // slots carrying any traffic: occupancy = occupied_slots / slot_records

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
        recv_cache_bytes += o.recv_cache_bytes;
        derivative_layout_bytes += o.derivative_layout_bytes;
        layer_cores += o.layer_cores;
        slot_records += o.slot_records;
        occupied_slots += o.occupied_slots;
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
