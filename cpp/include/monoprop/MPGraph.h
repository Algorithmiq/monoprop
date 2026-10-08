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
#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "monoprop/detail/graph/MPGraphLayers.h"
#include "monoprop/detail/graph/MPGraphViews.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

/// Ordered per-rank record of the evolution circuit, one Layer per generator.
class monoprop_EXPORT MPGraph : public LayerWindow {
private:
    using LayerIterator = std::vector<Layer>::iterator;

    bool schrodinger_;
    std::vector<Layer> layers_;
    size_t front_offset_ = 0;

    auto active_begin_index() const -> size_t { return front_offset_; }

    auto active_end_index() const -> size_t { return layers_.size(); }

    // Deducing this: const-ness of the returned iterator follows the object, so neither body is doubled.
    template <typename Self>
    auto active_begin_iterator(this Self &&self) {
        return self.layers_.begin() + static_cast<std::ptrdiff_t>(self.active_begin_index());
    }

    template <typename Self>
    auto active_end_iterator(this Self &&self) {
        return self.layers_.end();
    }

    auto append_position() -> LayerIterator { return schrodinger_ ? active_begin_iterator() : active_end_iterator(); }

    auto append_layer(Layer layer) -> void { layers_.emplace(append_position(), std::move(layer)); }

    auto checked_layer_offset(size_t layer_idx) const -> size_t {
        if (layer_idx >= layers()) {
            throw LayerIndexOutOfRange(std::format("Layer {} is out of range (layers={})", layer_idx, layers()));
        }
        return active_begin_index() + layer_idx;
    }

public:
    explicit MPGraph(bool schrodinger) : schrodinger_(schrodinger) {}

    explicit MPGraph(bool schrodinger, std::vector<Layer> layers)
        : schrodinger_(schrodinger),
          layers_(std::move(layers)) {}

    /// Gate info (param_index, gen_coeff, gate_index) is written onto `storage` here while it is still
    /// mutable, before it is frozen into the Layer's shared const core.
    auto append(std::shared_ptr<LayerCore> storage,
                size_t param_index = 0,
                double gen_coeff = 0.0,
                size_t gate_index = 0) -> void {
        storage->param_index = param_index;
        storage->gen_coeff = gen_coeff;
        storage->gate_index = gate_index;
        append_layer(Layer(std::move(storage)));
    }

    /// Slice the graph at `key` (the number of earliest operations to include); `contract` also removes
    /// the sliced part from this graph.
    auto slice_graph(size_t key, bool contract = false) -> MPGraph;

    auto slice_view(size_t key) const -> MPGraphView;

    auto layers() const -> size_t { return active_end_index() - active_begin_index(); }

    /// The layer at `layer_idx` in build order; throws LayerIndexOutOfRange at or past the end.
    // Deducing this: const-ness of the returned reference follows the object, so one body serves both.
    template <typename Self>
    auto get_layer(this Self &&self, size_t layer_idx) -> auto & {
        return self.layers_[self.checked_layer_offset(layer_idx)];
    }

    /// Non-owning replay view over the active layers, in build order.
    auto replay_view() const -> MPGraphView { return {layers_, active_begin_index(), layers(), false}; }

    auto is_schrodinger() const -> bool { return schrodinger_; }

    /// A normally-built layer stores no cosine set, so the companion cosine-index count cannot come from
    /// the graph: only the operator's inverted index can supply it.
    auto total_cycles() const -> size_t;
    auto storage_memory_usage() const -> GraphMemoryBreakdown;
};
} // namespace monoprop
