#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <format>
#include "monoprop/detail/print_compat.h"

#include "monoprop/detail/graph/MPGraphLayers.h"

namespace monoprop {

// Per-rank graph memory accounting. `cos_data_bytes` now counts the cosine word lists stored on
// PrunedLayer entries (FoldLayer recomputes its cos from the sidecar fold and stores nothing).
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

        // reverse_ flips traversal order (Schrödinger replays newest-first) within the [base_, base_+count_) window.
        return base_ + (reverse_ ? count_ - 1 - layer_idx : layer_idx);
    }

    const std::vector<Layer> *layers_ = nullptr;
    size_t base_ = 0;
    size_t count_ = 0;
    bool reverse_ = false;
};

} // namespace monoprop
