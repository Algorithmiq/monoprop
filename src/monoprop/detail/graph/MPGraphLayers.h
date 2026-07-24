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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "monoprop/MPGraphEncoding.h"

namespace monoprop {

// A layer replays one of two ways by whether it carries a stored cosine list (pruned_cos_):
//   RECOMPUTE (nullopt)   — cosine recomputed from the generator's inverted-index columns at replay.
//   PRUNED    (has value) — cosine pre-filtered to a backward-reachable subset, stored explicitly
//                           (an EMPTY stored list is still PRUNED — replay as nothing, do NOT recompute).
// All layers share an immutable LayerCore; the pared graph reuses source cores (shared_ptr) + pruned cos.
// "Immutable" means immutable IN VALUE: a core carries two eval-time caches filled through const handles
// (LayerExchangeLayout::recv_cache and the lazy derivative layout), so materializing them on a core two
// threads share is a data race. No shipped path does that — shards own their propagators and the Python
// bindings hold the GIL — but a C++ caller evaluating two aliasing propagators concurrently must not.

/// @brief Read-only view over an immutable LayerCore plus an optional pruned-cosine word list.
/// Cross-rank data is always read verbatim (no logical→stored remap). num_cos_inds() reports the stored
/// count for pruned layers; recompute layers report 0 and rebuild cosine from the inverted index.
struct LayerTraversal final {
    explicit LayerTraversal(const LayerCore &core, const CosMask *pruned_cos = nullptr)
        : core_(&core),
          pruned_cos_(pruned_cos) {}

    // num_cos_inds() reports 0 for recompute layers (no stored cosine); pruned layers report the stored count.
    auto num_cos_inds() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->total_count : 0; }

    // Per-layer recompute metadata, read straight off the underlying LayerCore core.
    auto scaled_count() const -> uint64_t { return core_->scaled_count; }
    auto generator_words() const -> const std::vector<uint64_t> & { return core_->generator_words; }

    auto cross_rank_rank_count() const -> size_t { return core_->cross_rank.rank_count(); }

    auto cross_rank_sin_send_size(size_t rank) const -> size_t { return core_->cross_rank.sin_send_size(rank); }
    auto cross_rank_sin_recv_size(size_t rank) const -> size_t { return core_->cross_rank.sin_recv_size(rank); }
    auto cross_rank_in_count(size_t rank) const -> size_t { return core_->cross_rank.in_count(rank); }

    // O(1) random access into the verbatim D list (paired self-slot derivative fetches d[k], d[k+P]).
    auto cross_rank_sin_recv_index_at(size_t rank, size_t idx) const -> size_t {
        return detail::cross_rank_sin_recv_index(core_->cross_rank, rank, idx);
    }
    auto cross_rank_sin_recv_phase_at(size_t rank, size_t idx) const -> int {
        return detail::cross_rank_sin_recv_phase(core_->cross_rank, rank, idx);
    }

    template <typename Func>
    auto for_each_cross_rank_sin_send_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx, detail::cross_rank_sin_send_index(core_->cross_rank, rank, idx));
        }
    }

    template <typename Func>
    auto for_each_cross_rank_sin_recv_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx,
                 detail::cross_rank_sin_recv_index(core_->cross_rank, rank, idx),
                 detail::cross_rank_sin_recv_phase(core_->cross_rank, rank, idx));
        }
    }

    auto evolution_exchange_layout() const -> const LayerExchangeLayout & { return core_->evolution_exchange_layout; }
    auto derivative_exchange_layout() const -> const LayerExchangeLayout & {
        return core_->derivative_exchange_layout();
    }

    auto param_index() const -> size_t { return core_->param_index; }
    auto gen_coeff() const -> double { return core_->gen_coeff; }
    auto gate_index() const -> size_t { return core_->gate_index; }

    // Rotations (Givens cycles) = sum of per-rank in-counts (one in-entry per rotation). sin_recv_size
    // would double-count self-rank rotations (in+out).
    auto total_cycles() const -> size_t {
        size_t count = 0;
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            count += cross_rank_in_count(rank);
        }
        return count;
    }

    // Total rotation endpoints (in+out) across ranks. Every endpoint is also in cos_data, so cosine-only
    // indices = num_cos_inds() - total_rotation_endpoints(). Used by graph_size() reporting.
    auto total_rotation_endpoints() const -> size_t {
        size_t count = 0;
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            count += cross_rank_sin_recv_size(rank);
        }
        return count;
    }

private:
    const LayerCore *core_;
    const CosMask *pruned_cos_;
};

/// @brief Owning graph layer: a shared immutable LayerCore, plus an owned cosine list for pruned layers.
/// All read-only access goes through traversal(); this type only adds ownership over the core + pruned cos.
struct Layer final {
    Layer() : core_(std::make_shared<LayerCore>()) {}

    explicit Layer(std::shared_ptr<const LayerCore> core) : core_(std::move(core)) {}
    // Pruned layer: carries an explicitly-stored (possibly empty) filtered cosine list.
    Layer(std::shared_ptr<const LayerCore> core, CosMask pruned_cos)
        : core_(std::move(core)),
          pruned_cos_(std::move(pruned_cos)) {}

    auto core() const -> const LayerCore & { return *core_; }
    auto shared_core() const -> std::shared_ptr<const LayerCore> { return core_; }
    auto pruned_cos() const -> const CosMask * { return pruned_cos_ ? &*pruned_cos_ : nullptr; }

    auto traversal() const -> LayerTraversal { return LayerTraversal(core(), pruned_cos()); }

private:
    std::shared_ptr<const LayerCore> core_;
    std::optional<CosMask> pruned_cos_; // nullopt == fold layer (recompute); value == pruned
};

} // namespace monoprop
