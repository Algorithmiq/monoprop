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

// A layer replays one of two ways, by whether it carries a stored cosine list (pruned_cos_):
//   recompute (nullopt)   — cosine rebuilt from the generator's inverted-index columns at replay.
//   pruned    (has value) — cosine pre-filtered to a backward-reachable subset, stored explicitly; an
//                           empty stored list is still pruned (replay as nothing, do not recompute).
// Cores are shared and immutable in value only: their eval-time caches (recv_cache, the lazy derivative
// layout) are filled through const handles, so evaluating two aliasing propagators concurrently is a race.

// Read-only view over an immutable LayerCore plus an optional pruned-cosine word list. Cross-rank data is
// always read verbatim (no logical→stored remap).
struct LayerTraversal final {
    explicit LayerTraversal(const LayerCore &core, const CosMask *pruned_cos = nullptr)
        : core_(&core),
          pruned_cos_(pruned_cos) {}

    // Reports 0 for recompute layers (no stored cosine); check has_stored_cos() first, or a normally-built
    // graph — every layer of which recomputes — looks like it has no cosine indices at all.
    auto num_cos_inds() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->total_count : 0; }

    auto has_stored_cos() const -> bool { return pruned_cos_ != nullptr; }

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

struct Layer final {
    Layer() : core_(std::make_shared<LayerCore>()) {}

    explicit Layer(std::shared_ptr<const LayerCore> core) : core_(std::move(core)) {}
    Layer(std::shared_ptr<const LayerCore> core, CosMask pruned_cos)
        : core_(std::move(core)),
          pruned_cos_(std::move(pruned_cos)) {}

    auto core() const -> const LayerCore & { return *core_; }
    auto shared_core() const -> std::shared_ptr<const LayerCore> { return core_; }
    auto pruned_cos() const -> const CosMask * { return pruned_cos_ ? &*pruned_cos_ : nullptr; }

    auto traversal() const -> LayerTraversal { return LayerTraversal(core(), pruned_cos()); }

private:
    std::shared_ptr<const LayerCore> core_;
    std::optional<CosMask> pruned_cos_;
};

} // namespace monoprop
