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

// A graph layer is replayed in one of two ways, distinguished by whether it carries a stored cosine
// list (pruned_cos_):
//   RECOMPUTE (pruned_cos_ == nullopt) — cosine recomputed from the generator's inverted-index columns
//                                        at replay (stores nothing); the main build path emits these.
//   PRUNED    (pruned_cos_ has value)  — cosine pre-filtered to a backward-reachable subset, stored
//                                        explicitly (an EMPTY stored list is still PRUNED — replay it
//                                        as nothing, do NOT recompute the full set).
// All layers share an immutable LayerCore core (cross-rank, exchange layouts, generator words,
// scaled_count). The pared graph reuses the source cores (shared_ptr), adding only the pruned cos.

/// @brief Read-only view over an immutable LayerCore plus an optional pruned-cosine word list.
///
/// Flag-free window used to replay a layer. Cross-rank data is ALWAYS read verbatim from the core (the
/// assemble_partners layout is never masked at replay, so there is no logical→stored remapping).
/// cos_data() is valid ONLY for a pruned layer (pruned_cos_ != nullptr); recompute layers store no cosine and
/// rebuild it from the inverted index.
struct LayerTraversal final {
    explicit LayerTraversal(const LayerCore &core, const CosMask *pruned_cos = nullptr)
        : core_(&core),
          pruned_cos_(pruned_cos) {}

    // cos_data() is valid ONLY for pruned layers (pruned_cos_ != nullptr). Fold layers recompute cos
    // from the inverted index fold and never call cos_data(); num_cos_inds()/cos_span_count() report 0 there.
    auto cos_data() const -> const CosMask & { return *pruned_cos_; }
    auto num_cos_inds() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->total_count : 0; }
    auto cos_span_count() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->span_count() : 0; }

    // Per-layer recompute metadata, read straight off the underlying LayerCore core.
    auto scaled_count() const -> uint64_t { return core_->scaled_count; }
    auto generator_words() const -> const std::vector<uint64_t> & { return core_->generator_words; }

    auto cross_rank_rank_count() const -> size_t { return core_->cross_rank.rank_count(); }

    auto cross_rank_sin_send_size(size_t rank) const -> size_t { return core_->cross_rank.sin_send_size(rank); }
    auto cross_rank_sin_recv_size(size_t rank) const -> size_t { return core_->cross_rank.sin_recv_size(rank); }

    // O(1) random access into the verbatim self/cross-rank D list. Used by the paired self-slot
    // derivative to fetch d[k] and d[k+P].
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
    auto derivative_exchange_layout() const -> const LayerExchangeLayout & { return core_->derivative_exchange_layout; }

    auto param_index() const -> size_t { return core_->param_index; }
    auto gen_coeff() const -> double { return core_->gen_coeff; }
    auto gate_index() const -> size_t { return core_->gate_index; }

private:
    const LayerCore *core_;
    const CosMask *pruned_cos_;
};

/// @brief Owning graph layer: a shared immutable LayerCore, plus an owned cosine list for pruned layers.
///
/// Pared graphs share source cores via shared_ptr and add only the pruned cosine list. Read-only
/// accessors delegate to a cheap LayerTraversal so replay logic lives in one place.
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

    // These accessors delegate to traversal(); the returned references point into the owned LayerCore
    // (not the temporary traversal), so they stay valid. num_cos_inds()/cos_span_count() count the
    // stored pruned cos (0 for recompute layers) and exist for the diagnostic formatters.
    // Ownership-specific queries LayerTraversal does not expose stay defined below.
    auto num_cos_inds() const -> size_t { return traversal().num_cos_inds(); }
    auto cos_span_count() const -> size_t { return traversal().cos_span_count(); }
    auto scaled_count() const -> uint64_t { return traversal().scaled_count(); }
    auto generator_words() const -> const std::vector<uint64_t> & { return traversal().generator_words(); }

    auto cross_rank_rank_count() const -> size_t { return traversal().cross_rank_rank_count(); }
    auto cross_rank_sin_send_size(size_t rank) const -> size_t { return traversal().cross_rank_sin_send_size(rank); }
    auto cross_rank_sin_recv_size(size_t rank) const -> size_t { return traversal().cross_rank_sin_recv_size(rank); }
    auto cross_rank_in_count(size_t rank) const -> size_t { return core().cross_rank.in_count(rank); }

    template <typename Func>
    auto for_each_cross_rank_sin_send_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        traversal().for_each_cross_rank_sin_send_range(rank, begin, end, std::forward<Func>(func));
    }

    template <typename Func>
    auto for_each_cross_rank_sin_recv_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        traversal().for_each_cross_rank_sin_recv_range(rank, begin, end, std::forward<Func>(func));
    }

    auto evolution_exchange_layout() const -> const LayerExchangeLayout & {
        return traversal().evolution_exchange_layout();
    }

    // Gate information owned by this layer (see LayerCore). Set on the mutable LayerCore at
    // build time (before it is frozen into the shared const core); read by evaluation.
    auto param_index() const -> size_t { return traversal().param_index(); }
    auto gen_coeff() const -> double { return traversal().gen_coeff(); }
    auto gate_index() const -> size_t { return traversal().gate_index(); }

    auto empty() const -> bool {
        if (num_cos_inds() != 0) {
            return false;
        }
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            if (cross_rank_sin_send_size(rank) != 0 || cross_rank_sin_recv_size(rank) != 0) {
                return false;
            }
        }
        return true;
    }

    // Number of rotations (Givens cycles) in this layer = sum of per-rank in-counts. Each rotation
    // contributes exactly one in-entry (its target), so this counts rotations once; sin_recv_size would
    // count in+out = 2 per self-rank rotation (the historical over-count fixed here).
    auto total_cycles() const -> size_t {
        size_t count = 0;
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            count += cross_rank_in_count(rank);
        }
        return count;
    }

    // Total rotation endpoints (in+out) across ranks. Every endpoint is also in cos_data (sources are
    // anticommuting; inserted targets are added in finish()), so cosine-ONLY indices =
    // num_cos_inds() - total_rotation_endpoints(). Used by graph_size() reporting.
    auto total_rotation_endpoints() const -> size_t {
        size_t count = 0;
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            count += cross_rank_sin_recv_size(rank);
        }
        return count;
    }

private:
    std::shared_ptr<const LayerCore> core_;
    std::optional<CosMask> pruned_cos_; // nullopt == fold layer (recompute); value == pruned
};

} // namespace monoprop
