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
#include <utility>
#include <variant>
#include <vector>

#include "monoprop/MPGraphEncoding.h"

namespace monoprop {

// A graph layer is one of two kinds, replayed differently:
//   FoldLayer   — cosine recomputed from the generator/sidecar fold at replay (stores nothing).
//   PrunedLayer — cosine pre-filtered to a backward-reachable subset, stored explicitly.
// Both share an immutable LayerCore core (cross-rank, exchange layouts, generator words,
// cos_count). The pared graph reuses the source cores (shared_ptr), adding only the pruned cos.
struct FoldLayer final {
    std::shared_ptr<const LayerCore> core;
};
struct PrunedLayer final {
    std::shared_ptr<const LayerCore> core;
    CosineWordList cos;
};
using LayerKind = std::variant<FoldLayer, PrunedLayer>;

// Thin, flag-free view over an immutable LayerCore core plus an optional pointer to a pruned
// cosine word list. Cross-rank is ALWAYS read verbatim from the core — at replay the assemble_partners
// layout is never masked, so there is no logical→stored position remapping anywhere. cos_data() is
// valid ONLY when has_pruned_cos() (a pruned/filtered layer); fold layers recompute their cosine set
// from the sidecar fold and store nothing.
struct LayerTraversal final {
    explicit LayerTraversal(const LayerCore &core, const CosineWordList *pruned_cos = nullptr)
        : core_(&core),
          pruned_cos_(pruned_cos) {}

    auto has_pruned_cos() const -> bool { return pruned_cos_ != nullptr; }

    // cos_data() is valid ONLY for pruned layers (pruned_cos_ != nullptr). Fold layers recompute cos
    // from the sidecar fold and never call cos_data(); num_cos_inds()/cos_span_count() report 0 there.
    auto cos_data() const -> const CosineWordList & { return *pruned_cos_; }
    auto num_cos_inds() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->total_count : 0; }
    auto cos_span_count() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->span_count() : 0; }

    // Per-layer recompute metadata, read straight off the underlying LayerCore core.
    auto cos_count() const -> uint64_t { return core_->cos_count; }
    auto generator_words() const -> const std::vector<uint64_t> & { return core_->generator_words; }

    auto cross_rank_rank_count() const -> size_t { return core_->cross_rank.rank_count(); }

    auto cross_rank_b_size(size_t rank) const -> size_t { return core_->cross_rank.b_size(rank); }
    auto cross_rank_d_size(size_t rank) const -> size_t { return core_->cross_rank.d_size(rank); }

    // Cross-rank is always stored verbatim at replay (no masked/pared filtering of the cross-rank
    // lists), so the assemble_partners layout is intact (d = [{out}]++[{in}], P==Q) and d-entry k
    // pairs with k+P as the two endpoints of one Givens rotation — required by the snapshot-free
    // self-slot path. Always true with the thin traversal.
    auto cross_rank_unmasked() const -> bool { return true; }

    // O(1) random access into the verbatim self/cross-rank D list. Used by the paired self-slot
    // derivative to fetch d[k] and d[k+P].
    auto cross_rank_d_index_at(size_t rank, size_t idx) const -> size_t {
        return detail::cross_rank_d_index(core_->cross_rank, rank, idx);
    }
    auto cross_rank_d_phase_at(size_t rank, size_t idx) const -> int {
        return detail::cross_rank_d_phase(core_->cross_rank, rank, idx);
    }

    template <typename Func>
    auto for_each_cross_rank_b_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx, detail::cross_rank_b_index(core_->cross_rank, rank, idx));
        }
    }

    template <typename Func>
    auto for_each_cross_rank_d_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx,
                 detail::cross_rank_d_index(core_->cross_rank, rank, idx),
                 detail::cross_rank_d_phase(core_->cross_rank, rank, idx));
        }
    }

    auto evolution_exchange_layout() const -> const LayerExchangeLayout & { return core_->evolution_exchange_layout; }
    auto derivative_exchange_layout() const -> const LayerExchangeLayout & {
        return core_->derivative_exchange_layout;
    }

private:
    const LayerCore *core_;
    const CosineWordList *pruned_cos_;
};

struct Layer final {
    Layer() : kind(FoldLayer{std::make_shared<LayerCore>()}) {}

    explicit Layer(std::shared_ptr<const LayerCore> core) : kind(FoldLayer{std::move(core)}) {}
    explicit Layer(LayerKind k) : kind(std::move(k)) {}

    auto core() const -> const LayerCore & {
        return *std::visit([](const auto &l) -> const LayerCore * { return l.core.get(); }, kind);
    }
    auto shared_core() const -> std::shared_ptr<const LayerCore> {
        return std::visit([](const auto &l) { return l.core; }, kind);
    }
    auto pruned_cos() const -> const CosineWordList * {
        if (const auto *p = std::get_if<PrunedLayer>(&kind)) {
            return &p->cos;
        }
        return nullptr;
    }

    auto traversal() const -> LayerTraversal { return LayerTraversal(core(), pruned_cos()); }

    // num_cos_inds()/cos_span_count() count the explicitly-stored pruned cos (0 for fold layers,
    // which recompute cos from the sidecar fold). They survive for diagnostic formatters
    // (Layer formatter, graph_size reporting).
    auto num_cos_inds() const -> size_t {
        const auto *cos = pruned_cos();
        return cos != nullptr ? cos->total_count : 0;
    }
    auto cos_span_count() const -> size_t {
        const auto *cos = pruned_cos();
        return cos != nullptr ? cos->span_count() : 0;
    }

    // Per-layer recompute metadata, carried in the layer's shared LayerCore core.
    auto cos_count() const -> uint64_t { return core().cos_count; }
    auto generator_words() const -> const std::vector<uint64_t> & { return core().generator_words; }

    auto cross_rank_rank_count() const -> size_t { return core().cross_rank.rank_count(); }
    auto cross_rank_b_size(size_t rank) const -> size_t { return core().cross_rank.b_size(rank); }
    auto cross_rank_d_size(size_t rank) const -> size_t { return core().cross_rank.d_size(rank); }
    auto cross_rank_in_count(size_t rank) const -> size_t { return core().cross_rank.in_count(rank); }

    template <typename Func>
    auto for_each_cross_rank_b_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx, detail::cross_rank_b_index(core().cross_rank, rank, idx));
        }
    }

    template <typename Func>
    auto for_each_cross_rank_d_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx,
                 detail::cross_rank_d_index(core().cross_rank, rank, idx),
                 detail::cross_rank_d_phase(core().cross_rank, rank, idx));
        }
    }

    auto evolution_exchange_layout() const -> const LayerExchangeLayout & { return core().evolution_exchange_layout; }

    auto empty() const -> bool {
        if (num_cos_inds() != 0) {
            return false;
        }
        for (size_t rank = 0; rank < cross_rank_rank_count(); ++rank) {
            if (cross_rank_b_size(rank) != 0 || cross_rank_d_size(rank) != 0) {
                return false;
            }
        }
        return true;
    }

    // Number of rotations (Givens cycles) in this layer = sum of per-rank in-counts. Each rotation
    // contributes exactly one in-entry (its target), so this counts rotations once; d_size would
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
            count += cross_rank_d_size(rank);
        }
        return count;
    }

    LayerKind kind;
};

} // namespace monoprop
