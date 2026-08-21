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
// Cores are shared and immutable: they hold no eval-time cache, so no const handle mutates one.

// Cross-rank data is always read verbatim; only the cosine set is ever filtered.
struct LayerTraversal final {
    explicit LayerTraversal(const LayerCore &core, const CosMask *pruned_cos = nullptr)
        : core_(&core),
          pruned_cos_(pruned_cos) {}

    // Reports 0 for recompute layers (no stored cosine); check has_stored_cos() first, or a normally-built
    // graph — every layer of which recomputes — looks like it has no cosine indices at all.
    auto num_cos_inds() const -> size_t { return pruned_cos_ != nullptr ? pruned_cos_->total_count : 0; }

    auto has_stored_cos() const -> bool { return pruned_cos_ != nullptr; }

    // The stored set itself, for readers that need its indices and not just the count; nullptr on a
    // recompute layer.
    auto stored_cos() const -> const CosMask * { return pruned_cos_; }

    auto scaled_count() const -> uint64_t { return core_->scaled_count; }
    auto generator_words() const -> const std::vector<uint64_t> & { return core_->generator_words; }

    auto cross_rank_rank_count() const -> size_t { return core_->cross_rank.rank_count(); }

    auto cross_rank_sin_send_size(size_t rank) const -> size_t { return core_->cross_rank.sin_send_size(rank); }
    auto cross_rank_sin_recv_size(size_t rank) const -> size_t { return core_->cross_rank.sin_recv_size(rank); }
    auto cross_rank_in_count(size_t rank) const -> size_t { return core_->cross_rank.in_count(rank); }

    // O(1); the self slot is read per rotation pair in the innermost gradient loop.
    auto cross_rank_self_slot() const -> detail::CrossRankSlotView {
        return detail::cross_rank_self_slot(core_->cross_rank);
    }

    // Every slot carrying traffic, ascending, each with its offset. func(slot_id, view), or
    // func(occupied_pos, slot_id, view) to index a per-slot array by occupied position instead of P.
    //
    // This is what a partner sweep should use. The old shape -- loop 0..P, ask each slot its size,
    // `continue` on zero -- walked the whole world to find the part of it that had anything in it.
    template <typename Func>
    auto for_each_occupied_slot(Func &&func) const -> void {
        detail::for_each_occupied_slot(core_->cross_rank, std::forward<Func>(func));
    }

    // The size an array indexed by occupied position needs; the sweep above visits exactly this many.
    auto occupied_slot_count() const -> size_t { return detail::cross_rank_occupied_slots(core_->cross_rank); }

    // The slot is resolved ONCE, outside the loop: the lookup it costs is indexed by the flat world P,
    // so doing it per endpoint made per-term work out of what is per-slot work.
    template <typename Func>
    auto for_each_cross_rank_sin_send_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        const auto slot = detail::cross_rank_slot(core_->cross_rank, rank);
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx, detail::slot_sin_send_index(slot, idx));
        }
    }

    template <typename Func>
    auto for_each_cross_rank_sin_recv_range(size_t rank, size_t begin, size_t end, Func &&func) const -> void {
        const auto slot = detail::cross_rank_slot(core_->cross_rank, rank);
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx, detail::slot_sin_recv_index(slot, idx), detail::slot_sin_recv_phase(slot, idx));
        }
    }

    // The exchange layout -- both sides of it -- is derived at the call site from these and
    // nothing else is stored: see detail::derive_exchange_layout and Evolution.cpp.
    auto cross_rank() const -> const PackedCrossRankStorage & { return core_->cross_rank; }

    auto param_index() const -> size_t { return core_->param_index; }
    auto gen_coeff() const -> double { return core_->gen_coeff; }
    auto gate_index() const -> size_t { return core_->gate_index; }

    // Rotations (Givens cycles) = sum of per-rank in-counts (one in-entry per rotation). sin_recv_size
    // would double-count self-rank rotations (in+out). Empty slots contribute nothing, so summing over
    // the occupied ones is the same total the full sweep gave.
    auto total_cycles() const -> size_t {
        size_t count = 0;
        for_each_occupied_slot([&count](size_t, const detail::CrossRankSlotView &slot) { count += slot.in_count; });
        return count;
    }

    // Endpoints are counted in+out across ranks. Every endpoint is also in cos_data, so cosine-only
    // indices = num_cos_inds() - total_rotation_endpoints().
    auto total_rotation_endpoints() const -> size_t {
        size_t count = 0;
        for_each_occupied_slot(
            [&count](size_t, const detail::CrossRankSlotView &slot) { count += slot.sin_send_count; });
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
