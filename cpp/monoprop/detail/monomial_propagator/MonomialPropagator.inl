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

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/LayerBuilder.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/Construction.h"
#include "monoprop/detail/sharded/Evaluation.h"
#include "monoprop/detail/sharded/State.h"

namespace monoprop {

// The requested operation does not agree with the graph this propagator currently holds -- either it
// requires no stored graph, or its parameter_mapping matches neither the stored layer nor gate count.
// The caller recovers by contracting or rebuilding the graph, not by fixing an isolated argument.
class GraphStateConflict : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The (basis, cutoff_type, basis_change) triple is inconsistent: a Pauli basis with a Length cutoff or
// a basis change, or a basis-change table that is not 2*logical_num_modes rows.
class CutoffConfigError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// A coefficient-informed build_graph() was given fewer parameter values than replaying the stored graph
// as a seed needs.
class SeedParametersTooShort : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

template <size_t NumModes>
MonomialPropagator<NumModes>::~MonomialPropagator() = default;

template <size_t NumModes>
auto MonomialPropagator<NumModes>::checked_source_(const MonomialPropagator &other) -> const MonomialPropagator & {
    other.require_valid_();
    return other;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::require_valid_() const -> void {
    if (invalid_) {
        throw InvalidPropagatorError(
            "This propagator is no longer usable: an earlier operation failed after it had started changing the "
            "operator, graph or cached state, so that state is incomplete. Construct a new propagator (or use a "
            "copy made before the failure); the failed object can only be destroyed.");
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::enter_operation_(bool uses_mpi) const -> void {
    // At every P and T: the caller becomes the primary of the operation's team, which alone makes its MPI calls.
    if (uses_mpi) {
        mpi::require_initializing_thread();
    }
    require_valid_();
}

template <size_t NumModes>
template <typename Body>
auto MonomialPropagator<NumModes>::run_operation_(bool uses_mpi, Body &&body) -> decltype(auto) {
    enter_operation_(uses_mpi);
    bool mutation_started = false;
    try {
        return std::forward<Body>(body)(mutation_started);
    }
    catch (...) {
        if (!mutation_started) {
            throw;
        }
        mutation_failed_(std::current_exception());
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::validate_generators_(const std::vector<VecZ> &majoranas) const -> void {
    for (const auto &gate : majoranas) {
        (void)indices_to_bitset_checked<NumModes>(gate, 2 * logical_num_modes_);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::validate_cutoff_config_(CutoffType cutoff_type,
                                                           const std::optional<std::vector<VecZ>> &basis_change) const
    -> void {
    with_algebra<NumModes>(basis_, [&]<typename A>() {
        if (A::requires_support_cutoff && cutoff_type != CutoffType::Support) {
            throw CutoffConfigError("Pauli basis requires cutoff_type == Support "
                                    "(Length has no Pauli-weight meaning under the Pauli encoding).");
        }
        if (!A::allows_basis_change && basis_change.has_value()) {
            throw CutoffConfigError("Pauli basis does not accept a basis_change "
                                    "(the encoding is already the Jordan-Wigner image).");
        }
    });
    // regenerate_cutoff_fn_ indexes rows [0, 2*logical_num_modes) unconditionally, so a short
    // basis_change is an out-of-bounds read.
    if (basis_change.has_value() && basis_change->size() != 2 * logical_num_modes_) {
        throw CutoffConfigError(std::format("basis_change must have exactly 2*logical_num_modes ({}) rows; got {}.",
                                            2 * logical_num_modes_,
                                            basis_change->size()));
    }
    // regenerate_cutoff_fn_ would reject these too, but only after an update started changing settings.
    if (basis_change.has_value()) {
        for (const auto &row : *basis_change) {
            (void)indices_to_bitset_checked<NumModes>(row, 2 * logical_num_modes_);
        }
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::regenerate_cutoff_fn_() -> void {
    if (basis_change_.has_value()) {
        MonomialList<NumModes> basis;
        basis.reserve(2 * logical_num_modes_);
        for (size_t i = 0; i < 2 * logical_num_modes_; ++i) {
            basis.push_back(indices_to_bitset_checked<NumModes>(basis_change_.value()[i], 2 * logical_num_modes_));
        }
        cutoff_fn_ = detail::cutoff_function_basis_change<NumModes>(cutoff_type_, cutoff_, basis, logical_num_modes_);
    }
    else {
        cutoff_fn_ = detail::cutoff_function<NumModes>(cutoff_type_, cutoff_, logical_num_modes_);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_data_of_(const MPGraph &graph,
                                                  const detail::MPOperator<NumModes> &op,
                                                  Basis basis) -> std::vector<LayerData> {
    std::vector<LayerData> layers;
    const auto num_layers = graph.layers();
    layers.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        const auto traversal = graph.get_layer_traversal(i);
        const size_t rank_count = traversal.cross_rank_rank_count();

        // Always empty: local cycles are folded into cross_rank[my_rank].
        std::vector<LocalCycleData> local_cyc_data;

        // The exported shape stays dense (callers index by rank), but it is filled by scattering the
        // occupied slots rather than by asking every possible slot how much it holds.
        std::vector<CrossRankData> b_data(rank_count), d_data(rank_count);
        traversal.for_each_occupied_slot([&](size_t rank, const detail::CrossRankSlotView &slot) {
            const size_t count = slot.sin_send_count;
            VecZ sin_send_indices(count);
            VecI b_phases(count, 0);
            VecZ d_indices(count);
            VecI sin_recv_phases(count);
            for (size_t k = 0; k < count; ++k) {
                sin_send_indices[k] = detail::slot_sin_send_index(slot, k);
                d_indices[k] = detail::slot_sin_recv_index(slot, k);
                sin_recv_phases[k] = detail::slot_sin_recv_phase(slot, k);
            }
            b_data[rank] = CrossRankData{std::move(sin_send_indices), std::move(b_phases)};
            d_data[rank] = CrossRankData{std::move(d_indices), std::move(sin_recv_phases)};
        });
        // Same two-way read as cos_index_count_(): a pared layer's stored set is authoritative, and
        // recomputing the fold over it would report the indices the pare removed.
        VecZ cos_inds;
        if (const CosMask *stored = traversal.stored_cos(); stored != nullptr) {
            cos_inds.reserve(stored->total_count);
            for (const auto &[base, bits] : stored->blocks) {
                detail::for_each_cos_index(base, bits, [&](size_t idx) { cos_inds.push_back(idx); });
            }
        }
        else if (const auto &gw = traversal.generator_words(); !gw.empty()) {
            const auto gen = detail::generator_from_words<NumModes>(gw);
            auto p = detail::make_fold_cache<NumModes>(op.inverted_index(), gen, traversal.scaled_count(), basis);
            cos_inds = detail::fold_to_indices<NumModes>(p);
        }
        layers.emplace_back(std::move(cos_inds), std::move(local_cyc_data), std::move(b_data), std::move(d_data));
    }
    return layers;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::cos_index_count_of_(const MPGraph &graph,
                                                       const detail::MPOperator<NumModes> &op,
                                                       Basis basis) -> size_t {
    // Only a pared layer stores a cosine set; otherwise recompute the fold here. Cosine-only = cos-scaled
    // minus the rotation endpoints, saturating at 0.
    size_t total = 0;
    const auto num_layers = graph.layers();
    for (size_t i = 0; i < num_layers; ++i) {
        const auto traversal = graph.get_layer_traversal(i);
        size_t cos_total = 0;
        if (traversal.has_stored_cos()) {
            cos_total = traversal.num_cos_inds();
        }
        else if (const auto &gw = traversal.generator_words(); !gw.empty()) {
            const auto gen = detail::generator_from_words<NumModes>(gw);
            const auto fold =
                detail::make_fold_cache<NumModes>(op.inverted_index(), gen, traversal.scaled_count(), basis);
            cos_total = detail::fold_popcount<NumModes>(fold);
        }
        const size_t endpoints = traversal.total_rotation_endpoints();
        total += (cos_total > endpoints) ? (cos_total - endpoints) : 0;
    }
    return total;
}

/// Per-layer cosine callbacks replaying `graph` against `inverted_index`; each closure captures `options`
/// for its kernels. The closures keep raw pointers into the index, so they must not outlive it.
template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index,
                         const MPGraphView &graph,
                         Basis basis = Basis::Majorana,
                         detail::parallel::Options options = {}) -> detail::CosCallbacks;

template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index,
                         const MPGraphView &graph,
                         Basis basis,
                         detail::parallel::Options options) -> detail::CosCallbacks {
    return detail::make_cos_callbacks<NumModes>(inverted_index, graph, basis, options);
}

} // namespace monoprop

#include "monoprop/detail/monomial_propagator/ShardedPropagator.inl"
