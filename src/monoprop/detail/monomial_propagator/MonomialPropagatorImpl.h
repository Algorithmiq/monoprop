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
#include <functional>
#include <numeric>
#include <string_view>
#include <thread>
#include <utility>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/LayerBuilder.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/shard/ShardGroup.h" // needs the complete type

namespace monoprop {

template <size_t NumModes>
MonomialPropagator<NumModes>::MonomialPropagator(const OperatorDict &initial_operator,
                                                 unsigned int cutoff,
                                                 const VecZ &initial_state,
                                                 std::optional<unsigned int> schrodinger_cutoff,
                                                 mpi::Comm comm,
                                                 std::optional<double> lower_atol,
                                                 std::optional<double> upper_atol,
                                                 CutoffType cutoff_type,
                                                 std::optional<std::vector<VecZ>> basis_change,
                                                 size_t logical_num_modes,
                                                 Basis basis,
                                                 size_t shards)
    : schrodinger_{schrodinger_cutoff.has_value()},
      comm_{comm},
      mp_op_{},
      graph_(schrodinger_cutoff.has_value()),
      cutoff_{cutoff},
      lower_atol_{lower_atol},
      upper_atol_{upper_atol},
      logical_num_modes_{logical_num_modes},
      cutoff_type_{cutoff_type},
      basis_change_{basis_change},
      basis_{basis} {
    if (logical_num_modes_ == 0 || logical_num_modes_ > NumModes) {
        throw std::runtime_error(
            std::format("logical_num_modes ({}) must be in the range [1, {}].", logical_num_modes_, NumModes));
    }

    validate_cutoff_config_(cutoff_type_, basis_change_);

    mp_op_.basis = basis_;

    if (upper_atol.has_value() && lower_atol.has_value() && (upper_atol.value() < lower_atol.value())) {
        throw std::runtime_error(std::format("upper_atol ({}) must be greater than or equal to lower_atol ({}).",
                                             upper_atol.value(),
                                             lower_atol.value()));
    }

    // shards>1 makes this a FACADE owning S single-shard propagators, one hash partition each on its own
    // pinned master thread; its own mp_op_/graph_ stay empty.
    const size_t n_shards = resolve_shard_count_(shards, comm);
    // Every MPI rank must resolve the SAME shard count: the R ranks x S shards form one flat P = R*S
    // SPMD world, so a mismatch would deadlock at the first hybrid collective.
    if (comm.kind == mpi::Comm::Kind::Mpi && mpi::size(comm) > 1
        && mpi::allreduce_sum<size_t>(n_shards, comm) != n_shards * static_cast<size_t>(mpi::size(comm))) {
        throw std::runtime_error("Shard count differs across MPI ranks — every rank must resolve the same "
                                 "shards= / monoprop_SHARDS / monoprop_NUM_THREADS so R*S is a consistent world.");
    }
    if (n_shards > 1) {
        auto factory = [=](mpi::Comm shard_comm) {
            return std::make_unique<MonomialPropagator<NumModes>>(initial_operator,
                                                                  cutoff,
                                                                  initial_state,
                                                                  schrodinger_cutoff,
                                                                  shard_comm,
                                                                  lower_atol,
                                                                  upper_atol,
                                                                  cutoff_type,
                                                                  basis_change,
                                                                  logical_num_modes,
                                                                  basis,
                                                                  /*shards=*/1);
        };
        shard_group_ = std::make_unique<detail::shard::ShardGroup<NumModes>>(static_cast<int>(n_shards), factory, comm);
        return;
    }

    const size_t num_ranks = static_cast<size_t>(mpi::size(comm_));
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm_));
    MonomialList<NumModes> local_heisenberg_terms;

    double core_term = 0.0;
    for (const auto &[indices, coefficient] : initial_operator) {
        const auto majorana_bitset = indices_to_bitset_checked<NumModes>(indices, 2 * logical_num_modes_);
        const auto encoded_coeff = algebra_encode_coeff<NumModes>(basis_, coefficient, majorana_bitset);

        // Store the core term separately as it is orders of magnitude larger than the other terms
        if (indices.empty()) {
            core_term = encoded_coeff;
            continue;
        }
        if (my_rank == find_rank<NumModes>(majorana_bitset, num_ranks)) {
            mp_op_.init_op_map[majorana_bitset] = encoded_coeff;
            local_heisenberg_terms.push_back(majorana_bitset);
        }
    }

    auto sc = schrodinger_cutoff.value_or(cutoff + 2);
    sc = std::min(sc, static_cast<unsigned int>(2 * logical_num_modes_));
    auto op = schrodinger_ ? generate_paired_op<NumModes>(sc / 2 + sc % 2, logical_num_modes_) : local_heisenberg_terms;

    const size_t expected_local_terms = std::max<size_t>(1, op.size() / std::max<size_t>(1, num_ranks));
    // Must run BEFORE the store: packed_inline_width_() derives the packed-row width from cutoff_fn_.
    regenerate_cutoff_fn_();
    mp_op_.store = std::make_unique<detail::OperatorIndex<NumModes>>(packed_inline_width_());
    mp_op_.store->reserve(expected_local_terms);
    // Store replaced: drop the stale lazy inverted index so it rebuilds against the new store.
    mp_op_.inverted_index_.reset();

    size_t i = 0;
    // The initial monomials are DISTINCT, so emplace (insert-if-absent) is an assigning insert here.
    for (size_t r = 0; r < op.size(); ++r) {
        const auto &mono = materialize_row<NumModes>(op, r);
        if (my_rank == find_rank<NumModes>(mono, num_ranks)) {
            mp_op_.append_term(mono);
            mp_op_.store->emplace(mono, i++);
        }
    }

    mp_op_.initial_state = initial_state;
    core_term_ = core_term;

    initialize_operator_caches_();
}

template <size_t NumModes>
MonomialPropagator<NumModes>::~MonomialPropagator() = default;

template <size_t NumModes>
MonomialPropagator<NumModes>::MonomialPropagator(const MonomialPropagator &other)
    : schrodinger_(other.schrodinger_),
      comm_(other.comm_),
      cutoff_fn_(other.cutoff_fn_),
      mp_op_(other.mp_op_),
      graph_(other.graph_),
      matched_scratch_(other.matched_scratch_),
      cutoff_(other.cutoff_),
      lower_atol_(other.lower_atol_),
      upper_atol_(other.upper_atol_),
      core_term_(other.core_term_),
      logical_num_modes_(other.logical_num_modes_),
      cutoff_type_(other.cutoff_type_),
      basis_change_(other.basis_change_),
      basis_(other.basis_),
      shard_group_(other.shard_group_ ? std::make_unique<detail::shard::ShardGroup<NumModes>>(*other.shard_group_)
                                      : nullptr) {}

// Precedence: an explicit ctor `shards`>=1, else monoprop_SHARDS, else the auto policy below.
template <size_t NumModes>
auto MonomialPropagator<NumModes>::resolve_shard_count_(size_t requested, mpi::Comm comm) -> size_t {
    if (requested >= 1) {
        return requested;
    }
    // AUTO policy: one serial shard per physical core, capped by monoprop_NUM_THREADS. On a multi-rank
    // comm it engages ONLY when threads were explicitly requested, so a pure-MPI run is not oversubscribed.
    const auto compute_auto = [&]() -> size_t {
        const int ranks = mpi::size(comm);
        size_t cores = detail::shard::enumerate_physical_cores().size();
        if (cores == 0) {
            // Topology unreadable: use half the hardware threads so SMT siblings aren't counted as cores.
            cores = std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()) / 2);
        }
        const auto num_threads = config::get().num_threads;
        const size_t budget =
            num_threads.has_value() ? static_cast<size_t>(*num_threads) : (ranks == 1 ? cores : size_t{1});
        return std::max<size_t>(1, std::min(budget, cores));
    };
    if (const char *env = std::getenv("monoprop_SHARDS")) {
        const std::string_view v(env);
        if (v == "auto") {
            return compute_auto();
        }
        if (v == "off") {
            return 1;
        }
        char *end = nullptr;
        const long n = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && n >= 1) {
            return static_cast<size_t>(n);
        }
    }
    return compute_auto();
}

// Sharded read accessors: fan out over the quiescent shards from the facade thread.
template <size_t NumModes>
auto MonomialPropagator<NumModes>::sharded_size_() const -> size_t {
    size_t total = 0;
    for (int r = 0; r < shard_group_->shard_count(); ++r) {
        total += shard_group_->shard(r).size();
    }
    return total;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sharded_graph_size_() const -> std::pair<size_t, size_t> {
    size_t cos = 0;
    size_t cyc = 0;
    for (int r = 0; r < shard_group_->shard_count(); ++r) {
        const auto [c, y] = shard_group_->shard(r).graph_size();
        cos += c;
        cyc += y;
    }
    return {cos, cyc};
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sharded_graph_layers_() const -> size_t {
    // Graph structure is identical on every shard, so shard 0 is authoritative for structural queries.
    return shard_group_->shard(0).graph_layers();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sharded_core_term_() const -> double {
    // The core (identity) term is stored on every shard (not hash-partitioned), so any shard is full.
    return shard_group_->shard(0).core_term();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sharded_operator_memory_usage_() const
    -> detail::MPOperatorMemoryBreakdown<NumModes> {
    detail::MPOperatorMemoryBreakdown<NumModes> total;
    for (int r = 0; r < shard_group_->shard_count(); ++r) {
        total += shard_group_->shard(r).operator_memory_usage();
    }
    return total;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sharded_graph_memory_usage_() const -> GraphMemoryBreakdown {
    GraphMemoryBreakdown total;
    for (int r = 0; r < shard_group_->shard_count(); ++r) {
        total += shard_group_->shard(r).graph_memory_usage();
    }
    return total;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::for_each_shard_(const std::function<void(MonomialPropagator &)> &fn) -> void {
    shard_group_->run_on_all([&](int r) { fn(shard_group_->shard(r)); });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::packed_inline_width_() const -> size_t {
    constexpr size_t kMax = detail::OperatorIndex<NumModes>::kMaxInlinePositions;
    constexpr size_t kDefault = detail::OperatorIndex<NumModes>::kDefaultInlinePositions;
    if (schrodinger_) {
        return kDefault;
    }
    // The bound is already in physical slots (CutoffEvaluator::max_slot_bound), so nothing to scale.
    const auto bound = detail::CutoffEvaluator<NumModes>(cutoff_fn_).max_slot_bound();
    if (!bound) {
        return kDefault;
    }
    return std::min<size_t>(*bound, kMax);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::apply_initial_operator_(const OperatorDict &op_dict)
    -> std::pair<MonomialList<NumModes>, VecD> {
    if (shard_group_) {
        // The facade holds no local terms of its own, so the return is empty.
        shard_group_->run_on_all([&](int r) { shard_group_->shard(r).update_initial_operator(op_dict); });
        return {};
    }
    const size_t num_ranks = static_cast<size_t>(mpi::size(comm_));
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm_));

    OperatorDict new_op;
    for (const auto &[ind, coeff] : op_dict) {
        const auto mono = indices_to_bitset_checked<NumModes>(ind, 2 * logical_num_modes_);
        if (ind.empty()) { // Core term, store in all
            core_term_ = algebra_encode_coeff<NumModes>(basis_, coeff, mono);
            continue;
        }
        if (my_rank == find_rank<NumModes>(mono, num_ranks)) {
            const auto mono_indices = bitset_to_indices<NumModes>(mono);
            new_op[mono_indices] = coeff;
        }
    }

    return mp_op_.update_initial_operator(new_op, schrodinger_);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_data() const -> std::vector<LayerData> {
    std::vector<LayerData> layers;
    const auto num_layers = graph_.layers();
    layers.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        const auto traversal = graph_.get_layer_traversal(i);
        const size_t rank_count = traversal.cross_rank_rank_count();

        // Always empty: local cycles are folded into cross_rank[my_rank].
        std::vector<LocalCycleData> local_cyc_data;

        std::vector<CrossRankData> b_data, d_data;
        b_data.reserve(rank_count);
        d_data.reserve(rank_count);
        for (size_t rank = 0; rank < rank_count; ++rank) {
            VecZ sin_send_indices(traversal.cross_rank_sin_send_size(rank));
            VecI b_phases(traversal.cross_rank_sin_send_size(rank), 0);
            VecZ d_indices(traversal.cross_rank_sin_recv_size(rank));
            VecI sin_recv_phases(traversal.cross_rank_sin_recv_size(rank));

            traversal.for_each_cross_rank_sin_send_range(
                rank,
                0,
                traversal.cross_rank_sin_send_size(rank),
                [&](size_t logical_idx, size_t value_idx) { sin_send_indices[logical_idx] = value_idx; });
            traversal.for_each_cross_rank_sin_recv_range(rank,
                                                         0,
                                                         traversal.cross_rank_sin_recv_size(rank),
                                                         [&](size_t logical_idx, size_t value_idx, int phase) {
                                                             d_indices[logical_idx] = value_idx;
                                                             sin_recv_phases[logical_idx] = phase;
                                                         });

            b_data.emplace_back(std::move(sin_send_indices), std::move(b_phases));
            d_data.emplace_back(std::move(d_indices), std::move(sin_recv_phases));
        }
        // cos is not stored per-layer; recompute it from the inverted-index fold.
        VecZ cos_inds;
        const auto &gw = traversal.generator_words();
        if (!gw.empty()) {
            const auto gen = detail::generator_from_words<NumModes>(gw);
            auto p = detail::make_fold_cache<NumModes>(mp_op_.inverted_index(), gen, traversal.scaled_count(), basis_);
            cos_inds = detail::fold_to_indices<NumModes>(p);
        }
        layers.emplace_back(std::move(cos_inds), std::move(local_cyc_data), std::move(b_data), std::move(d_data));
    }
    return layers;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::cos_index_count_() const -> size_t {
    // A non-pared layer stores no cosine set, so recompute the fold here; only a pared layer has a
    // stored count. Cosine-ONLY = cos-scaled minus the rotation endpoints, saturating at 0.
    size_t total = 0;
    const auto num_layers = graph_.layers();
    for (size_t i = 0; i < num_layers; ++i) {
        const auto traversal = graph_.get_layer_traversal(i);
        size_t cos_total = 0;
        if (traversal.has_stored_cos()) {
            cos_total = traversal.num_cos_inds();
        }
        else if (const auto &gw = traversal.generator_words(); !gw.empty()) {
            const auto gen = detail::generator_from_words<NumModes>(gw);
            const auto fold =
                detail::make_fold_cache<NumModes>(mp_op_.inverted_index(), gen, traversal.scaled_count(), basis_);
            cos_total = detail::fold_popcount<NumModes>(fold);
        }
        const size_t endpoints = traversal.total_rotation_endpoints();
        total += (cos_total > endpoints) ? (cos_total - endpoints) : 0;
    }
    return total;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::validate_cutoff_config_(CutoffType cutoff_type,
                                                           const std::optional<std::vector<VecZ>> &basis_change) const
    -> void {
    with_algebra<NumModes>(basis_, [&]<class A>() {
        if (A::requires_support_cutoff && cutoff_type != CutoffType::Support) {
            throw std::invalid_argument("Pauli basis requires cutoff_type == Support "
                                        "(Length has no Pauli-weight meaning under the Pauli encoding).");
        }
        if (!A::allows_basis_change && basis_change.has_value()) {
            throw std::invalid_argument("Pauli basis does not accept a basis_change "
                                        "(the encoding is already the Jordan-Wigner image).");
        }
    });
    // regenerate_cutoff_fn_ indexes rows [0, 2*logical_num_modes) unconditionally, so a short
    // basis_change is an out-of-bounds read.
    if (basis_change.has_value() && basis_change->size() != 2 * logical_num_modes_) {
        throw std::invalid_argument(std::format("basis_change must have exactly 2*logical_num_modes ({}) rows; got {}.",
                                                2 * logical_num_modes_,
                                                basis_change->size()));
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
auto MonomialPropagator<NumModes>::initialize_operator_caches_() -> void {
    // Pre-warm the lazy operator/state/inverted-index caches, then trim the now-stable coeff vectors.
    (void)mp_op_.get_operator();
    // Heisenberg warms the SPARSE state only: a dense vector here would reinstate the 99.9%-zero array
    // the sparse form exists to avoid. Schrödinger's dense vector IS the live evolved vector.
    if (schrodinger_) {
        (void)mp_op_.dense_state();
    }
    else {
        (void)mp_op_.sparse_state();
    }
    (void)mp_op_.inverted_index();
    mp_op_.op_coeffs.shrink_to_fit();
    mp_op_.shrink_state_to_fit();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::current_picture_coeffs_() -> const VecD & {
    return schrodinger_ ? mp_op_.dense_state() : mp_op_.get_operator();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::extend_coeffs_from_current_picture_if_needed_(VecD &coeffs) -> void {
    if (coeffs.size() >= mp_op_.size()) {
        return;
    }

    const auto &current = current_picture_coeffs_();
    if (&coeffs == &current) {
        return;
    }

    if (coeffs.size() < current.size()) {
        coeffs.insert(coeffs.end(), current.begin() + static_cast<std::ptrdiff_t>(coeffs.size()), current.end());
    }
    coeffs.resize(mp_op_.size(), 0.0);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_build_graph_(const std::vector<VecZ> &majoranas,
                                                            const VecZ &parameter_mapping,
                                                            const VecD &gen_coeffs,
                                                            const VecZ &gate_indices,
                                                            int only_rotate_len_k) -> void {
    const auto majoranas_size = majoranas.size();
    run_gate_loop_(majoranas,
                   only_rotate_len_k,
                   [this, &parameter_mapping, &gen_coeffs, &gate_indices, majoranas_size](const VecZ &mono,
                                                                                          int rot_len,
                                                                                          size_t i) {
                       const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
                       propagate_one_(mono,
                                      rot_len,
                                      std::nullopt,
                                      std::nullopt,
                                      parameter_mapping[idx],
                                      gen_coeffs[idx],
                                      gate_indices[idx]);
                   });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_graph_with_coeffs_(const std::vector<VecZ> &majoranas,
                                                                  const VecZ &parameter_mapping,
                                                                  const VecD &gen_coeffs,
                                                                  const VecZ &gate_indices,
                                                                  const VecD &parameters,
                                                                  const VecD &operator_coeffs,
                                                                  int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    auto coeffs = operator_coeffs;
    const auto majoranas_size = majoranas.size();

    run_gate_loop_(majoranas,
                   only_rotate_len_k,
                   [this, &parameter_mapping, &gen_coeffs, &gate_indices, &mapped_params, &coeffs, majoranas_size](
                       const VecZ &mono,
                       int rot_len,
                       size_t i) {
                       const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
                       const auto [build_angle, apply_angle] = gate_angle_(mapped_params, i, majoranas_size);
                       // The cos word list is not persisted on the layer; the builder moves it out transiently.
                       auto cos = std::make_shared<CosMask>();
                       auto storage = build_evolve_result_(mono, rot_len, std::cref(coeffs), build_angle, cos.get());
                       graph_.append(storage, parameter_mapping[idx], gen_coeffs[idx], gate_indices[idx]);

                       extend_coeffs_from_current_picture_if_needed_(coeffs);

                       Layer layer(std::move(storage));
                       detail::LayerCosScale cos_scale = [cos](size_t, double *c, double v) {
                           detail::scale_cos_mask(c, *cos, v);
                       };
                       evolve_step(coeffs, layer, apply_angle, cos_scale, comm_);
                   });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_contract_immediately_(const std::vector<VecZ> &majoranas,
                                                                     const VecZ &parameter_mapping,
                                                                     const VecD &gen_coeffs,
                                                                     const VecD &parameters,
                                                                     int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    // Called for the side effect alone: it returns a reference to the very vector selected below.
    (void)current_picture_coeffs_();
    VecD *op_coeffs = schrodinger_ ? &mp_op_.state_coeffs : &mp_op_.op_coeffs;
    const auto majoranas_size = majoranas.size();
    // The build reports its fused-cos-sweep decision through `fused_scale`, so the apply cannot disagree.
    run_gate_loop_(
        majoranas,
        only_rotate_len_k,
        [this, &mapped_params, op_coeffs, majoranas_size](const VecZ &mono, int rot_len, size_t i) {
            const auto [build_angle, apply_angle] = gate_angle_(mapped_params, i, majoranas_size);
            // extend_coeffs must run AFTER build_evolve_result_'s self-rank grow and BEFORE the apply.
            CosMask cos;
            detail::FusedContract fc;
            bool fused_scale = false;
            build_evolve_result_(mono, rot_len, std::cref(*op_coeffs), build_angle, &cos, &fc, op_coeffs, &fused_scale);
            extend_coeffs_from_current_picture_if_needed_(*op_coeffs);
            detail::apply_fused_contract(fc, *op_coeffs, cos, apply_angle, schrodinger_, fused_scale);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_graph(const std::vector<VecZ> &majoranas,
                                               const VecZ &parameter_mapping,
                                               const VecD &gen_coeffs,
                                               std::optional<VecZ> gate_indices,
                                               std::optional<VecD> parameters,
                                               int only_rotate_len_k) -> void {
    if (shard_group_) {
        shard_group_->run_on_all([&](int r) {
            shard_group_->shard(r)
                .build_graph(majoranas, parameter_mapping, gen_coeffs, gate_indices, parameters, only_rotate_len_k);
        });
        return;
    }
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);

    // Gate indices are 0-based per call, so offset by the existing gate count to make them absolute.
    VecZ local_gates;
    if (gate_indices.has_value()) {
        local_gates = std::move(*gate_indices);
    }
    else {
        local_gates.resize(majoranas.size());
        std::iota(local_gates.begin(), local_gates.end(), size_t{0});
    }
    validate_gate_indices(local_gates, majoranas.size());
    const size_t gate_offset = n_gates();
    for (auto &g : local_gates) {
        g += gate_offset;
    }

    if (!parameters.has_value()) {
        evolve_mode_build_graph_(majoranas, parameter_mapping, gen_coeffs, local_gates, only_rotate_len_k);
    }
    else {
        // map_params() indexes `parameters` by parameter_mapping, so a too-short vector reads OOB.
        validate_parameters_length(*parameters, parameter_mapping);
        // Coefficient-informed build: seed by contracting the existing graph so atol truncation sees
        // realistic coefficients. That graph covers the parameter prefix [0, m); slice so the check passes.
        VecD seed;
        if (graph_layers() > 0) {
            const auto existing = graph_gate_arrays_();
            const size_t m = expected_num_params(existing.first);
            const VecD existing_params(
                parameters->begin(),
                parameters->begin() + static_cast<std::ptrdiff_t>(std::min(m, parameters->size())));
            seed = contract_partially(existing_params, false);
        }
        else {
            seed = current_picture_coeffs_();
        }
        evolve_mode_graph_with_coeffs_(majoranas,
                                       parameter_mapping,
                                       gen_coeffs,
                                       local_gates,
                                       *parameters,
                                       seed,
                                       only_rotate_len_k);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate(const std::vector<VecZ> &majoranas,
                                             const VecZ &parameter_mapping,
                                             const VecD &gen_coeffs,
                                             const VecD &parameters,
                                             int only_rotate_len_k) -> void {
    if (shard_group_) {
        shard_group_->run_on_all([&](int r) {
            shard_group_->shard(r).propagate(majoranas, parameter_mapping, gen_coeffs, parameters, only_rotate_len_k);
        });
        return;
    }
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);
    validate_parameters_length(parameters, parameter_mapping);
    // propagate() contracts in place and would consume any pre-existing graph layers from the front,
    // corrupting the result — reject a non-empty graph (contract_partially() folds one instead).
    if (graph_layers() > 0) {
        throw std::runtime_error(std::format("Cannot propagate() on top of a non-empty graph of {} layer(s): "
                                             "propagate() evolves and contracts in place and assumes no stored graph. "
                                             "Call contract_partially() to fold the existing graph first, or use "
                                             "build_graph() to extend it.",
                                             graph_layers()));
    }
    evolve_mode_contract_immediately_(majoranas, parameter_mapping, gen_coeffs, parameters, only_rotate_len_k);
}

template <size_t NumModes>
template <typename EvolutionFunc>
auto MonomialPropagator<NumModes>::run_gate_loop_(const std::vector<VecZ> &majoranas,
                                                  int only_rotate_len_k,
                                                  EvolutionFunc evolution_func) -> void {
    // Serial per shard; parallelism comes from sharding the operator across cores.
    for (size_t i = 0; i < majoranas.size(); ++i) {
        const auto idx = !schrodinger_ ? majoranas.size() - 1 - i : i;
        const auto &mono = majoranas[idx];
        evolution_func(mono, only_rotate_len_k, i);
    }

    initialize_operator_caches_();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_evolve_result_(const VecZ &gen_vec,
                                                        int only_rotate_len_k,
                                                        std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                        std::optional<double> param,
                                                        CosMask *out_cos,
                                                        detail::FusedContract *fused_contract,
                                                        VecD *fused_scale_coeffs,
                                                        bool *fused_scale) -> std::shared_ptr<LayerCore> {
    // The single choke point for every gate generator reaching the engine, and the only place they are
    // bounds-checked: nothing between the public entry points and here constrains a generator's indices.
    const auto gen_mono = indices_to_bitset_checked<NumModes>(gen_vec, 2 * logical_num_modes_);

    // Both parities go through the parity-corrected inverted-index scan (odd generators add the g_odd
    // parity(|M|) correction). The recompute metadata is written onto the returned LayerCore.
    return detail::build_layer<NumModes>(mp_op_,
                                         gen_mono,
                                         cutoff_fn_,
                                         lower_atol_,
                                         coeffs,
                                         upper_atol_,
                                         param,
                                         only_rotate_len_k,
                                         matched_scratch_,
                                         comm_,
                                         out_cos,
                                         fused_contract,
                                         schrodinger_,
                                         fused_scale_coeffs,
                                         fused_scale,
                                         basis_);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate_one_(const VecZ &gen_vec,
                                                  int only_rotate_len_k,
                                                  std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                  std::optional<double> param,
                                                  size_t param_index,
                                                  double gen_coeff,
                                                  size_t gate_index) -> void {
    // Gate info and recompute metadata ride on the LayerCore, so evaluation needs only the parameters.
    graph_.append(build_evolve_result_(gen_vec, only_rotate_len_k, coeffs, param), param_index, gen_coeff, gate_index);
}

// Forward declaration; defined below.
template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index,
                         const MPGraphView &graph,
                         Basis basis = Basis::Majorana) -> std::pair<detail::LayerCosScale, detail::LayerCosAccumulate>;

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_operator_with_recompute_(VecD &&coeffs,
                                                                   const MPGraphView &graph,
                                                                   const VecD &params) -> VecD {
    const auto &inverted_index = mp_op_.inverted_index();
    // Only the scale side is consumed; build the pair through the shared builder for consistency.
    auto cos_scale = build_cos_callbacks<NumModes>(inverted_index, graph, basis_).first;
    return evolve_operator(std::move(coeffs), graph, params, cos_scale, comm_);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::n_gates() const -> size_t {
    if (shard_group_) {
        return shard_group_->shard(0).n_gates();
    }
    const size_t count = graph_.layers();
    size_t max_gate = 0;
    bool any = false;
    for (size_t layer = 0; layer < count; ++layer) {
        const size_t g = graph_.get_layer_traversal(layer).gate_index();
        max_gate = any ? std::max(max_gate, g) : g;
        any = true;
    }
    return any ? max_gate + 1 : 0;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::set_parameter_mapping(const VecZ &parameter_mapping) -> void {
    if (shard_group_) {
        shard_group_->run_on_all([&](int r) { shard_group_->shard(r).set_parameter_mapping(parameter_mapping); });
        return;
    }
    const size_t count = graph_.layers();
    const size_t gates = n_gates();

    // The LayerCore is shared and immutable, so relabelling copies it and replaces the layer's core in
    // place (preserving any stored pruned cos).
    auto relabel = [this](size_t layer, size_t new_param_index) {
        auto &target = graph_.get_layer(layer);
        auto new_core = std::make_shared<LayerCore>(target.core());
        // Drop the inherited eval-time derivative layout: it must not depend on a prior gradient run.
        new_core->reset_derivative_exchange_layout();
        new_core->param_index = new_param_index;
        if (const CosMask *pruned = target.pruned_cos()) {
            target = Layer(std::move(new_core), *pruned);
        }
        else {
            target = Layer(std::move(new_core));
        }
    };

    if (parameter_mapping.size() == count) {
        // Per-layer mapping in optimizer order.
        for (size_t layer = 0; layer < count; ++layer) {
            relabel(layer, parameter_mapping[count - 1 - layer]);
        }
    }
    else if (parameter_mapping.size() == gates) {
        // Per-gate mapping indexed by absolute gate index: relabel each layer via its stored gate index.
        for (size_t layer = 0; layer < count; ++layer) {
            relabel(layer, parameter_mapping[graph_.get_layer_traversal(layer).gate_index()]);
        }
    }
    else {
        throw std::runtime_error(std::format("parameter_mapping has {} entries; expected {} (per graph "
                                             "layer) or {} (per gate).",
                                             parameter_mapping.size(),
                                             count,
                                             gates));
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_gate_arrays_() const -> std::pair<VecZ, VecD> {
    if (shard_group_) {
        return shard_group_->shard(0).graph_gate_arrays_();
    }
    const size_t count = graph_.layers();
    VecZ parameter_mapping(count);
    VecD gen_coeffs(count);
    // Layers store gate info in simulation order; the evaluation machinery expects optimizer order (the
    // reverse), so layer `layer` lands at optimizer index count-1-layer.
    for (size_t layer = 0; layer < count; ++layer) {
        const auto traversal = graph_.get_layer_traversal(layer);
        const size_t optimizer_index = count - 1 - layer;
        parameter_mapping[optimizer_index] = traversal.param_index();
        gen_coeffs[optimizer_index] = traversal.gen_coeff();
    }
    return {std::move(parameter_mapping), std::move(gen_coeffs)};
}

// Per-layer (scale, accumulate) cos callbacks for a replayed graph; they own their recipes, not the graph.
template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index, const MPGraphView &graph, Basis basis)
    -> std::pair<detail::LayerCosScale, detail::LayerCosAccumulate> {
    // Fold layers recompute cos on the fly (LazyFold), never retained. Recompute lower_bounds each
    // sparse column, which the row-order fill keeps ascending (asserted in InvertedIndex::fill_rows).

    struct LayerCos {
        bool recomputes_cos = false;
        detail::LazyFold<NumModes> recipe{}; // used iff recomputes_cos
        const CosMask *filtered = nullptr;   // points into a pruned layer's stored cos
    };
    auto cache = std::make_shared<std::vector<LayerCos>>();
    cache->reserve(graph.layers());
    for (size_t i = 0; i < graph.layers(); ++i) {
        const auto &layer = graph.get_layer(i);
        LayerCos entry;
        if (const CosMask *pruned = layer.pruned_cos(); pruned != nullptr) {
            entry.recomputes_cos = false;
            entry.filtered = pruned;
        }
        else {
            entry.recomputes_cos = true;
            const auto t = layer.traversal();
            const auto gen = detail::generator_from_words<NumModes>(t.generator_words());
            entry.recipe = detail::make_lazy_fold<NumModes>(inverted_index, gen, t.scaled_count(), basis);
        }
        cache->push_back(std::move(entry));
    }

    const auto *sc = &inverted_index;
    detail::LayerCosScale cos_scale = [cache, sc](size_t i, double *c, double v) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            detail::scale_cos_mask(c, *e.filtered, v);
        }
        else {
            detail::scale_cos_lazy<NumModes>(*sc, e.recipe, c, v);
        }
    };
    detail::LayerCosAccumulate cos_acc = [cache, sc](size_t i, double *s, double *h, double v, double sec) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            return detail::accumulate_cos_mask(s, h, *e.filtered, v, sec);
        }
        return detail::accumulate_cos_lazy<NumModes>(*sc, e.recipe, s, h, v, sec);
    };
    return {std::move(cos_scale), std::move(cos_acc)};
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_functional_(Fn &&func, std::optional<double> pare_threshold)
    -> std::function<R(const VecD &)> {
    auto gate_arrays = graph_gate_arrays_();
    auto parameter_mapping = std::move(gate_arrays.first);
    auto gen_coeffs = std::move(gate_arrays.second);
    const auto num_params = expected_num_params(parameter_mapping);

    // NOTHING here needs a dense state: energy only dots it against the evolved operator, and the gradient
    // scatters it into its own thread-local scratch before back-evolving. So Heisenberg hands over just the
    // sparse scores (~0.07% of rows); Schrödinger's state is the live evolved vector, snapshotted whole.
    // EvalState OWNS its rows and snapshots the term count -- a later append push_backs onto the operator's
    // sparse rows, which would both dangle a view and outrun the `op` captured below.
    const auto num_terms = mp_op_.size();
    auto state = [&] {
        if (schrodinger_) {
            return EvalState::dense(mp_op_.dense_state());
        }
        const auto sparse = mp_op_.sparse_state();
        return EvalState::sparse(num_terms, sparse.rows, sparse.values);
    }();
    VecD op = mp_op_.get_operator();
    const auto core_term = this->core_term();
    const auto comm = comm_;

    const auto expected_layers = graph_layers();
    const auto &inverted_index = mp_op_.inverted_index();

    // ONE owning handle either way: pare hands back a heap-owned MPGraph the functional must keep alive
    // (build_cos_callbacks holds pointers into its layers' stored cos); non-pare aliases graph_.
    std::shared_ptr<const MPGraph> graph;
    if (pare_threshold.has_value()) {
        auto full_cos_of_layer = [this, &inverted_index](size_t i) -> CosMask {
            const auto layer = graph_.get_layer_traversal(i);
            const auto gen = detail::generator_from_words<NumModes>(layer.generator_words());
            const auto combined = detail::make_fold_cache<NumModes>(inverted_index, gen, layer.scaled_count(), basis_);
            return detail::fold_to_cos_mask<NumModes>(combined);
        };
        // Keep the indices whose amplitude clears the threshold in the picture's driving vector: the
        // Hamiltonian in Schrödinger, the state otherwise.
        const auto keep = schrodinger_ ? indices_above(op, *pare_threshold) : state.indices_above(*pare_threshold);
        const auto count = schrodinger_ ? op.size() : state.length();
        graph =
            std::make_shared<const MPGraph>(pare_graph(graph_, keep, count, schrodinger_, comm_, full_cos_of_layer));
    }
    else {
        graph = std::shared_ptr<const MPGraph>(std::shared_ptr<const void>{}, &graph_);
    }

    // The inverted index outlives this simulator, so the folds' column pointers stay valid in the closure.
    auto callbacks = build_cos_callbacks<NumModes>(inverted_index, graph->replay_view(), basis_);
    detail::LayerCosScale cos_scale = std::move(callbacks.first);
    detail::LayerCosAccumulate cos_acc = std::move(callbacks.second);

    return make_parameter_validated_functional(
        num_params,
        [func = std::move(func),
         core_term,
         state = std::move(state),
         op = std::move(op),
         graph = std::move(graph),
         parameter_mapping,
         gen_coeffs,
         expected_layers,
         cos_scale = std::move(cos_scale),
         cos_acc = std::move(cos_acc),
         comm](const VecD &params) -> R {
            validate_expected_graph_layers(graph->layers(), expected_layers);
            return func(core_term, state, op, parameter_mapping, gen_coeffs, *graph, params, comm, cos_scale, cos_acc);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_functional(std::optional<double> pare_threshold)
    -> std::function<double(const VecD &)> {
    if (shard_group_) {
        // Each shard allreduces internally, so shard 0 is the global value. The group is captured by
        // raw pointer, so the returned callable must not outlive this propagator.
        auto fns = std::make_shared<std::vector<std::function<double(const VecD &)>>>(
            static_cast<size_t>(shard_group_->shard_count()));
        shard_group_->run_on_all([&](int r) {
            (*fns)[static_cast<size_t>(r)] = shard_group_->shard(r).expectation_value_functional(pare_threshold);
        });
        auto *grp = shard_group_.get();
        return [grp, fns](const VecD &params) -> double {
            std::vector<double> vals(fns->size());
            grp->run_on_all([&](int r) { vals[static_cast<size_t>(r)] = (*fns)[static_cast<size_t>(r)](params); });
            return vals[0];
        };
    }
    return make_functional_(ev_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient_functional(std::optional<double> pare_threshold)
    -> std::function<std::pair<double, VecD>(const VecD &)> {
    if (shard_group_) {
        auto fns = std::make_shared<std::vector<std::function<std::pair<double, VecD>(const VecD &)>>>(
            static_cast<size_t>(shard_group_->shard_count()));
        shard_group_->run_on_all([&](int r) {
            (*fns)[static_cast<size_t>(r)] =
                shard_group_->shard(r).expectation_value_and_gradient_functional(pare_threshold);
        });
        auto *grp = shard_group_.get();
        return [grp, fns](const VecD &params) -> std::pair<double, VecD> {
            std::vector<std::pair<double, VecD>> res(fns->size());
            grp->run_on_all([&](int r) { res[static_cast<size_t>(r)] = (*fns)[static_cast<size_t>(r)](params); });
            return res[0];
        };
    }
    return make_functional_(ev_and_grad_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value(const VecD &parameters) -> double {
    if (shard_group_) {
        // Each shard allreduces internally, so every shard returns the GLOBAL value; take shard 0.
        std::vector<double> vals(static_cast<size_t>(shard_group_->shard_count()));
        shard_group_->run_on_all(
            [&](int r) { vals[static_cast<size_t>(r)] = shard_group_->shard(r).expectation_value(parameters); });
        return vals[0];
    }
    return expectation_value_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD> {
    if (shard_group_) {
        std::vector<std::pair<double, VecD>> res(static_cast<size_t>(shard_group_->shard_count()));
        shard_group_->run_on_all([&](int r) {
            res[static_cast<size_t>(r)] = shard_group_->shard(r).expectation_value_and_gradient(parameters);
        });
        return res[0];
    }
    return expectation_value_and_gradient_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::contract_partially(const VecD &parameters, bool inplace) -> VecD {
    if (shard_group_) {
        // Partitions are disjoint, so concatenating in shard order enumerates the whole operator
        // (deterministic for a fixed shard count). Core term excluded, as on the unsharded path.
        std::vector<VecD> res(static_cast<size_t>(shard_group_->shard_count()));
        shard_group_->run_on_all([&](int r) {
            res[static_cast<size_t>(r)] = shard_group_->shard(r).contract_partially(parameters, inplace);
        });
        VecD merged;
        size_t total = 0;
        for (const auto &v : res) {
            total += v.size();
        }
        merged.reserve(total);
        for (auto &v : res) {
            merged.insert(merged.end(), v.begin(), v.end());
        }
        return merged;
    }
    const auto gate_arrays = graph_gate_arrays_();
    const auto &parameter_mapping = gate_arrays.first;
    const auto &gen_coeffs = gate_arrays.second;
    validate_parameters_length(parameters, parameter_mapping);

    if (parameters.empty()) {
        return current_picture_coeffs_();
    }

    const size_t num_majoranas = parameter_mapping.size();
    // Inplace slicing produces an owned MPGraph that must be bound to a named local before viewing
    // (never view a temporary); slice_view() views this graph's still-live layers directly.
    if (schrodinger_) {
        const auto &state = mp_op_.dense_state();
        const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, -1.0);
        VecD evolved_state;
        if (inplace) {
            const MPGraph sliced = graph_.slice_graph(num_majoranas, true);
            evolved_state = evolve_operator_with_recompute_(VecD(state), sliced.replay_view(), mapped_params);
            mp_op_.state_coeffs = evolved_state;
        }
        else {
            evolved_state =
                evolve_operator_with_recompute_(VecD(state), graph_.slice_view(num_majoranas), mapped_params);
        }
        return evolved_state;
    }

    const auto &op = mp_op_.get_operator();
    const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0, true);
    VecD evolved_op;
    if (inplace) {
        const MPGraph sliced = graph_.slice_graph(num_majoranas, true);
        evolved_op = evolve_operator_with_recompute_(VecD(op), sliced.replay_view(), mapped_params);
        mp_op_.op_coeffs = evolved_op;
    }
    else {
        evolved_op = evolve_operator_with_recompute_(VecD(op), graph_.slice_view(num_majoranas), mapped_params);
    }
    return evolved_op;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolved_operator_terms(const VecD &parameters, double atol)
    -> std::vector<std::pair<VecZ, std::complex<double>>> {
    using Term = std::pair<VecZ, std::complex<double>>;
    // `p` is always unsharded here (a shard, or *this), so indexing() is available.
    const auto collect = [&](MonomialPropagator &p) -> std::vector<Term> {
        std::vector<Term> terms;
        const VecD evolved = p.contract_partially(parameters, false);
        p.indexing().for_each([&](const auto &mono, size_t idx) {
            if (idx >= evolved.size()) {
                return;
            }
            const double coeff = evolved[idx];
            if (std::abs(coeff) < atol) {
                return;
            }
            // Round to drop anti-hermitian numerical noise (Majorana un-applies the Hermitian phase).
            const auto decoded = algebra_decode_coeff<NumModes>(basis_, coeff, mono);
            const std::complex<double> rounded(std::round(decoded.real() * 1e12) / 1e12,
                                               std::round(decoded.imag() * 1e12) / 1e12);
            terms.emplace_back(bitset_to_indices<NumModes>(mono), rounded);
        });
        return terms;
    };
    if (!shard_group_) {
        return collect(*this);
    }
    std::vector<std::vector<Term>> per(static_cast<size_t>(shard_group_->shard_count()));
    shard_group_->run_on_all([&](int r) { per[static_cast<size_t>(r)] = collect(shard_group_->shard(r)); });
    std::vector<Term> merged;
    size_t total = 0;
    for (const auto &v : per) {
        total += v.size();
    }
    merged.reserve(total);
    for (const auto &v : per) {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}

} // namespace monoprop
