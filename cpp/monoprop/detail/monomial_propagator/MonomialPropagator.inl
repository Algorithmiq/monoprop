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
#include "monoprop/detail/partition/PartitionGroup.h"

namespace monoprop {

// The ranks disagree on S. The count comes from partitions= or the environment on
// every rank independently, so the fix is to the launch, and it may belong to a different rank.
class PartitionCountMismatch : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

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
                                                 size_t partitions,
                                                 PartitionChildFactory child_factory)
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
        throw PropagatorConfigError(
            std::format("logical_num_modes ({}) must be in the range [1, {}].", logical_num_modes_, NumModes));
    }

    validate_cutoff_config_(cutoff_type_, basis_change_);

    mp_op_.basis = basis_;

    if (upper_atol.has_value() && lower_atol.has_value() && (upper_atol.value() < lower_atol.value())) {
        throw PropagatorConfigError(std::format("upper_atol ({}) must be greater than or equal to lower_atol ({}).",
                                                upper_atol.value(),
                                                lower_atol.value()));
    }

    const size_t n_partitions = resolve_partition_count_(partitions, comm);
    // The R ranks x S partitions form one flat P = R*S SPMD world, so a mismatch across ranks would
    // deadlock at the first hybrid collective.
    if (comm.kind == mpi::Comm::Kind::Mpi && mpi::size(comm) > 1
        && mpi::allreduce_sum<size_t>(n_partitions, comm) != n_partitions * static_cast<size_t>(mpi::size(comm))) {
        throw PartitionCountMismatch(
            "Partition count differs across MPI ranks — every rank must resolve the same "
            "partitions= / monoprop_PARTITIONS / monoprop_NUM_THREADS so R*S is a consistent world.");
    }
    if (n_partitions > 1) {
        PartitionChildFactory factory =
            child_factory ? std::move(child_factory) : PartitionChildFactory{[=](mpi::Comm partition_comm) {
                return std::make_unique<MonomialPropagator<NumModes>>(initial_operator,
                                                                      cutoff,
                                                                      initial_state,
                                                                      schrodinger_cutoff,
                                                                      partition_comm,
                                                                      lower_atol,
                                                                      upper_atol,
                                                                      cutoff_type,
                                                                      basis_change,
                                                                      logical_num_modes,
                                                                      basis,
                                                                      /*partitions=*/1);
            }};
        partition_group_ = std::make_unique<detail::partition::PartitionGroup<NumModes>>(static_cast<int>(n_partitions),
                                                                                         factory,
                                                                                         comm);
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

    // Schrodinger's initial rows are the global paired basis, hash-partitioned over the P = R*S world
    // slots; Heisenberg's local_heisenberg_terms was already filtered to this slot's share above. Hence the
    // two reserves: a global count needs its 1/P share, a local one already is the share.
    size_t max_pairs = 0;
    size_t expected_local_terms = std::max<size_t>(1, local_heisenberg_terms.size());
    if (schrodinger_) {
        // schrodinger_ is schrodinger_cutoff.has_value(), set in the member-init list, so the deref holds.
        const auto sc = std::min(*schrodinger_cutoff, static_cast<unsigned int>(2 * logical_num_modes_));
        max_pairs = sc / 2 + sc % 2;
        const size_t global_terms = paired_op_size(max_pairs, logical_num_modes_);
        // paired_op_size saturates rather than wrapping, so this is "too large to count", not a size.
        const bool uncountable = global_terms == std::numeric_limits<size_t>::max();
        const size_t share = global_terms / std::max<size_t>(1, num_ranks);
        // A cutoff near the mode count makes the basis 2^logical_num_modes. Two independent ways that
        // is hopeless, and both are needed: the count does not fit size_t at all, which is
        // rank-independent because every slot walks the whole global basis however the rows are
        // divided; or the slot's share of rows cannot be addressed by a term index. Rejected here by
        // name rather than as a walk that does not finish.
        if (uncountable || share >= detail::OperatorIndex<NumModes>::kIndexCeiling) {
            const auto how_many = uncountable ? std::string("more than 2^64") : std::format("{}", global_terms);
            const auto per_slot = uncountable ? std::string("as many") : std::format("{}", share);
            throw PropagatorConfigError(
                std::format("schrodinger_cutoff ({}) admits {} paired basis terms over {} active modes, "
                            "about {} per world slot — more than can be walked or addressed. Lower "
                            "schrodinger_cutoff, or raise the rank x partition count (currently {}).",
                            *schrodinger_cutoff,
                            how_many,
                            logical_num_modes_,
                            per_slot,
                            num_ranks));
        }
        expected_local_terms = std::max<size_t>(1, share);
    }

    // Must run before the store: packed_inline_width_() derives the packed-row width from cutoff_fn_.
    regenerate_cutoff_fn_();
    mp_op_.store = std::make_unique<detail::OperatorIndex<NumModes>>(packed_inline_width_());
    mp_op_.store->reserve(expected_local_terms);
    // Store replaced: drop the stale lazy inverted index so it rebuilds against the new store.
    mp_op_.inverted_index_.reset();

    size_t i = 0;
    // The initial monomials are distinct, so emplace (insert-if-absent) is an assigning insert here. A row
    // index is a position in the kept subsequence, so the enumeration order below is load-bearing.
    const auto keep_if_owned = [&](const Monomial<NumModes> &mono) {
        if (my_rank == find_rank<NumModes>(mono, num_ranks)) {
            mp_op_.append_term(mono);
            mp_op_.store->emplace(mono, i++);
        }
    };
    if (schrodinger_) {
        // Enumerated, never listed: PartitionGroup constructs the S partitions concurrently on their own
        // masters, so a list here would hold P full copies of the global basis at one instant.
        for_each_paired_monomial<NumModes>(max_pairs, logical_num_modes_, keep_if_owned);
    }
    else {
        for (const auto &mono : local_heisenberg_terms) { // already this slot's share; the test is a no-op
            keep_if_owned(mono);
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
      initial_operator_epoch_(other.initial_operator_epoch_),
      logical_num_modes_(other.logical_num_modes_),
      cutoff_type_(other.cutoff_type_),
      basis_change_(other.basis_change_),
      basis_(other.basis_),
      partition_group_(other.partition_group_
                           ? std::make_unique<detail::partition::PartitionGroup<NumModes>>(*other.partition_group_)
                           : nullptr) {}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::resolve_partition_count_(size_t requested, mpi::Comm comm) -> size_t {
    if (requested >= 1) {
        return requested;
    }
    // One serial partition per physical core, capped by monoprop_NUM_THREADS. On a multi-rank comm this
    // engages only when threads were explicitly requested, so a pure-MPI run is not oversubscribed.
    const auto compute_auto = [&]() -> size_t {
        const int ranks = mpi::size(comm);
        size_t cores = detail::partition::enumerate_physical_cores().size();
        if (cores == 0) {
            // Topology unreadable: use half the hardware threads so smt siblings aren't counted as cores.
            cores = std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()) / 2);
        }
        const auto num_threads = config::get().num_threads;
        const size_t budget =
            num_threads.has_value() ? static_cast<size_t>(*num_threads) : (ranks == 1 ? cores : size_t{1});
        return std::max<size_t>(1, std::min(budget, cores));
    };
    if (const char *env = std::getenv("monoprop_PARTITIONS")) {
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

// Partition fan-out vocabulary; the declarations record which helper is legal where.

template <size_t NumModes>
auto MonomialPropagator<NumModes>::for_each_partition_(const std::function<void(MonomialPropagator &)> &fn) -> void {
    partition_group_->run_on_all([&](int r) { fn(partition_group_->partition(r)); });
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::map_partitions_(Fn fn) -> std::vector<R> {
    return detail::partition::map_partitions(*partition_group_, fn);
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::map_partitions_indexed_(Fn fn) -> std::vector<R> {
    return detail::partition::collect_on_all(*partition_group_,
                                             [&](int r) -> R { return fn(r, partition_group_->partition(r)); });
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::concat_partitions_(Fn fn) -> R {
    const auto per_partition = map_partitions_(fn);
    size_t total = 0;
    for (const auto &v : per_partition) {
        total += v.size();
    }
    R merged;
    merged.reserve(total);
    for (const auto &v : per_partition) {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}

template <size_t NumModes>
template <typename Proj, typename Accumulate, typename R>
auto MonomialPropagator<NumModes>::fold_partitions_(Proj proj, Accumulate accumulate) const -> R {
    R total{};
    for (int r = 0; r < partition_group_->partition_count(); ++r) {
        accumulate(total, proj(partition_group_->partition(r)));
    }
    return total;
}

template <size_t NumModes>
template <typename Proj, typename R>
auto MonomialPropagator<NumModes>::sum_partitions_(Proj proj) const -> R {
    return fold_partitions_(proj, [](R &total, const R &value) { total += value; });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::first_partition_() const -> const MonomialPropagator & {
    return partition_group_->partition(0);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::partitioned_size_() const -> size_t {
    return sum_partitions_([](const MonomialPropagator &s) { return s.size(); });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::partitioned_graph_size_() const -> std::pair<size_t, size_t> {
    // One pass: graph_size() recomputes the cosine-only count, so it must not be projected twice.
    return fold_partitions_([](const MonomialPropagator &s) { return s.graph_size(); },
                            [](std::pair<size_t, size_t> &total, const std::pair<size_t, size_t> &value) {
                                total.first += value.first;
                                total.second += value.second;
                            });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::partitioned_graph_layers_() const -> size_t {
    return first_partition_().graph_layers();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::partitioned_core_term_() const -> double {
    return first_partition_().core_term();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::partitioned_operator_memory_usage_() const
    -> detail::MPOperatorMemoryBreakdown<NumModes> {
    return sum_partitions_([](const MonomialPropagator &s) { return s.operator_memory_usage(); });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::partitioned_graph_memory_usage_() const -> GraphMemoryBreakdown {
    return sum_partitions_([](const MonomialPropagator &s) { return s.graph_memory_usage(); });
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
    ++initial_operator_epoch_;
    if (partition_group_) {
        // The facade holds no local terms of its own, so the return is empty.
        for_each_partition_([&](MonomialPropagator &s) { s.update_initial_operator(op_dict); });
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
    require_single_partition_("graph_data()");
    std::vector<LayerData> layers;
    const auto num_layers = graph_.layers();
    layers.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        const auto traversal = graph_.get_layer_traversal(i);
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
            auto p = detail::make_fold_cache<NumModes>(mp_op_.inverted_index(), gen, traversal.scaled_count(), basis_);
            cos_inds = detail::fold_to_indices<NumModes>(p);
        }
        layers.emplace_back(std::move(cos_inds), std::move(local_cyc_data), std::move(b_data), std::move(d_data));
    }
    return layers;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::cos_index_count_() const -> size_t {
    // Only a pared layer stores a cosine set; otherwise recompute the fold here. Cosine-only = cos-scaled
    // minus the rotation endpoints, saturating at 0.
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
    (void)mp_op_.get_operator();
    // Heisenberg warms the sparse state only; densifying here would defeat it. Schrödinger's dense vector
    // IS the live evolved vector.
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
                                                            std::optional<size_t> only_rotate_len_k) -> void {
    const auto majoranas_size = majoranas.size();
    run_gate_loop_(majoranas,
                   only_rotate_len_k,
                   [this, &parameter_mapping, &gen_coeffs, &gate_indices, majoranas_size](const VecZ &mono,
                                                                                          std::optional<size_t> rot_len,
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
                                                                  std::optional<size_t> only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    auto coeffs = operator_coeffs;
    const auto majoranas_size = majoranas.size();

    run_gate_loop_(majoranas,
                   only_rotate_len_k,
                   [this, &parameter_mapping, &gen_coeffs, &gate_indices, &mapped_params, &coeffs, majoranas_size](
                       const VecZ &mono,
                       std::optional<size_t> rot_len,
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
                       evolve_step(coeffs, layer, apply_angle, comm_, cos_scale);
                   });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_contract_immediately_(const std::vector<VecZ> &majoranas,
                                                                     const VecZ &parameter_mapping,
                                                                     const VecD &gen_coeffs,
                                                                     const VecD &parameters,
                                                                     std::optional<size_t> only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    // Called for the side effect alone: it returns a reference to the very vector selected below.
    (void)current_picture_coeffs_();
    VecD *op_coeffs = schrodinger_ ? &mp_op_.state_coeffs : &mp_op_.op_coeffs;
    const auto majoranas_size = majoranas.size();
    run_gate_loop_(
        majoranas,
        only_rotate_len_k,
        [this, &mapped_params, op_coeffs, majoranas_size](const VecZ &mono, std::optional<size_t> rot_len, size_t i) {
            const auto [build_angle, apply_angle] = gate_angle_(mapped_params, i, majoranas_size);
            // extend_coeffs must run after build_evolve_result_'s self-rank grow and before the apply.
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
                                               std::optional<size_t> only_rotate_len_k) -> void {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * logical_num_modes_);
    if (partition_group_) {
        for_each_partition_([&](MonomialPropagator &s) {
            s.build_graph(majoranas, parameter_mapping, gen_coeffs, gate_indices, parameters, only_rotate_len_k);
        });
        return;
    }
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);

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
        // map_params() indexes `parameters` by parameter_mapping, so a too-short vector reads out of bounds.
        validate_parameters_length(*parameters, parameter_mapping);
        // Coefficient-informed build: seed by contracting the existing graph so atol truncation sees
        // realistic coefficients. That graph covers the parameter prefix [0, m).
        VecD seed;
        if (graph_layers() > 0) {
            const auto existing = graph_gate_arrays_();
            const size_t m = expected_num_params(existing.first);
            // The per-mapping check above only covers this call's indices, which may all sit above the
            // prefix the stored graph needs. Truncating instead would replay the existing graph at a silently
            // different point on the axis, and map_params would fail one layer down on the sliced vector.
            if (parameters->size() < m) {
                throw SeedParametersTooShort(
                    std::format("Coefficient-informed build_graph() needs at least {} parameter value(s) to replay the "
                                "existing {}-layer graph as a seed, but got {}.",
                                m,
                                graph_layers(),
                                parameters->size()));
            }
            const VecD existing_params(parameters->begin(), parameters->begin() + static_cast<std::ptrdiff_t>(m));
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
                                             std::optional<size_t> only_rotate_len_k) -> void {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * logical_num_modes_);
    if (partition_group_) {
        for_each_partition_([&](MonomialPropagator &s) {
            s.propagate(majoranas, parameter_mapping, gen_coeffs, parameters, only_rotate_len_k);
        });
        return;
    }
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);
    validate_parameters_length(parameters, parameter_mapping);
    if (graph_layers() > 0) {
        throw GraphStateConflict(std::format("Cannot propagate() on top of a non-empty graph of {} layer(s): "
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
                                                  std::optional<size_t> only_rotate_len_k,
                                                  EvolutionFunc evolution_func) -> void {
    // Serial per partition; parallelism comes from partitioning the operator across cores.
    for (size_t i = 0; i < majoranas.size(); ++i) {
        const auto idx = !schrodinger_ ? majoranas.size() - 1 - i : i;
        const auto &mono = majoranas[idx];
        evolution_func(mono, only_rotate_len_k, i);
    }

    initialize_operator_caches_();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_evolve_result_(const VecZ &gen_vec,
                                                        std::optional<size_t> only_rotate_len_k,
                                                        std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                        std::optional<double> param,
                                                        CosMask *out_cos,
                                                        detail::FusedContract *fused_contract,
                                                        VecD *fused_scale_coeffs,
                                                        bool *fused_scale) -> std::shared_ptr<LayerCore> {
    // The only place a gate generator's indices are bounds-checked: nothing between the public entry
    // points and here constrains them.
    const auto gen_mono = indices_to_bitset_checked<NumModes>(gen_vec, 2 * logical_num_modes_);

    // The cos-recompute metadata is written onto the returned LayerCore.
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
                                                  std::optional<size_t> only_rotate_len_k,
                                                  std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                  std::optional<double> param,
                                                  size_t param_index,
                                                  double gen_coeff,
                                                  size_t gate_index) -> void {
    graph_.append(build_evolve_result_(gen_vec, only_rotate_len_k, coeffs, param), param_index, gen_coeff, gate_index);
}

template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index,
                         const MPGraphView &graph,
                         Basis basis = Basis::Majorana) -> detail::CosCallbacks;

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_operator_with_recompute_(VecD &&coeffs,
                                                                   const MPGraphView &graph,
                                                                   const VecD &params) -> VecD {
    const auto &inverted_index = mp_op_.inverted_index();
    // Only the scale side is consumed; build both through the shared builder for consistency.
    auto cos_scale = build_cos_callbacks<NumModes>(inverted_index, graph, basis_).scale;
    return evolve_operator(std::move(coeffs), graph, params, comm_, cos_scale);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::n_gates() const -> size_t {
    if (partition_group_) {
        return first_partition_().n_gates();
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
    if (partition_group_) {
        for_each_partition_([&](MonomialPropagator &s) { s.set_parameter_mapping(parameter_mapping); });
        return;
    }
    const size_t count = graph_.layers();
    const size_t gates = n_gates();

    // The LayerCore is shared and immutable, so relabelling copies it and replaces the layer's core.
    auto relabel = [this](size_t layer, size_t new_param_index) {
        auto &target = graph_.get_layer(layer);
        auto new_core = std::make_shared<LayerCore>(target.core());
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
        // Per-gate mapping, indexed by absolute gate index.
        for (size_t layer = 0; layer < count; ++layer) {
            relabel(layer, parameter_mapping[graph_.get_layer_traversal(layer).gate_index()]);
        }
    }
    else {
        throw GraphStateConflict(std::format("parameter_mapping has {} entries; expected {} (per graph "
                                             "layer) or {} (per gate).",
                                             parameter_mapping.size(),
                                             count,
                                             gates));
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_gate_arrays_() const -> std::pair<VecZ, VecD> {
    if (partition_group_) {
        return first_partition_().graph_gate_arrays_();
    }
    const size_t count = graph_.layers();
    VecZ parameter_mapping(count);
    VecD gen_coeffs(count);
    // Layers store gate info in simulation order; the evaluation machinery expects optimizer order (the reverse).
    for (size_t layer = 0; layer < count; ++layer) {
        const auto traversal = graph_.get_layer_traversal(layer);
        const size_t optimizer_index = count - 1 - layer;
        parameter_mapping[optimizer_index] = traversal.param_index();
        gen_coeffs[optimizer_index] = traversal.gen_coeff();
    }
    return {std::move(parameter_mapping), std::move(gen_coeffs)};
}

template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index, const MPGraphView &graph, Basis basis)
    -> detail::CosCallbacks {
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
    detail::LayerCosIndices cos_inds = [cache, sc](size_t i, std::vector<TermIndex> &out) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            detail::cos_indices_mask(*e.filtered, out);
            return;
        }
        detail::cos_indices_lazy<NumModes>(*sc, e.recipe, out);
    };
    return {.scale = std::move(cos_scale), .accumulate = std::move(cos_acc), .indices = std::move(cos_inds)};
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_functional_(Fn &&func, std::optional<double> pare_threshold)
    -> std::function<R(const VecD &)> {
    auto gate_arrays = graph_gate_arrays_();
    auto parameter_mapping = std::move(gate_arrays.first);
    auto gen_coeffs = std::move(gate_arrays.second);
    const auto num_params = expected_num_params(parameter_mapping);

    // Nothing here needs a dense state: energy only dots it against the evolved operator, and the gradient
    // scatters it into its own thread-local scratch before back-evolving. So Heisenberg hands over just the
    // sparse scores; Schrödinger's state is the live evolved vector, snapshotted whole.
    // EvalState owns its rows and snapshots the term count -- a later append push_backs onto the operator's
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
    // Aliased rather than copied: the check below needs the live counter, like graph->layers().
    const auto *epoch = &initial_operator_epoch_;
    const auto expected_epoch = initial_operator_epoch_;
    const auto &inverted_index = mp_op_.inverted_index();

    // One owning handle either way: pare hands back a heap-owned MPGraph the functional must keep alive
    // (build_cos_callbacks holds pointers into its layers' stored cos); non-pare aliases graph_.
    std::shared_ptr<const MPGraph> graph;
    if (pare_threshold.has_value()) {
        auto full_cos_of_layer = [this, &inverted_index](size_t i) -> CosMask {
            const auto layer = graph_.get_layer_traversal(i);
            const auto gen = detail::generator_from_words<NumModes>(layer.generator_words());
            const auto combined = detail::make_fold_cache<NumModes>(inverted_index, gen, layer.scaled_count(), basis_);
            return detail::fold_to_cos_mask<NumModes>(combined);
        };
        // Threshold the picture's driving vector: the Hamiltonian in Schrödinger, the state otherwise.
        const auto keep = schrodinger_ ? indices_above(op, *pare_threshold) : state.indices_above(*pare_threshold);
        const auto count = schrodinger_ ? op.size() : state.length();
        graph =
            std::make_shared<const MPGraph>(pare_graph(graph_, keep, count, schrodinger_, comm_, full_cos_of_layer));
    }
    else {
        graph = std::shared_ptr<const MPGraph>(std::shared_ptr<const void>{}, &graph_);
    }

    // The folds keep raw column pointers into this propagator's inverted index, so the returned callable
    // must not outlive the propagator.
    auto cos = build_cos_callbacks<NumModes>(inverted_index, graph->replay_view(), basis_);

    return [func = std::move(func),
            core_term,
            state = std::move(state),
            op = std::move(op),
            graph = std::move(graph),
            parameter_mapping,
            gen_coeffs,
            num_params,
            epoch,
            expected_epoch,
            expected_layers,
            cos = std::move(cos),
            comm](const VecD &params) -> R {
        validate_expected_initial_operator(*epoch, expected_epoch);
        validate_functional_call(params, num_params);
        validate_expected_graph_layers(graph->layers(), expected_layers);
        return func(EvalRequest{.e_core = core_term,
                                .state = state,
                                .op = op,
                                .parameter_mapping = parameter_mapping,
                                .gen_coeffs = gen_coeffs,
                                .graph = graph->replay_view(),
                                .params = params},
                    comm,
                    cos);
    };
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_functional(std::optional<double> pare_threshold)
    -> std::function<double(const VecD &)> {
    if (partition_group_) {
        // Each partition allreduces internally, so partition 0 is the global value. The group is captured by
        // raw pointer, so the returned callable must not outlive this propagator.
        auto fns = std::make_shared<std::vector<std::function<double(const VecD &)>>>(
            map_partitions_([&](MonomialPropagator &s) { return s.expectation_value_functional(pare_threshold); }));
        auto *grp = partition_group_.get();
        return [grp, fns](const VecD &params) -> double {
            return detail::partition::collect_on_all(*grp,
                                                     [&](int r) { return (*fns)[static_cast<size_t>(r)](params); })[0];
        };
    }
    return make_functional_(ev_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient_functional(std::optional<double> pare_threshold)
    -> std::function<std::pair<double, VecD>(const VecD &)> {
    if (partition_group_) {
        auto fns = std::make_shared<std::vector<std::function<std::pair<double, VecD>(const VecD &)>>>(map_partitions_(
            [&](MonomialPropagator &s) { return s.expectation_value_and_gradient_functional(pare_threshold); }));
        auto *grp = partition_group_.get();
        return [grp, fns](const VecD &params) -> std::pair<double, VecD> {
            return detail::partition::collect_on_all(*grp,
                                                     [&](int r) { return (*fns)[static_cast<size_t>(r)](params); })[0];
        };
    }
    return make_functional_(ev_and_grad_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value(const VecD &parameters) -> double {
    if (partition_group_) {
        // Each partition allreduces internally, so every partition returns the global value; take partition 0.
        return map_partitions_([&](MonomialPropagator &s) { return s.expectation_value(parameters); })[0];
    }
    return expectation_value_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD> {
    if (partition_group_) {
        // As in expectation_value(): the gradient is allreduced inside each partition.
        return map_partitions_([&](MonomialPropagator &s) { return s.expectation_value_and_gradient(parameters); })[0];
    }
    return expectation_value_and_gradient_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::contract_partially(const VecD &parameters, bool inplace) -> VecD {
    if (partition_group_) {
        return concat_partitions_([&](MonomialPropagator &s) { return s.contract_partially(parameters, inplace); });
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
    // `p` is always unpartitioned here (a partition, or *this), so indexing() is available.
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
    if (!partition_group_) {
        return collect(*this);
    }
    return concat_partitions_(collect);
}

} // namespace monoprop
