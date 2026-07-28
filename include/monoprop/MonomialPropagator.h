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
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <format>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/Evolution.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/Validation.h"
#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop {
namespace detail {
// Fused-contraction record sink (layer_build/Common.h).
struct FusedContract;
namespace partition {
// Intra-process partition runtime; held by unique_ptr, so a forward declaration suffices here.
template <size_t NumModes>
class PartitionGroup;
} // namespace partition
} // namespace detail

template <size_t NumModes>
class MonomialPropagator {
public:
    MonomialPropagator(const OperatorDict &initial_operator,
                       unsigned int cutoff,
                       const VecZ &initial_state,
                       std::optional<unsigned int> schrodinger_cutoff,
                       mpi::Comm comm,
                       std::optional<double> lower_atol = std::nullopt,
                       std::optional<double> upper_atol = std::nullopt,
                       CutoffType cutoff_type = CutoffType::Length,
                       std::optional<std::vector<VecZ>> basis_change = std::nullopt,
                       size_t logical_num_modes = NumModes,
                       Basis basis = Basis::Majorana,
                       size_t partitions = 0);

    /// Out-of-line because partition_group_ is a unique_ptr to an incomplete type here.
    virtual ~MonomialPropagator();

    /// Deep copy: clones the operator store, shares the immutable graph cores, and clones the whole
    /// partition group on a facade. The virtual destructor suppresses implicit moves, so a "move" deep-copies.
    MonomialPropagator(const MonomialPropagator &other);
    auto operator=(const MonomialPropagator &) -> MonomialPropagator & = delete;

    static constexpr auto num_modes{NumModes};
    static constexpr auto storage_num_modes{NumModes};

    auto logical_num_modes() const -> size_t { return logical_num_modes_; }

    /// Term count on this rank (allreduce for global).
    auto size() const -> size_t { return partition_group_ ? partitioned_size_() : mp_op_.size(); }

    /// (cosine-only indices, cycles) on this rank; cosine-only = cos-scaled but not a rotation endpoint.
    auto graph_size() const -> std::pair<size_t, size_t> {
        return partition_group_ ? partitioned_graph_size_() : std::pair{cos_index_count_(), graph_.total_cycles()};
    }

    /// The propagation graph (local to this rank). Single-partition only — see require_single_partition_.
    auto graph() const -> const MPGraph & {
        require_single_partition_("graph()");
        return graph_;
    }

    /// This rank's operator storage. Single-partition only — see require_single_partition_.
    auto mp_op() -> detail::MPOperator<NumModes> & {
        require_single_partition_("mp_op()");
        return mp_op_;
    }
    auto mp_op() const -> const detail::MPOperator<NumModes> & {
        require_single_partition_("mp_op()");
        return mp_op_;
    }

    // Memory breakdowns sum across partitions (the fields are additive over disjoint hash partitions).
    auto graph_memory_usage() const -> GraphMemoryBreakdown {
        if (partition_group_) {
            return partitioned_graph_memory_usage_();
        }
        return graph_.storage_memory_usage();
    }

    auto operator_memory_usage() const -> detail::MPOperatorMemoryBreakdown<NumModes> {
        if (partition_group_) {
            return partitioned_operator_memory_usage_();
        }
        return detail::estimate_memory_usage(mp_op_);
    }

    auto graph_layers() const -> size_t { return partition_group_ ? partitioned_graph_layers_() : graph_.layers(); }

    /// Distinct gate count. A multi-term gate spans several layers, so n_gates() <= graph_layers().
    auto n_gates() const -> size_t;

    /// The per-layer parameter mapping owned by the graph, in optimizer order (length = graph_layers()).
    auto parameter_mapping() const -> VecZ { return graph_gate_arrays_().first; }

    /// Re-wire which parameter drives each graph layer, in place. Accepts a per-layer mapping (length
    /// graph_layers(), optimizer order) or a per-gate one (length n_gates()); on a tie, per-layer wins.
    auto set_parameter_mapping(const VecZ &parameter_mapping) -> void;

    /// This rank's indexing map (monomial → coefficient index). C++-only, single-partition only.
    auto indexing() -> detail::OperatorIndex<NumModes> & {
        require_single_partition_("indexing()");
        return *mp_op_.store;
    }
    auto indexing() const -> const detail::OperatorIndex<NumModes> & {
        require_single_partition_("indexing()");
        return *mp_op_.store;
    }

    /// Per-layer (cos_inds, local_cycles, cross_rank_sin_send, cross_rank_sin_recv) for this
    /// rank/partition. local_cycles is always empty: local cycles are folded into cross_rank[my_rank].
    using LocalCycleData = std::tuple<size_t, size_t, int>;
    using CrossRankData = std::tuple<VecZ, VecI>; // (indices, phases)
    using LayerData =
        std::tuple<VecZ, std::vector<LocalCycleData>, std::vector<CrossRankData>, std::vector<CrossRankData>>;
    auto graph_data() const -> std::vector<LayerData>;

    /// Raise or lower the truncation floor; must not exceed the current upper_atol.
    auto update_lower_atol(std::optional<double> new_lower_atol) -> void {
        if (upper_atol_.has_value() && new_lower_atol.has_value() && (new_lower_atol.value() > upper_atol_.value())) {
            throw std::runtime_error(
                std::format("New lower_atol ({}) must be less than or equal to current upper_atol ({}).",
                            new_lower_atol.value(),
                            upper_atol_.value()));
        }
        update_setting_([&](MonomialPropagator &p) { p.lower_atol_ = new_lower_atol; });
    }

    /// Raise or lower the exact-retention ceiling; must not fall below the current lower_atol.
    auto update_upper_atol(std::optional<double> new_upper_atol) -> void {
        if (lower_atol_.has_value() && new_upper_atol.has_value() && new_upper_atol.value() < lower_atol_.value()) {
            throw std::runtime_error(
                std::format("New upper_atol ({}) must be greater than or equal to current lower_atol ({}).",
                            new_upper_atol.value(),
                            lower_atol_.value()));
        }
        update_setting_([&](MonomialPropagator &p) { p.upper_atol_ = new_upper_atol; });
    }

    /// Re-cut at a new cutoff, regenerating the cutoff function. Existing terms are not re-truncated.
    auto update_cutoff(unsigned int new_cutoff) -> void {
        update_setting_([&](MonomialPropagator &p) {
            p.cutoff_ = new_cutoff;
            p.regenerate_cutoff_fn_();
        });
    }

    /// Switch the cutoff metric (Length / Support); rejected if this algebra cannot honour it.
    auto update_cutoff_type(CutoffType new_cutoff_type) -> void {
        validate_cutoff_config_(new_cutoff_type, basis_change_);
        update_setting_([&](MonomialPropagator &p) {
            p.cutoff_type_ = new_cutoff_type;
            p.regenerate_cutoff_fn_();
        });
    }

    /// Replace the basis the cutoff is measured in (nullopt ⇒ the native basis).
    auto update_basis_change(std::optional<std::vector<VecZ>> new_basis_change) -> void {
        validate_cutoff_config_(cutoff_type_, new_basis_change);
        update_setting_([&](MonomialPropagator &p) {
            p.basis_change_ = new_basis_change;
            p.regenerate_cutoff_fn_();
        });
    }

    auto schrodinger() const -> bool { return schrodinger_; }

    auto basis() const -> Basis { return basis_; }

    auto core_term() const -> double { return partition_group_ ? partitioned_core_term_() : core_term_; }

    auto cutoff() const -> unsigned int { return cutoff_; }

    auto lower_atol() const -> std::optional<double> { return lower_atol_; }

    auto upper_atol() const -> std::optional<double> { return upper_atol_; }

    auto cutoff_type() const -> CutoffType { return cutoff_type_; }

    auto basis_change() const -> std::optional<std::vector<VecZ>> { return basis_change_; }

    /// The MPI communicator (MPI_COMM_SELF for a partition, which trades over an in-process comm).
    auto comm() const -> MPI_Comm { return comm_.mpi; }

    /// Build the propagation graph, one layer per generator, recording each layer's gate info
    /// (angle = parameters[mapping[i]] * gen_coeffs[i]). Accumulates across calls. `gate_indices` is
    /// 0-based per call, offset internally by the gate count already in the graph. Pass `parameters` to
    /// seed atol truncation while extending a non-empty graph. `only_rotate_len_k` > 0 applies gates to
    /// monomials of length <= k even if they anticommute. Heisenberg consumes each call's sequence in
    /// reverse, so a forward split across calls is not equivalent; Schrodinger is front-to-back, so it is.
    auto build_graph(const std::vector<VecZ> &majoranas,
                     const VecZ &parameter_mapping,
                     const VecD &gen_coeffs,
                     std::optional<VecZ> gate_indices = std::nullopt,
                     std::optional<VecD> parameters = std::nullopt,
                     int only_rotate_len_k = 0) -> void;

    /// Evolve and contract immediately, without storing a graph.
    auto propagate(const std::vector<VecZ> &majoranas,
                   const VecZ &parameter_mapping,
                   const VecD &gen_coeffs,
                   const VecD &parameters,
                   int only_rotate_len_k = 0) -> void;

    /// Expectation value at the given variational parameters (gate info owned by the graph).
    auto expectation_value(const VecD &parameters) -> double;

    auto expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD>;

    /// Reusable callable. `pare_threshold` is the edge-retention cutoff for a masked plan; nullopt
    /// keeps the exact graph.
    auto expectation_value_functional(std::optional<double> pare_threshold = std::nullopt)
        -> std::function<double(const VecD &)>;

    /// Reusable callable computing (expectation value, gradient); `pare_threshold` as above.
    auto expectation_value_and_gradient_functional(std::optional<double> pare_threshold = std::nullopt)
        -> std::function<std::pair<double, VecD>(const VecD &)>;

    /// Contract the graph into the operator (Heisenberg) or state (Schrodinger). `inplace` consumes the
    /// graph and updates internal state; otherwise nothing is mutated. Core term excluded either way.
    /// Coefficients are positioned by the owning partition's indexing(), so on a facade the result is
    /// the per-partition blocks concatenated in partition order: the same multiset as an unpartitioned
    /// run, but not positionally stable across partition counts — and the count is auto-picked from the
    /// host's core count unless pinned. Use evolved_operator_terms() when positions must mean something.
    auto contract_partially(const VecD &parameters, bool inplace) -> VecD;

    /// Decoded (indices, coefficient) terms with |coeff| >= atol, gathered across every partition's
    /// disjoint partition. Contracts non-inplace. Core term excluded.
    auto evolved_operator_terms(const VecD &parameters, double atol)
        -> std::vector<std::pair<VecZ, std::complex<double>>>;

    virtual auto update_initial_operator(const OperatorDict &op_dict) -> void { apply_initial_operator_(op_dict); }

protected:
    // Reusable evaluation callbacks for make_functional_
    static inline const auto ev_fn = [](double e_core,
                                        const EvalState &state,
                                        const VecD &op,
                                        const VecZ &parameter_mapping,
                                        const VecD &gen_coeffs,
                                        const auto &graph,
                                        const VecD &params,
                                        mpi::Comm comm,
                                        const detail::LayerCosScale &cos_scale = {},
                                        const detail::LayerCosAccumulate & = {}) -> double {
        // cos_acc unused for the energy path; accepted so both functionals share one call arity in make_functional_.
        return ev(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm, cos_scale);
    };

    static inline const auto ev_and_grad_fn =
        [](double e_core,
           const EvalState &state,
           const VecD &op,
           const VecZ &parameter_mapping,
           const VecD &gen_coeffs,
           const auto &graph,
           const VecD &params,
           mpi::Comm comm,
           const detail::LayerCosScale &cos_scale = {},
           const detail::LayerCosAccumulate &cos_acc = {}) -> std::pair<double, VecD> {
        return ev_and_grad(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm, cos_scale, cos_acc);
    };

    static auto expected_num_params(const VecZ &parameter_mapping) -> size_t;

    template <typename Fn, typename R = std::invoke_result_t<Fn, const VecD &>>
    static auto make_parameter_validated_functional(size_t expected_num_params, Fn func)
        -> std::function<R(const VecD &)>;

    /// Distribute op_dict across ranks and apply this rank's share (shared impl of
    /// update_initial_operator); returns its new (terms, coeffs) so caches can refresh.
    auto apply_initial_operator_(const OperatorDict &op_dict) -> std::pair<MonomialList<NumModes>, VecD>;

    bool schrodinger_;
    mpi::Comm comm_; // real MPI across nodes, or an in-process comm across partitions
    CutoffFn<NumModes> cutoff_fn_;
    detail::MPOperator<NumModes> mp_op_;
    MPGraph graph_;
    // Per-gate layer-build scratch, reused across gates; carries no state between them.
    detail::MatchedEpochSet matched_scratch_;

    // A perf hint, never a correctness constraint: overflow spills losslessly. Sized to the cutoff's
    // structural position bound when it has one. Protected so derived classes size their rows the same.
    auto packed_inline_width_() const -> size_t;

private:
    unsigned int cutoff_;

    std::optional<double> lower_atol_, upper_atol_;
    double core_term_{0.0};

    size_t logical_num_modes_{NumModes};

    CutoffType cutoff_type_;
    std::optional<std::vector<VecZ>> basis_change_;

    // Operator basis (Majorana default / native Pauli), immutable after construction.
    Basis basis_{Basis::Majorana};

    // Intra-process partition runtime. Null ⇒ ordinary single-partition propagator; non-null ⇒ a partition facade
    // whose own mp_op_/graph_ are unused and every method fans out to the S partition propagators.
    std::unique_ptr<detail::partition::PartitionGroup<NumModes>> partition_group_;
    // PartitionGroup rebinds a cloned partition's comm_ to its own transport during a deep copy.
    friend class detail::partition::PartitionGroup<NumModes>;

    // Guard for the raw per-partition accessors (graph(), mp_op(), indexing(), graph_data()). A facade's
    // own graph_/mp_op_ are never populated, so handing them out would return plausible-looking empty
    // state; there is no meaningful merge either, since the callers want one partition's raw layout.
    // Throwing sends a C++ consumer to the partition-transparent accessors, or to partitions=1.
    auto require_single_partition_(const char *what) const -> void {
        if (partition_group_) {
            throw std::runtime_error(std::format("{} is not available on a multi-partition propagator: the facade owns "
                                                 "no operator or graph of its own. Use the partition-transparent "
                                                 "accessors (size(), graph_size(), evolved_operator_terms(), ...) or "
                                                 "construct with partitions=1.",
                                                 what));
        }
    }

    // Resolve the effective partition count (ctor arg 0 ⇒ env/auto). Returns 1 for the ordinary path.
    static auto resolve_partition_count_(size_t requested, mpi::Comm comm) -> size_t;
    // Fan-out helpers for the inline accessors (defined in the impl, where PartitionGroup is complete).
    auto partitioned_size_() const -> size_t;
    auto partitioned_graph_size_() const -> std::pair<size_t, size_t>;
    auto partitioned_graph_layers_() const -> size_t;
    auto partitioned_core_term_() const -> double; // core term is replicated on every partition; read partition 0
    auto partitioned_operator_memory_usage_() const -> detail::MPOperatorMemoryBreakdown<NumModes>;
    auto partitioned_graph_memory_usage_() const -> GraphMemoryBreakdown;

    // Partition fan-out vocabulary. Every one of these is facade-only: partition_group_ != nullptr is a
    // precondition, and all are defined in the impl, where PartitionGroup is complete.
    //
    // `for_each_partition_` / `map_partitions_` / `concat_partitions_` dispatch to the partitions' own pinned master
    // threads, which is mandatory for anything that touches partition state: the partitions trade over an
    // in-process comm and their collectives are barrier-synced, so all S must run together.
    // `fold_partitions_` / `sum_partitions_` / `first_partition_` instead read quiescent partitions straight from the
    // facade thread, which is safe precisely because they mutate nothing.

    // Run `fn` on every partition concurrently.
    auto for_each_partition_(const std::function<void(MonomialPropagator &)> &fn) -> void;

    // Run `fn` on every partition concurrently; one result per partition, in partition order.
    template <typename Fn, typename R = std::invoke_result_t<Fn &, MonomialPropagator &>>
    auto map_partitions_(Fn fn) -> std::vector<R>;

    // map_partitions_ with the per-partition sequences concatenated in partition order. The partitions are
    // disjoint, so the result enumerates the whole operator (deterministic for a fixed partition count).
    template <typename Fn, typename R = std::invoke_result_t<Fn &, MonomialPropagator &>>
    auto concat_partitions_(Fn fn) -> R;

    // Sequential fold of `proj(partition)` over the quiescent partitions; `accumulate(total, value)` combines.
    template <typename Proj, typename Accumulate, typename R = std::invoke_result_t<Proj &, const MonomialPropagator &>>
    auto fold_partitions_(Proj proj, Accumulate accumulate) const -> R;

    // fold_partitions_ for the additive breakdowns: the fields sum over the disjoint hash partitions.
    template <typename Proj, typename R = std::invoke_result_t<Proj &, const MonomialPropagator &>>
    auto sum_partitions_(Proj proj) const -> R;

    // Partition 0 for values that are not partitioned: the graph structure and gate info are identical on
    // every partition, the core (identity) term is replicated on all of them, and an allreduced scalar is
    // already global on each. Anything hash-partitioned must go through sum_/concat_partitions_ instead.
    auto first_partition_() const -> const MonomialPropagator &;

    /// Cosine-only index count across the active layers (see graph_size()).
    auto cos_index_count_() const -> size_t;

    // The shared tail of every update_* setter: apply `mutate` here, then replicate it to each partition.
    // Validation stays at the call site, so a rejected value throws before anything is mutated, and
    // the partitions do not re-validate what the facade already checked against identical field values.
    auto update_setting_(const std::function<void(MonomialPropagator &)> &mutate) -> void {
        mutate(*this);
        if (partition_group_) {
            for_each_partition_(mutate);
        }
    }

    auto regenerate_cutoff_fn_() -> void;

    /// Reject a (cutoff_type, basis_change) pair this algebra or system size cannot honour.
    auto validate_cutoff_config_(CutoffType cutoff_type, const std::optional<std::vector<VecZ>> &basis_change) const
        -> void;

    auto initialize_operator_caches_() -> void;

    auto current_picture_coeffs_() -> const VecD &;

    auto extend_coeffs_from_current_picture_if_needed_(VecD &coeffs) -> void;

    // Structural graph build recording per-layer gate info (no contraction).
    auto evolve_mode_build_graph_(const std::vector<VecZ> &majoranas,
                                  const VecZ &parameter_mapping,
                                  const VecD &gen_coeffs,
                                  const VecZ &gate_indices,
                                  int only_rotate_len_k) -> void;

    // Picture direction in one place: Heisenberg replays in reverse (size-1-i), Schrödinger forward.
    // Returns {build_angle, apply_angle}; apply is the build angle, negated in Schrödinger.
    auto gate_angle_(const VecD &mapped_params, size_t i, size_t majoranas_size) const -> std::pair<double, double> {
        const size_t idx = schrodinger_ ? i : majoranas_size - 1 - i;
        const double build_angle = mapped_params[idx];
        return {build_angle, schrodinger_ ? -build_angle : build_angle};
    }

    auto evolve_mode_graph_with_coeffs_(const std::vector<VecZ> &majoranas,
                                        const VecZ &parameter_mapping,
                                        const VecD &gen_coeffs,
                                        const VecZ &gate_indices,
                                        const VecD &parameters,
                                        const VecD &operator_coeffs,
                                        int only_rotate_len_k) -> void;

    auto evolve_mode_contract_immediately_(const std::vector<VecZ> &majoranas,
                                           const VecZ &parameter_mapping,
                                           const VecD &gen_coeffs,
                                           const VecD &parameters,
                                           int only_rotate_len_k) -> void;

    template <typename EvolutionFunc>
    auto run_gate_loop_(const std::vector<VecZ> &majoranas, int only_rotate_len_k, EvolutionFunc evolution_func)
        -> void;

    auto propagate_one_(const VecZ &gen_vec,
                        int only_rotate_len_k,
                        std::optional<std::reference_wrapper<const VecD>> coeffs = std::nullopt,
                        std::optional<double> param = std::nullopt,
                        size_t param_index = 0,
                        double gen_coeff = 0.0,
                        size_t gate_index = 0) -> void;

    // fused_scale_coeffs (ContractImmediately only): the picture's mutable coeff vector for the k==0 fused
    // cos sweep; the taken decision is reported via fused_scale so the apply matches. See build_layer.
    auto build_evolve_result_(const VecZ &gen_vec,
                              int only_rotate_len_k,
                              std::optional<std::reference_wrapper<const VecD>> coeffs = std::nullopt,
                              std::optional<double> param = std::nullopt,
                              CosMask *out_cos = nullptr,
                              detail::FusedContract *fused_contract = nullptr,
                              VecD *fused_scale_coeffs = nullptr,
                              bool *fused_scale = nullptr) -> std::shared_ptr<LayerCore>;

    template <typename Fn,
              typename R = std::invoke_result_t<Fn,
                                                double,
                                                const EvalState &,
                                                const VecD &,
                                                const VecZ &,
                                                const VecD &,
                                                const MPGraph &,
                                                const VecD &,
                                                mpi::Comm>>
    auto make_functional_(Fn &&func, std::optional<double> pare_threshold) -> std::function<R(const VecD &)>;

    // Reconstruct the optimizer-order (parameter_mapping, gen_coeffs) arrays from the layers' gate info.
    auto graph_gate_arrays_() const -> std::pair<VecZ, VecD>;

    // Replay `graph` over `coeffs`, recomputing each layer's cosine set from the inverted-index fold.
    auto evolve_operator_with_recompute_(VecD &&coeffs, const MPGraphView &graph, const VecD &params) -> VecD;
};

} // namespace monoprop

// These includes are here on purpose and should not be moved to the top
#include "monoprop/detail/monomial_propagator/MonomialPropagatorHelpers.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorImpl.h"
