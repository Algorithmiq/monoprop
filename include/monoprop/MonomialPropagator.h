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
// Fused-contraction record sink (defined in layer_build/Common.h); forward-declared to stay decoupled.
struct FusedContract;
namespace shard {
// Intra-process shard runtime (defined in detail/shard/ShardGroup.h). Held by unique_ptr, so a forward
// declaration suffices; the ctor/copy-ctor/dtor that need the complete type are out-of-line in the impl.
template <size_t NumModes>
class ShardGroup;
} // namespace shard
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
                       size_t shards = 0);

    // Declared (not defaulted inline) because shard_group_ is a unique_ptr to an incomplete type here;
    // defined in the impl. Still virtual + effectively defaulted.
    virtual ~MonomialPropagator();

    // Deep-copyable value: the copy ctor (out-of-line) deep-clones the per-rank operator store (MPOperator's
    // copy ctor repairs the index back-pointer) and shares immutable graph cores via shared_ptr; a shard
    // facade clones its whole shard group. User-declared only because the shard_group_ unique_ptr would
    // otherwise delete it. Copy assignment is implicitly deleted (unique_ptr store).
    MonomialPropagator(const MonomialPropagator &other);
    auto operator=(const MonomialPropagator &) -> MonomialPropagator & = delete;
    // NOTE: the virtual destructor suppresses implicit moves, so a "move" selects the deep-cloning COPY
    // ctor (not a pointer steal). Fine today; to make moves O(1), default all four special members after
    // confirming MPOperator's move ctor repairs the index back-pointer.

    static constexpr auto num_modes{NumModes};
    static constexpr auto storage_num_modes{NumModes};

    auto logical_num_modes() const -> size_t { return logical_num_modes_; }

    /// @brief Number of Majorana operators in the operator, local to this rank (allreduce for global).
    auto size() const -> size_t { return shard_group_ ? sharded_size_() : mp_op_.size(); }

    /// @brief Number of (cosine-only indices, cycles) in the MBS graph, local to this rank (allreduce
    /// for global). "Cosine-only" = cos-scaled but not a rotation endpoint.
    auto graph_size() const -> std::pair<size_t, size_t> {
        return shard_group_ ? sharded_graph_size_() : std::pair{cos_index_count_(), graph_.total_cycles()};
    }

    /// @brief Get the Majorana Branch Simulator graph (local to this rank).
    auto graph() const -> const MPGraph & { return graph_; }

    auto mp_op() -> detail::MPOperator<NumModes> & { return mp_op_; }
    auto mp_op() const -> const detail::MPOperator<NumModes> & { return mp_op_; }

    // Memory breakdowns sum across shards on a facade (fields are additive over disjoint hash-partitions).
    auto graph_memory_usage() const -> GraphMemoryBreakdown {
        if (shard_group_) {
            return sharded_graph_memory_usage_();
        }
        return graph_.storage_memory_usage();
    }

    auto operator_memory_usage() const -> detail::MPOperatorMemoryBreakdown<NumModes> {
        if (shard_group_) {
            return sharded_operator_memory_usage_();
        }
        return detail::estimate_memory_usage(mp_op_);
    }

    /// @brief Number of evolved Majoranas (graph layers).
    auto graph_layers() const -> size_t { return shard_group_ ? sharded_graph_layers_() : graph_.layers(); }

    /// @brief Number of distinct gates forming the surrogate graph.
    /// A multi-term gate expands to several layers sharing one gate index, so n_gates() <= graph_layers().
    auto n_gates() const -> size_t;

    /// @brief The per-layer parameter mapping owned by the graph, in optimizer order (length = graph_layers()).
    auto parameter_mapping() const -> VecZ { return graph_gate_arrays_().first; }

    /// @brief Re-wire which variational parameter drives each graph layer, in place.
    ///
    /// Accepts per-layer (length graph_layers(), optimizer order) or per-gate (length n_gates(),
    /// expanded via each layer's stored gate index) granularity; when lengths coincide the per-layer
    auto set_parameter_mapping(const VecZ &parameter_mapping) -> void;

    /// @brief This rank's indexing map (Majorana bitset term → coefficient index). C++-only.
    auto indexing() -> detail::OperatorIndex<NumModes> & { return *mp_op_.store; }
    auto indexing() const -> const detail::OperatorIndex<NumModes> & { return *mp_op_.store; }

    /// @brief Return graph layer data as per-layer tuples (cos_inds, local_cycles, cross_rank_sin_send,
    /// cross_rank_sin_recv), local to this rank/shard.
    using LocalCycleData = std::tuple<size_t, size_t, int>;
    using CrossRankData = std::tuple<VecZ, VecI>; // (indices, phases)
    using LayerData =
        std::tuple<VecZ, std::vector<LocalCycleData>, std::vector<CrossRankData>, std::vector<CrossRankData>>;
    auto graph_data() const -> std::vector<LayerData>;

    /// @brief Update the lower absolute tolerance (std::nullopt for none).
    auto update_lower_atol(std::optional<double> new_lower_atol) -> void {
        if (upper_atol_.has_value() && new_lower_atol.has_value() && (new_lower_atol.value() > upper_atol_.value())) {
            throw std::runtime_error(
                std::format("New lower_atol ({}) must be less than or equal to current upper_atol ({}).",
                            new_lower_atol.value(),
                            upper_atol_.value()));
        }
        lower_atol_ = new_lower_atol;
        if (shard_group_) {
            for_each_shard_([&](MonomialPropagator &s) { s.update_lower_atol(new_lower_atol); });
        }
    }

    /// @brief Update the upper absolute tolerance (std::nullopt for none).
    auto update_upper_atol(std::optional<double> new_upper_atol) -> void {
        if (lower_atol_.has_value() && new_upper_atol.has_value() && new_upper_atol.value() < lower_atol_.value()) {
            throw std::runtime_error(
                std::format("New upper_atol ({}) must be greater than or equal to current lower_atol ({}).",
                            new_upper_atol.value(),
                            lower_atol_.value()));
        }
        upper_atol_ = new_upper_atol;
        if (shard_group_) {
            for_each_shard_([&](MonomialPropagator &s) { s.update_upper_atol(new_upper_atol); });
        }
    }

    /// @brief Update the cutoff value and regenerate the cutoff function.
    auto update_cutoff(unsigned int new_cutoff) -> void {
        cutoff_ = new_cutoff;
        regenerate_cutoff_fn_();
        if (shard_group_) {
            for_each_shard_([&](MonomialPropagator &s) { s.update_cutoff(new_cutoff); });
        }
    }

    /// @brief Update the cutoff type and regenerate the cutoff function.
    auto update_cutoff_type(CutoffType new_cutoff_type) -> void {
        validate_cutoff_config_(new_cutoff_type, basis_change_);
        cutoff_type_ = new_cutoff_type;
        regenerate_cutoff_fn_();
        if (shard_group_) {
            for_each_shard_([&](MonomialPropagator &s) { s.update_cutoff_type(new_cutoff_type); });
        }
    }

    /// @brief Update the basis change and regenerate the cutoff function (std::nullopt disables it).
    auto update_basis_change(std::optional<std::vector<VecZ>> new_basis_change) -> void {
        validate_cutoff_config_(cutoff_type_, new_basis_change);
        basis_change_ = new_basis_change;
        regenerate_cutoff_fn_();
        if (shard_group_) {
            for_each_shard_([&](MonomialPropagator &s) { s.update_basis_change(new_basis_change); });
        }
    }

    /// @brief Whether the simulation is in the Schrodinger picture (else Heisenberg).
    auto schrodinger() const -> bool { return schrodinger_; }

    /// @brief The operator basis: Majorana monomials or native Pauli strings.
    auto basis() const -> Basis { return basis_; }

    /// @brief The core term of the operator.
    auto core_term() const -> double { return shard_group_ ? sharded_core_term_() : core_term_; }

    /// @brief The current cutoff value.
    auto cutoff() const -> unsigned int { return cutoff_; }

    /// @brief The current lower absolute tolerance (std::nullopt if unset).
    auto lower_atol() const -> std::optional<double> { return lower_atol_; }

    /// @brief The current upper absolute tolerance (std::nullopt if unset).
    auto upper_atol() const -> std::optional<double> { return upper_atol_; }

    /// @brief The current cutoff type.
    auto cutoff_type() const -> CutoffType { return cutoff_type_; }

    /// @brief The current basis change (std::nullopt if unset).
    auto basis_change() const -> std::optional<std::vector<VecZ>> { return basis_change_; }

    /// @brief The MPI communicator (MPI_COMM_SELF for a shard-backed propagator, which uses an in-process ShmComm).
    auto comm() const -> MPI_Comm { return comm_.mpi; }

    /**
     * @brief Build the propagation graph from a sequence of Majorana generators, one layer per
     *        generator, recording each layer's gate info (angle = parameters[mapping[i]] * gen_coeffs[i]).
     *        The graph accumulates across calls.
     *
     * @param gate_indices Optional per-generator gate index, local and 0-based per call (offset
     *        internally by the gate count already in the graph). Omit for one gate per generator (iota).
     * @param parameters Optional; provide (covering the existing graph and these gates) to seed atol-based
     *        truncation while extending a non-empty graph. Omit for a pure structural build.
     * @param only_rotate_len_k If > 0, apply gates to monomials of length <= k even if they anticommute.
     *
     * @note Heisenberg applies gates back-to-front (each call consumes its sequence in reverse), so a
     *       forward split across calls is NOT equivalent; Schrodinger applies front-to-back, so it is.
     */
    auto build_graph(const std::vector<VecZ> &majoranas,
                     const VecZ &parameter_mapping,
                     const VecD &gen_coeffs,
                     std::optional<VecZ> gate_indices = std::nullopt,
                     std::optional<VecD> parameters = std::nullopt,
                     int only_rotate_len_k = 0) -> void;

    /// @brief Evolve and contract immediately, without storing a graph.
    /// Applies the gates at `parameters` directly to the operator (Heisenberg) or state (Schrodinger).
    auto propagate(const std::vector<VecZ> &majoranas,
                   const VecZ &parameter_mapping,
                   const VecD &gen_coeffs,
                   const VecD &parameters,
                   int only_rotate_len_k = 0) -> void;

    /// @brief Compute the expectation value at the given variational parameters (gate info owned by the graph).
    auto expectation_value(const VecD &parameters) -> double;

    /// @brief Compute the expectation value and its gradient at the given parameters.
    auto expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD>;

    /// @brief Return a reusable callable computing the expectation value from parameters.
    /// @param pare_threshold Edge-retention cutoff for a masked plan; std::nullopt disables paring (exact graph).
    auto expectation_value_functional(std::optional<double> pare_threshold = std::nullopt)
        -> std::function<double(const VecD &)>;

    /// @brief Return a reusable callable computing (expectation value, gradient) from parameters.
    /// @param pare_threshold See expectation_value_functional.
    auto expectation_value_and_gradient_functional(std::optional<double> pare_threshold = std::nullopt)
        -> std::function<std::pair<double, VecD>(const VecD &)>;

    /// @brief Contract the evolution graph into the operator (Heisenberg) or state (Schrodinger) at `parameters`.
    /// @param inplace If true, consume the graph and update internal state; if false, return the evolved
    ///        coefficients without modifying state (core term excluded from the returned vector).
    auto contract_partially(const VecD &parameters, bool inplace) -> VecD;

    /// @brief The full evolved operator as decoded (indices, coefficient) terms with |coeff| >= atol.
    /// Contracts at `parameters` (non-inplace); shard-transparent (gathers every shard's disjoint
    /// partition). Core term excluded (the Python binding adds it).
    auto evolved_operator_terms(const VecD &parameters, double atol)
        -> std::vector<std::pair<VecZ, std::complex<double>>>;

    virtual auto update_initial_operator(const OperatorDict &op_dict) -> void { apply_initial_operator_(op_dict); }

protected:
    // Reusable evaluation callbacks for make_functional_
    static inline const auto ev_fn = [](double e_core,
                                        const VecD &state,
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
           const VecD &state,
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

    /// @brief Distribute op_dict across ranks and apply it to this rank's operator (shared impl of
    /// update_initial_operator). Returns this rank's new (Majorana terms, encoded coeffs) so caches can refresh.
    auto apply_initial_operator_(const OperatorDict &op_dict) -> std::pair<MonomialList<NumModes>, VecD>;

    bool schrodinger_;
    mpi::Comm comm_; // communicator handle (real MPI across nodes, or in-process ShmComm across shards)
    CutoffFn<NumModes> cutoff_fn_;
    detail::MPOperator<NumModes> mp_op_; // Single MPOperator for this MPI rank
    MPGraph graph_;                      // Single MPGraph for this MPI rank
    // Persistent matched-follower scratch for the per-gate layer build (see MatchedEpochSet): reused
    // across gates (no per-gate allocate+memset). Pure scratch, carries no state between gates.
    detail::MatchedEpochSet matched_scratch_;

    // Inline-width hint for the packed operator rows (overflow spills losslessly, so it's a perf hint,
    // never a correctness constraint). Sized to the cutoff's structural position bound when it has one
    // (CutoffEvaluator::max_slot_bound; nullopt for arbitrary/basis-changed cutoffs and Schrodinger,
    // where we keep the full width). Protected so derived classes size their operator identically.
    auto packed_inline_width_() const -> size_t;

private:
    unsigned int cutoff_;

    // Evolution state
    std::optional<double> lower_atol_, upper_atol_;
    double core_term_{0.0};

    size_t logical_num_modes_{NumModes};

    // Store cutoff type and basis change for updating cutoff function
    CutoffType cutoff_type_;
    std::optional<std::vector<VecZ>> basis_change_;

    // Operator basis (Majorana default / native Pauli), immutable after construction.
    Basis basis_{Basis::Majorana};

    // Intra-process shard runtime. Null ⇒ ordinary single-partition propagator; non-null ⇒ a shard FACADE
    // whose own mp_op_/graph_ are unused and every method fans out to the S shard propagators. Constructed
    // only when the resolved shard count exceeds 1; requires a single MPI rank (shards don't nest with MPI).
    std::unique_ptr<detail::shard::ShardGroup<NumModes>> shard_group_;
    // ShardGroup rebinds a cloned shard's comm_ to its own ShmComm during a deep copy.
    friend class detail::shard::ShardGroup<NumModes>;

    // Resolve the effective shard count from the ctor `shards` arg (0 ⇒ env/auto), basis, thread budget,
    // and topology. Returns 1 for the ordinary path.
    static auto resolve_shard_count_(size_t requested, mpi::Comm comm) -> size_t;
    // Fan-out helpers for the inline accessors (defined in the impl, where ShardGroup is complete).
    auto sharded_size_() const -> size_t;
    auto sharded_graph_size_() const -> std::pair<size_t, size_t>;
    auto sharded_graph_layers_() const -> size_t;
    auto sharded_core_term_() const -> double; // core term is replicated on every shard; read shard 0
    // Sum the per-shard memory breakdowns (each shard owns a disjoint hash-partition, so fields add).
    auto sharded_operator_memory_usage_() const -> detail::MPOperatorMemoryBreakdown<NumModes>;
    auto sharded_graph_memory_usage_() const -> GraphMemoryBreakdown;
    // Run `fn` on every shard's propagator concurrently. Out-of-line because ShardGroup is incomplete here.
    auto for_each_shard_(const std::function<void(MonomialPropagator &)> &fn) -> void;

    /// @brief Cosine-only index count across the active layers (see graph_size()).
    auto cos_index_count_() const -> size_t;

    auto regenerate_cutoff_fn_() -> void;

    /// @brief Reject a (cutoff_type, basis_change) pair this algebra or system size cannot honour.
    /// Shared by the constructor and the update_* setters, which previously wrote straight through
    /// to regenerate_cutoff_fn_() and could install a configuration construction rejects.
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

    // Per-gate replay index + rotation angle (picture-direction logic in one place): Heisenberg replays
    // in reverse (size-1-i), Schrödinger forward (i) with the applied angle negated. Returns
    // {build_angle, apply_angle} (apply = build, negated in Schrödinger).
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

    /// @brief Common gate loop over majoranas, with timing.
    template <typename EvolutionFunc>
    auto run_gate_loop_(const std::vector<VecZ> &majoranas, int only_rotate_len_k, EvolutionFunc evolution_func)
        -> void;

    /// @brief Propagate the system by a single Majorana generator, updating the graph and operator.
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

    /// @brief Build a closure for expectation-value or gradient evaluation..
    template <typename Fn,
              typename R = std::invoke_result_t<Fn,
                                                double,
                                                const VecD &,
                                                const VecD &,
                                                const VecZ &,
                                                const VecD &,
                                                const MPGraph &,
                                                const VecD &,
                                                mpi::Comm>>
    auto make_functional_(Fn &&func, std::optional<double> pare_threshold) -> std::function<R(const VecD &)>;

    // Reconstruct the optimizer-order (parameter_mapping, gen_coeffs) arrays from the graph layers' gate info.
    auto graph_gate_arrays_() const -> std::pair<VecZ, VecD>;

    // Replay `graph` over `coeffs`, recomputing each layer's cosine set from the inverted-index fold. Used by
    // contract_partially.
    auto evolve_operator_with_recompute_(VecD &&coeffs, const MPGraphView &graph, const VecD &params) -> VecD;
};

} // namespace monoprop

// These includes are here on purpose and should not be moved to the top
#include "monoprop/detail/monomial_propagator/MonomialPropagatorHelpers.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorImpl.h"
