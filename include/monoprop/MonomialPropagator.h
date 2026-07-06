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
#include <cstring>
#include <format>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <tbb/task_arena.h>

#include "monoprop/Evolution.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/Validation.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/profiling/RegionProfiler.h"
#include "monoprop/logging/QuillWrapper.h"
#include "monoprop/logging/Utils.h"

namespace monoprop {
namespace detail {
// Fused-contraction record sink (defined in layer_build/Common.h). build_evolve_result_ only takes a
// pointer, so a forward declaration keeps this header decoupled from the layer-build internals.
struct FusedContract;
} // namespace detail

template <size_t NumModes>
class MonomialPropagator {
public:
    MonomialPropagator(const FermiOperatorMap &initial_operator,
                       unsigned int cutoff,
                       const VecZ &slater_determinant,
                       std::optional<unsigned int> schrodinger_cutoff,
                       MPI_Comm comm,
                       std::optional<double> lower_atol = std::nullopt,
                       std::optional<double> upper_atol = std::nullopt,
                       CutoffType cutoff_type = CutoffType::Length,
                       std::optional<std::vector<VecZ>> basis_change = std::nullopt,
                       size_t logical_num_modes = NumModes);

    virtual ~MonomialPropagator() = default;

    // The simulator is an independent deep-copyable value. The implicit copy constructor does the
    // right thing: it deep-clones the per-rank operator store (MPOperator's copy ctor repairs the
    // index back-pointer) and shares the immutable graph layer cores via shared_ptr. The MPI
    // communicator handle is copied as-is (shared, not MPI_Comm_dup'd). No special members are
    // declared, so move stays implicit too; copy assignment is implicitly deleted (unique_ptr store).

    static constexpr auto num_modes{NumModes};
    static constexpr auto storage_num_modes{NumModes};

    auto logical_num_modes() const -> size_t { return logical_num_modes_; }

    /**
     * @brief Returns the size of the Operator
     *
     * Provides the number of Majorana operators in the Operator (local to this rank).
     * To get global size, use MPI allreduce.
     *
     * @return Size of the Operator on this rank
     */
    auto size() const -> size_t { return mp_op_.size(); }

    /**
     * @brief Pre-reserve operator storage for an expected final (this-rank) term count.
     *
     * Purely an allocation hint — it changes no results. The operator vector and index-map shards
     * otherwise grow by geometric doubling during evolution, so a run that ends at N terms pays a
     * sequence of serial multi-GB reallocations/rehashes (an Amdahl anchor in deferred_self_inserts)
     * and carries up to ~2× transient over-allocation at the largest doubling. Reserving once to a
     * known scale removes both. For an R-rank run, pass the per-rank estimate (≈ global / R), since
     * each rank stores only the terms it owns. Safe to call any time before/between evolution steps;
     * a smaller value than the current size is a no-op.
     */
    auto reserve_operator(size_t expected_local_terms) -> void {
        // Sizes BOTH the packed rows and the hash index to a known final per-rank count. Called on a
        // non-empty store between steps, so it only ever reserves capacity (width was fixed at setup).
        mp_op_.store->reserve(expected_local_terms);
    }

    /**
     * @brief Returns the size of the graph.
     * To get global size, use MPI allreduce.
     *
     * @return the number of indices and cycles in the MBS graph (local to this rank).
     */
    auto graph_size() const -> std::pair<size_t, size_t> { return graph_.num_cos_inds_and_cycles(); }

    /**
     * @brief Get the Majorana Branch Simulator graph (local to this rank).
     *
     * @return The MPGraph object representing the simulation graph for this rank.
     */
    auto graph() const -> const MPGraph & { return graph_; }

    // Direct access to this rank's MPOperator: the per-rank coefficient vectors (get_state /
    // get_operator), the packed term store, and the persistent even-parity inverted index. Used by tests
    // to drive get_pared_graph directly and fold the per-layer full cosine set.
    auto mp_op() -> detail::MPOperator<NumModes> & { return mp_op_; }
    auto mp_op() const -> const detail::MPOperator<NumModes> & { return mp_op_; }

    auto graph_memory_usage() const -> GraphMemoryBreakdown { return graph_.storage_memory_usage(); }

    auto operator_memory_usage() const -> detail::MPOperatorMemoryBreakdown<NumModes> {
        return detail::estimate_memory_usage(mp_op_);
    }

    auto print_object_memory_report(std::string_view label) const { print_object_memory_report_(label); }

    /**
     * @brief Get the number of evolved Majoranas (graph layers).
     *
     * @return The number of Majorana operators that have been evolved (number of graph layers).
     */
    auto graph_layers() const -> size_t { return graph_.layers(); }

    /**
     * @brief Number of gates ingested into the graph.
     *
     * Derived from the layers as max(gate_index) + 1 over the active layers (0 if empty),
     * so it stays correct after prefix layers are consumed by contract_partially/propagate.
     * A single-term gate expands to one layer; a multi-term gate expands to several layers
     * that share one gate index, so n_gates() <= graph_layers().
     *
     * @return The number of distinct gate indices recorded across the graph's layers.
     */
    auto n_gates() const -> size_t;

    /**
     * @brief The parameter mapping owned by the graph, in optimizer order.
     *
     * Entry i is the variational-parameter index driving the i-th graph layer (a generated
     * Majorana monomial); this is the mapping the graph uses when binding parameters.
     *
     * @return The per-layer parameter mapping (length equals graph_layers()).
     */
    auto parameter_mapping() const -> VecZ { return graph_gate_arrays_().first; }

    /**
     * @brief Re-wire which variational parameter drives each graph layer, in place.
     *
     * Relabels the graph layers' parameter indices without rebuilding the graph. The graph
     * structure depends only on the generators, not on the parameter labels, so this is a
     * cheap O(layers) relabel that changes only the parameter binding. Existing functionals
     * created by expectation_value_functional() keep the mapping they captured at creation
     * (they snapshot it), so rebuild a functional to pick up the new mapping.
     *
     * The mapping may be given at either granularity: per-layer (length graph_layers(), in
     * optimizer order) or per-gate (length n_gates(), indexed by absolute gate index). A
     * per-gate mapping is expanded to per-layer via each layer's stored gate index, which is
     * order-agnostic and correct in both pictures and across build_graph calls. When the two
     * lengths coincide (every gate is single-term in a single build) the per-layer reading is
     * used; note that across multiple Heisenberg builds optimizer order differs from gate
     * order even for single-term gates, so pass a per-gate mapping when tying by gate.
     *
     * @param parameter_mapping New parameter index per layer (length graph_layers()) or per
     *        gate (length n_gates()).
     */
    auto set_parameter_mapping(const VecZ &parameter_mapping) -> void;

    /**
     * @brief Access this rank's indexing map.
     *
     * Returns the mapping from Majorana bitset terms to their coefficient indices
     * for this rank.
     */
    auto indexing() -> detail::OperatorIndex<NumModes> & { return *mp_op_.store; }
    auto indexing() const -> const detail::OperatorIndex<NumModes> & { return *mp_op_.store; }

    /**
     * @brief Return graph layer data in Python-friendly structures.
     *
     * Provides a vector of tuples containing (cos_inds, local_cycles,
     * cross_rank_out, cross_rank_in) for every layer in the evolution graph.
     *
     * Storage format:
     * - local_cycles: Cycles (src, tgt, phase) where both indices are on this rank
     * - cross_rank_sin_send[rank]: (indices, dummy_phases) for the send recipe B^{(r')} on this rank
     * - cross_rank_sin_recv[rank]: (indices, signed_phases) for the apply recipe D^{(r')} (D- then D+)
     */
    using LocalCycleData = std::tuple<size_t, size_t, int>;
    using CrossRankData = std::tuple<VecZ, VecI>; // (indices, phases)
    using LayerData =
        std::tuple<VecZ, std::vector<LocalCycleData>, std::vector<CrossRankData>, std::vector<CrossRankData>>;
    auto graph_data() const -> std::vector<LayerData>;

    /**
     * @brief Updates the lower absolute tolerance.
     *
     * @param new_lower_atol New lower absolute tolerance (std::nullopt for no tolerance)
     */
    auto update_lower_atol(std::optional<double> new_lower_atol) -> void {
        if (upper_atol_.has_value() && new_lower_atol.has_value() && (new_lower_atol.value() > upper_atol_.value())) {
            throw std::runtime_error(
                std::format("New lower_atol ({}) must be less than or equal to current upper_atol ({}).",
                            new_lower_atol.value(),
                            upper_atol_.value()));
        }
        lower_atol_ = new_lower_atol;
    }

    /**
     * @brief Updates the upper absolute tolerance.
     *
     * @param new_upper_atol New upper absolute tolerance (std::nullopt for no tolerance)
     */
    auto update_upper_atol(std::optional<double> new_upper_atol) -> void {
        if (lower_atol_.has_value() && new_upper_atol.has_value() && new_upper_atol.value() < lower_atol_.value()) {
            throw std::runtime_error(
                std::format("New upper_atol ({}) must be greater than or equal to current lower_atol ({}).",
                            new_upper_atol.value(),
                            lower_atol_.value()));
        }
        upper_atol_ = new_upper_atol;
    }

    /**
     * @brief Updates the cutoff value and regenerates the cutoff function.
     *
     * This method updates the cutoff value and regenerates the cutoff function
     * using the stored cutoff type and basis change information.
     *
     * @param new_cutoff New cutoff value
     */
    auto update_cutoff(unsigned int new_cutoff) -> void {
        cutoff_ = new_cutoff;
        regenerate_cutoff_fn_();
    }

    /**
     * @brief Updates the cutoff type and regenerates the cutoff function.
     *
     * This method updates the cutoff type and regenerates the cutoff function
     * using the current cutoff value and stored basis change information.
     *
     * @param new_cutoff_type New cutoff type
     */
    auto update_cutoff_type(CutoffType new_cutoff_type) -> void {
        cutoff_type_ = new_cutoff_type;
        regenerate_cutoff_fn_();
    }

    /**
     * @brief Updates the basis change and regenerates the cutoff function.
     *
     * This method updates the basis transformation used by the cutoff function.
     * Setting basis_change to std::nullopt will disable basis transformation.
     *
     * @param new_basis_change New basis change vectors (std::nullopt for no basis change)
     */
    auto update_basis_change(std::optional<std::vector<VecZ>> new_basis_change) -> void {
        basis_change_ = new_basis_change;
        regenerate_cutoff_fn_();
    }

    /**
     * @brief Check if the simulation is in Schrodinger picture.
     *
     * @return True if in Schrodinger picture, false if in Heisenberg picture.
     */
    auto schrodinger() const -> bool { return schrodinger_; }

    /**
     * @brief Get the core term of the operator.
     *
     * @return The core term as a float.
     */
    auto core_term() const -> double { return core_term_; }

    /**
     * @brief Get the current cutoff value.
     *
     * @return The current cutoff value.
     */
    auto cutoff() const -> unsigned int { return cutoff_; }

    /**
     * @brief Get the current lower absolute tolerance.
     *
     * @return The current lower absolute tolerance (std::nullopt if not set).
     */
    auto lower_atol() const -> std::optional<double> { return lower_atol_; }

    /**
     * @brief Get the current upper absolute tolerance.
     *
     * @return The current upper absolute tolerance (std::nullopt if not set).
     */
    auto upper_atol() const -> std::optional<double> { return upper_atol_; }

    /**
     * @brief Get the current cutoff type.
     *
     * @return The current cutoff type.
     */
    auto cutoff_type() const -> CutoffType { return cutoff_type_; }

    /**
     * @brief Get the current basis change.
     *
     * @return The current basis change (std::nullopt if not set).
     */
    auto basis_change() const -> std::optional<std::vector<VecZ>> { return basis_change_; }

    /**
     * @brief Get the MPI communicator used by this simulator.
     *
     * @return The MPI_Comm associated with this simulator instance.
     */
    auto comm() const -> MPI_Comm { return comm_; }

    /**
     * @brief Build the propagation graph from a sequence of Majorana generators.
     *
     * Appends one graph layer per Majorana generator, recording each layer's gate
     * information (parameter_mapping[i] and gen_coeffs[i]) on the layer itself so that
     * evaluation later needs only the variational `parameters`. The graph accumulates
     * across successive calls.
     *
     * @param majoranas Majorana generators to apply (each a vector of indices).
     * @param parameter_mapping Per-generator index into the variational parameter vector.
     * @param gen_coeffs Per-generator coefficient g (angle = parameters[mapping[i]] * g).
     * @param gate_indices Optional per-generator gate index (which ingested gate each
     *        monomial belongs to), local and 0-based per call. Unlike parameter_mapping
     *        these are offset internally by the gate count already in the graph, so callers
     *        pass local indices and never track the running gate count. Omit (nullopt) for
     *        one gate per generator (iota); required to be contiguous runs from 0.
     * @param parameters Optional. When the graph is already non-empty and coefficient
     *        information is needed for atol-based truncation while extending it, provide
     *        the full parameter vector covering the existing graph *and* these new gates;
     *        the seed coefficients are regenerated internally by contracting the existing
     *        graph at `parameters` (there is no operator_coeffs input). Omit for a pure
     *        structural build.
     * @param only_rotate_len_k If > 0, apply gates to monomials of length <= k even if
     *        they anticommute (see class docs).
     *
     * @note In the Heisenberg picture gates are applied back-to-front, so each call
     *       consumes its sequence in reverse; splitting a circuit into forward chunks
     *       across calls is NOT equivalent to one call. In the Schrodinger picture gates
     *       are applied front-to-back, so a forward split IS equivalent.
     */
    auto build_graph(const std::vector<VecZ> &majoranas,
                     const VecZ &parameter_mapping,
                     const VecD &gen_coeffs,
                     std::optional<VecZ> gate_indices = std::nullopt,
                     std::optional<VecD> parameters = std::nullopt,
                     int only_rotate_len_k = 0) -> void;

    /**
     * @brief Evolve and contract immediately, without storing a propagation graph.
     *
     * Memory-efficient path: applies the gates with the given `parameters` directly to the
     * operator (Heisenberg) or state (Schrodinger) without retaining a graph.
     *
     * @param majoranas Majorana generators to apply.
     * @param parameter_mapping Per-generator index into `parameters`.
     * @param gen_coeffs Per-generator coefficient g.
     * @param parameters Variational parameter values.
     * @param only_rotate_len_k See build_graph.
     */
    auto propagate(const std::vector<VecZ> &majoranas,
                   const VecZ &parameter_mapping,
                   const VecD &gen_coeffs,
                   const VecD &parameters,
                   int only_rotate_len_k = 0) -> void;

    /**
     * @brief Compute the expectation value at the given variational parameters.
     *
     * Gate information (parameter mapping and generator coefficients) is owned by the
     * graph, so only `parameters` is required.
     */
    auto expectation_value(const VecD &parameters) -> double;

    /**
     * @brief Compute the expectation value and its gradient at the given parameters.
     */
    auto expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD>;

    /**
     * @brief Return a reusable callable computing the expectation value from parameters.
     *
     * @param pare_threshold Absolute-value cutoff for retaining edges in a masked execution
     *        plan built for this functional. std::nullopt disables paring (exact graph).
     */
    auto expectation_value_functional(std::optional<double> pare_threshold = std::nullopt)
        -> std::function<double(const VecD &)>;

    /**
     * @brief Return a reusable callable computing (expectation value, gradient) from parameters.
     *
     * @param pare_threshold See expectation_value_functional.
     */
    auto expectation_value_and_gradient_functional(std::optional<double> pare_threshold = std::nullopt)
        -> std::function<std::pair<double, VecD>(const VecD &)>;

    /**
     * @brief Contract the evolution graph into the operator/state at the given parameters.
     *
     * Applies every gate in the current graph with the supplied `parameters` (gate mapping
     * and generator coefficients are owned by the graph). In Heisenberg picture the gates are
     * contracted into the operator; in Schrodinger picture into the state.
     *
     * @param parameters Variational parameter values.
     * @param inplace If true, updates the simulator's internal state (consuming the graph);
     *                if false, returns the evolved coefficients without modifying state.
     * @return The evolved coefficients (evolved state in Schrodinger, evolved operator in
     *         Heisenberg). The core term is not included in the returned vector.
     */
    auto contract_partially(const VecD &parameters, bool inplace) -> VecD;

    virtual auto update_initial_operator(const FermiOperatorMap &op_dict) -> void { apply_initial_operator_(op_dict); }

protected:
    // FROZEN EXTENSION SURFACE. Everything in this `protected:` block (the ev/ev_and_grad callbacks,
    // the static utilities, apply_initial_operator_, the data members, packed_inline_width_) plus the
    // two `virtual` methods above exist for the out-of-tree subclass `MonomialPropagatorExtra` (no C++
    // definition lives in this repo — only in a downstream/private repo that builds against this
    // header). Do NOT change these signatures/layout without coordinating that repo; refactors must
    // delegate underneath them.
    // Reusable evaluation callbacks for make_functional_ / the pare functionals — also used by
    // MonomialPropagatorExtra.
    // The trailing cos_scale/cos_acc recompute the per-layer cosine set from the prepared fold and are
    // required for any evolving path (the "stored-cos" fallback no longer exists — no layer keeps its
    // cosine bitmap). ev_fn ignores cos_acc because the energy path has no reverse sweep.
    static inline const auto ev_fn = [](double e_core,
                                        const VecD &state,
                                        const VecD &op,
                                        const VecZ &parameter_mapping,
                                        const VecD &gen_coeffs,
                                        const auto &graph,
                                        const VecD &params,
                                        MPI_Comm comm,
                                        const detail::LayerCosScale &cos_scale = {},
                                        const detail::LayerCosAccumulate & = {}) -> double {
        // cos_acc is unused for the energy path (no reverse sweep); accepted so both functionals share
        // the same call arity in make_functional_.
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
           MPI_Comm comm,
           const detail::LayerCosScale &cos_scale = {},
           const detail::LayerCosAccumulate &cos_acc = {}) -> std::pair<double, VecD> {
        return ev_and_grad(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm, cos_scale, cos_acc);
    };

    // Static utility methods also needed by MonomialPropagatorExtra.
    static auto expected_num_params(const VecZ &parameter_mapping) -> size_t;

    template <typename Fn, typename R = std::invoke_result_t<Fn, const VecD &>>
    static auto make_parameter_validated_functional(size_t expected_num_params, Fn func)
        -> std::function<R(const VecD &)>;

    /**
     * @brief Distributes op_dict across ranks and applies it to this rank's operator.
     *
     * Shared implementation of update_initial_operator. Returns the new initial-operator terms
     * for this rank as a (Majorana terms, encoded coefficients) pair, so overrides that maintain
     * caches keyed on the initial operator can refresh them from the return value.
     */
    auto apply_initial_operator_(const FermiOperatorMap &op_dict) -> std::pair<MajoranaVector<NumModes>, VecD>;

    // Data members also needed by MonomialPropagatorExtra.
    bool schrodinger_;
    MPI_Comm comm_; // MPI communicator
    CutoffFn<NumModes> cutoff_fn_;
    detail::MPOperator<NumModes> mp_op_; // Single MPOperator for this MPI rank
    MPGraph graph_;              // Single MPGraph for this MPI rank
    // Persistent matched-follower scratch for the per-gate layer build (see MatchedEpochSet):
    // reused across gates so no per-gate O(operator) allocate+memset. Pure scratch — carries no
    // state between gates (each build bumps the epoch), so copies may share or reset it freely.
    detail::MatchedEpochSet matched_scratch_;

    // Inline-width hint for the packed operator rows. The store reserves this many Majorana
    // positions per row inline; terms with more positions spill losslessly into the overflow
    // arena, so this is purely a memory/perf hint and never a correctness constraint. When the
    // cutoff structurally bounds a surviving term's position count, we size rows to that bound
    // instead of the maximum; CutoffEvaluator owns the cutoff -> bound mapping (max_positions_bound,
    // which reports a bound only for the structural length/mode cutoffs and std::nullopt for an
    // arbitrary cutoff_fn — including a basis-changed cutoff, whose bound is on the mapped term,
    // not the stored one). In the Schrodinger picture the operator grows under a rule the cutoff
    // does not bound, so we keep the full width.
    // Protected so derived classes (MonomialPropagatorExtra) can size their operator identically.
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

    static auto format_bytes_(size_t bytes) -> std::string;

    auto print_memory_row_(std::string_view name, size_t local_bytes) const -> void;

    auto print_object_memory_report_(std::string_view label) const -> void;

    auto regenerate_cutoff_fn_() -> void;

    auto initialize_operator_caches_() -> void;

    auto current_picture_coeffs_() -> const VecD &;

    auto extend_coeffs_from_current_picture_if_needed_(VecD &coeffs) -> void;

    // Structural graph build recording per-layer gate info (no contraction).
    auto evolve_mode_build_graph_(const std::vector<VecZ> &majoranas,
                                  const VecZ &parameter_mapping,
                                  const VecD &gen_coeffs,
                                  const VecZ &gate_indices,
                                  int only_rotate_len_k) -> void;

    // Per-gate replay index + rotation angle, shared by the graph-with-coeffs and contract-immediately
    // drivers so the picture-direction logic lives in one place: Heisenberg replays gates in reverse
    // (majoranas_size-1-i), Schrödinger forward (i); the applied angle is negated in the Schrödinger
    // picture. Returns {build_angle (fed to the layer build), apply_angle (fed to the apply — the
    // build angle, negated in the Schrödinger picture)}.
    auto gate_angle_(const VecD &mapped_params, size_t i, size_t majoranas_size) const -> std::pair<double, double> {
        const size_t idx = schrodinger_ ? i : majoranas_size - 1 - i;
        const double build_angle = mapped_params[idx];
        return {build_angle, schrodinger_ ? -build_angle : build_angle};
    }

    // Graph build that also contracts into a running coeffs vector seeded by the regenerated seed
    // (used to inform atol truncation while extending a non-empty graph).
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

    /**
     * @brief Common function to propagate majoranas with timing
     */
    template <typename EvolutionFunc>
    auto propagate_with_timing_(const std::vector<VecZ> &majoranas, int only_rotate_len_k, EvolutionFunc evolution_func)
        -> void;

    /**
     * @brief Propagates the system by a single Majorana operator
     *
     * Applies the propagation corresponding to a single Majorana generator to the system,
     * updating the internal MP graph structure and the operator representation.
     *
     * @param gen_vec Vector of indices representing the Majorana generator
     * @param only_rotate_len_k Apply gates only to propagated monomials of exact length k (0 = no filtering).
     *                           Defaults to 0.
     * @param coeffs Optional coefficients for the propagation
     * @param param Optional propagation parameter used by cutoff-aware update paths.
     */
    auto propagate_one_(const VecZ &gen_vec,
                        int only_rotate_len_k,
                        std::optional<std::reference_wrapper<const VecD>> coeffs = std::nullopt,
                        std::optional<double> param = std::nullopt,
                        size_t param_index = 0,
                        double gen_coeff = 0.0,
                        size_t gate_index = 0) -> void;

    auto build_evolve_result_(const VecZ &gen_vec,
                              int only_rotate_len_k,
                              std::optional<std::reference_wrapper<const VecD>> coeffs = std::nullopt,
                              std::optional<double> param = std::nullopt,
                              CosMask *out_cos = nullptr,
                              detail::FusedContract *fused_contract = nullptr) -> std::shared_ptr<LayerCore>;

    /**
     * @brief Creates a functional (closure) for expectation value or gradient calculations.
     *
     * Derives the per-layer gate information (parameter mapping and generator coefficients)
     * from the graph, and uses the cached pared plan when one is present (see pare()).
     *
     * @tparam Fn Function type for evaluation (ev or ev_and_grad)
     * @param func The function to use for evaluation (ev or ev_and_grad)
     * @return Function object that computes expectation value or expectation_value+gradient for parameters
     */
    template <typename Fn,
              typename R = std::invoke_result_t<Fn,
                                                double,
                                                const VecD &,
                                                const VecD &,
                                                const VecZ &,
                                                const VecD &,
                                                const MPGraph &,
                                                const VecD &,
                                                MPI_Comm>>
    auto make_functional_(Fn &&func, std::optional<double> pare_threshold) -> std::function<R(const VecD &)>;

    // Reconstruct the optimizer-order (parameter_mapping, gen_coeffs) arrays from the gate
    // information owned by the graph layers. Provably identical to the arrays that used to be
    // supplied by callers, for both Heisenberg and Schrodinger pictures.
    auto graph_gate_arrays_() const -> std::pair<VecZ, VecD>;

    // Replay `graph` over `coeffs` recomputing each layer's cosine set from the persistent inverted index
    // fold (main-built layers no longer store the cos bitmap). Used by contract_partially.
    auto evolve_operator_with_recompute_(VecD &&coeffs, const MPGraphView &graph, const VecD &params) -> VecD;
};

} // namespace monoprop

// These includes are here on purpose and should not be moved to the top
#include "monoprop/detail/monomial_propagator/MonomialPropagatorHelpers.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorImpl.h"
