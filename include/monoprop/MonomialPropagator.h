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
#include <format>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "monoprop/Evolution.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/Validation.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/logging/QuillWrapper.h"
#include "monoprop/logging/Utils.h"

namespace monoprop {
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
                       size_t logical_num_modes = NumModes)
        : schrodinger_{schrodinger_cutoff.has_value()},
          comm_{comm},
          mp_op_{},
          graph_(schrodinger_cutoff.has_value()),
          cutoff_{cutoff},
          lower_atol_{lower_atol},
          upper_atol_{upper_atol},
          logical_num_modes_{logical_num_modes},
          cutoff_type_{cutoff_type},
          basis_change_{basis_change} {
        if (logical_num_modes_ == 0 || logical_num_modes_ > NumModes) {
            throw std::runtime_error(
                std::format("logical_num_modes ({}) must be in the range [1, {}].", logical_num_modes_, NumModes));
        }

        // Validate atol parameters
        if (upper_atol.has_value() && lower_atol.has_value() && (upper_atol.value() < lower_atol.value())) {
            throw std::runtime_error(std::format("upper_atol ({}) must be greater than or equal to lower_atol ({}).",
                                                 upper_atol.value(),
                                                 lower_atol.value()));
        }

        const size_t num_ranks = static_cast<size_t>(mpi::size(comm_));
        const size_t my_rank = static_cast<size_t>(mpi::rank(comm_));
        MajoranaVector<NumModes> local_heisenberg_terms;

        // convert the operator to the internal format
        double core_term = 0.0;
        for (const auto &[indices, coefficient] : initial_operator) {
            for (const auto &index : indices) {
                if (index >= 2 * logical_num_modes_) {
                    throw std::runtime_error(
                        std::format("Operator term contains an index greater than {}", 2 * logical_num_modes_));
                }
            }
            const auto majorana_bitset = indices_to_bitset<NumModes>(indices);
            const auto encoded_coeff = encode_coeff<NumModes>(coefficient, majorana_bitset);

            // Store the core term separately as it is orders of magnitude larger than the other terms
            if (indices.empty()) {
                core_term = encoded_coeff;
                continue;
            }
            if (my_rank == find_rank<NumModes>(majorana_bitset, num_ranks)) {
                mp_op_.init_op_map_[majorana_bitset] = encoded_coeff;
                local_heisenberg_terms.push_back(majorana_bitset);
            }
        }

        auto sc = schrodinger_cutoff.value_or(cutoff + 2);
        sc = std::min(sc, static_cast<unsigned int>(2 * logical_num_modes_));
        auto op =
            schrodinger_ ? generate_paired_op<NumModes>(sc / 2 + sc % 2, logical_num_modes_) : local_heisenberg_terms;

        const size_t expected_local_terms = std::max<size_t>(1, op.size() / std::max<size_t>(1, num_ranks));
        mp_op_.indexing.reset(threading::effective_parallelism());
        mp_op_.indexing.reserve(expected_local_terms);

        auto i = 0;
        for (const auto &maj : op) {
            if (my_rank == find_rank<NumModes>(maj, num_ranks)) {
                mp_op_.op.push_back(maj);
                mp_op_.indexing[maj] = i++;
            }
        }

        // Initialize this rank's MPOperator
        mp_op_.slater_determinant_ = slater_determinant;
        core_term_ = core_term;

        // initialize the cutoff function
        regenerate_cutoff_fn_();

        initialize_operator_caches_();
    }

    virtual ~MonomialPropagator() = default;

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
     * @brief Returns the size of the graph.
     * To get global size, use MPI allreduce.
     *
     * @return the number of indices and cycles in the MP graph (local to this rank).
     */
    auto graph_size() const -> std::pair<size_t, size_t> { return graph_.num_cos_inds_and_cycles(); }

    /**
     * @brief Get the Monomial Propagator graph (local to this rank).
     *
     * @return The MPGraph object representing the simulation graph for this rank.
     */
    auto graph() const -> const MPGraph & { return graph_; }

    auto graph_memory_usage() const -> GraphMemoryBreakdown { return graph_.storage_memory_usage(); }

    auto operator_memory_usage() const -> MPOperatorMemoryBreakdown<NumModes> { return estimate_memory_usage(mp_op_); }

    auto print_object_memory_report(std::string_view label) const -> void { print_object_memory_report_(label); }

    /**
     * @brief Get the number of evolved Majoranas (graph layers).
     *
     * @return The number of Majorana operators that have been evolved (number of graph layers).
     */
    auto graph_layers() const -> size_t { return graph_.layers(); }

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
     * @param parameter_mapping New per-layer parameter index in optimizer order; its length
     *        must equal graph_layers().
     */
    auto set_parameter_mapping(const VecZ &parameter_mapping) -> void;

    /**
     * @brief Access this rank's indexing map.
     *
     * Returns the mapping from Majorana bitset terms to their coefficient indices
     * for this rank.
     */
    auto indexing() -> ShardedIndexMap<NumModes> & { return mp_op_.indexing; }
    auto indexing() const -> const ShardedIndexMap<NumModes> & { return mp_op_.indexing; }

    /**
     * @brief Return graph layer data in Python-friendly structures.
     *
     * Provides a vector of tuples containing (cos_inds, local_cycles,
     * cross_rank_out, cross_rank_in) for every layer in the evolution graph.
     *
     * Storage format:
     * - local_cycles: Cycles (src, tgt, phase) where both indices are on this rank
     * - cross_rank_out[rank]: (indices, phases) for outgoing cycles to that rank
     * - cross_rank_in[rank]: (indices, phases) for incoming cycles from that rank
     */
    using LocalCycleData = std::tuple<size_t, size_t, int>;
    using CrossRankData = std::tuple<VecZ, VecI>; // (indices, phases)
    using LayerData =
        std::tuple<VecZ, std::vector<LocalCycleData>, std::vector<CrossRankData>, std::vector<CrossRankData>>;
    auto graph_data() const -> std::vector<LayerData> {
        std::vector<LayerData> layers;
        const auto num_layers = graph_.layers();
        layers.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            const auto traversal = graph_.get_layer_traversal(i);
            const size_t rank_count = traversal.cross_rank_rank_count();

            std::vector<LocalCycleData> local_cyc_data;
            local_cyc_data.reserve(traversal.local_cycle_count());
            traversal.for_each_local_cycle_range(0,
                                                 traversal.local_cycle_count(),
                                                 [&local_cyc_data](size_t, size_t src, size_t tgt, int phase) {
                                                     local_cyc_data.emplace_back(src, tgt, phase);
                                                 });

            std::vector<CrossRankData> out_data, in_data;
            out_data.reserve(rank_count);
            in_data.reserve(rank_count);
            for (size_t rank = 0; rank < rank_count; ++rank) {
                VecZ out_indices(traversal.cross_rank_out_size(rank));
                VecI out_phases(traversal.cross_rank_out_size(rank));
                VecZ in_indices(traversal.cross_rank_in_size(rank));
                VecI in_phases(traversal.cross_rank_in_size(rank));

                traversal.for_each_cross_rank_out_range(
                    rank,
                    0,
                    traversal.cross_rank_out_size(rank),
                    [&out_indices, &out_phases](size_t logical_idx, size_t value_idx, int phase) {
                        out_indices[logical_idx] = value_idx;
                        out_phases[logical_idx] = phase;
                    });
                traversal.for_each_cross_rank_in_range(
                    rank,
                    0,
                    traversal.cross_rank_in_size(rank),
                    [&in_indices, &in_phases](size_t logical_idx, size_t value_idx, int phase) {
                        in_indices[logical_idx] = value_idx;
                        in_phases[logical_idx] = phase;
                    });

                out_data.emplace_back(std::move(out_indices), std::move(out_phases));
                in_data.emplace_back(std::move(in_indices), std::move(in_phases));
            }
            layers.emplace_back(detail::expand_compressed_cosine_data(traversal.cos_data()),
                                std::move(local_cyc_data),
                                std::move(out_data),
                                std::move(in_data));
        }
        return layers;
    }

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
    auto propagate_build_graph(const std::vector<VecZ> &majoranas,
                               const VecZ &parameter_mapping,
                               const VecD &gen_coeffs,
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
     * @param only_rotate_len_k See propagate_build_graph.
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
    // Reusable evaluation callbacks for make_functional — also used by MonomialPropagatorExtra.
    static inline const auto ev_fn = [](double e_core,
                                        const VecD &state,
                                        const VecD &op,
                                        const VecZ &parameter_mapping,
                                        const VecD &gen_coeffs,
                                        const auto &graph,
                                        const VecD &params,
                                        MPI_Comm comm) -> double {
        return ev(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
    };

    static inline const auto ev_and_grad_fn = [](double e_core,
                                                 const VecD &state,
                                                 const VecD &op,
                                                 const VecZ &parameter_mapping,
                                                 const VecD &gen_coeffs,
                                                 const auto &graph,
                                                 const VecD &params,
                                                 MPI_Comm comm) -> std::pair<double, VecD> {
        return ev_and_grad(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
    };

    // Static utility methods also needed by MonomialPropagatorExtra.
    static auto append_to_graph(MPGraph &graph,
                                VecZ &cos_inds,
                                std::optional<CompressedCosineData> &compressed_cos_data,
                                SplitCycleResult &split,
                                MPI_Comm comm,
                                size_t param_index = 0,
                                double gen_coeff = 0.0) -> void;

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
    auto apply_initial_operator_(const FermiOperatorMap &op_dict) -> std::pair<MajoranaVector<NumModes>, VecD> {
        const size_t num_ranks = static_cast<size_t>(mpi::size(comm_));
        const size_t my_rank = static_cast<size_t>(mpi::rank(comm_));

        // Convert the input operator to the internal format and distribute terms to ranks
        FermiOperatorMap new_op;
        for (const auto &[ind, coeff] : op_dict) {
            const auto maj = indices_to_bitset<NumModes>(ind);
            if (ind.empty()) { // Core term, store in all
                core_term_ = encode_coeff<NumModes>(coeff, maj);
                continue;
            }
            if (my_rank == find_rank<NumModes>(maj, num_ranks)) {
                const auto maj_indices = bitset_to_indices<NumModes>(maj);
                new_op[maj_indices] = coeff;
            }
        }

        // Update this rank's operator
        auto res = mp_op_.update_initial_operator(new_op, schrodinger_);
        return std::move(std::get<2>(res));
    }

    // Data members also needed by MonomialPropagatorExtra.
    bool schrodinger_;
    MPI_Comm comm_; // MPI communicator
    CutoffFn<NumModes> cutoff_fn_;
    MPOperator<NumModes> mp_op_; // Single MPOperator for this MPI rank
    MPGraph graph_;              // Single MPGraph for this MPI rank

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
                                  int only_rotate_len_k) -> void;

    // Graph build that also contracts into a running coeffs vector seeded by operator_coeffs
    // (used to inform atol truncation while extending a non-empty graph).
    auto evolve_mode_graph_with_coeffs_(const std::vector<VecZ> &majoranas,
                                        const VecZ &parameter_mapping,
                                        const VecD &gen_coeffs,
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
                        double gen_coeff = 0.0) -> void;

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
};

} // namespace monoprop

// These includes are here on purpose and should not be moved to the top
#include "monoprop/detail/monomial_propagator/MonomialPropagatorHelpers.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorImpl.h"
