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

// Included only by MonomialPropagator.inl.

#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/Construction.h"
#include "monoprop/detail/sharded/Evaluation.h"
#include "monoprop/detail/sharded/RootObserver.h"
#include "monoprop/detail/sharded/State.h"
#include "monoprop/detail/sharded/Team.h"

/*
 * The sharded OpenMP root: MonomialPropagator over the T shard states of one rank.
 *
 * Ownership. The root owns configuration, the ordinary communicator of P ranks, the captured budget (T threads = T
 * shards), the prepared (P, T) router, the replicated identity coefficient, the initial-operator epoch, validity, and
 * this rank's T shard states (flat owners rank * T + t). It holds no operator, graph or matched marks of its own and
 * never dispatches to child propagators. Every operation that touches substantial shard state runs it on the shard's
 * owner in one operation-scoped team: seeding and copying (State.h), construction and propagation (Construction.h),
 * retained preparation, evaluation and replay (Evaluation.h), and the root's own owner phases (initial-operator
 * updates, remapping, picture copies). The caller performs only rank-level metadata work and small per-shard
 * view/callback setup before a team.
 *
 * Failure. Arguments are validated before any team. A seam returns its error after its team has joined, with
 * whether mutation began; the root transfers the mutation flag before rethrowing under run_operation_, so a
 * post-mutation failure invalidates this object (mutation_failed_) and reaches mpi::operation_failed() before any
 * result reduction, while a pre-mutation failure leaves it usable. There is no rollback.
 *
 * Ranks. Every rank constructs and calls the root collectively, with the same arguments. Rank-local results
 * (exports, contraction blocks, size() and the memory aggregates) cover this rank's shards; energies and gradients
 * are reduced over the communicator after the ascending-shard fold, with the identity once. Inside a team only the
 * primary, which is MPI's initializing thread, makes MPI calls, so MPI_THREAD_FUNNELED suffices.
 *
 * Launch contract (not checked): exactly T workers per team, OMP_DYNAMIC=FALSE and no limit below T, the same T on
 * every rank; public calls outside any OpenMP region. Checked: every MPI-using public call comes from MPI's
 * initializing thread (a wrong thread fails fast, locally), and initialized MPI provides at least FUNNELED.
 */

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
                                                 Basis basis)
    : MonomialPropagator(ObservedTag{},
                         nullptr,
                         initial_operator,
                         cutoff,
                         initial_state,
                         schrodinger_cutoff,
                         comm,
                         lower_atol,
                         upper_atol,
                         cutoff_type,
                         std::move(basis_change),
                         logical_num_modes,
                         basis) {}

template <size_t NumModes>
MonomialPropagator<NumModes>::MonomialPropagator(ObservedTag /*tag*/,
                                                 const detail::sharded::RootObserver *observer,
                                                 const OperatorDict &initial_operator,
                                                 unsigned int cutoff,
                                                 const VecZ &initial_state,
                                                 std::optional<unsigned int> schrodinger_cutoff,
                                                 mpi::Comm comm,
                                                 std::optional<double> lower_atol,
                                                 std::optional<double> upper_atol,
                                                 CutoffType cutoff_type,
                                                 std::optional<std::vector<VecZ>> basis_change,
                                                 size_t logical_num_modes,
                                                 Basis basis)
    : schrodinger_{schrodinger_cutoff.has_value()},
      comm_{comm},
      cutoff_{cutoff},
      lower_atol_{lower_atol},
      upper_atol_{upper_atol},
      logical_num_modes_{logical_num_modes},
      cutoff_type_{cutoff_type},
      basis_change_{std::move(basis_change)},
      basis_{basis},
      observer_{observer} {
    // The geometry is the launch's: P ranks of comm x the captured budget T. No argument or other environment variable
    // selects or checks it (the removed partition runtime's monoprop_PARTITIONS is never read).
    // Before any MPI query or collective in this constructor.
    mpi::require_initializing_thread();
    try {
        mpi::require_thread_support();
    }
    catch (...) {
        // A rank that threw here would strand its peers in the routing agreement below.
        mpi::operation_failed(comm, std::current_exception());
    }
    const auto ranks = static_cast<size_t>(mpi::size(comm));
    try {
        parallel_ = detail::parallel::capture_thread_budget();
    }
    catch (const std::invalid_argument &e) {
        throw PropagatorConfigError(e.what());
    }
    const auto threads = static_cast<size_t>(parallel_.threads);

    if (logical_num_modes_ == 0 || logical_num_modes_ > NumModes) {
        throw PropagatorConfigError(
            std::format("logical_num_modes ({}) must be in the range [1, {}].", logical_num_modes_, NumModes));
    }
    validate_cutoff_config_(cutoff_type_, basis_change_);
    if (upper_atol.has_value() && lower_atol.has_value() && (upper_atol.value() < lower_atol.value())) {
        throw PropagatorConfigError(std::format("upper_atol ({}) must be greater than or equal to lower_atol ({}).",
                                                upper_atol.value(),
                                                lower_atol.value()));
    }

    std::optional<detail::sharded::PairedBasisBounds> paired;
    if (schrodinger_) {
        // The basis is shared by the P * T flat owners.
        paired = detail::sharded::paired_basis_bounds(*schrodinger_cutoff, logical_num_modes_, ranks * threads);
        if (!paired->countable() || paired->share >= detail::OperatorIndex<NumModes>::kIndexCeiling) {
            const auto how_many =
                paired->countable() ? std::format("{}", paired->global_terms) : std::string("more than 2^64");
            const auto per_shard = paired->countable() ? std::format("{}", paired->share) : std::string("as many");
            throw PropagatorConfigError(
                std::format("schrodinger_cutoff ({}) admits {} paired basis terms over {} active modes, about {} per "
                            "shard -- more than can be walked or addressed. Lower schrodinger_cutoff, or raise the "
                            "rank x thread count P * T (currently {}).",
                            *schrodinger_cutoff,
                            how_many,
                            logical_num_modes_,
                            per_shard,
                            ranks * threads));
        }
    }

    // Routing mode and seed agree across the communicator (collective; every rank got here on replicated input). A
    // linear router then rejects a non-power-of-two P on every rank together.
    check_routing_agreement(comm_);
    world_ = detail::sharded::PhysicalWorld::of(comm_);
    router_.emplace(routing::make_router<NumModes>(ranks, threads));

    const double core_term =
        detail::sharded::validate_initial_operator<NumModes>(initial_operator, basis_, logical_num_modes_);
    // Before seeding: the packed-row width derives from the cutoff predicate.
    regenerate_cutoff_fn_();
    const auto seed = detail::sharded::OperatorSeed<NumModes>{
        .initial_operator = initial_operator,
        .initial_state = initial_state,
        .router = *router_,
        .basis = basis_,
        .logical_num_modes = logical_num_modes_,
        .paired = paired,
        .inline_width = detail::sharded::packed_inline_width<NumModes>(schrodinger_, cutoff_fn_)};
    shards_ = detail::sharded::seed_shards<NumModes>(parallel_,
                                                     seed,
                                                     world_.rank,
                                                     detail::sharded::RootShardObserver{observer_});
    core_term_ = core_term;
}

template <size_t NumModes>
MonomialPropagator<NumModes>::MonomialPropagator(const MonomialPropagator &other)
    : schrodinger_(checked_source_(other).schrodinger_),
      comm_(other.comm_),
      cutoff_fn_(other.cutoff_fn_),
      cutoff_(other.cutoff_),
      lower_atol_(other.lower_atol_),
      upper_atol_(other.upper_atol_),
      core_term_(other.core_term_),
      initial_operator_epoch_(other.initial_operator_epoch_),
      routing_coverage_reported_(other.routing_coverage_reported_),
      logical_num_modes_(other.logical_num_modes_),
      cutoff_type_(other.cutoff_type_),
      basis_change_(other.basis_change_),
      basis_(other.basis_),
      parallel_(other.parallel_),
      router_(other.router_),
      world_(other.world_),
      observer_(other.observer_),
      // The source's budget, not a new capture: a copy keeps T.
      shards_(detail::sharded::copy_shards<NumModes>(other.parallel_,
                                                     other.shards_,
                                                     detail::sharded::RootShardObserver{other.observer_})) {}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::mutation_failed_(std::exception_ptr error) -> void {
    invalidate_();
    mpi::operation_failed(comm_, std::move(error));
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sole_shard_(const char *what) const
    -> const detail::sharded::ShardState<NumModes> & {
    require_valid_();
    if (shards_.size() != 1) {
        throw MultiShardUnsupported(std::format(
            "{} is only available when this rank holds one shard, but it holds {} (one per thread of the "
            "budget captured at construction), and no single store represents the rank. Launch with "
            "monoprop_NUM_THREADS=1 (on every rank) to inspect the raw layout, or use the shard-transparent "
            "accessors (size(), graph_size(), the memory breakdowns, evolved_operator_terms(), ...).",
            what,
            shards_.size()));
    }
    return *shards_.front();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::sole_shard_(const char *what) -> detail::sharded::ShardState<NumModes> & {
    static_cast<void>(std::as_const(*this).sole_shard_(what)); // the checks
    return *shards_.front();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_size() const -> std::pair<size_t, size_t> {
    require_valid_();
    auto total = std::pair<size_t, size_t>{0, 0};
    for (const auto &state : shards_) {
        // May rebuild a stale inverted index (a lazy cache); construction leaves every index current.
        total.first += cos_index_count_of_(state->graph, state->op, basis_);
        total.second += state->graph.total_cycles();
    }
    return total;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_data() const -> std::vector<LayerData> {
    const auto &state = sole_shard_("graph_data()");
    return graph_data_of_(state.graph, state.op, basis_);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::n_gates() const -> size_t {
    require_valid_();
    // Layer metadata is identical on every shard.
    const MPGraph &graph = shards_.front()->graph;
    const size_t count = graph.layers();
    size_t max_gate = 0;
    bool any = false;
    for (size_t layer = 0; layer < count; ++layer) {
        const size_t g = graph.get_layer_traversal(layer).gate_index();
        max_gate = any ? std::max(max_gate, g) : g;
        any = true;
    }
    return any ? max_gate + 1 : 0;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_gate_arrays_() const -> std::pair<VecZ, VecD> {
    const MPGraph &graph = shards_.front()->graph;
    const size_t count = graph.layers();
    VecZ parameter_mapping(count);
    VecD gen_coeffs(count);
    // Layers store gate info in simulation order; the evaluation machinery expects optimizer order (the reverse).
    for (size_t layer = 0; layer < count; ++layer) {
        const auto traversal = graph.get_layer_traversal(layer);
        const size_t optimizer_index = count - 1 - layer;
        parameter_mapping[optimizer_index] = traversal.param_index();
        gen_coeffs[optimizer_index] = traversal.gen_coeff();
    }
    return {std::move(parameter_mapping), std::move(gen_coeffs)};
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::report_routing_coverage_(const std::vector<VecZ> &majoranas) -> void {
    if (routing_coverage_reported_) {
        return;
    }
    const routing::Router &router = *router_;
    if (!router.is_linear()) {
        // One rank (or splitmix): no rank subspace to fall short of.
        routing_coverage_reported_ = true;
        return;
    }
    std::vector<uint64_t> shifts;
    shifts.reserve(majoranas.size());
    for (const auto &gate : majoranas) {
        shifts.push_back(static_cast<uint64_t>(router.rank_shift<NumModes>(indices_to_bitset<NumModes>(gate))));
    }
    routing_coverage_reported_ = true;
    std::ranges::sort(shifts);
    shifts.erase(std::ranges::unique(shifts).begin(), shifts.end());
    const size_t span = routing::gf2_rank(shifts);
    if (span >= router.linear_bits()) {
        return;
    }
    const auto line =
        std::format("COMMROUTE rank={} linear_bits={} shift_rank={} shifts={} ranks_per_coset={} rank_cosets={}\n",
                    mpi::rank(comm_),
                    router.linear_bits(),
                    span,
                    shifts.size(),
                    size_t{1} << span,
                    router.ranks() >> span);
    std::fputs(line.c_str(), stderr);
    std::fflush(stderr);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::generators_(const std::vector<VecZ> &majoranas) const -> MonomialList<NumModes> {
    MonomialList<NumModes> generators;
    generators.reserve(majoranas.size());
    for (const auto &gate : majoranas) {
        generators.push_back(indices_to_bitset_checked<NumModes>(gate, 2 * logical_num_modes_));
    }
    return generators;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::construction_context_() const -> detail::sharded::ConstructionContext<NumModes> {
    return detail::sharded::ConstructionContext<NumModes>{.cutoff_fn = cutoff_fn_,
                                                          .router = *router_,
                                                          .lower_atol = lower_atol_,
                                                          .upper_atol = upper_atol_,
                                                          .basis = basis_,
                                                          .schrodinger = schrodinger_,
                                                          .world = world_,
                                                          .rounds = rounds_.get()};
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::finish_construction_(const detail::sharded::ConstructionOutcome &outcome,
                                                        bool &mutation_started) -> void {
    mutation_started = mutation_started || outcome.mutation_started;
    if (outcome.error) {
        std::rethrow_exception(outcome.error);
    }
}

template <size_t NumModes>
template <typename Body>
auto MonomialPropagator<NumModes>::run_owner_phase_(detail::sharded::RootWork work, Body &&body) -> std::exception_ptr {
    return detail::sharded::run_team(parallel_, [&](size_t shard, detail::sharded::TeamFailure &failure) noexcept {
        // The only phase: its decision ends the sequence either way.
        static_cast<void>(detail::sharded::phase(failure, shard, [&] {
            detail::sharded::notify_root(observer_, work, shard);
            body(shard);
        }));
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_graph(const std::vector<VecZ> &majoranas,
                                               const VecZ &parameter_mapping,
                                               const VecD &gen_coeffs,
                                               std::optional<VecZ> gate_indices,
                                               std::optional<VecD> parameters,
                                               std::optional<size_t> only_rotate_len_k) -> void {
    run_operation_(true, [&](bool &mutation_started) {
        validate_only_rotate_len_k_(only_rotate_len_k, 2 * logical_num_modes_);
        if (majoranas.empty()) {
            return;
        }
        validate_coefficient_lengths(parameter_mapping, gen_coeffs);
        validate_generators_(majoranas);
        VecZ local_gates;
        if (gate_indices.has_value()) {
            local_gates = *gate_indices;
        }
        else {
            local_gates.resize(majoranas.size());
            std::iota(local_gates.begin(), local_gates.end(), size_t{0});
        }
        validate_gate_indices(local_gates, majoranas.size());
        VecD mapped_params;
        // Replay angles of the existing layers, exactly as contract_partially() computes them, for the seed.
        std::optional<VecD> seed_params;
        if (parameters.has_value()) {
            validate_parameters_length(*parameters, parameter_mapping);
            if (graph_layers() > 0) {
                const auto [existing_mapping, existing_gen_coeffs] = graph_gate_arrays_();
                const size_t m = expected_num_params(existing_mapping);
                if (parameters->size() < m) {
                    throw SeedParametersTooShort(std::format(
                        "Coefficient-informed build_graph() needs at least {} parameter value(s) to replay the "
                        "existing {}-layer graph as a seed, but got {}.",
                        m,
                        graph_layers(),
                        parameters->size()));
                }
                const VecD prefix(parameters->begin(), parameters->begin() + static_cast<std::ptrdiff_t>(m));
                seed_params = schrodinger_ ? map_params(prefix, existing_mapping, existing_gen_coeffs, -1.0)
                                           : map_params(prefix, existing_mapping, existing_gen_coeffs, 1.0, true);
            }
            mapped_params = map_params(*parameters, parameter_mapping, gen_coeffs, 1.0);
        }
        const size_t gate_offset = n_gates();
        for (auto &g : local_gates) {
            g += gate_offset;
        }
        const auto generators = generators_(majoranas);
        report_routing_coverage_(majoranas);
        const auto ctx = construction_context_();
        const auto circuit = detail::sharded::GraphCircuit<NumModes>{.generators = generators,
                                                                     .parameter_mapping = parameter_mapping,
                                                                     .gen_coeffs = gen_coeffs,
                                                                     .gate_indices = local_gates,
                                                                     .only_rotate_len_k = only_rotate_len_k};
        const auto observer = detail::sharded::RootConstructionObserver{observer_};
        if (!parameters.has_value()) {
            finish_construction_(detail::sharded::build_graph<NumModes>(parallel_, shards_, ctx, circuit, observer),
                                 mutation_started);
            return;
        }
        const auto informed = detail::sharded::InformedCircuit<NumModes>{
            .graph = circuit,
            .mapped_params = mapped_params,
            .seed_params = seed_params ? std::optional<std::span<const double>>(*seed_params) : std::nullopt};
        finish_construction_(
            detail::sharded::build_graph_informed<NumModes>(parallel_, shards_, ctx, informed, observer),
            mutation_started);
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate(const std::vector<VecZ> &majoranas,
                                             const VecZ &parameter_mapping,
                                             const VecD &gen_coeffs,
                                             const VecD &parameters,
                                             std::optional<size_t> only_rotate_len_k) -> void {
    run_operation_(true, [&](bool &mutation_started) {
        validate_only_rotate_len_k_(only_rotate_len_k, 2 * logical_num_modes_);
        if (majoranas.empty()) {
            return;
        }
        validate_coefficient_lengths(parameter_mapping, gen_coeffs);
        validate_parameters_length(parameters, parameter_mapping);
        validate_generators_(majoranas);
        if (graph_layers() > 0) {
            throw GraphStateConflict(
                std::format("Cannot propagate() on top of a non-empty graph of {} layer(s): "
                            "propagate() evolves and contracts in place and assumes no stored graph. "
                            "Call contract_partially() to fold the existing graph first, or use "
                            "build_graph() to extend it.",
                            graph_layers()));
        }
        const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
        const auto generators = generators_(majoranas);
        report_routing_coverage_(majoranas);
        const auto ctx = construction_context_();
        const auto circuit = detail::sharded::PropagationCircuit<NumModes>{.generators = generators,
                                                                           .mapped_params = mapped_params,
                                                                           .only_rotate_len_k = only_rotate_len_k};
        finish_construction_(detail::sharded::propagate<NumModes>(parallel_,
                                                                  shards_,
                                                                  ctx,
                                                                  circuit,
                                                                  detail::sharded::RootConstructionObserver{observer_}),
                             mutation_started);
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::set_parameter_mapping(const VecZ &parameter_mapping) -> void {
    run_operation_(false, [&](bool &mutation_started) {
        const size_t count = graph_layers();
        const size_t gates = n_gates();
        const bool per_layer = parameter_mapping.size() == count;
        if (!per_layer && parameter_mapping.size() != gates) {
            throw GraphStateConflict(std::format("parameter_mapping has {} entries; expected {} (per graph "
                                                 "layer) or {} (per gate).",
                                                 parameter_mapping.size(),
                                                 count,
                                                 gates));
        }
        mutation_started = true;
        // The LayerCore is shared and immutable, so relabelling copies it and replaces the layer's core; each owner
        // copies its own shard's cores.
        const auto error = run_owner_phase_(detail::sharded::RootWork::remap, [&](size_t shard) {
            MPGraph &graph = shards_[shard]->graph;
            for (size_t layer = 0; layer < count; ++layer) {
                // Per-layer mapping in optimizer order; per-gate mapping indexed by absolute gate index.
                const size_t new_param_index = per_layer
                                                   ? parameter_mapping[count - 1 - layer]
                                                   : parameter_mapping[graph.get_layer_traversal(layer).gate_index()];
                auto &target = graph.get_layer(layer);
                auto new_core = std::make_shared<LayerCore>(target.core());
                new_core->param_index = new_param_index;
                if (const CosMask *pruned = target.pruned_cos()) {
                    target = Layer(std::move(new_core), *pruned);
                }
                else {
                    target = Layer(std::move(new_core));
                }
            }
        });
        if (error) {
            std::rethrow_exception(error);
        }
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::apply_initial_operator_(const OperatorDict &op_dict)
    -> std::pair<MonomialList<NumModes>, VecD> {
    return run_operation_(true, [&](bool &mutation_started) -> std::pair<MonomialList<NumModes>, VecD> {
        // Read-only pass on the caller: every index is checked and routed with the shards' own (P, T) ownership
        // before anything, the epoch included, changes. Each rank applies its own shards' share only.
        const routing::Router &router = *router_;
        const size_t first_local = world_.rank * shards_.size();
        std::vector<OperatorDict> shares(shards_.size());
        std::optional<double> new_core_term;
        for (const auto &[ind, coeff] : op_dict) {
            const auto mono = indices_to_bitset_checked<NumModes>(ind, 2 * logical_num_modes_);
            if (ind.empty()) { // the replicated identity is rank-level metadata
                new_core_term = algebra_encode_coeff<NumModes>(basis_, coeff, mono);
                continue;
            }
            const size_t owner = find_rank<NumModes>(mono, router);
            if (owner >= first_local && owner - first_local < shares.size()) {
                shares[owner - first_local][bitset_to_indices<NumModes>(mono)] = coeff;
            }
        }

        mutation_started = true;
        ++initial_operator_epoch_;
        if (new_core_term) {
            core_term_ = *new_core_term;
        }
        std::vector<std::pair<MonomialList<NumModes>, VecD>> updated(shards_.size());
        const auto error = run_owner_phase_(detail::sharded::RootWork::initial_operator, [&](size_t shard) {
            updated[shard] = shards_[shard]->op.update_initial_operator(shares[shard], schrodinger_);
        });
        if (error) {
            std::rethrow_exception(error);
        }
        // This rank's new (terms, coefficients), shard blocks in shard order.
        std::pair<MonomialList<NumModes>, VecD> merged;
        for (auto &[terms, coeffs] : updated) {
            merged.first.insert(merged.first.end(), terms.begin(), terms.end());
            merged.second.insert(merged.second.end(), coeffs.begin(), coeffs.end());
        }
        return merged;
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::prepare_retained_(std::optional<double> pare_threshold, bool &mutation_started)
    -> detail::sharded::RetainedEvaluation {
    auto [parameter_mapping, gen_coeffs] = graph_gate_arrays_();
    const auto context = detail::sharded::RetainedContext{.core_term = core_term_,
                                                          .parameter_mapping = std::move(parameter_mapping),
                                                          .gen_coeffs = std::move(gen_coeffs),
                                                          .pare_threshold = pare_threshold,
                                                          .basis = basis_,
                                                          .schrodinger = schrodinger_,
                                                          .rank = world_.rank};
    auto outcome = detail::sharded::prepare_retained<NumModes>(parallel_, shards_, context, observer_);
    mutation_started = mutation_started || outcome.mutation_started;
    if (outcome.error) {
        std::rethrow_exception(outcome.error);
    }
    return std::move(*outcome.retained);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evaluate_retained_(const detail::sharded::RetainedEvaluation &retained,
                                                      const VecD &params,
                                                      bool gradient,
                                                      bool *mutation_started) -> std::pair<double, VecD> {
    // Views of the retained snapshots and `params`, built for this call only. A failure up to and including the
    // evaluator's own argument checks happens before its team and mutates nothing.
    const auto requests = retained.requests(params);
    if (requests.size() == 1 && world_.ranks == 1 && observer_ == nullptr) {
        /*
         * One shard on one rank: nothing to exchange and no team to coordinate, so the sole shard goes to the
         * low-level serial evaluator (MPFunctions.cpp), the loop the sharded phases reduce to at T = 1, without
         * their per-step publication, staging and checkpoints. Bitwise the sharded result (sharded_root_tests:
         * sharded_root_sole_shard_serial_evaluation_matches_the_phases). A test observer keeps the phases, which it
         * observes.
         */
        detail::sharded::check_shard_request(requests.front());
        if (mutation_started != nullptr) {
            *mutation_started = true; // the evaluator warms the shard's lazy caches and the thread's scratch
        }
        try {
            if (gradient) {
                return ev_and_grad(requests.front(), comm_, retained.callbacks.front());
            }
            return {ev(requests.front(), comm_, retained.callbacks.front()), VecD{}};
        }
        catch (...) {
            if (mutation_started != nullptr) {
                throw; // the enclosing run_operation_ applies the failure policy once
            }
            mutation_failed_(std::current_exception());
        }
    }
    auto outcome = detail::sharded::evaluate_shards(requests,
                                                    retained.callbacks,
                                                    retained.options,
                                                    gradient,
                                                    observer_,
                                                    world_,
                                                    rounds_.get());
    // The team has run: owners warmed their lazy caches and per-thread scratch.
    if (mutation_started != nullptr) {
        *mutation_started = true;
    }
    try {
        if (outcome.error) {
            std::rethrow_exception(outcome.error);
        }
        detail::sharded::notify_root(observer_, detail::sharded::RootWork::combine, 0);
        // Ascending-shard fold, then the physical reduction, then the identity once (as ev_sharded()).
        const double local = detail::sharded::combine_contributions(outcome.contributions);
        const double expectation_value = mpi::allreduce_sum(local, comm_);
        VecD total_gradient;
        if (gradient) {
            total_gradient = detail::sharded::combine_gradients(outcome.gradients);
            if (!params.empty()) {
                mpi::allreduce_sum_inplace(total_gradient, comm_);
            }
        }
        return {retained.core_term + expectation_value, std::move(total_gradient)};
    }
    catch (...) {
        if (mutation_started != nullptr) {
            throw; // the enclosing run_operation_ applies the failure policy once
        }
        mutation_failed_(std::current_exception());
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_functional(std::optional<double> pare_threshold)
    -> std::function<double(const VecD &)> {
    return run_operation_(true, [&](bool &mutation_started) -> std::function<double(const VecD &)> {
        // Prepared once; every call reuses the snapshots. The callbacks borrow the shards' inverted indices, so the
        // callable must not outlive this propagator.
        auto retained = std::make_shared<const detail::sharded::RetainedEvaluation>(
            prepare_retained_(pare_threshold, mutation_started));
        const auto expected_epoch = initial_operator_epoch_;
        return [owner = this, expected_epoch, retained](const VecD &params) -> double {
            owner->enter_operation_(true);
            validate_expected_initial_operator(owner->initial_operator_epoch_, expected_epoch);
            retained->validate_call(params);
            return owner->evaluate_retained_(*retained, params, /*gradient=*/false, nullptr).first;
        };
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient_functional(std::optional<double> pare_threshold)
    -> std::function<std::pair<double, VecD>(const VecD &)> {
    using Functional = std::function<std::pair<double, VecD>(const VecD &)>;
    return run_operation_(true, [&](bool &mutation_started) -> Functional {
        auto retained = std::make_shared<const detail::sharded::RetainedEvaluation>(
            prepare_retained_(pare_threshold, mutation_started));
        const auto expected_epoch = initial_operator_epoch_;
        return [owner = this, expected_epoch, retained](const VecD &params) -> std::pair<double, VecD> {
            owner->enter_operation_(true);
            validate_expected_initial_operator(owner->initial_operator_epoch_, expected_epoch);
            retained->validate_call(params);
            return owner->evaluate_retained_(*retained, params, /*gradient=*/true, nullptr);
        };
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value(const VecD &parameters) -> double {
    return run_operation_(true, [&](bool &mutation_started) -> double {
        validate_functional_call(parameters, expected_num_params(graph_gate_arrays_().first));
        // A preparation team, then one evaluation team: two operation-scoped regions, independent of depth.
        const auto retained = prepare_retained_(std::nullopt, mutation_started);
        return evaluate_retained_(retained, parameters, /*gradient=*/false, &mutation_started).first;
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD> {
    return run_operation_(true, [&](bool &mutation_started) -> std::pair<double, VecD> {
        validate_functional_call(parameters, expected_num_params(graph_gate_arrays_().first));
        const auto retained = prepare_retained_(std::nullopt, mutation_started);
        return evaluate_retained_(retained, parameters, /*gradient=*/true, &mutation_started);
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::contract_blocks_(const VecD &parameters,
                                                    bool inplace,
                                                    bool &mutation_started,
                                                    VecD *concatenated) -> std::vector<VecD> {
    const size_t threads = shards_.size();
    const auto concatenate = [&](const std::vector<VecD> &blocks) {
        if (concatenated == nullptr) {
            return;
        }
        size_t total = 0;
        for (const auto &block : blocks) {
            total += block.size();
        }
        concatenated->clear();
        concatenated->reserve(total);
        for (const auto &block : blocks) {
            concatenated->insert(concatenated->end(), block.begin(), block.end());
        }
    };
    if (parameters.empty()) {
        // Nothing to replay: each owner copies its current picture (materializing pending entries, a lazy cache).
        mutation_started = true;
        std::vector<VecD> blocks(threads);
        const auto error = run_owner_phase_(detail::sharded::RootWork::picture, [&](size_t shard) {
            blocks[shard] = shards_[shard]->op.current_picture(schrodinger_);
        });
        if (error) {
            std::rethrow_exception(error);
        }
        concatenate(blocks);
        return blocks;
    }

    const auto [parameter_mapping, gen_coeffs] = graph_gate_arrays_();
    const size_t num_layers = parameter_mapping.size();
    // Heisenberg replays the oldest layers first in reverse optimizer order; Schrödinger replays the reversed window
    // of slice_view() at the negated angles, starting from the dense state.
    const VecD mapped_params = schrodinger_ ? map_params(parameters, parameter_mapping, gen_coeffs, -1.0)
                                            : map_params(parameters, parameter_mapping, gen_coeffs, 1.0, true);
    // From here the pictures and inverted indices (lazy caches) are read, and an in-place contraction consumes the
    // graphs.
    mutation_started = true;
    // An in-place contraction's sliced graphs own the consumed layers; they are bound here, before any view of them,
    // and outlive the team.
    std::vector<std::optional<MPGraph>> sliced(threads);
    std::vector<detail::sharded::ReplayRequest> requests;
    std::vector<detail::CosCallbacks> callbacks;
    requests.reserve(threads);
    callbacks.reserve(threads);
    for (size_t shard = 0; shard < threads; ++shard) {
        auto &state = *shards_[shard];
        const VecD &picture = state.op.current_picture(schrodinger_);
        const MPGraphView view = inplace
                                     ? sliced[shard].emplace(state.graph.slice_graph(num_layers, true)).replay_view()
                                     : state.graph.slice_view(num_layers);
        requests.push_back(detail::sharded::ReplayRequest{.coeffs = picture, .graph = view});
        callbacks.push_back(
            detail::make_cos_callbacks<NumModes>(state.op.inverted_index(), view, basis_, detail::parallel::Options{}));
    }
    auto outcome =
        detail::sharded::replay_shards(requests, mapped_params, callbacks, parallel_, observer_, world_, rounds_.get());
    if (outcome.error) {
        std::rethrow_exception(outcome.error);
    }
    concatenate(outcome.coeffs);
    if (inplace) {
        // Each block replaces its own shard's picture, which owns the rows the block is indexed by.
        for (size_t shard = 0; shard < threads; ++shard) {
            auto &op = shards_[shard]->op;
            (schrodinger_ ? op.state_coeffs : op.op_coeffs) = std::move(outcome.coeffs[shard]);
        }
        return {};
    }
    return std::move(outcome.coeffs);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::contract_partially(const VecD &parameters, bool inplace) -> VecD {
    return run_operation_(true, [&](bool &mutation_started) -> VecD {
        validate_parameters_length(parameters, graph_gate_arrays_().first);
        VecD concatenated;
        static_cast<void>(contract_blocks_(parameters, inplace, mutation_started, &concatenated));
        return concatenated;
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolved_operator_terms(const VecD &parameters, double atol)
    -> std::vector<std::pair<VecZ, std::complex<double>>> {
    using Term = std::pair<VecZ, std::complex<double>>;
    return run_operation_(true, [&](bool &mutation_started) -> std::vector<Term> {
        validate_parameters_length(parameters, graph_gate_arrays_().first);
        const auto blocks = contract_blocks_(parameters, false, mutation_started, nullptr);
        std::vector<Term> terms;
        // Each shard's index decodes that shard's own block; shards in ascending order.
        for (size_t shard = 0; shard < shards_.size(); ++shard) {
            detail::sharded::notify_root(observer_, detail::sharded::RootWork::export_block, shard);
            const VecD &evolved = blocks[shard];
            shards_[shard]->op.store->for_each([&](const auto &mono, size_t idx) {
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
        }
        return terms;
    });
}

} // namespace monoprop
