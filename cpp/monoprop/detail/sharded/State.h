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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/sharded/Team.h"

/*
 * Owner-initialized shard state for sharded execution.
 *
 * A rank-level propagator owns T shard states, one per worker of its fixed team. Shard t is the flat routing owner
 * rank * T + t at geometry (P, T). Its state is mathematical only: an operator store with its coefficients, state and
 * caches, a primary graph whose endpoints are rows of that store, and the layer build's matched marks. It is not a
 * propagator, communicator or executor.
 *
 * Memory ownership:
 *
 * - The caller allocates only the Shards pointer vector (T pointers) and the team's error slots.
 * - Each owner allocates its whole ShardState inside its protected phase: the state object, the packed rows, hash
 *   index and overflow map, coefficient/state vectors, pending initial-operator entries, inverted index, graph layer
 *   list and matched marks. The pointees never move, so their addresses are stable for the state's lifetime,
 *   including when the vector itself is moved.
 * - A copy deep-copies every mutable member; only the immutable graph LayerCore objects stay shared between the
 *   source's and the copy's layer lists.
 * - On failure, shards other owners already published are destroyed by the caller after the team has joined.
 *
 * Seeding is split into caller-side preparation, which may validate, throw configuration errors and query MPI, and
 * an owner-local seed that only reads the prepared inputs.
 */

namespace monoprop::detail::sharded {

/*!
 * \brief Size of the Schrödinger paired initial basis and its per-owner share.
 */
struct PairedBasisBounds {
    size_t max_pairs = 0;    //!< Pair bound: the effective Schrödinger cutoff halved, rounded up.
    size_t global_terms = 0; //!< Paired basis size; `SIZE_MAX` when too large to count.
    size_t share = 0;        //!< `global_terms` divided by the number of flat owners.

    //! Whether the basis size was countable at all.
    [[nodiscard]] auto countable() const noexcept -> bool { return global_terms != std::numeric_limits<size_t>::max(); }

    //! Row reservation for each owner's store; never zero.
    [[nodiscard]] auto reserve() const noexcept -> size_t { return std::max<size_t>(1, share); }
};

/*!
 * \brief Bound the streamed paired basis for a Schrödinger cutoff.
 *
 * Pure. The cutoff is first capped at `2 * logical_num_modes`, then rounded up to whole pairs, so an odd cutoff
 * admits one more pair than it covers. Rejecting an unaddressable share is the caller's configuration policy.
 *
 * \param schrodinger_cutoff The configured Schrödinger cutoff.
 * \param logical_num_modes  Active logical modes.
 * \param flat_owners        Flat owners P * T sharing the basis; 0 is treated as 1.
 * \return The pair bound, global size and per-owner share.
 */
inline auto paired_basis_bounds(unsigned int schrodinger_cutoff, size_t logical_num_modes, size_t flat_owners)
    -> PairedBasisBounds {
    const auto effective = std::min(schrodinger_cutoff, static_cast<unsigned int>(2 * logical_num_modes));
    auto bounds = PairedBasisBounds{};
    bounds.max_pairs = (effective / 2) + (effective % 2);
    bounds.global_terms = paired_op_size(bounds.max_pairs, logical_num_modes);
    bounds.share = bounds.global_terms / std::max<size_t>(1, flat_owners);
    return bounds;
}

/*!
 * \brief Packed-row inline width for a new store.
 *
 * A performance hint, never a correctness constraint: longer rows spill losslessly to the overflow map.
 * Heisenberg uses the cutoff's structural position bound, capped at the store's maximum; Schrödinger and cutoffs
 * without a bound use the store's default.
 *
 * \param schrodinger Whether the propagator evolves the state.
 * \param cutoff_fn   The propagator's cutoff predicate.
 * \return The width to construct each owner's OperatorIndex with.
 */
template <size_t NumModes>
auto packed_inline_width(bool schrodinger, const CutoffFn<NumModes> &cutoff_fn) -> size_t {
    constexpr auto max_width = OperatorIndex<NumModes>::kMaxInlinePositions;
    constexpr auto default_width = OperatorIndex<NumModes>::kDefaultInlinePositions;
    if (schrodinger) {
        return default_width;
    }
    // Already in physical slots (CutoffEvaluator::max_slot_bound), so nothing to scale.
    const auto bound = CutoffEvaluator<NumModes>(cutoff_fn).max_slot_bound();
    if (!bound) {
        return default_width;
    }
    return std::min<size_t>(*bound, max_width);
}

/*!
 * \brief Caller-side check of an initial operator; returns its encoded identity coefficient.
 *
 * Converts every term within the logical-mode bound and encodes its coefficient, in the dictionary's order, so the
 * first invalid term throws before any owner starts. The identity coefficient is rank-level metadata: it is seeded
 * into no shard.
 *
 * \param initial_operator  The operator dictionary.
 * \param basis             The coefficient encoding.
 * \param logical_num_modes Active logical modes; indices must lie below `2 * logical_num_modes`.
 * \return The encoded identity coefficient, or 0 when the operator has no identity term.
 * \throws As indices_to_bitset_checked() for an out-of-range index, and as algebra_encode_coeff() for a coefficient
 *         the basis cannot encode.
 */
template <size_t NumModes>
auto validate_initial_operator(const OperatorDict &initial_operator, Basis basis, size_t logical_num_modes) -> double {
    auto core_term = 0.0;
    for (const auto &[indices, coefficient] : initial_operator) {
        const auto mono = indices_to_bitset_checked<NumModes>(indices, 2 * logical_num_modes);
        const auto encoded = algebra_encode_coeff<NumModes>(basis, coefficient, mono);
        if (indices.empty()) {
            core_term = encoded;
        }
    }
    return core_term;
}

/*!
 * \brief Read-only, caller-prepared inputs for seeding every owner of one propagator.
 *
 * Built once on the caller after validate_initial_operator() and any MPI-dependent configuration (router geometry,
 * routing agreement). Owners read it concurrently; nothing here is mutated during seeding.
 */
template <size_t NumModes>
struct OperatorSeed {
    const OperatorDict &initial_operator;    //!< Validated operator; must outlive seeding.
    const VecZ &initial_state;               //!< Copied into every owner's store.
    routing::Router router;                  //!< Flat-owner routing at geometry (P, T).
    Basis basis;                             //!< Coefficient encoding and state scoring.
    size_t logical_num_modes;                //!< Active logical modes.
    std::optional<PairedBasisBounds> paired; //!< Schrödinger's streamed paired basis; empty for Heisenberg.
    size_t inline_width;                     //!< From packed_inline_width().
};

/*!
 * \brief Seed one owner's operator: its routed share of the initial rows, pending entries and warm caches.
 *
 * Owner-local: reads `seed` and allocates only into `op` plus transient owner-local lists, so it may run concurrently
 * for different owners. Rows are this owner's keys in input order (Heisenberg) or in the paired basis's emission order
 * (Schrödinger), numbered 0, 1, 2, ... as they are kept, so fixed geometry reproduces the same rows. The paired basis
 * is streamed, never materialized. Heisenberg owners reserve their own term count; Schrödinger owners reserve the
 * basis's per-owner share. The identity term is skipped here: its coefficient is rank-level metadata, while
 * Schrödinger's identity basis row is an ordinary routed row with coefficient 0.
 *
 * The operator is filled where it lives rather than returned: an MPOperator is about 112 NumModes bytes (its inverted
 * index holds its columns inline), and a copy in a worker's frame would stay resident on every worker's stack.
 *
 * \param seed       Prepared inputs; the operator must have passed validate_initial_operator().
 * \param flat_owner This owner's flat routing slot, `rank * T + shard`.
 * \param op         A default-constructed operator; on return it is seeded, with the sparse state warmed for
 *                   Heisenberg and the dense live state for Schrödinger, then the inverted index.
 */
template <size_t NumModes>
auto seed_operator(const OperatorSeed<NumModes> &seed, size_t flat_owner, MPOperator<NumModes> &op) -> void {
    op.basis = seed.basis;
    const auto owns = [&](const Monomial<NumModes> &mono) {
        return seed.router.template dest<NumModes>(mono) == flat_owner;
    };

    MonomialList<NumModes> heisenberg_rows;
    for (const auto &[indices, coefficient] : seed.initial_operator) {
        if (indices.empty()) {
            continue;
        }
        const auto mono = indices_to_bitset_checked<NumModes>(indices, 2 * seed.logical_num_modes);
        if (!owns(mono)) {
            continue;
        }
        op.init_op_map[mono] = algebra_encode_coeff<NumModes>(seed.basis, coefficient, mono);
        if (!seed.paired) {
            heisenberg_rows.push_back(mono);
        }
    }

    const auto reserve = seed.paired ? seed.paired->reserve() : std::max<size_t>(1, heisenberg_rows.size());
    op.store = std::make_unique<OperatorIndex<NumModes>>(seed.inline_width);
    op.store->reserve(reserve);
    // Store replaced: drop any lazy inverted index so it rebuilds against the new store.
    op.inverted_index_.reset();

    // The kept monomials are distinct, so emplace (insert-if-absent) is an assigning insert. A row index is a
    // position in the kept subsequence, so the enumeration order is load-bearing.
    size_t row = 0;
    const auto keep_if_owned = [&](const Monomial<NumModes> &mono) {
        if (owns(mono)) {
            op.append_term(mono);
            op.store->emplace(mono, row++);
        }
    };
    if (seed.paired) {
        // Streamed: every owner walks the basis concurrently, so a materialized list would hold P * T copies.
        for_each_paired_monomial<NumModes>(seed.paired->max_pairs, seed.logical_num_modes, keep_if_owned);
    }
    else {
        for (const auto &mono : heisenberg_rows) {
            keep_if_owned(mono);
        }
    }

    op.initial_state = seed.initial_state;
    op.initialize_caches(seed.paired.has_value());
}

/*!
 * \brief One owner's mutable mathematical state.
 *
 * Owned through Shards at a stable address and mutated only by its owner during ordinary phases. Copying
 * deep-copies the store (OperatorIndex::clone()), coefficients, pending entries, sparse/dense state, inverted index,
 * layer list and matched marks; the layers' immutable LayerCore objects stay shared.
 */
template <size_t NumModes>
struct ShardState {
    MPOperator<NumModes> op; //!< Store, coefficients, state and lazy caches of this shard.
    MPGraph graph;           //!< Primary graph; its endpoints are rows of `op`.
    MatchedEpochSet matched; //!< Layer-build follower marks, indexed by rows of `op`.

    /*!
     * \brief Adopt an operator with an empty graph.
     * \param op          The operator, moved in.
     * \param schrodinger Whether the graph records Schrödinger layers.
     */
    ShardState(MPOperator<NumModes> op, bool schrodinger) : op(std::move(op)), graph(schrodinger) {}

    /*!
     * \brief An empty operator built in place, with an empty graph; seed_operator() then fills it where it lives.
     * \param schrodinger Whether the graph records Schrödinger layers.
     */
    explicit ShardState(bool schrodinger) : graph(schrodinger) {}

    ShardState(const ShardState &) = default;                    //!< Deep copy; graph cores stay shared.
    auto operator=(const ShardState &) -> ShardState & = delete; //!< States are replaced, not assigned.
    ShardState(ShardState &&) = delete;                          //!< Held at a stable address.
    auto operator=(ShardState &&) -> ShardState & = delete;      //!< Held at a stable address.
    ~ShardState() = default;                                     //!< Releases every owned allocation.

    //! Rows in this shard's store.
    [[nodiscard]] auto size() const -> size_t { return op.size(); }

    //! This shard's operator accounting, including its matched marks.
    [[nodiscard]] auto operator_memory_usage() const -> MPOperatorMemoryBreakdown<NumModes> {
        auto breakdown = estimate_memory_usage(op);
        breakdown.matched_scratch_bytes = matched.memory_bytes();
        return breakdown;
    }

    //! This shard's graph accounting; shared cores are counted by every graph that holds them.
    [[nodiscard]] auto graph_memory_usage() const -> GraphMemoryBreakdown { return graph.storage_memory_usage(); }
};

//! The shard states of one rank-level propagator, indexed by shard; entries are never null once built.
template <size_t NumModes>
using Shards = std::vector<std::unique_ptr<ShardState<NumModes>>>;

//! The substantial work an owner performs in seed_shards() or copy_shards(), as reported to an observer.
enum class ShardWork : std::uint8_t {
    seed, //!< seed_operator() and the new state.
    copy, //!< Copy construction from the source shard.
};

/*!
 * \brief The observer production code uses: it does nothing and compiles away.
 *
 * seed_shards() and copy_shards() call `begin(work, shard)` on the owner inside its protected phase before any of
 * the shard's substantial allocation, and `end(work, shard)` after the state is complete and before it is
 * published. Tests substitute an observer that records the executing worker or arms a failure; no production caller
 * passes one. An exception thrown by an observer is that owner's failure.
 */
struct NoShardObserver {
    auto begin(ShardWork /*work*/, size_t /*shard*/) const noexcept -> void {}
    auto end(ShardWork /*work*/, size_t /*shard*/) const noexcept -> void {}
};

/*!
 * \brief Build T shard states, each allocated and initialized on its owner worker.
 *
 * Allocates the pointer vector on the caller, then runs one team of `options.threads` workers in which worker t
 * calls `initialize(t)` inside a protected phase and publishes the result in slot t. The initializer is invoked
 * concurrently from every worker on the same object; it must perform only owner-local initialization (no MPI, no
 * synchronization, no mutation of shared state). Configuration, routing and any distributed failure guard belong to
 * the caller, and this helper owns no MPI requests.
 *
 * \param options    The already-captured budget; its thread count is T. The environment is not read.
 * \param initialize Callable as `initialize(size_t shard)`, returning a non-null
 *                   `std::unique_ptr<ShardState<NumModes>>`.
 * \return T non-null states, in shard order.
 * \throws std::invalid_argument if `options.threads` is not positive, before anything is allocated.
 * \throws std::logic_error if an initializer returns a null state.
 * \throws The original exception of the lowest-numbered failing owner, after the team has joined. States other owners
 *         had published are then destroyed on the caller; a failing initializer's own locals unwind inside it.
 */
template <size_t NumModes, class Initialize>
    requires std::is_same_v<std::invoke_result_t<Initialize &, size_t>, std::unique_ptr<ShardState<NumModes>>>
auto make_shards(parallel::Options options, Initialize &&initialize) -> Shards<NumModes> {
    // Checked before the negative value could become a container size.
    if (options.threads < 1) {
        throw std::invalid_argument(
            std::format("sharded::make_shards: the team size must be positive, got {}", options.threads));
    }
    auto shards = Shards<NumModes>(static_cast<size_t>(options.threads));
    const auto error = run_team(options, [&](size_t shard, TeamFailure &failure) noexcept {
        // The only phase: its decision ends the sequence either way.
        static_cast<void>(phase(failure, shard, [&] {
            auto state = std::invoke(initialize, shard);
            if (!state) {
                throw std::logic_error(
                    std::format("sharded::make_shards: the initializer for shard {} returned no state", shard));
            }
            shards[shard] = std::move(state);
        }));
    });
    if (error) {
        std::rethrow_exception(error);
    }
    return shards;
}

/*!
 * \brief Seed one rank's T shards from caller-prepared inputs, each on its owner.
 *
 * Shard t seeds flat owner `rank * T + t` through seed_operator(). An empty share still yields a valid empty state,
 * and its owner still participates in the team.
 *
 * \param options  The already-captured budget; its thread count T must equal `seed.router.partitions()`.
 * \param seed     Prepared inputs; see OperatorSeed.
 * \param rank     This process's rank, below `seed.router.ranks()`.
 * \param observer Test-only seam; see NoShardObserver.
 * \return This rank's T seeded states.
 * \throws std::invalid_argument if T is not positive, disagrees with the router's shard count, or `rank` is out of
 *         range, before the team starts.
 * \throws As make_shards() for owner failures.
 */
template <size_t NumModes, class Observer = NoShardObserver>
auto seed_shards(parallel::Options options,
                 const OperatorSeed<NumModes> &seed,
                 size_t rank,
                 const Observer &observer = {}) -> Shards<NumModes> {
    if (options.threads < 1 || seed.router.partitions() != static_cast<size_t>(options.threads)
        || rank >= seed.router.ranks()) {
        throw std::invalid_argument(std::format("sharded::seed_shards: {} shards on rank {} do not fit routing "
                                                "geometry ({} ranks, {} shards per rank)",
                                                options.threads,
                                                rank,
                                                seed.router.ranks(),
                                                seed.router.partitions()));
    }
    const auto threads = static_cast<size_t>(options.threads);
    return make_shards<NumModes>(options, [&](size_t shard) {
        observer.begin(ShardWork::seed, shard);
        // Built on the heap and seeded in place: no operator passes through this worker's frame.
        auto state = std::make_unique<ShardState<NumModes>>(seed.paired.has_value());
        seed_operator(seed, (rank * threads) + shard, state->op);
        observer.end(ShardWork::seed, shard);
        return state;
    });
}

/*!
 * \brief Copy T shards, each on the owner of the destination shard.
 *
 * Owner t deep-copies `source[t]` only. The source must not be mutated during the call; validity of the object that
 * owns it is the caller's check, made before calling. The copy has the source's shard count, so a copy keeps the
 * source's captured budget rather than resolving a new one.
 *
 * \param options  The source owner's captured budget; its thread count must equal `source.size()`.
 * \param source   The states to copy; every entry non-null.
 * \param observer Test-only seam; see NoShardObserver.
 * \return T independent states.
 * \throws std::invalid_argument if the shard count does not match or an entry is null, before the team starts.
 * \throws As make_shards() for owner failures; the source is left unchanged.
 */
template <size_t NumModes, class Observer = NoShardObserver>
auto copy_shards(parallel::Options options, const Shards<NumModes> &source, const Observer &observer = {})
    -> Shards<NumModes> {
    if (options.threads < 1 || source.size() != static_cast<size_t>(options.threads)
        || std::ranges::any_of(source, [](const auto &state) { return !state; })) {
        throw std::invalid_argument(std::format("sharded::copy_shards: expected {} non-null source shards, got {}",
                                                options.threads,
                                                source.size()));
    }
    return make_shards<NumModes>(options, [&](size_t shard) {
        observer.begin(ShardWork::copy, shard);
        auto state = std::make_unique<ShardState<NumModes>>(*source[shard]);
        observer.end(ShardWork::copy, shard);
        return state;
    });
}

//! Rows over all shards of this rank, the rank-local operator size.
template <size_t NumModes>
auto total_size(const Shards<NumModes> &shards) -> size_t {
    size_t total = 0;
    for (const auto &state : shards) {
        total += state->size();
    }
    return total;
}

//! Operator accounting summed over this rank's shards, in shard order; the fields are additive.
template <size_t NumModes>
auto operator_memory_usage(const Shards<NumModes> &shards) -> MPOperatorMemoryBreakdown<NumModes> {
    auto total = MPOperatorMemoryBreakdown<NumModes>{};
    for (const auto &state : shards) {
        total += state->operator_memory_usage();
    }
    return total;
}

//! Graph accounting summed over this rank's shards, in shard order.
template <size_t NumModes>
auto graph_memory_usage(const Shards<NumModes> &shards) -> GraphMemoryBreakdown {
    auto total = GraphMemoryBreakdown{};
    for (const auto &state : shards) {
        total += state->graph_memory_usage();
    }
    return total;
}

} // namespace monoprop::detail::sharded
