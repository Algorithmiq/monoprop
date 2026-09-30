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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/sharded/State.h"
#include "monoprop/detail/sharded/Team.h"

/*
 * Graph construction and graph-free propagation over the T shards of one process (P = 1).
 *
 * One run_team() spans the whole gate loop. Worker t is the owner of shard t, whose flat routing slot is
 * rank * T + t; every other shard of the process is a cross owner, exactly like a shard of another process, so the
 * owners exchange partner queries through published buffers instead of communicators. Each phase below ends in a
 * collective checkpoint (phase()); a phase body mutates only its owner's shard and gate buffers, and reads other
 * owners' buffers only after the checkpoint that published them. Within-shard kernels run serially: the parallelism
 * is across owners.
 *
 * Per gate, in this order (P = publication checkpoint):
 *
 *   traverse + prepare leaders   scan the shard, resolve same-shard leaders, publish leader payloads   (P1)
 *   resolve leaders              each destination resolves every source's block, in ascending source
 *                                order, inserts its misses and publishes its answers                    (P2)
 *   consume leaders + prepare    fold the answers into the sink; then filter followers against the now
 *     followers                  complete leader marks, resolve same-shard followers, publish payloads   (P3)
 *   resolve followers            as for leaders                                                         (P4)
 *   consume followers + finish   fold the answers; insert the deferred same-shard leader then follower
 *                                misses; finalize the layer, or extend and apply the fused records      (P5)
 *
 * The identity generator skips the exchange phases; with T = 1 there is no cross owner, so only the resolve and
 * consume work is skipped, and same-shard resolution still runs. Row insertion order is therefore cross-owner
 * leaders, cross-owner followers, deferred same-shard leaders, deferred same-shard followers, as in the legacy
 * runtime at the same (1, T) geometry.
 *
 * Buffers (owner t writes only its own; no buffer is resized, moved or freed while another owner may read it):
 *
 * | Buffer                                  | Writer  | Published | Readers                  | Last read | Reset       |
 * | --------------------------------------- | ------- | --------- | ------------------------ | --------- | ----------- |
 * | leader payload (engine queries_r or     | owner t | P1        | every destination        | P2        | prepare of  |
 * |   fused scratch), board payloads[t]     |         |           |   (resolve leaders)      |           |   followers |
 * | leader answers, board answers[t]        | owner t | P2        | every source (consume)   | P3        | resolve of  |
 * |                                         |         |           |                          |           |   followers |
 * | follower payload, board payloads[t]     | owner t | P3        | every destination        | P4        | next gate   |
 * | follower answers, board answers[t]      | owner t | P4        | every source (consume)   | P5        | next gate   |
 * | own sources, plain queries, values      | owner t | (private) | owner t (consume)        | P3 / P5   | next pass   |
 * | same-shard stage, deferred misses,      | owner t | (private) | owner t                  | P5        | next gate   |
 * |   decoded incoming positions, scratch   |         |           |                          |           |             |
 * | fused records, cosine set (propagate)   | owner t | (private) | owner t (apply)          | P5        | next gate   |
 *
 * "Next gate" means the owner's first phase of the following gate, which starts after P5 has passed, or the final
 * cache phase, which releases the gate buffers. The board itself (2T pointers) and the frames' pointer vector are
 * allocated on the caller and outlive the team, so every buffer stays alive through the join on failure.
 *
 * Opaque callbacks: a cutoff predicate that is not a typed built-in (CutoffEvaluator::parallel_safe()), including a
 * basis-change closure, is never called from two threads at once. The traversal then runs on the primary, shard by
 * shard, in a phase every worker enters; the other owners wait at its checkpoint and resume with their own
 * resolution. This is an exclusive phase, not a smaller team: the shards and their owners are unchanged.
 *
 * Failure: every call that can throw runs inside a phase. A failure suppresses every later phase body on every
 * worker, the team joins, and the original exception of the lowest-numbered failing worker is returned (not
 * rethrown) together with whether mutation had started. Nothing is rolled back: once mutation started, the shards
 * are partly updated and the rank-level owner must be treated as invalid. Independent copies are unaffected.
 */

namespace monoprop::detail::sharded {

//! The owner work a construction reports to its observer, in phase order.
enum class ConstructionWork : std::uint8_t {
    frame,             //!< Allocate the owner's operation frame; mutates no shard.
    picture,           //!< Propagation only: bring the live picture coefficients up to date.
    traverse,          //!< Scan the shard for this gate's partner queries.
    prepare_leaders,   //!< Resolve same-shard leaders and expose the leader payload.
    resolve_leaders,   //!< Resolve the leader queries published to this shard; insert and index misses.
    consume_leaders,   //!< Fold the answers to this shard's leader queries.
    prepare_followers, //!< Filter followers, resolve same-shard followers and expose the follower payload.
    resolve_followers, //!< Resolve the follower queries published to this shard; insert and index misses.
    consume_followers, //!< Fold the answers to this shard's follower queries.
    finalize,          //!< Insert deferred same-shard misses; finalize the layer and append it to the graph.
    apply,             //!< Propagation only: extend the coefficients and apply the fused records.
    caches,            //!< After the last gate: release gate buffers and warm the shard's caches.
};

//! The step reported for work outside the gate loop.
inline constexpr size_t kNoStep = std::numeric_limits<size_t>::max();

/*!
 * \brief The observer production code uses: it does nothing and compiles away.
 *
 * `visit(work, step, shard)` runs inside the protected phase, on the thread that performs `work` for `shard`, before
 * that work; `step` is the gate-loop step or kNoStep. `kernels(shard)` supplies the range observer the shard's
 * kernels and exchange phases report to (see NoRangeObserver and LayerBuildEngine); it is called once, by the owner,
 * in its frame phase. Tests substitute observers that record the executing worker, capture streams or throw; an
 * exception from either method is that phase's failure.
 */
struct NoConstructionObserver {
    auto visit(ConstructionWork /*work*/, size_t /*step*/, size_t /*shard*/) const noexcept -> void {}
    [[nodiscard]] auto kernels(size_t /*shard*/) const noexcept -> NoRangeObserver { return {}; }
};

/*!
 * \brief Rank-level configuration of a construction, read concurrently and never mutated by the owners.
 */
template <size_t NumModes>
struct ConstructionContext {
    const CutoffFn<NumModes> &cutoff_fn;   //!< Structural cutoff on partners; must outlive the call.
    routing::Router router;                //!< Flat-owner routing at geometry (1, T), as the shards were seeded.
    std::optional<double> lower_atol;      //!< Lower sine cutoff; applies only where coefficients are supplied.
    std::optional<double> upper_atol;      //!< Upper rescue threshold; applies only where coefficients are supplied.
    Basis basis = Basis::Majorana;         //!< Rotation signs, and Schrödinger fresh-insert scoring.
    bool schrodinger = false;              //!< Picture: forward gate order and state coefficients when true.
    size_t rank = 0;                       //!< This process's rank; 0 at P = 1.
};

/*!
 * \brief A validated circuit for graph construction, in circuit order, with every array one entry per gate.
 *
 * Views the caller's arrays, which must outlive the call. Generators are already converted and bounds-checked.
 */
template <size_t NumModes>
struct GraphCircuit {
    std::span<const Monomial<NumModes>> generators; //!< Gate generators, in circuit order.
    std::span<const size_t> parameter_mapping;      //!< Parameter index of each gate's layer.
    std::span<const double> gen_coeffs;             //!< Generator coefficient of each gate's layer.
    std::span<const size_t> gate_indices;           //!< Absolute gate index of each layer (offset already applied).
    std::optional<size_t> only_rotate_len_k;        //!< Rotate only sources of at most this many slots.
};

/*!
 * \brief A validated circuit for graph-free propagation, in circuit order, with one angle per gate.
 *
 * Views the caller's arrays, which must outlive the call.
 */
template <size_t NumModes>
struct PropagationCircuit {
    std::span<const Monomial<NumModes>> generators; //!< Gate generators, in circuit order.
    std::span<const double> mapped_params;          //!< Build angle of each gate: map_params(..., 1.0).
    std::optional<size_t> only_rotate_len_k;        //!< Rotate only sources of at most this many slots.
};

/*!
 * \brief What a construction reports after its team has joined.
 *
 * The seam a rank-level caller applies its validity policy at: with an error and `mutation_started`, the shards are
 * partly updated and the owner must be invalidated (and, in a multi-rank run, the communicator aborted) before the
 * error is rethrown; with an error and no mutation, the shards are unchanged and the owner stays usable.
 */
struct ConstructionOutcome {
    std::exception_ptr error;      //!< The lowest-numbered failing worker's original exception, or empty.
    bool mutation_started = false; //!< Whether any phase that may change a shard began.
};

/*!
 * \brief Read-only views of the blocks the owners of one process published for `owner`, one per window slot.
 *
 * Slot `s` of `window` is the flat owner whose local index is `s - first_local`. Its view is that owner's block for
 * `owner` when it published a buffer whose window reaches `owner`; a slot outside this process's owners, an owner
 * that published nothing, or a buffer whose window does not reach `owner` yields an empty view. Views alias the
 * published buffers, which must stay unchanged while the views are read; nothing is copied.
 *
 * \param window      The reading owner's window over flat slots.
 * \param owner       The reading owner's flat slot.
 * \param first_local The flat slot of this process's first shard, `rank * T`.
 * \param published   Per local shard, its published buffer or null.
 * \return One view per slot of `window`, in slot order.
 */
template <typename T>
[[nodiscard]] auto gather_published(mpi::SlotWindow window,
                                    size_t owner,
                                    size_t first_local,
                                    std::span<const mpi::WindowVec<std::vector<T>> *const> published)
    -> mpi::WindowVec<std::span<const T>> {
    mpi::WindowVec<std::span<const T>> views(window);
    for (const auto wi : window.indices()) {
        const size_t slot = window.slot(wi);
        if (slot < first_local || slot - first_local >= published.size()) {
            continue;
        }
        const auto *const buffer = published[slot - first_local];
        if (buffer != nullptr && buffer->window().contains(owner)) {
            views[wi] = std::span<const T>(buffer->at_slot(owner));
        }
    }
    return views;
}

namespace construction_detail {

//! Within-shard kernels run serially: the team's parallelism is across owners.
inline constexpr parallel::Options kSerial{};

// One owner's buffers for the current gate. The fused records precede the engine, which refers to them.
template <size_t NumModes, typename Sink, class KernelObserver>
struct GateWork {
    using Response = typename Sink::Response;
    GateScan<NumModes> scan;
    FusedContract fc;
    CosMask cos;
    std::optional<LayerBuildEngine<NumModes, Sink, KernelObserver>> engine;
    mpi::WindowVec<std::vector<Response>> answers; // this owner's answers for the current pass
};

// One owner's operation frame, allocated by the owner in the frame phase.
template <size_t NumModes, typename Sink, class KernelObserver>
struct OwnerFrame {
    explicit OwnerFrame(KernelObserver kernels_) : kernels(std::move(kernels_)) {}
    KernelObserver kernels;
    VecD *coeffs = nullptr; // propagation: the live picture coefficients of this shard
    std::optional<GateWork<NumModes, Sink, KernelObserver>> work;
};

template <size_t NumModes>
auto check_arguments(const char *what,
                     parallel::Options options,
                     const Shards<NumModes> &shards,
                     const ConstructionContext<NumModes> &ctx,
                     size_t gates,
                     std::initializer_list<size_t> lengths,
                     std::optional<size_t> only_rotate_len_k) -> void {
    if (options.threads < 1 || shards.size() != static_cast<size_t>(options.threads)
        || std::ranges::any_of(shards, [](const auto &state) { return !state; })) {
        throw std::invalid_argument(std::format("sharded::{}: expected {} non-null shards, got {}",
                                                what,
                                                options.threads,
                                                shards.size()));
    }
    // The physical exchange does not exist yet, so every flat owner must be a shard of this process.
    if (ctx.router.ranks() != 1 || ctx.rank != 0 || ctx.router.partitions() != shards.size()) {
        throw std::invalid_argument(std::format("sharded::{}: routing geometry ({} ranks, {} shards per rank, rank {}) "
                                                "is not one process of {} shards",
                                                what,
                                                ctx.router.ranks(),
                                                ctx.router.partitions(),
                                                ctx.rank,
                                                shards.size()));
    }
    for (const size_t length : lengths) {
        if (length != gates) {
            throw std::invalid_argument(
                std::format("sharded::{}: a per-gate array has {} entries for {} gates", what, length, gates));
        }
    }
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
}

/*
 * The gate loop shared by graph construction (Fused = false, GraphSink) and propagation (Fused = true, ContractSink).
 * `angle_of(idx)` is the build angle of gate idx (propagation); `append(state, storage, idx)` appends a finished
 * layer (graph construction).
 */
template <size_t NumModes, bool Fused, class Observer, class AngleOf, class Append>
auto run_gates(parallel::Options options,
               Shards<NumModes> &shards,
               const ConstructionContext<NumModes> &ctx,
               std::span<const Monomial<NumModes>> generators,
               std::optional<size_t> only_rotate_len_k,
               const Observer &observer,
               AngleOf &&angle_of,
               Append &&append) -> ConstructionOutcome {
    using Sink = std::conditional_t<Fused, ContractSink<NumModes>, GraphSink<NumModes>>;
    using Response = typename Sink::Response;
    using KernelObserver = std::remove_cvref_t<decltype(std::declval<const Observer &>().kernels(size_t{}))>;
    using Frame = OwnerFrame<NumModes, Sink, KernelObserver>;
    using W = ConstructionWork;

    const size_t threads = shards.size();
    const size_t gates = generators.size();
    const size_t flat_world = ctx.router.flat_world();
    const size_t first_local = ctx.rank * threads;
    const bool schrodinger = ctx.schrodinger;
    // Opaque predicates keep the exclusive traversal; see the header comment.
    const bool owner_traversal = CutoffEvaluator<NumModes>(ctx.cutoff_fn).parallel_safe();

    // Caller-side metadata: T frame pointers and the 2T-pointer board. Entry t has one writer, owner t; readers
    // dereference it only after the checkpoint that published it.
    std::vector<std::unique_ptr<Frame>> frames(threads);
    std::vector<const mpi::WindowVec<VecZ> *> payloads(threads, nullptr);
    std::vector<const mpi::WindowVec<std::vector<Response>> *> answers(threads, nullptr);
    bool mutation_started = false; // written by the primary only, read after the join

    const auto error = run_team(options, [&](size_t t, TeamFailure &failure) noexcept {
        const size_t flat = first_local + t;
        ShardState<NumModes> &own = *shards[t];

        if (!phase(failure, t, [&] {
                observer.visit(W::frame, kNoStep, t);
                frames[t] = std::make_unique<Frame>(observer.kernels(t));
            })) {
            return;
        }
        if (t == 0) {
            mutation_started = true;
        }
        Frame &frame = *frames[t];
        if constexpr (Fused) {
            if (!phase(failure, t, [&] {
                    observer.visit(W::picture, kNoStep, t);
                    // Materializes pending initial entries (Heisenberg) or scores the state (Schrödinger).
                    (void)own.op.current_picture(schrodinger);
                    frame.coeffs = schrodinger ? &own.op.state_coeffs : &own.op.op_coeffs;
                })) {
                return;
            }
        }

        for (size_t step = 0; step < gates; ++step) {
            // Heisenberg applies the circuit in reverse; Schrödinger forward, with the apply angle negated.
            const size_t idx = schrodinger ? step : gates - 1 - step;
            const Monomial<NumModes> &gen = generators[idx];
            const bool passes = gen.any(); // identity: nothing anticommutes, nothing is exchanged
            const bool cross = passes && flat_world > 1;

            // Scans shard s into a fresh gate buffer; the previous gate's buffers were last read before P5.
            const auto traverse = [&](size_t s) {
                observer.visit(W::traverse, step, s);
                Frame &f = *frames[s];
                ShardState<NumModes> &state = *shards[s];
                f.work.reset();
                auto &work = f.work.emplace();
                if constexpr (Fused) {
                    work.scan = scan_gate<NumModes, KernelObserver>(state.op,
                                                                    gen,
                                                                    ctx.cutoff_fn,
                                                                    ctx.lower_atol,
                                                                    std::cref(*f.coeffs),
                                                                    ctx.upper_atol,
                                                                    angle_of(idx),
                                                                    only_rotate_len_k,
                                                                    ctx.router,
                                                                    first_local + s,
                                                                    /*use_fused=*/true,
                                                                    f.coeffs,
                                                                    ctx.basis,
                                                                    kSerial,
                                                                    f.kernels);
                }
                else {
                    work.scan = scan_gate<NumModes, KernelObserver>(state.op,
                                                                    gen,
                                                                    ctx.cutoff_fn,
                                                                    ctx.lower_atol,
                                                                    std::nullopt,
                                                                    ctx.upper_atol,
                                                                    std::nullopt,
                                                                    only_rotate_len_k,
                                                                    ctx.router,
                                                                    first_local + s,
                                                                    /*use_fused=*/false,
                                                                    nullptr,
                                                                    ctx.basis,
                                                                    kSerial,
                                                                    f.kernels);
                }
            };
            // Owner-local: builds the engine (which starts the matched epoch) and runs the leader prepare.
            const auto prepare_leaders = [&] {
                observer.visit(W::prepare_leaders, step, t);
                auto &work = *frame.work;
                auto &engine = [&]() -> auto & {
                    if constexpr (Fused) {
                        return work.engine.emplace(own.op,
                                                   flat_world,
                                                   flat,
                                                   own.matched,
                                                   work.scan.combined_size,
                                                   Sink{.R = flat_world,
                                                        .my_rank = flat,
                                                        .fc = work.fc,
                                                        .op_coeffs = *frame.coeffs,
                                                        .fused_scale = work.scan.fused_scale,
                                                        .inv_cos = work.scan.inv_cos,
                                                        .schrodinger = schrodinger,
                                                        .basis = ctx.basis},
                                                   work.scan.window,
                                                   kSerial,
                                                   frame.kernels);
                    }
                    else {
                        return work.engine.emplace(own.op,
                                                   flat_world,
                                                   flat,
                                                   own.matched,
                                                   work.scan.combined_size,
                                                   Sink(flat_world, flat),
                                                   work.scan.window,
                                                   kSerial,
                                                   frame.kernels);
                    }
                }();
                if (passes) {
                    auto &streams = work.scan.streams;
                    payloads[t] = &engine.prepare_exchange(true,
                                                           std::move(streams.leader_queries),
                                                           std::move(streams.leader_src),
                                                           std::move(streams.leader_val),
                                                           std::move(streams.leader_self));
                }
            };
            const auto resolve = [&](bool leaders) {
                observer.visit(leaders ? W::resolve_leaders : W::resolve_followers, step, t);
                auto &work = *frame.work;
                const auto incoming = gather_published<size_t>(
                    work.engine->window,
                    flat,
                    first_local,
                    std::span<const mpi::WindowVec<VecZ> *const>(payloads));
                work.answers = work.engine->resolve_published(incoming, leaders);
                answers[t] = &work.answers;
            };
            const auto consume = [&](bool leaders) {
                observer.visit(leaders ? W::consume_leaders : W::consume_followers, step, t);
                auto &work = *frame.work;
                work.engine->consume_published(gather_published<Response>(
                    work.engine->window,
                    flat,
                    first_local,
                    std::span<const mpi::WindowVec<std::vector<Response>> *const>(answers)));
            };

            // P1: traversal and leader preparation.
            bool ok = false;
            if (owner_traversal) {
                ok = phase(failure, t, [&] {
                    traverse(t);
                    prepare_leaders();
                });
            }
            else {
                ok = phase(failure, t, [&] {
                         if (t == 0) {
                             for (size_t s = 0; s < threads; ++s) {
                                 traverse(s);
                             }
                         }
                     })
                     && phase(failure, t, prepare_leaders);
            }
            if (!ok) {
                return;
            }
            // P2: every destination resolves the leader payloads published at P1.
            if (cross && !phase(failure, t, [&] { resolve(true); })) {
                return;
            }
            // P3: leader answers consumed; the follower filter below reads this owner's completed leader marks.
            if (passes && !phase(failure, t, [&] {
                    if (cross) {
                        consume(true);
                    }
                    observer.visit(W::prepare_followers, step, t);
                    auto &streams = frame.work->scan.streams;
                    payloads[t] = &frame.work->engine->prepare_exchange(false,
                                                                        std::move(streams.follower_queries),
                                                                        std::move(streams.follower_src),
                                                                        std::move(streams.follower_val),
                                                                        std::move(streams.follower_self));
                })) {
                return;
            }
            // P4: every destination resolves the follower payloads published at P3.
            if (cross && !phase(failure, t, [&] { resolve(false); })) {
                return;
            }
            // P5: follower answers consumed, then the owner-local finish.
            if (!phase(failure, t, [&] {
                    if (cross) {
                        consume(false);
                    }
                    observer.visit(W::finalize, step, t);
                    auto &work = *frame.work;
                    if constexpr (Fused) {
                        (void)work.engine->finish(std::move(work.scan.cos_all), &work.cos);
                        observer.visit(W::apply, step, t);
                        // After the inserts, before the insert-target gather of the apply.
                        own.op.extend_from_current_picture(*frame.coeffs, schrodinger);
                        const double build_angle = angle_of(idx);
                        apply_fused_contract(work.fc,
                                             *frame.coeffs,
                                             work.cos,
                                             schrodinger ? -build_angle : build_angle,
                                             schrodinger,
                                             work.scan.fused_scale,
                                             kSerial,
                                             frame.kernels);
                    }
                    else {
                        auto storage = work.engine->finish(std::move(work.scan.cos_all), nullptr);
                        stamp_layer_metadata<NumModes>(*storage, gen, own.op);
                        append(own, std::move(storage), idx);
                    }
                })) {
                return;
            }
        }

        // The last phase: its decision ends the sequence either way.
        static_cast<void>(phase(failure, t, [&] {
            observer.visit(W::caches, kNoStep, t);
            frame.work.reset();
            own.op.initialize_caches(schrodinger);
        }));
    });
    return ConstructionOutcome{.error = error, .mutation_started = mutation_started};
}

} // namespace construction_detail

/*!
 * \brief Append one layer per gate to every shard's graph, building the partner rows the layers need.
 *
 * The graph-construction gate loop at geometry (1, T), with one team for the whole circuit. Gates run in reverse
 * circuit order (Heisenberg) or in circuit order (Schrödinger); layer i carries `parameter_mapping[i]`,
 * `gen_coeffs[i]` and `gate_indices[i]`, its generator words and its post-insertion `scaled_count`, and is appended
 * with the picture's graph semantics. After the last gate every shard's caches are warmed. An empty circuit returns
 * at once, without a team.
 *
 * \param options  The shards' captured budget; its thread count T is the team size and the shard count.
 * \param shards   The process's T shards, seeded or copied before the call; mutated in place, each by its owner.
 * \param ctx      Rank-level configuration; see ConstructionContext.
 * \param circuit  The validated circuit; see GraphCircuit.
 * \param observer Test-only seam; see NoConstructionObserver.
 * \return The outcome after the team has joined; see ConstructionOutcome. Phase failures are returned, not thrown.
 * \throws std::invalid_argument before the team, with nothing mutated, if the shards, geometry or per-gate arrays do
 *         not fit; as validate_only_rotate_len_k_() for an out-of-range `only_rotate_len_k`; std::bad_alloc if the
 *         caller-side metadata cannot be allocated.
 */
template <size_t NumModes, class Observer = NoConstructionObserver>
[[nodiscard]] auto build_graph(parallel::Options options,
                               Shards<NumModes> &shards,
                               const ConstructionContext<NumModes> &ctx,
                               const GraphCircuit<NumModes> &circuit,
                               const Observer &observer = {}) -> ConstructionOutcome {
    const size_t gates = circuit.generators.size();
    construction_detail::check_arguments("build_graph",
                                         options,
                                         shards,
                                         ctx,
                                         gates,
                                         {circuit.parameter_mapping.size(),
                                          circuit.gen_coeffs.size(),
                                          circuit.gate_indices.size()},
                                         circuit.only_rotate_len_k);
    if (gates == 0) {
        return {};
    }
    return construction_detail::run_gates<NumModes, false>(
        options,
        shards,
        ctx,
        circuit.generators,
        circuit.only_rotate_len_k,
        observer,
        [](size_t) { return 0.0; },
        [&](ShardState<NumModes> &state, std::shared_ptr<LayerCore> storage, size_t idx) {
            state.graph.append(std::move(storage),
                               circuit.parameter_mapping[idx],
                               circuit.gen_coeffs[idx],
                               circuit.gate_indices[idx]);
        });
}

/*!
 * \brief Propagate every shard's picture coefficients through the circuit, contracting each gate immediately.
 *
 * The graph-free gate loop at geometry (1, T), with one team for the whole circuit: Heisenberg evolves the operator
 * coefficients in reverse circuit order at angle `mapped_params[i]`; Schrödinger evolves the state coefficients in
 * circuit order at the negated angle. Coefficients are captured before the fused cosine sweep overwrites them, the
 * two-pass fallback runs where the sweep cannot (a length cap or a zero cosine), and fresh rows are extended from
 * the picture before the apply. No graph layer is recorded. After the last gate every shard's caches are warmed.
 * An empty circuit returns at once, without a team.
 *
 * \param options  The shards' captured budget; its thread count T is the team size and the shard count.
 * \param shards   The process's T shards; mutated in place, each by its owner.
 * \param ctx      Rank-level configuration; see ConstructionContext.
 * \param circuit  The validated circuit; see PropagationCircuit.
 * \param observer Test-only seam; see NoConstructionObserver.
 * \return The outcome after the team has joined; see ConstructionOutcome. Phase failures are returned, not thrown.
 * \throws As build_graph(), before the team, with nothing mutated.
 */
template <size_t NumModes, class Observer = NoConstructionObserver>
[[nodiscard]] auto propagate(parallel::Options options,
                             Shards<NumModes> &shards,
                             const ConstructionContext<NumModes> &ctx,
                             const PropagationCircuit<NumModes> &circuit,
                             const Observer &observer = {}) -> ConstructionOutcome {
    const size_t gates = circuit.generators.size();
    construction_detail::check_arguments("propagate",
                                         options,
                                         shards,
                                         ctx,
                                         gates,
                                         {circuit.mapped_params.size()},
                                         circuit.only_rotate_len_k);
    if (gates == 0) {
        return {};
    }
    return construction_detail::run_gates<NumModes, true>(
        options,
        shards,
        ctx,
        circuit.generators,
        circuit.only_rotate_len_k,
        observer,
        [&](size_t idx) { return circuit.mapped_params[idx]; },
        [](ShardState<NumModes> &, std::shared_ptr<LayerCore>, size_t) {});
}

} // namespace monoprop::detail::sharded
