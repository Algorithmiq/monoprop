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
#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/sharded/Evaluation.h"
#include "monoprop/detail/sharded/Exchange.h"
#include "monoprop/detail/sharded/State.h"
#include "monoprop/detail/sharded/Team.h"

/*
 * Graph construction and graph-free propagation over the T shards of one process, at geometry (P, T).
 *
 * One run_team() spans the whole gate loop. Worker t is the owner of shard t, whose flat routing slot is
 * rank * T + t; every other shard is a cross owner, in this process or another. Owners of this process exchange
 * partner queries through published buffers instead of communicators; blocks for and from other processes go through
 * one physical round per handoff (Exchange.h), whose MPI calls the primary makes. Each phase below ends in a
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
 * Coefficient-informed graph construction (build_graph_informed) runs the same gate phases with the owner's evolving
 * coefficients and build angle, after a seed phase that copies the current picture and, when a graph already exists,
 * replays it through replay_forward_in_team() in the same team. Its P5 then also extends the coefficients and begins
 * the replay of the new layer over its transient cosine set (snapshot, publish, cosine pass), and one more checkpoint
 * (P6) finishes it from the partners' published snapshots before the next gate's traversal reads the coefficients.
 *
 * The identity generator skips the exchange phases; with one flat owner there is no cross owner, so only the resolve
 * and consume work is skipped, and same-shard resolution still runs. Row insertion order is therefore cross-owner
 * leaders, cross-owner followers, deferred same-shard leaders, deferred same-shard followers, as in the legacy
 * runtime at the same (P, T) geometry.
 *
 * Other processes. A gate's window reaches another rank exactly when its peer plan does: the single peer
 * rank ^ shift(generator) under linear routing (none when the shift is zero), every other rank under splitmix. That
 * depends on the replicated generator and router only, so every rank takes the same branch. Each pass then adds:
 *
 *   P1 / P3   the owner also writes its query send row: the size of its block for every remote destination
 *   Q1        primary: lay out the query round, post its count round
 *   Q2        owners pack their remote query blocks into the round's send staging
 *   Q3        primary: complete the counts, lay out the receive side, post and complete the queries
 *   P2 / P4   destinations resolve local views and received blocks together, in ascending flat-slot order, then
 *             write their answer rows: one answer per query resolved for each remote source, and one per own query
 *             to each remote destination (answer counts are query counts, not the streams' word counts)
 *   R1        primary: lay out the answer round
 *   R2        owners pack the answers remote sources wait for
 *   R3        primary: post and complete the answers
 *   P3 / P5   sources consume local and received answers, in window order
 *
 * The query staging is read in P2 (P4) and rewritten only by the next pass's Q3; the answer staging is read in P3
 * (P5) and rewritten only by the next pass's R3. Construction's peer plan picks the transport: pairwise for the single
 * linear peer, one collective MPI_Ialltoallv under splitmix. Informed construction's new-layer replay goes through
 * a replay round as evaluation does (Evaluation.h), with the communicator-agreed transport: P5 publishes and writes
 * the replay rows, a primary phase lays the round out, owners pack, and a phase posts, runs every owner's cosine
 * pass while the transfer is in flight and completes it before P6 reads the received blocks.
 *
 * Buffers (owner t writes only its own; no buffer is resized, moved or freed while another owner may read it):
 *
 * | Buffer                                  | Writer  | Published | Readers                | Last read | Reset       |
 * | --------------------------------------- | ------- | --------- | ---------------------- | --------- | ----------- |
 * | leader payload (engine queries_r or     | owner t | P1        | every destination      | P2        | prepare of  |
 * |   fused scratch), query inbox row t     |         |           |   (resolve leaders)    |           |   followers |
 * | leader answers, answer inbox row t      | owner t | P2        | every source (consume) | P3        | resolve of  |
 * |                                         |         |           |                        |           |   followers |
 * | follower payload, query inbox row t     | owner t | P3        | every destination      | P4        | next gate   |
 * | follower answers, answer inbox row t    | owner t | P4        | every source (consume) | P5        | next gate   |
 * | own sources, plain queries, values      | owner t | (private) | owner t (consume)      | P3 / P5   | next pass   |
 * | same-shard stage, deferred misses,      | owner t | (private) | owner t                | P5        | next gate   |
 * |   decoded incoming positions, scratch   |         |           |                        |           |             |
 * | fused records, cosine set (propagate)   | owner t | (private) | owner t (apply)        | P5        | next gate   |
 * | informed: new-layer replay publication, | owner t | P5        | every partner (P6)     | P6        | next P5     |
 * |   board replay_pairs[t]                 |         |           |                        |           |             |
 * | informed: evolving coefficients, self   | owner t | (private) | owner t                | caches    | frame freed |
 * |   sources, transient cosine set         |         |           |                        |           |             |
 *
 * "Next gate" means the owner's first phase of the following gate, which starts after P5 has passed, or the final
 * cache phase, which releases the gate buffers. The two T x T inboxes of views (row t written by owner t, column d read
 * by owner d), the replay board, the frames' pointer vector and the physical rounds are allocated on the caller and
 * outlive the team, so every buffer and request stays alive through the join on failure.
 *
 * Opaque callbacks: a cutoff predicate that is not a typed built-in (CutoffEvaluator::parallel_safe()), including a
 * basis-change closure, is never called from two threads at once. The traversal then runs on the primary, shard by
 * shard, in a phase every worker enters; the other owners wait at its checkpoint and resume with their own
 * resolution. This is an exclusive phase, not a smaller team: the shards and their owners are unchanged.
 *
 * Failure: every call that can throw runs inside a phase. A failure suppresses every later phase body on every
 * worker, the team joins, and the original exception of the lowest-numbered failing worker is returned (not
 * rethrown) together with whether mutation had started. Nothing is rolled back: once mutation started, the shards
 * are partly updated and the rank-level owner must be treated as invalid. Independent copies are unaffected. If a
 * physical round still has live requests after a failed join, the seam hands the error to mpi::operation_failed()
 * itself, before any request owner is destroyed: on a multi-rank communicator that aborts it.
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
    seed,              //!< Informed construction: copy the picture, prepare and run the seed replay.
    replay,            //!< Informed construction: begin (P5) or finish (P6) the replay of the new layer.
    exchange,          //!< Multi-rank: the primary lays out, posts or completes a physical round (shard 0 only).
    pack,              //!< Multi-rank: copy this owner's remote blocks into a round's send staging.
};

/*!
 * \brief The observer production code uses: it does nothing and compiles away.
 *
 * `visit(work, step, shard)` runs inside the protected phase, on the thread that performs `work` for `shard`, before
 * that work; `step` is the gate-loop step or kNoStep. `kernels(shard)` supplies the range observer the shard's
 * kernels and exchange phases report to (see NoRangeObserver and LayerBuildEngine); it is called once, by the owner,
 * in its frame phase. Tests substitute observers that record the executing worker, capture streams or throw; an
 * exception from either method is that phase's failure. A test observer may also define
 * `replayed(shard, step, const VecD &coeffs)`, detected at compile time, which informed construction calls after each
 * new layer's replay with the owner's evolving coefficients, `queries_packed(step, shard, PhysicalExchange &)`,
 * called after the owner packed its remote query blocks, which may rewrite them to inject malformed traffic, and
 * `fused_records(step, shard, const FusedGateView &)`, which propagation calls with the owner's real fused records
 * just before it applies them.
 */
struct NoConstructionObserver {
    auto visit(ConstructionWork /*work*/, size_t /*step*/, size_t /*shard*/) const noexcept -> void {}
    [[nodiscard]] auto kernels(size_t /*shard*/) const noexcept -> NoRangeObserver { return {}; }
};

/*!
 * \brief Test-only view of one owner's fused records for one propagation gate, just before they are applied.
 *
 * Everything apply_fused_contract() receives, read-only: an observer may copy it and replay the apply itself.
 */
struct FusedGateView {
    const FusedContract &records;   //!< The gate's hit, insert and cross-rank-half records for this shard.
    const CosMask &cos;             //!< The gate's cosine set on this shard.
    std::span<const double> coeffs; //!< The picture coefficients the records apply to, already extended.
    double apply_angle;             //!< The angle apply_fused_contract() receives.
    bool schrodinger;               //!< Picture.
    bool fused_scale;               //!< Whether the traversal already applied the cosine (fused sweep).
};

/*!
 * \brief Rank-level configuration of a construction, read concurrently and never mutated by the owners.
 */
template <size_t NumModes>
struct ConstructionContext {
    const CutoffFn<NumModes> &cutoff_fn; //!< Structural cutoff on partners; must outlive the call.
    routing::Router router;              //!< Flat-owner routing at geometry (P, T), as the shards were seeded.
    std::optional<double> lower_atol;    //!< Lower sine cutoff; applies only where coefficients are supplied.
    std::optional<double> upper_atol;    //!< Upper rescue threshold; applies only where coefficients are supplied.
    Basis basis = Basis::Majorana;       //!< Rotation signs, and Schrödinger fresh-insert scoring.
    bool schrodinger = false;            //!< Picture: forward gate order and state coefficients when true.
    //! This process among the router's ranks; read on the caller (PhysicalWorld::of()) before the call.
    PhysicalWorld world{};
    //! The owner's physical rounds, reused across its calls; null allocates rounds for this call only.
    PhysicalRounds *rounds = nullptr;
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
 * \brief A validated circuit for coefficient-informed graph construction.
 *
 * Views the caller's arrays, which must outlive the call.
 */
template <size_t NumModes>
struct InformedCircuit {
    GraphCircuit<NumModes> graph;          //!< Generators and layer metadata, as for build_graph().
    std::span<const double> mapped_params; //!< Build angle of each gate: map_params(parameters, ..., 1.0).
    //! Seed replay angle of each existing layer, in replay order (see replay_shards()); required exactly when the
    //! shards already hold layers, which are then replayed to form the seed. Without it the seed is the picture.
    std::optional<std::span<const double>> seed_params;
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

/*
 * Post one source's published blocks into its row of the `threads` x `threads` inbox of views: entry
 * source * threads + d views the block for local destination d, or is empty when the buffer's window does not reach
 * it. Every entry of the row is rewritten, so no view from an earlier pass survives, and a row has one writer, so posts
 * by different sources share no cache line except at row boundaries. The views alias `buffer`, which must stay
 * unchanged until its readers are done. Read from the destination's side, this equals gather_published().
 */
template <typename T>
auto post_blocks(const mpi::WindowVec<std::vector<T>> &buffer,
                 size_t source,
                 size_t first_local,
                 size_t threads,
                 std::span<std::span<const T>> inbox) -> void {
    const mpi::SlotWindow window = buffer.window();
    const auto row = inbox.subspan(source * threads, threads);
    for (size_t d = 0; d < threads; ++d) {
        const size_t slot = first_local + d;
        row[d] = window.contains(slot) ? std::span<const T>(buffer.at_slot(slot)) : std::span<const T>{};
    }
}

/*
 * Destination `dest`'s views over `window` from its inbox column: slot s of the window gets the entry local source
 * s - first_local posted for `dest` when s is a local shard, and an empty view otherwise. The column's entries are
 * independent loads, one per source row.
 */
template <typename T>
[[nodiscard]] auto collect_blocks(mpi::SlotWindow window,
                                  size_t first_local,
                                  size_t dest,
                                  size_t threads,
                                  std::span<const std::span<const T>> inbox) -> mpi::WindowVec<std::span<const T>> {
    mpi::WindowVec<std::span<const T>> views(window);
    for (const auto wi : window.indices()) {
        const size_t slot = window.slot(wi);
        if (slot >= first_local && slot - first_local < threads) {
            views[wi] = inbox[((slot - first_local) * threads) + dest];
        }
    }
    return views;
}

//! Within-shard kernels run serially: the team's parallelism is across owners.
inline constexpr parallel::Options kSerial{};

/*
 * The peer ranks a gate's handoffs reach: the single linear peer rank ^ shift (none when that is this rank), or every
 * other rank under a dense plan. A function of the replicated generator and router only, so every rank computes the
 * same set for the same gate.
 */
inline auto gate_peers(mpi::PeerPlan plan, const PhysicalWorld &world) -> std::vector<size_t> {
    if (plan.sparse) {
        const size_t peer = world.rank ^ static_cast<size_t>(plan.shift);
        return peer == world.rank ? std::vector<size_t>{} : std::vector<size_t>{peer};
    }
    return other_ranks(world);
}

// The flat slot of shard `shard` of rank `rank`, checked against the window it must index.
inline auto window_slot(mpi::SlotWindow window, size_t rank, size_t shard, size_t threads) -> size_t {
    const size_t slot = (rank * threads) + shard;
    if (!window.contains(slot)) {
        throw std::logic_error(
            std::format("sharded construction: remote slot {} lies outside the gate's window [{}, {})",
                        slot,
                        window.base,
                        window.stop()));
    }
    return slot;
}

// Owner `own`'s send row: the size of its block for every shard of every peer.
template <typename T>
auto write_send_rows(PhysicalExchange &round,
                     const mpi::WindowVec<std::vector<T>> &blocks,
                     std::span<const size_t> peers,
                     size_t threads,
                     size_t own) -> void {
    round.reset_rows(own);
    for (size_t k = 0; k < peers.size(); ++k) {
        for (size_t t = 0; t < threads; ++t) {
            round.set_send_count(own, k, t, blocks.at_slot(window_slot(blocks.window(), peers[k], t, threads)).size());
        }
    }
}

/*
 * Owner `own`'s rows of an answer round: it sends one answer per query it resolved for each remote source, and receives
 * one per query of its own (`sources`, one entry per query) addressed to each remote destination. Answer counts are
 * query counts, not the query streams' word counts, so they are known only after resolution.
 */
template <typename Response>
auto write_answer_rows(PhysicalExchange &round,
                       const mpi::WindowVec<std::vector<Response>> &answers,
                       const mpi::WindowVec<std::vector<size_t>> &sources,
                       std::span<const size_t> peers,
                       size_t threads,
                       size_t own) -> void {
    round.reset_rows(own);
    for (size_t k = 0; k < peers.size(); ++k) {
        for (size_t t = 0; t < threads; ++t) {
            round.set_send_count(own,
                                 k,
                                 t,
                                 answers.at_slot(window_slot(answers.window(), peers[k], t, threads)).size());
            round.set_recv_count(own,
                                 k,
                                 t,
                                 sources.at_slot(window_slot(sources.window(), peers[k], t, threads)).size());
        }
    }
}

// Owner `own` copies its block for every remote shard into its slices of the round's send staging.
template <typename T>
auto pack_blocks(PhysicalExchange &round,
                 const mpi::WindowVec<std::vector<T>> &blocks,
                 std::span<const size_t> peers,
                 size_t threads,
                 size_t own) -> void {
    for (size_t k = 0; k < peers.size(); ++k) {
        for (size_t t = 0; t < threads; ++t) {
            const auto &block = blocks.at_slot(window_slot(blocks.window(), peers[k], t, threads));
            const auto slice = round.send_block<T>(own, k, t);
            if (slice.size() != block.size()) {
                throw std::logic_error(std::format("sharded construction: a {}-element block meets a {}-element slice",
                                                   block.size(),
                                                   slice.size()));
            }
            std::ranges::copy(block, slice.begin());
        }
    }
}

// Views of what owner `own` received from every remote shard, placed at their slots of `views`.
template <typename T>
auto add_received(mpi::WindowVec<std::span<const T>> &views,
                  const PhysicalExchange &round,
                  std::span<const size_t> peers,
                  size_t threads,
                  size_t own) -> void {
    for (size_t k = 0; k < peers.size(); ++k) {
        for (size_t su = 0; su < threads; ++su) {
            views.at_slot(window_slot(views.window(), peers[k], su, threads)) = round.recv_block<T>(own, k, su);
        }
    }
}

// One owner's buffers for the current gate. The fused records precede the engine, which refers to them.
template <size_t NumModes, typename Sink, class KernelObserver>
struct GateWork {
    using Response = typename Sink::Response;
    GateScan<NumModes> scan;
    FusedContract fc;
    CosMask cos;
    std::optional<LayerBuildEngine<NumModes, Sink, KernelObserver>> engine;
    const mpi::WindowVec<VecZ> *payload = nullptr; // the engine's payload for the current pass (multi-rank packing)
    mpi::WindowVec<std::vector<Response>> answers; // this owner's answers for the current pass
    std::shared_ptr<LayerCore> core;               // informed: the new layer, replayed at P5/P6
};

// One owner's operation frame, allocated by the owner in the frame phase.
template <size_t NumModes, typename Sink, class KernelObserver>
struct OwnerFrame {
    explicit OwnerFrame(KernelObserver kernels_) : kernels(std::move(kernels_)) {}
    KernelObserver kernels;
    VecD *coeffs = nullptr; // propagation: the live picture coefficients; informed: `informed` below
    std::optional<GateWork<NumModes, Sink, KernelObserver>> work;
    VecD informed;                        // informed: the evolving coefficients (seed, then gate by gate)
    std::optional<MPGraphView> seed_view; // informed: the existing graph's replay window
    CosCallbacks seed_callbacks;          // informed: the seed replay's callbacks
    ForwardScratch replay_scratch;        // informed: self sources of the new layer's replay
    PublishedPair replay_published;       // informed: read by partners at P6
};

// Construction's work reported through an evaluation observer: the seed replay's steps.
template <class Observer>
class SeedObserver final : public EvaluationObserver {
public:
    explicit SeedObserver(const Observer &observer) : observer_(&observer) {}
    auto visit(EvaluationWork /*work*/, size_t step, size_t shard) const -> void override {
        observer_->visit(ConstructionWork::seed, step, shard);
    }

private:
    const Observer *observer_;
};

//! Which gate loop run_gates() executes.
enum class GateMode : std::uint8_t { graph, informed, propagate };

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
        throw std::invalid_argument(
            std::format("sharded::{}: expected {} non-null shards, got {}", what, options.threads, shards.size()));
    }
    // The router's geometry must be this process's: its ranks, and T shards per rank.
    if (ctx.router.ranks() != ctx.world.ranks || ctx.world.rank >= ctx.world.ranks
        || ctx.router.partitions() != shards.size()) {
        throw std::invalid_argument(std::format("sharded::{}: routing geometry ({} ranks, {} shards per rank) does not "
                                                "match rank {} of {} with {} shards",
                                                what,
                                                ctx.router.ranks(),
                                                ctx.router.partitions(),
                                                ctx.world.rank,
                                                ctx.world.ranks,
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
 * The gate loop shared by graph construction (GraphSink), coefficient-informed graph construction (GraphSink with
 * coefficients) and propagation (ContractSink). `angle_of(idx)` is the build angle of gate idx (informed and
 * propagation); `append(state, storage, idx)` appends a finished layer (both graph modes); `seed_params` are the
 * informed seed replay's angles, or null for the picture seed.
 */
template <size_t NumModes, GateMode Mode, class Observer, class AngleOf, class Append>
auto run_gates(parallel::Options options,
               Shards<NumModes> &shards,
               const ConstructionContext<NumModes> &ctx,
               std::span<const Monomial<NumModes>> generators,
               std::optional<size_t> only_rotate_len_k,
               const Observer &observer,
               AngleOf &&angle_of,
               Append &&append,
               const std::span<const double> *seed_params = nullptr) -> ConstructionOutcome {
    constexpr bool Fused = Mode == GateMode::propagate;
    constexpr bool Informed = Mode == GateMode::informed;
    using Sink = std::conditional_t<Fused, ContractSink<NumModes>, GraphSink<NumModes>>;
    using Response = typename Sink::Response;
    using KernelObserver = std::remove_cvref_t<decltype(std::declval<const Observer &>().kernels(size_t{}))>;
    using Frame = OwnerFrame<NumModes, Sink, KernelObserver>;
    using W = ConstructionWork;

    const size_t threads = shards.size();
    const size_t gates = generators.size();
    const size_t flat_world = ctx.router.flat_world();
    const size_t first_local = ctx.world.rank * threads;
    const bool schrodinger = ctx.schrodinger;
    // Opaque predicates keep the exclusive traversal; see the header comment.
    const bool owner_traversal = CutoffEvaluator<NumModes>(ctx.cutoff_fn).parallel_safe();

    // Caller-side metadata: T frame pointers. Entry t has one writer, owner t; readers dereference it only after the
    // checkpoint that published it.
    std::vector<std::unique_ptr<Frame>> frames(threads);
    // Handoff tables (T x T): entry s * T + d views source s's block for destination d. Source s writes its row in the
    // phase that publishes the blocks; destination d reads its column only after that phase's checkpoint, with one
    // independent load per source instead of chasing every source's buffer.
    std::vector<std::span<const size_t>> query_inbox(threads * threads);
    std::vector<std::span<const Response>> answer_inbox(threads * threads);
    // Informed only: the new layers' replay publications (entry t written by owner t in its frame phase) and the
    // seed replay job (entry t filled by owner t in the seed phase, read by partners after its checkpoint).
    std::vector<const PublishedPair *> replay_pairs(threads, nullptr);
    ForwardReplayJob seed_job;
    if constexpr (Informed) {
        if (seed_params != nullptr) {
            seed_job.params = *seed_params;
            seed_job.first_local = first_local;
            seed_job.owners.resize(threads);
        }
    }
    const SeedObserver<Observer> seed_observer(observer);
    bool mutation_started = false; // written by the primary only, read after the join

    // Physical rounds, only when there are other processes; allocated on the caller and alive through the join.
    const PhysicalWorld &world = ctx.world;
    const bool multirank = world.ranks > 1;
    // The owner's rounds when it supplies them (ctx.rounds), so their staging persists across calls; otherwise this
    // call's own.
    PhysicalRounds own_rounds;
    PhysicalExchange *queries = nullptr;
    PhysicalExchange *answers = nullptr;
    PhysicalExchange *replay = nullptr;
    if (multirank) {
        PhysicalRounds &rounds = ctx.rounds != nullptr ? *ctx.rounds : own_rounds;
        using Kind = PhysicalRounds::Kind;
        queries = &rounds.get(Kind::queries, world, threads);
        answers = &rounds.get(Fused ? Kind::fused_answers : Kind::graph_answers, world, threads);
        if constexpr (Informed) {
            replay = &rounds.get(Kind::replay, world, threads);
        }
    }
    const routing::Router &router = ctx.router;
    // Construction's transport follows its peer plan: pairwise for the linear peer, collective under splitmix.
    const auto transport = router.is_linear() ? ExchangeTransport::pairwise : ExchangeTransport::collective;
    const ReplayRound replay_round{
        .round = replay,
        .transport = world.replay_pairwise ? ExchangeTransport::pairwise : ExchangeTransport::collective};
    if constexpr (Informed) {
        seed_job.remote = replay_round;
    }

    const auto error = run_team(options, [&](size_t t, TeamFailure &failure) noexcept {
        const size_t flat = first_local + t;
        ShardState<NumModes> &own = *shards[t];

        if (!phase(failure, t, [&] {
                observer.visit(W::frame, kNoStep, t);
                frames[t] = std::make_unique<Frame>(observer.kernels(t));
                replay_pairs[t] = &frames[t]->replay_published;
            })) {
            return;
        }
        if (t == 0) {
            mutation_started = true;
        }
        Frame &frame = *frames[t];
        if constexpr (Informed) {
            // The seed: the current picture, or the existing graph replayed from it at the seed angles.
            if (!phase(failure, t, [&] {
                    observer.visit(W::seed, kNoStep, t);
                    frame.informed = own.op.current_picture(schrodinger);
                    frame.coeffs = &frame.informed;
                    if (seed_params != nullptr) {
                        frame.seed_view.emplace(own.graph.slice_view(own.graph.layers()));
                        frame.seed_callbacks =
                            make_cos_callbacks<NumModes>(own.op.inverted_index(), *frame.seed_view, ctx.basis, kSerial);
                        auto &entry = seed_job.owners[t];
                        entry.coeffs = &frame.informed;
                        entry.graph = &*frame.seed_view;
                        entry.callbacks = &frame.seed_callbacks;
                    }
                })) {
                return;
            }
            if (seed_params != nullptr && !replay_forward_in_team(failure, t, seed_job, &seed_observer)) {
                return;
            }
        }
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
            // The other processes this gate's handoffs reach; the same set on every rank (see the header comment).
            const auto peers =
                multirank && passes
                    ? gate_peers(mpi::PeerPlan{.sparse = router.is_linear(),
                                               .shift = static_cast<int>(router.rank_shift<NumModes>(gen))},
                                 world)
                    : std::vector<size_t>{};
            const bool remote = !peers.empty();

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
                else if constexpr (Informed) {
                    // As the legacy coefficient-informed build: coefficients and build angle, no fused sweep.
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
                                                                    /*use_fused=*/false,
                                                                    nullptr,
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
                    work.payload = &engine.prepare_exchange(true,
                                                            std::move(streams.leader_queries),
                                                            std::move(streams.leader_src),
                                                            std::move(streams.leader_val),
                                                            std::move(streams.leader_self));
                    post_blocks<size_t>(*work.payload,
                                        t,
                                        first_local,
                                        threads,
                                        std::span<std::span<const size_t>>(query_inbox));
                    if (remote) {
                        write_send_rows<size_t>(*queries, *work.payload, peers, threads, t);
                    }
                }
            };
            const auto resolve = [&](bool leaders) {
                observer.visit(leaders ? W::resolve_leaders : W::resolve_followers, step, t);
                auto &work = *frame.work;
                auto incoming = collect_blocks<size_t>(work.engine->window,
                                                       first_local,
                                                       t,
                                                       threads,
                                                       std::span<const std::span<const size_t>>(query_inbox));
                if (remote) {
                    add_received<size_t>(incoming, *queries, peers, threads, t);
                }
                work.answers = work.engine->resolve_published(incoming, leaders);
                post_blocks<Response>(work.answers,
                                      t,
                                      first_local,
                                      threads,
                                      std::span<std::span<const Response>>(answer_inbox));
                if (remote) {
                    write_answer_rows<Response>(*answers, work.answers, work.engine->src_idx_r, peers, threads, t);
                }
            };
            const auto consume = [&](bool leaders) {
                observer.visit(leaders ? W::consume_leaders : W::consume_followers, step, t);
                auto &work = *frame.work;
                auto received = collect_blocks<Response>(work.engine->window,
                                                         first_local,
                                                         t,
                                                         threads,
                                                         std::span<const std::span<const Response>>(answer_inbox));
                if (remote) {
                    add_received<Response>(received, *answers, peers, threads, t);
                }
                work.engine->consume_published(received);
            };
            // Q1-Q3 of a pass whose queries reach other processes; see the header comment.
            const auto query_round = [&]() -> bool {
                if (!remote) {
                    return true;
                }
                return phase(failure,
                             t,
                             [&] {
                                 if (t == 0) {
                                     observer.visit(W::exchange, step, t);
                                     queries->plan_send(peers, transport);
                                     queries->post_counts();
                                 }
                             })
                       && phase(failure,
                                t,
                                [&] {
                                    observer.visit(W::pack, step, t);
                                    pack_blocks<size_t>(*queries, *frame.work->payload, peers, threads, t);
                                    // Test-only: an observer that asks may rewrite this owner's slices.
                                    if constexpr (requires { observer.queries_packed(step, t, *queries); }) {
                                        observer.queries_packed(step, t, *queries);
                                    }
                                })
                       && phase(failure, t, [&] {
                              if (t == 0) {
                                  observer.visit(W::exchange, step, t);
                                  queries->wait_counts();
                                  queries->plan_recv();
                                  queries->post();
                                  queries->wait();
                              }
                          });
            };
            // R1-R3: the answers remote sources wait for, laid out from the rows the resolvers wrote.
            const auto answer_round = [&]() -> bool {
                if (!remote) {
                    return true;
                }
                return phase(failure,
                             t,
                             [&] {
                                 if (t == 0) {
                                     observer.visit(W::exchange, step, t);
                                     answers->plan_send(peers, transport);
                                     answers->plan_recv();
                                 }
                             })
                       && phase(failure,
                                t,
                                [&] {
                                    observer.visit(W::pack, step, t);
                                    pack_blocks<Response>(*answers, frame.work->answers, peers, threads, t);
                                })
                       && phase(failure, t, [&] {
                              if (t == 0) {
                                  observer.visit(W::exchange, step, t);
                                  answers->post();
                                  answers->wait();
                              }
                          });
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
                ok = phase(failure,
                           t,
                           [&] {
                               if (t == 0) {
                                   for (size_t s = 0; s < threads; ++s) {
                                       traverse(s);
                                   }
                               }
                           })
                     && phase(failure, t, prepare_leaders);
            }
            if (!ok || !query_round()) {
                return;
            }
            // P2: every destination resolves the leader payloads published at P1.
            if (cross && !phase(failure, t, [&] { resolve(true); })) {
                return;
            }
            if (!answer_round()) {
                return;
            }
            // P3: leader answers consumed; the follower filter below reads this owner's completed leader marks.
            if (passes && !phase(failure, t, [&] {
                    if (cross) {
                        consume(true);
                    }
                    observer.visit(W::prepare_followers, step, t);
                    auto &work = *frame.work;
                    auto &streams = work.scan.streams;
                    work.payload = &work.engine->prepare_exchange(false,
                                                                  std::move(streams.follower_queries),
                                                                  std::move(streams.follower_src),
                                                                  std::move(streams.follower_val),
                                                                  std::move(streams.follower_self));
                    post_blocks<size_t>(*work.payload,
                                        t,
                                        first_local,
                                        threads,
                                        std::span<std::span<const size_t>>(query_inbox));
                    if (remote) {
                        write_send_rows<size_t>(*queries, *work.payload, peers, threads, t);
                    }
                })) {
                return;
            }
            if (!query_round()) {
                return;
            }
            // P4: every destination resolves the follower payloads published at P3.
            if (cross && !phase(failure, t, [&] { resolve(false); })) {
                return;
            }
            if (!answer_round()) {
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
                        if constexpr (requires(const FusedGateView &view) { observer.fused_records(step, t, view); }) {
                            observer.fused_records(
                                step,
                                t,
                                FusedGateView{.records = work.fc,
                                              .cos = work.cos,
                                              .coeffs = *frame.coeffs,
                                              .apply_angle = schrodinger ? -build_angle : build_angle,
                                              .schrodinger = schrodinger,
                                              .fused_scale = work.scan.fused_scale});
                        }
                        apply_fused_contract(work.fc,
                                             *frame.coeffs,
                                             work.cos,
                                             schrodinger ? -build_angle : build_angle,
                                             schrodinger,
                                             work.scan.fused_scale,
                                             kSerial,
                                             frame.kernels);
                    }
                    else if constexpr (Informed) {
                        // The cosine set is not persisted on the layer; it drives only this gate's replay.
                        work.core = work.engine->finish(std::move(work.scan.cos_all), &work.cos);
                        stamp_layer_metadata<NumModes>(*work.core, gen, own.op);
                        append(own, work.core, idx);
                        observer.visit(W::replay, step, t);
                        // After the inserts, before the new layer's endpoints are read.
                        own.op.extend_from_current_picture(*frame.coeffs, schrodinger);
                        const double build_angle = angle_of(idx);
                        const double apply_angle = schrodinger ? -build_angle : build_angle;
                        forward_publish(*frame.coeffs,
                                        LayerTraversal(*work.core),
                                        flat,
                                        frame.replay_scratch,
                                        frame.replay_published[0]);
                        if (multirank) {
                            // The cosine pass waits for the replay round's post, so the transfer overlaps it.
                            write_replay_rows(*replay, frame.replay_published[0], t);
                        }
                        else {
                            forward_scale_mask(*frame.coeffs, work.cos, apply_angle);
                        }
                    }
                    else {
                        auto storage = work.engine->finish(std::move(work.scan.cos_all), nullptr);
                        stamp_layer_metadata<NumModes>(*storage, gen, own.op);
                        append(own, std::move(storage), idx);
                    }
                })) {
                return;
            }
            // P6 (informed): finish the new layer's replay from the partners' snapshots published at P5. With other
            // processes, the replay round runs first, and the owners' cosine passes run while it is in flight.
            if constexpr (Informed) {
                if (multirank
                    && !replay_round.run(
                        failure,
                        t,
                        [&] {
                            observer.visit(W::pack, step, t);
                            pack_replay_column(*replay, replay_pairs, 0, t);
                        },
                        [&] {
                            const double build_angle = angle_of(idx);
                            forward_scale_mask(*frame.coeffs,
                                               frame.work->cos,
                                               schrodinger ? -build_angle : build_angle);
                        })) {
                    return;
                }
                if (!phase(failure, t, [&] {
                        observer.visit(W::replay, step, t);
                        const double build_angle = angle_of(idx);
                        const EndpointBoard board{.owners = replay_pairs,
                                                  .buffer = 0,
                                                  .first_local = first_local,
                                                  .remote = replay};
                        forward_finish(*frame.coeffs,
                                       LayerTraversal(*frame.work->core),
                                       schrodinger ? -build_angle : build_angle,
                                       flat,
                                       frame.replay_scratch,
                                       board);
                        // Test-only: an observer that asks sees the evolving coefficients; production has no hook.
                        if constexpr (requires { observer.replayed(t, step, std::as_const(*frame.coeffs)); }) {
                            observer.replayed(t, step, std::as_const(*frame.coeffs));
                        }
                    })) {
                    return;
                }
            }
        }

        // The last phase: its decision ends the sequence either way.
        static_cast<void>(phase(failure, t, [&] {
            observer.visit(W::caches, kNoStep, t);
            frame.work.reset();
            own.op.initialize_caches(schrodinger);
        }));
    });
    // A request still live after a failed join belongs to a round some peer may never complete: hand the failure to
    // the distributed policy now, before any request owner is destroyed (see the header comment).
    const auto live = [](const PhysicalExchange *round) { return round != nullptr && round->live() != 0; };
    if (error && (live(queries) || live(answers) || live(replay))) {
        mpi::operation_failed(world.comm, error);
    }
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
    construction_detail::check_arguments(
        "build_graph",
        options,
        shards,
        ctx,
        gates,
        {circuit.parameter_mapping.size(), circuit.gen_coeffs.size(), circuit.gate_indices.size()},
        circuit.only_rotate_len_k);
    if (gates == 0) {
        return {};
    }
    return construction_detail::run_gates<NumModes, construction_detail::GateMode::graph>(
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
 * \brief Coefficient-informed graph construction: build_graph() with the lower/upper atol cutoffs applied to the
 *        evolving coefficients.
 *
 * The legacy evolve_mode_graph_with_coeffs_() at geometry (1, T), in one team for the seed and the whole circuit.
 * Each owner's seed is a copy of its current picture; when the shards already hold layers, every shard replays its
 * existing graph from that copy at `seed_params` (replay_forward_in_team(), no nested team), as contract_partially()
 * would. Each gate then traverses with the coefficients and its build angle `mapped_params[i]`, appends the layer
 * with unchanged metadata (as build_graph()), extends the coefficients to the new rows, and replays the new layer over
 * its transient cosine set at the picture's apply angle (negated for Schrödinger), so the next gate's cutoff
 * decisions see the evolved values. The coefficients are then discarded; after the last gate every shard's caches
 * are warmed. An empty circuit returns at once, without a team.
 *
 * \param options  The shards' captured budget; its thread count T is the team size and the shard count.
 * \param shards   The process's T shards; mutated in place, each by its owner.
 * \param ctx      Rank-level configuration; its atols apply.
 * \param circuit  The validated circuit and angles; see InformedCircuit.
 * \param observer Test-only seam; see NoConstructionObserver. Seed replay steps report ConstructionWork::seed.
 * \return The outcome after the team has joined; see ConstructionOutcome. Phase failures are returned, not thrown.
 * \throws As build_graph(), before the team, with nothing mutated; std::invalid_argument if `mapped_params` does not
 *         have one angle per gate, or if `seed_params` is absent while the shards hold layers or does not have one
 *         angle per existing layer of every shard.
 */
template <size_t NumModes, class Observer = NoConstructionObserver>
[[nodiscard]] auto build_graph_informed(parallel::Options options,
                                        Shards<NumModes> &shards,
                                        const ConstructionContext<NumModes> &ctx,
                                        const InformedCircuit<NumModes> &circuit,
                                        const Observer &observer = {}) -> ConstructionOutcome {
    const auto &graph = circuit.graph;
    const size_t gates = graph.generators.size();
    construction_detail::check_arguments("build_graph_informed",
                                         options,
                                         shards,
                                         ctx,
                                         gates,
                                         {graph.parameter_mapping.size(),
                                          graph.gen_coeffs.size(),
                                          graph.gate_indices.size(),
                                          circuit.mapped_params.size()},
                                         graph.only_rotate_len_k);
    if (gates == 0) {
        return {};
    }
    const bool has_layers = std::ranges::any_of(shards, [](const auto &state) { return state->graph.layers() > 0; });
    if (has_layers != circuit.seed_params.has_value()
        || (circuit.seed_params && std::ranges::any_of(shards, [&](const auto &state) {
                return state->graph.layers() != circuit.seed_params->size();
            }))) {
        throw std::invalid_argument(
            "sharded::build_graph_informed: the seed angles must be given exactly when the shards "
            "hold layers, one per existing layer of every shard");
    }
    const std::span<const double> *seed = circuit.seed_params ? &*circuit.seed_params : nullptr;
    return construction_detail::run_gates<NumModes, construction_detail::GateMode::informed>(
        options,
        shards,
        ctx,
        graph.generators,
        graph.only_rotate_len_k,
        observer,
        [&](size_t idx) { return circuit.mapped_params[idx]; },
        [&](ShardState<NumModes> &state, std::shared_ptr<LayerCore> storage, size_t idx) {
            state.graph.append(std::move(storage),
                               graph.parameter_mapping[idx],
                               graph.gen_coeffs[idx],
                               graph.gate_indices[idx]);
        },
        seed);
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
    return construction_detail::run_gates<NumModes, construction_detail::GateMode::propagate>(
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
