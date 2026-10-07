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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/MPGraphEncoding.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/pare/PareGraph.h"
#include "monoprop/detail/sharded/Exchange.h"
#include "monoprop/detail/sharded/State.h"
#include "monoprop/detail/sharded/Team.h"
#include "monoprop/monopropExport.h"

/*
 * Snapshot-safe replay, energy, gradients and retained functionals over the T shards of one process, at geometry
 * (P, T).
 *
 * One run_team() spans a whole evaluation, forward and reverse loops included. Worker t owns shard t, whose flat
 * routing slot is rank * T + t; every other shard is a cross owner, in this process or another. A replay step of layer
 * l is split at one checkpoint: its begin (snapshot and publish the endpoint values partners read, then the cosine
 * pass) and its finish (read the partners' published snapshots, add the sine terms). Finishes read published
 * snapshots, never a partner's live coefficients, which that partner's cosine pass has already overwritten.
 *
 * Other processes (P > 1). Every step then also moves the blocks of partners on other ranks through one physical
 * round (Exchange.h, ReplayRound), with the communicator-agreed transport: the begin publishes and writes the owner's
 * replay rows (send and receive counts are equal: a layer's partner layout is symmetric); a primary phase lays the
 * round out over every other rank; owners pack their remote blocks from the published snapshots; then the primary
 * posts, every owner runs its callback part (record, cosine pass or reverse accumulation) while the transfer is in
 * flight, and the primary completes it. The finish then reads remote partners' blocks from the round's receive
 * staging, which the next step's round rewrites only after that finish. Each step thus has four checkpoints instead
 * of one; P = 1 keeps the one-checkpoint sequence.
 *
 * Phase p (one checkpoint) = finish of step p-1, then, at p = L, the owner's local contribution, then the begin of
 * step p. Steps 0 .. L-1 are the forward replay; a gradient continues with the reverse steps L .. 2L-1 (reverse layer
 * L-1 first). Step p publishes into buffer p % 2 of its owner's pair:
 *
 * | Buffer                         | Writer  | Published      | Readers           | Last read   | Earliest reuse     |
 * | ------------------------------ | ------- | -------------- | ----------------- | ----------- | ------------------ |
 * | published[p % 2] (step p)      | owner t | end of phase p | partners' finish  | phase p + 1 | begin, phase p + 2 |
 * | forward self sources           | owner t | (private)      | owner t (finish)  | phase p + 1 | begin, phase p + 1 |
 * | derivative snapshots, A        | owner t | (private)      | owner t (finish)  | phase p + 1 | begin, phase p + 1 |
 * | partner staging (finish)       | owner t | (private)      | owner t (finish)  | phase p + 1 | next finish        |
 * | records (gradient)             | owner t | (private)      | owner t (reverse) | last phase  | next call          |
 * | working op / state             | owner t | (private)      | owner t           | last phase  | next call          |
 * | contributions[t], gradients[t] | owner t | join           | caller            | after join  | caller's           |
 *
 * Working storage is thread-local to the executing worker thread, retained across calls and rebound and reset in the
 * owner's first phase of every call: an OpenMP worker index is not a permanent OS-thread identity. Frames, published
 * pairs and results stay alive through the join on failure. Hot paths reach the frame through the pointer the frame
 * phase stores, never through the thread-local itself. A finish copies its partners' published blocks, in runs of
 * consecutive partners, into the owner's partner staging before applying them (PartnerStaging), and a gradient's frame
 * phase reserves the records exactly (replay::reserve_records()), so recording never reallocates.
 *
 * Callbacks: a set marked CosCallbacks::owner_parallel (built by detail::make_cos_callbacks) runs on its owner, in
 * parallel with the other owners. If any shard's set is opaque, every callback call (records, cosine pass, reverse
 * accumulation) runs on the primary, shard by shard, in an exclusive phase every worker enters after each begin;
 * snapshots, publication, finishes and contributions stay owner-parallel. Within-shard kernels are serial.
 *
 * Failure: arguments are checked before the team, with nothing mutated. Every call that can throw runs inside a
 * phase; a failure suppresses every later phase body, the team joins, and the lowest failing worker's original
 * exception is returned by evaluate_shards() / replay_shards(), and delivered by ev_sharded() /
 * ev_and_grad_sharded() through mpi::operation_failed() before any result reduction.
 */

namespace monoprop::detail::sharded {

//! The step reported for work outside a step loop.
inline constexpr size_t kNoStep = std::numeric_limits<size_t>::max();

//! The owner work an evaluation reports to its observer.
enum class EvaluationWork : std::uint8_t {
    frame,              //!< Rebind and reset the owner's working frame; copy its operator (and state).
    record,             //!< Forward step: take the selective pre-layer record.
    publish,            //!< Forward step: snapshot own sources, publish partner endpoints.
    cosine,             //!< Forward step: the cosine pass (callback).
    finish,             //!< Forward step: add partners' published and own snapshotted sine terms.
    contribution,       //!< The owner's local energy term.
    reverse_publish,    //!< Reverse step: snapshot endpoints, publish partner pairs.
    reverse_accumulate, //!< Reverse step: predivide, cosine accumulation (callback), restore.
    reverse_finish,     //!< Reverse step: self pairs, partner endpoints, the local parameter contribution.
    retain,             //!< Retained preparation: snapshot, pare and build callbacks.
};

/*!
 * \brief Test-only observation seam; production passes none.
 *
 * `visit(work, step, shard)` runs inside the protected phase, on the thread that performs `work` for `shard` (its
 * owner, or the primary for an opaque callback), before that work. `step` is the replay step or kNoStep. An
 * exception it throws is that phase's failure.
 */
//! The values one run of staged partner blocks holds at least: small enough to stay cache-resident until applied.
inline constexpr size_t kStagingRunValues = 4096;

class EvaluationObserver {
public:
    EvaluationObserver() = default;                                               //!< Stateless base.
    EvaluationObserver(const EvaluationObserver &) = default;                     //!< Copyable.
    auto operator=(const EvaluationObserver &) -> EvaluationObserver & = default; //!< Copyable.
    EvaluationObserver(EvaluationObserver &&) = default;                          //!< Movable.
    auto operator=(EvaluationObserver &&) -> EvaluationObserver & = default;      //!< Movable.
    virtual ~EvaluationObserver() = default;                                      //!< Polymorphic base.

    //! Called before `work` of `step` for `shard`; may throw.
    virtual auto visit(EvaluationWork work, size_t step, size_t shard) const -> void = 0;

    //! Test-only: the minimum staging run of every owner's finishes, so small layers can cover multi-run staging.
    [[nodiscard]] virtual auto staging_run() const noexcept -> size_t { return kStagingRunValues; }
};

// --- Owner-local replay phases ----------------------------------------------------------------------------------

/*!
 * \brief One owner's endpoint values for one replay step, published to its partners at a checkpoint.
 */
struct PublishedEndpoints {
    VecD values;                //!< Per partner slot, its block of pre-cosine values (pairs for a reverse step).
    LayerExchangeLayout layout; //!< Blocks over flat slots: counts and offsets; the owner's own slot is empty.
};

//! The double buffer one owner publishes through: step p uses entry p % 2.
using PublishedPair = std::array<PublishedEndpoints, 2>;

/*!
 * \brief Read-only view of the publications for one replay step: the local owners' pairs, and the step's physical
 *        round for partners on other ranks.
 */
struct EndpointBoard {
    std::span<const PublishedPair *const> owners; //!< Per local shard, its publication pair.
    size_t buffer = 0;                            //!< The entry of each pair that holds this step.
    size_t first_local = 0;                       //!< Flat slot of local shard 0, `rank * T`.
    const PhysicalExchange *remote = nullptr;     //!< The step's completed round, or null when P = 1.

    /*!
     * \brief The block `source` published for `reader` at this step.
     *
     * \param source The partner's flat slot.
     * \param reader The reading owner's flat slot; a shard of this process.
     * \param count  The number of values the reader's layer expects from `source`.
     * \throws std::logic_error if `source` is neither a local shard nor a shard of a peer of `remote`, or its block
     *         does not hold exactly `count` values.
     */
    [[nodiscard]] auto block(size_t source, size_t reader, size_t count) const -> const double * {
        if (source < first_local || source - first_local >= owners.size()) {
            return remote_block_(source, reader, count);
        }
        const auto &published = (*owners[source - first_local])[buffer];
        if (reader >= published.layout.counts.size() || static_cast<size_t>(published.layout.counts[reader]) != count) {
            throw std::logic_error(
                std::format("sharded replay: partner {} did not publish {} values for slot {}", source, count, reader));
        }
        return published.values.data() + published.layout.displs[reader];
    }

private:
    [[nodiscard]] auto remote_block_(size_t source, size_t reader, size_t count) const -> const double * {
        if (remote == nullptr) {
            throw std::logic_error(
                std::format("sharded replay: partner slot {} is not a shard of this process", source));
        }
        const size_t threads = remote->threads();
        const auto peers = remote->peers();
        const auto found = std::ranges::lower_bound(peers, source / threads);
        if (found == peers.end() || *found != source / threads) {
            throw std::logic_error(std::format("sharded replay: partner slot {} is on no peer of the round", source));
        }
        const auto received = remote->recv_block<double>(reader - first_local,
                                                         static_cast<size_t>(found - peers.begin()),
                                                         source % threads);
        if (received.size() != count) {
            throw std::logic_error(std::format("sharded replay: partner {} sent {} values for slot {}, expected {}",
                                               source,
                                               received.size(),
                                               reader,
                                               count));
        }
        return received.data();
    }
};

/*!
 * \brief Owner `shard`'s rows of a replay round from its publication: what it sends to and receives from every shard
 *        of every other rank, then its row totals (PhysicalExchange::close_rows()). A layer's partner layout is
 *        symmetric, so both counts are the published block size.
 */
monoprop_EXPORT auto write_replay_rows(PhysicalExchange &round, const PublishedEndpoints &out, size_t shard) -> void;

/*!
 * \brief Owner `shard` copies its published blocks for every shard of every other rank into the round's send slices.
 *
 * Requires the round's offsets for this owner's row, which plan_send() lays out; ReplayRound plans owner-parallel
 * and packs by column instead (pack_replay_column()).
 */
monoprop_EXPORT auto pack_replay_blocks(PhysicalExchange &round, const PublishedEndpoints &out, size_t shard) -> void;

/*!
 * \brief Owner `shard` copies every local owner's published block for remote shard `shard` of every other rank into
 *        the round's send slices: column `shard`, whose offsets it placed itself (PhysicalExchange::place_column()).
 *
 * Reads other owners' publications, which are immutable from their publication checkpoint until consumed.
 *
 * \param round  The step's round, planned with plan_totals().
 * \param owners Per local shard, its publication pair; every entry non-null.
 * \param buffer The entry of each pair that holds this step.
 * \param shard  This owner, the remote shard index of the column it fills.
 */
monoprop_EXPORT auto pack_replay_column(PhysicalExchange &round,
                                        std::span<const PublishedPair *const> owners,
                                        size_t buffer,
                                        size_t shard) -> void;

/*!
 * \brief A replay step's physical round with the communicator-agreed transport; inert (null) when P = 1.
 */
struct ReplayRound {
    PhysicalExchange *round = nullptr;                         //!< Caller-owned; outlives the team.
    ExchangeTransport transport = ExchangeTransport::pairwise; //!< mpi::routes_pairwise() of the communicator.

    /*!
     * \brief The round's three phases, after the phase in which every owner wrote its rows (write_replay_rows()).
     *
     * The round is planned owner-parallel over every other rank: the primary sizes it from the owners' row totals
     * while every owner sums its column; every owner then places its column's offsets and runs `pack()`, which fills
     * that column (pack_replay_column()); finally the primary posts, every owner runs `overlap()` while the transfer
     * is in flight, and the primary completes it. Every worker must call this at the same point of its sequence.
     *
     * \return The last checkpoint decision, identical on every worker; on false the caller ends its sequence.
     */
    template <class Pack, class Overlap>
    auto run(TeamFailure &failure, size_t shard, Pack &&pack, Overlap &&overlap) const noexcept -> bool {
        return phase(failure,
                     shard,
                     [&] {
                         if (shard == 0) {
                             const auto peers =
                                 other_ranks(PhysicalWorld{.rank = round->rank(), .ranks = round->ranks()});
                             round->plan_totals(peers, transport);
                         }
                         round->plan_column(shard);
                     })
               && phase(failure,
                        shard,
                        [&] {
                            round->place_column(shard);
                            pack();
                        })
               && phase(failure, shard, [&] {
                      if (shard == 0) {
                          round->post();
                      }
                      overlap();
                      if (shard == 0) {
                          round->wait();
                      }
                  });
    }
};

/*!
 * \brief One owner's staging of its partners' blocks while it finishes a step.
 *
 * Private to the owner and reused across steps. `values` holds one run of consecutive partner blocks: the larger of
 * `run_values` and the step's largest block.
 */
struct PartnerStaging {
    //! One partner of the step.
    struct Block {
        size_t rank = 0;              //!< The partner's flat slot.
        size_t count = 0;             //!< The values it published for this owner.
        const double *data = nullptr; //!< Where the apply reads them, once staged.
    };
    std::vector<size_t> position;          //!< Per flat slot: its index in `blocks`.
    std::vector<Block> blocks;             //!< The step's partners, in the order the apply visits them.
    VecD values;                           //!< The current run.
    size_t run_values = kStagingRunValues; //!< The minimum run; EvaluationObserver::staging_run() in tests.
};

//! One owner's private forward-step scratch: its self slot's pre-cosine sources and its partner staging.
struct ForwardScratch {
    VecD self_sources;       //!< The self slot's sin_send values before the cosine pass.
    PartnerStaging partners; //!< Partner blocks staged by the finish; the reverse finish reuses it.
};

/*!
 * \brief Begin of a forward step, before the cosine pass: snapshot the self slot and publish every partner's block.
 *
 * Owner-local; reads `coeffs`, writes `scratch` and `out` only.
 *
 * \param coeffs     The owner's coefficients, pre-layer.
 * \param layer      The owner's layer.
 * \param flat_owner The owner's flat slot.
 * \param scratch    The owner's scratch; its self snapshot is overwritten.
 * \param out        The publication for this step; overwritten, then read by partners until their finish.
 */
monoprop_EXPORT auto forward_publish(const VecD &coeffs,
                                     const LayerTraversal &layer,
                                     size_t flat_owner,
                                     ForwardScratch &scratch,
                                     PublishedEndpoints &out) -> void;

/*!
 * \brief The cosine pass of a forward step: `scale(layer_idx, coeffs, cos(2 * param))`.
 */
monoprop_EXPORT auto forward_scale(VecD &coeffs, size_t layer_idx, double param, const LayerCosScale &scale) -> void;

/*!
 * \brief The cosine pass of a forward step over a transient cosine mask, with serial kernels.
 */
monoprop_EXPORT auto forward_scale_mask(VecD &coeffs, const CosMask &cos, double param) -> void;

/*!
 * \brief Finish of a forward step, after every partner published and after the cosine pass.
 *
 * Adds `sin(2 * param)` times each partner's published pre-cosine value into this owner's targets, partners in
 * ascending slot order, then the self slot's snapshotted sources.
 *
 * \param coeffs     The owner's coefficients, already cos-scaled.
 * \param layer      The same layer as the begin.
 * \param param      The replay angle.
 * \param flat_owner The owner's flat slot.
 * \param scratch    The owner's scratch from the begin; its partner staging is overwritten.
 * \param board      The step's publications.
 */
monoprop_EXPORT auto forward_finish(VecD &coeffs,
                                    const LayerTraversal &layer,
                                    double param,
                                    size_t flat_owner,
                                    ForwardScratch &scratch,
                                    const EndpointBoard &board) -> void;

/*!
 * \brief One owner's entry of a multi-step forward replay run inside an existing team.
 */
struct ForwardReplayOwner {
    VecD *coeffs = nullptr;                  //!< The owner's coefficients, evolved in place.
    const MPGraphView *graph = nullptr;      //!< The replay window; must outlive the replay.
    const CosCallbacks *callbacks = nullptr; //!< Its `scale` is the cosine pass.
    ForwardScratch scratch;                  //!< Private.
    PublishedPair published;                 //!< Read by partners.
};

/*!
 * \brief A forward replay of every shard of a running team, sized and owned by the caller.
 */
struct ForwardReplayJob {
    std::span<const double> params;         //!< Replay angle of each step, shared by every shard.
    size_t first_local = 0;                 //!< Flat slot of local shard 0.
    bool owner_parallel = true;             //!< False: every cosine pass runs on the primary.
    std::vector<ForwardReplayOwner> owners; //!< One per shard; entry t is filled by owner t.
    ReplayRound remote{};                   //!< The physical round for partners on other ranks, or inert at P = 1.
};

/*!
 * \brief Replay `job.params.size()` forward steps of every shard inside the calling team; opens no team.
 *
 * Every worker of the team must call it with its own shard, after the checkpoint that published every entry of
 * `job.owners`. Uses `params.size() + 1` checkpoints (twice as many cosine phases with opaque callbacks), each
 * through phase(); its return value is the last checkpoint decision, and on false the caller must end its sequence.
 *
 * \param failure  The team's failure state.
 * \param shard    The calling worker's shard.
 * \param job      The replay; the caller keeps it alive through the join.
 * \param observer Test-only seam, or null.
 * \return Whether every checkpoint passed, identically on every worker.
 */
monoprop_EXPORT auto replay_forward_in_team(TeamFailure &failure,
                                            size_t shard,
                                            ForwardReplayJob &job,
                                            const EvaluationObserver *observer = nullptr) noexcept -> bool;

// --- Rank-level evaluation --------------------------------------------------------------------------------------

/*!
 * \brief What evaluate_shards() reports after its team has joined.
 */
struct EvaluationOutcome {
    std::exception_ptr error;          //!< The lowest failing worker's original exception, or empty.
    std::vector<double> contributions; //!< Per shard: its local energy term (identity excluded).
    std::vector<VecD> gradients;       //!< Per shard: its local gradient; empty for an energy-only evaluation.
};

/*!
 * \brief Evaluate T shards in one team and return their local contributions, without combining them.
 *
 * The S4 seam beneath ev_sharded() and ev_and_grad_sharded(). Request t must view shard t's retained state,
 * operator and graph; parameters, mapping, generator coefficients and identity are replicated. Energy-only
 * evaluation contracts with EvalState::dot (sparse stays sparse); a gradient scatters the state densely and
 * contracts with inner_product(). Empty parameters use EvalState::dot and need no callbacks.
 *
 * \param requests  One request per shard, in shard order.
 * \param callbacks One callback set per shard, built for that shard's index and graph.
 * \param options   The captured budget; its thread count T is the team size.
 * \param gradient  Whether to run the reverse pass.
 * \param observer  Test-only seam, or null.
 * \param world     This process among the ranks the shards' graphs were built over (PhysicalWorld::of() on the
 *                  caller); the default is one process. Every rank of a multi-rank world must call this together.
 * \return The outcome; a phase failure is returned, never thrown, except that a failure leaving a physical round's
 *         requests live goes to mpi::operation_failed() before the round is destroyed.
 * \throws std::invalid_argument before the team, with nothing mutated, for mismatched counts or shapes;
 *         MissingLayerCallback for a missing callback the path needs; EvalStateArgumentError for a state longer
 *         than its operator.
 */
monoprop_EXPORT auto evaluate_shards(std::span<const EvalRequest> requests,
                                     std::span<const CosCallbacks> callbacks,
                                     parallel::Options options,
                                     bool gradient,
                                     const EvaluationObserver *observer = nullptr,
                                     const PhysicalWorld &world = {},
                                     PhysicalRounds *rounds = nullptr) -> EvaluationOutcome;

/*!
 * \brief The reference fold of per-shard scalars: `0.0 + c[0] + c[1] + ...`, in ascending shard order.
 */
monoprop_EXPORT auto combine_contributions(std::span<const double> contributions) -> double;

/*!
 * \brief Per component, the reference fold of per-shard gradients in ascending shard order.
 * \throws std::invalid_argument if the shards' gradients differ in length.
 */
monoprop_EXPORT auto combine_gradients(std::span<const VecD> gradients) -> VecD;

/*!
 * \brief Expectation value over T shards: the replicated identity plus the ascending-shard fold of their terms,
 *        reduced over `comm`.
 *
 * One team spans the whole replay. Outside the team the communicator is used only on the caller: its rank, size and
 * agreed replay transport are read before the team (PhysicalWorld::of()), and the result reduction runs after the
 * join. Inside it, only the primary posts and completes the replay rounds. A phase failure goes to
 * mpi::operation_failed(comm, error) before the reduction, which at one rank rethrows the original exception.
 *
 * \param requests  One request per shard; see evaluate_shards().
 * \param callbacks One callback set per shard.
 * \param options   The captured budget; its thread count T is the team size.
 * \param comm      The ordinary physical communicator of P ranks the shards' graphs were built over.
 * \param observer  Test-only seam, or null.
 * \throws As evaluate_shards() before the team, and std::invalid_argument if `comm` is not an ordinary communicator.
 */
monoprop_EXPORT auto ev_sharded(std::span<const EvalRequest> requests,
                                std::span<const CosCallbacks> callbacks,
                                parallel::Options options,
                                mpi::Comm comm,
                                const EvaluationObserver *observer = nullptr) -> double;

/*!
 * \brief As ev_sharded(), plus the full gradient: per component, the ascending-shard fold, reduced over `comm`.
 */
monoprop_EXPORT auto ev_and_grad_sharded(std::span<const EvalRequest> requests,
                                         std::span<const CosCallbacks> callbacks,
                                         parallel::Options options,
                                         mpi::Comm comm,
                                         const EvaluationObserver *observer = nullptr) -> std::pair<double, VecD>;

// --- Partial contraction ----------------------------------------------------------------------------------------

/*!
 * \brief One shard's input to replay_shards(): the coefficients to evolve and the replay window.
 */
struct ReplayRequest {
    const VecD &coeffs; //!< Un-evolved coefficients; copied, never modified.
    MPGraphView graph;  //!< Replay window over layers the caller keeps alive.
};

/*!
 * \brief What replay_shards() reports after its team has joined.
 */
struct ReplayOutcome {
    std::exception_ptr error; //!< The lowest failing worker's original exception, or empty.
    std::vector<VecD> coeffs; //!< Per shard: its evolved coefficients, allocated by its owner.
};

/*!
 * \brief Evolve every shard's coefficients through its replay window in one team: the partial-contraction seam.
 *
 * `mapped_params` gives each replay step's angle. For a Heisenberg partial contraction they are
 * `map_params(parameters, mapping, gen_coeffs, 1.0, true)` over `graph.slice_view(k)`; for Schrödinger
 * `map_params(parameters, mapping, gen_coeffs, -1.0)` over the reversed `slice_view(k)`, starting from the dense
 * state. Shard coefficient blocks concatenated in shard order form the rank-local result.
 *
 * `world` is this process among the ranks the graphs were built over, as for evaluate_shards(); every rank of a
 * multi-rank world must call this together, and a failure leaving the round's requests live goes to
 * mpi::operation_failed() before the round is destroyed.
 *
 * \throws std::invalid_argument before the team for mismatched counts or window lengths, MissingLayerCallback for
 *         a missing `scale` when there is a step to replay.
 */
monoprop_EXPORT auto replay_shards(std::span<const ReplayRequest> requests,
                                   std::span<const double> mapped_params,
                                   std::span<const CosCallbacks> callbacks,
                                   parallel::Options options,
                                   const EvaluationObserver *observer = nullptr,
                                   const PhysicalWorld &world = {},
                                   PhysicalRounds *rounds = nullptr) -> ReplayOutcome;

// --- Retained functionals ---------------------------------------------------------------------------------------

/*!
 * \brief Rank-level inputs of a retained functional, replicated over the shards.
 */
struct RetainedContext {
    double core_term = 0.0;               //!< The replicated identity coefficient, added once per evaluation.
    VecZ parameter_mapping;               //!< Optimizer order, as MonomialPropagator::graph_gate_arrays_().
    VecD gen_coeffs;                      //!< Optimizer order, parallel to parameter_mapping.
    std::optional<double> pare_threshold; //!< Pare each shard's graph against this threshold, if set.
    Basis basis = Basis::Majorana;        //!< Fold encoding of the callbacks.
    bool schrodinger = false;             //!< Picture: which vector is thresholded, and the state's form.
    size_t rank = 0;                      //!< This process's rank; flat owner of shard t is rank * T + t.
};

/*!
 * \brief One shard's retained evaluation inputs.
 *
 * `graph` is either a pared graph owned here or an aliasing handle to the shard's live graph. `callbacks` borrow the
 * shard's inverted index and `graph`'s stored masks: they must not outlive the shards, and `graph` keeps the masks
 * alive.
 */
struct RetainedShard {
    EvalState state;                      //!< Owned snapshot: sparse Heisenberg scores, or the dense Schrödinger state.
    VecD op;                              //!< Owned snapshot of the un-evolved operator coefficients.
    std::shared_ptr<const MPGraph> graph; //!< The replayed graph.
};

/*!
 * \brief A retained evaluation over T shards: what a functional captures, with views built only at call time.
 */
struct RetainedEvaluation {
    double core_term = 0.0;              //!< Replicated identity.
    VecZ parameter_mapping;              //!< Optimizer order.
    VecD gen_coeffs;                     //!< Optimizer order.
    size_t num_params = 0;               //!< Required parameter count.
    size_t expected_layers = 0;          //!< Graph layers at preparation (identical on every shard).
    parallel::Options options;           //!< The captured budget.
    std::vector<RetainedShard> shards;   //!< One per shard.
    std::vector<CosCallbacks> callbacks; //!< One per shard, parallel to `shards`.

    /*!
     * \brief The functional's own argument checks: parameter count and unchanged graph layer counts.
     * \throws As validate_functional_call() and validate_expected_graph_layers().
     */
    auto validate_call(const VecD &params) const -> void {
        validate_functional_call(params, num_params);
        for (const auto &shard : shards) {
            validate_expected_graph_layers(shard.graph->layers(), expected_layers);
        }
    }

    /*!
     * \brief One request per shard viewing this object's snapshots and `params`; valid while both live unchanged.
     */
    [[nodiscard]] auto requests(const VecD &params) const -> std::vector<EvalRequest> {
        std::vector<EvalRequest> out;
        out.reserve(shards.size());
        for (const auto &shard : shards) {
            out.push_back(EvalRequest{.e_core = core_term,
                                      .state = shard.state,
                                      .op = shard.op,
                                      .parameter_mapping = parameter_mapping,
                                      .gen_coeffs = gen_coeffs,
                                      .graph = shard.graph->replay_view(),
                                      .params = params,
                                      .parallel = parallel::Options{}});
        }
        return out;
    }
};

/*!
 * \brief What prepare_retained() reports after its team has joined.
 */
struct RetainedOutcome {
    std::exception_ptr error;                   //!< The lowest failing worker's original exception, or empty.
    bool mutation_started = false;              //!< Whether any owner began warming caches or paring.
    std::optional<RetainedEvaluation> retained; //!< Set on success.
};

/*!
 * \brief Snapshot, optionally pare, and build callbacks for every shard, each on its owner, in one team.
 *
 * Per shard, as MonomialPropagator::make_functional_() does for one store: the state (sparse Heisenberg scores, or
 * the dense Schrödinger state) with its length snapshotted, a copy of the operator coefficients, then -- with a
 * threshold -- the keep set of the picture's driving vector and pare_graph_owner() at the shard's flat slot, and the
 * callbacks, built with serial kernels. Preparation warms the shard's lazy caches, so a failure after the first
 * phase started leaves `mutation_started` set.
 *
 * \param options  The shards' captured budget.
 * \param shards   The process's T shards; their caches are warmed in place.
 * \param context  Rank-level inputs.
 * \param observer Test-only seam, or null.
 * \throws std::invalid_argument before the team if the shards do not fit `options` or disagree on layer counts.
 */
template <size_t NumModes>
auto prepare_retained(parallel::Options options,
                      Shards<NumModes> &shards,
                      const RetainedContext &context,
                      const EvaluationObserver *observer = nullptr) -> RetainedOutcome {
    if (options.threads < 1 || shards.size() != static_cast<size_t>(options.threads)
        || std::ranges::any_of(shards, [](const auto &state) { return !state; })) {
        throw std::invalid_argument(std::format("sharded::prepare_retained: expected {} non-null shards, got {}",
                                                options.threads,
                                                shards.size()));
    }
    const size_t layers = shards.front()->graph.layers();
    for (const auto &state : shards) {
        if (state->graph.layers() != layers) {
            throw std::invalid_argument("sharded::prepare_retained: the shards' graphs disagree on the layer count");
        }
    }
    if (context.parameter_mapping.size() != layers || context.gen_coeffs.size() != layers) {
        throw std::invalid_argument("sharded::prepare_retained: the gate arrays do not match the graph layers");
    }
    const size_t threads = shards.size();
    // Caller-owned slots: entry t is written by owner t only, and read after the join.
    std::vector<std::optional<RetainedShard>> slots(threads);
    std::vector<CosCallbacks> callbacks(threads);
    const parallel::Options serial{}; // within-shard kernels are serial
    bool mutation_started = false;
    const auto error = run_team(options, [&](size_t t, TeamFailure &failure) noexcept {
        if (t == 0) {
            mutation_started = true;
        }
        static_cast<void>(phase(failure, t, [&] {
            if (observer != nullptr) {
                observer->visit(EvaluationWork::retain, kNoStep, t);
            }
            auto &op = shards[t]->op;
            const auto num_terms = op.size();
            auto state = [&] {
                if (context.schrodinger) {
                    return EvalState::dense(op.dense_state());
                }
                const auto sparse = op.sparse_state();
                return EvalState::sparse(num_terms, sparse.rows, sparse.values);
            }();
            VecD coeffs = op.get_operator();
            const auto &inverted_index = op.inverted_index();
            std::shared_ptr<const MPGraph> graph;
            const MPGraph &live = shards[t]->graph;
            if (context.pare_threshold.has_value()) {
                const auto full_cos_of_layer = [&](size_t i) -> CosMask {
                    return full_cos_mask<NumModes>(inverted_index, live.get_layer_traversal(i), context.basis);
                };
                const auto keep = context.schrodinger ? indices_above(coeffs, *context.pare_threshold)
                                                      : state.indices_above(*context.pare_threshold);
                const auto count = context.schrodinger ? coeffs.size() : state.length();
                graph = std::make_shared<const MPGraph>(pare_graph_owner(live,
                                                                         keep,
                                                                         count,
                                                                         context.schrodinger,
                                                                         (context.rank * threads) + t,
                                                                         full_cos_of_layer));
            }
            else {
                graph = std::shared_ptr<const MPGraph>(std::shared_ptr<const void>{}, &live);
            }
            callbacks[t] = make_cos_callbacks<NumModes>(inverted_index, graph->replay_view(), context.basis, serial);
            slots[t].emplace(
                RetainedShard{.state = std::move(state), .op = std::move(coeffs), .graph = std::move(graph)});
        }));
    });
    RetainedOutcome outcome{.error = error, .mutation_started = mutation_started, .retained = std::nullopt};
    if (error) {
        return outcome;
    }
    // Assembled on the caller after the join; a failure here still follows the post-mutation outcome contract.
    try {
        RetainedEvaluation retained{.core_term = context.core_term,
                                    .parameter_mapping = context.parameter_mapping,
                                    .gen_coeffs = context.gen_coeffs,
                                    .num_params = expected_num_params(context.parameter_mapping),
                                    .expected_layers = layers,
                                    .options = options,
                                    .shards = {},
                                    .callbacks = std::move(callbacks)};
        retained.shards.reserve(threads);
        for (auto &slot : slots) {
            retained.shards.push_back(std::move(*slot));
        }
        outcome.retained.emplace(std::move(retained));
    }
    catch (...) {
        outcome.error = std::current_exception();
    }
    return outcome;
}

} // namespace monoprop::detail::sharded
