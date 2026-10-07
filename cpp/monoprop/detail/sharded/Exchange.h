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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/monopropExport.h"

/*
 * One physical-MPI round of the sharded runtime: the inter-process part of a handoff between the T owners of this
 * process and the T owners of each peer process. It is not a collective facade: it exposes physical post and wait
 * operations to the primary, and disjoint payload slices and read-only received blocks to the owners. Shards of the
 * same process never go through it; they read each other's published buffers directly.
 *
 * A round, in the phases of one team (O: every owner, inside its protected phase; P: the primary only):
 *
 *   O  reset_rows(t), then set_send_count(t, ...) for every remote destination, and set_recv_count(t, ...) when the
 *      receive counts are known (replay: the layer is symmetric; answers: one per query, known on both sides)
 *   P  plan_send(peers, transport); post_counts() when the receive counts are unknown
 *   O  pack send_block<T>(t, peer, dest)
 *   P  wait_counts() if counted, plan_recv(), post(), and later wait()
 *   O  read recv_block<T>(t, peer, source)
 *
 * with a team checkpoint between consecutive lines. Planning and MPI calls are serial on the primary; packing and
 * reading are owner-parallel over disjoint memory.
 *
 * A round whose receive counts its owners know (a replay step) can be planned owner-parallel instead, with the same
 * number of checkpoints and the same offsets as plan_send() + plan_recv(), but O(P x T) work on the primary instead of
 * O(P x T^2):
 *
 *   O  reset_rows(t), set_send_count / set_recv_count for every index of every other rank, then close_rows(t)
 *   P  plan_totals(peers, transport)        O  plan_column(t)
 *   O  place_column(t), then pack send_block<T>(u, peer, t) for every local source u: owner t fills column t
 *   P  post(), and later wait()
 *   O  read recv_block<T>(t, peer, source)
 *
 * Wire layout, as the legacy hybrid transport: the message to peer rank b holds, for each of b's shards t in
 * ascending order, each local source shard's block for (b, t) in ascending source order. Received messages have the
 * same layout, so a local destination's part of a message is contiguous and its blocks ascend by source shard.
 * Every count, per-peer total and displacement is a checked int; a violation throws before anything is posted.
 *
 * Transports: pairwise posts an Irecv and an Isend per non-empty leg of each peer (both ends skip the same legs,
 * since receive counts are exchanged or known to match the sender's); collective posts one MPI_Ialltoallv over the
 * whole communicator, which every rank joins even with nothing to send. The count round is always pairwise over the
 * peer set: T x T counts per peer, each peer's block in the narrowest of 1, 2 or 4 bytes per count that holds its
 * largest, so light rounds stay below the transport's eager limit; the receiver reads the width from the received size.
 *
 * MPI ownership: post_counts(), wait_counts(), post() and wait() throw std::logic_error unless they run on OpenMP
 * thread 0 and, in MPI builds, on the thread that initialized MPI. Only the primary of a team, which is the thread
 * that entered it, ever posts or waits.
 *
 * Lifetime: the state lives on the heap, so a move keeps every buffer and request MPI points into. The move
 * constructor adopts; move assignment drains the destination's live requests first; the destructor drains. A caller
 * that may fail while requests are live must keep the exchange alive until it has handed the failure to
 * mpi::operation_failed(), which aborts a multi-rank communicator: draining a request a failed peer never completes
 * would hang.
 */

namespace monoprop::detail::sharded {

//! Element type of a round's payload.
enum class ExchangeElement : std::uint8_t {
    u32, //!< 32-bit unsigned (graph answers: TermIndex).
    u64, //!< 64-bit unsigned (queries: size_t words).
    f64, //!< double (replay endpoints, fused answers).
};

//! How a round's payload is posted.
enum class ExchangeTransport : std::uint8_t {
    pairwise,   //!< Irecv and Isend per non-empty leg of each peer.
    collective, //!< One MPI_Ialltoallv over every rank of the communicator.
};

//! Tags of the sharded rounds, distinct from each other and from the legacy transports' (Pairwise.h).
inline constexpr int kShardedQueryTag = 0x6D75;
inline constexpr int kShardedAnswerTag = 0x6D76;
inline constexpr int kShardedReplayTag = 0x6D77;
inline constexpr int kShardedCountTag = 0x6D78; //!< Every count round.

//! The element kind of a payload type.
template <class T>
inline constexpr ExchangeElement exchange_element_of = [] {
    if constexpr (std::is_same_v<T, std::uint32_t>) {
        return ExchangeElement::u32;
    }
    else if constexpr (std::is_same_v<T, std::uint64_t>
                       || (std::is_same_v<T, std::size_t> && sizeof(std::size_t) == sizeof(std::uint64_t))) {
        return ExchangeElement::u64;
    }
    else {
        static_assert(std::is_same_v<T, double>, "a sharded round carries u32, u64 or double payloads");
        return ExchangeElement::f64;
    }
}();

/*!
 * \brief This process's place among the physical ranks of an ordinary communicator, read once on the caller.
 */
struct PhysicalWorld {
    mpi::Comm comm{};             //!< The ordinary communicator; must outlive every exchange built from this.
    size_t rank = 0;              //!< This process's rank.
    size_t ranks = 1;             //!< The communicator's size P.
    bool replay_pairwise = false; //!< The communicator-agreed replay transport (mpi::routes_pairwise()).

    /*!
     * \brief Rank, size and agreed replay transport of `comm`.
     *
     * Call on MPI's initializing thread, outside any team: the first call on a multi-rank communicator agrees the
     * routing configuration collectively and caches the answer on the communicator.
     *
     * \throws std::invalid_argument if `comm` is not an ordinary communicator; routing::RoutingDisagreement on every
     *         rank together if the ranks resolved different routing configurations.
     */
    monoprop_EXPORT static auto of(const mpi::Comm &comm) -> PhysicalWorld;
};

//! Every rank of `world` except this one, ascending: the peer set of a dense round.
[[nodiscard]] inline auto other_ranks(const PhysicalWorld &world) -> std::vector<size_t> {
    std::vector<size_t> peers;
    peers.reserve(world.ranks - 1);
    for (size_t b = 0; b < world.ranks; ++b) {
        if (b != world.rank) {
            peers.push_back(b);
        }
    }
    return peers;
}

/*!
 * \brief A reusable physical round between this process's T owners and those of a set of peer ranks.
 */
class PhysicalExchange {
public:
    monoprop_EXPORT
    PhysicalExchange() noexcept; //!< An inert exchange: every operation except destruction and assignment throws.

    /*!
     * \brief An exchange for `threads` owners per process, over the ranks of `world`.
     *
     * Allocates the count tables for every possible peer, P - 1 of them.
     *
     * \param world   The physical world; its communicator must outlive the exchange.
     * \param threads The team size T, identical on every rank.
     * \param element The payload type of every round.
     * \param tag     The payload tag; rounds that may be in flight together need different tags.
     * \throws std::invalid_argument if `threads` is zero or `world` has no ranks; std::bad_alloc.
     */
    monoprop_EXPORT PhysicalExchange(const PhysicalWorld &world, size_t threads, ExchangeElement element, int tag);

    PhysicalExchange(const PhysicalExchange &) = delete;                     //!< Owns requests: move-only.
    auto operator=(const PhysicalExchange &) -> PhysicalExchange & = delete; //!< Owns requests: move-only.
    monoprop_EXPORT PhysicalExchange(PhysicalExchange &&other) noexcept;     //!< Adopts, live requests included.
    //! Drains this exchange's live requests, then adopts `other`'s.
    monoprop_EXPORT auto operator=(PhysicalExchange &&other) noexcept -> PhysicalExchange &;
    monoprop_EXPORT ~PhysicalExchange(); //!< Drains live requests; see the header comment.

    [[nodiscard]] monoprop_EXPORT auto threads() const noexcept -> size_t; //!< T.
    [[nodiscard]] monoprop_EXPORT auto rank() const noexcept -> size_t;    //!< This process's rank.
    [[nodiscard]] monoprop_EXPORT auto ranks() const noexcept -> size_t;   //!< The communicator's size P.
    //! The peer ranks of the current round, ascending, as plan_send() received them.
    [[nodiscard]] monoprop_EXPORT auto peers() const noexcept -> std::span<const size_t>;
    //! Requests posted and not yet waited, payload and counts together.
    [[nodiscard]] monoprop_EXPORT auto live() const noexcept -> int;

    // --- Owner side: count rows (row `shard` has one writer, owner `shard`) ------------------------------------------

    //! Zero both of owner `shard`'s count rows, for every possible peer.
    monoprop_EXPORT auto reset_rows(size_t shard) -> void;
    /*!
     * \brief Set how many elements local `shard` sends to shard `dest` of peer number `peer`.
     * \throws std::out_of_range for an index outside the table; std::length_error for a count above INT_MAX.
     */
    monoprop_EXPORT auto set_send_count(size_t shard, size_t peer, size_t dest, size_t count) -> void;
    /*!
     * \brief Set how many elements local `shard` receives from shard `source` of peer number `peer`.
     * \throws As set_send_count().
     */
    monoprop_EXPORT auto set_recv_count(size_t shard, size_t peer, size_t source, size_t count) -> void;
    //! The send count set for (shard, peer, dest).
    [[nodiscard]] monoprop_EXPORT auto send_count(size_t shard, size_t peer, size_t dest) const -> size_t;
    //! The receive count for (shard, peer, source): set by its owner or by wait_counts().
    [[nodiscard]] monoprop_EXPORT auto recv_count(size_t shard, size_t peer, size_t source) const -> size_t;

    // --- Owner-parallel planning (rounds whose rows cover every other rank, such as replay steps) ------------------

    //! Owner `shard`, after writing both of its rows: record its per-peer row totals for plan_totals().
    monoprop_EXPORT auto close_rows(size_t shard) -> void;
    /*!
     * \brief Start a round over `peers` from the owners' row totals (close_rows()): per-peer MPI counts and
     *        displacements and both staging sizes, in O(P x T) on the primary.
     *
     * The owners' offsets follow from plan_column() (same phase) and place_column() (next phase); they equal what
     * plan_send() and plan_recv() lay out. Row index k must be the k-th peer of `peers` for every possible peer index.
     *
     * \throws As plan_send(): std::invalid_argument for an invalid peer set, std::length_error if a total does not fit
     *         an int, std::logic_error while requests are live. Nothing is posted.
     */
    monoprop_EXPORT auto plan_totals(std::span<const size_t> peers, ExchangeTransport transport) -> void;
    //! Owner `shard`, in plan_totals()' phase: the send-side sum of column `shard` (local sources to remote shard
    //! `shard`) for every possible peer index.
    monoprop_EXPORT auto plan_column(size_t shard) -> void;
    /*!
     * \brief Owner `shard`, one checkpoint after plan_totals() and every plan_column(): the send offsets of column
     *        `shard` (every local source's block for remote shard `shard`) and the receive offsets of row `shard`.
     * \throws std::logic_error before plan_totals(); std::out_of_range for a shard outside the table.
     */
    monoprop_EXPORT auto place_column(size_t shard) -> void;

    // --- Primary side: planning and MPI -----------------------------------------------------------------------------

    /*!
     * \brief Start a round over `peers`: lay out and size the send staging from every owner's send row.
     *
     * \param peers     Peer ranks, strictly ascending, none of them this rank.
     * \param transport How post() moves the payload; must be the same on every rank of the round.
     * \throws std::invalid_argument for an invalid peer set; std::length_error if a per-peer total or offset does not
     *         fit an int; std::logic_error while requests are live. Nothing is posted.
     */
    monoprop_EXPORT auto plan_send(std::span<const size_t> peers, ExchangeTransport transport) -> void;
    //! Post the count round over the peers. Primary only; requires plan_send().
    monoprop_EXPORT auto post_counts() -> void;
    //! Complete the count round and fill every owner's receive row from it. Primary only.
    monoprop_EXPORT auto wait_counts() -> void;
    /*!
     * \brief Lay out and size the receive staging from every owner's receive row.
     * \throws std::length_error as plan_send(); std::logic_error without plan_send() or while requests are live.
     */
    monoprop_EXPORT auto plan_recv() -> void;
    //! Post the payload. Primary only; requires plan_recv().
    monoprop_EXPORT auto post() -> void;
    //! Complete the payload. Primary only.
    monoprop_EXPORT auto wait() -> void;

    // --- Owner side: payload ----------------------------------------------------------------------------------------

    /*!
     * \brief Owner `shard`'s slice for shard `dest` of peer number `peer`, valid until the next plan_send().
     * \throws std::logic_error if `T` is not the round's element type or no round is planned; std::out_of_range.
     */
    template <class T>
    [[nodiscard]] auto send_block(size_t shard, size_t peer, size_t dest) -> std::span<T> {
        const auto bytes = send_bytes_(shard, peer, dest, exchange_element_of<T>);
        return {reinterpret_cast<T *>(bytes.data()), bytes.size() / sizeof(T)};
    }

    /*!
     * \brief What local `shard` received from shard `source` of peer number `peer`, valid until the next plan_recv().
     * \throws As send_block(); the round's receive side must have been planned with plan_recv().
     */
    template <class T>
    [[nodiscard]] auto recv_block(size_t shard, size_t peer, size_t source) const -> std::span<const T> {
        const auto bytes = recv_bytes_(shard, peer, source, exchange_element_of<T>);
        return {reinterpret_cast<const T *>(bytes.data()), bytes.size() / sizeof(T)};
    }

private:
    struct State;

    monoprop_EXPORT auto send_bytes_(size_t shard, size_t peer, size_t dest, ExchangeElement element)
        -> std::span<std::byte>;
    [[nodiscard]] monoprop_EXPORT auto recv_bytes_(size_t shard,
                                                   size_t peer,
                                                   size_t source,
                                                   ExchangeElement element) const -> std::span<const std::byte>;

    std::unique_ptr<State> state_; //!< Heap-stable: MPI holds pointers into it once a round is posted.
};

/*!
 * \brief The physical rounds of one rank-level owner, kept across its operations.
 *
 * Every construction, evaluation and replay call of an owner reuses these rounds instead of allocating its own, so
 * their staging grows to a high-water mark once, as the legacy communicator's staging does. Rounds allocated per call
 * free their staging at every call's end: each such free of a large mmapped block raised glibc's dynamic mmap threshold
 * a step further, and the later per-gate transient allocations then stayed resident in fragmented thread arenas.
 *
 * One owner uses its rounds from one operation at a time, on its calling thread; copies of an owner get their own empty
 * set, never a shared one. A round is created on first use, and recreated if the geometry it was created for changed.
 */
class PhysicalRounds {
public:
    //! The round kinds; rounds of different kinds may be in flight in the same operation.
    enum class Kind : std::uint8_t {
        queries,       //!< Construction queries (u64 words).
        graph_answers, //!< Graph construction answers (u32 term indices).
        fused_answers, //!< Propagation answers (double coefficients).
        replay,        //!< Replay endpoints of evaluation, partial contraction and informed construction (double).
    };

    /*!
     * \brief The round of `kind` for `threads` owners per process over the ranks of `world`.
     * \throws As PhysicalExchange's constructor, when the round has to be created.
     */
    monoprop_EXPORT auto get(Kind kind, const PhysicalWorld &world, size_t threads) -> PhysicalExchange &;

    //! The round of `kind` as it is, inert (threads() == 0) if never used; for inspection, never creates one.
    [[nodiscard]] auto peek(Kind kind) const -> const PhysicalExchange & {
        return rounds_.at(static_cast<size_t>(kind));
    }

private:
    std::array<PhysicalExchange, 4> rounds_; //!< Indexed by Kind; inert until first use.
};

} // namespace monoprop::detail::sharded
