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

#include "monoprop/detail/sharded/Exchange.h"

#include <omp.h>

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop::detail::sharded {

namespace {

auto element_bytes(ExchangeElement element) -> size_t {
    switch (element) {
        case ExchangeElement::u32:
            return 4;
        case ExchangeElement::u64:
        case ExchangeElement::f64:
            return 8;
    }
    return 8;
}

#ifdef monoprop_ENABLE_MPI
auto element_datatype(ExchangeElement element) -> MPI_Datatype {
    switch (element) {
        case ExchangeElement::u32:
            return MPI_UINT32_T;
        case ExchangeElement::u64:
            return MPI_UINT64_T;
        case ExchangeElement::f64:
            return MPI_DOUBLE;
    }
    return MPI_DOUBLE;
}
#endif

// A wide running total narrowed to an MPI int, or a length_error naming what overflowed.
auto checked_int(long long value, const char *what) -> int {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::length_error(std::format("sharded exchange: {} {} does not fit an MPI int", what, value));
    }
    return static_cast<int>(value);
}

// MPI ownership: the primary of the team, which is the thread that entered it and initialized MPI.
auto require_primary(const char *what) -> void {
    if (omp_get_thread_num() != 0) {
        throw std::logic_error(std::format("sharded exchange: {} called on OpenMP worker {}, not the primary",
                                           what,
                                           omp_get_thread_num()));
    }
#ifdef monoprop_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        int is_main = 0;
        MPI_Is_thread_main(&is_main);
        if (is_main == 0) {
            throw std::logic_error(
                std::format("sharded exchange: {} called on a thread other than the one that initialized MPI", what));
        }
    }
#endif
}

} // namespace

struct PhysicalExchange::State {
    mpi::Comm comm;
    size_t rank = 0;
    size_t ranks = 1;
    size_t threads = 1;
    ExchangeElement element = ExchangeElement::f64;
    size_t elem = 8;
    int tag = 0;
    size_t max_peers = 0; // ranks - 1
    size_t stride = 0;    // max_peers * threads: one row's length

    // Rows: entry [shard * stride + peer * threads + other] is the count between local `shard` and shard `other` of
    // peer `peer`. Row `shard` has one writer, its owner, except where the primary fills receive rows.
    std::vector<int> send_rows;
    std::vector<int> recv_rows;
    // Element offsets into the staging, same shape; derived by the primary's planning.
    std::vector<size_t> send_off;
    std::vector<size_t> recv_off;

    std::vector<size_t> peers;
    ExchangeTransport transport = ExchangeTransport::pairwise;
    bool send_planned = false;
    bool recv_planned = false;
    // Per communicator rank, zero outside the peers: what MPI_Ialltoallv reads, and the pairwise legs.
    std::vector<int> mpi_send_counts;
    std::vector<int> mpi_send_displs;
    std::vector<int> mpi_recv_counts;
    std::vector<int> mpi_recv_displs;
    // High-water-mark staging, never resized while a request may point into it.
    std::vector<std::byte> send_stage;
    std::vector<std::byte> recv_stage;
    // Count round: [peer][dest shard][source shard] ints in both directions.
    std::vector<int> count_send;
    std::vector<int> count_recv;
#ifdef monoprop_ENABLE_MPI
    std::vector<MPI_Request> requests;
    std::vector<MPI_Request> count_requests;
#endif
    int posted = 0;
    int count_posted = 0;

    auto drain() noexcept -> void {
#ifdef monoprop_ENABLE_MPI
        if (count_posted != 0) {
            MPI_Waitall(count_posted, count_requests.data(), MPI_STATUSES_IGNORE);
            count_posted = 0;
        }
        if (posted != 0) {
            MPI_Waitall(posted, requests.data(), MPI_STATUSES_IGNORE);
            posted = 0;
        }
#endif
    }

    [[nodiscard]] auto at(size_t shard, size_t peer, size_t other) const -> size_t {
        if (shard >= threads || peer >= max_peers || other >= threads) {
            throw std::out_of_range(std::format("sharded exchange: (shard {}, peer {}, shard {}) is outside a table of "
                                                "{} shards and {} peers",
                                                shard,
                                                peer,
                                                other,
                                                threads,
                                                max_peers));
        }
        return (shard * stride) + (peer * threads) + other;
    }

    auto require_idle(const char *what) const -> void {
        if (posted != 0 || count_posted != 0) {
            throw std::logic_error(std::format("sharded exchange: {} while a posted round is still in flight", what));
        }
    }

    auto set_peers(std::span<const size_t> next, ExchangeTransport how) -> void {
        if (next.size() > max_peers) {
            throw std::invalid_argument(
                std::format("sharded exchange: {} peers in a world of {} ranks", next.size(), ranks));
        }
        for (size_t k = 0; k < next.size(); ++k) {
            if (next[k] >= ranks || next[k] == rank || (k > 0 && next[k] <= next[k - 1])) {
                throw std::invalid_argument(
                    std::format("sharded exchange: peer {} is not another rank in strictly ascending order (rank {} "
                                "of {})",
                                next[k],
                                rank,
                                ranks));
            }
        }
        peers.assign(next.begin(), next.end());
        transport = how;
    }

    // Offsets of one side from its rows: the message of peer k is destination-shard major, source-shard minor. On
    // the send side the row owner is the source; on the receive side it is the destination.
    auto lay_out(const std::vector<int> &rows,
                 std::vector<size_t> &off,
                 std::vector<int> &mpi_counts,
                 std::vector<int> &mpi_displs,
                 bool send_side,
                 const char *what) -> size_t {
        std::ranges::fill(mpi_counts, 0);
        std::ranges::fill(mpi_displs, 0);
        long long running = 0;
        for (size_t k = 0; k < peers.size(); ++k) {
            const long long base = running;
            for (size_t major = 0; major < threads; ++major) {
                for (size_t minor = 0; minor < threads; ++minor) {
                    // Send: major = remote destination t, minor = local source u, row u. Receive: major = local
                    // destination t, minor = remote source su, row t.
                    const size_t row = send_side ? minor : major;
                    const size_t other = send_side ? major : minor;
                    const size_t i = (row * stride) + (k * threads) + other;
                    off[i] = static_cast<size_t>(checked_int(running, what));
                    running += rows[i];
                }
            }
            const size_t r = peers[k];
            mpi_displs[r] = checked_int(base, what);
            mpi_counts[r] = checked_int(running - base, what);
        }
        return static_cast<size_t>(checked_int(running, what));
    }

    auto size_stage(std::vector<std::byte> &stage, size_t elements) const -> void {
        // At least one element, so MPI never sees a null buffer.
        const size_t need = std::max<size_t>(elements, 1) * elem;
        if (stage.size() < need) {
            stage.resize(need);
        }
    }
};

auto PhysicalWorld::of(const mpi::Comm &comm) -> PhysicalWorld {
    if (comm.kind != mpi::Comm::Kind::Mpi) {
        throw std::invalid_argument("sharded exchange: the physical world needs an ordinary MPI communicator");
    }
    PhysicalWorld world;
    world.comm = comm;
    world.rank = static_cast<size_t>(mpi::rank(comm));
    world.ranks = static_cast<size_t>(mpi::size(comm));
    world.replay_pairwise = mpi::routes_pairwise(comm);
    return world;
}

PhysicalExchange::PhysicalExchange() noexcept = default;

PhysicalExchange::PhysicalExchange(const PhysicalWorld &world, size_t threads, ExchangeElement element, int tag)
    : state_(std::make_unique<State>()) {
    if (threads == 0 || world.ranks == 0 || world.rank >= world.ranks) {
        throw std::invalid_argument(
            std::format("sharded exchange: {} threads at rank {} of {}", threads, world.rank, world.ranks));
    }
    State &s = *state_;
    s.comm = world.comm;
    s.rank = world.rank;
    s.ranks = world.ranks;
    s.threads = threads;
    s.element = element;
    s.elem = element_bytes(element);
    s.tag = tag;
    s.max_peers = world.ranks - 1;
    s.stride = s.max_peers * threads;
    const size_t table = threads * s.stride;
    s.send_rows.assign(table, 0);
    s.recv_rows.assign(table, 0);
    s.send_off.assign(table, 0);
    s.recv_off.assign(table, 0);
    s.mpi_send_counts.assign(world.ranks, 0);
    s.mpi_send_displs.assign(world.ranks, 0);
    s.mpi_recv_counts.assign(world.ranks, 0);
    s.mpi_recv_displs.assign(world.ranks, 0);
    s.count_send.assign(s.max_peers * threads * threads, 0);
    s.count_recv.assign(s.max_peers * threads * threads, 0);
#ifdef monoprop_ENABLE_MPI
    // Sized once: MPI holds pointers into these while requests are live.
    s.requests.assign(2 * std::max<size_t>(s.max_peers, 1), MPI_REQUEST_NULL);
    s.count_requests.assign(2 * std::max<size_t>(s.max_peers, 1), MPI_REQUEST_NULL);
#endif
}

PhysicalExchange::PhysicalExchange(PhysicalExchange &&other) noexcept = default;

auto PhysicalExchange::operator=(PhysicalExchange &&other) noexcept -> PhysicalExchange & {
    if (this != &other) {
        if (state_) {
            state_->drain(); // never drop a request this exchange already owns
        }
        state_ = std::move(other.state_);
    }
    return *this;
}

PhysicalExchange::~PhysicalExchange() {
    if (state_) {
        state_->drain();
    }
}

namespace {

auto require_state(const void *state) -> void {
    if (state == nullptr) {
        throw std::logic_error("sharded exchange: an inert (default-constructed or moved-from) exchange was used");
    }
}

} // namespace

auto PhysicalExchange::threads() const noexcept -> size_t {
    return state_ ? state_->threads : 0;
}

auto PhysicalExchange::rank() const noexcept -> size_t {
    return state_ ? state_->rank : 0;
}

auto PhysicalExchange::ranks() const noexcept -> size_t {
    return state_ ? state_->ranks : 1;
}

auto PhysicalExchange::peers() const noexcept -> std::span<const size_t> {
    return state_ ? std::span<const size_t>(state_->peers) : std::span<const size_t>{};
}

auto PhysicalExchange::live() const noexcept -> int {
    return state_ ? state_->posted + state_->count_posted : 0;
}

auto PhysicalExchange::reset_rows(size_t shard) -> void {
    require_state(state_.get());
    State &s = *state_;
    if (shard >= s.threads) {
        throw std::out_of_range(std::format("sharded exchange: shard {} of {}", shard, s.threads));
    }
    std::fill_n(s.send_rows.begin() + static_cast<std::ptrdiff_t>(shard * s.stride), s.stride, 0);
    std::fill_n(s.recv_rows.begin() + static_cast<std::ptrdiff_t>(shard * s.stride), s.stride, 0);
}

auto PhysicalExchange::set_send_count(size_t shard, size_t peer, size_t dest, size_t count) -> void {
    require_state(state_.get());
    const size_t i = state_->at(shard, peer, dest);
    state_->send_rows[i] = checked_int(static_cast<long long>(std::min<size_t>(count, size_t{1} << 62)), "send count");
}

auto PhysicalExchange::set_recv_count(size_t shard, size_t peer, size_t source, size_t count) -> void {
    require_state(state_.get());
    const size_t i = state_->at(shard, peer, source);
    state_->recv_rows[i] =
        checked_int(static_cast<long long>(std::min<size_t>(count, size_t{1} << 62)), "receive count");
}

auto PhysicalExchange::send_count(size_t shard, size_t peer, size_t dest) const -> size_t {
    require_state(state_.get());
    return static_cast<size_t>(state_->send_rows[state_->at(shard, peer, dest)]);
}

auto PhysicalExchange::recv_count(size_t shard, size_t peer, size_t source) const -> size_t {
    require_state(state_.get());
    return static_cast<size_t>(state_->recv_rows[state_->at(shard, peer, source)]);
}

auto PhysicalExchange::plan_send(std::span<const size_t> peers, ExchangeTransport transport) -> void {
    require_state(state_.get());
    State &s = *state_;
    s.require_idle("plan_send");
    s.send_planned = false;
    s.recv_planned = false;
    s.set_peers(peers, transport);
    const size_t total =
        s.lay_out(s.send_rows, s.send_off, s.mpi_send_counts, s.mpi_send_displs, /*send_side=*/true, "send total");
    s.size_stage(s.send_stage, total);
    s.send_planned = true;
}

auto PhysicalExchange::post_counts() -> void {
    require_primary("post_counts");
    require_state(state_.get());
    State &s = *state_;
    if (!s.send_planned) {
        throw std::logic_error("sharded exchange: post_counts() before plan_send()");
    }
    s.require_idle("post_counts");
    const size_t t2 = s.threads * s.threads;
    for (size_t k = 0; k < s.peers.size(); ++k) {
        for (size_t t = 0; t < s.threads; ++t) {
            for (size_t u = 0; u < s.threads; ++u) {
                s.count_send[(k * t2) + (t * s.threads) + u] = s.send_rows[(u * s.stride) + (k * s.threads) + t];
            }
        }
    }
    if (s.peers.empty()) {
        return;
    }
#ifdef monoprop_ENABLE_MPI
    const int block = checked_int(static_cast<long long>(t2), "count block");
    int n = 0;
    for (size_t k = 0; k < s.peers.size(); ++k) {
        const int peer = static_cast<int>(s.peers[k]);
        MPI_Irecv(s.count_recv.data() + (k * t2),
                  block,
                  MPI_INT,
                  peer,
                  kShardedCountTag,
                  s.comm.mpi,
                  &s.count_requests[static_cast<size_t>(n++)]);
        MPI_Isend(s.count_send.data() + (k * t2),
                  block,
                  MPI_INT,
                  peer,
                  kShardedCountTag,
                  s.comm.mpi,
                  &s.count_requests[static_cast<size_t>(n++)]);
    }
    s.count_posted = n;
#else
    throw std::logic_error("sharded exchange: an MPI-off build has no peer ranks");
#endif
}

auto PhysicalExchange::wait_counts() -> void {
    require_primary("wait_counts");
    require_state(state_.get());
    State &s = *state_;
#ifdef monoprop_ENABLE_MPI
    if (s.count_posted != 0) {
        MPI_Waitall(s.count_posted, s.count_requests.data(), MPI_STATUSES_IGNORE);
        s.count_posted = 0;
    }
#endif
    // Peer k's block holds [my destination t][its source su].
    const size_t t2 = s.threads * s.threads;
    for (size_t k = 0; k < s.peers.size(); ++k) {
        for (size_t t = 0; t < s.threads; ++t) {
            for (size_t su = 0; su < s.threads; ++su) {
                const int count = s.count_recv[(k * t2) + (t * s.threads) + su];
                if (count < 0) {
                    throw std::length_error(
                        std::format("sharded exchange: rank {} announced a negative count {}", s.peers[k], count));
                }
                s.recv_rows[(t * s.stride) + (k * s.threads) + su] = count;
            }
        }
    }
}

auto PhysicalExchange::plan_recv() -> void {
    require_state(state_.get());
    State &s = *state_;
    if (!s.send_planned) {
        throw std::logic_error("sharded exchange: plan_recv() before plan_send()");
    }
    s.require_idle("plan_recv");
    s.recv_planned = false;
    const size_t total = s.lay_out(s.recv_rows,
                                   s.recv_off,
                                   s.mpi_recv_counts,
                                   s.mpi_recv_displs,
                                   /*send_side=*/false,
                                   "receive total");
    s.size_stage(s.recv_stage, total);
    s.recv_planned = true;
}

auto PhysicalExchange::post() -> void {
    require_primary("post");
    require_state(state_.get());
    State &s = *state_;
    if (!s.send_planned || !s.recv_planned) {
        throw std::logic_error("sharded exchange: post() before the round's layout is planned");
    }
    s.require_idle("post");
#ifdef monoprop_ENABLE_MPI
    const MPI_Datatype dt = element_datatype(s.element);
    if (s.transport == ExchangeTransport::collective) {
        // Every rank joins, even with nothing to send: a rank that skipped would strand the others.
        MPI_Ialltoallv(s.send_stage.data(),
                       s.mpi_send_counts.data(),
                       s.mpi_send_displs.data(),
                       dt,
                       s.recv_stage.data(),
                       s.mpi_recv_counts.data(),
                       s.mpi_recv_displs.data(),
                       dt,
                       s.comm.mpi,
                       s.requests.data());
        s.posted = 1;
        return;
    }
    int n = 0;
    for (const size_t peer : s.peers) {
        const int recv = s.mpi_recv_counts[peer];
        const int send = s.mpi_send_counts[peer];
        if (recv != 0) {
            MPI_Irecv(s.recv_stage.data() + (static_cast<size_t>(s.mpi_recv_displs[peer]) * s.elem),
                      recv,
                      dt,
                      static_cast<int>(peer),
                      s.tag,
                      s.comm.mpi,
                      &s.requests[static_cast<size_t>(n++)]);
        }
        if (send != 0) {
            MPI_Isend(s.send_stage.data() + (static_cast<size_t>(s.mpi_send_displs[peer]) * s.elem),
                      send,
                      dt,
                      static_cast<int>(peer),
                      s.tag,
                      s.comm.mpi,
                      &s.requests[static_cast<size_t>(n++)]);
        }
    }
    s.posted = n;
#else
    if (!s.peers.empty()) {
        throw std::logic_error("sharded exchange: an MPI-off build has no peer ranks");
    }
#endif
}

auto PhysicalExchange::wait() -> void {
    require_primary("wait");
    require_state(state_.get());
#ifdef monoprop_ENABLE_MPI
    State &s = *state_;
    if (s.posted != 0) {
        MPI_Waitall(s.posted, s.requests.data(), MPI_STATUSES_IGNORE);
        s.posted = 0;
    }
#endif
}

auto PhysicalExchange::send_bytes_(size_t shard, size_t peer, size_t dest, ExchangeElement element)
    -> std::span<std::byte> {
    require_state(state_.get());
    State &s = *state_;
    if (element != s.element || !s.send_planned) {
        throw std::logic_error("sharded exchange: send_block() with the wrong element type or before plan_send()");
    }
    const size_t i = s.at(shard, peer, dest);
    if (peer >= s.peers.size()) {
        throw std::out_of_range(std::format("sharded exchange: peer {} of a {}-peer round", peer, s.peers.size()));
    }
    return {s.send_stage.data() + (s.send_off[i] * s.elem), static_cast<size_t>(s.send_rows[i]) * s.elem};
}

auto PhysicalExchange::recv_bytes_(size_t shard, size_t peer, size_t source, ExchangeElement element) const
    -> std::span<const std::byte> {
    require_state(state_.get());
    const State &s = *state_;
    if (element != s.element || !s.recv_planned) {
        throw std::logic_error("sharded exchange: recv_block() with the wrong element type or before plan_recv()");
    }
    const size_t i = s.at(shard, peer, source);
    if (peer >= s.peers.size()) {
        throw std::out_of_range(std::format("sharded exchange: peer {} of a {}-peer round", peer, s.peers.size()));
    }
    return {s.recv_stage.data() + (s.recv_off[i] * s.elem), static_cast<size_t>(s.recv_rows[i]) * s.elem};
}

} // namespace monoprop::detail::sharded
