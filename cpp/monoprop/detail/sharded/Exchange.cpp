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
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
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

/*
 * Staging memory straight from the kernel. A round's staging is large and grows with the operator; returned to
 * malloc, every outgrown block would raise glibc's dynamic mmap threshold, after which the owners' per-gate transients
 * stay resident in fragmented thread arenas instead of being unmapped on free. Mapping it directly leaves malloc's
 * heuristics alone. Whole pages, zero-filled by the kernel.
 */
template <class T>
struct MappedAllocator {
    using value_type = T;

    MappedAllocator() noexcept = default;
    template <class U>
    MappedAllocator(const MappedAllocator<U> & /*other*/) noexcept {}

    [[nodiscard]] static auto bytes_for(size_t n) -> size_t {
        static const auto page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        if (n > (std::numeric_limits<size_t>::max() - page) / sizeof(T)) {
            throw std::bad_alloc();
        }
        return ((n * sizeof(T)) + page - 1) / page * page;
    }

    [[nodiscard]] auto allocate(size_t n) -> T * {
        if (n == 0) {
            return nullptr;
        }
        void *p = ::mmap(nullptr, bytes_for(n), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(p);
    }

    auto deallocate(T *p, size_t n) noexcept -> void {
        if (p != nullptr) {
            ::munmap(p, bytes_for(n));
        }
    }

    template <class U>
    auto operator==(const MappedAllocator<U> & /*other*/) const noexcept -> bool {
        return true;
    }
};

// AddressSanitizer neither poisons nor tracks memory mapped outside its allocator (no redzones, and its pointer-pair
// checks reject end - begin of such a block), so a sanitized build stages through malloc, where ASan checks every
// access; the mapped staging is a release-build memory-behaviour choice, not a correctness one.
#if defined(__SANITIZE_ADDRESS__)
constexpr bool sanitized_addresses = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
constexpr bool sanitized_addresses = true;
#else
constexpr bool sanitized_addresses = false;
#endif
#else
constexpr bool sanitized_addresses = false;
#endif
using Staging =
    std::vector<std::byte,
                std::conditional_t<sanitized_addresses, std::allocator<std::byte>, MappedAllocator<std::byte>>>;

// A wide running total narrowed to an MPI int, or a length_error naming what overflowed.
auto checked_int(long long value, const char *what) -> int {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::length_error(std::format("sharded exchange: {} {} does not fit an MPI int", what, value));
    }
    return static_cast<int>(value);
}

#ifdef monoprop_ENABLE_MPI
// Write `counts` (non-negative) to `wire` in the narrowest of 1, 2 or 4 unsigned bytes per count that holds the
// largest, little-endian as the host stores it; return that width.
auto encode_counts(std::span<const int> counts, std::byte *wire) -> size_t {
    int largest = 0;
    for (const int count : counts) {
        largest = std::max(largest, count);
    }
    const auto store = [&]<class U>(U /*tag*/) {
        for (size_t i = 0; i < counts.size(); ++i) {
            const auto value = static_cast<U>(counts[i]);
            std::memcpy(wire + (i * sizeof(U)), &value, sizeof(U));
        }
        return sizeof(U);
    };
    if (largest <= std::numeric_limits<std::uint8_t>::max()) {
        return store(std::uint8_t{});
    }
    if (largest <= std::numeric_limits<std::uint16_t>::max()) {
        return store(std::uint16_t{});
    }
    return store(std::uint32_t{});
}

// Count `i` of a block written by encode_counts() with `width` bytes per count.
auto decode_count(const std::byte *wire, size_t i, size_t width) -> int {
    if (width == 1) {
        return static_cast<int>(std::to_integer<std::uint8_t>(wire[i]));
    }
    if (width == 2) {
        std::uint16_t value = 0;
        std::memcpy(&value, wire + (i * 2), 2);
        return static_cast<int>(value);
    }
    std::uint32_t value = 0;
    std::memcpy(&value, wire + (i * 4), 4);
    return static_cast<int>(value);
}

#endif

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
    // Element offsets into the staging, same shape; derived by the primary's planning, or owner-parallel.
    // Eight bytes each, not four: owner-parallel planning has neighbouring owners write neighbouring entries of a
    // column (place_column()), and at four bytes twice as many owners share each line, so a replay step's packing
    // stalls on lines the other owners just wrote (+40 % per call on small evaluations at 2 x 48).
    std::vector<long long> send_off;
    std::vector<long long> recv_off;
    // Owner-parallel planning: per-owner row totals [shard * max_peers + peer] (one writer each, close_rows()), and
    // per-column send sums [peer * threads + column] (one writer per column, plan_column()).
    std::vector<long long> send_totals;
    std::vector<long long> recv_totals;
    std::vector<long long> column_sums;
    std::vector<long long> column_start; // lay_out() scratch: one entry per remote shard

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
    Staging send_stage;
    Staging recv_stage;
    // Count round: [peer][dest shard][source shard] counts. count_send holds them as ints; on the wire each peer's
    // block travels in the narrowest unsigned width (1, 2 or 4 bytes) that holds its largest count, so the small
    // blocks of a light round stay below the transport's eager limit instead of paying a rendezvous; the receiver
    // reads the width from the received size.
    std::vector<int> count_send;
    std::vector<std::byte> count_wire_send;
    std::vector<std::byte> count_wire_recv;
#ifdef monoprop_ENABLE_MPI
    std::vector<MPI_Request> requests;
    std::vector<MPI_Request> count_requests;
    std::vector<MPI_Status> count_statuses;
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
    // the send side the row owner is the source (minor), on the receive side the destination (major). Both passes walk
    // rows contiguously through local pointers (the planning runs serially on the primary, once per round side); only
    // each peer's end is checked, which bounds every partial sum below it.
    auto lay_out(const std::vector<int> &rows,
                 std::vector<long long> &off,
                 std::vector<int> &mpi_counts,
                 std::vector<int> &mpi_displs,
                 bool send_side,
                 const char *what) -> size_t {
        std::ranges::fill(mpi_counts, 0);
        std::ranges::fill(mpi_displs, 0);
        const size_t n = threads;
        const size_t row_stride = stride;
        const int *const in = rows.data();
        long long *const out = off.data();
        long long *const start = column_start.data();
        // The rows were just written by T different owners and the offsets were last read by them: request every line
        // of the round's segments at once, so those transfers overlap instead of arriving one row at a time.
        constexpr size_t line_counts = 64 / sizeof(int);
        constexpr size_t line_offsets = 64 / sizeof(long long);
        const size_t used = peers.size() * n;
        for (size_t row = 0; row < n; ++row) {
            for (size_t i = 0; i < used; i += line_counts) {
                __builtin_prefetch(in + (row * row_stride) + i, 0, 3);
            }
            for (size_t i = 0; i < used; i += line_offsets) {
                __builtin_prefetch(out + (row * row_stride) + i, 1, 3);
            }
        }
        long long running = 0;
        for (size_t k = 0; k < peers.size(); ++k) {
            const long long base = running;
            const size_t segment = k * n;
            if (send_side) {
                // Column t (remote destination t) starts after the earlier columns; local sources ascend within it.
                std::fill_n(start, n, 0LL);
                for (size_t u = 0; u < n; ++u) {
                    const int *row = in + (u * row_stride) + segment;
                    for (size_t t = 0; t < n; ++t) {
                        start[t] += row[t];
                    }
                }
                for (size_t t = 0; t < n; ++t) {
                    const long long width = start[t];
                    start[t] = running;
                    running += width;
                }
                for (size_t u = 0; u < n; ++u) {
                    const int *row = in + (u * row_stride) + segment;
                    long long *dst = out + (u * row_stride) + segment;
                    for (size_t t = 0; t < n; ++t) {
                        dst[t] = start[t];
                        start[t] += row[t];
                    }
                }
            }
            else {
                // Row t (local destination t) holds remote sources in ascending order: already contiguous.
                for (size_t t = 0; t < n; ++t) {
                    const int *row = in + (t * row_stride) + segment;
                    long long *dst = out + (t * row_stride) + segment;
                    for (size_t su = 0; su < n; ++su) {
                        dst[su] = running;
                        running += row[su];
                    }
                }
            }
            const size_t r = peers[k];
            mpi_displs[r] = checked_int(base, what);
            mpi_counts[r] = checked_int(running - base, what);
        }
        return static_cast<size_t>(checked_int(running, what));
    }

    auto size_stage(Staging &stage, size_t elements) const -> void {
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
    s.send_totals.assign(threads * s.max_peers, 0);
    s.recv_totals.assign(threads * s.max_peers, 0);
    s.column_sums.assign(s.max_peers * threads, 0);
    s.column_start.assign(threads, 0);
    s.mpi_send_counts.assign(world.ranks, 0);
    s.mpi_send_displs.assign(world.ranks, 0);
    s.mpi_recv_counts.assign(world.ranks, 0);
    s.mpi_recv_displs.assign(world.ranks, 0);
    s.count_send.assign(s.max_peers * threads * threads, 0);
    s.count_wire_send.assign(s.max_peers * threads * threads * sizeof(std::uint32_t), std::byte{0});
    s.count_wire_recv.assign(s.max_peers * threads * threads * sizeof(std::uint32_t), std::byte{0});
#ifdef monoprop_ENABLE_MPI
    // Sized once: MPI holds pointers into these while requests are live.
    s.requests.assign(2 * std::max<size_t>(s.max_peers, 1), MPI_REQUEST_NULL);
    s.count_requests.assign(2 * std::max<size_t>(s.max_peers, 1), MPI_REQUEST_NULL);
    s.count_statuses.resize(s.count_requests.size());
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

auto PhysicalExchange::close_rows(size_t shard) -> void {
    require_state(state_.get());
    State &s = *state_;
    if (shard >= s.threads) {
        throw std::out_of_range(std::format("sharded exchange: shard {} of {}", shard, s.threads));
    }
    for (size_t k = 0; k < s.max_peers; ++k) {
        long long sent = 0;
        long long received = 0;
        for (size_t t = 0; t < s.threads; ++t) {
            const size_t i = (shard * s.stride) + (k * s.threads) + t;
            sent += s.send_rows[i];
            received += s.recv_rows[i];
        }
        s.send_totals[(shard * s.max_peers) + k] = sent;
        s.recv_totals[(shard * s.max_peers) + k] = received;
    }
}

auto PhysicalExchange::plan_totals(std::span<const size_t> peers, ExchangeTransport transport) -> void {
    require_state(state_.get());
    State &s = *state_;
    s.require_idle("plan_totals");
    s.send_planned = false;
    s.recv_planned = false;
    s.set_peers(peers, transport);
    // As lay_out(): peer k's message occupies [base_k, base_k + total_k), the bases a running sum over the peers.
    const auto place = [&](const std::vector<long long> &totals,
                           std::vector<int> &mpi_counts,
                           std::vector<int> &mpi_displs,
                           const char *what) -> size_t {
        std::ranges::fill(mpi_counts, 0);
        std::ranges::fill(mpi_displs, 0);
        long long running = 0;
        for (size_t k = 0; k < s.peers.size(); ++k) {
            long long total = 0;
            for (size_t shard = 0; shard < s.threads; ++shard) {
                total += totals[(shard * s.max_peers) + k];
            }
            const size_t r = s.peers[k];
            mpi_displs[r] = checked_int(running, what);
            mpi_counts[r] = checked_int(total, what);
            running += total;
        }
        return static_cast<size_t>(checked_int(running, what));
    };
    const size_t sent = place(s.send_totals, s.mpi_send_counts, s.mpi_send_displs, "send total");
    const size_t received = place(s.recv_totals, s.mpi_recv_counts, s.mpi_recv_displs, "receive total");
    s.size_stage(s.send_stage, sent);
    s.size_stage(s.recv_stage, received);
    s.send_planned = true;
    s.recv_planned = true;
}

auto PhysicalExchange::plan_column(size_t shard) -> void {
    require_state(state_.get());
    State &s = *state_;
    if (shard >= s.threads) {
        throw std::out_of_range(std::format("sharded exchange: shard {} of {}", shard, s.threads));
    }
    for (size_t k = 0; k < s.max_peers; ++k) {
        long long sum = 0;
        for (size_t u = 0; u < s.threads; ++u) {
            sum += s.send_rows[(u * s.stride) + (k * s.threads) + shard];
        }
        s.column_sums[(k * s.threads) + shard] = sum;
    }
}

auto PhysicalExchange::place_column(size_t shard) -> void {
    require_state(state_.get());
    State &s = *state_;
    if (!s.send_planned || !s.recv_planned) {
        throw std::logic_error("sharded exchange: place_column() before plan_totals()");
    }
    if (shard >= s.threads) {
        throw std::out_of_range(std::format("sharded exchange: shard {} of {}", shard, s.threads));
    }
    // Every partial sum is bounded by its peer's checked total.
    for (size_t k = 0; k < s.peers.size(); ++k) {
        const size_t r = s.peers[k];
        // Send: destination-shard major, so column `shard` starts after the earlier columns; sources ascend within it.
        long long running = s.mpi_send_displs[r];
        for (size_t t = 0; t < shard; ++t) {
            running += s.column_sums[(k * s.threads) + t];
        }
        for (size_t u = 0; u < s.threads; ++u) {
            const size_t i = (u * s.stride) + (k * s.threads) + shard;
            s.send_off[i] = running;
            running += s.send_rows[i];
        }
        // Receive: local-destination major, so row `shard` starts after the earlier rows; sources ascend within it.
        running = s.mpi_recv_displs[r];
        for (size_t t = 0; t < shard; ++t) {
            running += s.recv_totals[(t * s.max_peers) + k];
        }
        for (size_t su = 0; su < s.threads; ++su) {
            const size_t i = (shard * s.stride) + (k * s.threads) + su;
            s.recv_off[i] = running;
            running += s.recv_rows[i];
        }
    }
}

auto PhysicalExchange::post_counts() -> void {
    require_primary("post_counts");
    require_state(state_.get());
    State &s = *state_;
    if (!s.send_planned) {
        throw std::logic_error("sharded exchange: post_counts() before plan_send()");
    }
    s.require_idle("post_counts");
    const size_t shards = s.threads;
    const size_t t2 = shards * shards;
    for (size_t k = 0; k < s.peers.size(); ++k) {
        int *const block = s.count_send.data() + (k * t2);
        for (size_t u = 0; u < shards; ++u) {
            const int *row = s.send_rows.data() + (u * s.stride) + (k * shards);
            for (size_t t = 0; t < shards; ++t) {
                block[(t * shards) + u] = row[t];
            }
        }
    }
    if (s.peers.empty()) {
        return;
    }
#ifdef monoprop_ENABLE_MPI
    const size_t capacity = t2 * sizeof(std::uint32_t);
    const int receivable = checked_int(static_cast<long long>(capacity), "count block");
    int n = 0;
    for (size_t k = 0; k < s.peers.size(); ++k) {
        const int peer = static_cast<int>(s.peers[k]);
        const int *const counts = s.count_send.data() + (k * t2);
        std::byte *const wire = s.count_wire_send.data() + (k * capacity);
        const size_t width = encode_counts(std::span<const int>(counts, t2), wire);
        MPI_Irecv(s.count_wire_recv.data() + (k * capacity),
                  receivable,
                  MPI_BYTE,
                  peer,
                  kShardedCountTag,
                  s.comm.mpi,
                  &s.count_requests[static_cast<size_t>(n++)]);
        MPI_Isend(wire,
                  checked_int(static_cast<long long>(t2 * width), "count block"),
                  MPI_BYTE,
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
    const size_t n = s.threads;
    const size_t t2 = n * n;
#ifdef monoprop_ENABLE_MPI
    const bool waited = s.count_posted != 0;
    if (waited) {
        MPI_Waitall(s.count_posted, s.count_requests.data(), s.count_statuses.data());
        s.count_posted = 0;
    }
    // Peer k's block holds [my destination t][its source su]: row t of the receive table. Its receive request is
    // number 2k (post_counts() posts the receive before the send, peer by peer).
    for (size_t k = 0; waited && k < s.peers.size(); ++k) {
        int bytes = 0;
        MPI_Get_count(&s.count_statuses[2 * k], MPI_BYTE, &bytes);
        const std::byte *const wire = s.count_wire_recv.data() + (k * t2 * sizeof(std::uint32_t));
        const size_t width = static_cast<size_t>(bytes) / t2;
        if (bytes < 0 || width * t2 != static_cast<size_t>(bytes) || (width != 1 && width != 2 && width != 4)) {
            throw std::length_error(std::format("sharded exchange: rank {} sent a {}-byte count block for {} counts",
                                                s.peers[k],
                                                bytes,
                                                t2));
        }
        int smallest = 0;
        for (size_t t = 0; t < n; ++t) {
            int *row = s.recv_rows.data() + (t * s.stride) + (k * n);
            for (size_t su = 0; su < n; ++su) {
                const int count = decode_count(wire, (t * n) + su, width);
                smallest = std::min(smallest, count);
                row[su] = count;
            }
        }
        if (smallest < 0) {
            throw std::length_error(
                std::format("sharded exchange: rank {} announced a negative count {}", s.peers[k], smallest));
        }
    }
#else
    static_cast<void>(t2);
#endif
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

auto PhysicalRounds::get(Kind kind, const PhysicalWorld &world, size_t threads) -> PhysicalExchange & {
    PhysicalExchange &round = rounds_.at(static_cast<size_t>(kind));
    if (round.threads() != threads || round.ranks() != world.ranks || round.rank() != world.rank) {
        switch (kind) {
            case Kind::queries:
                round = PhysicalExchange(world, threads, ExchangeElement::u64, kShardedQueryTag);
                break;
            case Kind::graph_answers:
                round = PhysicalExchange(world, threads, ExchangeElement::u32, kShardedAnswerTag);
                break;
            case Kind::fused_answers:
                round = PhysicalExchange(world, threads, ExchangeElement::f64, kShardedAnswerTag);
                break;
            case Kind::replay:
                round = PhysicalExchange(world, threads, ExchangeElement::f64, kShardedReplayTag);
                break;
        }
    }
    return round;
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
    return {s.send_stage.data() + (static_cast<size_t>(s.send_off[i]) * s.elem),
            static_cast<size_t>(s.send_rows[i]) * s.elem};
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
    return {s.recv_stage.data() + (static_cast<size_t>(s.recv_off[i]) * s.elem),
            static_cast<size_t>(s.recv_rows[i]) * s.elem};
}

} // namespace monoprop::detail::sharded
