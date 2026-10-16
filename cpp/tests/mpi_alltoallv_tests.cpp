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

/*
 * The real-MPI request helpers of the low-level one-owner-per-rank engine (detail/mpi/MPICompat.h): begin_alltoallv,
 * PendingAlltoallv, alltoall_counts and the PeerPlan narrowing, on an ordinary communicator. In the MPI-off build and
 * in the per-case serial runs the world has one rank and the stub self-copies; the cases that need peers run in the
 * whole-suite MPI variants. Ported from the plain-MPI arms of the in-process transport suites removed with the
 * partition runtime; their in-process (ShmComm/HybridComm) arms have no surviving subject.
 */

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h"

using monoprop::mpi::Comm;
using monoprop::mpi::PeerPlan;

namespace {

auto world_size() -> int {
    return monoprop::mpi::size(Comm(MPI_COMM_WORLD));
}

auto world_rank() -> int {
    return monoprop::mpi::rank(Comm(MPI_COMM_WORLD));
}

// The sparse XOR pairing needs a power-of-two world of at least two ranks.
auto pairable(int ranks) -> bool {
    return ranks >= 2 && (ranks & (ranks - 1)) == 0;
}

template <typename T>
auto full_window(const std::vector<std::vector<T>> &blocks) -> monoprop::mpi::WindowVec<std::vector<T>> {
    monoprop::mpi::WindowVec<std::vector<T>> w(monoprop::mpi::SlotWindow{.base = 0, .count = blocks.size()});
    std::ranges::copy(blocks, w.begin());
    return w;
}

// Slots outside the round's window come back empty.
template <typename T>
auto flat_blocks(const monoprop::mpi::WindowVec<std::vector<T>> &w, size_t world) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> out(world);
    for (const auto wi : w.window().indices()) {
        out[w.window().slot(wi)] = w[wi];
    }
    return out;
}

template <typename T>
auto exchange_blocks(const std::vector<std::vector<T>> &send,
                     const Comm &c,
                     bool skip_self = false,
                     const std::vector<int> *known_recv_counts = nullptr,
                     PeerPlan plan = {}) -> std::vector<std::vector<T>> {
    monoprop::mpi::WindowVec<std::vector<T>> got;
    monoprop::mpi::begin_alltoallv(full_window(send), c, skip_self, known_recv_counts, plan).wait_into(got);
    return flat_blocks(got, static_cast<size_t>(monoprop::mpi::size(c)));
}

auto sparse_tag(int src, int dst, int j) -> int {
    return (((src * 128) + dst) * 1000) + j;
}

} // namespace

BOOST_AUTO_TEST_CASE(mpi_alltoallv_sparse_plan_peer_is_an_involution) {
    for (const int shift : {0, 1, 2, 3, 5, 8, 13, 255}) {
        const PeerPlan plan{.sparse = true, .shift = shift};
        BOOST_REQUIRE(!plan.dense());
        BOOST_REQUIRE_EQUAL(plan.count(256), 1);
        for (int me = 0; me < 256; ++me) {
            const int peer = plan.peer(me, 0);
            BOOST_REQUIRE_EQUAL(plan.peer(peer, 0), me);
            BOOST_REQUIRE(plan.contains(me, peer));
            BOOST_REQUIRE(plan.contains(peer, me));
            BOOST_REQUIRE(!plan.contains(me, peer ^ 1)); // the peer set is a singleton
        }
    }
    const PeerPlan dense_plan{};
    BOOST_REQUIRE(dense_plan.dense());
    BOOST_REQUIRE_EQUAL(dense_plan.count(7), 7);
    for (int k = 0; k < 7; ++k) {
        BOOST_REQUIRE_EQUAL(dense_plan.peer(3, k), k); // dense ignores `me`: every rank is a peer
        BOOST_REQUIRE(dense_plan.contains(3, k));
    }
}

// alltoall_counts is a transpose: recv[s] on rank r is what s declared it sends to r.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_counts_transpose) {
    const int ranks = world_size();
    const int me = world_rank();
    std::vector<int> send(static_cast<size_t>(ranks));
    for (int t = 0; t < ranks; ++t) {
        send[static_cast<size_t>(t)] = (me * 100) + t;
    }
    std::vector<int> got(static_cast<size_t>(ranks), -1);
    monoprop::mpi::alltoall_counts(send.data(), got.data(), ranks, Comm(MPI_COMM_WORLD));
    for (int s = 0; s < ranks; ++s) {
        BOOST_CHECK_EQUAL(got[static_cast<size_t>(s)], (s * 100) + me);
    }
}

// Each source's block arrives contiguously in ascending source order, which Resolve.h's positional pairing relies on.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_dense_source_order_and_tags) {
    const int ranks = world_size();
    const int me = world_rank();
    const Comm c(MPI_COMM_WORLD);
    std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
    for (int t = 0; t < ranks; ++t) {
        for (int j = 0; j <= me; ++j) {
            send[static_cast<size_t>(t)].push_back((me * 1000) + j);
        }
    }
    const auto got = exchange_blocks(send, c);
    BOOST_REQUIRE_EQUAL(static_cast<int>(got.size()), ranks);
    for (int s = 0; s < ranks; ++s) {
        const auto &blk = got[static_cast<size_t>(s)];
        BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), s + 1); // source s sent s + 1
        for (int j = 0; j <= s; ++j) {
            BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], (s * 1000) + j);
        }
    }
    // skip_self: the self slot is neither sent nor received; every other source arrives intact.
    const auto skipped = exchange_blocks(send, c, /*skip_self=*/true);
    for (int s = 0; s < ranks; ++s) {
        BOOST_CHECK_EQUAL(skipped[static_cast<size_t>(s)].size(), s == me ? 0U : static_cast<size_t>(s + 1));
    }
}

// A sparse plan replaces the collectives with point-to-point over the one peer the plan names, so its failure modes
// are dropped data and a hang. Unknown recv layout (a pairwise count round) and the known layout of a response round.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_sparse_plan_delivers_only_to_its_peer) {
    const int ranks = world_size();
    if (!pairable(ranks)) {
        return;
    }
    const Comm c(MPI_COMM_WORLD);
    for (int shift = 0; shift < ranks; ++shift) {
        const PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(world_rank(), 0);
        std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
        for (int j = 0; j < 4; ++j) {
            send[static_cast<size_t>(peer)].push_back((world_rank() * 1000) + j);
        }
        const auto out = exchange_blocks(send, c, false, nullptr, plan);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), ranks);
        for (int src = 0; src < ranks; ++src) {
            if (src != peer) {
                BOOST_CHECK(out[static_cast<size_t>(src)].empty());
                continue;
            }
            BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(src)].size()), 4);
            for (int j = 0; j < 4; ++j) {
                BOOST_CHECK_EQUAL(out[static_cast<size_t>(src)][static_cast<size_t>(j)], (src * 1000) + j);
            }
        }
        std::vector<int> known(static_cast<size_t>(ranks), 0);
        known[static_cast<size_t>(peer)] = 4;
        const auto out2 = exchange_blocks(send, c, false, &known, plan);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out2.size()), ranks);
        BOOST_CHECK(out2[static_cast<size_t>(peer)] == out[static_cast<size_t>(peer)]);
    }
}

// known_recv_counts is caller-supplied and can name a rank the plan does not: no receive is posted for it, so an
// unmasked count would hand the caller unwritten bytes as data.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_known_recv_counts_are_masked_through_the_plan) {
    const int ranks = world_size();
    if (!pairable(ranks)) {
        return;
    }
    constexpr int kReal = 4;
    constexpr int kBogus = 7;
    const Comm c(MPI_COMM_WORLD);
    for (int shift = 0; shift < ranks; ++shift) {
        const PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(world_rank(), 0);
        const int bad = (peer + 1) % ranks;
        BOOST_REQUIRE(bad != peer);
        std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
        for (int j = 0; j < kReal; ++j) {
            send[static_cast<size_t>(peer)].push_back((world_rank() * 1000) + j);
        }
        std::vector<int> known(static_cast<size_t>(ranks), 0);
        known[static_cast<size_t>(peer)] = kReal;
        known[static_cast<size_t>(bad)] = kBogus;
        const auto out = exchange_blocks(send, c, false, &known, plan);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), ranks);
        BOOST_CHECK(out[static_cast<size_t>(bad)].empty());
        BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(peer)].size()), kReal);
        for (int j = 0; j < kReal; ++j) {
            BOOST_CHECK_EQUAL(out[static_cast<size_t>(peer)][static_cast<size_t>(j)], (peer * 1000) + j);
        }
    }
}

// A zero-count leg is where a send/recv posting asymmetry deadlocks: both ends must skip on the same value.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_sparse_plan_with_an_empty_leg) {
    const int ranks = world_size();
    if (!pairable(ranks)) {
        return;
    }
    const int me = world_rank();
    constexpr int kLen = 4;
    const Comm c(MPI_COMM_WORLD);
    for (int shift = 1; shift < ranks; ++shift) {
        const PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(me, 0);
        BOOST_REQUIRE(peer != me);
        const int my_len = me < peer ? 0 : kLen;
        const int peer_len = peer < me ? 0 : kLen;
        std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
        for (int j = 0; j < my_len; ++j) {
            send[static_cast<size_t>(peer)].push_back(sparse_tag(me, peer, j));
        }
        const auto out = exchange_blocks(send, c, false, nullptr, plan);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(peer)].size()), peer_len);
        for (int j = 0; j < peer_len; ++j) {
            BOOST_CHECK_EQUAL(out[static_cast<size_t>(peer)][static_cast<size_t>(j)], sparse_tag(peer, me, j));
        }
    }
}

// Shift zero makes every rank its own and only peer, and skip_self then drops that one leg.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_sparse_plan_skip_self_at_shift_zero) {
    const int ranks = world_size();
    if (!pairable(ranks)) {
        return;
    }
    const int me = world_rank();
    const PeerPlan plan{.sparse = true, .shift = 0};
    BOOST_REQUIRE_EQUAL(plan.peer(me, 0), me);
    std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
    for (int j = 0; j < 5; ++j) {
        send[static_cast<size_t>(me)].push_back(sparse_tag(me, me, j));
    }
    const auto out = exchange_blocks(send, Comm(MPI_COMM_WORLD), /*skip_self=*/true, nullptr, plan);
    BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), ranks);
    for (const auto &blk : out) {
        BOOST_CHECK(blk.empty());
    }
}

// Engine.h's run_exchange posts both its rounds on one communicator under one tag; MPI's non-overtaking order is what
// keeps a round-2 receive from matching a round-1 send.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_sparse_plan_back_to_back_rounds) {
    const int ranks = world_size();
    if (!pairable(ranks)) {
        return;
    }
    const int me = world_rank();
    const Comm c(MPI_COMM_WORLD);
    for (int shift = 0; shift < ranks; ++shift) {
        const PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(me, 0);
        const int len = 3 + (me % 2);
        std::vector<std::vector<int>> q(static_cast<size_t>(ranks));
        for (int j = 0; j < len; ++j) {
            q[static_cast<size_t>(peer)].push_back(sparse_tag(me, peer, j));
        }
        const auto q_out = exchange_blocks(q, c, false, nullptr, plan);
        std::vector<int> known(static_cast<size_t>(ranks), 0);
        known[static_cast<size_t>(peer)] = len;
        std::vector<std::vector<int>> r(static_cast<size_t>(ranks));
        const int back = static_cast<int>(q_out[static_cast<size_t>(peer)].size());
        for (int j = 0; j < back; ++j) {
            r[static_cast<size_t>(peer)].push_back(q_out[static_cast<size_t>(peer)][static_cast<size_t>(j)] + 7);
        }
        const auto r_out = exchange_blocks(r, c, false, &known, plan);
        BOOST_REQUIRE_EQUAL(back, 3 + (peer % 2));
        BOOST_REQUIRE_EQUAL(static_cast<int>(r_out[static_cast<size_t>(peer)].size()), len);
        for (int j = 0; j < len; ++j) {
            BOOST_CHECK_EQUAL(r_out[static_cast<size_t>(peer)][static_cast<size_t>(j)], sparse_tag(me, peer, j) + 7);
        }
        for (int src = 0; src < ranks; ++src) {
            if (src != peer) {
                BOOST_CHECK(r_out[static_cast<size_t>(src)].empty());
            }
        }
    }
}

// A shift outside the rank-index width would index out of range and name an invalid MPI rank: refused first, as an
// exception, before anything is posted.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_unroutable_peer_plan_is_refused) {
    const int ranks = world_size();
    const PeerPlan bad{.sparse = true, .shift = ranks};
    std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
    const Comm c(MPI_COMM_WORLD);
    BOOST_CHECK_THROW(static_cast<void>(monoprop::mpi::begin_alltoallv(full_window(send), c, false, nullptr, bad)),
                      std::invalid_argument);
    std::vector<int> counts(static_cast<size_t>(ranks), 0);
    std::vector<int> got(static_cast<size_t>(ranks), 0);
    BOOST_CHECK_THROW(monoprop::mpi::alltoall_counts(counts.data(), got.data(), ranks, c, bad), std::invalid_argument);
}

// MPI reads the payload buffers until the requests retire, so the owning handle is move-only and drains when dropped.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_pending_owns_its_requests) {
    using Pending = monoprop::mpi::PendingAlltoallv<int>;
    static_assert(!std::is_copy_constructible_v<Pending>);
    static_assert(!std::is_copy_assignable_v<Pending>);
    static_assert(std::is_nothrow_move_constructible_v<Pending>);
    static_assert(std::is_nothrow_move_assignable_v<Pending>);

    const int ranks = world_size();
    const int me = world_rank();
    const Comm c(MPI_COMM_WORLD);
    std::vector<std::vector<int>> send(static_cast<size_t>(ranks));
    for (int d = 0; d < ranks; ++d) {
        send[static_cast<size_t>(d)] = {(me * 1000) + d};
    }
    {
        auto h = monoprop::mpi::begin_alltoallv(full_window(send), c);
        auto moved = std::move(h); // the requests travel with the buffers MPI is reading
        monoprop::mpi::WindowVec<std::vector<int>> got;
        moved.wait_into(got);
        const auto out = flat_blocks(got, static_cast<size_t>(ranks));
        BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), ranks);
        for (int src = 0; src < ranks; ++src) {
            BOOST_REQUIRE_EQUAL(out[static_cast<size_t>(src)].size(), 1U);
            BOOST_CHECK_EQUAL(out[static_cast<size_t>(src)][0], (src * 1000) + me);
        }
    }
    // Never waited on: the destructor drains it, or the next collective would mismatch.
    {
        static_cast<void>(monoprop::mpi::begin_alltoallv(full_window(send), c));
    }
    // Move assignment drains the destination's own round before adopting the source's.
    {
        auto first = monoprop::mpi::begin_alltoallv(full_window(send), c);
        auto second = monoprop::mpi::begin_alltoallv(full_window(send), c);
        first = std::move(second);
        monoprop::mpi::WindowVec<std::vector<int>> got;
        first.wait_into(got);
        BOOST_CHECK_EQUAL(flat_blocks(got, static_cast<size_t>(ranks)).size(), static_cast<size_t>(ranks));
    }
    BOOST_CHECK_EQUAL(monoprop::mpi::allreduce_sum<int>(1, c), ranks);
}

// The physical reductions are ordinary MPI sums; every rank ends with the same value.
BOOST_AUTO_TEST_CASE(mpi_alltoallv_allreduce_sums_agree) {
    const int ranks = world_size();
    const int me = world_rank();
    const Comm c(MPI_COMM_WORLD);
    BOOST_CHECK_EQUAL(monoprop::mpi::allreduce_sum<size_t>(static_cast<size_t>(me) + 1, c),
                      static_cast<size_t>(ranks) * (static_cast<size_t>(ranks) + 1) / 2);
    monoprop::VecD values{static_cast<double>(me) + 0.5, 1.0, 0.0};
    monoprop::mpi::allreduce_sum_inplace(values, c);
    BOOST_CHECK_CLOSE(values[0], static_cast<double>(ranks) * static_cast<double>(ranks) / 2.0, 1e-12);
    BOOST_CHECK_EQUAL(values[1], static_cast<double>(ranks));
    BOOST_CHECK_EQUAL(values[2], 0.0);
}
