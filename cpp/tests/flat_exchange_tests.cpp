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

// post_flat_alltoallv's transport choice (the graph REPLAY path): a dense layout keeps MPI_Ialltoallv, a
// sparse one goes point-to-point over the active legs. Ticket::in_flight() is what tells the two apart --
// 1 for the collective, two requests per posted leg, 0 for a round with nothing to move.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Exchange.h"

#ifdef monoprop_ENABLE_MPI
#include <mpi.h>
#endif

using monoprop::mpi::Comm;
using monoprop::mpi::flat_exchange_prefers_pairwise;
using monoprop::mpi::post_flat_alltoallv;
using monoprop::mpi::sparse_leg_budget;

namespace {

// The prefix sum post_flat_alltoallv takes on both sides of a symmetric layout.
auto displs_of(const std::vector<int> &counts) -> std::vector<int> {
    std::vector<int> displs(counts.size());
    int running = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        displs[i] = running;
        running += counts[i];
    }
    return displs;
}

auto total_of(const std::vector<int> &counts) -> int {
    return std::accumulate(counts.begin(), counts.end(), 0);
}

#ifdef monoprop_ENABLE_MPI
auto world_size() -> int {
    int n = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}
auto world_rank() -> int {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}
#endif

} // namespace

// The boundary is <=, and the budget floors at 1 so an empty row and a one-peer row never split at N < 4
// -- a split is a deadlock, not a slow round.
BOOST_AUTO_TEST_CASE(flat_exchange_pairwise_budget_boundary) {
    for (const int n : {1, 2, 3, 4, 8, 16, 64}) {
        const int budget = std::max(1, n / 4);
        BOOST_REQUIRE_EQUAL(sparse_leg_budget(n), budget);
        std::vector<int> counts(static_cast<size_t>(n), 0);
        for (int legs = 0; legs <= n; ++legs) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill_n(counts.begin(), legs, 1);
            BOOST_CHECK_EQUAL(flat_exchange_prefers_pairwise(counts.data(), counts.data(), n), legs <= budget);
        }
    }
}

// A leg is active if EITHER side carries a payload: an asymmetric layout must not have one end post a
// receive the other never sends.
BOOST_AUTO_TEST_CASE(flat_exchange_active_legs_take_either_side) {
    constexpr int n = 8; // budget 2
    std::vector<int> send(static_cast<size_t>(n), 0);
    std::vector<int> recv(static_cast<size_t>(n), 0);
    send[1] = 4;
    recv[6] = 4;
    BOOST_CHECK(flat_exchange_prefers_pairwise(send.data(), recv.data(), n));
    recv[3] = 4;
    BOOST_CHECK(!flat_exchange_prefers_pairwise(send.data(), recv.data(), n));
}

// The self leg is a copy, not a message, so a one-rank world posts nothing at all and wait() is a no-op.
// Same on the non-MPI build, where the fallback self-copy runs instead.
BOOST_AUTO_TEST_CASE(flat_exchange_self_leg_is_copied_with_nothing_posted) {
    Comm c{MPI_COMM_SELF};
    const std::vector<int> counts{3};
    const auto displs = displs_of(counts);
    const std::vector<int> send{7, 8, 9};
    std::vector<int> out(static_cast<size_t>(total_of(counts)), -1);
    auto ticket = post_flat_alltoallv<int>({.send = send.data(),
                                            .send_counts = counts.data(),
                                            .send_displs = displs.data(),
                                            .recv = out.data(),
                                            .recv_counts = counts.data(),
                                            .recv_displs = displs.data()},
                                           1,
                                           c);
    BOOST_CHECK_EQUAL(ticket.in_flight(), 0);
    ticket.wait();
    BOOST_CHECK(out == send);
}

#ifdef monoprop_ENABLE_MPI

// Every leg active: N legs is above the budget at every N >= 2, so the round stays on MPI_Ialltoallv.
BOOST_AUTO_TEST_CASE(flat_exchange_dense_layout_takes_the_collective) {
    const int n = world_size();
    if (n < 2) {
        return;
    }
    const int me = world_rank();
    Comm c{MPI_COMM_WORLD};
    const std::vector<int> counts(static_cast<size_t>(n), 1);
    const auto displs = displs_of(counts);
    std::vector<int> send(static_cast<size_t>(n));
    for (int d = 0; d < n; ++d) {
        send[static_cast<size_t>(d)] = (me * 1000) + d;
    }
    std::vector<int> out(static_cast<size_t>(total_of(counts)), -1);
    auto ticket = post_flat_alltoallv<int>({.send = send.data(),
                                            .send_counts = counts.data(),
                                            .send_displs = displs.data(),
                                            .recv = out.data(),
                                            .recv_counts = counts.data(),
                                            .recv_displs = displs.data()},
                                           n,
                                           c);
    BOOST_CHECK_EQUAL(ticket.in_flight(), 1); // the collective, not 2N pairwise requests
    ticket.wait();
    for (int src = 0; src < n; ++src) {
        BOOST_CHECK_EQUAL(out[static_cast<size_t>(src)], (src * 1000) + me);
    }
}

// One peer, me ^ 1 -- an involution, so the count matrix stays symmetric and both ends drop the same
// N - 1 legs. Two requests posted (one Irecv, one Isend) and the same bytes delivered.
BOOST_AUTO_TEST_CASE(flat_exchange_single_leg_takes_the_pairwise_path) {
    const int n = world_size();
    if (n < 2 || (n % 2) != 0) {
        return;
    }
    const int me = world_rank();
    const int peer = me ^ 1;
    constexpr int len = 3;
    Comm c{MPI_COMM_WORLD};
    std::vector<int> counts(static_cast<size_t>(n), 0);
    counts[static_cast<size_t>(peer)] = len;
    const auto displs = displs_of(counts);
    BOOST_REQUIRE_EQUAL(total_of(counts), len);
    std::vector<int> send(static_cast<size_t>(len));
    for (int j = 0; j < len; ++j) {
        send[static_cast<size_t>(j)] = (me * 1000) + j;
    }
    std::vector<int> out(static_cast<size_t>(len), -1);
    auto ticket = post_flat_alltoallv<int>({.send = send.data(),
                                            .send_counts = counts.data(),
                                            .send_displs = displs.data(),
                                            .recv = out.data(),
                                            .recv_counts = counts.data(),
                                            .recv_displs = displs.data()},
                                           n,
                                           c);
    BOOST_CHECK_EQUAL(ticket.in_flight(), 2);
    ticket.wait();
    for (int j = 0; j < len; ++j) {
        BOOST_CHECK_EQUAL(out[static_cast<size_t>(j)], (peer * 1000) + j);
    }
}

// No active leg anywhere: nothing is posted, wait() drains nothing, and the recv buffer is left as the
// caller sized it. The shape a layer with no cross-rank partners takes.
BOOST_AUTO_TEST_CASE(flat_exchange_empty_layout_posts_nothing) {
    const int n = world_size();
    if (n < 2) {
        return;
    }
    Comm c{MPI_COMM_WORLD};
    const std::vector<int> counts(static_cast<size_t>(n), 0);
    const auto displs = displs_of(counts);
    BOOST_REQUIRE_EQUAL(total_of(counts), 0);
    const std::vector<int> send(1, 0);
    std::vector<int> out(1, -1); // Evolution sizes an empty round to 1, not 0
    auto ticket = post_flat_alltoallv<int>({.send = send.data(),
                                            .send_counts = counts.data(),
                                            .send_displs = displs.data(),
                                            .recv = out.data(),
                                            .recv_counts = counts.data(),
                                            .recv_displs = displs.data()},
                                           n,
                                           c);
    BOOST_CHECK_EQUAL(ticket.in_flight(), 0);
    ticket.wait();
    BOOST_CHECK_EQUAL(out[0], -1);
}

#endif // monoprop_ENABLE_MPI
