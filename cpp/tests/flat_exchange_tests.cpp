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

// post_flat_alltoallv's transport choice (the graph REPLAY path). `wire_bits` alone picks it -- never the
// layout -- because a data-dependent branch straddles: a rank inside MPI_Ialltoallv waits forever on one
// that chose point-to-point. Ticket::in_flight() is what tells the two apart: 1 for the collective, two
// requests per posted leg, 0 for a round with nothing to move.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Exchange.h"

#ifdef monoprop_ENABLE_MPI
#include <bit>

#include <mpi.h>
#endif

using monoprop::mpi::active_leg_count;
using monoprop::mpi::Comm;
using monoprop::mpi::post_flat_alltoallv;

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
// What Evolution's gate resolves to when routing gives fanout 1, which is the default at a power-of-two
// rank count. Tests must not derive it from their own layout -- that is the straddle.
auto wire_bits_for(int n) -> int {
    return std::countr_zero(static_cast<unsigned>(n));
}
#endif

} // namespace

// The upper bound the request vector is sized from: a leg counts if EITHER side carries a payload, so a
// bound too small can never be handed to sparse_pairwise.
BOOST_AUTO_TEST_CASE(flat_exchange_active_legs_take_either_side) {
    constexpr int n = 8;
    std::vector<int> send(static_cast<size_t>(n), 0);
    std::vector<int> recv(static_cast<size_t>(n), 0);
    BOOST_CHECK_EQUAL(active_leg_count(send.data(), recv.data(), n), 0);
    send[1] = 4;
    recv[6] = 4;
    BOOST_CHECK_EQUAL(active_leg_count(send.data(), recv.data(), n), 2);
    recv[1] = 4; // same leg, both sides
    BOOST_CHECK_EQUAL(active_leg_count(send.data(), recv.data(), n), 2);
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
                                           c,
                                           /*wire_bits=*/1);
    BOOST_CHECK_EQUAL(ticket.in_flight(), 0);
    ticket.wait();
    BOOST_CHECK(out == send);
}

#ifdef monoprop_ENABLE_MPI

namespace {

// One round over a symmetric layout, returning what wait() had to drain. `counts` is both sides.
auto run_round(int n,
               int wire_bits,
               const std::vector<int> &counts,
               const std::vector<int> &send,
               std::vector<int> &out) -> int {
    Comm c{MPI_COMM_WORLD};
    const auto displs = displs_of(counts);
    auto ticket = post_flat_alltoallv<int>({.send = send.data(),
                                            .send_counts = counts.data(),
                                            .send_displs = displs.data(),
                                            .recv = out.data(),
                                            .recv_counts = counts.data(),
                                            .recv_displs = displs.data()},
                                           n,
                                           c,
                                           wire_bits);
    const int drained = ticket.in_flight();
    ticket.wait();
    return drained;
}

} // namespace

// wire_bits == 0 is today's collective whatever the layout holds -- including the sparse layout the
// pairwise arm exists for. The layout must not be able to move the branch.
BOOST_AUTO_TEST_CASE(flat_exchange_zero_wire_bits_always_takes_the_collective) {
    const int n = world_size();
    if (n < 2 || (n % 2) != 0) {
        return;
    }
    const int me = world_rank();
    const int peer = me ^ 1;
    for (const bool dense : {true, false}) {
        std::vector<int> counts(static_cast<size_t>(n), 0);
        if (dense) {
            std::fill(counts.begin(), counts.end(), 1);
        }
        else {
            counts[static_cast<size_t>(peer)] = 1; // one leg: the shape the pairwise arm is for
        }
        const auto displs = displs_of(counts);
        std::vector<int> send(static_cast<size_t>(total_of(counts)), me);
        std::vector<int> out(static_cast<size_t>(total_of(counts)), -1);
        BOOST_CHECK_EQUAL(run_round(n, 0, counts, send, out), 1);
        for (int i = 0; i < n; ++i) {
            if (counts[static_cast<size_t>(i)] != 0) {
                BOOST_CHECK_EQUAL(out[static_cast<size_t>(displs[static_cast<size_t>(i)])], i);
            }
        }
    }
}

// One peer, me ^ 1 -- an involution, so the count matrix stays symmetric and both ends drop the same
// n - 1 legs. Two requests (one Irecv, one Isend) and the same bytes delivered.
BOOST_AUTO_TEST_CASE(flat_exchange_single_leg_takes_the_pairwise_path) {
    const int n = world_size();
    if (n < 2 || (n % 2) != 0) {
        return;
    }
    const int me = world_rank();
    const int peer = me ^ 1;
    constexpr int len = 3;
    std::vector<int> counts(static_cast<size_t>(n), 0);
    counts[static_cast<size_t>(peer)] = len;
    std::vector<int> send(static_cast<size_t>(len));
    for (int j = 0; j < len; ++j) {
        send[static_cast<size_t>(j)] = (me * 1000) + j;
    }
    std::vector<int> out(static_cast<size_t>(len), -1);
    BOOST_CHECK_EQUAL(run_round(n, wire_bits_for(n), counts, send, out), 2);
    for (int j = 0; j < len; ++j) {
        BOOST_CHECK_EQUAL(out[static_cast<size_t>(j)], (peer * 1000) + j);
    }
}

// Every leg active on the pairwise arm: it posts a pair per non-zero leg rather than falling back, so
// the transport really is the gate's choice and not the layout's. The self leg is a copy, not a pair.
BOOST_AUTO_TEST_CASE(flat_exchange_dense_layout_on_the_pairwise_arm) {
    const int n = world_size();
    if (n < 2 || (n & (n - 1)) != 0) {
        return; // off a power of two the gate resolves to 0 bits, i.e. the collective
    }
    const int me = world_rank();
    const std::vector<int> counts(static_cast<size_t>(n), 1);
    const auto displs = displs_of(counts);
    std::vector<int> send(static_cast<size_t>(n));
    for (int d = 0; d < n; ++d) {
        send[static_cast<size_t>(d)] = (me * 1000) + d;
    }
    std::vector<int> out(static_cast<size_t>(n), -1);
    BOOST_CHECK_EQUAL(run_round(n, wire_bits_for(n), counts, send, out), 2 * (n - 1));
    for (int src = 0; src < n; ++src) {
        BOOST_CHECK_EQUAL(out[static_cast<size_t>(src)], (src * 1000) + me);
    }
}

// No active leg anywhere: nothing is posted, wait() drains nothing, and the recv buffer is left as the
// caller sized it. A layer with no cross-rank partners takes this shape on EVERY rank at once -- the
// symmetric layout is what makes that true -- so the pairwise arm is still the agreed one.
BOOST_AUTO_TEST_CASE(flat_exchange_empty_layout_posts_nothing) {
    const int n = world_size();
    if (n < 2 || (n & (n - 1)) != 0) {
        return;
    }
    const std::vector<int> counts(static_cast<size_t>(n), 0);
    BOOST_REQUIRE_EQUAL(total_of(counts), 0);
    const std::vector<int> send(1, 0);
    std::vector<int> out(1, -1); // Evolution sizes an empty round to 1, not 0
    BOOST_CHECK_EQUAL(run_round(n, wire_bits_for(n), counts, send, out), 0);
    BOOST_CHECK_EQUAL(out[0], -1);
}

#endif // monoprop_ENABLE_MPI
