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

// HybridComm transport equivalence: R MPI ranks x S in-process shards must behave as one flat P=R*S
// SPMD world. Runs only under mpiexec with >= 2 ranks (a single rank exercises no cross-rank leg).
// Each rank spawns S threads sharing one HybridComm; only shard 0 touches MPI, exactly as ShardGroup
// drives it. Assertions run on the main thread (per-thread exceptions are captured and rethrown-checked).

#include <boost/test/unit_test.hpp>

#ifdef monoprop_ENABLE_MPI

#include <exception>
#include <numeric>
#include <thread>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/HybridComm.h"
#include "monoprop/detail/mpi/MPICompat.h"

using monoprop::mpi::Comm;
using monoprop::mpi::HybridComm;

namespace {

// Run body(hyb, local_shard) on S threads sharing one HybridComm over MPI_COMM_WORLD; join all.
template <class Body>
auto run_hybrid(int s, Body body) -> std::vector<std::exception_ptr> {
    HybridComm hyb(MPI_COMM_WORLD, s);
    std::vector<std::exception_ptr> errs(static_cast<size_t>(s));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(s));
    for (int r = 0; r < s; ++r) {
        threads.emplace_back([&, r]() {
            try {
                body(hyb, r);
            }
            catch (...) {
                errs[static_cast<size_t>(r)] = std::current_exception();
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    return errs;
}

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

} // namespace

// Global size() and rank() reflect the flat P=R*S world with rank-major ids.
BOOST_AUTO_TEST_CASE(hybrid_comm_flat_size_and_rank) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    for (const int S : {1, 2, 3}) {
        std::vector<int> seen_size(static_cast<size_t>(S), -1);
        std::vector<int> seen_rank(static_cast<size_t>(S), -1);
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            seen_size[static_cast<size_t>(u)] = monoprop::mpi::size(c);
            seen_rank[static_cast<size_t>(u)] = monoprop::mpi::rank(c);
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int u = 0; u < S; ++u) {
            BOOST_CHECK_EQUAL(seen_size[static_cast<size_t>(u)], R * S);
            BOOST_CHECK_EQUAL(seen_rank[static_cast<size_t>(u)], world_rank() * S + u); // rank-major
        }
    }
}

// allreduce_sum: partition g contributes its global id g; every partition ends with sum_{0..P-1} g.
BOOST_AUTO_TEST_CASE(hybrid_comm_allreduce_sum_global) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        const double expected = static_cast<double>(P) * (P - 1) / 2.0;
        std::vector<double> got(static_cast<size_t>(S), -1.0);
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            const double mine = static_cast<double>(monoprop::mpi::rank(c));
            got[static_cast<size_t>(u)] = monoprop::mpi::allreduce_sum<double>(mine, c);
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int u = 0; u < S; ++u) {
            BOOST_CHECK_EQUAL(got[static_cast<size_t>(u)], expected);
        }
    }
}

// begin_alltoallv over the hybrid comm must deliver each source's block CONTIGUOUSLY in ascending
// GLOBAL source order with the sender's tags intact — the property Resolve.h's positional pairing
// relies on. Partition g sends every partition a block of length (g % 3 + 1), each element tagged
// g * 1000 + j. Heterogeneous counts (incl. varying per source) and self blocks are exercised.
BOOST_AUTO_TEST_CASE(hybrid_comm_alltoallv_source_order_and_tags) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        // recv[u] = the vector-of-vectors this rank's shard u received (indexed by global source).
        std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            const int g = monoprop::mpi::rank(c);
            const int len = g % 3 + 1;
            std::vector<std::vector<int>> send(static_cast<size_t>(P));
            for (int d = 0; d < P; ++d) {
                for (int j = 0; j < len; ++j) {
                    send[static_cast<size_t>(d)].push_back(g * 1000 + j);
                }
            }
            auto h = monoprop::mpi::begin_alltoallv(send, c);
            std::vector<std::vector<int>> out;
            h.wait_into(out);
            recv[static_cast<size_t>(u)] = out;
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        // Every local shard u (global id world_rank*S+u) must have received, from each global source
        // src, a block of length (src%3+1) tagged src*1000+j.
        for (int u = 0; u < S; ++u) {
            const auto &out = recv[static_cast<size_t>(u)];
            BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
            for (int src = 0; src < P; ++src) {
                const int len = src % 3 + 1;
                const auto &blk = out[static_cast<size_t>(src)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), len);
                for (int j = 0; j < len; ++j) {
                    BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], src * 1000 + j);
                }
            }
        }
    }
}

#endif // monoprop_ENABLE_MPI
