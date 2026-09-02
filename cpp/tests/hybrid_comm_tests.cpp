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

// HybridComm transport equivalence: R MPI ranks x S in-process partitions must behave as one flat P=R*S
// SPMD world, with only partition 0 touching MPI, exactly as PartitionGroup drives it.
//
// The staging layout never reaches a caller, so no case here distinguishes a consistently transposed
// tiling from the shipped one; only the bit-identity comment in HybridComm.h pins that choice.

#include <boost/test/unit_test.hpp>

#ifdef monoprop_ENABLE_MPI

#include <atomic>
#include <bit>
#include <cstddef>
#include <exception>
#include <numeric>
#include <thread>
#include <vector>

#include <mpi.h>

#include "ThreadHarness.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/HybridComm.h"
#include "monoprop/detail/mpi/MPICompat.h"

using monoprop::mpi::Comm;
using monoprop::mpi::HybridComm;

namespace {

template <typename Body>
auto run_hybrid(int s, Body body) -> std::vector<std::exception_ptr> {
    HybridComm hyb(MPI_COMM_WORLD, s);
    return test_utils::run_comm_threads(hyb, s, body);
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
            BOOST_CHECK_EQUAL(seen_rank[static_cast<size_t>(u)], world_rank() * S + u);
        }
    }
}

// Each partition contributes its global id, so the expected total is sum_{g<P} g.
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

// begin_alltoallv must deliver each source's block contiguously in ascending global source order with
// tags intact (Resolve.h's positional pairing). With no known recv counts this drives the fused verb.
BOOST_AUTO_TEST_CASE(hybrid_comm_alltoallv_source_order_and_tags) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        // recv[u] = what this rank's partition u received, indexed by global source.
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

// Back-to-back alltoallvs with varying counts (zeros, growth, shrink) on one HybridComm: staging and
// offset-table reuse (a stale staged byte surfaces as a wrong tag), and the no-trailing-barrier rule.
BOOST_AUTO_TEST_CASE(hybrid_comm_repeated_alltoallv_varying_sizes) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    const int S = 3;
    const int P = R * S;
    const int rounds = 30;
    std::atomic<int> failures{0};
    auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
        Comm c = Comm::make_hybrid(&hyb, u);
        const int g = monoprop::mpi::rank(c);
        for (int round = 0; round < rounds; ++round) {
            // Every 7th round is "big": pushes the staging HWM up so later rounds expose stale bytes.
            const auto len_of = [&](int src) {
                return (round % 7 == 6) ? (src % 3 + 1) * 17 : (src + round) % 4; // includes 0
            };
            const int my_len = len_of(g);
            std::vector<std::vector<int>> send(static_cast<size_t>(P));
            for (int d = 0; d < P; ++d) {
                for (int j = 0; j < my_len; ++j) {
                    send[static_cast<size_t>(d)].push_back(g * 100000 + round * 100 + j);
                }
            }
            auto h = monoprop::mpi::begin_alltoallv(send, c);
            std::vector<std::vector<int>> out;
            h.wait_into(out);
            if (static_cast<int>(out.size()) != P) {
                failures.fetch_add(1);
                continue;
            }
            for (int src = 0; src < P; ++src) {
                const int len = len_of(src);
                const auto &blk = out[static_cast<size_t>(src)];
                if (static_cast<int>(blk.size()) != len) {
                    failures.fetch_add(1);
                    continue;
                }
                for (int j = 0; j < len; ++j) {
                    if (blk[static_cast<size_t>(j)] != src * 100000 + round * 100 + j) {
                        failures.fetch_add(1);
                    }
                }
            }
        }
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    BOOST_CHECK_EQUAL(failures.load(), 0);
}

// alltoallv_resolve driven directly: it folds the count round into the payload verb's barriered
// windows and sizes recv itself.
BOOST_AUTO_TEST_CASE(hybrid_comm_alltoallv_resolve_fused) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    const int S = 3;
    const int P = R * S;
    const int rounds = 30;
    std::atomic<int> failures{0};
    auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
        Comm c = Comm::make_hybrid(&hyb, u);
        const int g = monoprop::mpi::rank(c);
        std::vector<int> recv; // reused across rounds (staging + recv HWM)
        std::vector<int> rc(static_cast<size_t>(P)), rd(static_cast<size_t>(P));
        for (int round = 0; round < rounds; ++round) {
            const auto len_of = [&](int src) {
                return (round % 7 == 6) ? (src % 3 + 1) * 17 : (src + round) % 4; // includes 0
            };
            const int my_len = len_of(g);
            std::vector<int> send;
            std::vector<int> sc(static_cast<size_t>(P)), sd(static_cast<size_t>(P));
            int off = 0;
            for (int d = 0; d < P; ++d) {
                sc[static_cast<size_t>(d)] = my_len;
                sd[static_cast<size_t>(d)] = off;
                for (int j = 0; j < my_len; ++j) {
                    send.push_back(g * 100000 + round * 100 + j);
                }
                off += my_len;
            }
            hyb.alltoallv_resolve<int>(u,
                                       {.send = send.data(),
                                        .send_counts = sc.data(),
                                        .send_displs = sd.data(),
                                        .recv = recv,
                                        .recv_counts = rc.data(),
                                        .recv_displs = rd.data()},
                                       monoprop::mpi::datatype<int>::get());
            int expected_total = 0;
            for (int src = 0; src < P; ++src) {
                const int len = len_of(src);
                if (rc[static_cast<size_t>(src)] != len || rd[static_cast<size_t>(src)] != expected_total) {
                    failures.fetch_add(1);
                }
                for (int j = 0; j < len; ++j) {
                    if (recv[static_cast<size_t>(expected_total + j)] != src * 100000 + round * 100 + j) {
                        failures.fetch_add(1);
                    }
                }
                expected_total += len;
            }
            if (static_cast<int>(recv.size()) != expected_total) {
                failures.fetch_add(1);
            }
        }
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    BOOST_CHECK_EQUAL(failures.load(), 0);
}

namespace {

// Counts depending on both ends of the leg; every 5th round is a high-water round, so the smaller
// rounds after it run over stale staged bytes.
auto pair_count(int g, int d, int round) -> int {
    if (round % 5 == 4) {
        return (g * 3 + d * 7) % 11 + 1;
    }
    return (g * 2 + d * 3 + round) % 4;
}

// Unique per (source, destination, index), so a block from the wrong source cannot match.
auto pair_tag(int g, int d, int j) -> int {
    return (g * 100 + d) * 1000 + j;
}

} // namespace

// The only direct coverage of HybridComm::alltoallv, and the only count matrix varying along both
// indices, which is what pins the indexing of col_sum_ and recv_col_.
BOOST_AUTO_TEST_CASE(hybrid_comm_alltoallv_pairwise_counts) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    const int rounds = 12;
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        std::atomic<int> failures{0};
        std::atomic<int> payload_checks{0};
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            const int g = monoprop::mpi::rank(c);
            std::vector<int> sc(static_cast<size_t>(P)), sd(static_cast<size_t>(P));
            std::vector<int> rc(static_cast<size_t>(P)), rd(static_cast<size_t>(P));
            std::vector<int> send;
            std::vector<int> recv; // reassigned per round; stage_recv_'s HWM is what carries stale bytes
            for (int round = 0; round < rounds; ++round) {
                send.clear();
                int so = 0;
                int ro = 0;
                for (int d = 0; d < P; ++d) {
                    const int n = pair_count(g, d, round);
                    sc[static_cast<size_t>(d)] = n;
                    sd[static_cast<size_t>(d)] = so;
                    so += n;
                    for (int j = 0; j < n; ++j) {
                        send.push_back(pair_tag(g, d, j));
                    }
                    const int m = pair_count(d, g, round); // the transpose: what d sends me
                    rc[static_cast<size_t>(d)] = m;
                    rd[static_cast<size_t>(d)] = ro;
                    ro += m;
                }
                recv.assign(static_cast<size_t>(ro), -1);
                const auto args = monoprop::mpi::FlatAlltoallvArgs<int>{.send = send.data(),
                                                                        .send_counts = sc.data(),
                                                                        .send_displs = sd.data(),
                                                                        .recv = recv.data(),
                                                                        .recv_counts = rc.data(),
                                                                        .recv_displs = rd.data()}
                                      .bytes();
                hyb.alltoallv(u, args, monoprop::mpi::datatype<int>::get());
                for (int src = 0; src < P; ++src) {
                    const int m = rc[static_cast<size_t>(src)];
                    for (int j = 0; j < m; ++j) {
                        payload_checks.fetch_add(1);
                        if (recv[static_cast<size_t>(rd[static_cast<size_t>(src)] + j)] != pair_tag(src, g, j)) {
                            failures.fetch_add(1);
                        }
                    }
                }
            }
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        // Assertions inside count-dependent loops can all be skipped, so count the payload comparisons.
        BOOST_CHECK_GT(payload_checks.load(), 0);
        BOOST_CHECK_EQUAL(failures.load(), 0);
    }
}

// The same pairwise counts through the fused verb, whose recv staging is sized from the count matrix
// rather than from published rows. rc/rd are outputs here, and are checked.
BOOST_AUTO_TEST_CASE(hybrid_comm_alltoallv_resolve_pairwise_counts) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    const int rounds = 12;
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        std::atomic<int> failures{0};
        // Two counters: the layout checks run unconditionally, so one combined counter would pass on them.
        std::atomic<int> layout_checks{0};
        std::atomic<int> payload_checks{0};
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            const int g = monoprop::mpi::rank(c);
            std::vector<int> sc(static_cast<size_t>(P)), sd(static_cast<size_t>(P));
            std::vector<int> rc(static_cast<size_t>(P)), rd(static_cast<size_t>(P));
            std::vector<int> send;
            std::vector<int> recv; // reused across rounds (staging + recv HWM)
            for (int round = 0; round < rounds; ++round) {
                send.clear();
                int so = 0;
                for (int d = 0; d < P; ++d) {
                    const int n = pair_count(g, d, round);
                    sc[static_cast<size_t>(d)] = n;
                    sd[static_cast<size_t>(d)] = so;
                    so += n;
                    for (int j = 0; j < n; ++j) {
                        send.push_back(pair_tag(g, d, j));
                    }
                }
                hyb.alltoallv_resolve<int>(u,
                                           {.send = send.data(),
                                            .send_counts = sc.data(),
                                            .send_displs = sd.data(),
                                            .recv = recv,
                                            .recv_counts = rc.data(),
                                            .recv_displs = rd.data()},
                                           monoprop::mpi::datatype<int>::get());
                int total = 0;
                for (int src = 0; src < P; ++src) {
                    const int m = pair_count(src, g, round);
                    layout_checks.fetch_add(1);
                    if (rc[static_cast<size_t>(src)] != m || rd[static_cast<size_t>(src)] != total) {
                        failures.fetch_add(1);
                    }
                    for (int j = 0; j < m; ++j) {
                        payload_checks.fetch_add(1);
                        if (recv[static_cast<size_t>(total + j)] != pair_tag(src, g, j)) {
                            failures.fetch_add(1);
                        }
                    }
                    total += m;
                }
                layout_checks.fetch_add(1);
                if (static_cast<int>(recv.size()) != total) {
                    failures.fetch_add(1);
                }
            }
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        BOOST_CHECK_GT(layout_checks.load(), 0);
        BOOST_CHECK_GT(payload_checks.load(), 0);
        BOOST_CHECK_EQUAL(failures.load(), 0);
    }
}

BOOST_AUTO_TEST_CASE(hybrid_comm_allreduce_sum_inplace_global) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        // Lengths straddle the slice-partition edges: shorter than S, partial cache lines, many lines.
        for (const size_t N : {size_t{1}, size_t{5}, size_t{8 * 3 + 3}, size_t{257}}) {
            std::vector<std::vector<double>> res(static_cast<size_t>(S));
            auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
                Comm c = Comm::make_hybrid(&hyb, u);
                const int g = monoprop::mpi::rank(c);
                std::vector<double> v(N);
                for (size_t k = 0; k < N; ++k) {
                    v[k] = static_cast<double>(g + 1) * 0.25 + static_cast<double>(k);
                }
                hyb.allreduce_sum_inplace(u, v.data(), N);
                res[static_cast<size_t>(u)] = v;
            });
            for (const auto &e : errs) {
                BOOST_CHECK(e == nullptr);
            }
            // sum over g of ((g+1)*0.25 + k) = P(P+1)/8 + P*k
            for (int u = 0; u < S; ++u) {
                BOOST_REQUIRE_EQUAL(res[static_cast<size_t>(u)].size(), N);
                for (size_t k = 0; k < N; ++k) {
                    const double expect =
                        static_cast<double>(P) * (P + 1) / 8.0 + static_cast<double>(P) * static_cast<double>(k);
                    BOOST_CHECK_CLOSE(res[static_cast<size_t>(u)][k], expect, 1e-12);
                    BOOST_CHECK_EQUAL(res[static_cast<size_t>(u)][k], res[0][k]); // bit-identical
                }
            }
        }
    }
}

// Poison releases barrier waiters on every rank; the test completing at all proves that. Partition 0 poisons
// before entering a collective, so no rank is committed to MPI and the partition-0 guard deliberately does
// not fire. Poisoning inside a collective calls MPI_Abort instead, so it cannot be a ctest case.
BOOST_AUTO_TEST_CASE(hybrid_comm_poison_releases_waiters) {
    if (world_size() < 2) {
        return;
    }
    for (const int S : {2, 3}) {
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            if (u == 0) {
                hyb.poison();
                return;
            }
            std::vector<int> send(static_cast<size_t>(world_size() * S), 1);
            std::vector<int> got(static_cast<size_t>(world_size() * S));
            hyb.alltoall_counts(u, send.data(), got.data());
        });
        BOOST_CHECK(errs[0] == nullptr);
        for (int u = 1; u < S; ++u) {
            BOOST_REQUIRE(errs[static_cast<size_t>(u)] != nullptr);
            BOOST_CHECK_THROW(std::rethrow_exception(errs[static_cast<size_t>(u)]), monoprop::mpi::ShmCommPoisoned);
        }
    }
}

// The pairing is derived independently on both ends, so it must be an involution: whoever I send to
// sends back to me. A plan on which it fails deadlocks rather than mis-delivers, and no transport case
// below can distinguish the two.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_peer_is_an_involution) {
    for (const int shift : {0, 1, 2, 3, 5, 8, 13, 255}) {
        const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
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
    const monoprop::mpi::PeerPlan dense_plan{};
    BOOST_REQUIRE(dense_plan.dense());
    BOOST_REQUIRE_EQUAL(dense_plan.count(7), 7);
    for (int k = 0; k < 7; ++k) {
        BOOST_REQUIRE_EQUAL(dense_plan.peer(3, k), k); // dense ignores `me`: every rank is a peer
        BOOST_REQUIRE(dense_plan.contains(3, k));
    }
}

// A sparse PeerPlan replaces the collectives with point-to-point over the one peer the plan names, so
// the two failure modes it can have are DROPPED data and a HANG -- neither of which a dense-path test
// can see. Every rank derives the same pairing from the same shift, and a block whose destination is
// not the peer must be empty: send only to the plan's peer and check the delivery is exactly that.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_delivers_only_to_its_peers) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return; // the XOR pairing needs a power-of-two rank count
    }
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        for (int shift = 0; shift < R; ++shift) {
            const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
            const int peer = plan.peer(world_rank(), 0);
            BOOST_REQUIRE_EQUAL(plan.count(R), 1); // sparse is always pairwise
            std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
            auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
                Comm c = Comm::make_hybrid(&hyb, u);
                const int g = monoprop::mpi::rank(c);
                std::vector<std::vector<int>> send(static_cast<size_t>(P));
                for (int t = 0; t < S; ++t) {
                    auto &blk = send[static_cast<size_t>(peer * S + t)];
                    for (int j = 0; j <= t; ++j) {
                        blk.push_back(g * 1000 + t * 10 + j);
                    }
                }
                auto h = monoprop::mpi::begin_alltoallv(send,
                                                        c,
                                                        /*skip_self=*/false,
                                                        /*known_recv_counts=*/nullptr,
                                                        plan);
                std::vector<std::vector<int>> out;
                h.wait_into(out);
                recv[static_cast<size_t>(u)] = out;
            });
            for (const auto &e : errs) {
                BOOST_CHECK(e == nullptr);
            }
            // XOR is an involution, so whoever I send to sends to me: my sources are exactly `peer`'s
            // partitions, and the block from (peer, su) to my partition t has t+1 entries.
            for (int t = 0; t < S; ++t) {
                const auto &out = recv[static_cast<size_t>(t)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
                for (int src = 0; src < P; ++src) {
                    const auto &blk = out[static_cast<size_t>(src)];
                    if (src / S != peer) {
                        BOOST_CHECK(blk.empty()); // a non-peer must not appear at all
                        continue;
                    }
                    BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), t + 1);
                    for (int j = 0; j <= t; ++j) {
                        BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], src * 1000 + t * 10 + j);
                    }
                }
            }
        }
    }
}

// Same contract on the S == 1 world, which takes the pure-MPI branch (MPI_Ialltoallv vs Isend/Irecv)
// rather than HybridComm's staged one, and on the KNOWN-recv-counts path the response round uses.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_on_the_plain_mpi_path) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    Comm c{MPI_COMM_WORLD};
    for (int shift = 0; shift < R; ++shift) {
        const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(world_rank(), 0);
        std::vector<std::vector<int>> send(static_cast<size_t>(R));
        for (int j = 0; j < 4; ++j) {
            send[static_cast<size_t>(peer)].push_back(world_rank() * 1000 + j);
        }
        // Unknown recv layout: the counts round is point-to-point too.
        std::vector<std::vector<int>> out;
        monoprop::mpi::begin_alltoallv(send, c, false, nullptr, plan).wait_into(out);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), R);
        for (int src = 0; src < R; ++src) {
            if (src != peer) {
                BOOST_CHECK(out[static_cast<size_t>(src)].empty());
                continue;
            }
            BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(src)].size()), 4);
            for (int j = 0; j < 4; ++j) {
                BOOST_CHECK_EQUAL(out[static_cast<size_t>(src)][static_cast<size_t>(j)], src * 1000 + j);
            }
        }
        // Known recv layout (the response round): counts are the transpose, so peer-only again.
        std::vector<int> known(static_cast<size_t>(R), 0);
        known[static_cast<size_t>(peer)] = 4;
        std::vector<std::vector<int>> out2;
        monoprop::mpi::begin_alltoallv(send, c, false, &known, plan).wait_into(out2);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out2.size()), R);
        BOOST_CHECK(out2[static_cast<size_t>(peer)] == out[static_cast<size_t>(peer)]);
    }
}

// known_recv_counts is CALLER-supplied, so it can carry a count for a rank the plan does not name --
// the response round's transpose is only as masked as whatever produced it. No receive is ever posted
// for a non-peer, so an unmasked count sizes recv_buffer for bytes nothing writes and wait_into would
// hand that slot to the caller as data. Both the plain-MPI and the staged HybridComm path must drop it.
BOOST_AUTO_TEST_CASE(hybrid_comm_known_recv_counts_are_masked_through_the_plan) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    constexpr int kReal = 4;
    constexpr int kBogus = 7; // what a stale or unmasked transpose would claim a non-peer is sending
    for (int shift = 0; shift < R; ++shift) {
        const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(world_rank(), 0);
        const int bad = (peer + 1) % R; // the peer set is exactly {peer}
        BOOST_REQUIRE(bad != peer);

        // S == 1: the plain-MPI Isend/Irecv branch of begin_alltoallv.
        {
            Comm c{MPI_COMM_WORLD};
            std::vector<std::vector<int>> send(static_cast<size_t>(R));
            for (int j = 0; j < kReal; ++j) {
                send[static_cast<size_t>(peer)].push_back(world_rank() * 1000 + j);
            }
            std::vector<int> known(static_cast<size_t>(R), 0);
            known[static_cast<size_t>(peer)] = kReal;
            known[static_cast<size_t>(bad)] = kBogus;
            std::vector<std::vector<int>> out;
            monoprop::mpi::begin_alltoallv(send, c, false, &known, plan).wait_into(out);
            BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), R);
            BOOST_CHECK(out[static_cast<size_t>(bad)].empty()); // unmasked, this holds kBogus elements
            BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(peer)].size()), kReal);
            for (int j = 0; j < kReal; ++j) {
                BOOST_CHECK_EQUAL(out[static_cast<size_t>(peer)][static_cast<size_t>(j)], peer * 1000 + j);
            }
        }

        // S == 2: the same array through HybridComm's staged alltoallv.
        constexpr int S = 2;
        const int P = R * S;
        std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            const int g = monoprop::mpi::rank(c);
            std::vector<std::vector<int>> send(static_cast<size_t>(P));
            std::vector<int> known(static_cast<size_t>(P), 0);
            for (int t = 0; t < S; ++t) {
                for (int j = 0; j < kReal; ++j) {
                    send[static_cast<size_t>((peer * S) + t)].push_back((g * 1000) + j);
                }
                known[static_cast<size_t>((peer * S) + t)] = kReal;
                known[static_cast<size_t>((bad * S) + t)] = kBogus;
            }
            std::vector<std::vector<int>> out;
            monoprop::mpi::begin_alltoallv(send, c, false, &known, plan).wait_into(out);
            recv[static_cast<size_t>(u)] = out;
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int t = 0; t < S; ++t) {
            const auto &out = recv[static_cast<size_t>(t)];
            BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
            for (int su = 0; su < S; ++su) {
                BOOST_CHECK(out[static_cast<size_t>((bad * S) + su)].empty());
                const auto &blk = out[static_cast<size_t>((peer * S) + su)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), kReal);
                for (int j = 0; j < kReal; ++j) {
                    BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], (((peer * S) + su) * 1000) + j);
                }
            }
        }
    }
}

namespace {

// Varies along BOTH ends and hits 0, so a block landing on the wrong peer or the wrong partition
// changes a length, not just a value.
auto sparse_count(int src, int dst) -> int {
    return ((src * 3) + (dst * 5)) % 4;
}
auto sparse_tag(int src, int dst, int j) -> int {
    return (((src * 128) + dst) * 1000) + j;
}

} // namespace

// A zero-count leg is where a send/recv posting asymmetry deadlocks rather than mis-delivers: both ends
// must skip on the SAME value. Nothing above ever sends an empty block over a real message, so force
// one -- the lower-numbered end of every pair sends nothing while its peer sends four.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_with_an_empty_leg) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    const int me = world_rank();
    constexpr int kLen = 4;
    for (int shift = 1; shift < R; ++shift) { // shift 0 is the self peer, covered separately
        const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(me, 0);
        BOOST_REQUIRE(peer != me);
        const int my_len = me < peer ? 0 : kLen; // exactly one end of the pair is silent
        const int peer_len = peer < me ? 0 : kLen;

        {
            Comm c{MPI_COMM_WORLD};
            std::vector<std::vector<int>> send(static_cast<size_t>(R));
            for (int j = 0; j < my_len; ++j) {
                send[static_cast<size_t>(peer)].push_back(sparse_tag(me, peer, j));
            }
            std::vector<std::vector<int>> out;
            monoprop::mpi::begin_alltoallv(send, c, false, nullptr, plan).wait_into(out);
            BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), R);
            BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(peer)].size()), peer_len);
            for (int j = 0; j < peer_len; ++j) {
                BOOST_CHECK_EQUAL(out[static_cast<size_t>(peer)][static_cast<size_t>(j)], sparse_tag(peer, me, j));
            }
        }

        for (const int S : {1, 2}) {
            const int P = R * S;
            std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
            auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
                Comm c = Comm::make_hybrid(&hyb, u);
                const int g = monoprop::mpi::rank(c);
                std::vector<std::vector<int>> send(static_cast<size_t>(P));
                for (int t = 0; t < S; ++t) {
                    const int d = (peer * S) + t;
                    for (int j = 0; j < my_len; ++j) {
                        send[static_cast<size_t>(d)].push_back(sparse_tag(g, d, j));
                    }
                }
                std::vector<std::vector<int>> out;
                monoprop::mpi::begin_alltoallv(send, c, false, nullptr, plan).wait_into(out);
                recv[static_cast<size_t>(u)] = out;
            });
            for (const auto &e : errs) {
                BOOST_CHECK(e == nullptr);
            }
            for (int t = 0; t < S; ++t) {
                const int g = (me * S) + t;
                const auto &out = recv[static_cast<size_t>(t)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
                for (int su = 0; su < S; ++su) {
                    const auto &blk = out[static_cast<size_t>((peer * S) + su)];
                    BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), peer_len);
                    for (int j = 0; j < peer_len; ++j) {
                        BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], sparse_tag((peer * S) + su, g, j));
                    }
                }
            }
        }
    }
}

// shift == 0 makes every rank its own and only peer, so the sparse path's self slot is the whole
// exchange -- and skip_self then removes the one leg that would have moved anything for a partition.
// The remaining S-1 in-rank legs must still arrive.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_skip_self_at_shift_zero) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    const int me = world_rank();
    const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = 0};
    BOOST_REQUIRE_EQUAL(plan.peer(me, 0), me);

    {
        Comm c{MPI_COMM_WORLD}; // the self peer is the ONLY peer, and skip_self drops it
        std::vector<std::vector<int>> send(static_cast<size_t>(R));
        for (int j = 0; j < 5; ++j) {
            send[static_cast<size_t>(me)].push_back(sparse_tag(me, me, j));
        }
        std::vector<std::vector<int>> out;
        monoprop::mpi::begin_alltoallv(send, c, /*skip_self=*/true, nullptr, plan).wait_into(out);
        BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), R);
        for (const auto &blk : out) {
            BOOST_CHECK(blk.empty());
        }
    }

    for (const int S : {2, 3}) {
        const int P = R * S;
        std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            const int g = monoprop::mpi::rank(c);
            std::vector<std::vector<int>> send(static_cast<size_t>(P));
            for (int t = 0; t < S; ++t) {
                const int d = (me * S) + t;
                for (int j = 0; j <= t; ++j) {
                    send[static_cast<size_t>(d)].push_back(sparse_tag(g, d, j));
                }
            }
            std::vector<std::vector<int>> out;
            monoprop::mpi::begin_alltoallv(send, c, /*skip_self=*/true, nullptr, plan).wait_into(out);
            recv[static_cast<size_t>(u)] = out;
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int t = 0; t < S; ++t) {
            const int g = (me * S) + t;
            const auto &out = recv[static_cast<size_t>(t)];
            BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
            for (int src = 0; src < P; ++src) {
                const bool in_rank = (src / S) == me;
                const int want = (in_rank && src != g) ? t + 1 : 0; // own slot dropped by skip_self
                BOOST_REQUIRE_EQUAL(static_cast<int>(out[static_cast<size_t>(src)].size()), want);
                for (int j = 0; j < want; ++j) {
                    BOOST_CHECK_EQUAL(out[static_cast<size_t>(src)][static_cast<size_t>(j)], sparse_tag(src, g, j));
                }
            }
        }
    }
}

// Consecutive gate exchanges post their rounds on ONE communicator under kFlatPayloadTag, and the
// non-overtaking argument in Pairwise.h is what says a later receive cannot match an earlier send.
// Two back-to-back sparse rounds, the second on the transpose of the first's counts, assert it.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_back_to_back_rounds) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    const int me = world_rank();
    Comm c{MPI_COMM_WORLD};
    for (int shift = 0; shift < R; ++shift) {
        const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(me, 0);
        const int len = 3 + (me % 2); // asymmetric, so a swapped round is a length mismatch

        std::vector<std::vector<int>> q(static_cast<size_t>(R));
        for (int j = 0; j < len; ++j) {
            q[static_cast<size_t>(peer)].push_back(sparse_tag(me, peer, j));
        }
        std::vector<std::vector<int>> q_out;
        monoprop::mpi::begin_alltoallv(q, c, false, nullptr, plan).wait_into(q_out);

        // Round 2 immediately, same comm and tag, sized from round 1's transpose -- the response shape.
        std::vector<int> known(static_cast<size_t>(R), 0);
        known[static_cast<size_t>(peer)] = len;
        std::vector<std::vector<int>> r(static_cast<size_t>(R));
        const int back = static_cast<int>(q_out[static_cast<size_t>(peer)].size());
        for (int j = 0; j < back; ++j) {
            r[static_cast<size_t>(peer)].push_back(q_out[static_cast<size_t>(peer)][static_cast<size_t>(j)] + 7);
        }
        std::vector<std::vector<int>> r_out;
        monoprop::mpi::begin_alltoallv(r, c, false, &known, plan).wait_into(r_out);

        BOOST_REQUIRE_EQUAL(back, 3 + (peer % 2));
        BOOST_REQUIRE_EQUAL(static_cast<int>(r_out[static_cast<size_t>(peer)].size()), len);
        for (int j = 0; j < len; ++j) {
            BOOST_CHECK_EQUAL(r_out[static_cast<size_t>(peer)][static_cast<size_t>(j)], sparse_tag(me, peer, j) + 7);
        }
        for (int src = 0; src < R; ++src) {
            if (src != peer) {
                BOOST_CHECK(r_out[static_cast<size_t>(src)].empty());
            }
        }
    }
}

namespace {

// One fused-resolve round under `plan`: flattens send_blocks, runs the verb, and re-splits the payload
// by global source using the resolved counts, so a case compares delivered BLOCKS rather than offsets.
auto resolve_round(HybridComm &hyb,
                   int u,
                   int P,
                   const std::vector<std::vector<int>> &send_blocks,
                   monoprop::mpi::PeerPlan plan) -> std::vector<std::vector<int>> {
    std::vector<int> send;
    std::vector<int> sc(static_cast<size_t>(P)), sd(static_cast<size_t>(P));
    for (int d = 0; d < P; ++d) {
        const auto &blk = send_blocks[static_cast<size_t>(d)];
        sc[static_cast<size_t>(d)] = static_cast<int>(blk.size());
        sd[static_cast<size_t>(d)] = static_cast<int>(send.size());
        send.insert(send.end(), blk.begin(), blk.end());
    }
    std::vector<int> recv;
    std::vector<int> rc(static_cast<size_t>(P)), rd(static_cast<size_t>(P));
    hyb.alltoallv_resolve<int>(u,
                               {.send = send.data(),
                                .send_counts = sc.data(),
                                .send_displs = sd.data(),
                                .recv = recv,
                                .recv_counts = rc.data(),
                                .recv_displs = rd.data()},
                               monoprop::mpi::datatype<int>::get(),
                               plan);
    std::vector<std::vector<int>> out(static_cast<size_t>(P));
    for (int src = 0; src < P; ++src) {
        const auto off = static_cast<size_t>(rd[static_cast<size_t>(src)]);
        const auto n = static_cast<size_t>(rc[static_cast<size_t>(src)]);
        out[static_cast<size_t>(src)].assign(recv.begin() + static_cast<std::ptrdiff_t>(off),
                                             recv.begin() + static_cast<std::ptrdiff_t>(off + n));
    }
    return out;
}

// Every leg INTO an odd rank is empty, so a multi-peer plan always has a peer whose payload legs are
// skipped entirely while its S*S-int count block still travels. Both ends read the same function of
// (src, dst), which is what keeps the skip symmetric.
auto muted_count(int src, int dst, int S) -> int {
    return (dst / S) % 2 == 1 ? 0 : sparse_count(src, dst);
}

} // namespace

// The fused resolve POSTS its count round in B1→B2 and drains it in B3→B4, with every partition's
// pack_send_ running underneath. Only the sparse arm splits -- plan.dense() is a blocking MPI_Alltoall
// -- so drive the SAME peer-masked layout through both arms and require the delivered blocks to agree
// element for element. A dropped or torn count block changes a length here, not just a value.
BOOST_AUTO_TEST_CASE(hybrid_comm_resolve_split_count_round_matches_the_dense_arm) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return; // the XOR pairing needs a power-of-two rank count
    }
    const int me = world_rank();
    int cases = 0;
    {
        constexpr int f = 1; // a sparse plan is fanout 1 by construction; f > 1 is no longer expressible
        for (int shift = 0; shift < R; ++shift) {
            const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
            BOOST_REQUIRE_EQUAL(plan.count(R), f);
            ++cases;
            for (const int S : {1, 2, 3}) {
                const int P = R * S;
                // Peer-masked, so the dense plan carries the identical bytes: its non-peer blocks are
                // empty rather than absent, and a dense count of zero and a sparse absence must agree.
                const auto fill = [&](int g) {
                    std::vector<std::vector<int>> send(static_cast<size_t>(P));
                    for (int k = 0; k < f; ++k) {
                        const int b = plan.peer(me, k);
                        for (int t = 0; t < S; ++t) {
                            const int d = (b * S) + t;
                            for (int j = 0; j < muted_count(g, d, S); ++j) {
                                send[static_cast<size_t>(d)].push_back(sparse_tag(g, d, j));
                            }
                        }
                    }
                    return send;
                };
                std::vector<std::vector<std::vector<int>>> dense(static_cast<size_t>(S));
                std::vector<std::vector<std::vector<int>>> sparse(static_cast<size_t>(S));
                auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
                    const int g = (me * S) + u;
                    dense[static_cast<size_t>(u)] = resolve_round(hyb, u, P, fill(g), {});
                    sparse[static_cast<size_t>(u)] = resolve_round(hyb, u, P, fill(g), plan);
                });
                for (const auto &e : errs) {
                    BOOST_CHECK(e == nullptr);
                }
                for (int t = 0; t < S; ++t) {
                    const int g = (me * S) + t;
                    const auto &got = sparse[static_cast<size_t>(t)];
                    const auto &want_dense = dense[static_cast<size_t>(t)];
                    BOOST_REQUIRE_EQUAL(static_cast<int>(got.size()), P);
                    BOOST_REQUIRE_EQUAL(static_cast<int>(want_dense.size()), P);
                    for (int src = 0; src < P; ++src) {
                        const int want = plan.contains(me, src / S) ? muted_count(src, g, S) : 0;
                        const auto &blk = got[static_cast<size_t>(src)];
                        BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), want);
                        BOOST_REQUIRE_EQUAL(static_cast<int>(want_dense[static_cast<size_t>(src)].size()), want);
                        for (int j = 0; j < want; ++j) {
                            BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], sparse_tag(src, g, j));
                            BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)],
                                              want_dense[static_cast<size_t>(src)][static_cast<size_t>(j)]);
                        }
                    }
                }
            }
        }
    }
    BOOST_TEST(cases > 0); // at R < 2 the case is a no-op and must not read as coverage
}

// shift == 0 at full bits makes this rank its own and only peer, so the count round posts NOTHING:
// sparse_pairwise memcpys the S*S-int block in place and the B3→B4 wait must be a no-op rather than a
// wait on stale requests. The S^2 in-rank payload legs still have to arrive.
BOOST_AUTO_TEST_CASE(hybrid_comm_resolve_split_count_round_self_peer_only) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    const int me = world_rank();
    const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = 0};
    BOOST_REQUIRE_EQUAL(plan.peer(me, 0), me);
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            const int g = (me * S) + u;
            std::vector<std::vector<int>> send(static_cast<size_t>(P));
            for (int t = 0; t < S; ++t) {
                const int d = (me * S) + t;
                for (int j = 0; j <= t; ++j) {
                    send[static_cast<size_t>(d)].push_back(sparse_tag(g, d, j));
                }
            }
            recv[static_cast<size_t>(u)] = resolve_round(hyb, u, P, send, plan);
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int t = 0; t < S; ++t) {
            const int g = (me * S) + t;
            const auto &out = recv[static_cast<size_t>(t)];
            BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
            for (int src = 0; src < P; ++src) {
                const int want = (src / S) == me ? t + 1 : 0; // block (su -> t) has t+1 entries
                const auto &blk = out[static_cast<size_t>(src)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), want);
                for (int j = 0; j < want; ++j) {
                    BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], sparse_tag(src, g, j));
                }
            }
        }
    }
}

// A peer with nothing to send is where the split can hang: pack_send_ copies zero bytes in B2→B3 while
// the count round is still live, and the payload round posts no leg for it at all. Back-to-back rounds,
// the second silent, so the second also runs over the first's staging high-water bytes.
BOOST_AUTO_TEST_CASE(hybrid_comm_resolve_split_count_round_zero_count_peer) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return;
    }
    const int me = world_rank();
    for (int shift = 1; shift < R; ++shift) { // shift 0 is the self peer, covered above
        const monoprop::mpi::PeerPlan plan{.sparse = true, .shift = shift};
        const int peer = plan.peer(me, 0);
        BOOST_REQUIRE(peer != me);
        const int my_len = me < peer ? 0 : 4; // exactly one end of the pair is silent
        const int peer_len = peer < me ? 0 : 4;
        for (const int S : {1, 2}) {
            const int P = R * S;
            std::vector<std::vector<std::vector<int>>> loud(static_cast<size_t>(S));
            std::vector<std::vector<std::vector<int>>> quiet(static_cast<size_t>(S));
            auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
                const int g = (me * S) + u;
                std::vector<std::vector<int>> send(static_cast<size_t>(P));
                for (int t = 0; t < S; ++t) {
                    const int d = (peer * S) + t;
                    for (int j = 0; j < my_len; ++j) {
                        send[static_cast<size_t>(d)].push_back(sparse_tag(g, d, j));
                    }
                }
                loud[static_cast<size_t>(u)] = resolve_round(hyb, u, P, send, plan);
                quiet[static_cast<size_t>(u)] =
                    resolve_round(hyb, u, P, std::vector<std::vector<int>>(static_cast<size_t>(P)), plan);
            });
            for (const auto &e : errs) {
                BOOST_CHECK(e == nullptr);
            }
            for (int t = 0; t < S; ++t) {
                const int g = (me * S) + t;
                const auto &out = loud[static_cast<size_t>(t)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(out.size()), P);
                for (int su = 0; su < S; ++su) {
                    const auto &blk = out[static_cast<size_t>((peer * S) + su)];
                    BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), peer_len);
                    for (int j = 0; j < peer_len; ++j) {
                        BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], sparse_tag((peer * S) + su, g, j));
                    }
                }
                // The all-silent round: every resolved count is zero, so a stale staged byte cannot hide.
                for (const auto &blk : quiet[static_cast<size_t>(t)]) {
                    BOOST_CHECK(blk.empty());
                }
            }
        }
    }
}

// derived_wire_plan_ is what lets alltoallv narrow its own wire when the caller cannot: only partition 0
// reaches MPI, and ITS row may be the empty one while a sibling partition holds the rank's only traffic.
// Reading the peer set off partition 0's row alone resolves to the SELF peer, whose legs are all zero --
// every block is then dropped with no hang to show for it. So this case puts the traffic where partition
// 0 cannot see it, and checks the payload arrives carrying its source's global id.
//
// alltoallv's derive_wire_bits has no library caller today, so this is its only exercise.
BOOST_AUTO_TEST_CASE(hybrid_comm_derived_wire_plan_reads_every_partitions_row) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return; // the XOR pairing needs a power-of-two rank count
    }
    const int bits = std::countr_zero(static_cast<unsigned>(R));
    const int me = world_rank();
    const int peer = me ^ 1; // shift 1, so every rank derives the same pairing
    constexpr int kLen = 5;

    for (const int S : {2, 4}) {
        const int P = R * S;
        const int carrier = S - 1; // never partition 0 -- that is the whole point of the case
        std::vector<std::vector<int>> got(static_cast<size_t>(S));
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            const int g = (me * S) + u;
            std::vector<int> counts(static_cast<size_t>(P), 0);
            const std::vector<int> displs(static_cast<size_t>(P), 0);
            std::vector<int> send;
            if (u == carrier) {
                counts[static_cast<size_t>((peer * S) + carrier)] = kLen;
                for (int j = 0; j < kLen; ++j) {
                    send.push_back((g * 1000) + j);
                }
            }
            // Symmetric layout: my peer's carrier partition sends me exactly what I send it, which is
            // what derive_wire_bits requires and what lets the recv rows name the peer set. One block, so
            // both displacements are zero.
            std::vector<int> recv(static_cast<size_t>(u == carrier ? kLen : 0), -1);
            const monoprop::mpi::AlltoallvArgs args{.send = reinterpret_cast<const std::byte *>(send.data()),
                                                    .send_counts = counts.data(),
                                                    .send_displs = displs.data(),
                                                    .recv = reinterpret_cast<std::byte *>(recv.data()),
                                                    .recv_counts = counts.data(),
                                                    .recv_displs = displs.data(),
                                                    .elem = sizeof(int)};
            hyb.alltoallv(u, args, MPI_INT, monoprop::mpi::PeerPlan{}, /*derive_wire_bits=*/bits);
            got[static_cast<size_t>(u)] = recv;
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int u = 0; u < S; ++u) {
            if (u != carrier) {
                BOOST_CHECK(got[static_cast<size_t>(u)].empty());
                continue;
            }
            // The values name their sender, so this fails on a plan that named the wrong peer as well as
            // on one that named none.
            const int src = (peer * S) + carrier;
            BOOST_REQUIRE_EQUAL(static_cast<int>(got[static_cast<size_t>(u)].size()), kLen);
            for (int j = 0; j < kLen; ++j) {
                BOOST_CHECK_EQUAL(got[static_cast<size_t>(u)][static_cast<size_t>(j)], (src * 1000) + j);
            }
        }
    }
}

#endif // monoprop_ENABLE_MPI
