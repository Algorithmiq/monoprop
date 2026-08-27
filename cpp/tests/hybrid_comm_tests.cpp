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

// alltoallv_resolve driven directly: it folds the count MPI_Alltoall into the payload verb's B1→B2
// window and sizes recv itself.
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

// A sparse PeerPlan replaces the collectives with point-to-point over the peers the plan names, so the
// two failure modes it can have are DROPPED data and a HANG -- neither of which a dense-path test can
// see. Every rank derives the same pairing from the same (bits, shift), and a block whose destination is
// not a peer must be empty: send only to the plan's peer and check the delivery is exactly that.
BOOST_AUTO_TEST_CASE(hybrid_comm_sparse_plan_delivers_only_to_its_peers) {
    const int R = world_size();
    if (R < 2 || (R & (R - 1)) != 0) {
        return; // the XOR pairing needs a power-of-two rank count
    }
    const int bits = std::countr_zero(static_cast<unsigned>(R));
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        for (int shift = 0; shift < R; ++shift) {
            const monoprop::mpi::PeerPlan plan{.bits = bits, .shift = shift};
            const int peer = plan.peer(world_rank(), 0);
            BOOST_REQUIRE_EQUAL(plan.count(R), 1); // full bits => pairwise
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
    const int bits = std::countr_zero(static_cast<unsigned>(R));
    Comm c{MPI_COMM_WORLD};
    for (int shift = 0; shift < R; ++shift) {
        const monoprop::mpi::PeerPlan plan{.bits = bits, .shift = shift};
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
    const int bits = std::countr_zero(static_cast<unsigned>(R));
    constexpr int kReal = 4;
    constexpr int kBogus = 7; // what a stale or unmasked transpose would claim a non-peer is sending
    for (int shift = 0; shift < R; ++shift) {
        const monoprop::mpi::PeerPlan plan{.bits = bits, .shift = shift};
        const int peer = plan.peer(world_rank(), 0);
        const int bad = (peer + 1) % R; // at full bits the peer set is exactly {peer}
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

#endif // monoprop_ENABLE_MPI
