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
// ON THE STAGING BLOCK ORDER. The offset tables behind alltoallv/alltoallv_resolve were restructured
// (Phase P0 publish, contiguous P1 passes, scatter_off_ deleted), and the ordering expectations below
// were re-derived against the new tables rather than re-run. The conclusion is that they do not move,
// and it is worth writing down why, because it is not what the shape of the diff suggests:
//
//   * The order bytes are DELIVERED in is fixed by recv_displs, which the caller (or, in the fused
//     verb, the ascending prefix over global source) supplies. The staging layout never reaches the
//     caller, so no delivery expectation can depend on it.
//   * The order bytes are STAGED in is unchanged too. The message to rank b still runs destination
//     partition t ascending, and within each t the source partitions u ascending; only the arithmetic
//     that produces a block's start changed, and it is elementwise equal to the old running cursor
//     (see the bit-identity comment above size_staging_send_ in HybridComm.h). So the wire format is
//     the same and there is no mixed-version interop hazard between ranks.
//
// What the old cases could NOT pin is the indexing of the two DERIVED per-destination aggregates the
// change introduces: col_sum_ (the column sums W) on the send side and recv_col_ on the recv side.
// Neither existed before -- the old code carried one running cursor over the raw counts and never
// formed a per-destination total. In every old case a partition sends the same length to every
// destination, and that makes both aggregates CONSTANT along the very index a slip would scramble:
// col_sum_[g] is the same value for every g, and recv_col_[a*S+t] depends only on a. A wrong index
// then reads the right value. Modelled over 8 geometries with R, S in [1, 4]: a transposed or dropped
// destination index on either aggregate is caught in 0 of 8 geometries by the old count patterns, and
// in 6 of 8 (send side) and 4 of 8 (recv side) by the pairwise counts below.
//
// Two things the pairwise cases do NOT add, recorded so nobody re-derives them the hard way:
//
//   * A CONSISTENTLY (dest, source)-transposed staging layout is a genuinely valid alternative tiling.
//     It mismatches 0 blocks at every geometry and every count pattern tried, old or new, and no
//     black-box test can distinguish it at any count pattern. Only the bit-identity comment above
//     size_staging_send_ pins that choice.
//   * A send-side-ONLY transposition is caught by the old uniform cases at least as well as by the new
//     ones (63 of 81 blocks wrong at R=3, S=3, against 46-48 for the pairwise pattern). So is a flat
//     index built as t*R+b instead of b*S+t, when it is the STORE index that slips: 6 of the 9
//     destination bases move even under uniform counts. That slip is invisible only when it is the
//     col_sum_ READ index that slips, which is the first case above.
//
// The two cases below also reach HybridComm::alltoallv, which no case in this file drove at all.

#include <boost/test/unit_test.hpp>

#ifdef monoprop_ENABLE_MPI

#include <atomic>
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
// tags intact (Resolve.h's positional pairing). Note which verb this drives: begin_alltoallv routes a
// hybrid comm with no known recv counts into the FUSED resolve verb, so this and the varying-sizes
// case below exercise alltoallv_resolve, never HybridComm::alltoallv.
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

// Counts that depend on BOTH ends of the leg, unlike every case above, where a partition sends the
// same length to every destination. Round 4 of 5 is the no-zero high-water round that pushes the
// staging HWM up so the smaller rounds after it run over stale staged bytes; the rest include 0 legs.
auto pair_count(int g, int d, int round) -> int {
    if (round % 5 == 4) {
        return (g * 3 + d * 7) % 11 + 1;
    }
    return (g * 2 + d * 3 + round) % 4;
}

// Unique per (source, destination, index), so a block delivered to the right offset from the WRONG
// source is caught instead of silently matching.
auto pair_tag(int g, int d, int j) -> int {
    return (g * 100 + d) * 1000 + j;
}

} // namespace

// HybridComm::alltoallv driven directly. Two independent reasons this case exists:
//
//   * It is the only DIRECT coverage of HybridComm::alltoallv. begin_alltoallv sends a hybrid comm
//     with unknown recv counts to the fused verb, so every other case in this file drives that one
//     instead; the caller-supplied-recv-layout path is reached only through Engine.h's response round
//     (begin_alltoallv with known recv counts) and so only end to end, by
//     mpi_distributed_layer_equivalence, where a wrong staging offset arrives as a wrong answer.
//   * Its count matrix varies along BOTH indices, which is what pins the indexing of col_sum_ and
//     recv_col_, the two derived per-destination aggregates this change introduces. Under the
//     per-source-constant counts every case above uses, both aggregates are constant along the index a
//     slip would scramble, so a mis-indexed read returns the right value anyway. It does NOT pin the
//     tiling: see the file header for what is and is not distinguishable, with counts.
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
            std::vector<int> recv; // reused across rounds: a stale staged byte surfaces as a wrong tag
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
        // Not decoration: a case whose assertions sit inside count-dependent loops can reach none of
        // them and still pass. This counts the PAYLOAD comparisons specifically -- the ones that would
        // vanish if every count came back zero -- rather than any assertion at all.
        BOOST_CHECK_GT(payload_checks.load(), 0);
        BOOST_CHECK_EQUAL(failures.load(), 0);
    }
}

// The same pairwise counts through the fused verb, which resolves the recv layout itself: its recv-side
// staging is sized from the count matrix on partition 0 rather than from published rows, so it is a
// genuinely separate path through the offset tables. rc/rd are outputs here and are checked, not
// supplied.
BOOST_AUTO_TEST_CASE(hybrid_comm_alltoallv_resolve_pairwise_counts) {
    if (world_size() < 2) {
        return;
    }
    const int R = world_size();
    const int rounds = 12;
    for (const int S : {1, 2, 3}) {
        const int P = R * S;
        std::atomic<int> failures{0};
        // Two counters, not one: the layout checks below run once per source unconditionally, so a
        // single combined counter would be satisfied by P * rounds even if every resolved count came
        // back zero and not one payload element were ever examined.
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

#endif // monoprop_ENABLE_MPI
