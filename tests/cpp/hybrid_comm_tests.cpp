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

template <class Body>
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
            BOOST_CHECK_EQUAL(seen_rank[static_cast<size_t>(u)], world_rank() * S + u);
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

// begin_alltoallv must deliver each source's block contiguously in ascending global source order with
// tags intact (Resolve.h's positional pairing). Partition g sends everyone (g%3+1) elts tagged g*1000+j.
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
// window and sizes recv itself. Varying-size rounds check the total, the transpose, and source order.
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
                                       send.data(),
                                       sc.data(),
                                       sd.data(),
                                       recv,
                                       rc.data(),
                                       rd.data(),
                                       sizeof(int),
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

// allreduce_sum_inplace: element-wise global sum over all P partitions, bit-identical on every partition.
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
                hyb.poison(); // partition 0 unwinds before reaching the collective — on every rank
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
