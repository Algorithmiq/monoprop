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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <numeric>
#include <thread>
#include <vector>

#include "ThreadHarness.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/ShmComm.h"

using monoprop::mpi::Comm;
using monoprop::mpi::ShmComm;
using monoprop::mpi::ShmCommPoisoned;

namespace {

template <typename Body>
auto run_shm(int s, Body body) -> std::vector<std::exception_ptr> {
    ShmComm sh(s);
    return test_utils::run_comm_threads(sh, s, body);
}

} // namespace

// alltoall_counts is a transpose: recv[s] on rank r == what s declared it sends to r.
BOOST_AUTO_TEST_CASE(shm_comm_alltoall_counts_transpose) {
    for (const int S : {2, 4, 8}) {
        std::vector<std::vector<int>> recv(static_cast<size_t>(S));
        auto errs = run_shm(S, [&](ShmComm &sh, int r) {
            std::vector<int> send(static_cast<size_t>(S));
            for (int t = 0; t < S; ++t) {
                send[static_cast<size_t>(t)] = r * 100 + t;
            }
            std::vector<int> got(static_cast<size_t>(S));
            sh.alltoall_counts(r, send.data(), got.data());
            recv[static_cast<size_t>(r)] = got;
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int r = 0; r < S; ++r) {
            for (int s = 0; s < S; ++s) {
                BOOST_CHECK_EQUAL(recv[static_cast<size_t>(r)][static_cast<size_t>(s)], s * 100 + r);
            }
        }
    }
}

// begin_alltoallv (the vector-of-vectors facade) must deliver each source's block contiguously in
// ascending source order — what Resolve.h's positional pairing depends on. Rank r sends (r+1) tagged elts.
BOOST_AUTO_TEST_CASE(shm_comm_begin_alltoallv_source_order_and_tags) {
    for (const int S : {2, 4, 8}) {
        std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
        auto errs = run_shm(S, [&](ShmComm &sh, int r) {
            Comm c = Comm::make_shm(&sh, r);
            std::vector<std::vector<int>> send(static_cast<size_t>(S));
            for (int t = 0; t < S; ++t) {
                for (int j = 0; j <= r; ++j) {
                    send[static_cast<size_t>(t)].push_back(r * 1000 + j);
                }
            }
            auto h = monoprop::mpi::begin_alltoallv(send, c);
            std::vector<std::vector<int>> got;
            h.wait_into(got);
            recv[static_cast<size_t>(r)] = got;
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        for (int r = 0; r < S; ++r) {
            const auto &got = recv[static_cast<size_t>(r)];
            BOOST_REQUIRE_EQUAL(static_cast<int>(got.size()), S);
            for (int s = 0; s < S; ++s) {
                const auto &blk = got[static_cast<size_t>(s)];
                BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), s + 1); // source s sent (s+1)
                for (int j = 0; j <= s; ++j) {
                    BOOST_CHECK_EQUAL(blk[static_cast<size_t>(j)], s * 1000 + j);
                }
            }
        }
    }
}

// skip_self: the self slot is neither sent nor received; every other source arrives intact.
BOOST_AUTO_TEST_CASE(shm_comm_begin_alltoallv_skip_self) {
    const int S = 4;
    std::vector<std::vector<std::vector<int>>> recv(static_cast<size_t>(S));
    auto errs = run_shm(S, [&](ShmComm &sh, int r) {
        Comm c = Comm::make_shm(&sh, r);
        std::vector<std::vector<int>> send(static_cast<size_t>(S));
        for (int t = 0; t < S; ++t) {
            send[static_cast<size_t>(t)] = {r * 10 + 1, r * 10 + 2};
        }
        auto h = monoprop::mpi::begin_alltoallv(send, c, /*skip_self=*/true);
        std::vector<std::vector<int>> got;
        h.wait_into(got);
        recv[static_cast<size_t>(r)] = got;
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    for (int r = 0; r < S; ++r) {
        for (int s = 0; s < S; ++s) {
            const auto &blk = recv[static_cast<size_t>(r)][static_cast<size_t>(s)];
            if (s == r) {
                BOOST_CHECK(blk.empty());
            }
            else {
                BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), 2);
                BOOST_CHECK_EQUAL(blk[0], s * 10 + 1);
            }
        }
    }
}

// allreduce_sum is bit-identical on every rank and equals the fixed-order reference.
BOOST_AUTO_TEST_CASE(shm_comm_allreduce_sum_bit_identical) {
    for (const int S : {2, 4, 8}) {
        std::vector<size_t> int_res(static_cast<size_t>(S));
        std::vector<double> dbl_res(static_cast<size_t>(S));
        auto errs = run_shm(S, [&](ShmComm &sh, int r) {
            int_res[static_cast<size_t>(r)] = sh.allreduce_sum<size_t>(r, static_cast<size_t>(r) + 1);
            dbl_res[static_cast<size_t>(r)] = sh.allreduce_sum<double>(r, static_cast<double>(r) + 0.5);
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        const size_t expect_int = static_cast<size_t>(S) * (static_cast<size_t>(S) + 1) / 2; // sum 1..S
        const double expect_dbl = static_cast<double>(S) * static_cast<double>(S) / 2.0;     // sum (r+0.5)
        for (int r = 0; r < S; ++r) {
            BOOST_CHECK_EQUAL(int_res[static_cast<size_t>(r)], expect_int);
            BOOST_CHECK_EQUAL(dbl_res[static_cast<size_t>(r)], dbl_res[0]);
            BOOST_CHECK_CLOSE(dbl_res[static_cast<size_t>(r)], expect_dbl, 1e-12);
        }
    }
}

// allreduce_sum_inplace sums element-wise; every rank ends bit-identical to the ascending-rank-order
// reference. Lengths straddle the slice edges: shorter than S (empty slices), partial lines, many lines.
BOOST_AUTO_TEST_CASE(shm_comm_allreduce_sum_inplace_vector) {
    for (const int S : {2, 4, 8}) {
        for (const size_t N :
             {size_t{1}, size_t{5}, size_t{8 * 2 + 3}, size_t{8} * static_cast<size_t>(S) + 7, size_t{257}}) {
            // Materialize every rank's input once and reduce those exact stored doubles at both sites:
            // aarch64 gcc-14 -ffp-contract=fast fuses `r*0.3 + k*1.7` at one site and not the other (1 ulp).
            std::vector<std::vector<double>> inputs(static_cast<size_t>(S), std::vector<double>(N));
            for (int r = 0; r < S; ++r) {
                for (size_t k = 0; k < N; ++k) {
                    inputs[static_cast<size_t>(r)][k] = static_cast<double>(r + 1) * 0.3 + static_cast<double>(k) * 1.7;
                }
            }
            std::vector<std::vector<double>> res(static_cast<size_t>(S));
            auto errs = run_shm(S, [&](ShmComm &sh, int r) {
                std::vector<double> v = inputs[static_cast<size_t>(r)];
                sh.allreduce_sum_inplace(r, v.data(), N);
                res[static_cast<size_t>(r)] = v;
            });
            for (const auto &e : errs) {
                BOOST_CHECK(e == nullptr);
            }
            std::vector<double> ref(N); // ascending-rank-order reference over the identical stored inputs
            for (size_t k = 0; k < N; ++k) {
                double acc = 0.0;
                for (int r = 0; r < S; ++r) {
                    acc += inputs[static_cast<size_t>(r)][k];
                }
                ref[k] = acc;
            }
            for (int r = 0; r < S; ++r) {
                BOOST_REQUIRE_EQUAL(res[static_cast<size_t>(r)].size(), N);
                for (size_t k = 0; k < N; ++k) {
                    BOOST_CHECK_EQUAL(res[static_cast<size_t>(r)][k], ref[k]);
                }
            }
        }
    }
}

// post_flat_alltoallv over caller-owned flat buffers (the Evolution/Pare replay path): each rank sends
// its own id, so every target must receive [0,1,..,S-1] in source order.
BOOST_AUTO_TEST_CASE(shm_comm_post_flat_alltoallv_flat_buffers) {
    const int S = 4;
    std::vector<std::vector<int>> recv(static_cast<size_t>(S));
    auto errs = run_shm(S, [&](ShmComm &sh, int r) {
        Comm c = Comm::make_shm(&sh, r);
        std::vector<int> send(static_cast<size_t>(S), r);
        std::vector<int> sc(static_cast<size_t>(S), 1), sd(static_cast<size_t>(S));
        for (int i = 0; i < S; ++i) {
            sd[static_cast<size_t>(i)] = i;
        }
        std::vector<int> rc(static_cast<size_t>(S), 1), rd(static_cast<size_t>(S));
        for (int i = 0; i < S; ++i) {
            rd[static_cast<size_t>(i)] = i;
        }
        std::vector<int> out(static_cast<size_t>(S), -1);
        auto ticket = monoprop::mpi::post_flat_alltoallv<int>({.send = send.data(),
                                                               .send_counts = sc.data(),
                                                               .send_displs = sd.data(),
                                                               .recv = out.data(),
                                                               .recv_counts = rc.data(),
                                                               .recv_displs = rd.data()},
                                                              S,
                                                              c);
        ticket.wait();
        recv[static_cast<size_t>(r)] = out;
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    for (int r = 0; r < S; ++r) {
        for (int s = 0; s < S; ++s) {
            BOOST_CHECK_EQUAL(recv[static_cast<size_t>(r)][static_cast<size_t>(s)], s);
        }
    }
}

// alltoallv_resolve resolves recv_counts (the transpose) AND moves the payload in one 2-sync round,
// sizing recv itself — what begin_alltoallv takes for unknown-layout Shm rounds.
BOOST_AUTO_TEST_CASE(shm_comm_alltoallv_resolve_fused) {
    for (const int S : {2, 4, 8}) {
        const int rounds = 25;
        std::atomic<int> failures{0};
        auto errs = run_shm(S, [&](ShmComm &sh, int r) {
            std::vector<int> recv; // reused across rounds (HWM)
            std::vector<int> rc(static_cast<size_t>(S)), rd(static_cast<size_t>(S));
            for (int round = 0; round < rounds; ++round) {
                // Every 5th round is big: pushes the recv HWM up so later rounds run over a larger buffer.
                const auto len_of = [&](int src) {
                    return (round % 5 == 4) ? (src % 3 + 1) * 11 : (src + round) % 4; // includes 0
                };
                const int my_len = len_of(r);
                std::vector<int> send;
                std::vector<int> sc(static_cast<size_t>(S)), sd(static_cast<size_t>(S));
                int off = 0;
                for (int t = 0; t < S; ++t) {
                    sc[static_cast<size_t>(t)] = my_len;
                    sd[static_cast<size_t>(t)] = off;
                    for (int j = 0; j < my_len; ++j) {
                        send.push_back(r * 100000 + round * 100 + j);
                    }
                    off += my_len;
                }
                sh.alltoallv_resolve<int>(r,
                                          {.send = send.data(),
                                           .send_counts = sc.data(),
                                           .send_displs = sd.data(),
                                           .recv = recv,
                                           .recv_counts = rc.data(),
                                           .recv_displs = rd.data()});
                int expected_total = 0;
                for (int s = 0; s < S; ++s) {
                    const int len = len_of(s); // s sends me exactly len(s) (same block to every target)
                    if (rc[static_cast<size_t>(s)] != len) {
                        failures.fetch_add(1);
                    }
                    if (rd[static_cast<size_t>(s)] != expected_total) {
                        failures.fetch_add(1);
                    }
                    for (int j = 0; j < len; ++j) {
                        if (recv[static_cast<size_t>(expected_total + j)] != s * 100000 + round * 100 + j) {
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
}

// Repeated collectives reuse one ShmComm across many rounds without drift (barrier generation reuse).
BOOST_AUTO_TEST_CASE(shm_comm_repeated_collectives) {
    const int S = 8;
    std::atomic<int> failures{0};
    auto errs = run_shm(S, [&](ShmComm &sh, int r) {
        for (int round = 0; round < 200; ++round) {
            const size_t got = sh.allreduce_sum<size_t>(r, static_cast<size_t>(round));
            if (got != static_cast<size_t>(round) * static_cast<size_t>(S)) {
                failures.fetch_add(1);
            }
        }
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    BOOST_CHECK_EQUAL(failures.load(), 0);
}

// Oversubscribed: the barrier's bounded spin must fall back to yielding or spinners starve the completer
// of a core. The test completing at all proves liveness.
BOOST_AUTO_TEST_CASE(shm_comm_oversubscribed_repeated_collectives) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int S = static_cast<int>(std::min(64u, std::max(8u, 2 * hw)));
    std::atomic<int> failures{0};
    auto errs = run_shm(S, [&](ShmComm &sh, int r) {
        for (int round = 0; round < 50; ++round) {
            const size_t got = sh.allreduce_sum<size_t>(r, static_cast<size_t>(round));
            if (got != static_cast<size_t>(round) * static_cast<size_t>(S)) {
                failures.fetch_add(1);
            }
        }
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    BOOST_CHECK_EQUAL(failures.load(), 0);
}

// Poison: if one participant unwinds instead of arriving, peers waiting in a barrier must throw
// ShmCommPoisoned rather than hang. The test completing at all proves no deadlock.
BOOST_AUTO_TEST_CASE(shm_comm_poison_releases_waiters) {
    for (const int S : {2, 4, 8}) {
        auto errs = run_shm(S, [&](ShmComm &sh, int r) {
            if (r == 0) {
                sh.poison(); // simulate an engine exception on rank 0 before it reaches the collective
                return;
            }
            // Peers enter a collective; only S-1 arrive, so they must observe poison and throw.
            std::vector<int> send(static_cast<size_t>(S), 1), got(static_cast<size_t>(S));
            sh.alltoall_counts(r, send.data(), got.data());
        });
        BOOST_CHECK(errs[0] == nullptr);
        for (int r = 1; r < S; ++r) {
            BOOST_REQUIRE(errs[static_cast<size_t>(r)] != nullptr);
            BOOST_CHECK_THROW(std::rethrow_exception(errs[static_cast<size_t>(r)]), ShmCommPoisoned);
        }
    }
}
