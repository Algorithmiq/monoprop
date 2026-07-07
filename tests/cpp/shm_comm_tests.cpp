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

#include <atomic>
#include <exception>
#include <numeric>
#include <thread>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/ShmComm.h"

using monoprop::mpi::Comm;
using monoprop::mpi::ShmComm;
using monoprop::mpi::ShmCommPoisoned;

namespace {

// Run `body(sh, rank)` on S participant threads sharing one ShmComm; join all. Exceptions thrown by a
// body are captured per-rank (so Boost.Test assertions stay on the main thread, where they are safe).
template <class Body>
auto run_shm(int s, Body body) -> std::vector<std::exception_ptr> {
    ShmComm sh(s);
    std::vector<std::exception_ptr> errs(static_cast<size_t>(s));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(s));
    for (int r = 0; r < s; ++r) {
        threads.emplace_back([&, r]() {
            try {
                body(sh, r);
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

} // namespace


// alltoall_counts is a transpose: recv[s] on rank r == what s declared it sends to r.
BOOST_AUTO_TEST_CASE(shm_comm_alltoall_counts_transpose) {
    for (const int S : {2, 4, 8}) {
        std::vector<std::vector<int>> recv(static_cast<size_t>(S));
        auto errs = run_shm(S, [&](ShmComm &sh, int r) {
            std::vector<int> send(static_cast<size_t>(S));
            for (int t = 0; t < S; ++t) {
                send[static_cast<size_t>(t)] = r * 100 + t; // r sends (r*100+t) to t
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

// begin_alltoallv (the vector-of-vectors facade) over a Shm comm must deliver each source's block
// contiguously and tagged, in ascending source order — the property Resolve.h's positional pairing
// depends on. Rank r sends target t a block of length (r+1) tagged r*1000+j (sender-determined count).
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
                BOOST_CHECK(blk.empty()); // self slot skipped
            }
            else {
                BOOST_REQUIRE_EQUAL(static_cast<int>(blk.size()), 2);
                BOOST_CHECK_EQUAL(blk[0], s * 10 + 1);
            }
        }
    }
}

// allreduce_sum is bit-identical on every rank and equals the fixed-order reference. Integer + double.
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
        const double expect_dbl = static_cast<double>(S) * static_cast<double>(S) / 2.0;      // sum (r+0.5)
        for (int r = 0; r < S; ++r) {
            BOOST_CHECK_EQUAL(int_res[static_cast<size_t>(r)], expect_int);
            BOOST_CHECK_EQUAL(dbl_res[static_cast<size_t>(r)], dbl_res[0]); // identical across ranks
            BOOST_CHECK_CLOSE(dbl_res[static_cast<size_t>(r)], expect_dbl, 1e-12);
        }
    }
}

// allreduce_sum_inplace on a per-rank vector sums element-wise; every rank ends identical.
BOOST_AUTO_TEST_CASE(shm_comm_allreduce_sum_inplace_vector) {
    const int S = 4;
    const size_t N = 5;
    std::vector<std::vector<double>> res(static_cast<size_t>(S));
    auto errs = run_shm(S, [&](ShmComm &sh, int r) {
        std::vector<double> v(N);
        for (size_t k = 0; k < N; ++k) {
            v[k] = static_cast<double>(r) * 10.0 + static_cast<double>(k);
        }
        sh.allreduce_sum_inplace(r, v.data(), N);
        res[static_cast<size_t>(r)] = v;
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    // sum_r (r*10 + k) over r=0..3 = 60 + 4k
    for (int r = 0; r < S; ++r) {
        for (size_t k = 0; k < N; ++k) {
            BOOST_CHECK_CLOSE(res[static_cast<size_t>(r)][k], 60.0 + 4.0 * static_cast<double>(k), 1e-12);
        }
    }
}

// post_flat_alltoallv over caller-owned flat buffers (the Evolution/Pare replay path). Each rank sends
// one element (its rank) to every target; target r receives [0,1,..,S-1] in source order.
BOOST_AUTO_TEST_CASE(shm_comm_post_flat_alltoallv_flat_buffers) {
    const int S = 4;
    std::vector<std::vector<int>> recv(static_cast<size_t>(S));
    auto errs = run_shm(S, [&](ShmComm &sh, int r) {
        Comm c = Comm::make_shm(&sh, r);
        std::vector<int> send(static_cast<size_t>(S), r); // one element per target, value = my rank
        std::vector<int> sc(static_cast<size_t>(S), 1), sd(static_cast<size_t>(S));
        for (int i = 0; i < S; ++i) {
            sd[static_cast<size_t>(i)] = i;
        }
        std::vector<int> rc(static_cast<size_t>(S), 1), rd(static_cast<size_t>(S));
        for (int i = 0; i < S; ++i) {
            rd[static_cast<size_t>(i)] = i;
        }
        std::vector<int> out(static_cast<size_t>(S), -1);
        auto ticket = monoprop::mpi::post_flat_alltoallv<int>(
            send.data(), sc.data(), sd.data(), out.data(), rc.data(), rd.data(), S, c);
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

// Poison: if one participant unwinds instead of arriving, peers waiting in a barrier must throw
// ShmCommPoisoned rather than hang forever. The test completing at all proves no deadlock.
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
        BOOST_CHECK(errs[0] == nullptr); // rank 0 returned cleanly
        for (int r = 1; r < S; ++r) {
            BOOST_REQUIRE(errs[static_cast<size_t>(r)] != nullptr);
            BOOST_CHECK_THROW(std::rethrow_exception(errs[static_cast<size_t>(r)]), ShmCommPoisoned);
        }
    }
}

