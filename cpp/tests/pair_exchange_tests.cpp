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

// pair_exchange, the one-round gate exchange, over every transport it dispatches to: ShmComm in-rank,
// HybridComm in-rank and across ranks, and a raw communicator. Every word names its (source, destination,
// index), so a block that lands on the wrong partition, arrives out of source order, or is torn changes a
// value or a length rather than passing unnoticed. The MPI cases self-skip below two ranks.

#include <boost/test/unit_test.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#ifdef monoprop_ENABLE_MPI
#include <mpi.h>
#endif

#include "ThreadHarness.h"
#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/PairExchange.h"
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/HybridComm.h"
#endif

using monoprop::mpi::CollectiveArgumentError;
using monoprop::mpi::Comm;
using monoprop::mpi::PairRecv;
using monoprop::mpi::ShmComm;
using monoprop::mpi::ShmCommPoisoned;
using monoprop::mpi::SubStreams;

namespace {

// Global ids: slot = rank * S + partition, the same rank-major numbering HybridComm uses.
auto tag_word(size_t src_slot, size_t dst_slot, size_t j) -> size_t {
    return (src_slot << 44) | (dst_slot << 24) | j;
}

auto mix(uint64_t x) -> uint64_t {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// The traffic shapes. Each is a function of (gate, source, destination) both ends can evaluate, so the
// receiver knows the exact length it must see from every source.
enum class Pattern : std::uint8_t {
    Random,     // 0..200 words per (u, t), independent
    AllEmpty,   // every sub-stream empty: the header alone crosses
    OneSource,  // only the last partition of each rank sends, to every destination
    OneDest,    // every partition sends only to destination partition 0
    FlatLinear, // u sends only to t = u ^ d, d = gate % S -- the routing the engine actually produces
    LowerSilent // the lower-numbered rank of a pair sends nothing: the payload is zero on one side
};

constexpr std::array kInRankPatterns{Pattern::Random,
                                     Pattern::AllEmpty,
                                     Pattern::OneSource,
                                     Pattern::OneDest,
                                     Pattern::FlatLinear};
constexpr std::array kCrossRankPatterns{Pattern::Random,
                                        Pattern::AllEmpty,
                                        Pattern::OneSource,
                                        Pattern::OneDest,
                                        Pattern::FlatLinear,
                                        Pattern::LowerSilent};

auto expected_len(Pattern p, uint64_t gate, int S, int src_rank, int src_part, int dst_rank, int dst_part) -> size_t {
    const auto random = [&] {
        return static_cast<size_t>(mix((gate * 1000003ULL + static_cast<uint64_t>(src_rank * S + src_part)) * 7919ULL
                                       + static_cast<uint64_t>(dst_rank * S + dst_part))
                                   % 201);
    };
    switch (p) {
        case Pattern::Random:
            return random();
        case Pattern::AllEmpty:
            return 0;
        case Pattern::OneSource:
            return src_part == S - 1 ? 100 + static_cast<size_t>(dst_part) : 0;
        case Pattern::OneDest:
            return dst_part == 0 ? 50 + static_cast<size_t>(src_part) : 0;
        case Pattern::FlatLinear:
            return dst_part == (src_part ^ static_cast<int>(gate % static_cast<uint64_t>(S)))
                       ? 60 + 3 * static_cast<size_t>(src_part)
                       : 0;
        case Pattern::LowerSilent:
            return src_rank < dst_rank ? 0 : random();
    }
    return 0;
}

// One gate's outgoing buffers of one partition. A caller alternates two of these (the lifetime rule in
// PairExchange.h), so `fill` reuses the storage a gate two back last handed out.
struct Outgoing {
    std::vector<std::vector<size_t>> bufs;
    std::vector<std::span<const size_t>> spans;

    auto fill(Pattern p, uint64_t gate, int S, int my_rank, int u, int peer_rank) -> void {
        bufs.resize(static_cast<size_t>(S));
        spans.resize(static_cast<size_t>(S));
        for (int t = 0; t < S; ++t) {
            auto &b = bufs[static_cast<size_t>(t)];
            const size_t n = expected_len(p, gate, S, my_rank, u, peer_rank, t);
            b.resize(n);
            for (size_t j = 0; j < n; ++j) {
                b[j] = tag_word(static_cast<size_t>(my_rank * S + u), static_cast<size_t>(peer_rank * S + t), j);
            }
            spans[static_cast<size_t>(t)] = b;
        }
    }
    [[nodiscard]] auto view() const -> SubStreams { return spans; }
};

// What partition t of `my_rank` must have received from every partition of `peer_rank`. Counts every
// comparison so a case whose loops never ran cannot read as a pass.
auto verify(const PairRecv &got,
            Pattern p,
            uint64_t gate,
            int S,
            int my_rank,
            int t,
            int peer_rank,
            std::atomic<int> &failures,
            std::atomic<long> &checks) -> void {
    if (static_cast<int>(got.from.size()) != S) {
        failures.fetch_add(1);
        return;
    }
    for (int u = 0; u < S; ++u) {
        const auto blk = got.from[static_cast<size_t>(u)];
        const size_t want = expected_len(p, gate, S, peer_rank, u, my_rank, t);
        checks.fetch_add(1);
        if (blk.size() != want) {
            failures.fetch_add(1);
            continue;
        }
        for (size_t j = 0; j < want; ++j) {
            checks.fetch_add(1);
            if (blk[j] != tag_word(static_cast<size_t>(peer_rank * S + u), static_cast<size_t>(my_rank * S + t), j)) {
                failures.fetch_add(1);
            }
        }
    }
}

template <typename Body>
auto run_shm(int s, Body body) -> std::vector<std::exception_ptr> {
    ShmComm sh(s);
    return test_utils::run_comm_threads(sh, s, body);
}

} // namespace

// Every (u -> t) block of every shape arrives intact and in ascending source order, at S from one partition
// to eight. Each gate here is a fresh ShmComm: the back-to-back rule has its own case below.
BOOST_AUTO_TEST_CASE(pair_exchange_shm_delivers_every_block) {
    for (const int S : {1, 2, 3, 4, 8}) {
        for (const Pattern p : kInRankPatterns) {
            std::atomic<int> failures{0};
            std::atomic<long> checks{0};
            // Owned outside the threads: a partition that returns first must not free what its peers
            // still view (rule 1 in PairExchange.h has no next call to bound the last gate).
            std::vector<Outgoing> outs(static_cast<size_t>(S));
            auto errs = run_shm(S, [&](ShmComm &sh, int u) {
                Comm c = Comm::make_shm(&sh, u);
                Outgoing &out = outs[static_cast<size_t>(u)];
                out.fill(p, /*gate=*/1, S, /*my_rank=*/0, u, /*peer_rank=*/0);
                const PairRecv got = monoprop::mpi::pair_exchange(c, /*rank_shift=*/0, out.view());
                verify(got, p, 1, S, 0, u, 0, failures, checks);
            });
            for (const auto &e : errs) {
                BOOST_CHECK(e == nullptr);
            }
            BOOST_CHECK_GT(checks.load(), 0);
            BOOST_CHECK_EQUAL(failures.load(), 0);
        }
    }
}

// Ten gates on one ShmComm, the shape changing every gate and the caller alternating two buffer sets as
// the lifetime rule requires. A peer publishing gate g+1 while this partition still holds gate g's views
// is the race the descriptor parity exists for, so gate g is re-verified after gate g+1's buffers are
// prepared and the peers have had time to run ahead into their next call.
BOOST_AUTO_TEST_CASE(pair_exchange_shm_back_to_back_gates) {
    constexpr int kGates = 10;
    for (const int S : {2, 3, 4, 8}) {
        std::atomic<int> failures{0};
        std::atomic<long> checks{0};
        std::vector<std::array<Outgoing, 2>> all_sets(static_cast<size_t>(S)); // outlives the threads
        auto errs = run_shm(S, [&](ShmComm &sh, int u) {
            Comm c = Comm::make_shm(&sh, u);
            std::array<Outgoing, 2> &sets = all_sets[static_cast<size_t>(u)];
            PairRecv prev{};
            Pattern prev_pattern = Pattern::AllEmpty;
            for (int g = 0; g < kGates; ++g) {
                const Pattern p = kInRankPatterns[static_cast<size_t>(g) % kInRankPatterns.size()];
                Outgoing &out = sets[static_cast<size_t>(g & 1)];
                out.fill(p, static_cast<uint64_t>(g), S, 0, u, 0);
                if (g > 0) {
                    std::this_thread::yield();
                    verify(prev, prev_pattern, static_cast<uint64_t>(g - 1), S, 0, u, 0, failures, checks);
                }
                prev = monoprop::mpi::pair_exchange(c, 0, out.view());
                prev_pattern = p;
                verify(prev, p, static_cast<uint64_t>(g), S, 0, u, 0, failures, checks);
            }
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        BOOST_CHECK_GT(checks.load(), 0);
        BOOST_CHECK_EQUAL(failures.load(), 0);
    }
}

// The argument checks run before any barrier and on replicated inputs, so every participant throws and
// nobody is left spinning. A single-rank world has no peer: any non-zero shift is an error, as is a
// sub-stream count that is not S.
BOOST_AUTO_TEST_CASE(pair_exchange_shm_rejects_bad_arguments) {
    const int S = 3;
    {
        auto errs = run_shm(S, [&](ShmComm &sh, int u) {
            Comm c = Comm::make_shm(&sh, u);
            Outgoing out;
            out.fill(Pattern::Random, 0, S, 0, u, 0);
            monoprop::mpi::pair_exchange(c, /*rank_shift=*/1, out.view());
        });
        for (const auto &e : errs) {
            BOOST_REQUIRE(e != nullptr);
            BOOST_CHECK_THROW(std::rethrow_exception(e), CollectiveArgumentError);
        }
    }
    {
        auto errs = run_shm(S, [&](ShmComm &sh, int u) {
            Comm c = Comm::make_shm(&sh, u);
            Outgoing out;
            out.fill(Pattern::Random, 0, S + 1, 0, u, 0); // S+1 sub-streams
            monoprop::mpi::pair_exchange(c, 0, out.view());
        });
        for (const auto &e : errs) {
            BOOST_REQUIRE(e != nullptr);
            BOOST_CHECK_THROW(std::rethrow_exception(e), CollectiveArgumentError);
        }
    }
}

// A participant that unwinds instead of arriving must release the peers spinning in the verb's barrier.
BOOST_AUTO_TEST_CASE(pair_exchange_shm_poison_releases_waiters) {
    for (const int S : {2, 4, 8}) {
        auto errs = run_shm(S, [&](ShmComm &sh, int u) {
            if (u == 0) {
                sh.poison();
                return;
            }
            Comm c = Comm::make_shm(&sh, u);
            Outgoing out;
            out.fill(Pattern::Random, 0, S, 0, u, 0);
            monoprop::mpi::pair_exchange(c, 0, out.view());
        });
        BOOST_CHECK(errs[0] == nullptr);
        for (int u = 1; u < S; ++u) {
            BOOST_REQUIRE(errs[static_cast<size_t>(u)] != nullptr);
            BOOST_CHECK_THROW(std::rethrow_exception(errs[static_cast<size_t>(u)]), ShmCommPoisoned);
        }
    }
}

#ifdef monoprop_ENABLE_MPI

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

// The shifts a case runs: the in-rank one and the two cross-rank ones the engine's routing produces most
// (deduplicated at R == 2, where they coincide).
auto shifts_for(int R) -> std::vector<int> {
    std::vector<int> s{0, 1};
    if (R - 1 != 1) {
        s.push_back(R - 1);
    }
    return s;
}

auto power_of_two(int n) -> bool {
    return n >= 2 && (n & (n - 1)) == 0;
}

} // namespace

// Every (rank r, partition u) -> (rank r ^ shift, partition t) block arrives bit for bit, in ascending u,
// on every shape including empty rows, empty columns, and a side with nothing to send at all.
BOOST_AUTO_TEST_CASE(pair_exchange_hybrid_delivers_every_block) {
    const int R = world_size();
    if (!power_of_two(R)) {
        return;
    }
    const int me = world_rank();
    for (const int S : {1, 2, 4}) {
        for (const int shift : shifts_for(R)) {
            const int peer = me ^ shift;
            for (const Pattern p : kCrossRankPatterns) {
                std::atomic<int> failures{0};
                std::atomic<long> checks{0};
                std::vector<Outgoing> outs(static_cast<size_t>(S)); // outlives the threads
                auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
                    Comm c = Comm::make_hybrid(&hyb, u);
                    Outgoing &out = outs[static_cast<size_t>(u)];
                    out.fill(p, 3, S, me, u, peer);
                    const PairRecv got = monoprop::mpi::pair_exchange(c, shift, out.view());
                    verify(got, p, 3, S, me, u, peer, failures, checks);
                });
                for (const auto &e : errs) {
                    BOOST_CHECK(e == nullptr);
                }
                BOOST_CHECK_GT(checks.load(), 0);
                BOOST_CHECK_EQUAL(failures.load(), 0);
            }
        }
    }
}

// Consecutive gates with different shifts on one HybridComm: cross-rank (two barriers, one message each
// way) and in-rank (one barrier) gates interleave, the receive buffer grows and is reused, and the
// caller alternates its buffer sets. Gate g's views are re-verified after gate g+1 is prepared.
BOOST_AUTO_TEST_CASE(pair_exchange_hybrid_consecutive_gates_change_shift) {
    const int R = world_size();
    if (!power_of_two(R)) {
        return;
    }
    const int me = world_rank();
    const std::vector<int> shifts{1, 0, R - 1, 1, 1, 0, 0, R - 1, 1, R - 1, 0, 1};
    for (const int S : {1, 2, 4}) {
        std::atomic<int> failures{0};
        std::atomic<long> checks{0};
        std::vector<std::array<Outgoing, 2>> all_sets(static_cast<size_t>(S)); // outlives the threads
        auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
            Comm c = Comm::make_hybrid(&hyb, u);
            std::array<Outgoing, 2> &sets = all_sets[static_cast<size_t>(u)];
            PairRecv prev{};
            Pattern prev_pattern = Pattern::AllEmpty;
            int prev_peer = me;
            for (size_t g = 0; g < shifts.size(); ++g) {
                const int shift = shifts[g];
                const int peer = me ^ shift;
                const Pattern p = kCrossRankPatterns[g % kCrossRankPatterns.size()];
                Outgoing &out = sets[g & 1];
                out.fill(p, g, S, me, u, peer);
                if (g > 0) {
                    std::this_thread::yield();
                    verify(prev, prev_pattern, g - 1, S, me, u, prev_peer, failures, checks);
                }
                prev = monoprop::mpi::pair_exchange(c, shift, out.view());
                prev_pattern = p;
                prev_peer = peer;
                verify(prev, p, g, S, me, u, peer, failures, checks);
            }
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
        BOOST_CHECK_GT(checks.load(), 0);
        BOOST_CHECK_EQUAL(failures.load(), 0);
    }
}

// The raw-communicator kind is the S == 1 world: one sub-stream out, one view back, the self peer at
// shift 0 and one probed message each way otherwise. Back-to-back gates reuse the per-thread receive
// buffer across a growth and a silent side.
BOOST_AUTO_TEST_CASE(pair_exchange_plain_mpi_comm) {
    const int R = world_size();
    if (!power_of_two(R)) {
        return;
    }
    const int me = world_rank();
    Comm c{MPI_COMM_WORLD};
    std::atomic<int> failures{0};
    std::atomic<long> checks{0};
    std::array<Outgoing, 2> sets;
    uint64_t gate = 0;
    for (const int shift : shifts_for(R)) {
        const int peer = me ^ shift;
        for (const Pattern p : kCrossRankPatterns) {
            Outgoing &out = sets[gate & 1];
            out.fill(p, gate, 1, me, 0, peer);
            const PairRecv got = monoprop::mpi::pair_exchange(c, shift, out.view());
            verify(got, p, gate, 1, me, 0, peer, failures, checks);
            ++gate;
        }
    }
    BOOST_CHECK_GT(checks.load(), 0);
    BOOST_CHECK_EQUAL(failures.load(), 0);
}

// A shift that names no rank -- `R` itself sets a bit above every rank index -- is rejected on every
// participant before any barrier or MPI call, so the check cannot strand a rank. Both the hybrid and the
// raw kind.
BOOST_AUTO_TEST_CASE(pair_exchange_rejects_shift_outside_the_world) {
    const int R = world_size();
    if (!power_of_two(R)) {
        return;
    }
    const int me = world_rank();
    const int S = 2;
    auto errs = run_hybrid(S, [&](HybridComm &hyb, int u) {
        Comm c = Comm::make_hybrid(&hyb, u);
        Outgoing out;
        out.fill(Pattern::Random, 0, S, me, u, me);
        monoprop::mpi::pair_exchange(c, /*rank_shift=*/R, out.view());
    });
    for (const auto &e : errs) {
        BOOST_REQUIRE(e != nullptr);
        BOOST_CHECK_THROW(std::rethrow_exception(e), CollectiveArgumentError);
    }
    Comm raw{MPI_COMM_WORLD};
    Outgoing out;
    out.fill(Pattern::Random, 0, 1, me, 0, me);
    BOOST_CHECK_THROW(monoprop::mpi::pair_exchange(raw, R, out.view()), CollectiveArgumentError);
}

#endif // monoprop_ENABLE_MPI
