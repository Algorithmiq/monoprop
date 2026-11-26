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

// First on purpose: the header must compile on its own.
#include "monoprop/detail/sharded/Exchange.h"

/*
 * The physical-MPI round of the sharded runtime (detail/sharded/Exchange.h), driven the way the orchestration drives
 * it: owners fill their count rows and pack their own slices inside protected phases, the primary plans, posts and
 * waits in primary-only phases, and destination owners read the received blocks after the checkpoint that follows
 * the wait. The cases need two or more ranks: the whole-suite MPI variants run them at T = 1 and
 * cpp/tests/CMakeLists.txt launches them again at T = 2.
 *
 * Oracle: every block's size and contents are a closed-form function of (source rank, source shard, destination
 * rank, destination shard, element index, round), so each receiver checks exactly what it must have received
 * without trusting the sender's layout. Workers never call BOOST_TEST; they record mismatches in per-shard slots.
 */

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ShardedTestSupport.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/sharded/Team.h"

namespace {

namespace sharded = monoprop::detail::sharded;
using sharded::ExchangeElement;
using sharded::ExchangeTransport;
using sharded::PhysicalExchange;
using sharded::PhysicalWorld;
using sharded_test::message_of;
using sharded_test::team_options;

constexpr int exchange_tag = 0x5E01;

auto world() -> PhysicalWorld {
    return PhysicalWorld::of(monoprop::mpi::Comm(MPI_COMM_WORLD));
}

// Block size from source (rank a, shard u) to destination (rank b, shard t) in round `round`: 0 .. 5 elements, so
// empty legs, empty shards and whole empty peers all occur, and a scaled round grows the staging.
auto block_size(size_t a, size_t u, size_t b, size_t t, size_t round, size_t scale) -> size_t {
    return ((a * 7) + (u * 3) + (b * 5) + (t * 11) + (round * 13)) % 6 * scale;
}

template <class T>
auto encode(size_t a, size_t u, size_t b, size_t t, size_t i) -> T {
    // Unique per (source, destination, index) and exact in every element type.
    const auto v = static_cast<uint64_t>((((((a * 16) + u) * 16 + b) * 16 + t) << 12) + i);
    if constexpr (std::is_same_v<T, double>) {
        return static_cast<double>(v) + 0.25;
    }
    else {
        return static_cast<T>(v);
    }
}

template <class T>
constexpr auto element_of() -> ExchangeElement {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return ExchangeElement::u32;
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        return ExchangeElement::u64;
    }
    else {
        return ExchangeElement::f64;
    }
}

struct RoundSpec {
    std::vector<size_t> peers;
    ExchangeTransport transport = ExchangeTransport::pairwise;
    bool counts_known = false; // receivers fill their receive rows; otherwise the count round supplies them
    size_t round = 0;
    size_t scale = 1;
};

// One round through the phase protocol; returns per-shard mismatch descriptions and the join error.
template <class T>
auto run_round(PhysicalExchange &ex, const PhysicalWorld &w, const RoundSpec &spec)
    -> std::pair<std::exception_ptr, std::vector<std::string>> {
    const auto options = team_options();
    const auto threads = static_cast<size_t>(options.threads);
    const size_t me = w.rank;
    std::vector<std::vector<std::string>> problems(threads);
    const auto error = sharded::run_team(options, [&](size_t s, sharded::TeamFailure &failure) noexcept {
        // Owner s fills its rows: what it sends to every remote shard, and (when known) what it receives.
        if (!sharded::phase(failure, s, [&] {
                ex.reset_rows(s);
                for (size_t k = 0; k < spec.peers.size(); ++k) {
                    const size_t b = spec.peers[k];
                    for (size_t t = 0; t < threads; ++t) {
                        ex.set_send_count(s, k, t, block_size(me, s, b, t, spec.round, spec.scale));
                        if (spec.counts_known) {
                            ex.set_recv_count(s, k, t, block_size(b, t, me, s, spec.round, spec.scale));
                        }
                    }
                }
            })) {
            return;
        }
        if (!sharded::phase(failure, s, [&] {
                if (s == 0) {
                    ex.plan_send(spec.peers, spec.transport);
                    if (!spec.counts_known) {
                        ex.post_counts();
                    }
                }
            })) {
            return;
        }
        // Each owner packs its own disjoint slices.
        if (!sharded::phase(failure, s, [&] {
                for (size_t k = 0; k < spec.peers.size(); ++k) {
                    const size_t b = spec.peers[k];
                    for (size_t t = 0; t < threads; ++t) {
                        auto block = ex.send_block<T>(s, k, t);
                        if (block.size() != block_size(me, s, b, t, spec.round, spec.scale)) {
                            throw std::logic_error("send slice has the wrong size");
                        }
                        for (size_t i = 0; i < block.size(); ++i) {
                            block[i] = encode<T>(me, s, b, t, i);
                        }
                    }
                }
            })) {
            return;
        }
        if (!sharded::phase(failure, s, [&] {
                if (s == 0) {
                    if (!spec.counts_known) {
                        ex.wait_counts();
                    }
                    ex.plan_recv();
                    ex.post();
                    ex.wait();
                }
            })) {
            return;
        }
        // Destination owner s checks every block it received, against the closed form.
        static_cast<void>(sharded::phase(failure, s, [&] {
            for (size_t k = 0; k < spec.peers.size(); ++k) {
                const size_t a = spec.peers[k];
                for (size_t u = 0; u < threads; ++u) {
                    const size_t expected = block_size(a, u, me, s, spec.round, spec.scale);
                    if (ex.recv_count(s, k, u) != expected) {
                        problems[s].push_back(std::format("count from ({}, {}) is {}, expected {}",
                                                          a,
                                                          u,
                                                          ex.recv_count(s, k, u),
                                                          expected));
                    }
                    const auto block = ex.recv_block<T>(s, k, u);
                    if (block.size() != expected) {
                        problems[s].push_back(std::format("block from ({}, {}) has {} values, expected {}",
                                                          a,
                                                          u,
                                                          block.size(),
                                                          expected));
                        continue;
                    }
                    for (size_t i = 0; i < block.size(); ++i) {
                        if (block[i] != encode<T>(a, u, me, s, i)) {
                            problems[s].push_back(std::format("block from ({}, {}) differs at {}", a, u, i));
                            break;
                        }
                    }
                }
            }
        }));
    });
    std::vector<std::string> flat;
    for (auto &p : problems) {
        flat.insert(flat.end(), p.begin(), p.end());
    }
    return {error, flat};
}

auto describe(const std::exception_ptr &error) -> std::string {
    return error ? message_of(error) : "none";
}

template <class T>
auto check_rounds(const std::vector<RoundSpec> &specs) -> void {
    const auto w = world();
    PhysicalExchange ex(w, static_cast<size_t>(team_options().threads), element_of<T>(), exchange_tag);
    for (const auto &spec : specs) {
        BOOST_TEST_CONTEXT("rank " << w.rank << " of " << w.ranks << ", T " << team_options().threads << ", round "
                                   << spec.round << " scale " << spec.scale << ", "
                                   << (spec.transport == ExchangeTransport::pairwise ? "pairwise" : "collective")
                                   << ", counts " << (spec.counts_known ? "known" : "exchanged") << ", "
                                   << spec.peers.size() << " peer(s)") {
            const auto [error, problems] = run_round<T>(ex, w, spec);
            BOOST_TEST(describe(error) == "none");
            BOOST_TEST(problems.empty(), (problems.empty() ? std::string{} : problems.front()));
            BOOST_TEST(ex.live() == 0);
        }
    }
}

// By default the scales grow, shrink to an empty round and grow again on the same object: staging is reused, never
// assumed fresh.
auto every_variant(const std::vector<size_t> &peers, const std::vector<size_t> &scales = {1, 300, 0, 2})
    -> std::vector<RoundSpec> {
    std::vector<RoundSpec> specs;
    size_t round = 0;
    for (const size_t scale : scales) {
        for (const auto transport : {ExchangeTransport::pairwise, ExchangeTransport::collective}) {
            for (const bool known : {false, true}) {
                specs.push_back(
                    {.peers = peers, .transport = transport, .counts_known = known, .round = round++, .scale = scale});
            }
        }
    }
    return specs;
}

} // namespace

BOOST_AUTO_TEST_CASE(sharded_exchange_moves_every_block_to_its_owner, *test_utils::min_ranks(2)) {
    const auto peers = sharded::other_ranks(world());
    check_rounds<double>(every_variant(peers));
    check_rounds<uint64_t>(every_variant(peers));
    check_rounds<uint32_t>(every_variant(peers));
}

BOOST_AUTO_TEST_CASE(sharded_exchange_count_blocks_of_every_width_arrive, *test_utils::min_ranks(2)) {
    // Counts of at most 5, 300 and 100000 travel as 1-, 2- and 4-byte counts; a later light round after a heavy one
    // shrinks the block again. The receiver must read each block's width from its size.
    const auto peers = sharded::other_ranks(world());
    std::vector<RoundSpec> specs;
    size_t round = 0;
    for (const size_t scale : {size_t{1}, size_t{60}, size_t{20000}, size_t{1}, size_t{60}}) {
        for (const auto transport : {ExchangeTransport::pairwise, ExchangeTransport::collective}) {
            specs.push_back(
                {.peers = peers, .transport = transport, .counts_known = false, .round = round++, .scale = scale});
        }
    }
    check_rounds<uint32_t>(specs);
}

BOOST_AUTO_TEST_CASE(sharded_exchange_sparse_peer_round, *test_utils::even_ranks()) {
    const auto w = world();
    check_rounds<uint64_t>(every_variant({w.rank ^ 1U}));
}

BOOST_AUTO_TEST_CASE(sharded_exchange_payload_totals_must_fit_mpi_counts, *test_utils::min_ranks(2)) {
    // Per-peer totals are prefix sums of int counts; a sum past INT_MAX is refused by plan_send, before posting.
    const auto w = world();
    const size_t threads = 2;
    PhysicalExchange ex(w, threads, ExchangeElement::f64, exchange_tag);
    ex.reset_rows(0);
    ex.reset_rows(1);
    const int big = std::numeric_limits<int>::max();
    ex.set_send_count(0, 0, 0, static_cast<size_t>(big));
    ex.set_send_count(1, 0, 0, static_cast<size_t>(big));
    BOOST_CHECK_THROW(ex.plan_send(std::vector<size_t>{sharded::other_ranks(w).front()}, ExchangeTransport::pairwise),
                      std::length_error);
    BOOST_TEST(ex.live() == 0);
}
