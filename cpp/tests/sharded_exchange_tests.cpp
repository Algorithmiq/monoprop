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
 * the wait. cpp/tests/CMakeLists.txt launches the multi-rank cases under mpiexec with 2 and 3 processes at
 * T = 1, 2 and 4; at world size 1 they skip, and only the single-process contract cases run.
 *
 * Oracle: every block's size and contents are a closed-form function of (source rank, source shard, destination
 * rank, destination shard, element index, round), so each receiver checks exactly what it must have received
 * without trusting the sender's layout. Workers never call BOOST_TEST; they record mismatches in per-shard slots.
 */

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/Team.h"

namespace {

namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;
using sharded::ExchangeElement;
using sharded::ExchangeTransport;
using sharded::PhysicalExchange;
using sharded::PhysicalWorld;

constexpr int kTag = 0x5E01;

auto team_options() -> parallel::Options {
    return parallel::capture_thread_budget();
}

auto world() -> PhysicalWorld {
    return PhysicalWorld::of(monoprop::mpi::Comm(MPI_COMM_WORLD));
}

// Multi-rank cases run only in their dedicated launches, which name the world size (cpp/tests/CMakeLists.txt).
auto dedicated_launch() -> bool {
    const char *text = std::getenv("monoprop_TEST_SHARDED_RANKS");
    return text != nullptr && std::stoul(text) == world().ranks;
}

auto has_two_ranks(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const auto ranks = world().ranks;
    boost::test_tools::assertion_result result(ranks >= 2 && dedicated_launch());
    result.message() << "the launch has " << ranks << " MPI rank(s); this case needs a dedicated launch of at least 2";
    return result;
}

auto has_even_ranks(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const auto ranks = world().ranks;
    boost::test_tools::assertion_result result(ranks >= 2 && ranks % 2 == 0 && dedicated_launch());
    result.message() << "the launch has " << ranks << " MPI rank(s); the XOR peer needs a dedicated launch of an even "
                     << "count of at least 2";
    return result;
}

auto has_nonprimary_worker(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const auto threads = team_options().threads;
    boost::test_tools::assertion_result result(threads >= 2);
    result.message() << "the captured team budget is " << threads << "; this case needs a nonprimary worker";
    return result;
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

auto all_other_ranks(const PhysicalWorld &w) -> std::vector<size_t> {
    std::vector<size_t> peers;
    for (size_t b = 0; b < w.ranks; ++b) {
        if (b != w.rank) {
            peers.push_back(b);
        }
    }
    return peers;
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
    if (!error) {
        return "none";
    }
    try {
        std::rethrow_exception(error);
    }
    catch (const std::exception &e) {
        return e.what();
    }
    catch (...) {
        return "unknown";
    }
}

template <class T>
auto check_rounds(const std::vector<RoundSpec> &specs) -> void {
    const auto w = world();
    PhysicalExchange ex(w, static_cast<size_t>(team_options().threads), element_of<T>(), kTag);
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

auto every_variant(const std::vector<size_t> &peers) -> std::vector<RoundSpec> {
    std::vector<RoundSpec> specs;
    size_t round = 0;
    // Grow, shrink and grow again on the same object: staging is reused, never assumed fresh.
    for (const size_t scale : {size_t{1}, size_t{300}, size_t{1}, size_t{2}}) {
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

// --- Single-process contract --------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(sharded_exchange_world_reads_the_communicator) {
    const auto w = world();
    BOOST_TEST(w.ranks == static_cast<size_t>(monoprop::mpi::size(monoprop::mpi::Comm(MPI_COMM_WORLD))));
    BOOST_TEST(w.rank == static_cast<size_t>(monoprop::mpi::rank(monoprop::mpi::Comm(MPI_COMM_WORLD))));
    const auto self = PhysicalWorld::of(monoprop::mpi::Comm(MPI_COMM_SELF));
    BOOST_TEST(self.ranks == 1U);
    BOOST_TEST(self.rank == 0U);
    BOOST_TEST(!self.replay_pairwise);
}

BOOST_AUTO_TEST_CASE(sharded_exchange_rejects_invalid_layouts_before_posting) {
    const auto w = world();
    const size_t threads = 2;
    PhysicalExchange ex(w, threads, ExchangeElement::f64, kTag);
    // A peer set must name other ranks, ascending, each once.
    BOOST_CHECK_THROW(ex.plan_send(std::vector<size_t>{w.rank}, ExchangeTransport::pairwise), std::invalid_argument);
    BOOST_CHECK_THROW(ex.plan_send(std::vector<size_t>{w.ranks}, ExchangeTransport::pairwise), std::invalid_argument);
    if (w.ranks >= 3) {
        const auto peers = all_other_ranks(w);
        BOOST_CHECK_THROW(ex.plan_send(std::vector<size_t>{peers[1], peers[0]}, ExchangeTransport::pairwise),
                          std::invalid_argument);
        BOOST_CHECK_THROW(ex.plan_send(std::vector<size_t>{peers[0], peers[0]}, ExchangeTransport::pairwise),
                          std::invalid_argument);
    }
    // Counts must fit an MPI int, and indices must address the table (a one-rank world has no peer to address).
    ex.reset_rows(0);
    if (w.ranks >= 2) {
        BOOST_CHECK_THROW(ex.set_send_count(0, 0, 0, size_t{std::numeric_limits<int>::max()} + 1), std::length_error);
        BOOST_CHECK_THROW(ex.set_recv_count(0, 0, 0, size_t{std::numeric_limits<int>::max()} + 1), std::length_error);
    }
    else {
        BOOST_CHECK_THROW(ex.set_send_count(0, 0, 0, 1), std::out_of_range);
    }
    BOOST_CHECK_THROW(ex.set_send_count(threads, 0, 0, 1), std::out_of_range);
    BOOST_CHECK_THROW(ex.set_send_count(0, w.ranks, 0, 1), std::out_of_range);
    BOOST_CHECK_THROW(ex.set_send_count(0, 0, threads, 1), std::out_of_range);
    // The element type is fixed per exchange.
    ex.plan_send({}, ExchangeTransport::pairwise);
    BOOST_CHECK_THROW(static_cast<void>(ex.send_block<uint64_t>(0, 0, 0)), std::logic_error);
    // Nothing was posted by any of the failures.
    BOOST_TEST(ex.live() == 0);
}

// Owner-parallel planning (close_rows, plan_totals, plan_column, place_column) lays every block out exactly where
// plan_send + plan_recv do: same slice address and size for every (shard, peer, other shard), same staging sizes.
// No MPI call is made, so a world of several ranks can be described without launching them; every rank position is
// covered, sparse and dense count tables, empty rows and columns.
BOOST_AUTO_TEST_CASE(sharded_exchange_owner_parallel_planning_matches_the_primary_layout) {
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next = [&state](uint64_t bound) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state % bound;
    };
    for (const size_t ranks : {size_t{2}, size_t{3}, size_t{4}}) {
        for (const size_t threads : {size_t{1}, size_t{2}, size_t{5}}) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                for (const uint64_t density : {uint64_t{0}, uint64_t{3}, uint64_t{10}}) {
                    BOOST_TEST_CONTEXT("ranks " << ranks << " threads " << threads << " rank " << rank << " density "
                                                << density) {
                        const PhysicalWorld w{.comm = monoprop::mpi::Comm(MPI_COMM_SELF), .rank = rank, .ranks = ranks};
                        PhysicalExchange primary(w, threads, ExchangeElement::f64, kTag);
                        PhysicalExchange owners(w, threads, ExchangeElement::f64, kTag);
                        std::vector<size_t> peers;
                        for (size_t b = 0; b < ranks; ++b) {
                            if (b != rank) {
                                peers.push_back(b);
                            }
                        }
                        for (size_t t = 0; t < threads; ++t) {
                            primary.reset_rows(t);
                            owners.reset_rows(t);
                            for (size_t k = 0; k < peers.size(); ++k) {
                                for (size_t o = 0; o < threads; ++o) {
                                    // density 10: every entry nonzero; 3: about a third; 0: all empty.
                                    const size_t send = density == 0 || next(10) >= density ? 0 : 1 + next(50);
                                    const size_t recv = density == 0 || next(10) >= density ? 0 : 1 + next(50);
                                    primary.set_send_count(t, k, o, send);
                                    primary.set_recv_count(t, k, o, recv);
                                    owners.set_send_count(t, k, o, send);
                                    owners.set_recv_count(t, k, o, recv);
                                }
                            }
                            owners.close_rows(t);
                        }
                        primary.plan_send(peers, ExchangeTransport::pairwise);
                        primary.plan_recv();
                        owners.plan_totals(peers, ExchangeTransport::pairwise);
                        for (size_t t = 0; t < threads; ++t) {
                            owners.plan_column(t);
                        }
                        for (size_t t = threads; t-- > 0;) { // any order: each owner writes only its own column
                            owners.place_column(t);
                        }
                        // Offsets relative to each staging's start, since the two exchanges own different buffers.
                        const auto *send0 = primary.send_block<double>(0, 0, 0).data();
                        const auto *send1 = owners.send_block<double>(0, 0, 0).data();
                        const auto *recv0 = primary.recv_block<double>(0, 0, 0).data();
                        const auto *recv1 = owners.recv_block<double>(0, 0, 0).data();
                        size_t mismatches = 0;
                        for (size_t t = 0; t < threads; ++t) {
                            for (size_t k = 0; k < peers.size(); ++k) {
                                for (size_t o = 0; o < threads; ++o) {
                                    const auto a = primary.send_block<double>(t, k, o);
                                    const auto b = owners.send_block<double>(t, k, o);
                                    const auto c = primary.recv_block<double>(t, k, o);
                                    const auto d = owners.recv_block<double>(t, k, o);
                                    mismatches += static_cast<size_t>(
                                        a.size() != b.size() || a.data() - send0 != b.data() - send1
                                        || c.size() != d.size() || c.data() - recv0 != d.data() - recv1);
                                }
                            }
                        }
                        BOOST_TEST(mismatches == 0U);
                        BOOST_TEST(owners.live() == 0);
                    }
                }
            }
        }
    }
    // Placing a column before the round is sized is refused.
    const PhysicalWorld w{.comm = monoprop::mpi::Comm(MPI_COMM_SELF), .rank = 0, .ranks = 2};
    PhysicalExchange unplanned(w, 2, ExchangeElement::f64, kTag);
    BOOST_CHECK_THROW(unplanned.place_column(0), std::logic_error);
    BOOST_CHECK_THROW(unplanned.plan_column(2), std::out_of_range);
    BOOST_CHECK_THROW(unplanned.close_rows(2), std::out_of_range);
    // A per-peer total past INT_MAX is refused before anything is placed or posted, as by plan_send().
    PhysicalExchange big(w, 2, ExchangeElement::f64, kTag);
    for (size_t t = 0; t < 2; ++t) {
        big.reset_rows(t);
        big.set_send_count(t, 0, 0, static_cast<size_t>(std::numeric_limits<int>::max()));
        big.close_rows(t);
    }
    BOOST_CHECK_THROW(big.plan_totals(std::vector<size_t>{1}, ExchangeTransport::pairwise), std::length_error);
    BOOST_CHECK_THROW(big.place_column(0), std::logic_error);
    BOOST_TEST(big.live() == 0);
}

// A replay step's bulk path (write_symmetric_rows, the all-other-ranks plan_totals, send_column) produces the same
// tables, staging layout and slices as the entry-wise path, and send_column's view of column t is exactly the
// send_block slices (u, peer, t) of every local source u. Layouts shorter than the world fill the missing slots with
// zero counts; the owner's own rank is never a peer.
BOOST_AUTO_TEST_CASE(sharded_exchange_symmetric_rows_and_column_views_match_the_entrywise_path) {
    uint64_t state = 0xD1B54A32D192ED03ULL;
    const auto next = [&state](uint64_t bound) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state % bound;
    };
    for (const size_t ranks : {size_t{2}, size_t{3}, size_t{4}}) {
        for (const size_t threads : {size_t{1}, size_t{2}, size_t{5}}) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                for (const bool short_layouts : {false, true}) {
                    BOOST_TEST_CONTEXT("ranks " << ranks << " threads " << threads << " rank " << rank << " short "
                                                << short_layouts) {
                        const PhysicalWorld w{.comm = monoprop::mpi::Comm(MPI_COMM_SELF), .rank = rank, .ranks = ranks};
                        PhysicalExchange entrywise(w, threads, ExchangeElement::f64, kTag);
                        PhysicalExchange bulk(w, threads, ExchangeElement::f64, kTag);
                        std::vector<size_t> peers;
                        for (size_t b = 0; b < ranks; ++b) {
                            if (b != rank) {
                                peers.push_back(b);
                            }
                        }
                        for (size_t t = 0; t < threads; ++t) {
                            // One publication layout per owner, indexed by flat slot; about a third of the slots
                            // empty, the own rank's slots arbitrary (never read), and sometimes cut short.
                            const size_t slots = short_layouts ? next(ranks * threads + 1) : ranks * threads;
                            std::vector<int> counts(slots);
                            for (int &count : counts) {
                                count = next(3) == 0 ? 0 : static_cast<int>(1 + next(40));
                            }
                            entrywise.reset_rows(t);
                            for (size_t k = 0; k < peers.size(); ++k) {
                                for (size_t o = 0; o < threads; ++o) {
                                    const size_t slot = (peers[k] * threads) + o;
                                    const size_t count = slot < counts.size() ? static_cast<size_t>(counts[slot]) : 0;
                                    entrywise.set_send_count(t, k, o, count);
                                    entrywise.set_recv_count(t, k, o, count);
                                }
                            }
                            entrywise.close_rows(t);
                            bulk.write_symmetric_rows(t, counts);
                        }
                        entrywise.plan_totals(peers, ExchangeTransport::collective);
                        bulk.plan_totals(ExchangeTransport::collective);
                        BOOST_TEST(std::ranges::equal(bulk.peers(), peers));
                        for (size_t t = 0; t < threads; ++t) {
                            entrywise.plan_column(t);
                            bulk.plan_column(t);
                        }
                        for (size_t t = threads; t-- > 0;) {
                            entrywise.place_column(t);
                            bulk.place_column(t);
                        }
                        const auto *send0 = entrywise.send_block<double>(0, 0, 0).data();
                        const auto *send1 = bulk.send_block<double>(0, 0, 0).data();
                        const auto *recv0 = entrywise.recv_block<double>(0, 0, 0).data();
                        const auto *recv1 = bulk.recv_block<double>(0, 0, 0).data();
                        size_t mismatches = 0;
                        for (size_t t = 0; t < threads; ++t) {
                            for (size_t k = 0; k < peers.size(); ++k) {
                                const auto column = bulk.send_column<double>(t, k);
                                mismatches += static_cast<size_t>(column.offsets.size() != threads
                                                                  || column.counts.size() != threads);
                                for (size_t o = 0; o < threads; ++o) {
                                    mismatches += static_cast<size_t>(
                                        entrywise.send_count(t, k, o) != bulk.send_count(t, k, o)
                                        || entrywise.recv_count(t, k, o) != bulk.recv_count(t, k, o));
                                    const auto a = entrywise.send_block<double>(t, k, o);
                                    const auto b = bulk.send_block<double>(t, k, o);
                                    const auto c = entrywise.recv_block<double>(t, k, o);
                                    const auto d = bulk.recv_block<double>(t, k, o);
                                    mismatches += static_cast<size_t>(
                                        a.size() != b.size() || a.data() - send0 != b.data() - send1
                                        || c.size() != d.size() || c.data() - recv0 != d.data() - recv1);
                                    // Column t's entry for source o is send_block(o, k, t).
                                    const auto slice = bulk.send_block<double>(o, k, t);
                                    const auto block = column.block(o);
                                    mismatches += static_cast<size_t>(block.data() != slice.data()
                                                                      || block.size() != slice.size());
                                }
                            }
                        }
                        BOOST_TEST(mismatches == 0U);
                    }
                }
            }
        }
    }
    const PhysicalWorld w{.comm = monoprop::mpi::Comm(MPI_COMM_SELF), .rank = 1, .ranks = 3};
    PhysicalExchange round(w, 2, ExchangeElement::f64, kTag);
    BOOST_CHECK_THROW(round.write_symmetric_rows(2, std::vector<int>{}), std::out_of_range);
    BOOST_CHECK_THROW(round.write_symmetric_rows(0, std::vector<int>{0, 0, 0, 0, -1, 0}), std::length_error);
    // Columns are readable only once the round is planned, with the round's element type, inside the table.
    BOOST_CHECK_THROW(static_cast<void>(round.send_column<double>(0, 0)), std::logic_error);
    round.write_symmetric_rows(0, std::vector<int>{1, 1, 1, 1, 1, 1});
    round.write_symmetric_rows(1, std::vector<int>{1, 1, 1, 1, 1, 1});
    round.plan_totals(ExchangeTransport::pairwise);
    round.plan_column(0);
    round.plan_column(1);
    round.place_column(0);
    BOOST_CHECK_THROW(static_cast<void>(round.send_column<uint64_t>(0, 0)), std::logic_error);
    BOOST_CHECK_THROW(static_cast<void>(round.send_column<double>(2, 0)), std::out_of_range);
    BOOST_CHECK_THROW(static_cast<void>(round.send_column<double>(0, 2)), std::out_of_range);
    BOOST_TEST(round.send_column<double>(0, 1).block(1).size() == 1U);
    BOOST_TEST(round.live() == 0);
}

BOOST_AUTO_TEST_CASE(sharded_exchange_owner_rounds_persist_by_kind) {
    using Kind = sharded::PhysicalRounds::Kind;
    const PhysicalWorld w{.comm = monoprop::mpi::Comm(MPI_COMM_SELF), .rank = 0, .ranks = 2};
    sharded::PhysicalRounds rounds;
    // Nothing exists before first use.
    for (const auto kind : {Kind::queries, Kind::graph_answers, Kind::fused_answers, Kind::replay}) {
        BOOST_TEST(rounds.peek(kind).threads() == 0U);
    }
    PhysicalExchange &queries = rounds.get(Kind::queries, w, 2);
    // The same round on every later call with the same geometry: its staging persists.
    BOOST_TEST(&rounds.get(Kind::queries, w, 2) == &queries);
    BOOST_TEST(&rounds.peek(Kind::queries) == &queries);
    BOOST_TEST(queries.threads() == 2U);
    BOOST_TEST(rounds.peek(Kind::graph_answers).threads() == 0U); // kinds are separate rounds
    // Each kind carries its payload type.
    const std::vector<size_t> peer{1};
    for (const auto kind : {Kind::queries, Kind::graph_answers, Kind::fused_answers, Kind::replay}) {
        PhysicalExchange &round = rounds.get(kind, w, 2);
        round.reset_rows(0);
        round.reset_rows(1);
        round.set_send_count(0, 0, 0, 1);
        round.plan_send(peer, ExchangeTransport::pairwise);
        const bool u64 = kind == Kind::queries;
        const bool u32 = kind == Kind::graph_answers;
        const auto accepts = [&](auto element) {
            try {
                return round.send_block<decltype(element)>(0, 0, 0).size() == 1U;
            }
            catch (const std::logic_error &) {
                return false;
            }
        };
        BOOST_TEST(accepts(uint64_t{}) == u64);
        BOOST_TEST(accepts(uint32_t{}) == u32);
        BOOST_TEST(accepts(double{}) == (!u64 && !u32));
    }
    // A different geometry gets a fresh round.
    BOOST_TEST(rounds.get(Kind::queries, w, 3).threads() == 3U);
}

BOOST_AUTO_TEST_CASE(sharded_exchange_payload_totals_must_fit_mpi_counts) {
    // Per-peer totals are prefix sums of int counts; a sum past INT_MAX is refused by plan_send, before posting.
    const auto w = world();
    if (w.ranks < 2 || !dedicated_launch()) {
        // One process has no peer to address; the per-count bound is covered above.
        BOOST_TEST_MESSAGE("world size 1: no peer to overflow");
        return;
    }
    const size_t threads = 2;
    PhysicalExchange ex(w, threads, ExchangeElement::f64, kTag);
    ex.reset_rows(0);
    ex.reset_rows(1);
    const int big = std::numeric_limits<int>::max();
    ex.set_send_count(0, 0, 0, static_cast<size_t>(big));
    ex.set_send_count(1, 0, 0, static_cast<size_t>(big));
    BOOST_CHECK_THROW(ex.plan_send(std::vector<size_t>{all_other_ranks(w).front()}, ExchangeTransport::pairwise),
                      std::length_error);
    BOOST_TEST(ex.live() == 0);
}

BOOST_AUTO_TEST_CASE(sharded_exchange_mpi_calls_run_only_on_the_primary,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto w = world();
    const auto options = team_options();
    PhysicalExchange ex(w, static_cast<size_t>(options.threads), ExchangeElement::u64, kTag);
    // No peers: a primary post is a no-op, a worker's post is refused before it reaches MPI.
    std::vector<int> refused(static_cast<size_t>(options.threads), 0);
    const auto error = sharded::run_team(options, [&](size_t s, sharded::TeamFailure &failure) noexcept {
        if (!sharded::phase(failure, s, [&] {
                if (s == 0) {
                    ex.plan_send({}, ExchangeTransport::pairwise);
                }
            })) {
            return;
        }
        static_cast<void>(sharded::phase(failure, s, [&] {
            if (s == 1) {
                for (const auto &call : {+[](PhysicalExchange &e) { e.post_counts(); },
                                         +[](PhysicalExchange &e) { e.wait_counts(); },
                                         +[](PhysicalExchange &e) { e.post(); },
                                         +[](PhysicalExchange &e) { e.wait(); }}) {
                    try {
                        call(ex);
                    }
                    catch (const std::logic_error &) {
                        ++refused[s];
                    }
                }
            }
        }));
    });
    BOOST_TEST(describe(error) == "none");
    BOOST_TEST(refused[1] == 4);
    BOOST_TEST(ex.live() == 0);
}

// --- Multi-rank rounds --------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(sharded_exchange_moves_every_block_to_its_owner, *boost::unit_test::precondition(has_two_ranks)) {
    const auto peers = all_other_ranks(world());
    check_rounds<double>(every_variant(peers));
    check_rounds<uint64_t>(every_variant(peers));
    check_rounds<uint32_t>(every_variant(peers));
}

BOOST_AUTO_TEST_CASE(sharded_exchange_count_blocks_of_every_width_arrive,
                     *boost::unit_test::precondition(has_two_ranks)) {
    // Counts of at most 5, 300 and 100000 travel as 1-, 2- and 4-byte counts; a later light round after a heavy one
    // shrinks the block again. The receiver must read each block's width from its size.
    const auto peers = all_other_ranks(world());
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

BOOST_AUTO_TEST_CASE(sharded_exchange_sparse_peer_round, *boost::unit_test::precondition(has_even_ranks)) {
    const auto w = world();
    check_rounds<uint64_t>(every_variant({w.rank ^ 1U}));
}

BOOST_AUTO_TEST_CASE(sharded_exchange_empty_rounds_complete, *boost::unit_test::precondition(has_two_ranks)) {
    // A round whose every count is zero still completes on every rank, under both transports.
    const auto w = world();
    PhysicalExchange ex(w, static_cast<size_t>(team_options().threads), ExchangeElement::f64, kTag);
    for (const auto transport : {ExchangeTransport::pairwise, ExchangeTransport::collective}) {
        for (const bool known : {false, true}) {
            const RoundSpec spec{.peers = all_other_ranks(w),
                                 .transport = transport,
                                 .counts_known = known,
                                 .round = 0,
                                 .scale = 0};
            const auto [error, problems] = run_round<double>(ex, w, spec);
            BOOST_TEST(describe(error) == "none");
            BOOST_TEST(problems.empty());
            BOOST_TEST(ex.live() == 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_exchange_moves_keep_and_drain_requests, *boost::unit_test::precondition(has_two_ranks)) {
    // A posted round moved into a new owner keeps its requests; assigning over a live owner drains it first.
    const auto w = world();
    const auto peers = all_other_ranks(w);
    PhysicalExchange ex(w, 1, ExchangeElement::f64, kTag);
    const auto post_one = [&](PhysicalExchange &e, double value) {
        e.reset_rows(0);
        for (size_t k = 0; k < peers.size(); ++k) {
            e.set_send_count(0, k, 0, 1);
            e.set_recv_count(0, k, 0, 1);
        }
        e.plan_send(peers, ExchangeTransport::pairwise);
        for (size_t k = 0; k < peers.size(); ++k) {
            e.send_block<double>(0, k, 0)[0] = value + static_cast<double>(w.rank);
        }
        e.plan_recv();
        e.post();
    };
    post_one(ex, 10.0);
    BOOST_TEST(ex.live() > 0);
    PhysicalExchange moved(std::move(ex));
    BOOST_TEST(moved.live() > 0);
    moved.wait();
    for (size_t k = 0; k < peers.size(); ++k) {
        BOOST_TEST(moved.recv_block<double>(0, k, 0)[0] == 10.0 + static_cast<double>(peers[k]));
    }
    PhysicalExchange target(w, 1, ExchangeElement::f64, kTag + 2);
    post_one(target, 20.0);
    BOOST_TEST(target.live() > 0);
    target = std::move(moved); // drains the round target had posted
    BOOST_TEST(target.live() == 0);
#ifdef monoprop_ENABLE_MPI
    // The barrier keeps the second round's tag from meeting the next case's first round.
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}
