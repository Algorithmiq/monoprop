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
#include <chrono>
#include <exception>
#include <thread>
#include <vector>

#include "monoprop/detail/mpi/PartitionBarrier.h"

using monoprop::mpi::PartitionBarrier;
using monoprop::mpi::ShmCommPoisoned;

namespace {

// Every participant stamps its own slot with the round number; after the barrier every slot must read as
// stamped, so anyone released early sees a stale one. Tested directly because the two-level path needs
// pinning and >=2 L3 domains to engage through a real ShmComm, which no test host is guaranteed to have.
auto stamp_rounds(int n, const std::vector<int> &groups, int rounds) -> std::vector<std::exception_ptr> {
    PartitionBarrier barrier(n, groups);
    std::vector<int> slots(static_cast<size_t>(n), -1);
    std::vector<std::exception_ptr> errs(static_cast<size_t>(n));
    std::atomic<int> mismatches{0};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n));
    for (int p = 0; p < n; ++p) {
        threads.emplace_back([&, p] {
            try {
                for (int round = 0; round < rounds; ++round) {
                    slots[static_cast<size_t>(p)] = round;
                    barrier.sync(p);
                    for (int q = 0; q < n; ++q) {
                        if (slots[static_cast<size_t>(q)] != round) {
                            ++mismatches;
                        }
                    }
                    barrier.sync(p); // peers must finish reading before the next stamp overwrites
                }
            }
            catch (...) {
                errs[static_cast<size_t>(p)] = std::current_exception();
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    BOOST_CHECK_EQUAL(mismatches.load(), 0);
    return errs;
}

auto no_errors(const std::vector<std::exception_ptr> &errs) -> void {
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
}

} // namespace

// No group ids ⇒ the flat barrier, the path every unpinned run takes.
BOOST_AUTO_TEST_CASE(partition_barrier_flat_releases_together) {
    for (const int n : {1, 2, 3, 8}) {
        no_errors(stamp_rounds(n, {}, 20));
    }
}

// Two-level: same semantics, and the group sizes are deliberately uneven and not powers of two, so a
// barrier that mixed up "last in group" with "last overall" would deadlock or release early.
BOOST_AUTO_TEST_CASE(partition_barrier_grouped_releases_together) {
    no_errors(stamp_rounds(8, {0, 0, 0, 0, 1, 1, 1, 1}, 20));
    no_errors(stamp_rounds(7, {0, 0, 0, 1, 1, 2, 2}, 20));
    no_errors(stamp_rounds(6, {3, 9, 3, 9, 3, 9}, 20)); // interleaved, non-contiguous domain ids
    no_errors(stamp_rounds(5, {0, 1, 1, 1, 1}, 20));    // a domain of one
}

// One domain is no reason to pay for a root barrier: it must behave exactly like the flat path.
BOOST_AUTO_TEST_CASE(partition_barrier_single_group_is_flat) {
    no_errors(stamp_rounds(4, {2, 2, 2, 2}, 20));
}

// A group id list that does not cover every participant is a programming error upstream, not a reason
// to hang: it falls back to flat rather than indexing out of range.
BOOST_AUTO_TEST_CASE(partition_barrier_short_group_list_falls_back) {
    no_errors(stamp_rounds(4, {0, 1}, 20));
}

// poison() must release waiters in BOTH levels: a peer that throws mid-round leaves the others parked
// on their domain's word, not the root's.
BOOST_AUTO_TEST_CASE(partition_barrier_poison_releases_grouped_waiters) {
    constexpr int kN = 6;
    const std::vector<int> groups{0, 0, 0, 1, 1, 1};
    PartitionBarrier barrier(kN, groups);
    std::atomic<int> poisoned_count{0};
    std::vector<std::thread> threads;
    threads.reserve(kN - 1);
    for (int p = 0; p < kN - 1; ++p) { // participant kN-1 never arrives
        threads.emplace_back([&, p] {
            try {
                barrier.sync(p);
            }
            catch (const ShmCommPoisoned &) {
                ++poisoned_count;
            }
        });
    }
    // Let the arrivers reach the barrier before poisoning, so this exercises the release of parked
    // waiters rather than the post-barrier poison check.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    barrier.poison();
    for (auto &t : threads) {
        t.join();
    }
    // The last arriver of the complete domain parks at the root; the rest park on their domain word.
    BOOST_CHECK_EQUAL(poisoned_count.load(), kN - 1);
}

// reset() after an aborted round must leave no partial arrival count behind, in any group.
BOOST_AUTO_TEST_CASE(partition_barrier_reset_clears_partial_arrivals) {
    constexpr int kN = 4;
    const std::vector<int> groups{0, 0, 1, 1};
    PartitionBarrier barrier(kN, groups);
    std::thread lone([&] {
        try {
            barrier.sync(0); // arrives alone, then is released by the poison below
        }
        catch (const ShmCommPoisoned &) {
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    barrier.poison();
    lone.join();
    barrier.reset();

    std::atomic<int> through{0};
    std::vector<std::thread> threads;
    threads.reserve(kN);
    for (int p = 0; p < kN; ++p) {
        threads.emplace_back([&, p] {
            barrier.sync(p);
            ++through;
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    BOOST_CHECK_EQUAL(through.load(), kN);
}
