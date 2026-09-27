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

// Workers never call BOOST_TEST: each block writes only its own preallocated element (or an atomic),
// and every assertion runs on the caller after for_blocks has returned.

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

using monoprop::detail::parallel::for_blocks;
using monoprop::detail::parallel::Options;

namespace {

// The OpenMP runtime settings (ICVs) that a library helper must never change.
struct RuntimeSettings {
    int max_threads = 0;
    int dynamic = 0;
    int max_active_levels = 0;
    int thread_limit = 0;
    omp_sched_t schedule = omp_sched_static;
    int chunk = 0;
    omp_proc_bind_t proc_bind = omp_proc_bind_false;
    int level = 0;

    auto operator==(const RuntimeSettings &) const -> bool = default;
};

auto runtime_settings() -> RuntimeSettings {
    RuntimeSettings s;
    s.max_threads = omp_get_max_threads();
    s.dynamic = omp_get_dynamic();
    s.max_active_levels = omp_get_max_active_levels();
    s.thread_limit = omp_get_thread_limit();
    omp_get_schedule(&s.schedule, &s.chunk);
    s.proc_bind = omp_get_proc_bind();
    s.level = omp_get_level();
    return s;
}

// A test-side probe, not library behavior: can this runtime give a two-thread region right now?
auto runtime_offers_two_workers(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    if (omp_get_thread_limit() < 2) {
        return {false};
    }
    int team = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    boost::test_tools::assertion_result result(team >= 2);
    result.message() << "the OpenMP runtime provided " << team << " worker(s) for a two-thread request";
    return result;
}

} // namespace

BOOST_AUTO_TEST_CASE(openmp_blocks_visit_once) {
    std::vector<int> visits(19);
    for_blocks(19, {.threads = 3}, [&](size_t b) { ++visits[b]; });
    for (const auto n : visits) {
        BOOST_TEST(n == 1);
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_visit_once_across_counts_and_budgets) {
    for (const size_t count : {size_t{0}, size_t{1}, size_t{19}}) {
        for (const int threads : {1, 2, 3}) {
            std::vector<int> visits(count, 0);
            std::vector<int> worker(count, -1);
            for_blocks(count, {.threads = threads}, [&](size_t b) {
                ++visits[b];
                worker[b] = omp_get_thread_num();
            });
            const auto requested = std::min<size_t>(static_cast<size_t>(threads), count);
            for (size_t b = 0; b < count; ++b) {
                BOOST_TEST(visits[b] == 1, "count=" << count << " threads=" << threads << " block=" << b);
                BOOST_TEST(worker[b] >= 0);
                BOOST_TEST(static_cast<size_t>(worker[b]) < requested);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_more_requested_workers_than_blocks) {
    constexpr size_t count = 2;
    std::vector<int> visits(count, 0);
    std::vector<int> team(count, 0);
    std::vector<int> worker(count, -1);
    for_blocks(count, {.threads = 8}, [&](size_t b) {
        ++visits[b];
        team[b] = omp_get_num_threads();
        worker[b] = omp_get_thread_num();
    });
    for (size_t b = 0; b < count; ++b) {
        BOOST_TEST(visits[b] == 1);
        // The region requests min(threads, count) workers, so no idle third worker is created.
        BOOST_TEST(team[b] >= 1);
        BOOST_TEST(team[b] <= static_cast<int>(count));
        BOOST_TEST(worker[b] < static_cast<int>(count));
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_budget_one_runs_ascending_on_the_caller) {
    constexpr size_t count = 19;
    const auto caller = std::this_thread::get_id();
    std::vector<size_t> order;
    bool on_caller = true;
    int level = -1;
    for_blocks(count, {.threads = 1}, [&](size_t b) {
        order.push_back(b);
        on_caller = on_caller && std::this_thread::get_id() == caller;
        level = omp_get_level();
    });
    BOOST_REQUIRE_EQUAL(order.size(), count);
    for (size_t b = 0; b < count; ++b) {
        BOOST_TEST(order[b] == b);
    }
    BOOST_TEST(on_caller);
    BOOST_TEST(level == 0); // no OpenMP region was opened
}

// The serial path runs the body directly, so its first exception ends the loop at once.
BOOST_AUTO_TEST_CASE(openmp_blocks_budget_one_propagates_at_the_failing_block) {
    std::vector<size_t> order;
    BOOST_CHECK_EXCEPTION(for_blocks(19,
                                     {.threads = 1},
                                     [&](size_t b) {
                                         order.push_back(b);
                                         if (b == 5) {
                                             throw std::runtime_error("block 5");
                                         }
                                     }),
                          std::runtime_error,
                          [](const std::runtime_error &e) { return std::string(e.what()) == "block 5"; });
    const std::vector<size_t> expected{0, 1, 2, 3, 4, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(order.begin(), order.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(openmp_blocks_nested_invocation_runs_serially_in_order) {
    constexpr size_t outer = 4;
    constexpr size_t inner = 5;
    std::vector<int> visits(outer * inner, 0);
    std::vector<std::vector<size_t>> order(outer);
    std::vector<int> outer_level(outer, -1);
    std::vector<int> inner_level(outer * inner, -1);
    for_blocks(outer, {.threads = 2}, [&](size_t o) {
        outer_level[o] = omp_get_level();
        for_blocks(inner, {.threads = 3}, [&](size_t i) {
            ++visits[o * inner + i];
            order[o].push_back(i); // one writer: the inner loop runs on outer block o's worker
            inner_level[o * inner + i] = omp_get_level();
        });
    });
    for (size_t o = 0; o < outer; ++o) {
        BOOST_REQUIRE_EQUAL(order[o].size(), inner);
        for (size_t i = 0; i < inner; ++i) {
            BOOST_TEST(visits[o * inner + i] == 1);
            BOOST_TEST(order[o][i] == i);
            // No nested region: the inner body runs at the outer body's nesting level.
            BOOST_TEST(inner_level[o * inner + i] == outer_level[o]);
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_nested_invocation_propagates_at_the_failing_block) {
    std::vector<std::vector<size_t>> order(2);
    BOOST_CHECK_THROW(for_blocks(2,
                                 {.threads = 2},
                                 [&](size_t o) {
                                     for_blocks(6, {.threads = 3}, [&](size_t i) {
                                         order[o].push_back(i);
                                         if (i == 2) {
                                             throw std::runtime_error("inner");
                                         }
                                     });
                                 }),
                      std::runtime_error);
    // Each outer block that ran stopped its serial inner loop at the failing block.
    for (const auto &seen : order) {
        BOOST_TEST(seen.size() <= 3U);
        for (size_t i = 0; i < seen.size(); ++i) {
            BOOST_TEST(seen[i] == i);
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_invalid_budget_is_rejected_before_work) {
    for (const int threads : {0, -1, INT_MIN}) {
        for (const size_t count : {size_t{0}, size_t{5}}) {
            int calls = 0;
            BOOST_CHECK_THROW(for_blocks(count, {.threads = threads}, [&](size_t) { ++calls; }), std::invalid_argument);
            BOOST_TEST(calls == 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_unrepresentable_count_is_rejected_before_work) {
    const auto first_unrepresentable = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) + 1U;
    for (const size_t count : {first_unrepresentable, std::numeric_limits<size_t>::max()}) {
        for (const int threads : {1, 3}) {
            int calls = 0;
            BOOST_CHECK_THROW(for_blocks(count, {.threads = threads}, [&](size_t) { ++calls; }), std::length_error);
            BOOST_TEST(calls == 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_blocks_leave_runtime_settings_unchanged) {
    const auto before = runtime_settings();
    std::vector<int> sink(64, 0);
    for_blocks(sink.size(), {.threads = 1}, [&](size_t b) { sink[b] = 1; });
    for_blocks(sink.size(), {.threads = 3}, [&](size_t b) { sink[b] = 2; });
    for_blocks(sink.size(), {.threads = 1000}, [&](size_t b) { sink[b] = 3; });
    for_blocks(4, {.threads = 2}, [&](size_t o) {
        for_blocks(16, {.threads = 4}, [&](size_t i) { sink[o * 16 + i] = 4; });
    });
    BOOST_CHECK_THROW(for_blocks(8, {.threads = 3}, [](size_t) { throw std::runtime_error("fail"); }),
                      std::runtime_error);
    BOOST_CHECK_THROW(for_blocks(8, {.threads = 0}, [](size_t) {}), std::invalid_argument);
    const auto after = runtime_settings();
    BOOST_TEST((before == after));
    BOOST_TEST(after.max_threads == before.max_threads);
    BOOST_TEST(after.dynamic == before.dynamic);
    BOOST_TEST(after.max_active_levels == before.max_active_levels);
    BOOST_TEST(std::count(sink.begin(), sink.end(), 4) == 64);
}

// A worker failure is caught inside the region, every worker joins, and only then does the caller
// see the exception. After a failure no particular subset of blocks is required to have run.
BOOST_AUTO_TEST_CASE(openmp_blocks_worker_exception_is_rethrown_after_join) {
    constexpr size_t count = 19;
    std::atomic<int> in_flight{0};
    std::vector<char> finished(count, 0);
    // Block 7 opens the second worker's static chunk (19 blocks over 3 workers: 7/6/6), so the
    // exception is raised while the other workers are still sleeping inside their bodies.
    constexpr size_t failing = 7;
    int in_flight_at_catch = -1;
    try {
        for_blocks(count, {.threads = 3}, [&](size_t b) {
            in_flight.fetch_add(1);
            if (b == failing) {
                in_flight.fetch_sub(1);
                throw std::runtime_error("block 7");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            finished[b] = 1;
            in_flight.fetch_sub(1);
        });
        BOOST_FAIL("for_blocks returned normally after a worker threw");
    }
    catch (const std::runtime_error &e) {
        in_flight_at_catch = in_flight.load();
        BOOST_TEST(std::string(e.what()) == "block 7");
    }
    BOOST_TEST(in_flight_at_catch == 0);
    BOOST_TEST(finished[failing] == 0);
}

// Every block throws; worker 0 always receives block 0 first under a static schedule, whatever the
// actual team size, so the first populated slot is worker 0's and it holds block 0's exception.
BOOST_AUTO_TEST_CASE(openmp_blocks_first_populated_worker_slot_is_rethrown) {
    for (const int threads : {2, 3}) {
        BOOST_CHECK_EXCEPTION(
            for_blocks(19, {.threads = threads}, [](size_t b) { throw std::runtime_error(std::to_string(b)); }),
            std::runtime_error,
            [](const std::runtime_error &e) { return std::string(e.what()) == "0"; });
    }
}

// Test-only worker observation: each block records its actual OpenMP worker ID. A request for four
// workers is not evidence of participation; the recorded IDs are.
BOOST_AUTO_TEST_CASE(openmp_blocks_multiple_workers_participate,
                     *boost::unit_test::precondition(runtime_offers_two_workers)) {
    constexpr size_t count = 64;
    constexpr int threads = 4;
    std::vector<int> worker(count, -1);
    std::vector<int> team(count, 0);
    for_blocks(count, {.threads = threads}, [&](size_t b) {
        worker[b] = omp_get_thread_num();
        team[b] = omp_get_num_threads();
    });
    const auto actual = *std::max_element(team.begin(), team.end());
    const std::set<int> distinct(worker.begin(), worker.end());
    BOOST_TEST(actual >= 1);
    BOOST_TEST(actual <= threads);
    // A static schedule with count >= team hands every actual worker at least one block.
    BOOST_TEST(distinct.size() == static_cast<size_t>(actual));
    BOOST_TEST(*distinct.begin() == 0);
    BOOST_TEST(*distinct.rbegin() == actual - 1);
    BOOST_WARN_MESSAGE(
        actual >= 2,
        "multi-worker participation NOT demonstrated: the runtime gave this region " << actual << " worker(s)");
    BOOST_TEST_MESSAGE("openmp_blocks_multiple_workers_participate: observed "
                       << distinct.size() << " distinct workers of " << threads << " requested");
}
