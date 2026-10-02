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

/*
 * The fixed-team phase primitive (detail/sharded/Team.h). The team size T is the captured launch budget
 * (monoprop_NUM_THREADS, else omp_get_max_threads()); cpp/tests/CMakeLists.txt reruns these cases in fresh
 * processes at T = 1, 2 and 4. Cases that need a nonprimary worker skip below T = 2; the T = 2 and T = 4
 * launches run them.
 *
 * Workers never call BOOST_TEST. Each writes only its own preallocated observation slot (or an atomic, so a
 * mutated helper that repeats an owner cannot race), and every assertion runs on the caller after run_team()
 * has joined the team. Observing the actual team is test-side evidence, not a library check.
 *
 * The sleeps only shape interleavings that make a broken checkpoint visible: waiting workers park in the
 * barrier while a delayed worker is still inside its phase. Correctness never depends on them.
 */

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/Team.h"

namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;
using namespace std::chrono_literals;

namespace {

// Orchestration bodies must be noexcept; potentially throwing work belongs inside phase().
static_assert(sharded::TeamBody<decltype([](size_t, sharded::TeamFailure &) noexcept {})>);
static_assert(!sharded::TeamBody<decltype([](size_t, sharded::TeamFailure &) {})>);
static_assert(!sharded::TeamBody<decltype([](size_t) noexcept {})>);

auto team_options() -> parallel::Options {
    return parallel::capture_thread_budget();
}

auto team_size(parallel::Options options) -> size_t {
    return static_cast<size_t>(options.threads);
}

// The team a fixed-T launch in cpp/tests/CMakeLists.txt promises; absent in ordinary discovered runs.
auto expected_team() -> std::optional<int> {
    const auto *value = std::getenv("monoprop_TEST_EXPECT_TEAM");
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::stoi(value);
}

auto has_nonprimary_worker(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const auto threads = parallel::capture_thread_budget().threads;
    boost::test_tools::assertion_result result(threads >= 2);
    result.message() << "the captured team budget is " << threads << "; this case needs a nonprimary worker";
    return result;
}

// What one worker saw of its own team, written only by that worker.
struct alignas(64) Observation {
    std::atomic<int> calls{0}; //!< Orchestration-body invocations for this owner.
    int worker = -1;           //!< omp_get_thread_num() inside the body.
    int team = 0;              //!< omp_get_num_threads() inside the body.
    int level = -1;            //!< omp_get_level() inside the body.
    int active_level = -1;     //!< omp_get_active_level() inside the body.
    std::thread::id thread;    //!< The OS thread running the body.
};

auto observe(Observation &slot) -> void {
    slot.calls.fetch_add(1);
    slot.worker = omp_get_thread_num();
    slot.team = omp_get_num_threads();
    slot.level = omp_get_level();
    slot.active_level = omp_get_active_level();
    slot.thread = std::this_thread::get_id();
}

/*
 * Test-only orchestration for multi-phase failure cases: run `count` phases, `body(p)` in phase p, and record each
 * decision in `decisions` until the first false one. After a false decision the worker runs no further phase body,
 * but still reaches the remaining checkpoints. With the correct helper every worker stops at the same checkpoint, so
 * those extra checkpoints are uniform and change nothing. With a broken checkpoint, workers that disagree still
 * reach the same number of barriers, so the divergence is recorded and asserted after the join instead of
 * deadlocking the team.
 */
template <class Body>
auto run_phases(sharded::TeamFailure &failure, size_t shard, size_t count, std::vector<int> &decisions, Body &&body)
    -> void {
    auto proceeding = true;
    for (size_t p = 0; p < count; ++p) {
        if (!proceeding) {
            failure.checkpoint();
            continue;
        }
        proceeding = sharded::phase(failure, shard, [&] { body(p); });
        decisions.push_back(proceeding ? 1 : 0);
    }
}

// A non-std exception type that also identifies the thrower.
struct WorkerError {
    size_t worker;
};

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
    auto settings = RuntimeSettings{};
    settings.max_threads = omp_get_max_threads();
    settings.dynamic = omp_get_dynamic();
    settings.max_active_levels = omp_get_max_active_levels();
    settings.thread_limit = omp_get_thread_limit();
    omp_get_schedule(&settings.schedule, &settings.chunk);
    settings.proc_bind = omp_get_proc_bind();
    settings.level = omp_get_level();
    return settings;
}

} // namespace

BOOST_AUTO_TEST_CASE(sharded_team_visits_each_owner) {
    const auto options = team_options();
    const auto threads = team_size(options);
    auto visits = std::vector<std::atomic<int>>(threads);
    auto observations = std::vector<Observation>(threads);
    auto decisions = std::vector<int>(threads, -1);
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        observe(observations[shard]);
        decisions[shard] = sharded::phase(failure, shard, [&] { visits[shard].fetch_add(1); }) ? 1 : 0;
    });
    BOOST_TEST(!error);
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST(visits[shard].load() == 1, "shard " << shard);
        BOOST_TEST(observations[shard].calls.load() == 1, "shard " << shard);
        BOOST_TEST(observations[shard].team == options.threads, "shard " << shard);
        BOOST_TEST(decisions[shard] == 1, "shard " << shard);
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_observes_worker_identity) {
    const auto options = team_options();
    const auto threads = team_size(options);
    const auto caller = std::this_thread::get_id();
    const auto caller_level = omp_get_level();
    auto in_body = std::vector<Observation>(threads);
    auto in_phase = std::vector<Observation>(threads);
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        observe(in_body[shard]);
        sharded::phase(failure, shard, [&] { observe(in_phase[shard]); });
    });
    BOOST_TEST(!error);
    auto distinct = std::set<std::thread::id>{};
    for (size_t shard = 0; shard < threads; ++shard) {
        for (const auto *slot : {&in_body[shard], &in_phase[shard]}) {
            BOOST_TEST(slot->calls.load() == 1, "shard " << shard);
            BOOST_TEST(slot->worker == static_cast<int>(shard), "shard " << shard);
            BOOST_TEST(slot->team == options.threads, "shard " << shard);
            // One structured region even at T = 1, where the inactive region leaves omp_in_parallel() false.
            BOOST_TEST(slot->level == caller_level + 1, "shard " << shard);
            BOOST_TEST(slot->active_level == (threads > 1 ? 1 : 0), "shard " << shard);
            BOOST_TEST((slot->thread == in_body[shard].thread), "shard " << shard);
        }
        distinct.insert(in_body[shard].thread);
    }
    BOOST_TEST(distinct.size() == threads);
    BOOST_TEST((in_body[0].thread == caller)); // the entering caller is the OpenMP primary thread
    BOOST_TEST_MESSAGE("sharded_team_observes_worker_identity: " << distinct.size()
                                                                 << " distinct threads for T=" << threads);
}

BOOST_AUTO_TEST_CASE(sharded_team_keeps_owner_across_phases) {
    constexpr size_t phases = 6;
    const auto options = team_options();
    const auto threads = team_size(options);
    auto workers = std::vector<std::vector<int>>(threads, std::vector<int>(phases, -1));
    auto identities = std::vector<std::vector<std::thread::id>>(threads, std::vector<std::thread::id>(phases));
    auto decisions = std::vector<std::vector<int>>(threads, std::vector<int>(phases, -1));
    auto regions = std::vector<std::vector<int>>(threads, std::vector<int>(phases, -1));
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        for (size_t p = 0; p < phases; ++p) {
            const auto proceed = sharded::phase(failure, shard, [&] {
                workers[shard][p] = omp_get_thread_num();
                identities[shard][p] = std::this_thread::get_id();
                // Distinguishes one team from a region per phase.
                regions[shard][p] = omp_get_level();
            });
            decisions[shard][p] = proceed ? 1 : 0;
            if (!proceed) {
                return;
            }
        }
    });
    BOOST_TEST(!error);
    for (size_t shard = 0; shard < threads; ++shard) {
        for (size_t p = 0; p < phases; ++p) {
            BOOST_TEST(workers[shard][p] == static_cast<int>(shard), "shard " << shard << " phase " << p);
            BOOST_TEST((identities[shard][p] == identities[shard][0]), "shard " << shard << " phase " << p);
            BOOST_TEST(decisions[shard][p] == 1, "shard " << shard << " phase " << p);
            BOOST_TEST(regions[shard][p] == regions[0][0], "shard " << shard << " phase " << p);
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_primary_only_phase_runs_on_the_caller) {
    const auto options = team_options();
    const auto threads = team_size(options);
    const auto caller = std::this_thread::get_id();
    auto primary_runs = std::atomic<int>{0};
    auto primary_thread = std::thread::id{};
    auto published = 0;
    auto seen = std::vector<int>(threads, -1);
    auto decisions = std::vector<std::vector<int>>(threads);
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        auto proceed = sharded::phase(failure, shard, [&] {
            if (shard == 0) {
                // The other owners have no work here but still wait for this write.
                std::this_thread::sleep_for(10ms);
                primary_runs.fetch_add(1);
                primary_thread = std::this_thread::get_id();
                published = 42;
            }
        });
        decisions[shard].push_back(proceed ? 1 : 0);
        if (!proceed) {
            return;
        }
        proceed = sharded::phase(failure, shard, [&] { seen[shard] = published; });
        decisions[shard].push_back(proceed ? 1 : 0);
    });
    BOOST_TEST(!error);
    BOOST_TEST(primary_runs.load() == 1);
    BOOST_TEST((primary_thread == caller));
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST(seen[shard] == 42, "shard " << shard);
        BOOST_TEST((decisions[shard] == std::vector<int>{1, 1}), "shard " << shard);
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_empty_owners_reach_the_checkpoint) {
    const auto options = team_options();
    const auto threads = team_size(options);
    const auto last = threads - 1;
    auto value = 0;
    auto seen = std::vector<int>(threads, -1);
    auto empty_arrivals = std::vector<std::atomic<int>>(threads);
    auto decisions = std::vector<std::vector<int>>(threads);
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        // Only the last owner has work; every other owner's phase body is empty.
        auto proceed = sharded::phase(failure, shard, [&] {
            if (shard == last) {
                std::this_thread::sleep_for(10ms);
                value = 7;
            }
        });
        if (shard != last) {
            empty_arrivals[shard].fetch_add(1);
        }
        decisions[shard].push_back(proceed ? 1 : 0);
        if (!proceed) {
            return;
        }
        proceed = sharded::phase(failure, shard, [&] { seen[shard] = value; });
        decisions[shard].push_back(proceed ? 1 : 0);
    });
    BOOST_TEST(!error);
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST(seen[shard] == 7, "shard " << shard);
        BOOST_TEST((decisions[shard] == std::vector<int>{1, 1}), "shard " << shard);
        BOOST_TEST(empty_arrivals[shard].load() == (shard == last ? 0 : 1), "shard " << shard);
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_publishes_writes_to_the_next_phase) {
    constexpr size_t block = 4096;
    constexpr size_t rounds = 3;
    const auto options = team_options();
    const auto threads = team_size(options);
    auto buffers = std::vector<std::vector<size_t>>(threads, std::vector<size_t>(block, 0));
    auto sums = std::vector<std::vector<size_t>>(threads, std::vector<size_t>(rounds, 0));
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        for (size_t round = 0; round < rounds; ++round) {
            // Producers finish at staggered times; a consumer must still see its peer's whole block.
            if (!sharded::phase(failure, shard, [&] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2 * (shard + 1)));
                    for (size_t i = 0; i < block; ++i) {
                        buffers[shard][i] = (round + 1) * 1000003 + shard * block + i;
                    }
                })) {
                return;
            }
            const auto peer = (shard + 1) % threads;
            if (!sharded::phase(failure, shard, [&] {
                    sums[shard][round] = std::accumulate(buffers[peer].begin(), buffers[peer].end(), size_t{0});
                })) {
                return;
            }
        }
    });
    BOOST_TEST(!error);
    for (size_t shard = 0; shard < threads; ++shard) {
        const auto peer = (shard + 1) % threads;
        for (size_t round = 0; round < rounds; ++round) {
            auto expected = size_t{0};
            for (size_t i = 0; i < block; ++i) {
                expected += (round + 1) * 1000003 + peer * block + i;
            }
            BOOST_TEST(sums[shard][round] == expected, "shard " << shard << " round " << round);
        }
    }
}

// Explicit T = 1, independent of the launch budget: the failure is returned after the (inactive) region
// ends, never rethrown by run_team().
BOOST_AUTO_TEST_CASE(sharded_team_single_worker_failure_is_returned_after_join) {
    const auto caller = std::this_thread::get_id();
    const auto caller_level = omp_get_level();
    auto observation = Observation{};
    auto later_phase = 0;
    auto finished = 0;
    auto decision = -1;
    auto error = std::exception_ptr{};
    try {
        error = sharded::run_team(
            parallel::Options{.threads = 1},
            [&](size_t shard, sharded::TeamFailure &failure) noexcept {
                observe(observation);
                const auto proceed = sharded::phase(failure, shard, [] { throw std::runtime_error("single worker"); });
                decision = proceed ? 1 : 0;
                if (proceed) {
                    sharded::phase(failure, shard, [&] { ++later_phase; });
                }
                finished = 1;
            });
    }
    catch (...) {
        BOOST_FAIL("run_team rethrew a phase exception instead of returning it");
    }
    BOOST_TEST(observation.calls.load() == 1);
    BOOST_TEST(observation.worker == 0);
    BOOST_TEST(observation.team == 1);
    BOOST_TEST(observation.level == caller_level + 1);
    BOOST_TEST((observation.thread == caller));
    BOOST_TEST(decision == 0);
    BOOST_TEST(later_phase == 0);
    BOOST_TEST(finished == 1);
    BOOST_REQUIRE(error);
    BOOST_CHECK_EXCEPTION(std::rethrow_exception(error), std::runtime_error, [](const std::runtime_error &e) {
        return std::string(e.what()) == "single worker";
    });
}

BOOST_AUTO_TEST_CASE(sharded_team_failure_is_returned_not_rethrown) {
    const auto options = team_options();
    const auto threads = team_size(options);
    const auto failing = threads - 1; // the primary at T = 1
    auto finished = std::atomic<int>{0};
    auto error = std::exception_ptr{};
    try {
        error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
            sharded::phase(failure, shard, [&] {
                if (shard == failing) {
                    throw std::runtime_error("returned");
                }
            });
            finished.fetch_add(1);
        });
    }
    catch (...) {
        BOOST_FAIL("run_team rethrew a phase exception instead of returning it");
    }
    BOOST_TEST(finished.load() == options.threads);
    BOOST_REQUIRE(error);
    BOOST_CHECK_EXCEPTION(std::rethrow_exception(error), std::runtime_error, [](const std::runtime_error &e) {
        return std::string(e.what()) == "returned";
    });
}

BOOST_AUTO_TEST_CASE(sharded_team_nonprimary_failure_joins_and_skips_later_phases,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    constexpr size_t phases = 4;
    constexpr size_t failing_phase = 1;
    constexpr int iterations = 10;
    const auto options = team_options();
    const auto threads = team_size(options);
    const auto failing = threads - 1;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto executed = std::vector<std::vector<int>>(threads, std::vector<int>(phases, 0));
        auto decisions = std::vector<std::vector<int>>(threads);
        auto in_flight = std::atomic<int>{0};
        auto finished = std::atomic<int>{0};
        const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
            in_flight.fetch_add(1);
            run_phases(failure, shard, phases, decisions[shard], [&](size_t p) {
                if (p == failing_phase && shard == failing) {
                    // Every peer is already waiting in the checkpoint when this worker fails.
                    std::this_thread::sleep_for(5ms);
                    throw std::runtime_error("worker " + std::to_string(shard));
                }
                executed[shard][p] = 1;
            });
            finished.fetch_add(1);
            in_flight.fetch_sub(1);
        });
        const auto joined = in_flight.load();
        BOOST_TEST(joined == 0, "iteration " << iteration);
        BOOST_TEST(finished.load() == options.threads, "iteration " << iteration);
        for (size_t shard = 0; shard < threads; ++shard) {
            BOOST_TEST((decisions[shard] == std::vector<int>{1, 0}), "iteration " << iteration << " shard " << shard);
            BOOST_TEST(executed[shard][0] == 1, "iteration " << iteration << " shard " << shard);
            BOOST_TEST(executed[shard][failing_phase] == (shard == failing ? 0 : 1),
                       "iteration " << iteration << " shard " << shard);
            for (size_t p = failing_phase + 1; p < phases; ++p) {
                BOOST_TEST(executed[shard][p] == 0, "iteration " << iteration << " shard " << shard << " phase " << p);
            }
        }
        BOOST_REQUIRE(error);
        BOOST_CHECK_EXCEPTION(std::rethrow_exception(error), std::runtime_error, [&](const std::runtime_error &e) {
            return std::string(e.what()) == "worker " + std::to_string(failing);
        });
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_simultaneous_failures_select_the_lowest_worker,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    constexpr int iterations = 5;
    const auto options = team_options();
    const auto threads = team_size(options);
    // Higher workers fail first in time, so selection cannot follow arrival order.
    for (const size_t lowest_failing : {size_t{0}, size_t{1}}) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            auto later = std::atomic<int>{0};
            auto decisions = std::vector<std::vector<int>>(threads);
            const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
                run_phases(failure, shard, 2, decisions[shard], [&](size_t p) {
                    if (p == 1) {
                        later.fetch_add(1);
                    }
                    else if (shard >= lowest_failing) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2 * (threads - shard)));
                        throw WorkerError{shard};
                    }
                });
            });
            BOOST_TEST(later.load() == 0);
            for (size_t shard = 0; shard < threads; ++shard) {
                BOOST_TEST((decisions[shard] == std::vector<int>{0}), "iteration " << iteration << " shard " << shard);
            }
            BOOST_REQUIRE(error);
            try {
                std::rethrow_exception(error);
            }
            catch (const WorkerError &e) {
                BOOST_TEST(e.worker == lowest_failing, "iteration " << iteration);
            }
            catch (...) {
                BOOST_FAIL("the selected exception changed type");
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_failure_selects_the_lowest_recorded_slot) {
    auto failure = sharded::TeamFailure(4);
    BOOST_TEST(!failure.first_error());
    // Outside a parallel region the checkpoint binds to a one-thread team.
    BOOST_TEST(failure.checkpoint());
    const auto third = std::make_exception_ptr(WorkerError{3});
    const auto second = std::make_exception_ptr(WorkerError{2});
    const auto later = std::make_exception_ptr(WorkerError{99});
    failure.record(3, third);
    failure.record(2, second);
    failure.record(2, later); // a slot keeps its first error
    failure.record(1, std::exception_ptr{});
    BOOST_TEST((failure.first_error() == second));
    BOOST_TEST(!failure.checkpoint());
}

BOOST_AUTO_TEST_CASE(sharded_team_preserves_the_original_exception) {
    struct PayloadError : std::runtime_error {
        explicit PayloadError(int value) : std::runtime_error("payload"), payload(value) {}
        int payload;
    };
    struct Opaque {
        int code;
        std::string text;
    };
    const auto options = team_options();
    const auto failing = team_size(options) - 1; // a nonprimary worker when T > 1
    auto fail_with = [&](auto thrower) {
        return sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
            sharded::phase(failure, shard, [&] {
                if (shard == failing) {
                    thrower();
                }
            });
        });
    };

    const auto payload = fail_with([] { throw PayloadError(17); });
    BOOST_REQUIRE(payload);
    try {
        std::rethrow_exception(payload);
    }
    catch (const PayloadError &e) {
        BOOST_TEST(e.payload == 17);
        BOOST_TEST(std::string(e.what()) == "payload");
    }
    catch (...) {
        BOOST_FAIL("PayloadError changed type");
    }

    const auto opaque = fail_with([] { throw Opaque{5, "not a std::exception"}; });
    BOOST_REQUIRE(opaque);
    try {
        std::rethrow_exception(opaque);
    }
    catch (const Opaque &e) {
        BOOST_TEST(e.code == 5);
        BOOST_TEST(e.text == "not a std::exception");
    }
    catch (...) {
        BOOST_FAIL("Opaque changed type");
    }

    const auto integer = fail_with([] { throw 23; });
    BOOST_REQUIRE(integer);
    try {
        std::rethrow_exception(integer);
    }
    catch (const int value) {
        BOOST_TEST(value == 23);
    }
    catch (...) {
        BOOST_FAIL("int changed type");
    }
}

// Checkpoint N's decision must stay stable while slow workers consume it: a fast worker that passes
// checkpoint N and fails in phase N+1 must not turn N into a failure for the others.
BOOST_AUTO_TEST_CASE(sharded_team_checkpoint_generations_are_stable,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    constexpr int iterations = 20;
    const auto options = team_options();
    const auto threads = team_size(options);
    for (const size_t fast : {size_t{0}, threads - 1}) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            auto decisions = std::vector<std::vector<int>>(threads);
            auto executed_second = std::vector<int>(threads, 0);
            auto executed_third = std::vector<int>(threads, 0);
            const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
                run_phases(failure, shard, 3, decisions[shard], [&](size_t p) {
                    if (p == 0 && shard == fast) {
                        // The slow workers park in checkpoint N before the fast worker arrives last.
                        std::this_thread::sleep_for(5ms);
                    }
                    else if (p == 1) {
                        if (shard == fast) {
                            throw WorkerError{shard}; // at once, while the others may still be leaving checkpoint N
                        }
                        executed_second[shard] = 1;
                    }
                    else if (p == 2) {
                        executed_third[shard] = 1;
                    }
                });
            });
            for (size_t shard = 0; shard < threads; ++shard) {
                const auto context = [&] {
                    return "fast " + std::to_string(fast) + " iteration " + std::to_string(iteration) + " shard "
                           + std::to_string(shard);
                };
                BOOST_TEST((decisions[shard] == std::vector<int>{1, 0}), context());
                BOOST_TEST(executed_second[shard] == (shard == fast ? 0 : 1), context());
                BOOST_TEST(executed_third[shard] == 0, context());
            }
            BOOST_REQUIRE(error);
            try {
                std::rethrow_exception(error);
            }
            catch (const WorkerError &e) {
                BOOST_TEST(e.worker == fast);
            }
            catch (...) {
                BOOST_FAIL("the selected exception changed type");
            }
        }
    }
}

/*
 * The same property with the interleaving forced: every other worker reads checkpoint 0's decision only after the
 * fast worker has left it and recorded its failure in phase 1. A decision that is not generation-stamped (any recorded
 * failure fails every checkpoint still being read) would fail checkpoint 0 for the slow workers.
 */
BOOST_AUTO_TEST_CASE(sharded_team_checkpoint_ignores_a_later_generation_failure,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const auto threads = team_size(options);
    for (const size_t fast : {size_t{0}, threads - 1}) {
        auto decisions = std::vector<std::vector<int>>(threads);
        auto fast_recorded = std::atomic<bool>(false);
        const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
            const auto hold_until_recorded = [&](size_t generation) noexcept {
                if (generation == 0 && shard != fast) {
                    while (!fast_recorded.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                }
            };
            auto proceeding = true;
            for (size_t p = 0; p < 3; ++p) {
                if (proceeding) {
                    try {
                        if (p == 1 && shard == fast) {
                            failure.record(shard, std::make_exception_ptr(WorkerError{shard}));
                            fast_recorded.store(true, std::memory_order_release);
                            throw WorkerError{shard};
                        }
                    }
                    catch (...) {
                        failure.record(shard, std::current_exception());
                    }
                }
                const auto decision = failure.checkpoint(hold_until_recorded);
                if (proceeding) {
                    decisions[shard].push_back(decision ? 1 : 0);
                }
                proceeding = proceeding && decision;
            }
        });
        for (size_t shard = 0; shard < threads; ++shard) {
            BOOST_TEST((decisions[shard] == std::vector<int>{1, 0}),
                       "fast " + std::to_string(fast) + " shard " + std::to_string(shard));
        }
        BOOST_REQUIRE(error);
        try {
            std::rethrow_exception(error);
        }
        catch (const WorkerError &e) {
            BOOST_TEST(e.worker == fast);
        }
        catch (...) {
            BOOST_FAIL("the selected exception changed type");
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_fresh_invocation_has_no_stale_failure) {
    const auto options = team_options();
    const auto threads = team_size(options);
    const auto failed = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        sharded::phase(failure, shard, [&] {
            if (shard == threads - 1) {
                throw std::runtime_error("first invocation");
            }
        });
    });
    BOOST_REQUIRE(failed);

    auto executed = std::vector<std::vector<int>>(threads, std::vector<int>(3, 0));
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        for (size_t p = 0; p < 3; ++p) {
            if (!sharded::phase(failure, shard, [&] { executed[shard][p] = 1; })) {
                return;
            }
        }
    });
    BOOST_TEST(!error);
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST((executed[shard] == std::vector<int>{1, 1, 1}), "shard " << shard);
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_rejects_a_nonpositive_budget_before_the_region) {
    for (const int threads : {0, -1}) {
        auto calls = std::atomic<int>{0};
        BOOST_CHECK_THROW(sharded::run_team(parallel::Options{.threads = threads},
                                            [&](size_t, sharded::TeamFailure &) noexcept { calls.fetch_add(1); }),
                          std::invalid_argument);
        BOOST_TEST(calls.load() == 0);
    }
}

BOOST_AUTO_TEST_CASE(sharded_team_leaves_runtime_settings_unchanged) {
    const auto options = team_options();
    const auto before = runtime_settings();
    auto sink = std::vector<int>(team_size(options), 0);
    const auto ok = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        sharded::phase(failure, shard, [&] { sink[shard] = 1; });
    });
    const auto failed = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        sharded::phase(failure, shard, [] { throw std::runtime_error("fail"); });
    });
    const auto single =
        sharded::run_team(parallel::Options{.threads = 1}, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
            sharded::phase(failure, shard, [&] { sink[shard] = 2; });
        });
    const auto after = runtime_settings();
    BOOST_TEST(!ok);
    BOOST_TEST(static_cast<bool>(failed));
    BOOST_TEST(!single);
    BOOST_TEST((before == after));
    BOOST_TEST(after.max_threads == before.max_threads);
    BOOST_TEST(after.dynamic == before.dynamic);
    BOOST_TEST(after.thread_limit == before.thread_limit);
    BOOST_TEST(after.level == before.level);
}

// Only meaningful under the fixed-T launches, which set monoprop_TEST_EXPECT_TEAM; it checks that the
// launch really supplied the team the other cases are rerun with.
BOOST_AUTO_TEST_CASE(sharded_team_launch_supplies_the_expected_team) {
    const auto expected = expected_team();
    if (!expected) {
        BOOST_TEST_MESSAGE("monoprop_TEST_EXPECT_TEAM is unset; not a fixed-team launch");
        return;
    }
    const auto options = team_options();
    BOOST_TEST(options.threads == *expected);
    BOOST_TEST(omp_get_dynamic() == 0);
    const auto threads = team_size(options);
    auto observations = std::vector<Observation>(threads);
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        sharded::phase(failure, shard, [&] { observe(observations[shard]); });
    });
    BOOST_TEST(!error);
    auto distinct = std::set<std::thread::id>{};
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST(observations[shard].calls.load() == 1, "shard " << shard);
        BOOST_TEST(observations[shard].team == *expected, "shard " << shard);
        distinct.insert(observations[shard].thread);
    }
    BOOST_TEST(distinct.size() == static_cast<size_t>(*expected));
    BOOST_TEST_MESSAGE("sharded_team_launch_supplies_the_expected_team: observed "
                       << distinct.size() << " workers, expected " << *expected);
}
