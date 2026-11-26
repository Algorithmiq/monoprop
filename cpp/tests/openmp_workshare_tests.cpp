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

// Workers never call BOOST_TEST: each block writes only its own preallocated element (or an atomic), and every
// assertion runs on the caller after for_blocks has returned.

#include <boost/test/unit_test.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

using monoprop::detail::parallel::for_blocks;

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
