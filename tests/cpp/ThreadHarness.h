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

#pragma once

#include <exception>
#include <thread>
#include <vector>

namespace test_utils {

// Run `body(comm, rank)` on S participant threads sharing one transport `comm`; join all. Body
// exceptions are captured per-rank and returned: Boost.Test assertions are only safe on the main thread.
template <class Comm, class Body>
auto run_comm_threads(Comm &comm, int s, Body body) -> std::vector<std::exception_ptr> {
    std::vector<std::exception_ptr> errs(static_cast<size_t>(s));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(s));
    for (int r = 0; r < s; ++r) {
        threads.emplace_back([&, r]() {
            try {
                body(comm, r);
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

} // namespace test_utils
