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

// Test-only support shared by the sharded runtime suites: the launch's captured team, a scoped environment variable
// and failure inspection.

#include <cstdlib>
#include <exception>
#include <optional>
#include <string>

#include "RankPreconditions.h"
#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"

namespace sharded_test {

inline auto team_options() -> monoprop::detail::parallel::Options {
    return monoprop::detail::parallel::capture_thread_budget();
}

// Sets (or unsets, for nullopt) one environment variable for a scope, then restores it.
class ScopedEnv {
public:
    ScopedEnv(const char *name, std::optional<std::string> value) : name_(name) {
        if (const char *old = std::getenv(name)) {
            old_ = std::string(old);
        }
        if (value) {
            ::setenv(name, value->c_str(), 1);
        }
        else {
            ::unsetenv(name);
        }
    }
    ScopedEnv(const ScopedEnv &) = delete;
    auto operator=(const ScopedEnv &) -> ScopedEnv & = delete;
    ~ScopedEnv() {
        if (old_) {
            ::setenv(name_, old_->c_str(), 1);
        }
        else {
            ::unsetenv(name_);
        }
    }

private:
    const char *name_;
    std::optional<std::string> old_;
};

// The team the launch promises in monoprop_TEST_EXPECT_TEAM (cpp/tests/CMakeLists.txt): 1 in the per-case runs, T in
// the fixed-T launches; empty when the run names none.
inline auto expected_team() -> std::optional<int> {
    const char *value = std::getenv("monoprop_TEST_EXPECT_TEAM");
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::stoi(value);
}

// The message of a non-empty exception pointer.
inline auto message_of(const std::exception_ptr &error) -> std::string {
    return monoprop::mpi::describe_exception(error);
}

} // namespace sharded_test
