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

#include <chrono>
#include <optional>

#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/mpi/CpuRelax.h"
#include "monoprop/detail/mpi/PartitionBarrier.h"

using monoprop::config::detail::parse_flag;
using monoprop::config::detail::parse_positive_int;

BOOST_AUTO_TEST_CASE(env_config_parse_flag_default_when_unset_or_empty) {
    BOOST_CHECK_EQUAL(parse_flag(nullptr, true), true);
    BOOST_CHECK_EQUAL(parse_flag(nullptr, false), false);
    BOOST_CHECK_EQUAL(parse_flag("", true), true);
    BOOST_CHECK_EQUAL(parse_flag("", false), false);
}

BOOST_AUTO_TEST_CASE(env_config_parse_flag_falsey_first_char) {
    // Only the first character decides, so "0abc" is falsey too.
    for (const char *v : {"0", "f", "F", "n", "N"}) {
        BOOST_CHECK_MESSAGE(parse_flag(v, true) == false, v);
    }
    BOOST_CHECK_EQUAL(parse_flag("0abc", true), false);
}

BOOST_AUTO_TEST_CASE(env_config_parse_flag_truthy_first_char) {
    for (const char *v : {"1", "t", "T", "y", "Y", "on", "true", "anything"}) {
        BOOST_CHECK_MESSAGE(parse_flag(v, false) == true, v);
    }
}

BOOST_AUTO_TEST_CASE(env_config_parse_positive_int_null_and_malformed) {
    BOOST_CHECK(parse_positive_int(nullptr) == std::nullopt);
    BOOST_CHECK(parse_positive_int("") == std::nullopt);
    BOOST_CHECK(parse_positive_int("abc") == std::nullopt);
    BOOST_CHECK(parse_positive_int("12x") == std::nullopt); // trailing junk rejects, not a partial 12
    BOOST_CHECK(parse_positive_int("  ") == std::nullopt);  // strtol consumes ws, then end == text
}

BOOST_AUTO_TEST_CASE(env_config_parse_positive_int_range) {
    BOOST_CHECK(parse_positive_int("0") == std::nullopt);
    BOOST_CHECK(parse_positive_int("-5") == std::nullopt);
    BOOST_CHECK(parse_positive_int("1000001") == std::nullopt); // above the 1e6 ceiling
    BOOST_CHECK(parse_positive_int("1") == std::optional<int>(1));
    BOOST_CHECK(parse_positive_int("42") == std::optional<int>(42));
    BOOST_CHECK(parse_positive_int("1000000") == std::optional<int>(1'000'000)); // inclusive upper bound
}

// Observe the budget the barrier actually resolved, rather than recomputing
// `parse_positive_int(...).value_or(kDefault)` here: an assertion of that shape passes even if the
// barrier is never wired to the setting at all, which is the only thing worth checking. config::get()
// caches on first call, so the env path is unreachable in-process; the injectable override is not.
BOOST_AUTO_TEST_CASE(env_config_spin_budget_reaches_the_barrier) {
    using namespace std::chrono_literals;
    BOOST_CHECK_GT(monoprop::mpi::detail::kDefaultSpinBudgetUs, 0);

    // Default construction lands on the configured value, else the compiled-in default -- never on a
    // zero-length spin, which would send every waiter straight to sched_yield.
    const monoprop::mpi::PartitionBarrier configured(2);
    const auto expected = std::chrono::microseconds{
        monoprop::config::get().spin_budget_us.value_or(monoprop::mpi::detail::kDefaultSpinBudgetUs)};
    BOOST_CHECK(configured.spin_budget() == expected);
    BOOST_CHECK_GT(configured.spin_budget().count(), 0);

    // An explicit override wins over both, which is what lets the default be swept and justified.
    const monoprop::mpi::PartitionBarrier overridden(2, {}, 25us);
    BOOST_CHECK(overridden.spin_budget() == 25us);
}

BOOST_AUTO_TEST_CASE(env_config_settings_cached_singleton) {
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    BOOST_CHECK_EQUAL(&a, &b);
    // Touch a field so the Settings aggregate is actually read.
    BOOST_CHECK(a.partition_pinning == true || a.partition_pinning == false);
}
