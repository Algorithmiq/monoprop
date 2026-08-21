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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <optional>

#include "monoprop/detail/EnvConfig.h"

using monoprop::config::detail::parse_flag;
using monoprop::config::detail::parse_positive_int;

TEST_CASE("env_config_parse_flag_default_when_unset_or_empty") {
    CHECK((parse_flag(nullptr, true)) == (true));
    CHECK((parse_flag(nullptr, false)) == (false));
    CHECK((parse_flag("", true)) == (true));
    CHECK((parse_flag("", false)) == (false));
}

TEST_CASE("env_config_parse_flag_falsey_first_char") {
    // Only the first character decides, so "0abc" is falsey too.
    for (const char *v : {"0", "f", "F", "n", "N"}) {
        INFO(v);
        CHECK(parse_flag(v, true) == false);
    }
    CHECK((parse_flag("0abc", true)) == (false));
}

TEST_CASE("env_config_parse_flag_truthy_first_char") {
    for (const char *v : {"1", "t", "T", "y", "Y", "on", "true", "anything"}) {
        INFO(v);
        CHECK(parse_flag(v, false) == true);
    }
}

TEST_CASE("env_config_parse_positive_int_null_and_malformed") {
    CHECK(parse_positive_int(nullptr) == std::nullopt);
    CHECK(parse_positive_int("") == std::nullopt);
    CHECK(parse_positive_int("abc") == std::nullopt);
    CHECK(parse_positive_int("12x") == std::nullopt); // trailing junk rejects, not a partial 12
    CHECK(parse_positive_int("  ") == std::nullopt);  // strtol consumes ws, then end == text
}

TEST_CASE("env_config_parse_positive_int_range") {
    CHECK(parse_positive_int("0") == std::nullopt);
    CHECK(parse_positive_int("-5") == std::nullopt);
    CHECK(parse_positive_int("1000001") == std::nullopt); // above the 1e6 ceiling
    CHECK(parse_positive_int("1") == std::optional<int>(1));
    CHECK(parse_positive_int("42") == std::optional<int>(42));
    CHECK(parse_positive_int("1000000") == std::optional<int>(1'000'000)); // inclusive upper bound
}

TEST_CASE("env_config_settings_cached_singleton") {
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    CHECK((&a) == (&b));
    // Touch a field so the Settings aggregate is actually read.
    CHECK((a.partition_pinning == true || a.partition_pinning == false));
}
