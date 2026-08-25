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

#include <optional>

#include "monoprop/detail/EnvConfig.h"

using monoprop::config::detail::parse_positive_int;

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

BOOST_AUTO_TEST_CASE(env_config_settings_cached_singleton) {
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    BOOST_CHECK_EQUAL(&a, &b);
    // Touch a field so the Settings aggregate is actually read.
    BOOST_CHECK(a.num_threads == std::nullopt || *a.num_threads >= 1);
}
