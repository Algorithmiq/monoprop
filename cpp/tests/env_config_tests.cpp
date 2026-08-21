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

using monoprop::config::detail::parse_env_flag;
using monoprop::config::detail::parse_positive_int;

BOOST_AUTO_TEST_CASE(env_config_parse_env_flag_unset_and_empty_are_off) {
    BOOST_CHECK_EQUAL(parse_env_flag(nullptr), false);
    BOOST_CHECK_EQUAL(parse_env_flag(""), false);
}

BOOST_AUTO_TEST_CASE(env_config_parse_env_flag_falsey_words_in_any_case) {
    // `off` and `OFF` are the cases the deleted first-character parser read as ON.
    for (const char *v : {"0", "false", "FALSE", "False", "no", "NO", "No", "off", "OFF", "Off"}) {
        BOOST_CHECK_MESSAGE(parse_env_flag(v) == false, v);
    }
}

BOOST_AUTO_TEST_CASE(env_config_parse_env_flag_truthy_words) {
    for (const char *v : {"1", "true", "TRUE", "yes", "on", "ON", "anything"}) {
        BOOST_CHECK_MESSAGE(parse_env_flag(v), v);
    }
}

BOOST_AUTO_TEST_CASE(env_config_parse_env_flag_compares_whole_words) {
    // A first-character parser calls every one of these off; a whole-word one calls them all on.
    for (const char *v : {"offbeat", "november", "0abc", "nope", "falsey", "f", "n"}) {
        BOOST_CHECK_MESSAGE(parse_env_flag(v), v);
    }
    // The converse: a prefix of a falsey word is not that word either.
    BOOST_CHECK_EQUAL(parse_env_flag("of"), true);
    BOOST_CHECK_EQUAL(parse_env_flag("fals"), true);
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

BOOST_AUTO_TEST_CASE(env_config_settings_cached_singleton) {
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    BOOST_CHECK_EQUAL(&a, &b);
    // Touch a field so the Settings aggregate is actually read.
    BOOST_CHECK(a.commplace == true || a.commplace == false);
}
