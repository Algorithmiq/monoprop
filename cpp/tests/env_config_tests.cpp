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

#include <cstdlib>
#include <optional>

#include "monoprop/detail/EnvConfig.h"

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

// Every knob declares its default TWICE -- once as a Settings member initialiser, once as parse_flag's
// fallback in get() -- and get() unconditionally overwrites the first with the second. So changing only
// the member silently does nothing. That is not hypothetical: it happened once on a knob that was being
// turned OFF after measuring a 7.2% regression -- the member was set to false, the flip was documented in
// the header and in the docs, parse_flag's fallback was left at true, and the regression shipped enabled
// by default anyway. Any knob whose two declarations disagree fails here.
BOOST_AUTO_TEST_CASE(env_config_member_defaults_match_parsed_defaults) {
    const monoprop::config::Settings declared;
    const auto &effective = monoprop::config::get();

    struct Knob {
        const char *name;
        bool declared;
        bool effective;
    };
    const Knob knobs[] = {
        {"monoprop_PARTITION_PINNING", declared.partition_pinning, effective.partition_pinning},
        {"monoprop_LAYER_PROFILE", declared.layer_profile, effective.layer_profile},
        {"monoprop_DIGEST_CUTOFF", declared.digest_cutoff, effective.digest_cutoff},
    };

    size_t compared = 0;
    for (const auto &knob : knobs) {
        const char *override_value = std::getenv(knob.name);
        if (override_value != nullptr && override_value[0] != '\0') {
            continue; // the environment is meant to win; there is no default to compare against
        }
        ++compared;
        BOOST_CHECK_MESSAGE(knob.declared == knob.effective, knob.name);
    }
    // Under a fully-exported environment every knob would be skipped and the case would pass vacuously.
    BOOST_CHECK_GT(compared, 0U);
}

BOOST_AUTO_TEST_CASE(env_config_settings_cached_singleton) {
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    BOOST_CHECK_EQUAL(&a, &b);
    // Touch a field so the Settings aggregate is actually read.
    BOOST_CHECK(a.partition_pinning == true || a.partition_pinning == false);
}
