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

// Direct unit coverage of EnvConfig.h — the single home for monoprop_* environment parsing. The
// two pure parsers (parse_flag / parse_positive_int) are pinned here against every branch of their
// documented semantics; config::get() is exercised once for the cached-Settings path. The engine
// consumes these via config::get() on hot paths, but that reads the environment once at load, so the
// parsers themselves are only fully swept in isolation.
//
// Cases are flat top-level BOOST_AUTO_TEST_CASEs sharing an env_config_ prefix (not a
// BOOST_AUTO_TEST_SUITE) to match this suite's ctest discovery, which registers by leaf case name.

#include <boost/test/unit_test.hpp>

#include <optional>

#include "monoprop/detail/EnvConfig.h"

using monoprop::config::detail::parse_flag;
using monoprop::config::detail::parse_positive_int;

BOOST_AUTO_TEST_CASE(env_config_parse_flag_default_when_unset_or_empty) {
    // nullptr and empty string both fall through to the supplied default (either polarity).
    BOOST_CHECK_EQUAL(parse_flag(nullptr, true), true);
    BOOST_CHECK_EQUAL(parse_flag(nullptr, false), false);
    BOOST_CHECK_EQUAL(parse_flag("", true), true);
    BOOST_CHECK_EQUAL(parse_flag("", false), false);
}

BOOST_AUTO_TEST_CASE(env_config_parse_flag_falsey_first_char) {
    // Falsey iff the first character is one of {0,f,F,n,N}; the default is irrelevant once set.
    for (const char *v : {"0", "f", "F", "n", "N"}) {
        BOOST_CHECK_MESSAGE(parse_flag(v, true) == false, v);
    }
    BOOST_CHECK_EQUAL(parse_flag("0abc", true), false); // only the first char matters
}

BOOST_AUTO_TEST_CASE(env_config_parse_flag_truthy_first_char) {
    for (const char *v : {"1", "t", "T", "y", "Y", "on", "true", "anything"}) {
        BOOST_CHECK_MESSAGE(parse_flag(v, false) == true, v);
    }
}

BOOST_AUTO_TEST_CASE(env_config_parse_positive_int_null_and_malformed) {
    BOOST_CHECK(parse_positive_int(nullptr) == std::nullopt);
    BOOST_CHECK(parse_positive_int("") == std::nullopt);    // end == text
    BOOST_CHECK(parse_positive_int("abc") == std::nullopt); // end == text
    BOOST_CHECK(parse_positive_int("12x") == std::nullopt); // trailing junk (*end != '\0')
    BOOST_CHECK(parse_positive_int("  ") == std::nullopt);  // strtol consumes ws then end == text
}

BOOST_AUTO_TEST_CASE(env_config_parse_positive_int_range) {
    BOOST_CHECK(parse_positive_int("0") == std::nullopt);       // value <= 0
    BOOST_CHECK(parse_positive_int("-5") == std::nullopt);      // value <= 0
    BOOST_CHECK(parse_positive_int("1000001") == std::nullopt); // value > 1e6
    BOOST_CHECK(parse_positive_int("1") == std::optional<int>(1));
    BOOST_CHECK(parse_positive_int("42") == std::optional<int>(42));
    BOOST_CHECK(parse_positive_int("1000000") == std::optional<int>(1'000'000)); // inclusive upper bound
}

BOOST_AUTO_TEST_CASE(env_config_settings_cached_singleton) {
    // get() parses the environment once and returns the same immutable instance every call.
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    BOOST_CHECK_EQUAL(&a, &b);
    // Touch the fields so the Settings aggregate is read (documented defaults unless the environment
    // overrode them for this process).
    BOOST_CHECK(a.shard_pinning == true || a.shard_pinning == false);
    BOOST_CHECK(a.phase_timers == true || a.phase_timers == false);
}
