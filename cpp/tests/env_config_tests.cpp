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

#include <cstdint>
#include <optional>

#include "monoprop/detail/EnvConfig.h"

using monoprop::config::EnvConfigError;
using monoprop::config::RoutingMode;
using monoprop::config::detail::parse_positive_int;
using monoprop::config::detail::parse_routing_mode;
using monoprop::config::detail::parse_uint64;

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

// Both routing parsers throw where parse_positive_int returns nullopt: a routing knob that
// defaulted silently would change the transport with no diagnostic.
BOOST_AUTO_TEST_CASE(env_config_parse_uint64_unset_valid_and_rejected) {
    BOOST_CHECK(parse_uint64("k", nullptr) == std::nullopt);
    BOOST_CHECK(parse_uint64("k", "") == std::nullopt);
    BOOST_CHECK(parse_uint64("k", "0") == std::optional<std::uint64_t>(0));
    BOOST_CHECK(parse_uint64("k", "18446744073709551615") == std::optional<std::uint64_t>(~std::uint64_t{0}));
    BOOST_CHECK_THROW(parse_uint64("k", "abc"), EnvConfigError);
    BOOST_CHECK_THROW(parse_uint64("k", "12x"), EnvConfigError);
    BOOST_CHECK_THROW(parse_uint64("k", "-1"), EnvConfigError);                   // strtoull would WRAP it
    BOOST_CHECK_THROW(parse_uint64("k", "18446744073709551616"), EnvConfigError); // ERANGE
}

BOOST_AUTO_TEST_CASE(env_config_parse_routing_mode_rejects_a_typo) {
    BOOST_CHECK(parse_routing_mode("k", nullptr) == std::nullopt);
    BOOST_CHECK(parse_routing_mode("k", "") == std::nullopt);
    BOOST_CHECK(parse_routing_mode("k", "splitmix") == std::optional<RoutingMode>(RoutingMode::Splitmix));
    BOOST_CHECK(parse_routing_mode("k", "linear") == std::optional<RoutingMode>(RoutingMode::Linear));
    BOOST_CHECK_THROW(parse_routing_mode("k", "dense"), EnvConfigError); // all three used to mean linear
    BOOST_CHECK_THROW(parse_routing_mode("k", "off"), EnvConfigError);
    BOOST_CHECK_THROW(parse_routing_mode("k", "Linear"), EnvConfigError);
}
