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

#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "monoprop/detail/EnvConfig.h"

using monoprop::config::Profile;
using monoprop::config::detail::parse_flag;
using monoprop::config::detail::parse_positive_int;
#ifdef monoprop_ENABLE_PROFILE
using monoprop::config::detail::parse_profile;
#endif

namespace {

#ifdef monoprop_ENABLE_PROFILE

// Every region bit, not just the one a case is named for: an extra region set is as wrong as none.
auto check_profile(const char *text, const Profile &expected) -> void {
    const Profile got = parse_profile(text);
    const char *label = text == nullptr ? "<nullptr>" : text;
    BOOST_CHECK_MESSAGE(got.layer == expected.layer, label << ": layer");
    BOOST_CHECK_MESSAGE(got.comm == expected.comm, label << ": comm");
    BOOST_CHECK_MESSAGE(got.replay == expected.replay, label << ": replay");
    BOOST_CHECK_MESSAGE(got.mem == expected.mem, label << ": mem");
    BOOST_CHECK_MESSAGE(got.any() == expected.any(), label << ": any()");
}

#endif // monoprop_ENABLE_PROFILE

} // namespace

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

#ifdef monoprop_ENABLE_PROFILE

BOOST_AUTO_TEST_CASE(env_config_parse_profile_unset_is_every_region_off) {
    check_profile(nullptr, Profile{});
    check_profile("", Profile{});
}

BOOST_AUTO_TEST_CASE(env_config_parse_profile_single_region) {
    check_profile("layer", Profile{.layer = true});
    check_profile("comm", Profile{.comm = true});
    check_profile("replay", Profile{.replay = true});
    check_profile("mem", Profile{.mem = true});
}

BOOST_AUTO_TEST_CASE(env_config_parse_profile_comma_list) {
    check_profile("layer,comm", Profile{.layer = true, .comm = true});
    check_profile("replay,mem", Profile{.replay = true, .mem = true});
    check_profile("comm,comm", Profile{.comm = true}); // idempotent, not a toggle
}

BOOST_AUTO_TEST_CASE(env_config_parse_profile_tolerates_whitespace_and_empty_tokens) {
    check_profile(" layer , mem ", Profile{.layer = true, .mem = true});
    check_profile("\tcomm\t", Profile{.comm = true});
    check_profile("layer,,comm", Profile{.layer = true, .comm = true}); // doubled comma
    check_profile("layer,", Profile{.layer = true});                    // trailing comma
    check_profile(",layer", Profile{.layer = true});                    // leading comma
    check_profile(",", Profile{});                                      // separators and nothing else
    check_profile("   ", Profile{});
    // Whitespace is tolerated only AROUND a name: a missing comma must reach the unknown-region rule.
    BOOST_CHECK_THROW(parse_profile("layer comm"), std::invalid_argument);
}

// `all` must SET bits, not assign the struct, or `newregion,all` clears the region just named.
BOOST_AUTO_TEST_CASE(env_config_parse_profile_all_sets_every_region) {
    constexpr std::array<std::string_view, 4> kRegionNames{"layer", "comm", "replay", "mem"};
    static_assert(sizeof(Profile) == kRegionNames.size() * sizeof(bool),
                  "add the new Profile field's region name here, and a check for its bit in check_profile");

    const Profile every{.layer = true, .comm = true, .replay = true, .mem = true};
    check_profile("all", every);
    check_profile(" all ", every);
    for (const std::string_view name : kRegionNames) {
        check_profile((std::string(name) + ",all").c_str(), every);
        check_profile(("all," + std::string(name)).c_str(), every);
    }
}

// An unrecognised region throws: a diagnostic that silently does nothing looks like one whose subject
// did nothing.
BOOST_AUTO_TEST_CASE(env_config_parse_profile_unknown_region_throws) {
    BOOST_CHECK_THROW(parse_profile("Layer"), std::invalid_argument); // matching is case-SENSITIVE
    BOOST_CHECK_THROW(parse_profile("ALL"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("layers"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("layer,bogus"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("bogus,layer"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("1"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("0"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("true"), std::invalid_argument);
    // Unset and empty already turn every region off, so no disabling token may become one that parses.
    BOOST_CHECK_THROW(parse_profile("off"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("none"), std::invalid_argument);
    BOOST_CHECK_THROW(parse_profile("layer,off"), std::invalid_argument);
    // Throwing is only half the contract: the message has to name the token, or the fix is a guess.
    BOOST_CHECK_EXCEPTION(parse_profile("layer,bogus"), std::invalid_argument, [](const std::invalid_argument &error) {
        return std::string_view(error.what()).find("bogus") != std::string_view::npos;
    });
}

// Both wrong guesses -- ON (which regions exist) and OFF (`0`) -- by fact, not by wording.
BOOST_AUTO_TEST_CASE(env_config_parse_profile_error_message_teaches_both_halves) {
    const auto message = [](const char *text) {
        try {
            parse_profile(text);
        }
        catch (const std::invalid_argument &error) {
            return std::string(error.what());
        }
        return std::string{}; // did not throw at all, which the checks below then fail on
    };

    const std::string off_guess = message("0");
    BOOST_REQUIRE(!off_guess.empty());
    BOOST_CHECK(off_guess.find("'0'") != std::string::npos); // the token, quoted as typed
    BOOST_CHECK(off_guess.find("monoprop_PROFILE") != std::string::npos);
    for (const std::string_view name : {"layer", "comm", "replay", "mem", "all"}) {
        BOOST_CHECK_MESSAGE(off_guess.find(name) != std::string::npos, name << " missing from: " << off_guess);
    }
    BOOST_CHECK_MESSAGE(off_guess.find("unset") != std::string::npos, off_guess);
    BOOST_CHECK_MESSAGE(off_guess.find("empty") != std::string::npos, off_guess);

    for (const char *text : {"1", "off", "none", "layer,bogus"}) {
        const std::string what = message(text);
        BOOST_CHECK_MESSAGE(what.find("unset") != std::string::npos, text << ": " << what);
        BOOST_CHECK_MESSAGE(what.find("empty") != std::string::npos, text << ": " << what);
    }
}

#else // monoprop_ENABLE_PROFILE

// Without the instrument the parser does not exist, so the contract is the refusal: an exported
// monoprop_PROFILE must not be answered with an empty log. validate() is what bindings.cpp calls.
BOOST_AUTO_TEST_CASE(env_config_profile_without_instrument_is_refused) {
    const char *const set = std::getenv("monoprop_PROFILE");
    if (set != nullptr && set[0] != '\0') {
        BOOST_CHECK_THROW(monoprop::config::validate(), std::invalid_argument);
    }
    else {
        BOOST_CHECK_NO_THROW(monoprop::config::validate());
    }
}

#endif // monoprop_ENABLE_PROFILE

// Each knob declares its default TWICE -- Settings' member and get()'s parse fallback -- and get()
// overwrites the first with the second, so changing only the member silently does nothing.
BOOST_AUTO_TEST_CASE(env_config_member_defaults_match_parsed_defaults) {
    const monoprop::config::Settings declared;
    const auto &effective = monoprop::config::get();

    struct Knob {
        const char *env;    // the variable that would override the member, voiding the comparison
        const char *member; // what to name when the two declarations disagree
        bool declared;
        bool effective;
    };
    // The four profile bits share one variable, so these rows guard get()'s assignment path only.
    const Knob knobs[] = {
        {"monoprop_PARTITION_PINNING", "partition_pinning", declared.partition_pinning, effective.partition_pinning},
        {"monoprop_PROFILE", "profile.layer", declared.profile.layer, effective.profile.layer},
        {"monoprop_PROFILE", "profile.comm", declared.profile.comm, effective.profile.comm},
        {"monoprop_PROFILE", "profile.replay", declared.profile.replay, effective.profile.replay},
        {"monoprop_PROFILE", "profile.mem", declared.profile.mem, effective.profile.mem},
        {"monoprop_PROFILE", "profile.any()", declared.profile.any(), effective.profile.any()},
    };

    size_t compared = 0;
    for (const auto &knob : knobs) {
        const char *override_value = std::getenv(knob.env);
        if (override_value != nullptr && override_value[0] != '\0') {
            continue; // the environment is meant to win; there is no default to compare against
        }
        ++compared;
        BOOST_CHECK_MESSAGE(knob.declared == knob.effective, knob.member);
    }

    if (const char *value = std::getenv("monoprop_NUM_THREADS"); value == nullptr || value[0] == '\0') {
        ++compared;
        BOOST_CHECK_MESSAGE(declared.num_threads == effective.num_threads, "num_threads");
    }

    // The documented defaults, whatever this process's environment says.
    BOOST_CHECK(declared.partition_pinning);
    BOOST_CHECK(!declared.profile.any());
    BOOST_CHECK(declared.num_threads == std::nullopt);

    // Under a fully-exported environment every knob would be skipped and the case would pass vacuously.
    BOOST_CHECK_GT(compared, 0U);
}

BOOST_AUTO_TEST_CASE(env_config_settings_cached_singleton) {
    const auto &a = monoprop::config::get();
    const auto &b = monoprop::config::get();
    BOOST_CHECK_EQUAL(&a, &b);
}

// validate()'s FAILING path is unreachable here (the Settings are already cached), so what is checked
// is that it is callable twice and leaves the cache untouched.
BOOST_AUTO_TEST_CASE(env_config_validate_forces_the_parse_without_disturbing_it) {
    const auto &before = monoprop::config::get();
    BOOST_CHECK_NO_THROW(monoprop::config::validate());
    BOOST_CHECK_NO_THROW(monoprop::config::validate());
    BOOST_CHECK_EQUAL(&before, &monoprop::config::get());
}
