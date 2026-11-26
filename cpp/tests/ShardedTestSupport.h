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

// Test-only support shared by the sharded runtime suites: the launch's captured team and the Boost preconditions on
// it, a scoped environment variable, the dedicated multi-rank launch check, exact keys and bit patterns for bitwise
// comparisons, and the coefficient encoding the operator fixtures use.

#include <boost/test/unit_test.hpp>

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "AllocationProbe.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/Team.h"

namespace sharded_test {

using namespace monoprop;
namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;
namespace allocation = test_utils::allocation;
using Key = std::vector<uint64_t>;
using RetainedMap = std::map<Key, double>;

inline constexpr size_t num_modes = 8;

inline auto team_options() -> parallel::Options {
    return parallel::capture_thread_budget();
}

inline auto team_size(parallel::Options options = team_options()) -> size_t {
    return static_cast<size_t>(options.threads);
}

inline auto has_nonprimary_worker(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const auto threads = parallel::capture_thread_budget().threads;
    boost::test_tools::assertion_result result(threads >= 2);
    result.message() << "the captured team budget is " << threads << "; this case needs a nonprimary worker";
    return result;
}

inline auto has_allocation_probe(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    boost::test_tools::assertion_result result(allocation::available());
    result.message() << "this build's sanitizer runtime owns operator new/delete, so no allocation probe is linked";
    return result;
}

// After an injected allocation failure, no worker thread keeps an armed failure.
inline auto disarm_team() -> void {
    static_cast<void>(sharded::run_team(team_options(), [](size_t, sharded::TeamFailure &) noexcept {
        (void)allocation::disarm_failure();
    }));
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

// Multi-rank cases run only in their dedicated launches, which name the world size (cpp/tests/CMakeLists.txt).
inline auto dedicated_launch(size_t ranks) -> bool {
    const char *text = std::getenv("monoprop_TEST_SHARDED_RANKS");
    return text != nullptr && std::stoul(text) == ranks;
}

inline auto message_of(const std::exception_ptr &error) -> std::string {
    try {
        std::rethrow_exception(error);
    }
    catch (const std::exception &e) {
        return e.what();
    }
    catch (...) {
        return "non-standard exception";
    }
}

template <size_t N>
auto key_of(const Monomial<N> &m) -> Key {
    Key key;
    for (size_t w = 0; w < Monomial<N>::num_words(); ++w) {
        key.push_back(m.word(w));
    }
    return key;
}

inline auto bits(double v) -> uint64_t {
    return std::bit_cast<uint64_t>(v);
}

// Stores the coefficient whose encoding is `value`, so every fixture is encodable in its basis.
inline auto add_term(OperatorDict &op, Basis basis, const VecZ &indices, double value) -> void {
    op[indices] =
        algebra_decode_coeff<num_modes>(basis, std::complex<double>(value, 0.0), indices_to_bitset<num_modes>(indices));
}

} // namespace sharded_test
