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

// Which row backend a propagator ends up on, and that monoprop_ROW_STORE is wired to it.
//
// This is the guard against a false negative in the rest of the suite: every case runs a second time
// under monoprop_ROW_STORE=sparse (see cpp/tests/boostAddTests.cmake), and every fixture is far below
// the automatic crossover -- so if the variable reached nothing, those 246 extra passes would be the
// dense backend passing twice and nobody would notice.

#include <boost/test/unit_test.hpp>

#include <cstddef>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/operator/SparseRowStore.h"

#include "TestData.h"
#include "TestUtilities.h"

using namespace monoprop;

namespace {

constexpr size_t kNumModes = 8;

auto small_propagator() -> MonomialPropagator {
    const auto data = test_utils::load_case_data("random_exact.msgpack");
    return test_utils::build_simulator(kNumModes, data);
}

} // namespace

// The propagator's backend must be what the environment asked for -- and, unset, what the crossover
// says. Both arms are live: the default ctest variant takes the first, the sparse-rows variant the
// second, so this case is the one that fails if the variable is ignored.
BOOST_AUTO_TEST_CASE(row_store_selection_follows_the_environment) {
    const auto propagator = small_propagator();
    const size_t storage_modes = propagator.storage_num_modes();
    // A fixture-sized system: below every shipped crossover, so `auto` must pick dense here. If this
    // ever fails, the suite's sparse coverage has stopped being a second configuration.
    BOOST_REQUIRE(!monoprop::detail::SparseRowStore::preferred_for_modes(storage_modes));

    switch (config::get().row_store) {
        case config::RowStore::Sparse:
            BOOST_TEST(propagator.rows_are_sparse());
            break;
        case config::RowStore::Dense:
        case config::RowStore::Auto:
            BOOST_TEST(!propagator.rows_are_sparse());
            break;
    }
}

// The automatic rule, independent of any propagator: the crossover is a whole 32-mode block, which is
// what storage_modes_for() produces, so no storage width can straddle it.
BOOST_AUTO_TEST_CASE(row_store_auto_crossover_is_a_whole_storage_block) {
    constexpr size_t kMin = monoprop::detail::SparseRowStore::kMinModes;
    BOOST_TEST(kMin % 32 == 0U);
    BOOST_TEST(!monoprop::detail::SparseRowStore::preferred_for_modes(kMin - 1));
    BOOST_TEST(monoprop::detail::SparseRowStore::preferred_for_modes(kMin));
    BOOST_TEST(monoprop::detail::SparseRowStore::preferred_for_modes(kMin + 32));
    BOOST_TEST(monoprop::detail::storage_modes_for(kMin) == kMin);
}

// An unrecognized value must be rejected, not silently treated as `auto`: the whole reason to set the
// variable is to know which backend ran, and a typo that fell back would read as a passing run of a
// configuration that never happened. Parsed here rather than through config::get(), which caches the
// process environment once and so cannot be re-read per case.
BOOST_AUTO_TEST_CASE(row_store_env_parses_only_the_three_values) {
    BOOST_TEST((config::detail::parse_row_store(nullptr) == config::RowStore::Auto));
    BOOST_TEST((config::detail::parse_row_store("") == config::RowStore::Auto));
    BOOST_TEST((config::detail::parse_row_store("auto") == config::RowStore::Auto));
    BOOST_TEST((config::detail::parse_row_store("dense") == config::RowStore::Dense));
    BOOST_TEST((config::detail::parse_row_store("sparse") == config::RowStore::Sparse));
    for (const char *bad : {"Sparse", "SPARSE", "spars", "sparse ", "1", "on", "packed"}) {
        BOOST_TEST(!config::detail::parse_row_store(bad).has_value());
    }
}
