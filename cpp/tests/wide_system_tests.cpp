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

// The engine at a storage width that runs the support-form row store in a released wheel.
//
// Every checked-in fixture is 28 modes or fewer, so all of them store monomials in a single 32-mode
// block. That is below every sparse-row crossover, which leaves a gap the rest of the suite cannot
// close: monoprop_ROW_STORE=sparse can force the support-form store onto fixture-width rows and pass,
// while the width it was built for -- several words per monomial, an active window short of its
// storage, mode lanes spread over more than one codes word -- stays unexercised.
//
// So this case embeds a fixture instead of adding one: LiH's 12 modes relabelled into a 90-mode system
// (test_utils::ModeEmbedding), which stores at 96. The embedding is also the oracle -- a monotone mode
// relabelling is a canonical transformation, so the fixture's exact energy still applies, and the
// truncated run still owes the narrow run's value.

#include <boost/test/unit_test.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/EnvConfig.h"

#include "TestData.h"
#include "TestPropagator.h"
#include "TestUtilities.h"

using namespace monoprop;

namespace {

// The source fixture's own width. The embedding only relabels, so the physics never leaves these 12
// modes: no evolved term can carry more than 24 Majorana indices or occupy more than 12 modes, which is
// what makes the cutoffs below untruncated -- and the reference energy exact -- at 90 modes.
constexpr size_t kSourceModes = 12;

constexpr size_t kWideModes = 90;
constexpr size_t kWideStorageModes = 96;

// The twin of tests/cases.py's WIDE_EMBEDDING, position for position; see there for why these twelve.
// In short: 0 and 89 are the ends of the active window, 25/26 and 57/58 straddle the storage-word
// boundaries once the 6-mode window offset is applied, and 31/32 and 63/64 straddle the boundaries the
// same modes would fall on with a flush window.
auto wide_embedding() -> test_utils::ModeEmbedding {
    return {.num_modes = kWideModes, .modes = {0, 25, 26, 31, 32, 57, 58, 63, 64, 87, 88, 89}};
}

auto wide_case() -> test_utils::CaseData {
    return test_utils::embed_case(test_utils::load_case_data<kSourceModes>("lih_fermionic_spin_exact.msgpack"),
                                  wide_embedding());
}

// Storage pinned to 96 and the logical width to 90 -- which is what storage_modes_for(90) would have
// produced anyway, so this is the one place in the suite where the pinned width is the production one.
auto wide_propagator(const test_utils::CaseData &data, unsigned int cutoff, CutoffType cutoff_type)
    -> MonomialPropagator {
    return test_utils::make_propagator<kWideStorageModes>(data.hamiltonian,
                                                          cutoff,
                                                          data.initial_state,
                                                          /*schrodinger_cutoff=*/std::nullopt,
                                                          MPI_COMM_SELF,
                                                          /*lower_atol=*/std::nullopt,
                                                          /*upper_atol=*/std::nullopt,
                                                          cutoff_type,
                                                          /*basis_change=*/std::nullopt,
                                                          /*logical_num_modes=*/kWideModes);
}

// (cutoff_type, cutoff) pairs that truncate nothing for this problem, and the pair that does.
constexpr std::array<std::pair<CutoffType, unsigned int>, 2> kUntruncated{
    {{CutoffType::Length, 2 * kSourceModes}, {CutoffType::Support, kSourceModes}}};

} // namespace

// The premise the two cases below rest on. 96 storage modes is at or above the crossover a released
// wheel is built with (monoprop_SPARSE_ROW_MIN_MODES is 96 unless -march=native moves it to 1024), so
// on a wheel this width selects the support-form store by itself; a dev build with arch flags on gets
// there through monoprop_ROW_STORE=sparse, i.e. the sparse-rows ctest variant. Asserted so that variant
// cannot quietly have run dense rows a second time.
BOOST_AUTO_TEST_CASE(wide_case_stores_at_ninety_six_modes) {
    const auto data = wide_case();
    const auto propagator = wide_propagator(data, 2 * kSourceModes, CutoffType::Length);
    BOOST_TEST(data.num_modes == kWideModes);
    BOOST_TEST(propagator.logical_num_modes() == kWideModes);
    BOOST_TEST(propagator.storage_num_modes() == kWideStorageModes);
    BOOST_TEST(MonomialPropagator::storage_modes_for(kWideModes) == kWideStorageModes);
    if (config::get().row_store == config::RowStore::Sparse) {
        BOOST_TEST(propagator.rows_are_sparse());
    }
}

// The fixture's exact energy, reached at a width no fixture on disk has. Both cutoff kinds run: each
// has its own evaluator, and each has a codes-word form the sparse rows use instead.
BOOST_AUTO_TEST_CASE(wide_case_reaches_its_exact_expectation_value) {
    const auto data = wide_case();
    for (const auto [cutoff_type, cutoff] : kUntruncated) {
        BOOST_TEST_CONTEXT("cutoff_type=" << static_cast<int>(cutoff_type) << " cutoff=" << cutoff) {
            auto propagator = wide_propagator(data, cutoff, cutoff_type);
            const double expval = test_utils::evaluate_expval(propagator, data, /*pare=*/false);
            test_utils::check_expval_close("wide embedding", expval, data.actual_expval);
        }
    }
}

// With truncation on, which terms survive must not depend on the storage width: both cutoffs count
// something the relabelling preserves -- Majorana indices, or occupied modes. So the wide run owes the
// narrow run's energy, to rather more than the 1e-9 an exact-value check asks for. Compared as a value
// rather than term by term because that comparison is the Python suite's (tests/test_wide_system.py),
// which can hold the two term sets side by side.
BOOST_AUTO_TEST_CASE(truncated_wide_run_matches_the_narrow_run) {
    const auto narrow_data = test_utils::load_case_data<kSourceModes>("lih_fermionic_spin_exact.msgpack");
    const auto wide_data = test_utils::embed_case(narrow_data, wide_embedding());
    constexpr unsigned int kTruncating = 4;

    for (const auto cutoff_type : {CutoffType::Length, CutoffType::Support}) {
        BOOST_TEST_CONTEXT("cutoff_type=" << static_cast<int>(cutoff_type)) {
            auto narrow = test_utils::make_propagator<kSourceModes>(narrow_data.hamiltonian,
                                                                    kTruncating,
                                                                    narrow_data.initial_state,
                                                                    std::nullopt,
                                                                    MPI_COMM_SELF,
                                                                    std::nullopt,
                                                                    std::nullopt,
                                                                    cutoff_type);
            auto wide = wide_propagator(wide_data, kTruncating, cutoff_type);
            const double narrow_expval = test_utils::evaluate_expval(narrow, narrow_data, /*pare=*/false);
            const double wide_expval = test_utils::evaluate_expval(wide, wide_data, /*pare=*/false);
            BOOST_TEST(test_utils::near(wide_expval, narrow_expval, /*atol=*/1e-12, /*rtol=*/1e-12));
        }
    }
}
