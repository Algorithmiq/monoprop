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

// The differential test between CodesAlgebra.h and the dense implementations it must replace. Every
// function is checked to agree *exactly* -- these are integer and sign quantities, so there is no
// tolerance to spend -- over real fixture monomials and over randomized rows, at storage widths both
// equal to and wider than the logical width. Making the codes form the default is gated on this.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/CodesAlgebra.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/SparseRowStore.h"

#include "TestData.h"
#include "TestUtilities.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

// Rebuild a dense monomial from a row's mode lanes and an arbitrary codes word, so a codes-side
// transform (pair_swap) can be compared against its dense counterpart. sparse_row_to_monomial is the
// store's own materialization, which is the point: re-implementing it here would leave this oracle
// agreeing with a slot convention the store no longer uses. Only valid where the substituted codes word
// has the same occupancy as the row's own, which is the case for every transform here.
template <size_t NumModes>
auto to_monomial(const SparseRow &row, RowCodes codes) -> Monomial<NumModes> {
    return sparse_row_to_monomial<2 * NumModes>(SparseRow{.modes = row.modes, .codes = codes});
}

// Which outcomes the comparison actually reached. Every branch of every ported function must be
// exercised by the inputs, or agreement is vacuous -- an interleave phase stubbed to `return 1` agrees
// with the dense version on any input set that happens to contain only even permutations.
struct Seen {
    bool paired = false;
    bool unpaired = false;
    bool y_letters = false;
    bool phase_plus = false;
    bool phase_minus = false;
    bool anticommutes = false;
    bool commutes = false;
    bool cutoff_kept = false;
    bool cutoff_dropped = false;

    auto operator|=(const Seen &o) -> Seen & {
        paired |= o.paired;
        unpaired |= o.unpaired;
        y_letters |= o.y_letters;
        phase_plus |= o.phase_plus;
        phase_minus |= o.phase_minus;
        anticommutes |= o.anticommutes;
        commutes |= o.commutes;
        cutoff_kept |= o.cutoff_kept;
        cutoff_dropped |= o.cutoff_dropped;
        return *this;
    }
};

auto require_discriminating(const Seen &seen) -> void {
    BOOST_TEST(seen.paired);
    BOOST_TEST(seen.unpaired);
    BOOST_TEST(seen.y_letters);
    BOOST_TEST(seen.phase_plus);
    BOOST_TEST(seen.phase_minus);
    BOOST_TEST(seen.anticommutes);
    BOOST_TEST(seen.commutes);
    BOOST_TEST(seen.cutoff_kept);
    BOOST_TEST(seen.cutoff_dropped);
}

// Every single-row function at once, against the dense version of each.
template <size_t NumModes>
auto check_row(const Monomial<NumModes> &mono, const SparseRow &row, size_t logical_num_modes, Seen &seen) -> void {
    const size_t inactive_prefix = NumModes - logical_num_modes;

    const auto dense_sums = cutoff_sums<NumModes>(mono, logical_num_modes);
    const auto codes_sums = codes_cutoff_sums(row, inactive_prefix);
    BOOST_TEST(codes_sums.or_sum == dense_sums.or_sum);
    BOOST_TEST(codes_sums.popcount_sum == dense_sums.popcount_sum);
    BOOST_TEST(codes_sums.xor_sum == dense_sums.xor_sum);

    // Both sides of each cutoff's two branches: a cutoff below the term's measure exercises the
    // fully-paired escape, one above it the plain comparison.
    for (const unsigned int cutoff : {0U, 1U, 2U, 4U, 8U, 64U}) {
        const bool kept = codes_length_cutoff(row, cutoff, inactive_prefix);
        BOOST_TEST(kept == length_cutoff<NumModes>(mono, cutoff, logical_num_modes));
        BOOST_TEST(codes_support_cutoff(row, cutoff, inactive_prefix)
                   == support_cutoff<NumModes>(mono, cutoff, logical_num_modes));
        seen.cutoff_kept |= kept;
        seen.cutoff_dropped |= !kept;
    }

    const bool paired = codes_is_paired(row.codes);
    BOOST_TEST(paired == is_paired<NumModes>(mono));
    seen.paired |= paired;
    seen.unpaired |= !paired;

    const size_t y = codes_pauli_y_count(row.codes);
    BOOST_TEST(y == pauli_y_count<NumModes>(mono));
    seen.y_letters |= y > 0;

    BOOST_TEST((to_monomial<NumModes>(row, codes_pair_swap(row.codes)) == pair_swap<NumModes>(mono)));
}

// Encode monomials into one store and hand back both the store and the dense originals. All rows are
// pushed before any view is taken: a view borrows the store's arrays, so growth would dangle it.
template <size_t NumModes>
struct Encoded {
    std::vector<Monomial<NumModes>> dense;
    SparseRowStore<NumModes> store{SparseRowStore<NumModes>::kMaxSlots};

    auto add(const Monomial<NumModes> &mono) -> void {
        dense.push_back(mono);
        store.push_back(mono);
    }
};

template <size_t NumModes>
auto check_all(Encoded<NumModes> &enc, size_t logical_num_modes) -> Seen {
    Seen seen;
    BOOST_REQUIRE(enc.dense.size() == enc.store.size());
    for (size_t i = 0; i < enc.dense.size(); ++i) {
        BOOST_REQUIRE_MESSAGE(!enc.store.spilled(i), "row " << i << " spilled; the algebra needs a codes word");
        check_row<NumModes>(enc.dense[i], enc.store.view(i), logical_num_modes, seen);
    }
    // The two-row functions, over every ordered pair for small sets and a stride otherwise: they are
    // O(n^2) in the row count and the fixtures carry hundreds of terms.
    const size_t n = enc.dense.size();
    const size_t stride = n > 24 ? (n / 24) + 1 : 1;
    for (size_t i = 0; i < n; i += stride) {
        for (size_t k = 0; k < n; k += stride) {
            const auto maj = enc.store.view(i);
            const auto gen = enc.store.view(k);
            const int phase = codes_interleave_phase(maj, gen);
            BOOST_TEST(phase == interleave_phase<NumModes>(enc.dense[i], enc.dense[k]));
            const bool anti = codes_pauli_anticommutes(maj, gen);
            BOOST_TEST(anti == pauli_anticommutes<NumModes>(enc.dense[i], enc.dense[k]));
            seen.phase_plus |= phase > 0;
            seen.phase_minus |= phase < 0;
            seen.anticommutes |= anti;
            seen.commutes |= !anti;
        }
    }
    return seen;
}

// The fixtures' Hamiltonian keys and generator index lists are the real-world monomials: Hermitian
// Majorana products, so the set includes fully paired rows, which are the inputs both cutoffs treat
// specially. NumModes is the storage width; the fixture's own mode count is the logical one.
template <size_t NumModes>
auto check_fixture(const std::string &name) -> Seen {
    const auto data = test_utils::load_case_data<NumModes>(name);
    BOOST_REQUIRE(data.num_modes > 0);
    BOOST_REQUIRE(NumModes >= data.num_modes);
    const size_t max_index = 2 * data.num_modes;
    constexpr size_t kMaxSlots = SparseRowStore<NumModes>::kMaxSlots;

    Encoded<NumModes> enc;
    for (const auto &[inds, coeff] : data.hamiltonian) {
        if (inds.size() > kMaxSlots) {
            continue; // would spill; the store's own tests cover that path
        }
        enc.add(indices_to_bitset_checked<NumModes>(inds, max_index));
    }
    for (const auto &inds : data.majoranas) {
        if (inds.size() > kMaxSlots) {
            continue;
        }
        enc.add(indices_to_bitset_checked<NumModes>(inds, max_index));
    }
    BOOST_REQUIRE_MESSAGE(enc.dense.size() > 1, "fixture " << name << " yielded no monomials to compare");
    return check_all<NumModes>(enc, data.num_modes);
}

// Randomized rows over one (storage, logical) pair.
template <size_t NumModes>
auto check_random(std::mt19937_64 &rng, size_t logical_num_modes) -> Seen {
    Encoded<NumModes> enc;
    for (size_t trial = 0; trial < 120; ++trial) {
        Monomial<NumModes> mono;
        const size_t occupied = rng() % (SparseRowStore<NumModes>::kMaxSlots + 1);
        // Every fourth row is forced fully paired: that is the branch both cutoffs short-circuit on and
        // the only input is_paired accepts.
        const bool force_paired = (trial % 4) == 0;
        for (size_t k = 0; k < occupied; ++k) {
            const size_t mode = rng() % NumModes;
            const unsigned int code = force_paired ? 0b11U : 1U + static_cast<unsigned int>(rng() % 3U);
            if ((code & 1U) != 0U) {
                mono.set(2 * mode);
            }
            if ((code & 2U) != 0U) {
                mono.set((2 * mode) + 1);
            }
        }
        enc.add(mono);
    }
    return check_all<NumModes>(enc, logical_num_modes);
}

} // namespace

// Whole register: storage width equals the logical width, so every mode is active and the codes form
// takes its zero-prefix path. The storage width is a template parameter here, so each fixture is
// instantiated at its own mode count.
BOOST_AUTO_TEST_CASE(codes_algebra_matches_dense_on_fixtures_whole_register) {
    Seen seen;
    seen |= check_fixture<8>("random_exact.msgpack");
    seen |= check_fixture<12>("lih_fermionic_spin_exact.msgpack");
    seen |= check_fixture<16>("S0_8e8o_majoranic_c6.msgpack");
    seen |= check_fixture<16>("majorana_lattice_layer_30.msgpack");
    require_discriminating(seen);
}

// The production layout: the logical modes occupy the top of a wider register and the low physical modes
// are inactive. This is the case cutoff_sums applies active_bit_offset for, and the one the codes form
// has to reproduce by dropping a slot prefix.
BOOST_AUTO_TEST_CASE(codes_algebra_matches_dense_on_fixtures_padded_storage) {
    Seen seen;
    seen |= check_fixture<32>("random_exact.msgpack");
    seen |= check_fixture<32>("lih_fermionic_spin_exact.msgpack");
    seen |= check_fixture<32>("S0_8e8o_majoranic_c6.msgpack");
    seen |= check_fixture<64>("majorana_lattice_layer_30.msgpack");
    require_discriminating(seen);
}

// Randomized rows reach occupancies and code patterns the fixtures do not: single-position modes in
// every combination, rows at the slot capacity, empty rows, and inactive modes actually populated --
// which a propagator never produces but the dense functions accept, so the two must still agree.
BOOST_AUTO_TEST_CASE(codes_algebra_matches_dense_on_randomized_rows) {
    std::mt19937_64 rng(20260812U);
    Seen seen;
    seen |= check_random<32>(rng, 32);
    seen |= check_random<32>(rng, 16);
    seen |= check_random<32>(rng, 29);
    seen |= check_random<64>(rng, 64);
    seen |= check_random<64>(rng, 32);
    seen |= check_random<64>(rng, 61);
    seen |= check_random<128>(rng, 128);
    seen |= check_random<128>(rng, 64);
    seen |= check_random<128>(rng, 125);
    require_discriminating(seen);
}

// The identities in the header, spelled out on hand-built words so a regression names the broken one.
BOOST_AUTO_TEST_CASE(codes_algebra_word_identities) {
    // Slots 0..2 = 0b11, 0b10, 0b01: one paired mode, one upper-only, one lower-only.
    constexpr RowCodes codes = 0b01'10'11ULL;
    BOOST_TEST(row_slot_count(codes) == 3U);
    const auto sums = codes_cutoff_sums(codes);
    BOOST_TEST(sums.or_sum == 3U);       // n
    BOOST_TEST(sums.popcount_sum == 4U); // n + d, d = 1
    BOOST_TEST(sums.xor_sum == 2U);      // n - d
    BOOST_TEST(!codes_is_paired(codes));
    BOOST_TEST(codes_is_paired(0b11'11ULL));
    BOOST_TEST(codes_is_paired(0U)); // an empty row is vacuously paired, as dense is_paired agrees
    BOOST_TEST(codes_pauli_y_count(codes) == 1U);
    BOOST_TEST(codes_pair_swap(codes) == 0b10'01'11ULL);
    BOOST_TEST(codes_pair_swap(codes_pair_swap(codes)) == codes); // an involution

    BOOST_TEST(codes_popcount_below(codes, 0U) == 0U);
    BOOST_TEST(codes_popcount_below(codes, 1U) == 2U);
    BOOST_TEST(codes_popcount_below(codes, 2U) == 3U);
    BOOST_TEST(codes_popcount_below(codes, 3U) == 4U);
    // Past the last slot the answer is the whole word, and the shift that would express it is undefined.
    BOOST_TEST(codes_popcount_below(codes, SparseRowStore<32>::kMaxSlots) == 4U);
}
