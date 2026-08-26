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

// The scan's per-term kernel in support form, against the dense one it must replace. What
// emit_term_products computes per term is the product M(+)G, the overlap popcount(M&G), and the basis
// rotation sign; this asserts all three agree exactly, for both algebras, on real generators from the
// fixtures and on randomized rows -- including the capacity overflow, which must be reported rather than
// silently truncating a mode list.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/CodesAlgebra.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/algebra/PauliAlgebra.h"

#include "TestData.h"
#include "TestUtilities.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

// A row plus the lane storage behind it, so a test can hold several at once. The store's own rows borrow
// its arrays; these do not, which is what lets a product row be built and then compared.
struct OwnedRow {
    std::vector<RowMode> lanes;
    RowCodes codes = 0;

    explicit OwnedRow(size_t capacity) : lanes(capacity, 0) {}

    [[nodiscard]] auto view() const -> SparseRow { return SparseRow{lanes.data(), codes}; }

    static auto encode(const Bitset &mono, size_t capacity) -> OwnedRow {
        OwnedRow row(capacity);
        size_t used = 0;
        for_each_mode_slot(mono, [&](size_t mode, unsigned int code) {
            BOOST_REQUIRE_MESSAGE(used < capacity, "test row exceeded its capacity");
            row.lanes[used] = static_cast<RowMode>(mode);
            row.codes |= static_cast<RowCodes>(code) << (2 * used);
            ++used;
        });
        return row;
    }
};

struct Seen {
    bool cancelled_a_mode = false; // a mode present in both, cancelling to nothing
    bool nonzero_overlap = false;
    bool zero_overlap = false;
    bool majorana_minus = false;
    bool majorana_plus = false;
    bool pauli_minus = false;
    bool pauli_plus = false;

    auto operator|=(const Seen &o) -> Seen & {
        cancelled_a_mode |= o.cancelled_a_mode;
        nonzero_overlap |= o.nonzero_overlap;
        zero_overlap |= o.zero_overlap;
        majorana_minus |= o.majorana_minus;
        majorana_plus |= o.majorana_plus;
        pauli_minus |= o.pauli_minus;
        pauli_plus |= o.pauli_plus;
        return *this;
    }
};

// One term against one generator, every quantity emit_term_products would produce.
auto check_product(const Bitset &mono, const Bitset &gen, size_t capacity, Seen &seen) -> void {
    const size_t num_bits = mono.size();
    const auto mono_row = OwnedRow::encode(mono, capacity);
    const auto gen_row = OwnedRow::encode(gen, capacity);

    // The dense reference, exactly as the scan computes it.
    Bitset dense_product(num_bits);
    const auto fused = mono.fused_xor_into(gen, dense_product);

    std::vector<RowMode> out_lanes(capacity, 0);
    const auto product = sparse_toggle(mono_row.view(), gen_row.view(), out_lanes.data(), capacity);
    BOOST_REQUIRE(!product.overflowed);

    const SparseRow product_row{out_lanes.data(), product.codes};
    BOOST_TEST(product.overlap == fused.overlap);
    BOOST_TEST(product.num_slots == row_slot_count(product.codes));
    BOOST_TEST((sparse_row_to_bitset(product_row, num_bits) == dense_product));

    // The two rotation signs. Majorana's dense form goes through the per-layer interleave mask, which is
    // the hot path the sparse walk replaces, so compare against that and not only against
    // interleave_phase.
    const auto majorana_ctx = MajoranaAlgebra::make_gen_context(gen);
    const int dense_majorana = MajoranaAlgebra::rotation_sign(majorana_ctx, mono, dense_product);
    const int sparse_majorana = codes_interleave_phase(mono_row.view(), gen_row.view());
    BOOST_TEST(sparse_majorana == dense_majorana);

    const auto pauli_ctx = PauliAlgebra::make_gen_context(gen);
    const int dense_pauli = PauliAlgebra::rotation_sign(pauli_ctx, mono, dense_product);
    const int sparse_pauli = codes_pauli_rotation_sign(mono_row.view(), gen_row.view());
    BOOST_TEST(sparse_pauli == dense_pauli);

    seen.nonzero_overlap |= fused.overlap > 0;
    seen.zero_overlap |= fused.overlap == 0;
    seen.majorana_minus |= dense_majorana < 0;
    seen.majorana_plus |= dense_majorana > 0;
    seen.pauli_minus |= dense_pauli < 0;
    seen.pauli_plus |= dense_pauli > 0;
    // A cancelling mode is the case a naive union would get wrong: fewer product slots than the union of
    // the two inputs' modes.
    size_t shared_cancelling = 0;
    for_each_mode_slot(mono, [&](size_t mode, unsigned int code) {
        const auto row = gen_row.view();
        for (size_t j = 0; j < row.num_slots(); ++j) {
            if (row.mode(j) == mode && row.code(j) == code) {
                ++shared_cancelling;
            }
        }
    });
    seen.cancelled_a_mode |= shared_cancelling > 0;
}

auto random_mono(std::mt19937_64 &rng, size_t num_modes, size_t max_modes) -> Bitset {
    Bitset mono(2 * num_modes);
    const size_t occupied = rng() % (max_modes + 1);
    for (size_t k = 0; k < occupied; ++k) {
        const size_t mode = rng() % num_modes;
        const auto code = 1U + static_cast<unsigned int>(rng() % 3U);
        if ((code & 1U) != 0U) {
            mono.set(2 * mode);
        }
        if ((code & 2U) != 0U) {
            mono.set((2 * mode) + 1);
        }
    }
    return mono;
}

} // namespace

// Randomized terms against randomized generators. Generators are drawn from the same distribution and
// deliberately overlap the terms, since a disjoint generator exercises neither the overlap count nor the
// cancelling-mode branch.
BOOST_AUTO_TEST_CASE(codes_product_matches_dense_on_randomized_rows) {
    std::mt19937_64 rng(20260812U);
    Seen seen;
    for (const size_t num_modes : {32U, 64U, 300U}) {
        for (size_t trial = 0; trial < 400; ++trial) {
            const auto mono = random_mono(rng, num_modes, 6);
            const auto gen = random_mono(rng, num_modes, 4);
            check_product(mono, gen, SparseRowStore::kMaxSlots, seen);
        }
    }
    BOOST_TEST(seen.cancelled_a_mode);
    BOOST_TEST(seen.nonzero_overlap);
    BOOST_TEST(seen.zero_overlap);
    BOOST_TEST(seen.majorana_minus);
    BOOST_TEST(seen.majorana_plus);
    BOOST_TEST(seen.pauli_minus);
    BOOST_TEST(seen.pauli_plus);
}

// Small mode counts, so terms and generators collide constantly: nearly every product goes through the
// equal-mode branch, and many modes cancel outright.
BOOST_AUTO_TEST_CASE(codes_product_matches_dense_under_heavy_overlap) {
    std::mt19937_64 rng(4242U);
    Seen seen;
    for (size_t trial = 0; trial < 2000; ++trial) {
        const auto mono = random_mono(rng, 6, 6);
        const auto gen = random_mono(rng, 6, 6);
        check_product(mono, gen, SparseRowStore::kMaxSlots, seen);
    }
    BOOST_TEST(seen.cancelled_a_mode);
    BOOST_TEST(seen.nonzero_overlap);
    BOOST_TEST(seen.majorana_minus);
    BOOST_TEST(seen.pauli_minus);
}

// Real generators and real terms: the fixtures' Majorana generator list against their Hamiltonian keys.
BOOST_AUTO_TEST_CASE(codes_product_matches_dense_on_fixture_generators) {
    Seen seen;
    size_t pairs = 0;
    for (const std::string name : {"random_exact.msgpack", "lih_fermionic_spin_exact.msgpack"}) {
        const auto data = test_utils::load_case_data(name);
        const size_t num_bits = 2 * data.num_modes;
        const size_t max_index = 2 * data.num_modes;

        std::vector<Bitset> terms;
        for (const auto &[inds, coeff] : data.hamiltonian) {
            if (inds.size() <= 12) {
                terms.push_back(indices_to_bitset_checked(inds, max_index, num_bits));
            }
        }
        std::vector<Bitset> gens;
        for (const auto &inds : data.majoranas) {
            if (inds.size() <= 12) {
                gens.push_back(indices_to_bitset_checked(inds, max_index, num_bits));
            }
        }
        BOOST_REQUIRE(!terms.empty());
        BOOST_REQUIRE(!gens.empty());

        const size_t stride = terms.size() > 40 ? (terms.size() / 40) + 1 : 1;
        for (size_t i = 0; i < terms.size(); i += stride) {
            for (const auto &gen : gens) {
                check_product(terms[i], gen, SparseRowStore::kMaxSlots, seen);
                ++pairs;
            }
        }
    }
    BOOST_TEST(pairs > 100U);
    BOOST_TEST(seen.nonzero_overlap);
    BOOST_TEST(seen.majorana_plus);
    BOOST_TEST(seen.pauli_plus);
}

// The product occupies up to the term's modes plus the generator's, which is why a scratch row is sized
// max_mode_bound() + generator locality. Past that the answer must be "overflowed", never a truncated
// mode list beside a plausible codes word -- that combination is what made a Stage 3 capacity bug read
// as a speedup.
BOOST_AUTO_TEST_CASE(codes_product_reports_capacity_overflow) {
    constexpr size_t kNumBits = 64;
    Bitset mono(kNumBits);
    Bitset gen(kNumBits);
    for (const size_t mode : {0U, 1U, 2U}) { // three disjoint modes each
        mono.set(2 * mode);
    }
    for (const size_t mode : {10U, 11U, 12U}) {
        gen.set(2 * mode);
    }
    const auto mono_row = OwnedRow::encode(mono, 8);
    const auto gen_row = OwnedRow::encode(gen, 8);

    // Six modes in the product, so five lanes is one short and six is exactly enough.
    for (const size_t capacity : {1U, 3U, 5U}) {
        std::vector<RowMode> lanes(capacity, 0);
        const auto product = sparse_toggle(mono_row.view(), gen_row.view(), lanes.data(), capacity);
        BOOST_TEST(product.overflowed);
        BOOST_TEST(product.codes == 0U);
        BOOST_TEST(product.num_slots == 0U);
    }
    std::vector<RowMode> lanes(6, 0);
    const auto product = sparse_toggle(mono_row.view(), gen_row.view(), lanes.data(), 6);
    BOOST_TEST(!product.overflowed);
    BOOST_TEST(product.num_slots == 6U);
    BOOST_TEST(product.overlap == 0U);

    // A cancelling term needs *fewer* lanes than the union, so capacity is about the product and not
    // about the inputs: gen against itself is empty.
    std::vector<RowMode> same(1, 0);
    const auto cancelled = sparse_toggle(gen_row.view(), gen_row.view(), same.data(), 1);
    BOOST_TEST(!cancelled.overflowed);
    BOOST_TEST(cancelled.num_slots == 0U);
    BOOST_TEST(cancelled.codes == 0U);
    BOOST_TEST(cancelled.overlap == 3U);
}
