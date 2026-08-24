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

// The scan's per-term kernel, sparse against dense, through the interface the scan actually calls:
// product() -> passes() -> owner() -> push(). codes_algebra_tests.cpp and codes_product_tests.cpp already
// pin the pieces against their dense counterparts; what this adds is the *composition* -- that
// SparseTermProducts routes a term to the right one of them, and that its three fallbacks (a spilled store
// row, a product past the scratch capacity, a cutoff with no codes form) produce the dense answer rather
// than a wrong one.
//
// This is the gate on Stage 6's store swap: with MPOperator::Store still OperatorIndex the sparse kernel
// is unreachable from the library, so nothing else would instantiate it.
//
// The width-bound dense kernel, DenseTermProductsW<A, W>, is the third answer to the same questions and
// is compared here for the same reason: it restates four of the five off raw words with the storage word
// count fixed at compile time, so nothing but a differential test can tell a restatement that agrees
// from one that merely runs. Its cases live at the bottom of the file; WordKernel<W>'s own primitives
// are pinned separately in word_kernel_tests.cpp.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/evolution/layer_build/TermProduct.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

#include "InlineWidths.h"
#include "TestData.h"
#include "TestUtilities.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

// Which branches a run exercised. Every case asserts on these: a fallback that silently swallowed every
// term would otherwise pass by comparing the dense kernel against itself.
struct Seen {
    size_t sparse_terms = 0;
    size_t fallback_terms = 0;
    size_t cutoff_passed = 0;
    size_t cutoff_failed = 0;
    size_t sparse_cutoff_failed = 0; // the codes cutoff said no, not the dense one
    size_t row_records = 0;          // survivors pushed as a support-form record
    size_t escaped_records = 0;      // survivors pushed as an escape plus a tail entry
};

auto random_term(std::mt19937_64 &rng, size_t num_modes, size_t max_modes) -> Bitset {
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

// A fully paired term of `modes` modes. These are the rows that make support unbounded -- xor_sum == 0
// escapes the cutoff -- so they are what drives a store row to spill and a product to overflow.
auto paired_term(size_t num_modes, size_t modes) -> Bitset {
    Bitset mono(2 * num_modes);
    for (size_t m = 0; m < modes; ++m) {
        mono.set(2 * m);
        mono.set((2 * m) + 1);
    }
    return mono;
}

// One term against one generator through both kernels, comparing every answer the scan reads. The two
// stores must hold the same monomial at index i, which the caller guarantees by inserting in lockstep.
template <typename A>
auto check_term(DenseTermProducts<A> &dense,
                SparseTermProducts<A> &sparse_kernel,
                const OperatorIndex &packed,
                const SparseRowStore &sparse_store,
                size_t i,
                size_t gen_pop,
                Seen &seen) -> void {
    const size_t mono_pop = packed.popcount(i);
    BOOST_REQUIRE(sparse_store.popcount(i) == mono_pop);

    const auto reference = dense.product(packed, i);
    const auto candidate = sparse_kernel.product(sparse_store, i);
    BOOST_TEST(candidate.overlap == reference.overlap);
    BOOST_TEST(candidate.phase_factor == reference.phase_factor);

    const size_t new_pop = mono_pop + gen_pop - (2 * reference.overlap);
    const bool reference_passes = dense.passes(new_pop);
    const bool candidate_passes = sparse_kernel.passes(new_pop);
    BOOST_TEST(candidate_passes == reference_passes);

    // Both remaining answers are read only for a surviving term, so ask them only there. owner() is still
    // the dense hash on both sides, because owner routing is monomial_hash everywhere.
    if (reference_passes) {
        for (const size_t rank_count : {2U, 3U, 8U}) {
            BOOST_TEST(sparse_kernel.owner(rank_count) == dense.owner(rank_count));
        }

        // The two records no longer agree byte for byte -- that is what the support form is for -- so what
        // is compared is what a resolver reads back out of them: the same key, and the same phase.
        const size_t num_bits = packed.num_bits();
        const size_t capacity = sparse_kernel.record_capacity();
        BOOST_TEST(sparse_kernel.record_words() == query_words(sparse_payload_words(capacity)));

        VecZ unused_escapes;
        VecZ reference_record = query_buffer();
        dense.push(QueryOut{reference_record, unused_escapes}, -1);
        BOOST_TEST(unused_escapes.empty()); // a dense record has nowhere to escape to and never needs one

        VecZ candidate_record = query_buffer();
        VecZ candidate_escapes;
        sparse_kernel.push(QueryOut{candidate_record, candidate_escapes}, -1);
        const bool escaped = !candidate_escapes.empty();
        BOOST_TEST(escaped == sparse_kernel.fell_back());
        append_escape_tail(candidate_record, candidate_escapes);
        seen.row_records += escaped ? 0 : 1;
        seen.escaped_records += escaped ? 1 : 0;

        DenseQueryKeys dense_keys;
        dense_keys.configure(num_bits, 0);
        dense_keys.ensure(1);
        dense_keys.begin_batch();
        BOOST_TEST(dense_keys.read_record(reference_record, 0, dense.record_words(), 0) == -1);

        SparseQueryKeys sparse_keys;
        sparse_keys.configure(num_bits, capacity);
        sparse_keys.ensure(1);
        sparse_keys.begin_batch();
        BOOST_TEST(sparse_keys.read_record(candidate_record, 0, sparse_kernel.record_words(), 0) == -1);
        BOOST_TEST(sparse_keys[0].is_spilled() == escaped);
        BOOST_TEST((key_monomial(sparse_keys[0], num_bits) == dense_keys[0]));
        // And the store agrees with the key either way, which is what the resolve's find_batch rests on.
        BOOST_TEST(sparse_row_hash(sparse_keys[0]) == sparse_row_hash(dense_keys[0]));
    }

    if (sparse_kernel.fell_back()) {
        ++seen.fallback_terms;
    }
    else {
        ++seen.sparse_terms;
        seen.sparse_cutoff_failed += reference_passes ? 0 : 1;
    }
    seen.cutoff_passed += reference_passes ? 1 : 0;
    seen.cutoff_failed += reference_passes ? 0 : 1;
}

// Every term of `terms` against every generator of `gens`, over one algebra and one cutoff.
template <typename A>
auto sweep(const std::vector<Bitset> &terms,
           const std::vector<Bitset> &gens,
           const CutoffFn &cutoff_fn,
           size_t sparse_slots,
           Seen &seen) -> void {
    BOOST_REQUIRE(!terms.empty());
    const size_t num_bits = terms.front().size();
    OperatorIndex packed(num_bits);
    SparseRowStore sparse_store(num_bits, sparse_slots);
    for (const auto &mono : terms) {
        packed.push_back(mono);
        sparse_store.push_back(mono);
    }

    const CutoffEvaluator cutoff_eval{cutoff_fn};
    for (const auto &gen : gens) {
        DenseTermProducts<A> dense(gen, cutoff_eval);
        SparseTermProducts<A> sparse_kernel(gen, cutoff_eval);
        const size_t gen_pop = gen.count();
        for (size_t i = 0; i < terms.size(); ++i) {
            check_term(dense, sparse_kernel, packed, sparse_store, i, gen_pop, seen);
        }
    }
}

} // namespace

// Randomized terms and generators at three widths, both algebras, both cutoff kinds. Slots are generous
// enough that no row spills, so what is under test is the sparse path itself.
BOOST_AUTO_TEST_CASE(term_product_sparse_kernel_matches_dense_on_randomized_rows) {
    std::mt19937_64 rng(20260812U);
    Seen seen;
    for (const size_t num_modes : {32U, 64U, 300U}) {
        std::vector<Bitset> terms;
        for (size_t t = 0; t < 150; ++t) {
            terms.push_back(random_term(rng, num_modes, 6));
        }
        std::vector<Bitset> gens;
        for (size_t t = 0; t < 8; ++t) {
            gens.push_back(random_term(rng, num_modes, 4));
        }
        for (const unsigned int cutoff : {4U, 8U}) {
            const auto length = cutoff_function(CutoffType::Length, cutoff, num_modes, 2 * num_modes);
            const auto support = cutoff_function(CutoffType::Support, cutoff, num_modes, 2 * num_modes);
            sweep<MajoranaAlgebra>(terms, gens, length, SparseRowStore::kMaxSlots, seen);
            sweep<MajoranaAlgebra>(terms, gens, support, SparseRowStore::kMaxSlots, seen);
            sweep<PauliAlgebra>(terms, gens, support, SparseRowStore::kMaxSlots, seen);
            sweep<PauliAlgebra>(terms, gens, length, SparseRowStore::kMaxSlots, seen);
        }
    }
    // Most terms take the sparse path, and some do not: the terms are drawn up to 6 modes wide against a
    // cutoff of 4 or 8, and a term above the bound overflows a capacity that is sized from that bound.
    // Which is the real shape of the thing -- a stored row exceeds the bound whenever it is fully paired.
    BOOST_TEST(seen.sparse_terms > 20000U);
    BOOST_TEST(seen.fallback_terms > 0U);
    BOOST_TEST(seen.cutoff_passed > 0U);
    // The point of the codes cutoff is rejecting a product without materializing it, so a run where
    // nothing was rejected would not have tested it.
    BOOST_TEST(seen.sparse_cutoff_failed > 0U);
    // Every survivor here was pushed as a row: an overflowing product is wider than the cutoff's bound, so
    // unless it is fully paired the cutoff rejects it before a record is ever asked for. A *surviving*
    // escape needs a fully paired product, which has a case of its own below.
    BOOST_TEST(seen.row_records > 0U);
    BOOST_TEST(seen.escaped_records == 0U);
}

// Real terms and generators: the fixtures' Hamiltonian keys against their Majorana generator list, which
// is where the products and the ordering signs are the ones the engine actually computes.
BOOST_AUTO_TEST_CASE(term_product_sparse_kernel_matches_dense_on_fixture_generators) {
    Seen seen;
    for (const std::string name : {"random_exact.msgpack", "lih_fermionic_spin_exact.msgpack"}) {
        const auto data = test_utils::load_case_data<0>(name);
        const size_t num_bits = 2 * data.num_modes;
        const size_t max_index = 2 * data.num_modes;

        std::vector<Bitset> terms;
        for (const auto &[inds, coeff] : data.hamiltonian) {
            terms.push_back(indices_to_bitset_checked(inds, max_index, num_bits));
        }
        std::vector<Bitset> gens;
        for (const auto &inds : data.majoranas) {
            gens.push_back(indices_to_bitset_checked(inds, max_index, num_bits));
        }
        BOOST_REQUIRE(!terms.empty());
        BOOST_REQUIRE(!gens.empty());
        // A stride, not the whole cross product: the fixtures are large and the pairs are homogeneous.
        std::vector<Bitset> sampled_gens;
        const size_t stride = gens.size() > 12 ? (gens.size() / 12) + 1 : 1;
        for (size_t g = 0; g < gens.size(); g += stride) {
            sampled_gens.push_back(gens[g]);
        }

        for (const unsigned int cutoff : {4U, 6U}) {
            sweep<MajoranaAlgebra>(terms,
                                   sampled_gens,
                                   cutoff_function(CutoffType::Length, cutoff, data.num_modes, num_bits),
                                   SparseRowStore::kMaxSlots,
                                   seen);
            sweep<PauliAlgebra>(terms,
                                sampled_gens,
                                cutoff_function(CutoffType::Support, cutoff, data.num_modes, num_bits),
                                SparseRowStore::kMaxSlots,
                                seen);
        }
    }
    // No claim that nothing fell back: a wide fixture term against a wide generator legitimately overflows
    // a capacity of cutoff + |G|, and that case is covered on its own below.
    BOOST_TEST(seen.sparse_terms > 100U);
    BOOST_TEST(seen.cutoff_passed > 0U);
    BOOST_TEST(seen.sparse_cutoff_failed > 0U);
}

// A store row with no view: the kernel must take the dense path for that term alone and keep taking the
// sparse one for the rest. Rows are a mix, so both happen in the same gate.
BOOST_AUTO_TEST_CASE(term_product_falls_back_on_a_spilled_row) {
    std::mt19937_64 rng(99U);
    constexpr size_t kNumModes = 64;
    std::vector<Bitset> terms;
    for (size_t t = 0; t < 60; ++t) {
        terms.push_back(random_term(rng, kNumModes, 3));
        terms.push_back(paired_term(kNumModes, 9)); // 9 modes > the 4 slots below
    }
    std::vector<Bitset> gens{random_term(rng, kNumModes, 2), random_term(rng, kNumModes, 4)};

    Seen seen;
    sweep<MajoranaAlgebra>(terms, gens, cutoff_function(CutoffType::Length, 6, kNumModes, 2 * kNumModes), 4, seen);
    sweep<PauliAlgebra>(terms, gens, cutoff_function(CutoffType::Support, 6, kNumModes, 2 * kNumModes), 4, seen);
    BOOST_TEST(seen.fallback_terms > 0U);
    BOOST_TEST(seen.sparse_terms > 0U);
}

// A product past the scratch capacity, with the store row itself perfectly representable: capacity is
// max_mode_bound() + the generator's locality, so a fully paired row well above the cutoff overflows it.
// sparse_toggle reports that rather than truncating, and the kernel must then answer densely.
BOOST_AUTO_TEST_CASE(term_product_falls_back_on_a_capacity_overflow) {
    std::mt19937_64 rng(1010U);
    constexpr size_t kNumModes = 64;
    std::vector<Bitset> terms;
    for (size_t t = 0; t < 40; ++t) {
        terms.push_back(random_term(rng, kNumModes, 3));
        // 12 modes: inside the 20-slot store rows below, past a capacity of 4 + |G|.
        terms.push_back(paired_term(kNumModes, 12));
    }
    std::vector<Bitset> gens{random_term(rng, kNumModes, 2)};

    Seen seen;
    sweep<MajoranaAlgebra>(terms, gens, cutoff_function(CutoffType::Length, 4, kNumModes, 2 * kNumModes), 20, seen);
    sweep<PauliAlgebra>(terms, gens, cutoff_function(CutoffType::Support, 4, kNumModes, 2 * kNumModes), 20, seen);
    BOOST_TEST(seen.fallback_terms > 0U);
    BOOST_TEST(seen.sparse_terms > 0U);
    BOOST_TEST(seen.cutoff_passed > 0U);
}

// The escape's own case, and the reason no capacity can settle the question: a fully paired product is kept
// unconditionally (xor_sum == 0), so a product both wider than the record and kept by the cutoff exists by
// construction. The generator is fully paired on modes the term already holds, so the product is the term
// minus those modes -- still fully paired, still far wider than a capacity of bound + |G|.
BOOST_AUTO_TEST_CASE(term_product_escapes_a_surviving_product_no_record_can_hold) {
    constexpr size_t kNumModes = 64;
    const std::vector<Bitset> terms{paired_term(kNumModes, 12)};
    Bitset gen(2 * kNumModes);
    for (const size_t m : {3U, 4U}) {
        gen.set(2 * m);
        gen.set((2 * m) + 1);
    }
    const std::vector<Bitset> gens{gen};

    Seen seen;
    sweep<MajoranaAlgebra>(terms, gens, cutoff_function(CutoffType::Length, 4, kNumModes, 2 * kNumModes), 20, seen);
    sweep<PauliAlgebra>(terms, gens, cutoff_function(CutoffType::Support, 4, kNumModes, 2 * kNumModes), 20, seen);
    BOOST_TEST(seen.fallback_terms == 2U);
    BOOST_TEST(seen.sparse_terms == 0U);
    BOOST_TEST(seen.cutoff_passed == 2U);
    BOOST_TEST(seen.escaped_records == 2U);
    BOOST_TEST(seen.row_records == 0U);
}

// A logical width narrower than the storage width, which is what storage_modes_for() produces for any
// mode count that is not a whole 32-mode block. The inactive modes are the low ones, so the codes cutoff
// drops them as a prefix of the ascending slots where the dense one shifts the whole register -- and the
// terms here deliberately occupy modes on both sides of that boundary.
BOOST_AUTO_TEST_CASE(term_product_sparse_kernel_matches_dense_in_a_narrow_active_window) {
    std::mt19937_64 rng(7070U);
    constexpr size_t kStorageModes = 64;
    constexpr size_t kLogicalModes = 50; // an inactive prefix of 14 modes
    std::vector<Bitset> terms;
    for (size_t t = 0; t < 120; ++t) {
        terms.push_back(random_term(rng, kStorageModes, 6));
    }
    std::vector<Bitset> gens{random_term(rng, kStorageModes, 3), random_term(rng, kStorageModes, 2)};

    Seen seen;
    sweep<MajoranaAlgebra>(terms,
                           gens,
                           cutoff_function(CutoffType::Length, 4, kLogicalModes, 2 * kStorageModes),
                           SparseRowStore::kMaxSlots,
                           seen);
    sweep<PauliAlgebra>(terms,
                        gens,
                        cutoff_function(CutoffType::Support, 4, kLogicalModes, 2 * kStorageModes),
                        SparseRowStore::kMaxSlots,
                        seen);
    BOOST_TEST(seen.sparse_terms > 0U);
    BOOST_TEST(seen.cutoff_passed > 0U);
    BOOST_TEST(seen.sparse_cutoff_failed > 0U);
}

// A cutoff that is neither concrete functor -- the basis-change form, which is a lambda on purpose -- has
// no codes counterpart, so the whole gate falls back. Asserted because the alternative to falling back is
// answering with the wrong cutoff, which no other test would catch.
BOOST_AUTO_TEST_CASE(term_product_falls_back_when_the_cutoff_has_no_codes_form) {
    std::mt19937_64 rng(2020U);
    constexpr size_t kNumModes = 32;
    std::vector<Bitset> terms;
    for (size_t t = 0; t < 80; ++t) {
        terms.push_back(random_term(rng, kNumModes, 5));
    }
    std::vector<Bitset> gens{random_term(rng, kNumModes, 3)};

    // The identity basis, so the predicate is an ordinary length cutoff wrapped in a lambda: the answers
    // must still match, and every term must have gone the dense way to produce them.
    MonomialList basis;
    for (size_t b = 0; b < 2 * kNumModes; ++b) {
        Bitset single(2 * kNumModes);
        single.set(b);
        basis.push_back(single);
    }
    const auto wrapped = cutoff_function_basis_change(CutoffType::Length, 4, basis, kNumModes);
    BOOST_REQUIRE(CutoffEvaluator{wrapped}.length_cutoff() == nullptr);

    Seen seen;
    sweep<MajoranaAlgebra>(terms, gens, wrapped, SparseRowStore::kMaxSlots, seen);
    BOOST_TEST(seen.sparse_terms == 0U);
    BOOST_TEST(seen.fallback_terms == terms.size());
}

// A generator wider than one codes word cannot be encoded as a row at all, so the gate falls back
// wholesale. Exotic, but the branch exists and an unencodable generator must not be silently truncated
// into a *different* generator.
BOOST_AUTO_TEST_CASE(term_product_falls_back_on_a_generator_past_one_codes_word) {
    std::mt19937_64 rng(3030U);
    constexpr size_t kNumModes = 128;
    std::vector<Bitset> terms;
    for (size_t t = 0; t < 40; ++t) {
        terms.push_back(random_term(rng, kNumModes, 4));
    }
    std::vector<Bitset> gens{paired_term(kNumModes, SparseRowStore::kMaxSlots + 5)};

    Seen seen;
    sweep<MajoranaAlgebra>(terms, gens, cutoff_function(CutoffType::Length, 6, kNumModes, 2 * kNumModes), 40, seen);
    BOOST_TEST(seen.sparse_terms == 0U);
    BOOST_TEST(seen.fallback_terms == terms.size());
}

// ---------------------------------------------------------------------------------------------------
// The width-bound dense kernel against the width-agnostic one.
// ---------------------------------------------------------------------------------------------------

namespace {

// Which of the two cutoff paths a narrow-kernel run took, per term. Both have to occur across the
// cases below or one of them ships compared against nothing.
struct NarrowSeen {
    size_t terms = 0;
    size_t word_cutoff_terms = 0; // passes() answered off the words
    size_t evaluator_terms = 0;   // passes() went through the CutoffEvaluator
    size_t passed = 0;
    size_t failed = 0;
    size_t paired_rescues = 0; // kept although longer than the cutoff, i.e. the fully-paired clause
};

// One term through both dense kernels, comparing every answer the scan reads plus the product itself.
// Unlike the sparse comparison the records must agree *byte for byte*: both push a dense monomial, so
// any difference here is a difference in the product.
template <typename A, size_t W>
auto check_narrow_term(DenseTermProducts<A> &reference_kernel,
                       DenseTermProductsW<A, W> &candidate_kernel,
                       const OperatorIndex &packed,
                       size_t i,
                       size_t gen_pop,
                       unsigned int cutoff,
                       NarrowSeen &seen) -> void {
    const auto reference = reference_kernel.product(packed, i);
    const auto candidate = candidate_kernel.product(packed, i);
    BOOST_TEST(candidate.overlap == reference.overlap);
    BOOST_TEST(candidate.phase_factor == reference.phase_factor);
    BOOST_TEST((candidate_kernel.product_row() == reference_kernel.product_row()));
    BOOST_TEST(candidate_kernel.record_words() == reference_kernel.record_words());

    const size_t new_pop = packed.popcount(i) + gen_pop - (2 * reference.overlap);
    const bool reference_passes = reference_kernel.passes(new_pop);
    BOOST_TEST(candidate_kernel.passes(new_pop) == reference_passes);

    if (reference_passes) {
        for (const size_t rank_count : {2U, 3U, 8U}) {
            BOOST_TEST(candidate_kernel.owner(rank_count) == reference_kernel.owner(rank_count));
        }

        VecZ unused_escapes;
        VecZ reference_record = query_buffer();
        reference_kernel.push(QueryOut{reference_record, unused_escapes}, -1);
        VecZ candidate_record = query_buffer();
        candidate_kernel.push(QueryOut{candidate_record, unused_escapes}, -1);
        BOOST_TEST(candidate_record == reference_record, boost::test_tools::per_element());
        BOOST_TEST(unused_escapes.empty()); // neither dense kernel has anywhere to escape to
        seen.paired_rescues += new_pop > cutoff ? 1 : 0;
    }

    ++seen.terms;
    seen.word_cutoff_terms += candidate_kernel.uses_word_cutoff() ? 1 : 0;
    seen.evaluator_terms += candidate_kernel.uses_word_cutoff() ? 0 : 1;
    seen.passed += reference_passes ? 1 : 0;
    seen.failed += reference_passes ? 0 : 1;
}

// Every term of `terms` against every generator of `gens`, at the one width W the terms are built for.
template <typename A, size_t W>
auto sweep_narrow(const std::vector<Bitset> &terms,
                  const std::vector<Bitset> &gens,
                  const CutoffFn &cutoff_fn,
                  unsigned int cutoff,
                  NarrowSeen &seen) -> void {
    BOOST_REQUIRE(!terms.empty());
    const size_t num_bits = terms.front().size();
    BOOST_REQUIRE(num_bits == W * Bitset::word_width);
    OperatorIndex packed(num_bits);
    for (const auto &mono : terms) {
        packed.push_back(mono);
    }

    const CutoffEvaluator cutoff_eval{cutoff_fn};
    for (const auto &gen : gens) {
        DenseTermProducts<A> reference_kernel(gen, cutoff_eval);
        DenseTermProductsW<A, W> candidate_kernel(gen, cutoff_eval);
        const size_t gen_pop = gen.count();
        for (size_t i = 0; i < terms.size(); ++i) {
            check_narrow_term<A, W>(reference_kernel, candidate_kernel, packed, i, gen_pop, cutoff, seen);
        }
    }
}

// The W the dispatch bound, read back out of the arm it selected -- the only thing about the seam that
// is observable, since every arm answers identically.
template <typename Store>
auto bound_kernel_width(size_t num_words) -> size_t {
    return with_kernel_width<Store>(num_words, []<size_t W>(std::integral_constant<size_t, W>) -> size_t { return W; });
}

// Terms wide enough to be rejected, narrow enough to survive, and some fully paired well above the
// cutoff -- which is the clause the word cutoff answers with its own fold rather than a popcount.
auto narrow_kernel_terms(std::mt19937_64 &rng, size_t num_modes) -> std::vector<Bitset> {
    std::vector<Bitset> terms;
    for (size_t t = 0; t < 60; ++t) {
        terms.push_back(random_term(rng, num_modes, 7));
    }
    for (const size_t paired : {1U, 3U, 8U}) {
        terms.push_back(paired_term(num_modes, paired));
    }
    return terms;
}

} // namespace

// A length cutoff over the whole register: the one shape the word cutoff answers, so this is the case
// where the specialized path is actually under test. Asserted through uses_word_cutoff(), because a
// change that quietly stopped engaging it would leave every other assertion here still passing.
BOOST_AUTO_TEST_CASE(narrow_kernel_matches_dense_under_a_whole_register_length_cutoff) {
    std::mt19937_64 rng(20260821U);
    NarrowSeen seen;
    test_utils::for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        const size_t num_modes = (W * Bitset::word_width) / 2;
        const auto terms = narrow_kernel_terms(rng, num_modes);
        const std::vector<Bitset> gens{random_term(rng, num_modes, 2),
                                       random_term(rng, num_modes, 4),
                                       paired_term(num_modes, 2)};
        for (const unsigned int cutoff : {4U, 8U}) {
            const auto length = cutoff_function(CutoffType::Length, cutoff, num_modes, 2 * num_modes);
            sweep_narrow<MajoranaAlgebra, W>(terms, gens, length, cutoff, seen);
            // The Pauli algebra with a length cutoff: the kind and the basis are independent here even
            // though the shipping models pair them, and the kernel's cutoff arm must not depend on the
            // algebra it was instantiated with.
            sweep_narrow<PauliAlgebra, W>(terms, gens, length, cutoff, seen);
        }
    });
    BOOST_TEST(seen.terms > 0U);
    BOOST_TEST(seen.evaluator_terms == 0U); // every term took the word cutoff
    BOOST_TEST(seen.word_cutoff_terms == seen.terms);
    BOOST_TEST(seen.passed > 0U);
    BOOST_TEST(seen.failed > 0U);
    // Products kept although longer than the cutoff, i.e. the fully-paired fold answering yes. Without
    // these the word cutoff would be tested as nothing but `new_pop <= cutoff`.
    BOOST_TEST(seen.paired_rescues > 0U);
}

// The two shapes the word cutoff declines: a support cutoff, and a length cutoff whose active window is
// narrower than the storage register -- which is what storage_modes_for() produces for any mode count
// that is not a whole 32-mode block. Both must fall through to the evaluator and still agree; the
// failure this guards against is answering a *different* cutoff, which changes which terms survive.
BOOST_AUTO_TEST_CASE(narrow_kernel_defers_to_the_evaluator_off_the_whole_register_length_cutoff) {
    std::mt19937_64 rng(606U);
    NarrowSeen seen;
    test_utils::for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        const size_t num_modes = (W * Bitset::word_width) / 2;
        const size_t logical_modes = num_modes - 5; // an inactive prefix of 5 modes
        const auto terms = narrow_kernel_terms(rng, num_modes);
        const std::vector<Bitset> gens{random_term(rng, num_modes, 3), random_term(rng, num_modes, 2)};
        for (const unsigned int cutoff : {4U, 6U}) {
            const auto support = cutoff_function(CutoffType::Support, cutoff, num_modes, 2 * num_modes);
            const auto narrow = cutoff_function(CutoffType::Length, cutoff, logical_modes, 2 * num_modes);
            sweep_narrow<MajoranaAlgebra, W>(terms, gens, support, cutoff, seen);
            sweep_narrow<PauliAlgebra, W>(terms, gens, support, cutoff, seen);
            sweep_narrow<MajoranaAlgebra, W>(terms, gens, narrow, cutoff, seen);
        }
    });
    BOOST_TEST(seen.terms > 0U);
    BOOST_TEST(seen.word_cutoff_terms == 0U);
    BOOST_TEST(seen.evaluator_terms == seen.terms);
    BOOST_TEST(seen.passed > 0U);
    BOOST_TEST(seen.failed > 0U);
}

// A cutoff that is neither concrete functor, so CutoffEvaluator recovered nothing: the kernel keeps its
// word product and defers the whole predicate. Same case as the sparse kernel's, for the same reason.
BOOST_AUTO_TEST_CASE(narrow_kernel_defers_when_the_cutoff_has_no_concrete_functor) {
    std::mt19937_64 rng(707U);
    constexpr size_t kWords = 2;
    constexpr size_t kNumModes = (kWords * Bitset::word_width) / 2;
    const auto terms = narrow_kernel_terms(rng, kNumModes);
    const std::vector<Bitset> gens{random_term(rng, kNumModes, 3)};

    MonomialList basis;
    for (size_t b = 0; b < 2 * kNumModes; ++b) {
        Bitset single(2 * kNumModes);
        single.set(b);
        basis.push_back(single);
    }
    const auto wrapped = cutoff_function_basis_change(CutoffType::Length, 4, basis, kNumModes);
    BOOST_REQUIRE(CutoffEvaluator{wrapped}.length_cutoff() == nullptr);

    NarrowSeen seen;
    sweep_narrow<MajoranaAlgebra, kWords>(terms, gens, wrapped, 4, seen);
    BOOST_TEST(seen.terms == terms.size());
    BOOST_TEST(seen.word_cutoff_terms == 0U);
    BOOST_TEST(seen.passed > 0U);
    BOOST_TEST(seen.failed > 0U);
}

// The seam itself: which W the scan binds for a given storage width and store. Pinned because it is the
// one thing above that no differential test can see -- every arm computes the same answers, so a
// dispatch that always chose 0 would leave the whole suite green and only the benchmark different.
BOOST_AUTO_TEST_CASE(with_kernel_width_binds_the_capped_storage_word_count) {
    for (size_t words = 1; words <= Bitset::kInlineWords; ++words) {
        BOOST_TEST(bound_kernel_width<OperatorIndex>(words) == (words <= kNarrowKernelWords ? words : 0));
        // The sparse store is never specialized: its per-term work is O(slots), not O(storage words).
        BOOST_TEST(bound_kernel_width<SparseRowStore>(words) == 0U);
    }
    // Above the inline regime the words are on the heap, so there is no width to bind.
    BOOST_TEST(bound_kernel_width<OperatorIndex>(Bitset::kInlineWords + 1) == 0U);
}
