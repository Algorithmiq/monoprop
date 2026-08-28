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

// The scan's per-term kernel. For one anticommuting term it answers five questions -- the product M(+)G,
// the overlap the emitted phase needs, the basis rotation sign, whether the product survives the
// structural cutoff, and (for a survivor) its owner rank and query record -- and those five are exactly
// what changes when a row stops being a bitset. So they are gathered behind one per-gate object, chosen
// off the store type by TermProductsFor, rather than spread over the scan's emit lambda.
//
// The call sequence per term is product() -> passes() -> owner() -> push(), each reading the product the
// previous left. That is stateful on purpose: the product is per-gate scratch, since a term must not pay
// to construct the storage its product goes into (see the note on the scratch monomials below).

#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/CodesAlgebra.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

namespace monoprop::detail {

// phase_factor is the basis-specific sign only: Majorana interleave_phase, still to be folded with
// hermitian_phase at emit; Pauli pauli_rotation_sign, already rotation-ready.
// ham and the two monomials are deduced from their argument types; A stays the sole explicit template
// argument at call sites, same as before.
//
// `mono` and `new_mono` are scratch owned by the caller for the whole gate, not locals: both are
// overwritten whole here, and a term must not pay for constructing them. Constructing a Bitset means
// deriving the word count from a runtime width, testing it against the inline capacity and -- above
// that capacity -- allocating; with a compile-time width all of that used to fold away to nothing, so
// the scratch is what keeps it off the per-term path.
template <Algebra A>
[[gnu::always_inline]] inline auto emit_term_products(const auto &ham,
                                                      size_t i,
                                                      const typename A::GenContext &ctx,
                                                      MonomialLike auto &mono,
                                                      MonomialLike auto &new_mono,
                                                      size_t &overlap,
                                                      int &phase_factor) -> void {
    const auto &gen = A::generator(ctx);
    // for_each_position only sets bits, so the scratch has to start clear; reset() keeps the width,
    // unlike assigning a default-constructed Bitset.
    mono.reset();
    for_each_row_position(ham, i, [&mono](size_t pos) { mono.set(pos); });
    // One pass instead of two: mono ^ gen and popcount(mono & gen) are both always needed here, so
    // fused_xor_into() computes them together, straight into new_mono (see Bitset::fused_xor_into).
    // Its result_count (popcount of the XOR) goes unused -- the caller already has new_pop for free
    // via mono_pop + gen_pop - 2*overlap -- so it is not threaded through here.
    overlap = mono.fused_xor_into(gen, new_mono).overlap;
    phase_factor = A::rotation_sign(ctx, mono, new_mono);
}

// What one term's product yields. The product's popcount is not here: the caller gets it for free as
// mono_pop + gen_pop - 2*overlap, where the product itself would have to count it.
struct TermProduct {
    size_t overlap = 0;
    int phase_factor = 0;
};

// Dense rows -- the representation the engine has always used, and the reference the sparse one below
// must match term for term.
template <Algebra A>
class DenseTermProducts {
public:
    // gen fixes the width of both scratch monomials, and it is the operator's storage width, so every
    // word op below stays on Bitset's matched-width path. cutoff_eval is held by reference: it borrows
    // the caller's CutoffFn already, so it outlives no less than this does.
    DenseTermProducts(const Bitset &gen, const CutoffEvaluator &cutoff_eval)
        : ctx_(A::make_gen_context(gen)),
          cutoff_(&cutoff_eval),
          mono_(gen.size()),
          new_mono_(gen.size()) {}

    template <typename Store>
    [[gnu::always_inline]] auto product(const Store &store, size_t i) -> TermProduct {
        TermProduct out;
        emit_term_products<A>(store, i, ctx_, mono_, new_mono_, out.overlap, out.phase_factor);
        return out;
    }

    [[nodiscard]] auto passes(size_t new_pop) const -> bool {
        return cutoff_->passes_with_popcount(new_mono_, new_pop);
    }
    // find_rank's expression, minus its n_ranks == 0 guard, which the scan's rank_count > 1
    // short-circuit already covers. Owner routing is monomial_hash everywhere, including that initial
    // distribution, so this must not become anything else.
    [[nodiscard]] auto owner(size_t rank_count) const -> size_t { return monomial_hash(new_mono_) % rank_count; }
    auto push(QueryOut out, int phase) const -> void { query_push(out.records, new_mono_, phase); }

    // Record width for the per-rank query reserves, which run before the first product.
    [[nodiscard]] auto record_words() const -> size_t { return query_words(new_mono_.num_words()); }

    // The product monomial, for the sparse emitter's fallback and for the differential tests.
    [[nodiscard]] auto product_row() const -> const Bitset & { return new_mono_; }

private:
    typename A::GenContext ctx_;
    const CutoffEvaluator *cutoff_;
    Bitset mono_;
    Bitset new_mono_;
};

// Thrown by DenseTermProductsW's constructor: with_kernel_width picks W from the store's row word
// count and the generator's own width is independent of that (a stale generator bank, a basis
// change that resized the store but not the gates, ...), so unlike the per-term word ops in
// Bitset.h -- deliberately assert-only, since Release must keep their loops bare -- this binding
// happens once per gate. A real branch there costs nothing next to the per-term work it guards, so
// it stays a check even in Release rather than silently reading past W words of gen/mono/new_mono.
class KernelWidthMismatch : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The dense kernel with the storage word count bound at compile time, chosen once per gate by the
// scan (see with_kernel_width). Answers exactly what DenseTermProducts answers, in the same order
// and to the same values -- what differs is that every word loop inside has a known trip count and
// every operand's storage pointer is resolved once here instead of on each access.
//
// Why this is a separate class rather than a W parameter on DenseTermProducts: only the hot answers
// are worth specializing, and a fallback is still needed for W outside the inline regime, where the
// kernel's inline-operand precondition does not hold.
//
// The cutoff is specialized only for a length cutoff over the whole register. Both other cases -- a
// support cutoff, or an active window narrower than the storage width -- keep going through the
// evaluator. Not for lack of trying: a support arm folding or_sum the same way was measured and cost
// about 1% everywhere, gaining nothing even on the Pauli models that use it, because their per-term
// time is not in the cutoff. A narrow window would need a third kernel, and getting its shift wrong
// would silently change which terms survive.
template <Algebra A, size_t W>
class DenseTermProductsW {
public:
    DenseTermProductsW(const Bitset &gen, const CutoffEvaluator &cutoff_eval)
        : ctx_(A::make_gen_context(gen)),
          cutoff_(&cutoff_eval),
          mono_(gen.size()),
          new_mono_(gen.size()),
          gen_words_(A::generator(ctx_).data()),
          mono_words_(mono_.data()),
          new_words_(new_mono_.data()) {
        if (gen.num_words() != W) {
            throw KernelWidthMismatch(
                std::format("DenseTermProductsW<W={}> bound against a generator of {} words; the kernel's W must "
                            "be the generator's word count.",
                            W,
                            gen.num_words()));
        }
        if (const auto *length = cutoff_eval.length_cutoff(); length != nullptr && length->masks.whole_register()) {
            length_cutoff_ = length->cutoff;
        }
    }

    // The word pointers below point into this object's own bitsets, so a copy would leave the copy
    // reading and writing the original's storage. Nothing copies this -- it is the scan's per-gate
    // local -- so the case is made unrepresentable rather than documented.
    DenseTermProductsW(const DenseTermProductsW &) = delete;
    auto operator=(const DenseTermProductsW &) -> DenseTermProductsW & = delete;
    DenseTermProductsW(DenseTermProductsW &&) = delete;
    auto operator=(DenseTermProductsW &&) -> DenseTermProductsW & = delete;

    template <typename Store>
    [[gnu::always_inline]] auto product(const Store &store, size_t i) -> TermProduct {
        WordKernel<W>::clear(mono_words_);
        // Straight to the words: Bitset::set would re-select the storage pointer for every set bit,
        // and a row carries one per surviving slot.
        for_each_row_position(store, i, [this](size_t pos) {
            mono_words_[pos / Bitset::word_width] |= uint64_t{1} << (pos % Bitset::word_width);
        });
        const auto counts = WordKernel<W>::fused_xor_into(mono_words_, gen_words_, new_words_);
        return {counts.overlap, A::template rotation_sign_words<W>(ctx_, mono_words_, new_words_)};
    }

    [[nodiscard]] auto passes(size_t new_pop) const -> bool {
        if (length_cutoff_) {
            // Same two clauses as CutoffEvaluator::passes_with_popcount for a length cutoff, in the
            // same order: the popcount test proves keep without reading the monomial, and the paired
            // test is the xor_sum == 0 clause that rescues a fully paired term of any length.
            return new_pop <= *length_cutoff_ || fully_paired_words<W>(new_words_);
        }
        return cutoff_->passes_with_popcount(new_mono_, new_pop);
    }
    [[nodiscard]] auto owner(size_t rank_count) const -> size_t {
        return WordKernel<W>::splitmix(new_words_) % rank_count;
    }
    auto push(QueryOut out, int phase) const -> void { query_push(out.records, new_mono_, phase); }

    [[nodiscard]] auto record_words() const -> size_t { return query_words(W); }
    [[nodiscard]] auto product_row() const -> const Bitset & { return new_mono_; }

    // Whether passes() answers off the words or through the evaluator. Exists for the differential
    // tests, which otherwise cannot tell a run that exercised the word cutoff from one that compared
    // the evaluator against itself -- the same reason SparseTermProducts::fell_back() is observable.
    [[nodiscard]] auto uses_word_cutoff() const -> bool { return length_cutoff_.has_value(); }

private:
    typename A::GenContext ctx_;
    const CutoffEvaluator *cutoff_;
    Bitset mono_;
    Bitset new_mono_;
    // Resolved once, in the constructor's order: each is the data() of a member above, which does not
    // move for this object's lifetime because no member below is ever resized or reassigned.
    const uint64_t *gen_words_;
    uint64_t *mono_words_;
    uint64_t *new_words_;
    std::optional<unsigned int> length_cutoff_ = std::nullopt;
};

// The slot capacity a support-form query record is cut to, which is also the scan's scratch product
// capacity -- a query carries exactly such a product, so one number has to serve both or a product that
// fits the scratch would not fit the record. A product occupies at most the cutoff's mode bound plus the
// generator's own modes; an absent bound means the cutoff has no codes form and no term will reach the
// toggle, but the row is sized anyway so the capacity is never zero.
//
// Every rank derives this from the same circuit and cutoff, which is what lets it fix a wire stride with no
// communication -- the same agreement find_rank already needs for the hash width.
[[nodiscard]] inline auto sparse_record_capacity(const Bitset &gen, const CutoffEvaluator &cutoff_eval) -> size_t {
    return SparseRowStore::scratch_slots_for(cutoff_eval.max_mode_bound().value_or(SparseRowStore::kMaxSlots),
                                             occupied_mode_count(gen));
}

// Support-form rows. The five answers split three ways: product, overlap and rotation sign come off the
// codes word and the two mode lists (sparse_toggle plus A::codes_rotation_sign, O(slots) where the dense
// form is O(storage words)); the cutoff is a popcount test that reads the row only when the bound is
// exceeded; and push() writes the row itself, so a term that survives touches a storage word only when it
// escaped the sparse form or when owner() needs the dense hash at R>1. A term the cutoff rejects touches
// none at all, which is the whole point.
//
// Three cases fall back to the dense kernel for that term, none of them rare enough to assert away: a
// spilled store row (no view exists), a product past the scratch capacity (sparse_toggle reports it
// rather than truncating), and a generator too wide for one codes word. A cutoff that is neither of the
// two concrete functors has no codes form either, and falls back for every term of the gate.
template <Algebra A>
class SparseTermProducts {
public:
    SparseTermProducts(const Bitset &gen, const CutoffEvaluator &cutoff_eval)
        : fallback_(gen, cutoff_eval),
          dense_(gen.size()) {
        gen_lanes_.reserve(SparseRowStore::kMaxSlots);
        bool gen_fits = true;
        for_each_mode_slot(gen, [this, &gen_fits](size_t mode, unsigned int code) {
            if (gen_lanes_.size() == SparseRowStore::kMaxSlots) {
                gen_fits = false;
                return;
            }
            gen_codes_ |= static_cast<RowCodes>(code) << (2 * gen_lanes_.size());
            gen_lanes_.push_back(static_cast<RowMode>(mode));
        });

        // Which cutoff, its bound, and the inactive-mode prefix, all fixed for the propagator's
        // lifetime. active_bit_offset counts physical bits and a mode spans two, hence the halving.
        if (const auto *length = cutoff_eval.length_cutoff(); length != nullptr) {
            kind_ = Kind::Length;
            cutoff_value_ = length->cutoff;
            inactive_mode_prefix_ = length->masks.active_bit_offset / 2;
        }
        else if (const auto *support = cutoff_eval.support_cutoff(); support != nullptr) {
            kind_ = Kind::Support;
            cutoff_value_ = support->cutoff;
            inactive_mode_prefix_ = support->masks.active_bit_offset / 2;
        }
        sparse_usable_ = gen_fits && kind_ != Kind::None;
        // Shared with the record stride rather than derived here: a product that fits the scratch has to fit
        // the record it is pushed into.
        capacity_ = sparse_record_capacity(gen, cutoff_eval);
        out_lanes_.resize(capacity_);
    }

    template <typename Store>
    [[gnu::always_inline]] auto product(const Store &store, size_t i) -> TermProduct {
        dense_valid_ = false;
        if (sparse_usable_ && !store.spilled(i)) {
            const SparseRow mono = store.view(i);
            const auto toggled = sparse_toggle(mono, generator(), out_lanes_.data(), capacity_);
            if (!toggled.overflowed) {
                fallback_used_ = false;
                product_ = toggled;
                return {toggled.overlap, A::codes_rotation_sign(mono, generator())};
            }
        }
        fallback_used_ = true;
        return fallback_.product(store, i);
    }

    [[nodiscard]] auto passes(size_t new_pop) const -> bool {
        if (fallback_used_) {
            return fallback_.passes(new_pop);
        }
        if (kind_ == Kind::Length) {
            return codes_length_passes_with_popcount(product_row(), cutoff_value_, new_pop, inactive_mode_prefix_);
        }
        return codes_support_passes_with_popcount(product_row(), cutoff_value_, new_pop, inactive_mode_prefix_);
    }
    // Still the dense hash, and it has to be: owner routing is monomial_hash everywhere, including
    // find_rank's initial distribution, and the store's own probe hash is a different function for a
    // different purpose. So a *multi-rank* run still materializes once per surviving term here -- moving
    // that would mean changing find_rank too. A serial run never calls this (the scan short-circuits at
    // rank_count == 1), so it materializes only for the terms that escape.
    auto owner(size_t rank_count) -> size_t { return monomial_hash(dense_row()) % rank_count; }

    // The row when there is one, and the dense escape when there is not. Which of the two is not a tuning
    // choice: a fully paired product escapes the cutoff, so nothing bounds a query's support.
    auto push(QueryOut out, int phase) -> void {
        if (fallback_used_) {
            sparse_query_push_escape(out.records, out.escapes, dense_row(), capacity_, phase);
            return;
        }
        sparse_query_push(out.records, product_row(), capacity_, phase);
    }
    [[nodiscard]] auto record_words() const -> size_t { return query_words(sparse_payload_words(capacity_)); }

    // The product in support form. Meaningless when the term fell back to the dense kernel.
    [[nodiscard]] auto product_row() const -> SparseRow { return SparseRow{out_lanes_.data(), product_.codes}; }
    [[nodiscard]] auto fell_back() const -> bool { return fallback_used_; }
    // The record's lane capacity, which is also this kernel's scratch capacity -- see sparse_record_capacity.
    [[nodiscard]] auto record_capacity() const -> size_t { return capacity_; }

private:
    enum class Kind : uint8_t { None, Length, Support };

    [[nodiscard]] auto generator() const -> SparseRow { return SparseRow{gen_lanes_.data(), gen_codes_}; }

    // Memoized because owner() and push() both want it and only push() runs unconditionally.
    auto dense_row() -> const Bitset & {
        if (fallback_used_) {
            return fallback_.product_row();
        }
        if (!dense_valid_) {
            dense_.reset(); // as in the dense kernel: the slot walk only sets bits
            fill_from_sparse_row(product_row(), dense_);
            dense_valid_ = true;
        }
        return dense_;
    }

    DenseTermProducts<A> fallback_;
    std::vector<RowMode> gen_lanes_ = {};
    RowCodes gen_codes_ = 0;
    DefaultInitVector<RowMode> out_lanes_ = {};
    SparseProduct product_ = {};
    Bitset dense_;
    size_t capacity_ = 0;
    size_t inactive_mode_prefix_ = 0;
    unsigned int cutoff_value_ = 0;
    Kind kind_ = Kind::None;
    bool sparse_usable_ = false;
    bool fallback_used_ = true;
    bool dense_valid_ = false;
};

// Which kernel a store wants at a given bound width. Explicit specializations for the same reason
// QueryKeysFor has them: a store must not know what the scan does with its rows, and an unhandled
// store must fail to compile rather than pick a default. W == 0 is the unspecialized arm, and it is
// the only one the sparse store has -- a sparse row's per-term work is O(slots), not O(storage words),
// so <SparseRowStore, A, W != 0> is deliberately left incomplete rather than silently dense.
template <typename Store, Algebra A, size_t W = 0>
struct TermProductsFor;
template <Algebra A>
struct TermProductsFor<OperatorIndex, A, 0> {
    using type = DenseTermProducts<A>;
};
template <Algebra A, size_t W>
struct TermProductsFor<OperatorIndex, A, W> {
    using type = DenseTermProductsW<A, W>;
};
template <Algebra A>
struct TermProductsFor<SparseRowStore, A, 0> {
    using type = SparseTermProducts<A>;
};

// Bind the storage word count once per gate and let the scan build a kernel that knows it.
//
// Like with_algebra and with_store, this turns one runtime gate-wide property into a compile-time one
// at a single seam. Everything after that seam is templated on it. Doing this once per gate keeps
// compile time under control; encoding Bitset width in the type was too expensive.
//
// We use tag dispatch instead of passing in a kernel object. The body needs to declare the kernel as a
// local, and passing by reference was ~3% slower on the unspecialized arm because locals optimize better.
//
// Only W in [1, kNarrowKernelWords] is specialized. Above that, the runtime loop is already as fast or
// faster, and fewer code paths are better for instruction cache.
//
// This cap is fixed here, not a build option. Unlike monoprop_SPARSE_ROW_MIN_MODES (ISA-dependent),
// this is a storage-word count and maps to the same width regime on every machine.
//
// kNarrowKernelWords = 4 means 128 storage modes, which covers all current shipped models. Measured
// benefit is about 10% at 2-4 words, fading by 7 words, for about 11% extra `.text` from the four
// specializations. Raising the cap to Bitset::kInlineWords is correct but not faster; setting it to 0
// restores the pre-seam code path for re-measurement.
inline constexpr size_t kNarrowKernelWords = 4;
static_assert(kNarrowKernelWords <= Bitset::kInlineWords,
              "a specialized kernel assumes its operands are inline; above kInlineWords they spill");

// Whether a store's per-term work has a storage-word trip count for W to bind at all. A trait rather
// than an is_same_v in the dispatch, so a new backend states its own answer next to the kernel it
// asks for above, instead of silently inheriting "no".
template <typename Store>
inline constexpr bool kBindsKernelWidth = false;
template <>
inline constexpr bool kBindsKernelWidth<OperatorIndex> = true;

// W as the dispatch will actually pass it: itself while the build specializes that width, 0 once it
// does not. Mapping the arm rather than shortening the dispatch is what makes raising the cap a
// one-line change with no new arm.
template <size_t W>
inline constexpr size_t kCappedKernelWidth = W <= kNarrowKernelWords ? W : 0;

template <typename Store, typename F>
[[gnu::always_inline]] inline auto with_kernel_width(size_t num_words, F &&f) -> decltype(auto) {
    // The width regime, not the arm count, is what with_nwords' contract asks a caller to gate on:
    // above kInlineWords the words are on the heap, so there is no width to bind even in principle --
    // the kernel's precondition is that every operand is inline.
    if constexpr (kBindsKernelWidth<Store>) {
        if (num_words >= 1 && num_words <= Bitset::kInlineWords) {
            return with_nwords(num_words, [&f]<size_t W>(std::integral_constant<size_t, W>) -> decltype(auto) {
                return f(std::integral_constant<size_t, kCappedKernelWidth<W>>{});
            });
        }
    }
    return f(std::integral_constant<size_t, 0>{});
}

} // namespace monoprop::detail
