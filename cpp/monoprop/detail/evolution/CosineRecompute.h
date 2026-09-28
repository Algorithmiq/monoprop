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

// Recompute a layer's cosine index set from the persistent inverted index instead of a stored per-layer
// bitmap. A layer's cos = the terms anticommuting with its generator G = the per-word XOR-fold of G's
// inverted-index columns, with the odd-|G| row_parity(|M|) correction, truncated to the first
// `scaled_count` indices (the term count before that layer's own inserts).

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

namespace monoprop::detail {

/*!
 * \brief The kinds of logical work range a threaded kernel reports to its range observer (see NoRangeObserver).
 */
enum class KernelRange {
    cos_lazy,     //!< scale_cos_lazy: one range per kColumnBlockWords fold words.
    cos_mask,     //!< scale_cos_mask: one range per kCosMaskRangeBlocks stored mask blocks.
    fused_gather, //!< apply_fused_contract insert snapshots: one range per kFusedRecordRange records.
    fused_apply,  //!< apply_fused_contract rotations: one range per kFusedRecordRange records.
};

/*!
 * \brief The observer production kernels use: it does nothing and compiles away.
 *
 * A kernel calls `prepare(kind, ranges)` once on the calling thread before any range runs, then
 * `visit(kind, range)` at the start of each logical range, from whichever thread owns that range. Tests
 * substitute an observer that records the worker of each range; that seam is the only reason the kernels
 * take an observer, and no production caller passes one.
 */
struct NoRangeObserver {
    auto prepare(KernelRange /*kind*/, size_t /*ranges*/) const noexcept -> void {}
    auto visit(KernelRange /*kind*/, size_t /*range*/) const noexcept -> void {}
};

//! Stored mask blocks (one 64-bit word each) per scale_cos_mask range: as many words as a fold block.
inline constexpr size_t kCosMaskRangeBlocks = kColumnBlockWords;

//! Number of `per_range`-sized logical ranges covering `items`.
inline constexpr auto logical_ranges(size_t items, size_t per_range) -> size_t {
    return (items + per_range - 1) / per_range;
}

// Reconstruct a layer's generator Monomial from the raw words stored on its LayerCore.
template <size_t NumModes>
inline auto generator_from_words(const std::vector<uint64_t> &gw) -> Monomial<NumModes> {
    Monomial<NumModes> gen{};
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    return gen;
}

// Fold-word parameters shared by FoldCache and LazyFold, applied per word by apply_fold_mask.
//
// Deliberately holds no pointer into the index. row_parity_ is a lazily built, mutable vector that
// append_rows resizes and an index rebuild frees, so a pointer cached here would dangle: a
// LazyFold lives inside a retained functional closure (build_cos_callbacks keeps one per graph
// layer), which outlives any number of build_graph calls. Fetch it at use time instead --
// fold_row_parity() below costs one empty() test.
struct FoldMask {
    bool g_odd = false;    // false for Pauli and for even |G|
    size_t mask_words = 0; // min(inverted index words, ceil(scaled_count/64))
    size_t last_word = 0;
    uint64_t last_mask = ~uint64_t{0};
};

template <size_t NumModes>
inline auto make_fold_mask(const InvertedIndex<NumModes> &sc,
                           const Monomial<NumModes> &gen,
                           uint64_t scaled_count,
                           Basis basis = Basis::Majorana) -> FoldMask {
    FoldMask s;
    // Pauli folds J(G) and never needs the odd-|G| parity correction (see Scan.h); Majorana applies it
    // when |G| is odd. Truncation bounds are basis-independent.
    s.g_odd = algebra_fold_needs_odd_correction<NumModes>(basis, gen);
    const size_t full = sc.words();
    s.mask_words = std::min(full, static_cast<size_t>((scaled_count + 63) / 64));
    s.last_word = (s.mask_words == 0) ? 0 : s.mask_words - 1;
    s.last_mask = (scaled_count % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (scaled_count % 64)) - 1);
    return s;
}

// A layer's cosine fold materialised into one buffer. Backs the pare materializer and the
// recompute-equivalence test oracle.
template <size_t NumModes>
struct FoldCache {
    std::vector<uint64_t> combined; // the generator's columns XOR-combined over [0, fold.mask_words)
    FoldMask fold;
    // Safe to hold here (unlike in FoldMask): every FoldCache is built and consumed within one call,
    // with no operator growth in between.
    const uint64_t *row_parity = nullptr;
};

// The odd-|G| row-parity words for a fold, or nullptr when the correction does not apply.
template <size_t NumModes>
inline auto fold_row_parity(const InvertedIndex<NumModes> &sc, const FoldMask &f) -> const uint64_t * {
    return f.g_odd ? sc.row_parity_words() : nullptr;
}

template <size_t NumModes>
auto make_fold_cache(const InvertedIndex<NumModes> &sc,
                     const Monomial<NumModes> &gen,
                     uint64_t scaled_count,
                     Basis basis) -> FoldCache<NumModes> {
    FoldCache<NumModes> p;
    p.fold = make_fold_mask<NumModes>(sc, gen, scaled_count, basis);
    p.row_parity = fold_row_parity<NumModes>(sc, p.fold);
    // generator_words stores the real G; re-derive the fold generator (J(G) for Pauli) as the scan did.
    const auto fold_gen = algebra_fold_generator<NumModes>(basis, gen);
    const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);

    // One combine over [0, mask_words): words >= mask_words are never read, so dropping them is exact.
    p.combined.resize(p.fold.mask_words); // combine_columns_block zero-fills
    if (p.fold.mask_words != 0) {
        combine_columns_block<NumModes>(sc,
                                        {gen_columns.indices.data(), gen_columns.count},
                                        p.combined.data(),
                                        0,
                                        p.fold.mask_words);
    }
    return p;
}

// The one fold-word mask rule, shared by fold_word and recipe_fold_word and matching
// even_parity_scan_pass1. `row_parity` is passed in so callers hoist it out of the loop and no long-lived
// mask holds an index pointer (see FoldMask).
[[gnu::always_inline]] inline auto apply_fold_mask(uint64_t bits,
                                                   size_t wi,
                                                   const FoldMask &f,
                                                   const uint64_t *row_parity) -> uint64_t {
    if (f.g_odd) {
        bits ^= row_parity[wi];
    }
    if (wi == f.last_word) {
        bits &= f.last_mask;
    }
    return bits;
}

template <size_t NumModes>
[[gnu::always_inline]] inline auto fold_word(const FoldCache<NumModes> &p, size_t wi) -> uint64_t {
    return apply_fold_mask(p.combined[wi], wi, p.fold, p.row_parity);
}

// Visit each set bit of `bits` ascending, calling op(base + bit).
template <typename BitOp>
[[gnu::always_inline]] inline auto for_each_cos_index(size_t base, uint64_t bits, BitOp op) -> void {
    while (bits) {
        op(base + static_cast<size_t>(std::countr_zero(bits)));
        bits &= bits - 1;
    }
}

// Metadata to recompute a layer's cosine fold on the fly, with no per-layer cos buffer.
//
// `columns` is heap-sized to |G| (typically 2-4) rather than reusing EvenParityGeneratorColumns' fixed
// std::array<size_t, 2*NumModes>: a LazyFold is retained per graph layer, 4 KB each at NumModes=256.
template <size_t NumModes>
struct LazyFold {
    std::vector<size_t> columns;
    FoldMask fold;
};

template <size_t NumModes>
auto make_lazy_fold(const InvertedIndex<NumModes> &sc,
                    const Monomial<NumModes> &gen,
                    uint64_t scaled_count,
                    Basis basis) -> LazyFold<NumModes> {
    LazyFold<NumModes> r;
    r.fold = make_fold_mask<NumModes>(sc, gen, scaled_count, basis);
    const auto fold_gen = algebra_fold_generator<NumModes>(basis, gen);
    const auto columns = build_even_parity_generator_columns<NumModes>(fold_gen);
    r.columns.assign(columns.indices.begin(), columns.indices.begin() + columns.count);
    return r;
}

// The recompute analogue of fold_word, over a freshly-built block word (bb = the block's first fold word).
template <size_t NumModes>
[[gnu::always_inline]] inline auto recipe_fold_word(const LazyFold<NumModes> &r,
                                                    const uint64_t *blk,
                                                    size_t bb,
                                                    size_t wi,
                                                    const uint64_t *row_parity) -> uint64_t {
    return apply_fold_mask(blk[wi - bb], wi, r.fold, row_parity);
}

// Append a layer's cosine-set indices to `out`, walking the same blocks as scale_cos_* rather than sharing
// a visitor with them, so the scaling kernels stay verbatim.
template <size_t NumModes>
auto cos_indices_lazy(const InvertedIndex<NumModes> &sc, const LazyFold<NumModes> &r, std::vector<TermIndex> &out)
    -> void {
    const size_t mask_words = r.fold.mask_words;
    const uint64_t *row_parity = fold_row_parity<NumModes>(sc, r.fold);
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block<NumModes>(sc, {r.columns.data(), r.columns.size()}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64,
                               recipe_fold_word<NumModes>(r, blk.data(), bb, wi, row_parity),
                               [&out](size_t i) { out.push_back(static_cast<TermIndex>(i)); });
        }
    }
}

inline auto cos_indices_mask(const CosMask &cos, std::vector<TermIndex> &out) -> void {
    for (const auto &[base, bits] : cos.blocks) {
        for_each_cos_index(base, bits, [&out](size_t i) { out.push_back(static_cast<TermIndex>(i)); });
    }
}

/*!
 * \brief Multiply every coefficient in a layer's recomputed cosine set by `cos_val`.
 *
 * Each kColumnBlockWords fold block is one logical range, so the ranges touch disjoint coefficients.
 *
 * \param sc       The inverted index the fold reads; must not change during the call.
 * \param r        The layer's fold recipe.
 * \param coeff    Coefficients, covering at least `r.fold.mask_words * 64` rows or the index rows.
 * \param cos_val  The factor.
 * \param options  Thread budget.
 * \param observer Test seam; see NoRangeObserver.
 */
template <size_t NumModes, class Observer = NoRangeObserver>
auto scale_cos_lazy(const InvertedIndex<NumModes> &sc,
                    const LazyFold<NumModes> &r,
                    double *coeff,
                    double cos_val,
                    parallel::Options options = {},
                    const Observer &observer = {}) -> void {
    const size_t mask_words = r.fold.mask_words;
    // Fetched here, on the caller, for this call only: row_parity_words() builds its cache lazily, so it must
    // never run in a worker, and the pointer moves whenever the index grows.
    const uint64_t *const row_parity = fold_row_parity<NumModes>(sc, r.fold);
    const std::span<const size_t> columns{r.columns.data(), r.columns.size()};
    const size_t ranges = logical_ranges(mask_words, kColumnBlockWords);
    observer.prepare(KernelRange::cos_lazy, ranges);
    parallel::for_blocks(ranges, options, [&](size_t range) {
        observer.visit(KernelRange::cos_lazy, range);
        // Worker-private: never the caller's thread_local column_block_scratch(). combine_columns_block
        // writes every word of [bb, be) before any is read.
        std::array<uint64_t, kColumnBlockWords> blk;
        const size_t bb = range * kColumnBlockWords;
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block<NumModes>(sc, columns, blk.data(), bb, be);
        // apply_fold_mask compares wi with the fold's global last word, so only the real tail is masked.
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word<NumModes>(r, blk.data(), bb, wi, row_parity), [&](size_t i) {
                coeff[i] *= cos_val;
            });
        }
    });
}

template <size_t NumModes>
auto accumulate_cos_lazy(const InvertedIndex<NumModes> &sc,
                         const LazyFold<NumModes> &r,
                         double *state,
                         double *ham,
                         double cos_val,
                         double sec_val,
                         [[maybe_unused]] parallel::Options options = {}) -> double {
    const size_t mask_words = r.fold.mask_words;
    const uint64_t *row_parity = fold_row_parity<NumModes>(sc, r.fold);
    double loc = 0.0;
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block<NumModes>(sc, {r.columns.data(), r.columns.size()}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word<NumModes>(r, blk.data(), bb, wi, row_parity), [&](size_t i) {
                loc += state[i] * ham[i];
                ham[i] *= sec_val;
                state[i] *= cos_val;
            });
        }
    }
    return loc;
}

/*!
 * \brief Multiply every coefficient in a stored cosine mask by `cos_val`, leaving all others untouched.
 *
 * Each kCosMaskRangeBlocks run of mask blocks is one logical range. CosMask blocks are ascending with
 * distinct word-aligned bases, so the ranges touch disjoint coefficients.
 *
 * \param coeff    Coefficients, covering every index in `cos`.
 * \param cos      The mask.
 * \param cos_val  The factor.
 * \param options  Thread budget.
 * \param observer Test seam; see NoRangeObserver.
 */
template <class Observer = NoRangeObserver>
auto scale_cos_mask(double *coeff,
                    const CosMask &cos,
                    double cos_val,
                    parallel::Options options = {},
                    const Observer &observer = {}) -> void {
    const size_t n = cos.blocks.size();
    const size_t ranges = logical_ranges(n, kCosMaskRangeBlocks);
    observer.prepare(KernelRange::cos_mask, ranges);
    parallel::for_blocks(ranges, options, [&](size_t range) {
        observer.visit(KernelRange::cos_mask, range);
        const size_t hi = std::min(n, (range + 1) * kCosMaskRangeBlocks);
        for (size_t k = range * kCosMaskRangeBlocks; k < hi; ++k) {
            const auto [base, bits] = cos.blocks[k];
            for_each_cos_index(base, bits, [&](size_t i) { coeff[i] *= cos_val; });
        }
    });
}
inline auto accumulate_cos_mask(double *state,
                                double *ham,
                                const CosMask &cos,
                                double cos_val,
                                double sec_val,
                                [[maybe_unused]] parallel::Options options = {}) -> double {
    const size_t n = cos.blocks.size();
    double loc = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const auto [base, bits] = cos.blocks[k];
        for_each_cos_index(base, bits, [&](size_t i) {
            loc += state[i] * ham[i];
            ham[i] *= sec_val;
            state[i] *= cos_val;
        });
    }
    return loc;
}

template <size_t NumModes>
inline auto fold_to_cos_mask(const FoldCache<NumModes> &p) -> CosMask {
    CosMask c;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        const uint64_t b = fold_word<NumModes>(p, wi);
        if (b) {
            c.blocks.emplace_back(wi * 64, b);
            c.total_count += static_cast<size_t>(std::popcount(b));
        }
    }
    return c;
}
// Cos-index count without materialising the blocks; for diagnostics (graph_size).
template <size_t NumModes>
inline auto fold_popcount(const FoldCache<NumModes> &p) -> size_t {
    size_t total = 0;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        total += static_cast<size_t>(std::popcount(fold_word<NumModes>(p, wi)));
    }
    return total;
}

template <size_t NumModes>
inline auto fold_to_indices(const FoldCache<NumModes> &p) -> VecZ {
    VecZ inds;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        for_each_cos_index(wi * 64, fold_word<NumModes>(p, wi), [&](size_t i) { inds.push_back(i); });
    }
    return inds;
}

} // namespace monoprop::detail
