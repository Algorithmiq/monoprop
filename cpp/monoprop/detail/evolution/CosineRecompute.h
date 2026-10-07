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

#include "monoprop/MPGraph.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

namespace monoprop::detail {

//! Stored mask blocks (one 64-bit word each) per scale_cos_mask range: as many words as a fold block.
inline constexpr size_t kCosMaskRangeBlocks = kColumnBlockWords;

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
// LazyFold lives inside a retained functional closure (make_cos_callbacks keeps one per graph
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

// The number of indices cos_indices_lazy() appends: the same fold, popcounted.
template <size_t NumModes>
auto cos_count_lazy(const InvertedIndex<NumModes> &sc, const LazyFold<NumModes> &r) -> size_t {
    const size_t mask_words = r.fold.mask_words;
    const uint64_t *row_parity = fold_row_parity<NumModes>(sc, r.fold);
    std::vector<uint64_t> &blk = column_block_scratch();
    size_t count = 0;
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block<NumModes>(sc, {r.columns.data(), r.columns.size()}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            count += static_cast<size_t>(std::popcount(recipe_fold_word<NumModes>(r, blk.data(), bb, wi, row_parity)));
        }
    }
    return count;
}

// The number of indices cos_indices_mask() appends.
inline auto cos_count_mask(const CosMask &cos) -> size_t {
    size_t count = 0;
    for (const auto &block : cos.blocks) {
        count += static_cast<size_t>(std::popcount(block.second));
    }
    return count;
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

// Fold into `c`, replacing its contents; reuses its capacity.
template <size_t NumModes>
inline auto fold_to_cos_mask_into(const FoldCache<NumModes> &p, CosMask &c) -> void {
    c.blocks.clear();
    c.total_count = 0;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        const uint64_t b = fold_word<NumModes>(p, wi);
        if (b) {
            c.blocks.emplace_back(wi * 64, b);
            c.total_count += static_cast<size_t>(std::popcount(b));
        }
    }
}

template <size_t NumModes>
inline auto fold_to_cos_mask(const FoldCache<NumModes> &p) -> CosMask {
    CosMask c;
    fold_to_cos_mask_into<NumModes>(p, c);
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

/*!
 * \brief A layer's full cosine set (see full_cos_mask()) folded into `out`, replacing its contents and reusing its
 * capacity: a caller that folds layer after layer into one buffer allocates it once instead of once per layer.
 */
template <size_t NumModes>
auto full_cos_mask_into(const InvertedIndex<NumModes> &inverted_index,
                        const LayerTraversal &layer,
                        Basis basis,
                        CosMask &out) -> void {
    const auto gen = generator_from_words<NumModes>(layer.generator_words());
    const auto combined = make_fold_cache<NumModes>(inverted_index, gen, layer.scaled_count(), basis);
    fold_to_cos_mask_into<NumModes>(combined, out);
}

/*!
 * \brief A layer's full cosine set, folded from `inverted_index` and materialized as a mask; the paring input.
 * \param inverted_index The index of the store the layer's rows belong to.
 * \param layer          The layer; its generator words and scaled_count select the fold.
 * \param basis          The coefficient encoding.
 */
template <size_t NumModes>
auto full_cos_mask(const InvertedIndex<NumModes> &inverted_index, const LayerTraversal &layer, Basis basis) -> CosMask {
    CosMask mask;
    full_cos_mask_into<NumModes>(inverted_index, layer, basis, mask);
    return mask;
}

/*!
 * \brief Per-layer cosine callbacks replaying `graph` against `inverted_index`.
 *
 * A pared layer replays its stored mask (an empty stored mask replays nothing); every other layer recomputes its
 * fold from the index. Each closure captures `options` for its kernels. The closures keep raw pointers into the index
 * and into the layers' stored masks, so they must outlive neither the index nor the graph owning those layers. Row
 * parity words are fetched inside each call, never cached, since they move whenever the store grows. The result is
 * marked CosCallbacks::owner_parallel. Named apart from monoprop::build_cos_callbacks,
 * which delegates here, so argument-dependent lookup never makes an unqualified call ambiguous: the closures touch only
 * this index, these layers and thread-local scratch.
 *
 * \param inverted_index The index of the store whose rows the graph's endpoints name.
 * \param graph          The replay window; its layers must outlive the callbacks.
 * \param basis          The coefficient encoding (Pauli folds the generator's J image).
 * \param options        The kernels' thread budget; serial by default.
 */
template <size_t NumModes>
auto make_cos_callbacks(const InvertedIndex<NumModes> &inverted_index,
                        const MPGraphView &graph,
                        Basis basis = Basis::Majorana,
                        parallel::Options options = {}) -> CosCallbacks {
    struct LayerCos {
        bool recomputes_cos = false;
        LazyFold<NumModes> recipe{};       // used iff recomputes_cos
        const CosMask *filtered = nullptr; // points into a pruned layer's stored cos
    };
    auto cache = std::make_shared<std::vector<LayerCos>>();
    cache->reserve(graph.layers());
    for (size_t i = 0; i < graph.layers(); ++i) {
        const auto &layer = graph.get_layer(i);
        LayerCos entry;
        if (const CosMask *pruned = layer.pruned_cos(); pruned != nullptr) {
            entry.recomputes_cos = false;
            entry.filtered = pruned;
        }
        else {
            entry.recomputes_cos = true;
            const auto t = layer.traversal();
            const auto gen = generator_from_words<NumModes>(t.generator_words());
            entry.recipe = make_lazy_fold<NumModes>(inverted_index, gen, t.scaled_count(), basis);
        }
        cache->push_back(std::move(entry));
    }

    const auto *sc = &inverted_index;
    LayerCosScale cos_scale = [cache, sc, options](size_t i, double *c, double v) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            scale_cos_mask(c, *e.filtered, v, options);
        }
        else {
            scale_cos_lazy<NumModes>(*sc, e.recipe, c, v, options);
        }
    };
    LayerCosAccumulate cos_acc = [cache, sc, options](size_t i, double *s, double *h, double v, double sec) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            return accumulate_cos_mask(s, h, *e.filtered, v, sec, options);
        }
        return accumulate_cos_lazy<NumModes>(*sc, e.recipe, s, h, v, sec, options);
    };
    LayerCosIndices cos_inds = [cache, sc](size_t i, std::vector<TermIndex> &out) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            cos_indices_mask(*e.filtered, out);
            return;
        }
        cos_indices_lazy<NumModes>(*sc, e.recipe, out);
    };
    LayerCosCount cos_count = [cache, sc](size_t i) -> size_t {
        const auto &e = (*cache)[i];
        return e.recomputes_cos ? cos_count_lazy<NumModes>(*sc, e.recipe) : cos_count_mask(*e.filtered);
    };
    return {.scale = std::move(cos_scale),
            .accumulate = std::move(cos_acc),
            .indices = std::move(cos_inds),
            .count = std::move(cos_count),
            .owner_parallel = true};
}

} // namespace monoprop::detail
