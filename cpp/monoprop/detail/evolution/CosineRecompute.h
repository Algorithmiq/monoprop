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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/operator/InvertedIndex.h"

namespace monoprop::detail {

// Reconstruct a layer's generator from the raw words stored on its LayerCore.
//
// num_bits is passed rather than recovered as gw.size() * 64: the stored words are the generator's word
// count, which does not pin down its bit width -- a width that is not a whole multiple of 64 rounds up
// to the same word count, and the round trip would silently widen it (Bitset carries the used bits of
// the last word, so `size()` and the top-bit mask would both come back wrong). The caller has the real
// width in hand; the operator the generator is applied against is the one that defines it.
inline auto generator_from_words(const std::vector<uint64_t> &gw, size_t num_bits) -> Bitset {
    Bitset gen(num_bits);
    assert(gw.size() == gen.num_words() && "generator words must match the operator's storage width");
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

inline auto make_fold_mask(const auto &sc,
                           const MonomialLike auto &gen,
                           uint64_t scaled_count,
                           Basis basis = Basis::Majorana) -> FoldMask {
    FoldMask s;
    // Pauli folds J(G) and never needs the odd-|G| parity correction (see Scan.h); Majorana applies it
    // when |G| is odd. Truncation bounds are basis-independent.
    s.g_odd = algebra_fold_needs_odd_correction(basis, gen);
    const size_t full = sc.words();
    s.mask_words = std::min(full, static_cast<size_t>((scaled_count + 63) / 64));
    s.last_word = (s.mask_words == 0) ? 0 : s.mask_words - 1;
    s.last_mask = (scaled_count % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (scaled_count % 64)) - 1);
    return s;
}

// A layer's cosine fold materialised into one buffer. Backs the pare materializer and the
// recompute-equivalence test oracle.
// No width parameter: every member is a byte/word count or a heap buffer, and none was ever sized by
// one. It was templated only because its producer was.
struct FoldCache {
    std::vector<uint64_t> combined; // the generator's columns XOR-combined over [0, fold.mask_words)
    FoldMask fold;
    // Safe to hold here (unlike in FoldMask): every FoldCache is built and consumed within one call,
    // with no operator growth in between.
    const uint64_t *row_parity = nullptr;
};

// The odd-|G| row-parity words for a fold, or nullptr when the correction does not apply.
inline auto fold_row_parity(const auto &sc, const FoldMask &f) -> const uint64_t * {
    return f.g_odd ? sc.row_parity_words() : nullptr;
}

auto make_fold_cache(const auto &sc, const MonomialLike auto &gen, uint64_t scaled_count, Basis basis) -> FoldCache {
    FoldCache p;
    p.fold = make_fold_mask(sc, gen, scaled_count, basis);
    p.row_parity = fold_row_parity(sc, p.fold);
    // generator_words stores the real G; re-derive the fold generator (J(G) for Pauli) as the scan did.
    const auto fold_gen = algebra_fold_generator(basis, gen);
    const auto gen_columns = build_even_parity_generator_columns(fold_gen);

    // One combine over [0, mask_words): words >= mask_words are never read, so dropping them is exact.
    p.combined.resize(p.fold.mask_words); // combine_columns_block zero-fills
    if (p.fold.mask_words != 0) {
        combine_columns_block(sc,
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

[[gnu::always_inline]] inline auto fold_word(const FoldCache &p, size_t wi) -> uint64_t {
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
// `columns` is heap-sized to |G| (typically 2-4): a LazyFold is retained per graph layer, and sizing it
// by the register width instead would have cost 4 KB each at 256 modes. Carries no width parameter, for
// the same reason as FoldCache.
struct LazyFold {
    std::vector<size_t> columns;
    FoldMask fold;
};

auto make_lazy_fold(const auto &sc, const MonomialLike auto &gen, uint64_t scaled_count, Basis basis) -> LazyFold {
    LazyFold r;
    r.fold = make_fold_mask(sc, gen, scaled_count, basis);
    const auto fold_gen = algebra_fold_generator(basis, gen);
    const auto columns = build_even_parity_generator_columns(fold_gen);
    r.columns.assign(columns.indices.begin(), columns.indices.begin() + columns.count);
    return r;
}

// The recompute analogue of fold_word, over a freshly-built block word (bb = the block's first fold word).
[[gnu::always_inline]] inline auto recipe_fold_word(const LazyFold &r,
                                                    const uint64_t *blk,
                                                    size_t bb,
                                                    size_t wi,
                                                    const uint64_t *row_parity) -> uint64_t {
    return apply_fold_mask(blk[wi - bb], wi, r.fold, row_parity);
}

// Append a layer's cosine-set indices to `out`, walking the same blocks as scale_cos_* rather than sharing
// a visitor with them, so the scaling kernels stay verbatim.
auto cos_indices_lazy(const auto &sc, const LazyFold &r, std::vector<TermIndex> &out) -> void {
    const size_t mask_words = r.fold.mask_words;
    const uint64_t *row_parity = fold_row_parity(sc, r.fold);
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block(sc, {r.columns.data(), r.columns.size()}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word(r, blk.data(), bb, wi, row_parity), [&out](size_t i) {
                out.push_back(static_cast<TermIndex>(i));
            });
        }
    }
}

inline auto cos_indices_mask(const CosMask &cos, std::vector<TermIndex> &out) -> void {
    for (const auto &[base, bits] : cos.blocks) {
        for_each_cos_index(base, bits, [&out](size_t i) { out.push_back(static_cast<TermIndex>(i)); });
    }
}

auto scale_cos_lazy(const auto &sc, const LazyFold &r, double *coeff, double cos_val) -> void {
    const size_t mask_words = r.fold.mask_words;
    const uint64_t *row_parity = fold_row_parity(sc, r.fold);
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block(sc, {r.columns.data(), r.columns.size()}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word(r, blk.data(), bb, wi, row_parity), [&](size_t i) {
                coeff[i] *= cos_val;
            });
        }
    }
}

auto accumulate_cos_lazy(const auto &sc, const LazyFold &r, double *state, double *ham, double cos_val, double sec_val)
    -> double {
    const size_t mask_words = r.fold.mask_words;
    const uint64_t *row_parity = fold_row_parity(sc, r.fold);
    double loc = 0.0;
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block(sc, {r.columns.data(), r.columns.size()}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word(r, blk.data(), bb, wi, row_parity), [&](size_t i) {
                loc += state[i] * ham[i];
                ham[i] *= sec_val;
                state[i] *= cos_val;
            });
        }
    }
    return loc;
}

inline auto scale_cos_mask(double *coeff, const CosMask &cos, double cos_val) -> void {
    const size_t n = cos.blocks.size();
    for (size_t k = 0; k < n; ++k) {
        const auto [base, bits] = cos.blocks[k];
        for_each_cos_index(base, bits, [&](size_t i) { coeff[i] *= cos_val; });
    }
}
inline auto accumulate_cos_mask(double *state, double *ham, const CosMask &cos, double cos_val, double sec_val)
    -> double {
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

inline auto fold_to_cos_mask(const FoldCache &p) -> CosMask {
    CosMask c;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        const uint64_t b = fold_word(p, wi);
        if (b) {
            c.blocks.emplace_back(wi * 64, b);
            c.total_count += static_cast<size_t>(std::popcount(b));
        }
    }
    return c;
}
// Cos-index count without materialising the blocks; for diagnostics (graph_size).
inline auto fold_popcount(const FoldCache &p) -> size_t {
    size_t total = 0;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        total += static_cast<size_t>(std::popcount(fold_word(p, wi)));
    }
    return total;
}

inline auto fold_to_indices(const FoldCache &p) -> VecZ {
    VecZ inds;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        for_each_cos_index(wi * 64, fold_word(p, wi), [&](size_t i) { inds.push_back(i); });
    }
    return inds;
}

} // namespace monoprop::detail
