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

// CosineRecompute.h — recompute a layer's cosine index set from the persistent inverted index instead
// of a stored per-layer bitmap.
//
// Invariant: a layer's cos = the operator terms anticommuting with its generator G = the per-word
// XOR-fold of G's inverted-index columns (combine_columns_block), with the odd-|G| row_parity(|M|)
// correction, truncated to the first `scaled_count` operator indices (the term count BEFORE that
// layer's own inserts). LazyFold recomputes this on the fly (the sole runtime replay path); FoldCache
// materialises it into one buffer for the pare materializer and the equivalence-test oracle. Both
// share their fold-word parameters as one embedded FoldMask.

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
#include <utility>
#include <vector>

#include "monoprop/algebra/Algebra.h" // algebra_fold_generator / algebra_fold_needs_odd_correction
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h" // LayerCosScale, LayerCosAccumulate
#include "monoprop/detail/evolution/layer_build/Scan.h" // gen columns, inverted index, CosMask (only scan-side symbols used)
#include "monoprop/detail/operator/InvertedIndex.h" // combine_columns_block, column_block_scratch

namespace monoprop::detail {

// Reconstruct a layer's generator Monomial from the raw words stored on its LayerCore.
template <size_t NumModes>
inline auto generator_from_words(const std::vector<uint64_t> &gw) -> Monomial<NumModes> {
    Monomial<NumModes> gen{};
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    return gen;
}

/// Fold-word parameters shared by FoldCache and LazyFold: the odd-|G| row-parity correction and the
/// scaled_count truncation, applied per word by apply_fold_mask.
struct FoldMask {
    bool g_odd = false;
    const uint64_t *row_parity = nullptr; // null for even |G|
    size_t mask_words = 0;                // min(inverted index words, ceil(scaled_count/64))
    size_t last_word = 0;
    uint64_t last_mask = ~uint64_t{0};
};

template <size_t NumModes>
inline auto make_fold_mask(const InvertedIndex<NumModes> &sc,
                           const Monomial<NumModes> &gen,
                           uint64_t scaled_count,
                           Basis basis = Basis::Majorana) -> FoldMask {
    FoldMask s;
    // Pauli anticommutation folds J(G)'s columns and never needs the odd-|G| parity correction (see
    // Scan.h); Majorana applies it when |G| is odd. Truncation bounds are basis-independent.
    s.g_odd = algebra_fold_needs_odd_correction<NumModes>(basis, gen);
    if (s.g_odd) {
        s.row_parity = sc.row_parity_words();
    }
    const size_t full = sc.words();
    s.mask_words = std::min(full, static_cast<size_t>((scaled_count + 63) / 64));
    s.last_word = (s.mask_words == 0) ? 0 : s.mask_words - 1;
    s.last_mask = (scaled_count % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (scaled_count % 64)) - 1);
    return s;
}

/// A layer's cosine fold materialised into one buffer (the generator's columns XOR-combined over
/// fold.mask_words words). Backs the pare materializer and the recompute-equivalence test oracle; not a
/// runtime replay cache (retired — see the fold-recompute note below).
template <size_t NumModes>
struct FoldCache {
    std::vector<uint64_t> combined; // the generator's columns XOR-combined over [0, fold.mask_words)
    FoldMask fold;
};

template <size_t NumModes>
auto make_fold_cache(const InvertedIndex<NumModes> &sc,
                     const Monomial<NumModes> &gen,
                     uint64_t scaled_count,
                     Basis basis = Basis::Majorana) -> FoldCache<NumModes> {
    FoldCache<NumModes> p;
    p.fold = make_fold_mask<NumModes>(sc, gen, scaled_count, basis);
    // generator_words stores the REAL G; re-derive the fold generator (J(G) for Pauli) as the scan did.
    const auto fold_gen = algebra_fold_generator<NumModes>(basis, gen);
    const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);

    // One combine over [0, mask_words): words >= mask_words are never read, so dropping them is exact;
    // the odd-|G| row_parity and last-word mask are applied per-word later in fold_word.
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

// The one fold-word mask rule (shared by fold_word and recipe_fold_word, matching even_parity_scan_pass1):
// apply the odd-|G| row_parity correction, then the last-word scaled_count truncation mask.
[[gnu::always_inline]] inline uint64_t apply_fold_mask(uint64_t bits, size_t wi, const FoldMask &f) {
    if (f.g_odd) {
        bits ^= f.row_parity[wi];
    }
    if (wi == f.last_word) {
        bits &= f.last_mask;
    }
    return bits;
}

template <size_t NumModes>
[[gnu::always_inline]] inline uint64_t fold_word(const FoldCache<NumModes> &p, size_t wi) {
    return apply_fold_mask(p.combined[wi], wi, p.fold);
}

// Visit each set bit of `bits` ascending, calling op(base + bit). The single bit-scatter kernel behind
// every cos scale/accumulate loop; always_inline so the per-bit op has no call overhead.
template <typename BitOp>
[[gnu::always_inline]] inline void for_each_cos_index(size_t base, uint64_t bits, BitOp op) {
    while (bits) {
        op(base + static_cast<size_t>(std::countr_zero(bits)));
        bits &= bits - 1;
    }
}

// Fold RECOMPUTE — the SOLE runtime replay path: recompute each layer's fold on the fly rather than hold
// a per-layer buffer (multi-GB, and so cold that streaming it matched the recompute — measured 2026-07-20,
// so the persistent cache was retired). Fused with the scatter and parallelised over disjoint fold-word
// ranges (race-free; XOR associative → byte-identical to make_fold_cache); cache-blocked into L1 sub-blocks.

/// Metadata to recompute a layer's cosine fold on the fly (the sole runtime replay path): the
/// generator's ≤|G| inverted index column indices plus the cos truncation bounds — no per-layer buffer.
template <size_t NumModes>
struct LazyFold {
    EvenParityGeneratorColumns<NumModes> columns{};
    FoldMask fold;
};

template <size_t NumModes>
auto make_lazy_fold(const InvertedIndex<NumModes> &sc,
                    const Monomial<NumModes> &gen,
                    uint64_t scaled_count,
                    Basis basis = Basis::Majorana) -> LazyFold<NumModes> {
    LazyFold<NumModes> r;
    r.fold = make_fold_mask<NumModes>(sc, gen, scaled_count, basis);
    const auto fold_gen = algebra_fold_generator<NumModes>(basis, gen);
    r.columns = build_even_parity_generator_columns<NumModes>(fold_gen);
    return r;
}

// The recompute analogue of fold_word: apply the odd-|G| parity correction and last-word scaled_count
// mask to a freshly-built block word `blk[wi - bb]` (bb = the block's first fold word).
template <size_t NumModes>
[[gnu::always_inline]] inline uint64_t recipe_fold_word(const LazyFold<NumModes> &r,
                                                        const uint64_t *blk,
                                                        size_t bb,
                                                        size_t wi) {
    return apply_fold_mask(blk[wi - bb], wi, r.fold);
}

template <size_t NumModes>
void scale_cos_lazy(const InvertedIndex<NumModes> &sc, const LazyFold<NumModes> &r, double *coeff, double cos_val) {
    const size_t mask_words = r.fold.mask_words;
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block<NumModes>(sc, {r.columns.indices.data(), r.columns.count}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word<NumModes>(r, blk.data(), bb, wi), [&](size_t i) {
                coeff[i] *= cos_val;
            });
        }
    }
}

template <size_t NumModes>
double accumulate_cos_lazy(const InvertedIndex<NumModes> &sc,
                           const LazyFold<NumModes> &r,
                           double *state,
                           double *ham,
                           double cos_val,
                           double sec_val) {
    const size_t mask_words = r.fold.mask_words;
    double loc = 0.0;
    std::vector<uint64_t> &blk = column_block_scratch();
    for (size_t bb = 0; bb < mask_words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, mask_words);
        combine_columns_block<NumModes>(sc, {r.columns.indices.data(), r.columns.count}, blk.data(), bb, be);
        for (size_t wi = bb; wi < be; ++wi) {
            for_each_cos_index(wi * 64, recipe_fold_word<NumModes>(r, blk.data(), bb, wi), [&](size_t i) {
                loc += state[i] * ham[i];
                ham[i] *= sec_val;
                state[i] *= cos_val;
            });
        }
    }
    return loc;
}

// Scale/accumulate over a CosMask. Build- and pare-produced lists have 64-aligned disjoint blocks, so
// the block-range split is race-free.
inline void scale_cos_mask(double *coeff, const CosMask &cos, double cos_val) {
    const size_t n = cos.blocks.size();
    for (size_t k = 0; k < n; ++k) {
        const auto [base, bits] = cos.blocks[k];
        for_each_cos_index(base, bits, [&](size_t i) { coeff[i] *= cos_val; });
    }
}
inline double accumulate_cos_mask(double *state, double *ham, const CosMask &cos, double cos_val, double sec_val) {
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
template <size_t NumModes>
inline auto fold_to_indices(const FoldCache<NumModes> &p) -> VecZ {
    VecZ inds;
    for (size_t wi = 0; wi < p.fold.mask_words; ++wi) {
        for_each_cos_index(wi * 64, fold_word<NumModes>(p, wi), [&](size_t i) { inds.push_back(i); });
    }
    return inds;
}

} // namespace monoprop::detail
