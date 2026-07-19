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

// CosineRecompute.h — recompute a layer's cosine index set from the persistent inverted index
// instead of reading a stored per-layer bitmap.
//
// `cos` for a layer = the operator terms anticommuting with that layer's generator G = the per-word
// XOR-combine of G's inverted-index columns (combine_columns_block, InvertedIndex.h), with the
// odd-|G| row_parity(|M|) correction, truncated to the first `scaled_count` operator indices (the
// truncation bound = the operator term count BEFORE that layer's own inserts).
//
// FoldCache caches, once per functional, everything needed to recompute one layer's cos: the
// generator's columns XOR-combined into ONE self-owned buffer of exactly mask_words words, plus the
// (possibly odd-|G|) parity pointer. The per-word combine is then identical to even_parity_scan_pass1's,
// masked at the scaled_count boundary — one load per word at replay time.
//
// A layer's cosine set has FOUR representations across the codebase; the invariant tying them is: the
// per-word XOR-fold of the generator's inverted-index columns, truncated to `scaled_count` operator
// indices (with the odd-|G| parity correction), EQUALS the layer's anticommuting index set.
//   1. Layer::pruned_cos (a stored CosMask on PRUNED layers) — produced by pare
//      (filter_layer_cosine_data), replayed by scale_cos_mask / accumulate_cos_mask.
//   2. LayerCore.generator_words + scaled_count (recompute metadata) — stamped on the layer by build_layer,
//      consumed here by make_fold_mask / make_fold_cache / make_lazy_fold.
//   3. FoldCache / LazyFold (per-functional fold caches, this file) — built once per
//      functional from (2) in build_cos_callbacks, replayed by scale/accumulate_cos_{combined,recompute}.
//   4. A transient CosMask — emitted by the build scan for the in-build contraction
//      (evolve_step's transient closure) and apply_fused_contract; never persisted on the layer.
// FoldCache and LazyFold share their fold-word parameters as one embedded FoldMask.

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

#include "monoprop/PauliAlgebra.h" // pair_swap (Pauli fold columns = J(G))
#include "monoprop/Threading.h"
#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h" // LayerCosScale, LayerCosAccumulate
#include "monoprop/detail/evolution/layer_build/Scan.h" // gen columns, inverted index, CosMask (only scan-side symbols used)
#include "monoprop/detail/operator/InvertedIndex.h" // combine_columns_block, column_block_scratch

namespace monoprop::detail {

// Reconstruct a layer's generator MajoranaSet from the raw words stored on its LayerCore
// (generator_words()). Single definition for every replay/pare consumer.
template <size_t NumModes>
inline auto generator_from_words(const std::vector<uint64_t> &gw) -> MajoranaSet<NumModes> {
    MajoranaSet<NumModes> gen{};
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    return gen;
}

// ---- shared fold-word parameters ----
/// The fold-word parameters shared verbatim by the cached (FoldCache) and recompute
/// (LazyFold) paths: the odd-|G| row-parity(|M|) XOR correction and the scaled_count truncation,
/// applied per word by apply_fold_mask. Computed once by make_fold_mask; both fold types embed one.
struct FoldMask {
    bool g_odd = false;
    const uint64_t *row_parity = nullptr; // null for even |G|
    size_t mask_words = 0;                // min(inverted index words, ceil(scaled_count/64))
    size_t last_word = 0;
    uint64_t last_mask = ~uint64_t{0};
};

template <size_t NumModes>
inline auto make_fold_mask(const InvertedIndex<NumModes> &sc,
                           const MajoranaSet<NumModes> &gen,
                           uint64_t scaled_count,
                           Basis basis = Basis::Majorana) -> FoldMask {
    FoldMask s;
    // Pauli anticommutation folds J(G)'s columns and never needs the odd-|G| parity correction (see
    // Scan.h); Majorana applies it when |G| is odd. Truncation bounds are basis-independent.
    s.g_odd = (basis == Basis::Pauli) ? false : (gen.count() % 2 != 0);
    if (s.g_odd) {
        sc.ensure_row_parity();
        s.row_parity = sc.row_parity_word_ptr();
    }
    const size_t full = sc.words();
    s.mask_words = std::min(full, static_cast<size_t>((scaled_count + 63) / 64));
    s.last_word = (s.mask_words == 0) ? 0 : s.mask_words - 1;
    s.last_mask = (scaled_count % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (scaled_count % 64)) - 1);
    return s;
}

// ---- prepared fold per layer (built once per functional) ----
/// A layer's cosine fold, precomputed once per functional: the generator's inverted index columns XOR-
/// combined into one self-owned buffer of exactly `fold.mask_words` words (the only words ever read).
/// Replaying the cos is then a single load per word (vs one per column), holding one `mask_words`
/// buffer instead of a full-width buffer per sparse column — the memory that otherwise dominates
/// cosine recompute.
template <size_t NumModes>
struct FoldCache {
    std::vector<uint64_t> combined; // the generator's columns XOR-combined over [0, fold.mask_words)
    FoldMask fold;
};

template <size_t NumModes>
auto make_fold_cache(const InvertedIndex<NumModes> &sc,
                     const MajoranaSet<NumModes> &gen,
                     uint64_t scaled_count,
                     Basis basis = Basis::Majorana) -> FoldCache<NumModes> {
    FoldCache<NumModes> p;
    p.fold = make_fold_mask<NumModes>(sc, gen, scaled_count, basis);
    // generator_words stores the REAL G; re-derive J(G) here for Pauli exactly as the scan did.
    const auto fold_gen = (basis == Basis::Pauli) ? pair_swap<NumModes>(gen) : gen;
    const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);

    // One combine_columns_block call over [0, mask_words): dense columns XOR-read over the read words
    // only; sparse columns lower_bound to rows < mask_words*64 and scatter just those (rows in words
    // >= mask_words are never read, so dropping them is exact). The odd-|G| row_parity and last-word
    // mask are still applied per-word in fold_word.
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

// The one fold-word mask rule, shared by the cached (fold_word) and recompute (recipe_fold_word)
// paths and matching even_parity_scan_pass1's per-word derivation: apply the odd-|G| row_parity(|M|)
// XOR correction, then the last-word scaled_count truncation mask. always_inline so codegen is identical
// to the inlined form it replaces.
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

// Visit each set bit of `bits` in ascending order, calling op(operator_index) with the absolute
// index base + bit_position. The single bit-scatter kernel behind every cos scale/accumulate loop
// (fold, fold-recompute, and stored-word-list); always_inline so the per-bit op has no call overhead
// and the generated code matches the hand-written popcount loop it replaces.
template <typename BitOp>
[[gnu::always_inline]] inline void for_each_cos_index(size_t base, uint64_t bits, BitOp op) {
    while (bits) {
        op(base + static_cast<size_t>(std::countr_zero(bits)));
        bits &= bits - 1;
    }
}

template <size_t NumModes>
void scale_cos_cached(const FoldCache<NumModes> &p, double *coeff, double cos_val) {
    const size_t mask_words = p.fold.mask_words;
    for (size_t wi = 0; wi < mask_words; ++wi) {
        for_each_cos_index(wi * 64, fold_word<NumModes>(p, wi), [&](size_t i) { coeff[i] *= cos_val; });
    }
}

template <size_t NumModes>
double accumulate_cos_cached(const FoldCache<NumModes> &p, double *state, double *ham, double cos_val, double sec_val) {
    const size_t mask_words = p.fold.mask_words;
    double loc = 0.0;
    for (size_t wi = 0; wi < mask_words; ++wi) {
        for_each_cos_index(wi * 64, fold_word<NumModes>(p, wi), [&](size_t i) {
            loc += state[i] * ham[i];
            ham[i] *= sec_val;
            state[i] *= cos_val;
        });
    }
    return loc;
}

// ---- fold RECOMPUTE (no per-layer cache buffer) ----
// Above the fold-cache memory budget the eval recomputes each layer's fold on the fly instead of
// holding a `mask_words`-word buffer per layer (that cache is multi-GB for large operators × many
// generators, and being cold its own streaming dominates). A LazyFold stores only the generator's
// inverted index column indices + cos metadata. The recompute is fused with the scatter and parallelised
// over disjoint fold-word ranges (disjoint words → disjoint operator indices → race-free; XOR is
// associative so the per-word fold is byte-identical to make_fold_cache's combine). Each thread
// cache-blocks its range into kColumnBlockWords-word (L1-resident) sub-blocks so the fold is produced
// and consumed in-cache, avoiding the full-width scratch memset + readback the cache build pays.

/// Fold-cache memory budget in bytes. If the persistent per-layer fold cache (Σ mask_words · 8 B)
/// would exceed this, the functional switches to fold recompute (make_lazy_fold +
/// *_cos_fold_recompute), trading a small, largely bandwidth-hidden per-eval recompute for dropping a
/// multi-GB cold cache. Override with the `monoprop_RECOMPUTE_CACHE_MAX_MB` env var (0 ⇒ always
/// recompute); the 2048 MB default caps cache memory while keeping recompute rare.
inline auto recompute_cache_budget_bytes() -> size_t {
    // Parsed once in config::get(); the MB→bytes scaling stays here (0 MB ⇒ 0 ⇒ always recompute).
    return config::get().recompute_cache_max_mb * size_t{1024} * 1024;
}

/// Metadata to recompute a layer's cosine fold on the fly (used above the fold-cache budget): the
/// generator's ≤|G| inverted index column indices plus the cos truncation bounds — no per-layer buffer.
template <size_t NumModes>
struct LazyFold {
    EvenParityGeneratorColumns<NumModes> columns{}; // the generator's ≤|G| inverted index column indices
    FoldMask fold;
};

template <size_t NumModes>
auto make_lazy_fold(const InvertedIndex<NumModes> &sc,
                    const MajoranaSet<NumModes> &gen,
                    uint64_t scaled_count,
                    Basis basis = Basis::Majorana) -> LazyFold<NumModes> {
    LazyFold<NumModes> r;
    r.fold = make_fold_mask<NumModes>(sc, gen, scaled_count, basis);
    // generator_words stores the REAL G; re-derive J(G) here for Pauli exactly as the scan did.
    const auto fold_gen = (basis == Basis::Pauli) ? pair_swap<NumModes>(gen) : gen;
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

// ---- parallel/serial scale & accumulate over a CosMask ----
// Build- and pare-produced lists have 64-aligned, disjoint blocks → the block-range split is
// race-free (parallel). The combined fold path no longer materialises a CosMask,
// so only the parallel overload is needed.
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

// ---- fold → CosMask / index vector ----
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
