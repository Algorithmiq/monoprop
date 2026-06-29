#pragma once

// CosineRecompute.h — recompute a layer's cosine index set from the persistent even-parity sidecar
// instead of reading a stored per-layer bitmap.
//
// `cos` for a layer = the operator terms anticommuting with that layer's generator G = the per-word
// XOR-fold of G's even-parity sidecar columns (fold_overlap_word), with the odd-|G| row_parity(|M|)
// correction, truncated to the first `cos_count` operator indices (the fold-truncation bound = the
// operator term count BEFORE that layer's own inserts).
//
// PreparedFold caches, once per functional, everything needed to recompute one layer's cos:
// the hoisted fold columns and the (possibly odd-|G|) parity pointer. The fold is then identical to
// even_parity_scan_pass1's per-word fold, masked at the cos_count boundary.
//
// SCRATCH OWNERSHIP (important): the build-time prepare_fold (DistributedLayerBuilder.h) returns a
// BuildFoldGuard whose sparse-column pointers point into per-thread `thread_local` scratch, and whose
// destructor RESETS that scratch. It therefore CANNOT be cached for the functional's lifetime. So
// PreparedFold does NOT store a BuildFoldGuard; it builds a self-owned FoldColumns: dense sidecar
// columns are referenced by pointer (the sidecar is persistent for the simulator's lifetime), and
// the sparse columns are materialized once into owned word buffers held inside this struct. The
// stored `const uint64_t*` fold pointers then stay valid as long as the PreparedFold lives.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "monoprop/Threading.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"       // LayerCosScale, LayerCosAccumulate
#include "monoprop/detail/evolution/DistributedLayerBuilder.h"        // FoldColumns, fold_overlap_word, gen columns, sidecar

namespace monoprop::detail {

// ---- prepared fold per layer (built once per functional) ----
// Self-owned: see SCRATCH OWNERSHIP above. The fold's dense columns point into the persistent
// sidecar; the materialized sparse columns live in `owned_words` and the fold pointers reference
// those owned buffers, so the fold stays valid for this object's lifetime.
template <size_t NumModes>
struct PreparedFold {
    FoldColumns<NumModes> fold{};                  // hoisted dense / owned-materialized columns
    const uint64_t *pivot_ptr = nullptr;           // leader/follower pivot column (dense ptr or owned)
    std::vector<std::vector<uint64_t>> owned_words; // materialized sparse columns (kept alive)
    const uint64_t *row_parity = nullptr;          // null for even |G|
    bool g_odd = false;
    size_t cos_words = 0; // min(sidecar words, ceil(cos_count/64))
    size_t last_word = 0;
    uint64_t last_mask = ~uint64_t{0};
};

template <size_t NumModes>
auto make_prepared_fold(const EvenParityMajoranaScanSidecar<NumModes> &sc,
                              const MajoranaSet<NumModes> &gen,
                              uint64_t cos_count) -> PreparedFold<NumModes> {
    PreparedFold<NumModes> p;
    p.g_odd = (gen.count() % 2 != 0);
    const auto gen_columns = build_even_parity_generator_columns<NumModes>(gen);
    if (p.g_odd) {
        const_cast<EvenParityMajoranaScanSidecar<NumModes> &>(sc).ensure_row_parity();
    }

    const size_t full = sc.words();
    p.cos_words = std::min(full, static_cast<size_t>((cos_count + 63) / 64));
    p.last_word = (p.cos_words == 0) ? 0 : p.cos_words - 1;
    p.last_mask = (cos_count % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (cos_count % 64)) - 1);

    if (p.g_odd) {
        p.row_parity = sc.row_parity_word_ptr();
    }

    if (gen_columns.count == 0 || full == 0) {
        return p; // empty fold → recompute yields no cos (matches build's early return)
    }

    // Materialize one sparse column into an owned word buffer (XOR-scatter of its set rows).
    auto materialize_sparse = [&](size_t col) -> const uint64_t * {
        p.owned_words.emplace_back(full, 0);
        std::vector<uint64_t> &buf = p.owned_words.back();
        for (TermIndex r : sc.sparse_column_rows(col)) {
            buf[r >> 6] ^= (uint64_t{1} << (r & 63U));
        }
        return buf.data();
    };

    // owned_words must not reallocate after we hand out .data() pointers; reserve worst case.
    p.owned_words.reserve(p.owned_words.size() + gen_columns.count);

    FoldColumns<NumModes> &f = p.fold;
    f.count = 0;
    // Pivot column (gen_columns.indices[0]) — also folded into the overlap, matching prepare_fold.
    const size_t pivot_col = gen_columns.indices[0];
    if (sc.column_is_dense(pivot_col)) {
        p.pivot_ptr = sc.dense_column_data(pivot_col);
    } else {
        p.pivot_ptr = materialize_sparse(pivot_col);
    }
    f.ptrs[f.count++] = p.pivot_ptr;

    for (size_t k = 1; k < gen_columns.count; ++k) {
        const size_t c = gen_columns.indices[k];
        if (sc.column_is_dense(c)) {
            f.ptrs[f.count++] = sc.dense_column_data(c);
        } else {
            f.ptrs[f.count++] = materialize_sparse(c);
        }
    }
    return p;
}

template <size_t NumModes>
[[gnu::always_inline]] inline uint64_t fold_cos_word(const PreparedFold<NumModes> &p, size_t wi) {
    uint64_t overlap = fold_overlap_word<NumModes>(p.fold, wi);
    if (p.g_odd) {
        overlap ^= p.row_parity[wi];
    }
    if (wi == p.last_word) {
        overlap &= p.last_mask;
    }
    return overlap;
}

template <size_t NumModes>
void scale_cos_fold(const PreparedFold<NumModes> &p, double *coeff, double cos_val) {
    threading::parallel_for_ranges(
        p.cos_words,
        [&](size_t b, size_t e) {
            for (size_t wi = b; wi < e; ++wi) {
                uint64_t bits = fold_cos_word<NumModes>(p, wi);
                const size_t base = wi * 64;
                while (bits) {
                    coeff[base + std::countr_zero(bits)] *= cos_val;
                    bits &= bits - 1;
                }
            }
        },
        threading::range_grain_size(p.cos_words, 1));
}

template <size_t NumModes>
double accumulate_cos_fold(const PreparedFold<NumModes> &p,
                           double *state,
                           double *ham,
                           double cos_val,
                           double sec_val) {
    return threading::parallel_reduce_ranges(
        p.cos_words,
        0.0,
        [&](size_t b, size_t e, double loc) {
            for (size_t wi = b; wi < e; ++wi) {
                uint64_t bits = fold_cos_word<NumModes>(p, wi);
                const size_t base = wi * 64;
                while (bits) {
                    const size_t i = base + std::countr_zero(bits);
                    loc += state[i] * ham[i];
                    ham[i] *= sec_val;
                    state[i] *= cos_val;
                    bits &= bits - 1;
                }
            }
            return loc;
        },
        [](double a, double c) { return a + c; },
        threading::range_grain_size(p.cos_words, 1));
}

// ---- parallel/serial scale & accumulate over a CosineWordList ----
// Build- and pare-produced lists have 64-aligned, disjoint blocks → the block-range split is
// race-free (parallel). The combined fold path no longer materialises a CosineWordList,
// so only the parallel overload is needed.
inline void scale_cos_words(double *coeff, const CosineWordList &cos, double cos_val) {
    const size_t n = cos.blocks.size();
    threading::parallel_for_ranges(
        n,
        [&](size_t b, size_t e) {
            for (size_t k = b; k < e; ++k) {
                const auto [base, bits0] = cos.blocks[k];
                uint64_t bits = bits0;
                while (bits) { coeff[base + static_cast<size_t>(std::countr_zero(bits))] *= cos_val; bits &= bits - 1; }
            }
        },
        threading::range_grain_size(n, 1));
}
inline double accumulate_cos_words(double *state, double *ham, const CosineWordList &cos,
                                   double cos_val, double sec_val) {
    const size_t n = cos.blocks.size();
    return threading::parallel_reduce_ranges(
        n, 0.0,
        [&](size_t b, size_t e, double loc) {
            for (size_t k = b; k < e; ++k) {
                const auto [base, bits0] = cos.blocks[k];
                uint64_t bits = bits0;
                while (bits) {
                    const size_t i = base + static_cast<size_t>(std::countr_zero(bits));
                    loc += state[i] * ham[i]; ham[i] *= sec_val; state[i] *= cos_val; bits &= bits - 1;
                }
            }
            return loc;
        },
        [](double a, double c) { return a + c; },
        threading::range_grain_size(n, 1));
}


// ---- per-layer cos source for replaying a graph whose layers no longer store a CosineWordList ----
// One entry per graph layer. A layer is one of:
//   - FOLD: the main build path recomputes cos from the sidecar fold — left recompute metadata
//     (generator_words + cos_count); reproduce the full cosine set from the persistent sidecar fold.
//   - WORDS: the layer carries a stored CosineWordList (pruned/filtered layers); materialize it
//     verbatim as an absolute (base,bits) word list.
struct LayerCosEntry final {
    // fold is a heavy template; hold it type-erased via the scale/accumulate closures below so this
    // struct stays NumModes-agnostic for the caches that store a vector of these.
    std::function<void(double *, double)> scale;                          // (coeff, cos_val)
    std::function<double(double *, double *, double, double)> accumulate; // (state, ham, cos, sec)
};

template <size_t NumModes>
inline auto make_fold_layer_entry(const EvenParityMajoranaScanSidecar<NumModes> &sc,
                                  const std::vector<uint64_t> &gw,
                                  uint64_t cos_count) -> LayerCosEntry {
    MajoranaSet<NumModes> gen{};
    std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
    auto fold = std::make_shared<PreparedFold<NumModes>>(make_prepared_fold<NumModes>(sc, gen, cos_count));
    LayerCosEntry entry;
    entry.scale = [fold](double *c, double v) { scale_cos_fold<NumModes>(*fold, c, v); };
    entry.accumulate = [fold](double *s, double *h, double v, double sec) {
        return accumulate_cos_fold<NumModes>(*fold, s, h, v, sec);
    };
    return entry;
}

// Build a per-layer cos cache for `graph`, recomputing layers from `sidecar` (main build path) and
// reading the stored CosineWordList for layers that carry one. `graph` must expose layers()/get_layer(i) with a
// Layer interface (cos_data()/generator_words()/cos_count()). Used by the replay paths that do not
// (yet) get their cos via build-time transient words (e.g. contract_partially).
template <size_t NumModes, typename GraphLike>
inline auto build_layer_cos_cache(const EvenParityMajoranaScanSidecar<NumModes> &sidecar, const GraphLike &graph)
    -> std::shared_ptr<std::vector<LayerCosEntry>> {
    auto cache = std::make_shared<std::vector<LayerCosEntry>>();
    cache->reserve(graph.layers());
    for (size_t i = 0; i < graph.layers(); ++i) {
        const auto &layer = graph.get_layer(i);
        LayerCosEntry entry;
        if (layer.generator_words().empty()) {
            // No recompute metadata (empty cos with no generator ⇒ genuinely empty cosine set).
            entry.scale = [](double *, double) {};
            entry.accumulate = [](double *, double *, double, double) { return 0.0; };
        } else {
            // Main build path recomputes cos from the sidecar fold.
            entry = make_fold_layer_entry<NumModes>(sidecar, layer.generator_words(), layer.cos_count());
        }
        cache->push_back(std::move(entry));
    }
    return cache;
}

// Wrap a LayerCosEntry cache into the per-layer scale / accumulate callbacks evolve consumes.
inline auto make_cos_scale_from_cache(const std::shared_ptr<std::vector<LayerCosEntry>> &cache) -> LayerCosScale {
    return [cache](size_t i, double *c, double v) { (*cache)[i].scale(c, v); };
}
inline auto make_cos_acc_from_cache(const std::shared_ptr<std::vector<LayerCosEntry>> &cache) -> LayerCosAccumulate {
    return [cache](size_t i, double *s, double *h, double v, double sec) { return (*cache)[i].accumulate(s, h, v, sec); };
}

// ---- fold → CosineWordList / index vector ----
template <size_t NumModes>
inline auto fold_to_cos_words(const PreparedFold<NumModes> &p) -> CosineWordList {
    CosineWordList c;
    for (size_t wi = 0; wi < p.cos_words; ++wi) {
        const uint64_t b = fold_cos_word<NumModes>(p, wi);
        if (b) { c.blocks.emplace_back(wi * 64, b); c.total_count += static_cast<size_t>(std::popcount(b)); }
    }
    return c;
}
template <size_t NumModes>
inline auto fold_to_cos_indices(const PreparedFold<NumModes> &p) -> VecZ {
    VecZ inds;
    for (size_t wi = 0; wi < p.cos_words; ++wi) {
        uint64_t b = fold_cos_word<NumModes>(p, wi);
        const size_t base = wi * 64;
        while (b) { inds.push_back(base + static_cast<size_t>(std::countr_zero(b))); b &= b - 1; }
    }
    return inds;
}

} // namespace monoprop::detail
