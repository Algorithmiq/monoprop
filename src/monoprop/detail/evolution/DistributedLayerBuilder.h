#pragma once

// DistributedLayerBuilder.h — paper's BuildDistributedLayer algorithm (mpi_details Alg. 2)
//
// build_distributed_layer<NumModes>() replaces the fragmented
// fastpath/mainline/half-ham build pipeline with the paper's single pass.
//
// C_{j,r} = ALL anticommuting local indices. ContractLayer uses pre-cos B snapshots,
// so replay is independent of iteration order and rank count.
//
// FindAnticommuting is a single parity-agnostic pass: the sidecar XOR-column fold + pivot-bit
// leader/follower split. Even generators use the plain fold; odd generators add the per-row
// parity(|M|) correction (g_odd) so the same kernel is correct for both parities.
//
// Emits a graph layer directly: returns std::shared_ptr<LayerCore> assembled in-pass from a
// uniform per-rank PartnerAcc, ready to append to the surrogate graph.
//
// Overview (paper Algorithm 2):
// 1) FindAnticommuting → partition A_r into leaders L_r and followers F_r.
// 2) Leader pass: apply cutoffs → route surviving queries to owner of M' = M_i⊕G.
//    * Local leaders (owner == my_rank): resolve inline.
//    * Remote leaders: exchange queries via MPI (round 1a), receive responses (round 1b).
// 3) Follower pass: iterate F_r \ matched → same exchange (rounds 2a, 2b).
// 4) Insert-on-miss in the resolver: absent partners targeting REMOTE ranks are inserted by the
//    owner during resolve_incoming_queries and their real index is returned in the same response
//    round (no separate half-ham broadcast / index-return rounds). Absent partners targeting THIS
//    rank (self-rank queries, resolved inline) are deferred and inserted after both passes, inside
//    build_distributed_layer itself — they never travel over the wire.

// Outcomes: cutoff applied → cosine only; otherwise → sine between ranks. A matched follower
// (found by its leader) is skipped — the leader already accounts for it.

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

#include <memory>
#include <span>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::detail {

// ─── PartnerAcc ────────────────────────────────────────────────────────────────
// Uniform per-rank rotation accumulator. The self slot is just the partner with
// in_:=(tgt,φ), out_:=(src,φ). Cross-rank: in_=resolver side, out_=querier side.
// Assembly (identical for every rank): b=[in.idx]++[out.idx];
// d=[{out.idx,−φ}]++[{in.idx,+φ}]; endpoints += every in.idx and out.idx.
struct PartnerAcc {
    // Default-init storage: resize-then-overwrite paths skip the serial zero-fill (better parallel
    // scaling); push_back/emplace are unaffected.
    DefaultInitVector<std::pair<size_t, int>> in_entries;  // (local_target_idx, phase)
    DefaultInitVector<std::pair<size_t, int>> out_entries; // (local_source_idx, phase)
};

// ─── Chunked-parallel helpers (order-preserving, sort-free) ────────────────────
// Split a contiguous index/word space into disjoint ascending chunks, process each into its OWN
// output slot in parallel, then concatenate in chunk order. The result is globally sorted with no
// comparison sort and is deterministic regardless of thread scheduling (unlike tbb::combinable).
// Operator uniqueness + the XOR involution mean each term lands in exactly one chunk and exactly one
// of leaders/followers, so concatenation needs neither dedup nor merge.

// Chunk count for an n-element pass: ~256 elements/chunk, capped at 16× workers, serial below 512 or
// on one thread. The hot consumer (resolve_self_queries) probes a multi-hundred-MB index map and is
// DRAM-latency bound, so fine chunks expose more independent probes to overlap. Grain/floor/cap are
// env-tunable (MONOPROP_PCC_DIV / MONOPROP_PCC_MIN / MONOPROP_PCC_CAP), mirroring MONOPROP_FA_*.
inline auto partition_chunk_count(size_t n) -> size_t {
    if (n == 0) {
        return 0;
    }
    const size_t p = effective_parallelism();
    static const size_t div = [] {
        const char *e = std::getenv("MONOPROP_PCC_DIV");
        return e ? std::strtoul(e, nullptr, 10) : 256UL;
    }();
    static const size_t minn = [] {
        const char *e = std::getenv("MONOPROP_PCC_MIN");
        return e ? std::strtoul(e, nullptr, 10) : 512UL;
    }();
    static const size_t cap = [] {
        const char *e = std::getenv("MONOPROP_PCC_CAP");
        return e ? std::strtoul(e, nullptr, 10) : 16UL;
    }();
    if (p <= 1 || n < minn) {
        return 1;
    }
    const size_t by_size = std::max<size_t>(1, n / div);
    return std::min<size_t>(p * cap, by_size);
}

// Chunk count for a WORD-space pass (the sidecar XOR-column scan in fused_find_and_collect): the
// parallel dimension is the operator's word count = ceil(terms/64). Each word is ~|G| XOR/popcount
// ops over 64 terms, so we target a fine ~kWordsPerChunk words/chunk and let the count climb to 4×
// workers. Grain + floor are env-tunable (MONOPROP_FA_WORDS_PER_CHUNK / MONOPROP_FA_MIN_WORDS).
inline auto fa_words_per_chunk() -> size_t {
    static const size_t v = [] {
        if (const char *e = std::getenv("MONOPROP_FA_WORDS_PER_CHUNK")) {
            const size_t parsed = static_cast<size_t>(std::strtoul(e, nullptr, 10));
            if (parsed > 0) {
                return parsed;
            }
        }
        return size_t{256};
    }();
    return v;
}

inline auto fa_min_parallel_words() -> size_t {
    static const size_t v = [] {
        if (const char *e = std::getenv("MONOPROP_FA_MIN_WORDS")) {
            const size_t parsed = static_cast<size_t>(std::strtoul(e, nullptr, 10));
            if (parsed > 0) {
                return parsed;
            }
        }
        return size_t{256};
    }();
    return v;
}

inline auto partition_chunk_count_words(size_t word_count) -> size_t {
    if (word_count == 0) {
        return 0;
    }
    const size_t p = effective_parallelism();
    if (p <= 1 || word_count < fa_min_parallel_words()) {
        return 1;
    }
    const size_t by_size = std::max<size_t>(1, word_count / fa_words_per_chunk());
    return std::min<size_t>(p * 4, by_size);
}

// Run body(chunk_idx, lo, hi) over `chunks` contiguous sub-ranges of [0, n) in parallel.
// simple_partitioner + grain 1 forces one task per chunk so each writes a distinct slot.
template <typename Body>
inline auto for_each_chunk(size_t n, size_t chunks, Body &&body) -> void {
    if (n == 0 || chunks == 0) {
        return;
    }
    if (chunks == 1) {
        body(size_t{0}, size_t{0}, n);
        return;
    }
    const size_t per = (n + chunks - 1) / chunks;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, chunks, 1),
        [&](const tbb::blocked_range<size_t> &br) {
            for (size_t c = br.begin(); c < br.end(); ++c) {
                const size_t lo = c * per;
                if (lo >= n) {
                    continue;
                }
                body(c, lo, std::min(n, lo + per));
            }
        },
        tbb::simple_partitioner{});
}

// ── Order-preserving parallel gather core ──────────────────────────────────────
// Append `n_parts` per-chunk vectors (part_at(c) -> std::vector<T>&) onto `dst` in chunk order:
// chunk c lands at [base + prefix(c), base + prefix(c+1)). Deterministic and byte-identical
// regardless of thread count (unlike tbb::combinable). Each part is freed as consumed. Large totals
// scatter one task per chunk (sole writer per slice, no atomics); small/serial use one append pass.
template <typename Vec, typename PartAt>
inline auto append_parts_in_order(Vec &dst, size_t n_parts, PartAt &&part_at) -> void {
    using T = typename Vec::value_type;
    if (n_parts == 0) {
        return;
    }
    std::vector<size_t> offsets(n_parts + 1, 0);
    for (size_t c = 0; c < n_parts; ++c) {
        offsets[c + 1] = offsets[c] + part_at(c).size();
    }
    const size_t total = offsets[n_parts];
    if (total == 0) {
        return;
    }
    // Single-chunk (serial-pass) fast path: steal the lone buffer outright when dst is empty.
    if (dst.empty() && n_parts == 1) {
        dst = std::move(part_at(0));
        Vec{}.swap(part_at(0));
        return;
    }
    // Small or single-threaded: one serial append pass (cheaper than spawning tasks).
    if (effective_parallelism() <= 1 || total < 4096) {
        dst.reserve(dst.size() + total);
        for (size_t c = 0; c < n_parts; ++c) {
            auto &part = part_at(c);
            dst.insert(dst.end(), part.begin(), part.end());
            Vec{}.swap(part);
        }
        return;
    }
    // Large: preallocate, then scatter each chunk into its disjoint slice in parallel.
    const size_t base = dst.size();
    dst.resize(base + total);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n_parts, 1),
        [&](const tbb::blocked_range<size_t> &br) {
            for (size_t c = br.begin(); c < br.end(); ++c) {
                auto &part = part_at(c);
                std::copy(part.begin(), part.end(), dst.begin() + static_cast<std::ptrdiff_t>(base + offsets[c]));
                Vec{}.swap(part);
            }
        },
        tbb::simple_partitioner{});
}

// Append per-chunk vectors onto an existing destination in chunk order (frees inputs). Used by phases
// that accumulate across multiple passes (e.g. leader then follower) where replacing dst is not possible.
template <typename Vec>
inline auto append_gathered_chunks(Vec &dst, std::vector<Vec> &parts) -> void {
    append_parts_in_order(dst, parts.size(), [&](size_t c) -> Vec & { return parts[c]; });
}

// Append chunk-local per-rank vectors into per-rank destinations: for each rank r, the chunks
// chunk_by_rank[*][r] are appended onto dst_by_rank[r] in chunk order.
template <typename Vec>
inline auto append_chunked_rank_vectors(std::vector<Vec> &dst_by_rank, std::vector<std::vector<Vec>> &chunk_by_rank)
    -> void {
    const size_t chunks = chunk_by_rank.size();
    const size_t rank_count = dst_by_rank.size();
    if (chunks == 0 || rank_count == 0) {
        return;
    }
    for (size_t r = 0; r < rank_count; ++r) {
        append_parts_in_order(dst_by_rank[r], chunks, [&](size_t c) -> Vec & { return chunk_by_rank[c][r]; });
    }
}

// ─── concat_cosine_word_blocks ────────────────────────────────────────────────
// Concatenate disjoint, ascending CosineWordList blocks (per-chunk cosine, chunk order). The
// single-non-empty-block case is moved out wholesale (zero copy).
inline auto concat_cosine_word_blocks(const std::vector<CosineWordList *> &blocks) -> CosineWordList {
    CosineWordList out;
    CosineWordList *only = nullptr;
    size_t non_empty = 0, total_blocks = 0;
    for (auto *b : blocks) {
        if (!b->empty()) {
            only = b;
            ++non_empty;
            total_blocks += b->blocks.size();
        }
    }
    if (non_empty <= 1) {
        return only ? std::move(*only) : out;
    }
    out.blocks.reserve(total_blocks);
    for (auto *b : blocks) {
        out.total_count += b->total_count;
        out.blocks.insert(out.blocks.end(), b->blocks.begin(), b->blocks.end());
    }
    return out;
}

// ─── Even-parity scan + cutoff-state helpers ───────────────────────────────────
// The cutoff state read by the fused scan and the even-parity generator-column scan it uses.
struct MajoranaEvolutionCutoffState {
    CutoffContext cutoff_ctx;
};

inline auto build_majorana_evolution_cutoff_state(const std::optional<double> &atol,
                                                  std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                                                  const std::optional<double> &upper_atol,
                                                  const std::optional<double> &param) -> MajoranaEvolutionCutoffState {
    const bool check_atol = atol.has_value() && local_coeffs.has_value() && param.has_value();
    const bool check_upper_atol = upper_atol.has_value() && local_coeffs.has_value();
    const double sin_val = param.has_value() ? std::sin(2 * param.value()) : 1.0;
    const double cos_val = param.has_value() ? std::cos(2 * param.value()) : 1.0;

    return {
        .cutoff_ctx = CutoffContext{.check_atol = check_atol,
                                    .check_upper_atol = check_upper_atol,
                                    .atol_value = atol.value_or(0.0),
                                    .upper_atol_value = upper_atol.value_or(0.0),
                                    .abs_sin_val = std::abs(sin_val),
                                    .abs_cos_val = std::abs(cos_val),
                                    .use_coeff_checks = check_atol || check_upper_atol},
    };
}

template <size_t NumModes>
struct EvenParityGeneratorColumns {
    std::array<size_t, MajoranaSet<NumModes>::size()> indices{};
    size_t count = 0;
};

template <size_t NumModes>
auto build_even_parity_generator_columns(const MajoranaSet<NumModes> &gen_maj) -> EvenParityGeneratorColumns<NumModes> {
    EvenParityGeneratorColumns<NumModes> columns;
    for (size_t bit_idx = gen_maj.find_first(); bit_idx < gen_maj.size(); bit_idx = gen_maj.find_next(bit_idx)) {
        columns.indices[columns.count++] = bit_idx;
    }
    return columns;
}

// ─── Sidecar column-fold (pointer-hoisted, vectorizable) ──────────────────────
// The even-parity scan XOR-folds the generator's selected sidecar columns into a per-word overlap
// mask. Hoisting the column base pointers once turns the inner fold into a stride-0 walk over raw
// contiguous uint64_t arrays the compiler can vectorize. Columns don't resize mid-scan, so caching
// `.data()` is safe.
template <size_t NumModes>
struct FoldColumns {
    std::array<const uint64_t *, MajoranaSet<NumModes>::size()> ptrs{};
    size_t count = 0;
};

// Hoisted fold pointers + leader/follower pivot pointer for an even generator, handling SPARSE sidecar
// columns transparently: dense columns contribute their word array directly; sparse columns are
// scatter-XOR'd into a reused dense scratch appended as ONE column (so the fold reads ≤|G| columns and
// even_parity_scan_pass1 stays byte-identical). The pivot gets its own scratch if sparse. The guard
// owns the scratches and zeroes the touched words on destruction for reuse. Built once per generator
// on the build thread; the scratches are thread_local, so the parallel chunk scan reads them race-free.
template <size_t NumModes>
struct BuildFoldGuard {
    FoldColumns<NumModes> fold{};
    const uint64_t *pivot_ptr = nullptr;
    std::vector<uint64_t> *combined = nullptr;
    std::vector<uint64_t> *pivot = nullptr;
    std::vector<size_t> *combined_touched = nullptr;
    std::vector<size_t> *pivot_touched = nullptr;

    BuildFoldGuard() = default;
    BuildFoldGuard(const BuildFoldGuard &) = delete;
    auto operator=(const BuildFoldGuard &) -> BuildFoldGuard & = delete;
    auto operator=(BuildFoldGuard &&) -> BuildFoldGuard & = delete;
    BuildFoldGuard(BuildFoldGuard &&o) noexcept
        : fold(o.fold),
          pivot_ptr(o.pivot_ptr),
          combined(o.combined),
          pivot(o.pivot),
          combined_touched(o.combined_touched),
          pivot_touched(o.pivot_touched) {
        o.combined = nullptr;
        o.pivot = nullptr; // moved-from destructor must not re-clear the shared scratch
    }
    ~BuildFoldGuard() {
        if (pivot != nullptr) {
            for (size_t w : *pivot_touched) {
                (*pivot)[w] = 0;
            }
            pivot_touched->clear();
        }
        if (combined != nullptr) {
            for (size_t w : *combined_touched) {
                (*combined)[w] = 0;
            }
            combined_touched->clear();
        }
    }
};

template <size_t NumModes>
inline auto prepare_fold(const EvenParityMajoranaScanSidecar<NumModes> &sidecar,
                         const EvenParityGeneratorColumns<NumModes> &gen_columns) -> BuildFoldGuard<NumModes> {
    static thread_local std::vector<uint64_t> combined_scratch;
    static thread_local std::vector<uint64_t> pivot_scratch;
    static thread_local std::vector<size_t> combined_touched;
    static thread_local std::vector<size_t> pivot_touched;

    const size_t word_count = sidecar.words();
    BuildFoldGuard<NumModes> pf;
    FoldColumns<NumModes> &f = pf.fold;
    f.count = 0;

    const size_t pivot_col = gen_columns.indices[0];
    if (sidecar.column_is_dense(pivot_col)) {
        pf.pivot_ptr = sidecar.dense_column_data(pivot_col);
        f.ptrs[f.count++] = pf.pivot_ptr; // pivot column also folded into the overlap
    }
    else {
        if (pivot_scratch.size() < word_count) {
            pivot_scratch.resize(word_count, 0);
        }
        for (TermIndex r : sidecar.sparse_column_rows(pivot_col)) {
            const size_t w = r >> 6;
            if (pivot_scratch[w] == 0) {
                pivot_touched.push_back(w);
            }
            pivot_scratch[w] |= uint64_t{1} << (r & 63U);
        }
        pf.pivot_ptr = pivot_scratch.data();
        f.ptrs[f.count++] = pf.pivot_ptr;
        pf.pivot = &pivot_scratch;
        pf.pivot_touched = &pivot_touched;
    }

    bool any_sparse = false;
    for (size_t k = 1; k < gen_columns.count; ++k) {
        const size_t c = gen_columns.indices[k];
        if (sidecar.column_is_dense(c)) {
            f.ptrs[f.count++] = sidecar.dense_column_data(c);
        }
        else {
            if (combined_scratch.size() < word_count) {
                combined_scratch.resize(word_count, 0);
            }
            for (TermIndex r : sidecar.sparse_column_rows(c)) {
                const size_t w = r >> 6;
                if (combined_scratch[w] == 0) {
                    combined_touched.push_back(w);
                }
                combined_scratch[w] ^= uint64_t{1} << (r & 63U);
            }
            any_sparse = true;
        }
    }
    if (any_sparse) {
        f.ptrs[f.count++] = combined_scratch.data();
        pf.combined = &combined_scratch;
        pf.combined_touched = &combined_touched;
    }
    return pf;
}

// Single-word fold: XOR the generator's columns at word `wi`. Hot inner of pass 1.
template <size_t NumModes>
[[gnu::always_inline]] inline auto fold_overlap_word(const FoldColumns<NumModes> &f, size_t wi) -> uint64_t {
    uint64_t overlap = 0;
    for (size_t k = 0; k < f.count; ++k) {
        overlap ^= f.ptrs[k][wi];
    }
    return overlap;
}

// One nonzero-overlap word carried from the memory-bound scan (pass 1) to the emit (pass 2) in the
// even-parity sidecar scan. `foll` is the follower sub-mask (overlap & pivot); leaders are
// `overlap ^ foll` (disjoint). Used by fused_find_and_collect.
struct EvenParityNzWord {
    size_t base;
    uint64_t overlap;
    uint64_t foll;
};

// Even-parity scan pass 1 (memory-bound): over words [wlo,whi), XOR-fold the hoisted sidecar columns
// into a per-word overlap mask, keep nonzero words in `nz`, and tally popcounts (n_anti, n_foll) so
// pass 2 reserves its output runs once. Pass a thread_local `nz` to reuse its capacity across chunks.
// `g_odd` carries the odd-|G| correction: the anticommutation bit is (|M∩G| mod 2) XOR (|M| mod 2),
// so the per-row parity(|M|) bit (row_parity_ptr) is XORed in before foll/nonzero/pivot are derived.
// Even |G| (g_odd==false) ignores row_parity_ptr and is byte-identical.
template <size_t NumModes>
[[gnu::always_inline]] inline auto even_parity_scan_pass1(const FoldColumns<NumModes> &fold,
                                                          const uint64_t *pivot_ptr,
                                                          size_t wlo,
                                                          size_t whi,
                                                          size_t last_word,
                                                          uint64_t last_word_mask,
                                                          bool g_odd,
                                                          const uint64_t *row_parity_ptr,
                                                          std::vector<EvenParityNzWord> &nz,
                                                          size_t &n_anti,
                                                          size_t &n_foll) -> void {
    nz.clear();
    n_anti = 0;
    n_foll = 0;
    for (size_t wi = wlo; wi < whi; ++wi) {
        uint64_t overlap = fold_overlap_word(fold, wi);
        if (g_odd) {
            overlap ^= row_parity_ptr[wi];
        }
        if (wi == last_word) {
            overlap &= last_word_mask;
        }
        if (!overlap) {
            continue;
        }
        const uint64_t foll = overlap & pivot_ptr[wi];
        n_anti += static_cast<size_t>(std::popcount(overlap));
        n_foll += static_cast<size_t>(std::popcount(foll));
        nz.push_back(EvenParityNzWord{wi * 64, overlap, foll});
    }
}

// ─── Query serialization ──────────────────────────────────────────────────────
// Queries are exchanged as flat VecZ buffers: every kQueryWords elements = one query.
//   [word_0..word_{W-1}(new_maj), phase_as_uint64]
// The querier's source index is NOT in the payload: the resolver returns the partner by position and
// the querier holds the source in its parallel src list (src_idx_r[r][q]). Dropping it trims the
// query word count → less alltoallv volume and local buffer traffic.
template <size_t NumModes>
inline constexpr size_t kQueryWords = mpi_detail::kWords<NumModes> + 1; // +phase

template <size_t NumModes>
inline auto query_push(VecZ &buf, const MajoranaSet<NumModes> &maj, int phase) -> void {
    mpi_detail::append_majorana_words<NumModes>(maj, buf);
    buf.push_back(static_cast<size_t>(static_cast<unsigned int>(phase)));
}

template <size_t NumModes>
inline auto query_read(const VecZ &buf, size_t q, MajoranaSet<NumModes> &maj_out, int &phase_out) -> void {
    const size_t base = q * kQueryWords<NumModes>;
    maj_out = mpi_detail::read_majorana_from_words<NumModes>(buf, base);
    phase_out = static_cast<int>(static_cast<unsigned int>(buf[base + mpi_detail::kWords<NumModes>]));
}

// ─── Rotation gate (shared semantics) ─────────────────────────────────────────
// The per-term rotation gate splits into a DYNAMIC part (depends on current coeffs/param: orbital pop
// cap, upper-atol freeze, lower-atol sine cutoff) and a STATIC part (structural cutoff on M' = M⊕G,
// via CutoffEvaluator::passes_with_popcount). Every emitting path MUST use these helpers so the
// gate semantics cannot drift between paths.
inline auto rotation_dynamic_gate(int only_rotate_len_k, size_t maj_pop, const CutoffContext &ctx, double abs_c)
    -> bool {
    if (only_rotate_len_k > 0 && maj_pop > static_cast<size_t>(only_rotate_len_k)) {
        return false;
    }
    if (ctx.is_below_sin(abs_c)) {
        return false;
    }
    return true;
}

// ─── Rebuild-then-word-kernels emit (packed survivor products) ────────────────
// Per-generator context: the dense generator plus its popcount, built once per generator.
template <size_t NumModes>
struct GenEmitContext {
    const MajoranaSet<NumModes> &gen;
    size_t gen_pop;
};

template <size_t NumModes>
inline auto make_gen_emit_context(const MajoranaSet<NumModes> &gen, size_t gen_pop) -> GenEmitContext<NumModes> {
    return GenEmitContext<NumModes>{gen, gen_pop};
}

// Compute the three per-survivor products the cutoff/phase emit needs for term i:
//   new_maj  = M_i ⊕ G        (the rotated partner term that gets pushed as a query)
//   overlap  = |M_i ∩ G|      (feeds the new-popcount and the hermitian phase)
//   interleave = (−1)^x,  x = #{(m∈M_i, g∈G) : m<g}   (the interleave factor of the multiplicative phase)
//
// REBUILD-THEN-WORD-KERNELS: stream the term's stored ascending position list once to rebuild M_i as
// a dense W-word MajoranaSet in registers (k OR-shifts), then evaluate all three products with the
// branch-free W-word kernels (XOR, AND-popcount, prefix-xor interleave). The dynamic rotation gate
// (O(1) from the count byte) is applied by the caller BEFORE this kernel, so rejected terms cost
// zero reconstruction.
template <size_t NumModes>
[[gnu::always_inline]] inline void emit_term_products(const OperatorIndex<NumModes> &ham,
                                                      size_t i,
                                                      const GenEmitContext<NumModes> &ctx,
                                                      MajoranaSet<NumModes> &new_maj,
                                                      size_t &overlap,
                                                      int &interleave) {
    MajoranaSet<NumModes> maj; // zero-init, W words, lives in registers
    ham.for_each_position(i, [&](size_t pos) { maj.set(pos); });
    new_maj = maj ^ ctx.gen;
    overlap = maj.count_and(ctx.gen);
    interleave = interleave_phase<NumModes>(maj, ctx.gen);
}

// ─── fused_find_and_collect (any rank count) ──────────────────────────────────
// One pass over the operator fusing FindAnticommuting + apply_cutoffs: classify each anticommuting
// term leader/follower (sidecar XOR-column fold + pivot bit), compress it into the cosine block, and
// in the SAME walk apply the cutoffs and emit the surviving rotation query into its per-rank stream.
struct FusedScanResult {
    std::vector<CosineWordList> cos_blocks;        // ascending, disjoint, chunk order
    std::vector<VecZ> leader_queries;              // size R: serialized leader queries per owner rank
    std::vector<std::vector<size_t>> leader_src;   // size R: parallel to leader_queries (source op idx)
    std::vector<VecZ> follower_queries;            // size R: serialized follower queries per owner rank
    std::vector<std::vector<size_t>> follower_src; // size R: parallel to follower_queries
};

// Streams are routed to the owner of each partner M' = M⊕G (hash%R; self for single-rank, skipping
// the O(W) hash) and emitted in ascending source-index (chunk) order, so the downstream resolve and
// cross-rank index assignment are deterministic.
template <size_t NumModes>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const MajoranaSet<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const MajoranaEvolutionCutoffState &cut_st,
                            const VecD &coeffs,
                            int only_rotate_len_k,
                            size_t rank_count,
                            size_t my_rank) -> FusedScanResult {
    const size_t gen_pop = gen.count();
    const auto ectx = make_gen_emit_context<NumModes>(gen, gen_pop);

    // Cutoff + emit for one anticommuting term. Writes only the per-chunk per-rank sinks passed in
    // (safe under for_each_chunk). The dynamic gate (depends only on |M|) runs BEFORE
    // emit_term_products, so a gate-rejected term computes no products.
    auto emit = [&](size_t maj_pop,
                    size_t i,
                    bool is_follower,
                    std::vector<VecZ> &lq,
                    std::vector<std::vector<size_t>> &ls,
                    std::vector<VecZ> &fq,
                    std::vector<std::vector<size_t>> &fs) {
        const double abs_c = cut_st.cutoff_ctx.abs_coeff_for(i, coeffs);
        if (!rotation_dynamic_gate(only_rotate_len_k, maj_pop, cut_st.cutoff_ctx, abs_c)) {
            return;
        }
        MajoranaSet<NumModes> new_maj;
        size_t overlap = 0;
        int interleave = 0;
        emit_term_products<NumModes>(*op.op, i, ectx, new_maj, overlap, interleave);
        const size_t new_pop = maj_pop + gen_pop - 2 * overlap;
        // Structural cutoff on the partner M⊕G — UNLESS upper_atol rescues it (its sine coefficient is
        // large enough to keep alive despite exceeding the cutoff). See CutoffContext::is_above_upper.
        if (!cutoff_eval.passes_with_popcount(new_maj, new_pop) && !cut_st.cutoff_ctx.is_above_upper(abs_c)) {
            return;
        }
        const int phase = interleave * hermitian_phase(maj_pop, gen_pop, overlap);
        // Single rank: every partner is self-owned, skip the O(W) hash; multi-rank routes by owner.
        const size_t r_prime = (rank_count == 1) ? my_rank : (majorana_hash<NumModes>(new_maj) % rank_count);
        const size_t source = i;
        if (is_follower) {
            query_push<NumModes>(fq[r_prime], new_maj, phase);
            fs[r_prime].push_back(source);
        }
        else {
            query_push<NumModes>(lq[r_prime], new_maj, phase);
            ls[r_prime].push_back(source);
        }
    };

    FusedScanResult res;
    res.leader_queries.assign(rank_count, VecZ{});
    res.leader_src.assign(rank_count, std::vector<size_t>{});
    res.follower_queries.assign(rank_count, VecZ{});
    res.follower_src.assign(rank_count, std::vector<size_t>{});

    {
        // Odd |G| needs the per-row parity(|M|) correction (see even_parity_scan_pass1); even |G| is
        // byte-identical with no parity bitmap.
        const bool g_odd = (gen.count() % 2 != 0);
        const auto gen_columns = build_even_parity_generator_columns<NumModes>(gen);
        if (gen_columns.count == 0) {
            return res;
        }
        const auto &sidecar = op.even_parity_scan_sidecar();
        const size_t word_count = sidecar.words();
        if (word_count == 0) {
            return res;
        }
        // Build the row_parity bitmap once, only for odd generators (even workloads never allocate it).
        if (g_odd) {
            const_cast<EvenParityMajoranaScanSidecar<NumModes> &>(sidecar).ensure_row_parity();
        }
        const uint64_t *const row_parity_ptr = g_odd ? sidecar.row_parity_word_ptr() : nullptr;
        const size_t n = op.op->size();
        const size_t last_word = word_count - 1;
        const uint64_t last_word_mask = (n % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (n % 64)) - 1);
        // Hoist generator column base pointers once (expanding sparse columns into scratch; see
        // prepare_fold / FoldColumns).
        const BuildFoldGuard<NumModes> prepared = prepare_fold<NumModes>(sidecar, gen_columns);
        const FoldColumns<NumModes> &fold = prepared.fold;
        const uint64_t *const pivot_ptr = prepared.pivot_ptr;

        const size_t chunks = partition_chunk_count_words(word_count);
        std::vector<CosineWordList> ch_cos(chunks);
        std::vector<std::vector<VecZ>> ch_lq(chunks, std::vector<VecZ>(rank_count));
        std::vector<std::vector<std::vector<size_t>>> ch_ls(chunks, std::vector<std::vector<size_t>>(rank_count));
        std::vector<std::vector<VecZ>> ch_fq(chunks, std::vector<VecZ>(rank_count));
        std::vector<std::vector<std::vector<size_t>>> ch_fs(chunks, std::vector<std::vector<size_t>>(rank_count));
        for_each_chunk(word_count, chunks, [&](size_t c, size_t wlo, size_t whi) {
            auto &cos = ch_cos[c];
            auto &lq = ch_lq[c];
            auto &ls = ch_ls[c];
            auto &fq = ch_fq[c];
            auto &fs = ch_fs[c];

            // Pass 1 (memory-bound): see even_parity_scan_pass1. `nz` is thread_local to reuse capacity.
            thread_local std::vector<EvenParityNzWord> nz;
            size_t n_anti = 0;
            size_t n_foll = 0;
            even_parity_scan_pass1<NumModes>(fold,
                                             pivot_ptr,
                                             wlo,
                                             whi,
                                             last_word,
                                             last_word_mask,
                                             g_odd,
                                             row_parity_ptr,
                                             nz,
                                             n_anti,
                                             n_foll);
            if (rank_count == 1) {
                lq[my_rank].reserve((n_anti - n_foll) * kQueryWords<NumModes>);
                ls[my_rank].reserve(n_anti - n_foll);
                fq[my_rank].reserve(n_foll * kQueryWords<NumModes>);
                fs[my_rank].reserve(n_foll);
            }
            // Pass 2: collect cosine for EVERY anticommuting term, then apply cutoff + emit the query.
            // No orbital gate → store each nz word's full overlap whole (push_word); orbital gate →
            // per-index (push_index, ascending).
            const bool word_aligned_cos = only_rotate_len_k == 0;
            CosineWordBuilder cos_b;
            for (const auto &w : nz) {
                if (word_aligned_cos) {
                    cos_b.push_word(w.base, w.overlap);
                }
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const size_t maj_pop = op.op->popcount(i);
                    if (only_rotate_len_k > 0 && maj_pop > static_cast<size_t>(only_rotate_len_k)) {
                        continue;
                    }
                    if (!word_aligned_cos) {
                        cos_b.push_index(i);
                    }
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(maj_pop, i, is_follower, lq, ls, fq, fs);
                }
            }
            cos = cos_b.finish();
        });
        res.cos_blocks = std::move(ch_cos);
        append_chunked_rank_vectors(res.leader_queries, ch_lq);
        append_chunked_rank_vectors(res.leader_src, ch_ls);
        append_chunked_rank_vectors(res.follower_queries, ch_fq);
        append_chunked_rank_vectors(res.follower_src, ch_fs);
    }
    return res;
}

// ─── resolve_incoming_queries ─────────────────────────────────────────────────
// Resolver rank: for each query from sender s, look up M' locally; found → return its index, absent →
// INSERT it (new index i') and return i' in the SAME response round (the resolver is the sole inserter
// of cross-rank absent terms). It also records its inbound entry acc[s].in_entries in query order, so
// build_distributed_layer can assemble CrossRankPartnerData without a separate cycle-exchange round.
//
// ORDERING CONTRACT (load-bearing): the B/D exchange is positional — querier A's out_indices[k] must
// pair with resolver B's in_indices[k]. alltoallv preserves per-source order, so responses[s][q]
// answers incoming[s][q]. Every query yields exactly one resolution — DO NOT skip, reorder, or
// partition found vs. absent, or the pairing breaks and multi-rank energy diverges.
//
// PARALLELISM (load-bearing): all queries in one pass are source⊕G for globally-distinct sources (each
// term owned by one rank) and ⊕G is injective ⇒ queries pairwise distinct ⇒ misses distinct and absent.
// So miss j (in fixed (s,q) order) gets index base+j, byte-identical to a serial current_size++ loop.
// Returns per-sender response buffers (one size_t per query); symmetric-pair dedup is structural.
template <size_t NumModes>
auto resolve_incoming_queries(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                              MPOperator<NumModes> &op,
                              size_t rank_count,
                              bool is_leader_pass,
                              std::vector<uint8_t> &matched_bytes,
                              std::vector<PartnerAcc> &acc) -> std::vector<VecZ> {
    constexpr size_t W = kQueryWords<NumModes>;
    std::vector<VecZ> responses(rank_count);

    // Per-sender query counts and flat (sender-major, query-minor) offsets: g = goff[s] + q.
    std::vector<size_t> goff(rank_count + 1, 0);
    for (size_t s = 0; s < rank_count; ++s) {
        const size_t nq = incoming[s].empty() ? 0 : incoming[s].size() / W;
        responses[s].resize(nq, kMissingIndex);
        goff[s + 1] = goff[s] + nq;
    }
    const size_t nq_total = goff[rank_count];
    if (nq_total == 0) {
        return responses;
    }

    // ── Deterministic PARALLEL resolve + bulk insert (see PARALLELISM above) ──
    // Probes run lock-free; the bulk insert is atomics-free (disjoint op slots / map shards / sidecar
    // words, as insert_deferred_self_misses). Only the miss-rank prefix (Phase 2) is serial.
    MPOperator<NumModes> &ins = op;
    const size_t resp_bias = 0;
    DefaultInitVector<uint32_t> sender_of(nq_total);
    for (size_t s = 0; s < rank_count; ++s) {
        std::fill(sender_of.begin() + static_cast<std::ptrdiff_t>(goff[s]),
                  sender_of.begin() + static_cast<std::ptrdiff_t>(goff[s + 1]),
                  static_cast<uint32_t>(s));
    }

    // Phase 1 (parallel, read-only): deserialize + probe. idx_of[g] = found index or kMissingIndex.
    DefaultInitVector<MajoranaSet<NumModes>> maj(nq_total);
    DefaultInitVector<int> phase_of(nq_total);
    DefaultInitVector<size_t> idx_of(nq_total);
    parallel_for_indices(nq_total, [&](size_t g) {
        const size_t s = sender_of[g];
        const size_t q = g - goff[s];
        MajoranaSet<NumModes> m;
        int ph = 0;
        query_read<NumModes>(incoming[s], q, m, ph);
        maj[g] = m;
        phase_of[g] = ph;
        const auto i = op.op->find(m);
        idx_of[g] = (i && *i < op.op->size()) ? *i : kMissingIndex;
    });

    // Phase 2 (serial prefix, (sender,query) order): each miss takes the next index base+j. miss_g[j]
    // records which query g became miss j, so Phase 4 reads the deserialized maj[miss_g[j]] directly.
    const size_t base = ins.op->size(); // LOCAL insert base into the op being mutated
    std::vector<TermIndex> miss_g;
    for (size_t g = 0; g < nq_total; ++g) {
        if (idx_of[g] == kMissingIndex) {
            idx_of[g] = base + miss_g.size() + resp_bias; // COMBINED index downstream expects
            miss_g.push_back(static_cast<TermIndex>(g));
        }
    }
    const size_t n_miss = miss_g.size();

    // Phase 3 (parallel scatter): responses, resolver IN entries (q order), and matched-follower marks.
    // Found indices are distinct so matched_bytes has ≤1 writer per byte; freshly inserted partners
    // (ip ≥ base ≥ matched_bytes.size()) are skipped by the bound check.
    std::vector<size_t> in_base(rank_count);
    for (size_t s = 0; s < rank_count; ++s) {
        in_base[s] = acc[s].in_entries.size();
        acc[s].in_entries.resize(in_base[s] + responses[s].size());
    }
    parallel_for_indices(nq_total, [&](size_t g) {
        const size_t s = sender_of[g];
        const size_t q = g - goff[s];
        const size_t ip = idx_of[g];
        responses[s][q] = ip;
        acc[s].in_entries[in_base[s] + q] = {ip, phase_of[g]};
        if (is_leader_pass && ip < matched_bytes.size()) {
            matched_bytes[ip] = 1;
        }
    });

    // Phase 4 (parallel bulk insert of the n_miss distinct absent terms): scatter majs into the
    // disjoint op slots [base, base+n_miss), insert keys into disjoint map shards, resync the sidecar.
    if (n_miss > 0) {
        // Grow op GEOMETRICALLY (amortized O(1) per layer) — never an exact-fit reserve (would
        // realloc the whole operator every layer). Mirrors insert_deferred_self_misses.
        if (ins.op->capacity() < base + n_miss) {
            ins.op->reserve_rows(std::max(base + n_miss, ins.op->capacity() * 2 + 1));
        }
        ins.op->resize(base + n_miss);
        // One writer per miss slot base+j; the staged dense majorana is read straight out of the
        // deserialization buffer via miss_g — the packed row is written once, never re-read here.
        parallel_for_indices(n_miss, [&](size_t j) { assign_row<NumModes>(*ins.op, base + j, maj[miss_g[j]]); });
        ins.op->bulk_insert(n_miss, base, [&](size_t j) -> const MajoranaSet<NumModes> & {
            return maj[miss_g[j]];
        });
        ins.resync_sidecar_after_bulk_growth(base, n_miss);
    }

    return responses;
}

// ─── process_query_responses ──────────────────────────────────────────────────
// Querier rank: translate responses (found_idx or kMissingIndex) into cycles / half-terms.
template <size_t NumModes>
auto process_query_responses(const std::vector<VecZ> &responses,
                             const std::vector<std::vector<size_t>> &src_idx,
                             const std::vector<VecZ> &queries, // serialized query buffers (for phase recovery)
                             size_t rank_count,
                             size_t my_rank,
                             std::vector<PartnerAcc> &acc) -> void {
    for (size_t r = 0; r < rank_count; ++r) {
        if (r == my_rank) {
            continue;
        } // local already handled inline
        const auto &resp = responses[r];
        const auto &srcs = src_idx[r];
        const auto &qbuf = queries[r];
        for (size_t q = 0; q < resp.size(); ++q) {
            const size_t found_idx = resp[q];
            const size_t source_idx = srcs[q];
            int phase = 0;
            {
                MajoranaSet<NumModes> new_maj;
                query_read<NumModes>(qbuf, q, new_maj, phase);
                (void)new_maj;
            }
            // The resolver inserts on miss, so a remote response is always a real index. The remote
            // found_idx is not needed downstream (only source_idx + phase feed the OUT block).
            assert(found_idx != kMissingIndex && "resolver must insert absent cross-rank terms");
            (void)found_idx;
            // OUT block (querier side), in response (== q) order across leader-then-follower passes.
            acc[r].out_entries.push_back({source_idx, phase});
        }
    }
}

// ─── LayerBuildEngine ─────────────────────────────────────────────────────────
// Owns the machinery for build_distributed_layer: the per-rank accumulator, the matched-follower byte
// set, the per-rank query streams, the deferred self-misses, and the resolve/exchange/finalize
// operations. combined_size sets the matched_bytes length.
template <size_t NumModes>
struct LayerBuildEngine {
    struct DeferredSelfMiss {
        MajoranaSet<NumModes> maj;
        size_t src;
        int phase;
    };
    using PhasedEntry = std::pair<size_t, int>;

    // ── config (set at construction) ──
    MPOperator<NumModes> &local_op;  // scanned + looked up (primary operator)
    MPOperator<NumModes> &insert_op; // where absent terms are inserted (== local_op)
    const MajoranaSet<NumModes> &gen;
    const CutoffEvaluator<NumModes> &cut_eval;
    const MajoranaEvolutionCutoffState &cut_st;
    const VecD &coeffs;
    int only_rotate_len_k;
    MPI_Comm comm;
    size_t R;
    size_t my_rank;

    // ── state (grows during the build) ──
    std::vector<PartnerAcc> acc;
    // Follower-matched set over the combined index space. atomics-free byte array: ≤1 writer per byte
    // (distinct leaders → distinct found via injective ⊕G), leader-pass writes / follower-pass reads.
    std::vector<uint8_t> matched_bytes;
    std::vector<VecZ> queries_r;
    std::vector<std::vector<size_t>> src_idx_r;
    std::vector<DeferredSelfMiss> deferred_self_misses;

    LayerBuildEngine(MPOperator<NumModes> &local_op_,
                     MPOperator<NumModes> &insert_op_,
                     const MajoranaSet<NumModes> &gen_,
                     const CutoffEvaluator<NumModes> &cut_eval_,
                     const MajoranaEvolutionCutoffState &cut_st_,
                     const VecD &coeffs_,
                     int only_rotate_len_k_,
                     MPI_Comm comm_,
                     size_t R_,
                     size_t my_rank_,
                     size_t combined_size)
        : local_op(local_op_),
          insert_op(insert_op_),
          gen(gen_),
          cut_eval(cut_eval_),
          cut_st(cut_st_),
          coeffs(coeffs_),
          only_rotate_len_k(only_rotate_len_k_),
          comm(comm_),
          R(R_),
          my_rank(my_rank_),
          acc(R_),
          matched_bytes(combined_size, 0),
          queries_r(R_),
          src_idx_r(R_) {}

    auto lookup(const MajoranaSet<NumModes> &nm) const -> size_t {
        const auto it = local_op.op->find(nm);
        return (it && *it < local_op.op->size()) ? *it : kMissingIndex;
    }

    // Resolves THIS rank's own query stream (queries_r[my_rank]/src_idx_r[my_rank], populated by the
    // current pass) inline; clears it so the subsequent alltoallv never sends to self.
    auto resolve_self_queries(bool is_leader_pass) -> void {
        VecZ &lq = queries_r[my_rank];
        std::vector<size_t> &ls = src_idx_r[my_rank];
        const size_t nq = lq.empty() ? 0 : lq.size() / kQueryWords<NumModes>;
        const size_t chunks = partition_chunk_count(nq);
        if (chunks <= 1) {
            // Serial: append straight into the accumulator (zero staging copy).
            resolve_range_(lq,
                           ls,
                           0,
                           nq,
                           is_leader_pass,
                           acc[my_rank].in_entries,
                           acc[my_rank].out_entries,
                           deferred_self_misses);
            lq.clear();
            ls.clear();
            return;
        }
        // Probes run chunked in parallel (lock-free lookup, ≤1 writer per matched_bytes byte). The
        // order-sensitive outputs are collected per-chunk and concatenated in chunk order, so the
        // result (including deferred-miss insertion order) matches the serial scan.
        std::vector<DefaultInitVector<PhasedEntry>> in_parts(chunks);
        std::vector<DefaultInitVector<PhasedEntry>> out_parts(chunks);
        std::vector<std::vector<DeferredSelfMiss>> miss_parts(chunks);
        for_each_chunk(nq, chunks, [&](size_t c, size_t lo, size_t hi) {
            resolve_range_(lq, ls, lo, hi, is_leader_pass, in_parts[c], out_parts[c], miss_parts[c]);
        });
        append_gathered_chunks(acc[my_rank].in_entries, in_parts);
        append_gathered_chunks(acc[my_rank].out_entries, out_parts);
        append_gathered_chunks(deferred_self_misses, miss_parts);
        lq.clear();
        ls.clear();
    }

    auto run_exchange(bool is_leader_pass) -> void {
        resolve_self_queries(is_leader_pass);
        if (R > 1) {
            auto inc_q = mpi::alltoallv(queries_r, comm);
            auto resps =
                resolve_incoming_queries(inc_q, local_op, R, is_leader_pass, matched_bytes, acc);
            auto inc_r = mpi::alltoallv(resps, comm);
            process_query_responses<NumModes>(inc_r, src_idx_r, queries_r, R, my_rank, acc);
        }
    }

    // In-place compact the per-rank cross-rank follower query streams, dropping followers a leader
    // already matched in the leader pass (matched_bytes[src]) so they are not re-resolved over the wire.
    auto drop_matched_cross_rank_followers() -> void {
        constexpr size_t W = kQueryWords<NumModes>;
        for (size_t r = 0; r < R; ++r) {
            if (r == my_rank) {
                continue;
            }
            VecZ &q = queries_r[r];
            std::vector<size_t> &s = src_idx_r[r];
            const size_t nq = s.size();
            size_t kept = 0;
            for (size_t k = 0; k < nq; ++k) {
                if (matched_bytes[s[k]]) {
                    continue;
                }
                if (kept != k) { // slide the surviving query's W words down into the next kept slot
                    std::copy(q.begin() + static_cast<std::ptrdiff_t>(k * W),
                              q.begin() + static_cast<std::ptrdiff_t>((k + 1) * W),
                              q.begin() + static_cast<std::ptrdiff_t>(kept * W));
                }
                s[kept] = s[k];
                ++kept;
            }
            q.resize(kept * W);
            s.resize(kept);
        }
    }

    // Sub-step of finish() — do not call directly. Precondition (LOAD-BEARING): call only AFTER both
    // resolve passes complete. Inserting earlier would corrupt the base+k ↔ acc-slot index assignment
    // established below (and the per-miss distinctness argument relies on all passes having run).
    auto insert_deferred_self_misses() -> void {
        const size_t n_miss = deferred_self_misses.size();
        if (n_miss > 0) {
            // ── Parallel deterministic insert (any rank count) ──
            // The deferred SELF misses are pairwise-distinct (each maj is source⊕G over distinct op
            // terms, ⊕G injective) and still absent (a cross-rank term inserted mid-pass is some other
            // rank's source'⊕G, source'≠source). So miss k, in deterministic leader-then-follower
            // order, is assigned base+k — byte-identical to the serial loop — with no dedup and NO
            // ATOMICS: op slots, map shards, sidecar words and acc slots are written by disjoint tasks.
            const size_t base = insert_op.op->size();
            // Grow op GEOMETRICALLY (≥2× before resize) for amortized O(1)/layer; an exact-fit
            // reserve(base+n_miss) would realloc the whole persistent operator every layer.
            if (insert_op.op->capacity() < base + n_miss) {
                insert_op.op->reserve_rows(std::max(base + n_miss, insert_op.op->capacity() * 2 + 1));
            }
            insert_op.op->resize(base + n_miss);

            const size_t in_base = acc[my_rank].in_entries.size();
            const size_t out_base = acc[my_rank].out_entries.size();
            acc[my_rank].in_entries.resize(in_base + n_miss);
            acc[my_rank].out_entries.resize(out_base + n_miss);

            // Scatter majs into disjoint op slots, fill disjoint acc slots.
            parallel_for_indices(n_miss, [&](size_t k) {
                const auto &m = deferred_self_misses[k];
                assign_row<NumModes>(*insert_op.op, base + k, m.maj);
                acc[my_rank].in_entries[in_base + k] = {base + k, m.phase};
                acc[my_rank].out_entries[out_base + k] = {m.src, m.phase};
            });

            // Index map: shard-partitioned insert (key k → base+k), disjoint shards, no atomics.
            // key_at reads the staged dense MajoranaSet directly (no packed-row re-materialization).
            insert_op.op->bulk_insert(n_miss, base, [&](size_t k) -> const MajoranaSet<NumModes> & {
                return deferred_self_misses[k].maj;
            });

            // Sidecar: atomics-free word-partitioned append (only if present; has_value ⟹ rows==base).
            insert_op.resync_sidecar_after_bulk_growth(base, n_miss);
        }
    }

    // Sub-step of finish() — do not call directly (consumes acc after both passes + the deferred inserts).
    auto assemble_partners() -> std::vector<CrossRankPartnerData> {
        // Layout: b = [in.idx]++[out.idx]; d = [{out.idx,−φ}]++[{in.idx,+φ}]. cos covers ALL
        // anticommuting indices (endpoints included) since the D-apply only ADDS the sine term.
        std::vector<CrossRankPartnerData> partners(R);
        for (size_t r = 0; r < R; ++r) {
            const auto &a = acc[r];
            auto &p = partners[r];
            const size_t P = a.in_entries.size();
            const size_t Q = a.out_entries.size();
            if (P + Q == 0) {
                continue;
            }
            p.in_count = P; // boundary for deriving the D index list from B (D indices are not stored)
            p.b_indices.resize(P + Q);
            p.d.resize(P + Q);
            // One parallel region over P+Q: k<P fills in-entry slots, k>=P fills out-entry slots.
            parallel_for_indices(P + Q, [&](size_t k) {
                if (k < P) {
                    const auto &e = a.in_entries[k];
                    p.b_indices[k] = e.first;
                    p.d[Q + k] = {e.first, e.second};
                }
                else {
                    const size_t j = k - P;
                    const auto &e = a.out_entries[j];
                    p.b_indices[k] = e.first;
                    p.d[j] = {e.first, -e.second};
                }
            });
        }
        return partners;
    }

    // cos is not stored on the layer. When `out_cos` is non-null the in-build contraction needs the
    // full anticommuting cos for the immediate evolve_step, so we hand it over; otherwise cos is
    // discarded and recomputed from the sidecar fold at replay (generator_words + cos_count).
    auto finish(CosineWordList &&cos_all, CosineWordList *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        insert_deferred_self_misses();
        std::vector<CrossRankPartnerData> partners = assemble_partners();
        // Every rotation TARGET must be in cos so the gradient reverse-sweep can recover its pre-layer
        // coefficient by un-doing this layer's cosine scaling. Cycle targets are already in cos from
        // the fused scan; only freshly INSERTED half-terms can be absent. Forward energy is unaffected
        // (an inserted target's coefficient is 0 when the cos pass runs). Without this the reverse sweep
        // over-scales those endpoints — see test_infinite_cutoff.
        //
        // Inserts are APPENDED, occupying [combined_size, insert_op.size()), so we append just that
        // range (O(inserted)) instead of scanning every rotation target (a serial Amdahl anchor).
        if (out_cos != nullptr) {
            const size_t cos_lo = matched_bytes.size();
            const size_t cos_hi = insert_op.op->size();
            CosineWordBuilder end_b;
            for (size_t idx = cos_lo; idx < cos_hi; ++idx) {
                end_b.push_index(idx);
            }
            CosineWordList end_words = end_b.finish();
            // Endpoint bits (indices >= cos_lo) and scan bits (indices < cos_lo) are disjoint, so OR
            // is exact when they share the same boundary word (cos_lo not 64-aligned).
            cos_all.total_count += end_words.total_count;
            if (!cos_all.blocks.empty() && !end_words.blocks.empty()
                && end_words.blocks.front().first == cos_all.blocks.back().first) {
                cos_all.blocks.back().second |= end_words.blocks.front().second;
                cos_all.blocks.insert(cos_all.blocks.end(), end_words.blocks.begin() + 1, end_words.blocks.end());
            }
            else {
                cos_all.blocks.insert(cos_all.blocks.end(), end_words.blocks.begin(), end_words.blocks.end());
            }
            *out_cos = std::move(cos_all);
        }
        return build_layer_storage_unified(std::move(partners), my_rank);
    }

private:
    auto resolve_range_(VecZ &lq,
                        std::vector<size_t> &ls,
                        size_t lo,
                        size_t hi,
                        bool is_leader_pass,
                        DefaultInitVector<PhasedEntry> &in_sink,
                        DefaultInitVector<PhasedEntry> &out_sink,
                        std::vector<DeferredSelfMiss> &miss_sink) -> void {
        for (size_t q = lo; q < hi; ++q) {
            const size_t src = ls[q];
            if (!is_leader_pass && matched_bytes[src]) {
                continue; // follower already matched by a leader → not an independent rotation
            }
            MajoranaSet<NumModes> new_maj;
            int phase = 0;
            query_read<NumModes>(lq, q, new_maj, phase);
            const size_t found = lookup(new_maj);
            if (found != kMissingIndex) {
                if (is_leader_pass) {
                    matched_bytes[found] = 1; // distinct leaders → distinct found → no atomics
                }
                in_sink.push_back({found, phase});
                out_sink.push_back({src, phase});
            }
            else {
                miss_sink.push_back({new_maj, src, phase});
            }
        }
    }
};

// ─── build_distributed_layer ─────────────────────────────────────────────────
// Primary-path layer builder. Implements paper Algorithm 2 and emits a graph layer directly.
// Runs the fused scan (FindAnticommuting + apply_cutoffs in one walk) to produce the compressed
// cosine blocks and cutoff-applied per-rank leader/follower query streams. During the two exchange
// passes, rotation participants accumulate into a uniform per-rank PartnerAcc (self slot = partner
// with in:=tgt, out:=src). After both passes: self-rank absent partners are inserted (load-bearing:
// AFTER both resolves), the per-rank CrossRankPartnerData is assembled, and a LayerCore is built.
template <size_t NumModes>
auto build_distributed_layer(MPOperator<NumModes> &local_op,
                             const MajoranaSet<NumModes> &gen,
                             const CutoffFn<NumModes> &cutoff_fn,
                             const std::optional<double> &atol,
                             std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                             const std::optional<double> &upper_atol,
                             const std::optional<double> &param,
                             int only_rotate_len_k,
                             MPI_Comm comm,
                             CosineWordList *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = static_cast<size_t>(mpi::size(comm));
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs ? local_coeffs->get() : empty_coeffs();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // The cosine set covers operator indices < the operator size measured BEFORE this layer's
    // partner inserts; capture that bound now (the scan/inserts below grow local_op).
    const uint64_t cos_count_pre_insert = static_cast<uint64_t>(local_op.op->size());

    FusedScanResult fused =
        fused_find_and_collect<NumModes>(local_op, gen, cut_eval, cut_st, coeffs, only_rotate_len_k, R, my_rank);
    CosineWordList cos_all;
    {
        std::vector<CosineWordList *> blocks;
        blocks.reserve(fused.cos_blocks.size());
        for (auto &block : fused.cos_blocks) {
            blocks.push_back(&block);
        }
        cos_all = concat_cosine_word_blocks(blocks);
    }
    fused.cos_blocks = std::vector<CosineWordList>{};

    LayerBuildEngine<NumModes> eng(local_op,
                                   local_op,
                                   gen,
                                   cut_eval,
                                   cut_st,
                                   coeffs,
                                   only_rotate_len_k,
                                   comm,
                                   R,
                                   my_rank,
                                   /*combined_size=*/local_op.op->size());

    eng.queries_r = std::move(fused.leader_queries);
    eng.src_idx_r = std::move(fused.leader_src);
    eng.run_exchange(/*is_leader_pass=*/true);

    eng.queries_r = std::move(fused.follower_queries);
    eng.src_idx_r = std::move(fused.follower_src);
    if (R > 1) {
        eng.drop_matched_cross_rank_followers();
    }
    eng.run_exchange(/*is_leader_pass=*/false);

    auto storage = eng.finish(std::move(cos_all), out_cos);

    // Recompute metadata rides WITH the layer (in its LayerCore), so it survives every graph transform
    // (slice/union/consume/Schrödinger-prepend). cos_count is the POST-insert operator size (after
    // finish() ran this layer's partner inserts): the stored cos is "all anticommuting", so folding the
    // sidecar truncated to cos_count reproduces it bit-for-bit in both pictures with no stored bitmap.
    storage->generator_words.assign(gen.data(), gen.data() + mpi_detail::kWords<NumModes>);
    storage->cos_count = static_cast<uint64_t>(local_op.op->size());

    return storage;
}

} // namespace monoprop::detail
