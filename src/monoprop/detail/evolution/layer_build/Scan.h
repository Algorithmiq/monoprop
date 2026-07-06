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

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CoeffFrame.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Parallel.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/InvertedIndex.h"

namespace monoprop::detail {

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

// Collect G's set columns (the modes it touches) in ASCENDING bit order. Because they are ascending,
// indices[0] is the LOWEST set column = THE pivot — the column that splits each anticommuting pair
// into leader (pivot clear) and follower (pivot set). even_parity_scan_pass1 and the emit path both
// rely on gen_cols[0] being the pivot.
template <size_t NumModes>
auto build_even_parity_generator_columns(const MajoranaSet<NumModes> &gen_maj) -> EvenParityGeneratorColumns<NumModes> {
    EvenParityGeneratorColumns<NumModes> columns;
    for (size_t bit_idx = gen_maj.find_first(); bit_idx < gen_maj.size(); bit_idx = gen_maj.find_next(bit_idx)) {
        columns.indices[columns.count++] = bit_idx;
    }
    return columns;
}

// One nonzero-overlap word carried from the memory-bound scan (pass 1) to the emit (pass 2) in the
// even-parity inverted index scan. `overlap` bit t set ⟺ term (base+t) anticommutes with G. `foll` is the
// follower sub-mask (overlap & the pivot column): a term and its partner M⊕G split on the pivot bit,
// so masking overlap by the pivot column selects exactly the followers; leaders are the complement
// `overlap ^ foll` (disjoint). Used by fused_find_and_collect.
struct EvenParityNzWord {
    size_t base;
    uint64_t overlap;
    uint64_t foll;
};

// Even-parity scan pass 1: over words [wlo,whi), fold the generator's inverted index columns block-by-block
// (L1-resident sub-blocks; see combine_columns_block) into a per-word overlap mask, keep nonzero
// words in `nz`, and tally popcounts (n_anti, n_foll) so pass 2 reserves its
// output runs once. Pass a thread_local `nz` to reuse its capacity across chunks. `gen_cols` is
// the generator's full column list, pivot (leader/follower split column) FIRST — the fold includes
// it, and the pivot's own bits come from its dense words directly or a per-block scatter if sparse.
// `g_odd` carries the odd-|G| correction: the anticommutation bit is (|M∩G| mod 2) XOR (|M| mod 2),
// so the per-row parity(|M|) bit (row_parity_ptr) is XORed in before foll/nonzero/pivot are derived.
// Even |G| (g_odd==false) ignores row_parity_ptr and is byte-identical.
template <size_t NumModes>
inline auto even_parity_scan_pass1(const InvertedIndex<NumModes> &sc,
                                   std::span<const size_t> gen_cols,
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
    const size_t pivot_col = gen_cols[0];
    const bool pivot_dense = sc.column_is_dense(pivot_col);
    const uint64_t *const pivot_dense_ptr = pivot_dense ? sc.dense_column_data(pivot_col) : nullptr;
    std::vector<uint64_t> &blk = column_block_scratch();
    // Fold one word range [bb,be): combine G's columns, split leader/follower by the pivot bit, and
    // record every nonzero-overlap word. Shared by the plain (1024-word) and the block-skip (64-word)
    // block loops below.
    auto fold_range = [&](size_t bb, size_t be) {
        combine_columns_block<NumModes>(sc, gen_cols, blk.data(), bb, be);
        const uint64_t *pw; // pivot words for [bb,be), indexed [0, be-bb)
        if (pivot_dense) {
            pw = pivot_dense_ptr + bb;
        }
        else {
            std::vector<uint64_t> &pblk = pivot_column_block_scratch();
            combine_columns_block<NumModes>(sc, std::span<const size_t>(&pivot_col, 1), pblk.data(), bb, be);
            pw = pblk.data();
        }
        for (size_t wi = bb; wi < be; ++wi) {
            uint64_t overlap = blk[wi - bb];
            if (g_odd) {
                overlap ^= row_parity_ptr[wi];
            }
            if (wi == last_word) {
                overlap &= last_word_mask;
            }
            if (!overlap) {
                continue;
            }
            const uint64_t foll = overlap & pw[wi - bb];
            n_anti += static_cast<size_t>(std::popcount(overlap));
            n_foll += static_cast<size_t>(std::popcount(foll));
            nz.push_back(EvenParityNzWord{wi * 64, overlap, foll});
        }
    };
    for (size_t bb = wlo; bb < whi; bb += kColumnBlockWords) {
        fold_range(bb, std::min(bb + kColumnBlockWords, whi));
    }
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
    // Fixed-per-layer interleave mask W: interleave_phase(M,G) == (M.parity_and(W) ? -1 : 1).
    // Replaces the per-term prefix-XOR scan with one masked parity (see interleave_phase_mask).
    MajoranaSet<NumModes> interleave_mask;
};

template <size_t NumModes>
inline auto make_gen_emit_context(const MajoranaSet<NumModes> &gen, size_t gen_pop) -> GenEmitContext<NumModes> {
    return GenEmitContext<NumModes>{gen, gen_pop, interleave_phase_mask<NumModes>(gen)};
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
    // interleave_phase(maj, gen) via the fixed-per-layer mask (provably identical, branch/scan-free).
    interleave = maj.parity_and(ctx.interleave_mask) ? -1 : 1;
}

// ─── fused_find_and_collect (any rank count) ──────────────────────────────────
// One pass over the operator fusing FindAnticommuting + apply_cutoffs: classify each anticommuting
// term leader/follower (inverted index XOR-column fold + pivot bit), compress it into the cosine block, and
// in the SAME walk apply the cutoffs and emit the surviving rotation query into its per-rank stream.
struct FusedScanResult {
    std::vector<CosMask> cos_blocks;        // ascending, disjoint, chunk order
    std::vector<VecZ> leader_queries;              // size R: serialized leader queries per owner rank
    std::vector<std::vector<size_t>> leader_src;   // size R: parallel to leader_queries (source op idx)
    std::vector<VecZ> follower_queries;            // size R: serialized follower queries per owner rank
    std::vector<std::vector<size_t>> follower_src; // size R: parallel to follower_queries
    // Fused-contraction only (capture_values): signed pre-cos source coeff (v_src) parallel to
    // leader_src / follower_src. Empty (never populated) when capture_values is false.
    std::vector<std::vector<double>> leader_val;
    std::vector<std::vector<double>> follower_val;
};

// Streams are routed to the owner of each partner M' = M⊕G (hash%R; self for single-rank, skipping
// the O(W) hash) and emitted in ascending source-index (chunk) order, so the downstream resolve and
// cross-rank index assignment are deterministic.
// `capture_values` (fused contraction, R==1): also collect the signed pre-cos source coefficient
// (v_src) into leader_val/follower_val, parallel to leader_src/follower_src. Off by default so every
// other path is byte-for-byte unchanged.
template <size_t NumModes>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const MajoranaSet<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const MajoranaEvolutionCutoffState &cut_st,
                            const VecD &coeffs,
                            int only_rotate_len_k,
                            size_t rank_count,
                            size_t my_rank,
                            bool capture_values = false,
                            const CoeffFrame<NumModes> *frame = nullptr) -> FusedScanResult {
    const size_t gen_pop = gen.count();
    const auto ectx = make_gen_emit_context<NumModes>(gen, gen_pop);
    // Lazy-cosine frame: when `frame` is active the stored coeffs are frozen, so the gate reconstructs
    // the true value on demand; no eager cosine set is built (cos stays empty). The frame also carries
    // the magnitude byte prefilter: reject an anti term without reading its 8-byte coefficient when its
    // 1-byte upper bound proves it fails the sin gate (survivors exact-confirm). See CoeffFrame.h.
    const bool implicit = (frame != nullptr);
    const uint32_t *const stamp = implicit ? frame->stamp.data() : nullptr;
    const uint8_t *const mag_ptr =
        (implicit && frame->mag.size() == coeffs.size()) ? frame->mag.data() : nullptr;
    const double mag_rt =
        mag_reject_threshold(cut_st.cutoff_ctx.check_atol, cut_st.cutoff_ctx.abs_sin_val, cut_st.cutoff_ctx.atol_value);

    // Cutoff + emit for one anticommuting term. Writes only the per-chunk per-rank sinks passed in
    // (safe under for_each_chunk). The dynamic gate (depends only on |M|) runs BEFORE
    // emit_term_products, so a gate-rejected term computes no products.
    // abs_c = |coeff[i]| is passed in (the caller already loaded it for the pre-popcount atol gate on
    // the only_rotate_len_k==0 fast path), so emit does not re-read the coefficient. `v_src` is the
    // SIGNED coeff (derived from the same read); it is pushed into lv/fv only when capture_values.
    auto emit = [&](size_t maj_pop,
                    size_t i,
                    double abs_c,
                    double v_src,
                    bool is_follower,
                    std::vector<VecZ> &lq,
                    std::vector<std::vector<size_t>> &ls,
                    std::vector<std::vector<double>> &lv,
                    std::vector<VecZ> &fq,
                    std::vector<std::vector<size_t>> &fs,
                    std::vector<std::vector<double>> &fv) {
        if (!rotation_dynamic_gate(only_rotate_len_k, maj_pop, cut_st.cutoff_ctx, abs_c)) {
            return;
        }
        MajoranaSet<NumModes> new_maj;
        size_t overlap = 0;
        int interleave = 0;
        emit_term_products<NumModes>(*op.store, i, ectx, new_maj, overlap, interleave);
        const size_t new_pop = maj_pop + gen_pop - 2 * overlap;
        // Structural cutoff on the partner M⊕G — UNLESS upper_atol rescues it (its sine coefficient is
        // large enough to keep alive despite exceeding the cutoff). See CutoffContext::is_above_upper.
        const bool struct_pass = cutoff_eval.passes_with_popcount(new_maj, new_pop);
        if (!struct_pass && !cut_st.cutoff_ctx.is_above_upper(abs_c)) {
            return;
        }
        const int phase = interleave * hermitian_phase(maj_pop, gen_pop, overlap);
        // Single rank: every partner is self-owned, skip the O(W) hash; multi-rank routes by owner.
        const size_t r_prime = (rank_count == 1) ? my_rank : (majorana_hash<NumModes>(new_maj) % rank_count);
        const size_t source = i;
        if (is_follower) {
            query_push<NumModes>(fq[r_prime], new_maj, phase);
            fs[r_prime].push_back(source);
            if (capture_values) {
                fv[r_prime].push_back(v_src);
            }
        }
        else {
            query_push<NumModes>(lq[r_prime], new_maj, phase);
            ls[r_prime].push_back(source);
            if (capture_values) {
                lv[r_prime].push_back(v_src);
            }
        }
    };

    FusedScanResult res;
    res.leader_queries.assign(rank_count, VecZ{});
    res.leader_src.assign(rank_count, std::vector<size_t>{});
    res.follower_queries.assign(rank_count, VecZ{});
    res.follower_src.assign(rank_count, std::vector<size_t>{});
    // Sized to R even on the early-return paths below so the fused engine's per-rank src_val_r access
    // is always in bounds (parallel to leader_src / follower_src).
    if (capture_values) {
        res.leader_val.assign(rank_count, std::vector<double>{});
        res.follower_val.assign(rank_count, std::vector<double>{});
    }

    {
        // Odd |G| needs the per-row parity(|M|) correction (see even_parity_scan_pass1); even |G| is
        // byte-identical with no parity bitmap.
        const bool g_odd = (gen.count() % 2 != 0);
        const auto gen_columns = build_even_parity_generator_columns<NumModes>(gen);
        if (gen_columns.count == 0) {
            return res;
        }
        const auto &inverted_index = op.inverted_index();
        const size_t word_count = inverted_index.words();
        if (word_count == 0) {
            return res;
        }
        // Build the row_parity bitmap once, only for odd generators (even workloads never allocate it).
        if (g_odd) {
            inverted_index.ensure_row_parity();
        }
        const uint64_t *const row_parity_ptr = g_odd ? inverted_index.row_parity_word_ptr() : nullptr;
        const size_t n = op.store->size();

        const size_t last_word = word_count - 1;
        const uint64_t last_word_mask = (n % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (n % 64)) - 1);
        // Generator column list, pivot first. Each chunk folds its own L1-resident blocks inside
        // pass 1 (combine_columns_block) — no serial full-width sparse-scatter prologue.
        const std::span<const size_t> gen_cols(gen_columns.indices.data(), gen_columns.count);

        const size_t chunks = partition_chunk_count_words(word_count);
        std::vector<CosMask> ch_cos(chunks);
        std::vector<std::vector<VecZ>> ch_lq(chunks, std::vector<VecZ>(rank_count));
        std::vector<std::vector<std::vector<size_t>>> ch_ls(chunks, std::vector<std::vector<size_t>>(rank_count));
        std::vector<std::vector<VecZ>> ch_fq(chunks, std::vector<VecZ>(rank_count));
        std::vector<std::vector<std::vector<size_t>>> ch_fs(chunks, std::vector<std::vector<size_t>>(rank_count));
        // Fused v_src sinks: allocated (chunks × rank_count) only when capture_values; otherwise the
        // outer vectors hold `chunks` empty entries and are never indexed (emit guards on capture_values).
        std::vector<std::vector<std::vector<double>>> ch_lv(chunks);
        std::vector<std::vector<std::vector<double>>> ch_fv(chunks);
        if (capture_values) {
            for (size_t c = 0; c < chunks; ++c) {
                ch_lv[c].assign(rank_count, std::vector<double>{});
                ch_fv[c].assign(rank_count, std::vector<double>{});
            }
        }
        for_each_chunk(word_count, chunks, [&](size_t c, size_t wlo, size_t whi) {
            auto &cos = ch_cos[c];
            auto &lq = ch_lq[c];
            auto &ls = ch_ls[c];
            auto &lv = ch_lv[c];
            auto &fq = ch_fq[c];
            auto &fs = ch_fs[c];
            auto &fv = ch_fv[c];

            // Pass 1: fold the inverted index to find anticommuting terms (see even_parity_scan_pass1).
            // `nz` is thread_local to reuse capacity across chunks.
            thread_local std::vector<EvenParityNzWord> nz;
            size_t n_anti = 0;
            size_t n_foll = 0;
            even_parity_scan_pass1<NumModes>(inverted_index,
                                             gen_cols,
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
            // Derive (v_src, abs_c) for term i, shared by both pass-2 arms. Fused mode captures the
            // SIGNED coeff v_src and derives abs_c from it; the derived abs_c is bit-identical to
            // abs_coeff_for, so the non-capture (OFF) path is unchanged. Kept out of the arms so the
            // gate-before-popcount ordering in each arm stays explicit at the call site.
            auto derive_coeff = [&](size_t i) -> std::pair<double, double> {
                if (implicit) {
                    // Reconstruct the true (lazily-cos-scaled) coefficient from the frozen store.
                    const double v_src =
                        frame->true_value(*op.store, (i < coeffs.size()) ? coeffs[i] : 0.0, stamp[i], i);
                    return {v_src, std::abs(v_src)};
                }
                if (capture_values) {
                    const double v_src = (i < coeffs.size()) ? coeffs[i] : 0.0;
                    return {v_src, cut_st.cutoff_ctx.use_coeff_checks ? std::abs(v_src) : 0.0};
                }
                return {0.0, cut_st.cutoff_ctx.abs_coeff_for(i, coeffs)};
            };
            const bool word_aligned_cos = only_rotate_len_k == 0;
            CosineWordBuilder cos_b;
            for (const auto &w : nz) {
                if (word_aligned_cos) {
                    // No orbital gate: cosine-scale the whole word (all anticommuting terms), then per
                    // bit apply the ATOL coefficient gate BEFORE the popcount ROW read. ~90–97% of
                    // anticommuting terms fail this gate (their coefficient is below the sine cutoff),
                    // and the gate needs only |coeff[i]| — not the row — so deferring popcount until a
                    // term passes eliminates that many random packed-row cacheline loads (the dominant
                    // pass-2 memory traffic). Bit-identical: same emitted set/order, same cos word.
                    // Implicit frame: no eager cosine set — the firing log carries the cos factor.
                    if (!implicit) {
                        cos_b.push_word(w.base, w.overlap);
                    }
                    for (uint64_t m = w.overlap; m; m &= m - 1) {
                        const size_t tz = static_cast<size_t>(std::countr_zero(m));
                        const size_t i = w.base + tz;
                        // Byte prefilter: skip the coefficient read for terms the 1-byte upper bound
                        // already proves are below the sin cutoff (bit-exact — survivors exact-confirm).
                        if (mag_ptr != nullptr && static_cast<double>(mag_ptr[i]) <= mag_rt) {
                            continue;
                        }
                        const auto [v_src, abs_c] = derive_coeff(i);
                        if (cut_st.cutoff_ctx.is_below_sin(abs_c)) {
                            continue;
                        }
                        const size_t maj_pop = op.store->popcount(i);
                        const bool is_follower = (w.foll >> tz) & 1u;
                        emit(maj_pop, i, abs_c, v_src, is_follower, lq, ls, lv, fq, fs, fv);
                    }
                }
                else {
                    // Orbital gate active: it needs maj_pop, and the per-index cosine push covers only
                    // orbital-passing terms, so the popcount row read must precede both.
                    for (uint64_t m = w.overlap; m; m &= m - 1) {
                        const size_t tz = static_cast<size_t>(std::countr_zero(m));
                        const size_t i = w.base + tz;
                        const size_t maj_pop = op.store->popcount(i);
                        if (maj_pop > static_cast<size_t>(only_rotate_len_k)) {
                            continue;
                        }
                        cos_b.push_index(i);
                        const auto [v_src, abs_c] = derive_coeff(i);
                        const bool is_follower = (w.foll >> tz) & 1u;
                        emit(maj_pop, i, abs_c, v_src, is_follower, lq, ls, lv, fq, fs, fv);
                    }
                }
            }
            cos = cos_b.finish();
        });
        res.cos_blocks = std::move(ch_cos);
        append_chunked_rank_vectors(res.leader_queries, ch_lq);
        append_chunked_rank_vectors(res.leader_src, ch_ls);
        append_chunked_rank_vectors(res.follower_queries, ch_fq);
        append_chunked_rank_vectors(res.follower_src, ch_fs);
        if (capture_values) {
            // res.leader_val / follower_val were pre-sized to R above; append the per-chunk parts.
            append_chunked_rank_vectors(res.leader_val, ch_lv);
            append_chunked_rank_vectors(res.follower_val, ch_fv);
        }
    }
    return res;
}

} // namespace monoprop::detail
