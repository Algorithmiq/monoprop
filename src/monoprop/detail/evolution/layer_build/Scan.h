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
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/PauliAlgebra.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop::detail {

// ─── Even-parity scan + cutoff-state helpers ───────────────────────────────────
// The cutoff state read by the fused scan and the even-parity generator-column scan it uses.
inline auto build_majorana_evolution_cutoff_state(const std::optional<double> &atol,
                                                  std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                                                  const std::optional<double> &upper_atol,
                                                  const std::optional<double> &param) -> CutoffContext {
    const bool check_atol = atol.has_value() && local_coeffs.has_value() && param.has_value();
    const bool check_upper_atol = upper_atol.has_value() && local_coeffs.has_value();
    const double sin_val = param.has_value() ? std::sin(2 * param.value()) : 1.0;
    const double cos_val = param.has_value() ? std::cos(2 * param.value()) : 1.0;

    return CutoffContext{.check_atol = check_atol,
                         .check_upper_atol = check_upper_atol,
                         .atol_value = atol.value_or(0.0),
                         .upper_atol_value = upper_atol.value_or(0.0),
                         .abs_sin_val = std::abs(sin_val),
                         .abs_cos_val = std::abs(cos_val),
                         .use_coeff_checks = check_atol || check_upper_atol};
}

template <size_t NumModes>
struct EvenParityGeneratorColumns {
    std::array<size_t, MajoranaSet<NumModes>::size()> indices{};
    size_t count = 0;
};

// Collect a generator's set columns (the modes it touches) in ASCENDING bit order. indices[0] is the
// LOWEST set column; ordinary (Majorana) callers pass it to even_parity_scan_pass1 as the pivot — the
// column that splits each anticommuting pair into leader (pivot clear) and follower (pivot set).
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
// the column list the fold XORs into the anticommutation parity; `pivot_col` is the leader/follower
// split column and is read SEPARATELY (its bits come from its dense words directly or a per-block
// scatter if sparse). `pivot_col` must be a set bit that flips between every anticommuting partner
// pair; ordinary callers pass gen_cols[0]. Keeping it a distinct argument lets a caller fold one
// column set (e.g. a transformed generator) while splitting on a bit of the untransformed generator.
// `g_odd` carries the odd-|G| correction: the anticommutation bit is (|M∩G| mod 2) XOR (|M| mod 2),
// so the per-row parity(|M|) bit (row_parity_ptr) is XORed in before foll/nonzero/pivot are derived.
// Even |G| (g_odd==false) ignores row_parity_ptr and is byte-identical.
template <size_t NumModes>
inline auto even_parity_scan_pass1(const InvertedIndex<NumModes> &sc,
                                   std::span<const size_t> gen_cols,
                                   size_t pivot_col,
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
    const bool pivot_dense = sc.column_is_dense(pivot_col);
    const uint64_t *const pivot_dense_ptr = pivot_dense ? sc.dense_column_data(pivot_col) : nullptr;
    std::vector<uint64_t> &blk = column_block_scratch();
    // Fold one word range [bb,be): combine G's columns, split leader/follower by the pivot bit, and
    // record every nonzero-overlap word. Driven by the single kColumnBlockWords block loop below.
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
inline auto rotation_dynamic_gate(int only_rotate_len_k,
                                  size_t maj_pop,
                                  const CutoffContext &ctx,
                                  double abs_c) -> bool {
    if (only_rotate_len_k > 0 && maj_pop > static_cast<size_t>(only_rotate_len_k)) {
        return false;
    }
    if (ctx.is_below_sin(abs_c)) {
        return false;
    }
    return true;
}

// ─── Rebuild-then-word-kernels emit (packed survivor products) ────────────────
// Per-generator context, built once per generator. Two flavours selected at compile time by IsPauli:
// the Majorana arm caches the real generator G plus the fixed-per-layer interleave mask; the Pauli arm
// caches the rotation-sign kernel context (PauliGenContext, which itself holds G and |G| — the single
// source of truth). Either way emit_term_products reads the generator (for M⊕G / overlap) from here.
template <size_t NumModes, bool IsPauli>
struct GenEmitContext;

template <size_t NumModes>
struct GenEmitContext<NumModes, false> {
    const MajoranaSet<NumModes> &gen;
    // Fixed-per-layer interleave mask W: interleave_phase(M,G) == (M.parity_and(W) ? -1 : 1).
    // Replaces the per-term prefix-XOR scan with one masked parity (see interleave_phase_mask).
    MajoranaSet<NumModes> interleave_mask;
};

template <size_t NumModes>
struct GenEmitContext<NumModes, true> {
    // Precomputed context for the hot Pauli rotation-sign kernel (pauli_rotation_sign). It already
    // carries the generator G and |G|, so no separate gen/gen_pop members are duplicated here.
    PauliGenContext<NumModes> pauli_ctx;
};

template <size_t NumModes, bool IsPauli>
inline auto make_gen_emit_context(const MajoranaSet<NumModes> &gen) -> GenEmitContext<NumModes, IsPauli> {
    if constexpr (IsPauli) {
        return GenEmitContext<NumModes, true>{make_pauli_gen_context<NumModes>(gen)};
    }
    else {
        return GenEmitContext<NumModes, false>{gen, interleave_phase_mask<NumModes>(gen)};
    }
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
// The per-term phase_factor is the basis-specific multiplicative sign:
//   Majorana: interleave_phase(maj, gen) via the fixed-per-layer mask (branch/scan-free);
//   Pauli:    pauli_rotation_sign(pauli_ctx, maj, new_maj) — the ±1 ROTATION sign of maj·G (already
//             negated relative to the raw product sign, so no extra flip at the emit site).
// Majorana additionally folds in hermitian_phase to obtain the final query phase (see the emit lambda).
template <size_t NumModes, bool IsPauli>
[[gnu::always_inline]] inline void emit_term_products(const OperatorIndex<NumModes> &ham,
                                                      size_t i,
                                                      const GenEmitContext<NumModes, IsPauli> &ctx,
                                                      MajoranaSet<NumModes> &new_maj,
                                                      size_t &overlap,
                                                      int &phase_factor) {
    MajoranaSet<NumModes> maj; // zero-init, W words, lives in registers
    ham.for_each_position(i, [&](size_t pos) { maj.set(pos); });
    if constexpr (IsPauli) {
        const MajoranaSet<NumModes> &gen = ctx.pauli_ctx.gen;
        new_maj = maj ^ gen;
        overlap = maj.count_and(gen);
        phase_factor = pauli_rotation_sign<NumModes>(ctx.pauli_ctx, maj, new_maj);
    }
    else {
        new_maj = maj ^ ctx.gen;
        overlap = maj.count_and(ctx.gen);
        phase_factor = maj.parity_and(ctx.interleave_mask) ? -1 : 1;
    }
}

// ─── fused_find_and_collect (any rank count) ──────────────────────────────────
// One pass over the operator fusing FindAnticommuting + apply_cutoffs: classify each anticommuting
// term leader/follower (inverted index XOR-column fold + pivot bit), compress it into the cosine block, and
// in the SAME walk apply the cutoffs and emit the surviving rotation query into its per-rank stream.
struct FusedScanResult {
    std::vector<CosMask> cos_blocks;               // ascending, disjoint, chunk order
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
// `fused_scale_coeffs` (ContractImmediately, k==0 only; must alias coeffs.data()): fold the gate's
// cosine scale into this same pass — each anticommuting coefficient is loaded once (pre-cos, feeding
// the atol gate and v_src exactly as before) and stored back multiplied by `fused_scale_cos` =
// cos(2·build_angle). One coefficient sweep replaces the eager read-pass + CosMask + RMW-scale-pass,
// so no cosine set is built (cos_blocks stay empty). Chunks own disjoint word ranges ⇒ the in-place
// writes are race-free. Downstream, a hit partner's stored value is then POST-cos — resolve recovers
// the pre-cos v_tgt via 1/cos (see LayerBuildEngine::inv_cos_).
template <size_t NumModes, bool IsPauli = false>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const MajoranaSet<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const CutoffContext &cut_st,
                            const VecD &coeffs,
                            int only_rotate_len_k,
                            size_t rank_count,
                            size_t my_rank,
                            bool capture_values = false,
                            double *fused_scale_coeffs = nullptr,
                            double fused_scale_cos = 1.0) -> FusedScanResult {
    const size_t gen_pop = gen.count();
    const auto ectx = make_gen_emit_context<NumModes, IsPauli>(gen);

    // Cutoff + emit for one anticommuting term. Writes only the per-rank sinks passed in. The dynamic
    // gate (depends only on |M|) runs BEFORE emit_term_products, so a gate-rejected term computes no
    // products.
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
        // Gate emission on the SOURCE here (dynamic sine + orbital cap).
        if (!rotation_dynamic_gate(only_rotate_len_k, maj_pop, cut_st, abs_c)) {
            return;
        }
        MajoranaSet<NumModes> new_maj;
        size_t overlap = 0;
        int phase_factor = 0;
        emit_term_products<NumModes, IsPauli>(*op.store, i, ectx, new_maj, overlap, phase_factor);
        // Structural cutoff on the partner M⊕G — UNLESS upper_atol rescues it (its sine coefficient is
        // large enough to keep alive despite exceeding the cutoff). See CutoffContext::is_above_upper.
        const size_t new_pop = maj_pop + gen_pop - 2 * overlap;
        const bool struct_pass = cutoff_eval.passes_with_popcount(new_maj, new_pop);
        if (!struct_pass && !cut_st.is_above_upper(abs_c)) {
            return;
        }
        // Pauli: pauli_rotation_sign already returns the rotation-ready sign for U=exp(iθG), O'=U†OU
        // (the negated raw product sign of maj·G, pinned by pauli_build_layer_dense_matrix_ground_truth
        // / T7), so it is emitted directly. Majorana folds in hermitian_phase.
        int phase;
        if constexpr (IsPauli) {
            phase = phase_factor;
        }
        else {
            phase = phase_factor * hermitian_phase(maj_pop, gen_pop, overlap);
        }
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
        // The anticommutation fold runs over the generator's inverted-index columns. For Majorana that is
        // G itself; for Pauli it is J(G) = pair_swap(G), since pauli_anticommutes(M,G) = parity(|M ∩ J(G)|).
        // parity(|G ∩ J(G)|) = parity(2·#Z) = 0 ⇒ the self-commutation invariant holds and no odd-|G|
        // row-parity correction is ever needed for Pauli (g_odd forced false). J is a bijection so
        // J(G) ≠ 0 ⟺ G ≠ 0. The pivot that splits each anticommuting pair is a set bit of the REAL G
        // (gen.find_first()), NOT of J(G) — A and A⊕G differ exactly on G's bits (see the pivot arg below).
        const MajoranaSet<NumModes> fold_gen = IsPauli ? pair_swap<NumModes>(gen) : gen;
        // Odd |G| needs the per-row parity(|M|) correction (see even_parity_scan_pass1); even |G| is
        // byte-identical with no parity bitmap. Pauli never needs it (invariant above).
        const bool g_odd = IsPauli ? false : (gen.count() % 2 != 0);
        const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);
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
        // The fused sweep writes fused_scale_coeffs[i] for every anticommuting index i < n, so the coeff
        // vector must already cover the full operator and be the very array the reads come from. Both
        // hold in ContractImmediately (entry sync + per-gate extend); a violation is a caller bug —
        // assert, never a silent write-skip (a skipped scale would corrupt the 1/cos recovery later).
        assert(fused_scale_coeffs == nullptr || (fused_scale_coeffs == coeffs.data() && coeffs.size() >= n));

        const size_t last_word = word_count - 1;
        const uint64_t last_word_mask = (n % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (n % 64)) - 1);
        // Generator column list, pivot first. Pass 1 folds L1-resident blocks (combine_columns_block)
        // — no full-width sparse-scatter prologue.
        const std::span<const size_t> gen_cols(gen_columns.indices.data(), gen_columns.count);

        // Single serial sweep over all inverted-index words [0, word_count). Emit directly into the
        // result's per-rank query / source / value streams (each pre-sized to rank_count above). When
        // !capture_values, leader_val/follower_val stay size 0 and are never indexed (emit guards on it).
        auto &lq = res.leader_queries;
        auto &ls = res.leader_src;
        auto &lv = res.leader_val;
        auto &fq = res.follower_queries;
        auto &fs = res.follower_src;
        auto &fv = res.follower_val;

        // Pass 1: fold the inverted index to find anticommuting terms (see even_parity_scan_pass1).
        // Pass 1 and pass 2 stay FUSED in one loop over `nz`: splitting them (pass-1 fully, then pass-2)
        // measured +4-16% on the large fermionic workloads because `nz` spills out of L1 between them.
        // `nz` is thread_local so each shard master reuses its capacity across gates.
        thread_local std::vector<EvenParityNzWord> nz;
        size_t n_anti = 0;
        size_t n_foll = 0;
        even_parity_scan_pass1<NumModes>(inverted_index,
                                         gen_cols,
                                         gen.find_first(),
                                         /*wlo=*/0,
                                         /*whi=*/word_count,
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
            if (capture_values) {
                const double v_src = (i < coeffs.size()) ? coeffs[i] : 0.0;
                return {v_src, cut_st.use_coeff_checks ? std::abs(v_src) : 0.0};
            }
            return {0.0, cut_st.abs_coeff_for(i, coeffs)};
        };
        const bool word_aligned_cos = only_rotate_len_k == 0;
        CosineWordBuilder cos_b;
        for (const auto &w : nz) {
            if (word_aligned_cos && fused_scale_coeffs != nullptr) {
                // Fused cos sweep (ContractImmediately, k==0): every anticommuting coefficient is
                // loaded ONCE — the pre-cos value feeds the atol gate and v_src exactly as the eager
                // arm below — and stored back scaled, unconditionally and BEFORE any gate `continue`
                // (the sweep covers all anti terms; the gates only decide emission). No cosine set is
                // built: this store IS the gate's cos pass.
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const double v_src = fused_scale_coeffs[i];
                    fused_scale_coeffs[i] = v_src * fused_scale_cos;
                    const double abs_c = std::abs(v_src);
                    if (cut_st.is_below_sin(abs_c)) {
                        continue;
                    }
                    const size_t maj_pop = op.store->popcount(i);
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(maj_pop, i, abs_c, v_src, is_follower, lq, ls, lv, fq, fs, fv);
                }
            }
            else if (word_aligned_cos) {
                // No orbital gate: cosine-scale the whole word (all anticommuting terms), then per
                // bit apply the ATOL coefficient gate BEFORE the popcount ROW read. ~90–97% of
                // anticommuting terms fail this gate (their coefficient is below the sine cutoff),
                // and the gate needs only |coeff[i]| — not the row — so deferring popcount until a
                // term passes eliminates that many random packed-row cacheline loads (the dominant
                // pass-2 memory traffic). Bit-identical: same emitted set/order, same cos word.
                cos_b.push_word(w.base, w.overlap);
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const auto [v_src, abs_c] = derive_coeff(i);
                    if (cut_st.is_below_sin(abs_c)) {
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
        res.cos_blocks.push_back(cos_b.finish());
    }
    return res;
}

} // namespace monoprop::detail
