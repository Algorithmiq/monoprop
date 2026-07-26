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

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop::detail {

// The cutoff state read by the fused scan and the even-parity generator-column scan it uses.
inline auto build_majorana_evolution_cutoff_state(const std::optional<double> &atol,
                                                  std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                                                  const std::optional<double> &upper_atol,
                                                  const std::optional<double> &param) -> CutoffContext {
    const bool check_atol = atol.has_value() && local_coeffs.has_value() && param.has_value();
    const bool check_upper_atol = upper_atol.has_value() && local_coeffs.has_value();
    const double sin_val = param.has_value() ? std::sin(2 * param.value()) : 1.0;

    return CutoffContext{.check_atol = check_atol,
                         .check_upper_atol = check_upper_atol,
                         .atol_value = atol.value_or(0.0),
                         .upper_atol_value = upper_atol.value_or(0.0),
                         .abs_sin_val = std::abs(sin_val),
                         .use_coeff_checks = check_atol || check_upper_atol};
}

template <size_t NumModes>
struct EvenParityGeneratorColumns {
    std::array<size_t, Monomial<NumModes>::size()> indices{};
    size_t count = 0;
};

// Collect a generator's set columns in ASCENDING bit order. indices[0] (lowest) is the pivot ordinary
// callers pass to even_parity_scan_pass1 — the column that splits an anticommuting pair leader/follower.
template <size_t NumModes>
auto build_even_parity_generator_columns(const Monomial<NumModes> &gen_maj) -> EvenParityGeneratorColumns<NumModes> {
    EvenParityGeneratorColumns<NumModes> columns;
    for (size_t bit_idx = gen_maj.find_first(); bit_idx < gen_maj.size(); bit_idx = gen_maj.find_next(bit_idx)) {
        columns.indices[columns.count++] = bit_idx;
    }
    return columns;
}

// One nonzero-overlap word carried from scan pass 1 to emit pass 2. `overlap` bit t set ⟺ term (base+t)
// anticommutes with G; `foll` = overlap & pivot column = the followers (leaders are `overlap ^ foll`).
struct EvenParityNzWord {
    size_t base;
    uint64_t overlap;
    uint64_t foll;
};

// Even-parity scan pass 1: over words [wlo,whi), fold G's inverted index columns into a per-word overlap
// mask, keep nonzero words in `nz`, and tally popcounts (n_anti, n_foll) so pass 2 reserves once. `nz` is
// thread_local for capacity reuse. `pivot_col` (the leader/follower split bit) is read SEPARATELY from
// `gen_cols` so a caller can fold a transformed generator while splitting on the untransformed one;
// ordinary callers pass gen_cols[0]. `g_odd` XORs the per-row parity(|M|) correction (row_parity_ptr) in
// before followers are derived; even |G| ignores it and is byte-identical.
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
    // Fold one word range [bb,be): combine G's columns and record nonzero-overlap words. A DENSE pivot is
    // read inline; a SPARSE pivot is scatter-expanded LAZILY (only for blocks with a nonzero overlap, so
    // no-anticommuter blocks skip it) via a deferred follower fix-up — bit-identical to eager expansion.
    auto fold_range = [&](size_t bb, size_t be) {
        combine_columns_block<NumModes>(sc, gen_cols, blk.data(), bb, be);
        const size_t nz_block_start = nz.size();
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
            n_anti += static_cast<size_t>(std::popcount(overlap));
            uint64_t foll = 0;
            if (pivot_dense) {
                foll = overlap & pivot_dense_ptr[wi];
                n_foll += static_cast<size_t>(std::popcount(foll));
            }
            nz.push_back(EvenParityNzWord{wi * 64, overlap, foll});
        }
        if (pivot_dense || nz.size() == nz_block_start) {
            return; // dense pivot already folded in, or no anticommuting term — nothing to expand
        }
        std::vector<uint64_t> &pblk = pivot_column_block_scratch();
        combine_columns_block<NumModes>(sc, std::span<const size_t>(&pivot_col, 1), pblk.data(), bb, be);
        const uint64_t *pw = pblk.data();
        for (size_t k = nz_block_start; k < nz.size(); ++k) {
            EvenParityNzWord &e = nz[k];
            const size_t wi = e.base / 64;
            e.foll = e.overlap & pw[wi - bb];
            n_foll += static_cast<size_t>(std::popcount(e.foll));
        }
    };
    for (size_t bb = wlo; bb < whi; bb += kColumnBlockWords) {
        fold_range(bb, std::min(bb + kColumnBlockWords, whi));
    }
}

// The per-term rotation gate splits into a DYNAMIC part (orbital pop cap, upper-atol freeze, lower-atol
// sine cutoff) and a STATIC part (structural cutoff on M'=M⊕G). Every emitting path uses these helpers so
// the gate semantics cannot drift.
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

// The per-generator context, built once per layer, is owned by the algebra policy `A::GenContext`
// (Majorana: G + interleave mask; Pauli: PauliGenContext = G + |G|). See algebra/Algebra.h.

// Compute the three per-survivor products for term i: new_maj = M_i ⊕ G (the query partner),
// overlap = |M_i ∩ G| (feeds new-popcount + hermitian phase), and the phase_factor sign. Rebuilds M_i
// dense in registers from its stored position list, then evaluates with branch-free W-word kernels; the
// dynamic gate runs in the caller BEFORE this, so rejected terms cost no reconstruction.
// phase_factor is the basis-specific multiplicative sign: Majorana interleave_phase (folds hermitian_phase
// in later), Pauli pauli_rotation_sign (already rotation-ready — no extra flip at emit).
template <size_t NumModes, Algebra A>
[[gnu::always_inline]] inline auto emit_term_products(const OperatorIndex<NumModes> &ham,
                                                      size_t i,
                                                      const typename A::GenContext &ctx,
                                                      Monomial<NumModes> &new_maj,
                                                      size_t &overlap,
                                                      int &phase_factor) -> void {
    Monomial<NumModes> maj; // zero-init, W words, lives in registers
    ham.for_each_position(i, [&](size_t pos) { maj.set(pos); });
    const Monomial<NumModes> &gen = A::generator(ctx);
    new_maj = maj ^ gen;
    overlap = maj.count_and(gen);
    phase_factor = A::rotation_sign(ctx, maj, new_maj);
}

// fused_find_and_collect (any rank count): one pass fusing FindAnticommuting + apply_cutoffs — classify
// each anticommuting term leader/follower, compress it into the cosine block, and emit surviving queries.
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

// Streams are routed to the owner of each partner M'=M⊕G (hash%R; self at R==1) in ascending source-index
// order, so the downstream resolve + cross-rank index assignment are deterministic.
// `capture_values` (fused): also collect the signed pre-cos source coeff (v_src) into leader_val/follower_val.
// `fused_scale_coeffs` (k==0 only; must alias coeffs.data()): fold the gate's cosine scale into this pass —
// each anticommuting coeff is stored back ×`fused_scale_cos`=cos(2·build_angle), so no cosine set is built.
// Chunks own disjoint word ranges ⇒ race-free; a hit's stored value is then POST-cos (resolve recovers via 1/cos).
template <size_t NumModes, Algebra A>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const Monomial<NumModes> &gen,
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
    const auto ectx = A::make_gen_context(gen);

    // Cutoff + emit for one anticommuting term. The dynamic gate (|M| only) runs BEFORE emit_term_products,
    // so a gate-rejected term computes no products. abs_c/v_src are passed in from the caller's coeff read
    // (v_src the SIGNED coeff, pushed into lv/fv only when capture_values), so emit does not re-read it.
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
        Monomial<NumModes> new_maj;
        size_t overlap = 0;
        int phase_factor = 0;
        emit_term_products<NumModes, A>(*op.store, i, ectx, new_maj, overlap, phase_factor);
        // Structural cutoff on the partner M⊕G — UNLESS upper_atol rescues it (its sine coefficient is
        // large enough to keep alive despite exceeding the cutoff). See CutoffContext::is_above_upper.
        const size_t new_pop = maj_pop + gen_pop - 2 * overlap;
        const bool struct_pass = cutoff_eval.passes_with_popcount(new_maj, new_pop);
        if (!struct_pass && !cut_st.is_above_upper(abs_c)) {
            return;
        }
        // Emitted sine phase: the algebra folds the rotation sign into the final ±1 (Majorana folds in
        // hermitian_phase; Pauli's pauli_rotation_sign is already rotation-ready). See A::emit_phase.
        const int phase = A::emit_phase(phase_factor, maj_pop, gen_pop, overlap);
        // Single rank: every partner is self-owned, skip the O(W) hash; multi-rank routes by owner.
        const size_t r_prime = (rank_count == 1) ? my_rank : (monomial_hash<NumModes>(new_maj) % rank_count);
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
        // The anticommutation fold runs over G's inverted-index columns (Majorana: G; Pauli: J(G) =
        // pair_swap(G), so pauli_anticommutes = parity(|M ∩ J(G)|), and Pauli never needs the odd-|G|
        // correction since parity(|G ∩ J(G)|)=0). The pivot splitting each pair is a set bit of the REAL G
        // (gen.find_first()), NOT J(G) — A and A⊕G differ exactly on G's bits.
        const Monomial<NumModes> fold_gen = A::fold_generator(gen);
        // Odd |G| needs the per-row parity(|M|) correction (see even_parity_scan_pass1); even |G| is
        // byte-identical with no parity bitmap. Pauli never needs it (invariant above).
        const bool g_odd = A::fold_needs_odd_correction(gen);
        const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);
        if (gen_columns.count == 0) {
            return res;
        }
        const auto &inverted_index = op.inverted_index();
        const size_t word_count = inverted_index.words();
        if (word_count == 0) {
            return res;
        }
        const uint64_t *const row_parity_ptr = g_odd ? inverted_index.row_parity_words() : nullptr;
        const size_t n = op.store->size();
        // The fused sweep writes fused_scale_coeffs[i] for every anticommuting i < n, so it must be the
        // very array the reads come from and cover the full operator — a violation corrupts 1/cos recovery,
        // so assert rather than silently skip.
        assert(fused_scale_coeffs == nullptr || (fused_scale_coeffs == coeffs.data() && coeffs.size() >= n));

        const size_t last_word = word_count - 1;
        const uint64_t last_word_mask = (n % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (n % 64)) - 1);
        // Generator column list, pivot first. Pass 1 folds L1-resident blocks — no sparse-scatter prologue.
        const std::span<const size_t> gen_cols(gen_columns.indices.data(), gen_columns.count);

        // Classify the fold columns in O(|G|): the scan can be skipped only when every fold column is
        // empty (a dense column always holds ≥1 posting, so any dense column makes it non-empty).
        bool fold_cols_empty = true;
        for (size_t ci = 0; ci < gen_columns.count; ++ci) {
            const size_t c = gen_columns.indices[ci];
            if (inverted_index.column_is_dense(c)) {
                fold_cols_empty = false;
            }
            else if (!inverted_index.sparse_column_rows(c).empty()) {
                fold_cols_empty = false;
            }
        }
        // Zero-postings early-out: no term touches a fold column ⇒ nothing anticommutes (for even |G|),
        // so pass 1 would produce an empty nz — skip it, byte-identical downstream. The g_odd guard is
        // load-bearing: an odd Majorana generator anticommutes with disjoint odd-weight terms.
        const bool skip_scan = !g_odd && fold_cols_empty;

        // Single serial sweep over all inverted-index words, emitting directly into the result's per-rank
        // query/source/value streams. When !capture_values, leader_val/follower_val stay size 0 (emit guards).
        auto &lq = res.leader_queries;
        auto &ls = res.leader_src;
        auto &lv = res.leader_val;
        auto &fq = res.follower_queries;
        auto &fs = res.follower_src;
        auto &fv = res.follower_val;

        // Pass 1: fold the inverted index to find anticommuting terms (see even_parity_scan_pass1). Pass 1
        // and pass 2 stay FUSED over `nz` (splitting them measured +4-16% — `nz` spills L1 between them).
        // `nz` is thread_local so each shard master reuses its capacity across gates.
        thread_local std::vector<EvenParityNzWord> nz;
        size_t n_anti = 0;
        size_t n_foll = 0;
        if (skip_scan) {
            nz.clear(); // pass 1 clears it on entry; the skip must too (thread_local reuse)
        }
        else {
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
        }
        if (rank_count == 1) {
            lq[my_rank].reserve((n_anti - n_foll) * kQueryWords<NumModes>);
            ls[my_rank].reserve(n_anti - n_foll);
            fq[my_rank].reserve(n_foll * kQueryWords<NumModes>);
            fs[my_rank].reserve(n_foll);
        }
        // Pass 2: collect cosine for EVERY anticommuting term, then apply cutoff + emit the query. No
        // orbital gate → push each word's full overlap (push_word); orbital gate → per-index (push_index).
        // Derive (v_src, abs_c) for term i, shared by both pass-2 arms. Fused captures the SIGNED v_src and
        // derives abs_c from it (bit-identical to abs_coeff_for, so the OFF path is unchanged).
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
                // Fused cos sweep (k==0): cosine-scale in place all anticommuting terms and emit survivors.
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
                // No orbital gate: cosine-scale the whole word, then per bit apply the ATOL gate BEFORE the
                // popcount ROW read — deferring popcount until a term passes saves random packed-row loads.
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
