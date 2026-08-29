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
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop::detail {

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

// Set columns in ascending bit order.
template <size_t NumModes>
auto build_even_parity_generator_columns(const Monomial<NumModes> &gen_mono) -> EvenParityGeneratorColumns<NumModes> {
    EvenParityGeneratorColumns<NumModes> columns;
    for (size_t bit_idx = gen_mono.find_first(); bit_idx < gen_mono.size(); bit_idx = gen_mono.find_next(bit_idx)) {
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

// Even-parity scan pass 1 over words [wlo,whi). n_anti/n_foll are tallied here so pass 2 reserves once.
// `pivot_col` is read separately from `gen_cols` so a caller can fold a transformed generator while
// splitting on the untransformed one. `g_odd` XORs the per-row parity(|M|) correction (row_parity_ptr)
// in before followers are derived.
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
    // A dense pivot is read inline; a sparse pivot is scatter-expanded lazily (only for blocks with a
    // nonzero overlap, so no-anticommuter blocks skip it) via a deferred follower fix-up — bit-identical
    // to eager expansion.
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

// The per-term rotation gate splits into a dynamic part (orbital pop cap, lower-atol sine cutoff) and a
// static part (the structural cutoff on M'=M⊕G, applied in emit).
inline auto rotation_dynamic_gate(std::optional<size_t> only_rotate_len_k,
                                  size_t mono_pop,
                                  const CutoffContext &ctx,
                                  double abs_c) -> bool {
    if (only_rotate_len_k && mono_pop > static_cast<size_t>(*only_rotate_len_k)) {
        return false;
    }
    if (ctx.is_below_sin(abs_c)) {
        return false;
    }
    return true;
}

// The dense form is unavoidable: the owner hash folds every word and the basis sign reads the source
// bitset, so it is built regardless, and the merge below runs beside it.
template <size_t NumModes>
struct PartnerProduct {
    Monomial<NumModes> new_mono;
    size_t k = 0;       // popcount(M⊕G)
    size_t overlap = 0; // slots in both M and G, which cancel
    int phase_factor = 0;
};

// phase_factor is the basis-specific sign only: Majorana interleave_phase, still to be folded with
// hermitian_phase at emit; Pauli pauli_rotation_sign, already rotation-ready. `out_pos` receives the
// partner's ascending positions; a spilled source row has no position array, so that case walks the
// dense partner to fill it instead.
template <size_t NumModes, Algebra A, typename PosT, typename GenT>
[[gnu::always_inline]] inline auto emit_term_products(const OperatorIndex<NumModes> &ham,
                                                      size_t i,
                                                      const typename A::GenContext &ctx,
                                                      std::span<const GenT> gen_pos,
                                                      std::span<PosT> out_pos) -> PartnerProduct<NumModes> {
    const Monomial<NumModes> &gen = A::generator(ctx);
    PartnerProduct<NumModes> out;
    Monomial<NumModes> mono;
    if (const auto src = ham.row_positions(i); src.inlined()) {
        const auto merged = merge_partner_positions(src.pos, gen_pos, out_pos);
        out.k = merged.count;
        out.overlap = merged.overlap;
        for (const PosT q : src.pos) {
            mono.set(static_cast<size_t>(q));
        }
        out.new_mono = mono ^ gen;
    }
    else {
        ham.for_each_position(i, [&](size_t pos) { mono.set(pos); });
        out.new_mono = mono ^ gen;
        out.overlap = mono.count_and(gen);
        for (size_t b = out.new_mono.find_first(); b < out.new_mono.size(); b = out.new_mono.find_next(b)) {
            out_pos[out.k++] = static_cast<PosT>(b);
        }
    }
    out.phase_factor = A::rotation_sign(ctx, mono, out.new_mono);
    return out;
}

template <size_t NumModes>
struct FusedScanResult {
    std::vector<CosMask> cos_blocks;               // ascending, disjoint, chunk order
    std::vector<VecZ> leader_queries;              // size R: serialized leader queries per owner rank
    std::vector<std::vector<size_t>> leader_src;   // size R: parallel to leader_queries (source op idx)
    std::vector<VecZ> follower_queries;            // size R: serialized follower queries per owner rank
    std::vector<std::vector<size_t>> follower_src; // size R: parallel to follower_queries
    // Fused-contraction only (capture_values): signed pre-cos source coeff (v_src) parallel to
    // leader_src / follower_src. Empty when capture_values is false.
    std::vector<std::vector<double>> leader_val;
    std::vector<std::vector<double>> follower_val;
    // Self-owned queries, staged as positions instead of queued to the wire and resolved inline.
    // Order must match leader_src[my_rank] / follower_src[my_rank], or resolution attributes the wrong source.
    SelfQueryStage<NumModes> leader_self;
    SelfQueryStage<NumModes> follower_self;
};

// Classify, cut off and emit in one pass over the anticommuting terms. Queries go to the owner of
// M'=M⊕G (routing::Router; self at R==1) in ascending source-index order, so resolve and index assignment are
// deterministic. `fused_scale_coeffs` (no length cap only; must alias coeffs.data()) scales every anticommuting
// coeff in place by `fused_scale_cos`=cos(2·build_angle), so no cosine set is built and a hit's stored
// value is post-cos (resolve recovers it via 1/cos).
//
// `gen_shift` is router.rank_shift(gen), and `op` must hold only terms `my_rank` owns -- then the owner of
// M⊕G is rank(M) ^ gen_shift and the linear planes never run per term. Both are what mpi::PeerPlan
// already assumes; a violation moves ownership silently, so the fast path asserts against dest().
template <size_t NumModes, Algebra A>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const Monomial<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const CutoffContext &cut_st,
                            const VecD &coeffs,
                            std::optional<size_t> only_rotate_len_k,
                            size_t rank_count,
                            size_t my_rank,
                            const routing::Router &router,
                            size_t gen_shift,
                            bool capture_values = false,
                            double *fused_scale_coeffs = nullptr,
                            double fused_scale_cos = 1.0) -> FusedScanResult<NumModes> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t gen_pop = gen.count();
    const auto ectx = A::make_gen_context(gen);

    FusedScanResult<NumModes> res;
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
        // correction since parity(|G ∩ J(G)|)=0). The pivot splitting each pair is a set bit of the real G
        // (gen.find_first()), not J(G) — A and A⊕G differ exactly on G's bits.
        const Monomial<NumModes> fold_gen = A::fold_generator(gen);
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
        // very array the reads come from and cover the full operator — a violation corrupts 1/cos recovery.
        assert(fused_scale_coeffs == nullptr || (fused_scale_coeffs == coeffs.data() && coeffs.size() >= n));

        const size_t last_word = word_count - 1;
        const uint64_t last_word_mask = (n % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (n % 64)) - 1);
        const std::span<const size_t> gen_cols(gen_columns.indices.data(), gen_columns.count);

        // The scan can be skipped only when every fold column is empty; a dense column always holds ≥1
        // posting, so any dense column makes it non-empty.
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
        // Zero-postings early-out: no term touches a fold column ⇒ nothing anticommutes, so skip pass 1.
        // g_odd guard is load-bearing: an odd Majorana generator anticommutes with disjoint odd-weight terms.
        const bool skip_scan = !g_odd && fold_cols_empty;

        auto &lq = res.leader_queries;
        auto &ls = res.leader_src;
        auto &lv = res.leader_val;
        auto &fq = res.follower_queries;
        auto &fs = res.follower_src;
        auto &fv = res.follower_val;

        const OperatorIndex<NumModes> &ham = *op.store;
        using RowPosT = typename OperatorIndex<NumModes>::PosT;

        // The generator's positions, once per gate: the merge's second input.
        std::vector<uint16_t> gen_pos;
        gen_pos.reserve(gen_pop);
        for (size_t b = gen.find_first(); b < gen.size(); b = gen.find_next(b)) {
            gen_pos.push_back(static_cast<uint16_t>(b));
        }
        // pbuf capacity is 2*NumModes: the partner's positions are distinct, so this always suffices.
        std::vector<RowPosT> pbuf(2 * NumModes);

        auto push = [&](const Monomial<NumModes> &dense,
                        std::span<const RowPosT> pos,
                        int phase,
                        size_t i,
                        double v_src,
                        bool is_follower) {
            // Single rank: every partner is self-owned, skip the O(W) hash; multi-rank routes by owner.
            // routing::Router is the only owner function; find_rank (MPIUtils.h) must stay in step with it
            // or a term is placed and queried on different ranks, which duplicates a row silently.
            size_t r_prime = my_rank;
            if (rank_count != 1) {
                r_prime = router.dest_from_shift<NumModes>(dense, my_rank, gen_shift);
                assert(r_prime == router.dest<NumModes>(dense)); // an identity, not an approximation
            }
            if (r_prime == my_rank) {
                (is_follower ? res.follower_self : res.leader_self).push(pos, phase);
            }
            else {
                QueryWire<NumModes>::push(is_follower ? fq[r_prime] : lq[r_prime], pos, phase);
            }
            (is_follower ? fs[r_prime] : ls[r_prime]).push_back(i);
            if (capture_values) {
                (is_follower ? fv[r_prime] : lv[r_prime]).push_back(v_src);
            }
        };

        // The dynamic gate runs before emit_term_products, so a gate-rejected term computes no products.
        // abs_c/v_src come from the caller's coeff read, not re-read.
        auto emit = [&](size_t mono_pop, size_t i, double abs_c, double v_src, bool is_follower) {
            if (!rotation_dynamic_gate(only_rotate_len_k, mono_pop, cut_st, abs_c)) {
                return;
            }
            const auto p = emit_term_products<NumModes, A>(ham,
                                                           i,
                                                           ectx,
                                                           std::span<const uint16_t>(gen_pos),
                                                           std::span<RowPosT>(pbuf));
            // Structural cutoff on the partner M⊕G, unless upper_atol rescues it (CutoffContext::is_above_upper).
            const bool struct_pass = cutoff_eval.passes_with_popcount(p.new_mono, p.k);
            if (!struct_pass && !cut_st.is_above_upper(abs_c)) {
                return;
            }
            const int phase = A::emit_phase(p.phase_factor, mono_pop, gen_pop, p.overlap);
            push(p.new_mono, std::span<const RowPosT>(pbuf).first(p.k), phase, i, v_src, is_follower);
        };

        // Pass 1 and pass 2 stay fused over `nz`: splitting them regressed measurably, as `nz` spills L1
        // between them. `nz` is thread_local so each partition master reuses its capacity across gates.
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
            // A hint only; wider terms grow the buffer as needed.
            const size_t pq = QueryWire<NumModes>::kReservePositionsPerQuery;
            res.leader_self.reserve(n_anti - n_foll, pq);
            ls[my_rank].reserve(n_anti - n_foll);
            res.follower_self.reserve(n_foll, pq);
            fs[my_rank].reserve(n_foll);
        }
        auto derive_coeff = [&](size_t i) -> std::pair<double, double> {
            if (capture_values) {
                const double v_src = (i < coeffs.size()) ? coeffs[i] : 0.0;
                return {v_src, cut_st.use_coeff_checks ? std::abs(v_src) : 0.0};
            }
            return {0.0, cut_st.abs_coeff_for(i, coeffs)};
        };
        const bool word_aligned_cos = !only_rotate_len_k.has_value();
        CosineWordBuilder cos_b;
        for (const auto &w : nz) {
            if (word_aligned_cos && fused_scale_coeffs != nullptr) {
                // Fused cos sweep: scaling in place here is what replaces building a cosine set.
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const double v_src = fused_scale_coeffs[i];
                    fused_scale_coeffs[i] = v_src * fused_scale_cos;
                    const double abs_c = std::abs(v_src);
                    if (cut_st.is_below_sin(abs_c)) {
                        continue;
                    }
                    const size_t mono_pop = op.store->popcount(i);
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(mono_pop, i, abs_c, v_src, is_follower);
                }
            }
            else if (word_aligned_cos) {
                // No orbital gate: record the whole word in the cosine set, then per bit apply the atol
                // gate before the popcount row read — deferring popcount saves random packed-row loads.
                cos_b.push_word(w.base, w.overlap);
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const auto [v_src, abs_c] = derive_coeff(i);
                    if (cut_st.is_below_sin(abs_c)) {
                        continue;
                    }
                    const size_t mono_pop = op.store->popcount(i);
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(mono_pop, i, abs_c, v_src, is_follower);
                }
            }
            else {
                // Orbital gate active: it needs mono_pop, and the per-index cosine push covers only
                // orbital-passing terms, so the popcount row read must precede both.
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const size_t mono_pop = op.store->popcount(i);
                    if (mono_pop > static_cast<size_t>(*only_rotate_len_k)) {
                        continue;
                    }
                    cos_b.push_index(i);
                    const auto [v_src, abs_c] = derive_coeff(i);
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(mono_pop, i, abs_c, v_src, is_follower);
                }
            }
        }
        res.cos_blocks.push_back(cos_b.finish());
    }
    return res;
}

} // namespace monoprop::detail
