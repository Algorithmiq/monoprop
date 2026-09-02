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
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"
#include "monoprop/detail/mpi/Comm.h"
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

// One partner M⊕G as ascending positions plus the (k, d) digest the structural cutoff reads, the
// cancelled-slot count the Hermitian phase reads, and the basis sign -- all from the source row's
// positions. The dense form is built only where something still reads it: the splitmix router's hash, an
// opaque cutoff, or a sign kernel without a position form (a Pauli generator on more than 32 qubits).
template <size_t NumModes>
struct PartnerProduct {
    size_t k = 0;       // popcount(M⊕G)
    size_t overlap = 0; // slots in both M and G, which cancel
    size_t paired = 0;  // modes of M⊕G carrying both positions: the d of the (k, d) digest
    int phase_factor = 0;
    bool has_dense = false;
    Monomial<NumModes> dense; // M⊕G; valid iff has_dense
};

// Paired modes of an ascending position list: an even position immediately followed by its successor.
template <typename PosT>
[[nodiscard]] inline auto count_paired_positions(const PosT *pos, size_t k) noexcept -> size_t {
    size_t d = 0;
    for (size_t j = 0; j + 1 < k; ++j) {
        const auto p = static_cast<size_t>(pos[j]);
        if (p % 2 == 0 && static_cast<size_t>(pos[j + 1]) == p + 1) {
            ++d;
            ++j;
        }
    }
    return d;
}

// phase_factor is the basis-specific sign only: Majorana interleave_phase, still to be folded with
// hermitian_phase at emit; Pauli pauli_rotation_sign, already rotation-ready. `src` is row i's positions
// as the caller already read them (a spilled row has none, and that case walks the dense form instead).
template <size_t NumModes, Algebra A, typename PosT, typename GenT>
[[gnu::always_inline]] inline auto emit_partner(const OperatorIndex<NumModes> &ham,
                                                size_t i,
                                                typename OperatorIndex<NumModes>::RowPositions src,
                                                const typename A::GenContext &ctx,
                                                std::span<const GenT> gen_pos,
                                                std::span<PosT> out_pos,
                                                bool need_dense) -> PartnerProduct<NumModes> {
    const Monomial<NumModes> &gen = A::generator(ctx);
    PartnerProduct<NumModes> out;
    if (src.inlined()) {
        const auto merged = merge_partner_positions(src.pos, gen_pos, out_pos);
        out.k = merged.count;
        out.overlap = merged.overlap;
        out.paired = merged.paired;
        if (A::sign_from_positions_ok(ctx)) {
            out.phase_factor = A::rotation_sign_positions(ctx, src.pos.data(), src.pos.size());
            if (need_dense) {
                for (size_t j = 0; j < out.k; ++j) {
                    out.dense.set(static_cast<size_t>(out_pos[j]));
                }
                out.has_dense = true;
            }
        }
        else {
            Monomial<NumModes> mono;
            for (const PosT q : src.pos) {
                mono.set(static_cast<size_t>(q));
            }
            out.dense = mono ^ gen;
            out.has_dense = true;
            out.phase_factor = A::rotation_sign(ctx, mono, out.dense);
        }
        return out;
    }
    Monomial<NumModes> mono;
    ham.for_each_position(i, [&](size_t pos) { mono.set(pos); });
    out.dense = mono ^ gen;
    out.has_dense = true;
    out.overlap = mono.count_and(gen);
    for (size_t b = out.dense.find_first(); b < out.dense.size(); b = out.dense.find_next(b)) {
        out_pos[out.k++] = static_cast<PosT>(b);
    }
    out.paired = count_paired_positions(out_pos.data(), out.k);
    out.phase_factor = A::rotation_sign(ctx, mono, out.dense);
    return out;
}

template <size_t NumModes>
struct FusedScanResult {
    std::vector<CosMask> cos_blocks; // ascending, disjoint, chunk order
    // The six arrays below are indexed by DESTINATION SLOT through WindowVec::at_slot, and cover only
    // the slots this generator can reach: S of the P=R*S world under linear routing, all P under
    // splitmix. See mpi::PeerPlan::window.
    mpi::SlotWindow window;
    mpi::WindowVec<VecZ> leader_queries;              // serialized leader queries per owner slot
    mpi::WindowVec<std::vector<size_t>> leader_src;   // parallel to leader_queries (source op idx)
    mpi::WindowVec<VecZ> follower_queries;            // serialized follower queries per owner slot
    mpi::WindowVec<std::vector<size_t>> follower_src; // parallel to follower_queries
    // Fused-contraction only (capture_values): signed pre-cos source coeff (v_src) parallel to
    // leader_src / follower_src. Empty when capture_values is false.
    mpi::WindowVec<std::vector<double>> leader_val;
    mpi::WindowVec<std::vector<double>> follower_val;
    // Self-owned queries, staged as positions instead of encoded into the window's self slot, and resolved
    // inline. Order must match that slot's leader_src / follower_src, or resolution attributes the wrong
    // source. Empty unless the window contains my_rank, i.e. unless this generator's rank shift is zero.
    SelfQueryStage<NumModes> leader_self;
    SelfQueryStage<NumModes> follower_self;
};

// Classify, cut off and emit in one pass over the anticommuting terms. Queries go to the owner of
// M'=M⊕G (routing::Router; self at R==1) in ascending source-index order, so resolve and index assignment are
// deterministic. `fused_scale_coeffs` (no length cap only; must alias coeffs.data()) scales every anticommuting
// coeff in place by `fused_scale_cos`=cos(2·build_angle), so no cosine set is built and a hit's stored
// value is post-cos (resolve recovers it via 1/cos).
//
// Every anticommuting row -- emitted or not -- is entered into scratch.anti keyed by its fingerprint
// before any gate runs, because a term below the threshold or above the rotation cap is still somebody's
// partner (AntiTable.h). The partner's fingerprint is fp(M) ^ fp(G), which is also what routes it: under
// linear routing no per-term hash and no dense monomial exist on this path. `window` must be the peer
// plan's window for `my_rank`: it is what the six query arrays are sized to.
template <size_t NumModes, Algebra A>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const Monomial<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const CutoffContext &cut_st,
                            const VecD &coeffs,
                            std::optional<size_t> only_rotate_len_k,
                            mpi::SlotWindow window,
                            size_t my_rank,
                            const routing::Router &router,
                            GateScratch<NumModes> &scratch,
                            bool capture_values = false,
                            double *fused_scale_coeffs = nullptr,
                            double fused_scale_cos = 1.0) -> FusedScanResult<NumModes> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t gen_pop = gen.count();
    const size_t rank_count = router.flat_world();
    const auto ectx = A::make_gen_context(gen);
    assert(window.stop() <= rank_count && window.count != 0);

    FusedScanResult<NumModes> res;
    res.window = window;
    res.leader_queries.reset(window);
    res.leader_src.reset(window);
    res.follower_queries.reset(window);
    res.follower_src.reset(window);
    // Sized on the early-return paths below too, so the fused engine's per-slot src_val_r access is
    // always in bounds (parallel to leader_src / follower_src).
    if (capture_values) {
        res.leader_val.reset(window);
        res.follower_val.reset(window);
    }
    // An empty table on every early return: the resolve passes probe it for every query they carry.
    scratch.anti.begin(0);

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
        using RowPositions = typename OperatorIndex<NumModes>::RowPositions;
        AntiTable<NumModes> &table = scratch.anti;

        // The generator's positions, once per gate: the merge's second input. Its fingerprint is the
        // per-gate constant every partner's fingerprint is one XOR away from.
        std::vector<uint16_t> &gen_pos = scratch.gen;
        gen_pos.clear();
        for (size_t b = gen.find_first(); b < gen.size(); b = gen.find_next(b)) {
            gen_pos.push_back(static_cast<uint16_t>(b));
        }
        const uint64_t *const labels = routing::linear_basis<2 * NumModes>().data();
        const uint64_t fp_gen = routing::linear_hash<2 * NumModes>(gen);
        // pbuf capacity is 2*NumModes: the partner's positions are distinct, so this always suffices.
        std::vector<RowPosT> &pbuf = scratch.partner;
        if (pbuf.size() < 2 * NumModes) {
            pbuf.resize(2 * NumModes);
        }
        // What still reads the dense partner: the splitmix router (its hash), an opaque cutoff, or a sign
        // kernel with no position form. Under linear routing with a structural cutoff: nothing.
        const bool need_dense = (rank_count != 1 && !router.is_linear()) || !cutoff_eval.has_digest_form()
                                || !A::sign_from_positions_ok(ectx);

        auto push = [&](uint64_t fp_partner,
                        const PartnerProduct<NumModes> &p,
                        std::span<const RowPosT> pos,
                        int phase,
                        size_t i,
                        double v_src,
                        bool is_follower) {
            // Single rank: every partner is self-owned. Multi-rank routes by owner: off the fingerprint
            // under linear routing, off the dense hash under splitmix. routing::Router is the only owner
            // function; find_rank (MPIUtils.h) must stay in step with it or a term is placed and queried
            // on different ranks, which duplicates a row silently.
            size_t r_prime = my_rank;
            if (rank_count != 1) {
                if (router.is_linear()) {
                    r_prime = router.dest_from_fingerprint(fp_partner);
                    assert(!p.has_dense || r_prime == router.dest<NumModes>(p.dense)); // an identity
                }
                else {
                    assert(p.has_dense);
                    r_prime = router.dest<NumModes>(p.dense);
                }
            }
            // at_slot is the only re-basing door and asserts membership: a destination outside this
            // generator's window means the shift is wrong, and would otherwise land on another peer.
            if (r_prime == my_rank) {
                (is_follower ? res.follower_self : res.leader_self).push(pos, phase, fp_partner);
            }
            else {
                QueryWire<NumModes>::push(is_follower ? fq.at_slot(r_prime) : lq.at_slot(r_prime), pos, phase);
            }
            (is_follower ? fs : ls).at_slot(r_prime).push_back(i);
            if (capture_values) {
                (is_follower ? fv : lv).at_slot(r_prime).push_back(v_src);
            }
        };

        // Row i as the scan reads it: its positions (empty span for a spilled row), popcount and
        // fingerprint. Read once per anticommuting row, before any gate, because the row enters the
        // partner table whether or not it is emitted.
        struct RowRead {
            RowPositions src;
            size_t pop;
            uint64_t fp;
        };
        auto read_row = [&](size_t i) -> RowRead {
            const RowPositions src = ham.row_positions(i);
            if (src.inlined()) {
                return {src, src.pos.size(), routing::fingerprint_positions(labels, src.pos.data(), src.pos.size())};
            }
            const Monomial<NumModes> dense = ham.row(i);
            return {src, dense.count(), routing::linear_hash<2 * NumModes>(dense)};
        };

        // The dynamic gate runs before emit_partner, so a gate-rejected term computes no products.
        // abs_c/v_src come from the caller's coeff read, not re-read.
        auto emit = [&](const RowRead &row, size_t i, double abs_c, double v_src, bool is_follower) {
            if (!rotation_dynamic_gate(only_rotate_len_k, row.pop, cut_st, abs_c)) {
                return;
            }
            const auto p = emit_partner<NumModes, A>(ham,
                                                     i,
                                                     row.src,
                                                     ectx,
                                                     std::span<const uint16_t>(gen_pos),
                                                     std::span<RowPosT>(pbuf),
                                                     need_dense);
            // Structural cutoff on the partner M⊕G, unless upper_atol rescues it (CutoffContext::is_above_upper).
            const bool struct_pass = cutoff_eval.has_digest_form() ? cutoff_eval.passes_from_digest(p.k, p.paired)
                                                                   : cutoff_eval.passes_with_popcount(p.dense, p.k);
            if (!struct_pass && !cut_st.is_above_upper(abs_c)) {
                return;
            }
            const int phase = A::emit_phase(p.phase_factor, row.pop, gen_pop, p.overlap);
            push(row.fp ^ fp_gen, p, std::span<const RowPosT>(pbuf).first(p.k), phase, i, v_src, is_follower);
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
        table.begin(n_anti);
        if (rank_count == 1) {
            // A hint only; wider terms grow the buffer as needed.
            const size_t pq = QueryWire<NumModes>::kReservePositionsPerQuery;
            res.leader_self.reserve(n_anti - n_foll, pq);
            ls.at_slot(my_rank).reserve(n_anti - n_foll);
            res.follower_self.reserve(n_foll, pq);
            fs.at_slot(my_rank).reserve(n_foll);
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
                    const RowRead row = read_row(i);
                    table.add(static_cast<TermIndex>(i), row.fp);
                    const double v_src = fused_scale_coeffs[i];
                    fused_scale_coeffs[i] = v_src * fused_scale_cos;
                    const double abs_c = std::abs(v_src);
                    if (cut_st.is_below_sin(abs_c)) {
                        continue;
                    }
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(row, i, abs_c, v_src, is_follower);
                }
            }
            else if (word_aligned_cos) {
                // No orbital gate: record the whole word in the cosine set, then per bit apply the atol gate.
                cos_b.push_word(w.base, w.overlap);
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const RowRead row = read_row(i);
                    table.add(static_cast<TermIndex>(i), row.fp);
                    const auto [v_src, abs_c] = derive_coeff(i);
                    if (cut_st.is_below_sin(abs_c)) {
                        continue;
                    }
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(row, i, abs_c, v_src, is_follower);
                }
            }
            else {
                // Orbital gate active: the per-index cosine push covers only orbital-passing terms, so the
                // popcount test precedes it. A capped term is still entered into the partner table.
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const RowRead row = read_row(i);
                    table.add(static_cast<TermIndex>(i), row.fp);
                    if (row.pop > static_cast<size_t>(*only_rotate_len_k)) {
                        continue;
                    }
                    cos_b.push_index(i);
                    const auto [v_src, abs_c] = derive_coeff(i);
                    const bool is_follower = (w.foll >> tz) & 1u;
                    emit(row, i, abs_c, v_src, is_follower);
                }
            }
        }
        res.cos_blocks.push_back(cos_b.finish());
    }
    return res;
}

} // namespace monoprop::detail
