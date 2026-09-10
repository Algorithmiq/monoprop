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
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowKey.h"

namespace monoprop::detail {

// Floor of the self-slot reserve, so a gate that follows a silent one still starts with room: below this
// the estimate is not worth having and push()'s own first growth step is the same size.
inline constexpr size_t kSelfReserveFloor = 64;

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

// Even-parity scan pass 1 over words [wlo,whi). n_anti is tallied here so pass 2 reserves once.
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
                                   size_t &n_anti) -> void {
    nz.clear();
    n_anti = 0;
    const bool pivot_dense = sc.column_is_dense(pivot_col);
    std::vector<uint64_t> &blk = column_block_scratch();
    // A dense pivot is read inline; a sparse pivot is scatter-expanded lazily (only for blocks with a
    // nonzero overlap, so no-anticommuter blocks skip it) via a deferred follower fix-up — bit-identical
    // to eager expansion.
    auto fold_range = [&](size_t bb, size_t be) {
        combine_columns_block<NumModes>(sc, gen_cols, blk.data(), bb, be);
        // Block-scoped: a dense column is stored in chunks, so its words are contiguous only within
        // one fold block (InvertedIndex::dense_column_block).
        const uint64_t *const pivot_dense_ptr = pivot_dense ? sc.dense_column_block(pivot_col, bb, be) : nullptr;
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
            const uint64_t foll = pivot_dense ? (overlap & pivot_dense_ptr[wi - bb]) : uint64_t{0};
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
        }
    };
    for (size_t bb = wlo; bb < whi; bb += kColumnBlockWords) {
        fold_range(bb, std::min(bb + kColumnBlockWords, whi));
    }
}

// The per-term rotation gate splits into a dynamic part (orbital pop cap, lower-atol sine cutoff) and a
// static part (the structural cutoff on the partner, applied in emit). The dynamic part is two
// independent tests, kept separate because only one of them needs the row: the value path reads the
// coefficient half first and, when it fails, never reads the row at all (emit_row below).
inline auto rotation_coeff_gate(const CutoffContext &ctx, double abs_c) -> bool {
    return !ctx.is_below_sin(abs_c);
}
inline auto rotation_pop_gate(std::optional<size_t> only_rotate_len_k, size_t mono_pop) -> bool {
    return !only_rotate_len_k.has_value() || mono_pop <= static_cast<size_t>(*only_rotate_len_k);
}

/*! @brief One partner M^G as ascending positions plus what the emit reads off it.
 *
 *  The (k, d) digest the structural cutoff reads, the cancelled-slot count the Hermitian phase reads, and
 *  the basis sign -- all from the source row's positions. The dense form is built only where something
 *  still reads it: the splitmix router's hash, an opaque cutoff, or a sign kernel without a position form
 *  (a Pauli generator on more than 32 qubits).
 */
template <size_t NumModes>
struct PartnerProduct {
    size_t k = 0;       // popcount(M^G)
    size_t overlap = 0; // slots in both M and G, which cancel
    size_t paired = 0;  // modes of M^G carrying both positions: the d of the (k, d) digest
    int phase_factor = 0;
    bool sign_pending = false; // phase_factor not filled: the caller asked for the digest only
    bool has_dense = false;
    Monomial<NumModes> dense; // M^G; valid iff has_dense
};

/*! @brief The partner product of row @a i, from the positions the caller has already read.
 *
 *  `phase_factor` is the basis-specific sign only: Majorana interleave_phase, still to be folded with
 *  hermitian_phase at emit; Pauli pauli_rotation_sign, already rotation-ready. `src` is row i's positions
 *  (a spilled row has none, and that case walks the dense form instead).
 *
 *  `need_sign` false returns the digest with `sign_pending` set and no sign computed, for a caller whose
 *  send predicate reads the digest first and may drop the record: the sign is then taken from the same
 *  positions once the record is known to go out. The paths that build the dense partner compute the sign
 *  on the way there, so they honour the request only in the common inline case.
 */
template <size_t NumModes, Algebra A, typename PosT, typename GenT>
[[gnu::always_inline]] inline auto emit_partner(const OperatorIndex<NumModes> &ham,
                                                size_t i,
                                                typename OperatorIndex<NumModes>::RowPositions src,
                                                const typename A::GenContext &ctx,
                                                std::span<const GenT> gen_pos,
                                                std::span<PosT> out_pos,
                                                bool need_dense,
                                                bool need_sign = true) -> PartnerProduct<NumModes> {
    const Monomial<NumModes> &gen = A::generator(ctx);
    PartnerProduct<NumModes> out;
    if (src.inlined()) {
        const auto merged = merge_partner_positions(src.pos, gen_pos, out_pos);
        out.k = merged.count;
        out.overlap = merged.overlap;
        out.paired = merged.paired;
        if (A::sign_from_positions_ok(ctx)) {
            if (need_sign) {
                out.phase_factor = A::rotation_sign_positions(ctx, src.pos.data(), src.pos.size());
            }
            else {
                out.sign_pending = true;
            }
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
    // The wire records (QueryWire.h) per destination slot, in stream order = ascending source row. Fused
    // (value word after each record) iff capture_values.
    std::vector<VecZ> queries;
    // Per destination slot, the SOURCE row and emit phase of each record, parallel to the records of that
    // slot (the self slot's parallel to `self`). Ascending in row, because the scan walks rows ascending:
    // the absence pass and the graph sink's out lists rely on that order.
    std::vector<std::vector<SentRecord>> sent;
    // Parallel to `sent`: the pre-gate coefficient the partner would be minted with (Schrodinger: the
    // state score of a fully paired partner, else 0). Only filled when a state mask is given, i.e. for
    // the fused Schrodinger picture; read by the absence pass alone.
    std::vector<std::vector<double>> sent_c0;
    // Records addressed to this slot itself, staged as positions instead of encoded into the wire's self
    // slot, and joined inline.
    SelfQueryStage<NumModes> self;
};

/*! @brief Classify, cut off and emit in one pass over the anticommuting terms.
 *
 *  Every anticommuting row records its pivot bit (`foll`) in scratch.marks and then decides two things
 *  about its partner M^G (Engine.h has the protocol):
 *    E(M)    = coeff gate and pop gate and (struct_pass(partner) or is_above_upper(|c|))   its `rot` bit
 *    send(M) = CaptureValues ? E(M) : struct_pass(partner) or over_cutoff_possible
 *  The value path sends records for emitting terms ONLY: a silent term's contribution reaches its partner
 *  as a round-2 response instead, so it needs neither the row, the merge nor the wire, and the coefficient
 *  half of E is therefore tested first. Graph mode has no coefficients and so no silent terms to answer,
 *  and keeps the symmetric predicate its positional in/out pairing is proved on.
 *
 *  Records go to the owner of the partner (routing::Router; self at R==1) in ascending source-index order,
 *  so the join and the miss-index assignment are deterministic. The rows themselves are NOT indexed here:
 *  the records probe the operator's persistent term table after the exchange (TableJoin.h), so the only
 *  per-term row work the scan does is the partner merge every record needs anyway.
 *
 *  `fused_scale_coeffs` (no length cap only; must alias coeffs.data()) scales the anticommuting coeffs in
 *  place by `fused_scale_cos` = cos(2*build_angle), so no cosine set is built; the value a record carries
 *  is the PRE-cos coefficient, read before the store. EVERY anticommuting row is scaled here, in the one
 *  pass that already has its value in a register; a silent row's pre-gate value, which the join owes a
 *  partner that rotated it, is recovered from its swept slot as stored*(1/cos) (GateSinks.h ContractSink).
 *
 *  `state_mask` (Schrodinger fused picture only) switches on the c0 side channel in `sent_c0`.
 */
template <size_t NumModes, Algebra A, bool CaptureValues>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const Monomial<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const CutoffContext &cut_st,
                            const VecD &coeffs,
                            std::optional<size_t> only_rotate_len_k,
                            bool over_cutoff_possible,
                            size_t my_rank,
                            const routing::Router &router,
                            GateScratch<NumModes> &scratch,
                            double *fused_scale_coeffs = nullptr,
                            double fused_scale_cos = 1.0,
                            const Monomial<NumModes> *state_mask = nullptr) -> FusedScanResult<NumModes> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t gen_pop = gen.count();
    const size_t rank_count = router.flat_world();
    const auto ectx = A::make_gen_context(gen);

    // The value path is exactly the path that answers silent hits with a response (Engine.h,
    // ContractSink::wants_responses), and so the path whose send predicate is E(M) alone.
    constexpr bool kAnswerSilentHits = CaptureValues;

    FusedScanResult<NumModes> res;
    res.queries.assign(rank_count, VecZ{});
    res.sent.assign(rank_count, std::vector<SentRecord>{});
    res.self.keeps_values = CaptureValues;
    // Sized on the early-return paths below too, so the engine's per-slot access is always in bounds.
    if (state_mask != nullptr) {
        res.sent_c0.assign(rank_count, std::vector<double>{});
    }
    // No anticommuting words on an early return: nothing is sent, so the per-row marks stay untouched.
    scratch.nz.clear();

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
        // very array the reads come from and cover the full operator.
        assert(fused_scale_coeffs == nullptr || (fused_scale_coeffs == coeffs.data() && coeffs.size() >= n));
        // The value path answers a silent hit out of `coeffs`, so it already requires one coefficient per
        // pre-gate row. The hot arm below leans on that same contract to load a coefficient with no
        // per-row bound; the other arms run where the picture may carry none.
        assert((!CaptureValues || coeffs.size() >= n) && "the value path needs a coefficient per pre-gate row");

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

        const OperatorIndex<NumModes> &ham = *op.store;
        using RowPosT = typename OperatorIndex<NumModes>::PosT;
        using RowPositions = typename OperatorIndex<NumModes>::RowPositions;
        RowMarks &marks = scratch.marks;

        // The generator's positions, once per gate: the merge's second input.
        std::vector<uint16_t> &gen_pos = scratch.gen;
        gen_pos.clear();
        for (size_t b = gen.find_first(); b < gen.size(); b = gen.find_next(b)) {
            gen_pos.push_back(static_cast<uint16_t>(b));
        }
        // Bound once per gate so the per-record fold never re-enters routing's static-init guard.
        const uint64_t *const labels = routing::linear_basis<2 * NumModes>().data();
        // pbuf capacity is 2*NumModes: the partner's positions are distinct, so this always suffices.
        std::vector<RowPosT> &pbuf = scratch.partner;
        if (pbuf.size() < 2 * NumModes) {
            pbuf.resize(2 * NumModes);
        }
        // Every flag emit_row reads on every emitting row, by value rather than through the closure: a
        // noinline lambda reloads a captured reference per call, which is most of what its frame costs.
        const bool need_dense = (rank_count != 1 && !router.is_linear()) || !cutoff_eval.has_digest_form()
                                || !A::sign_from_positions_ok(ectx);
        const bool over_cutoff = over_cutoff_possible;
        const bool multi_rank = rank_count != 1;
        const Monomial<NumModes> *const mask = state_mask;
        double *const sweep_on_emit = fused_scale_coeffs;
        // And when the whole flat slot is linear, so is the destination: flat(M^G) = flat(M) ^
        // flat_shift(G), and flat(M) is this slot. Present exactly when nothing has to fold the partner's
        // fingerprint to route it (Routing.h is_flat_linear: splitmix, R == 1 and a non-power-of-two S
        // are the exceptions, and those keep the fold).
        const std::optional<size_t> gen_flat_shift = router.flat_shift<NumModes>(gen);
        const size_t partner_slot = gen_flat_shift.has_value() ? (my_rank ^ *gen_flat_shift) : my_rank;
        const bool shift_routes = gen_flat_shift.has_value();
#ifndef NDEBUG
        // Per gate, not per record: both identities are properties of the router and the key map, so one
        // dense partner witnesses them for the whole gate.
        bool routing_checked = false;
#endif

        // `r_prime` is the partner's owning slot, decided by the caller (routing::Router is the only owner
        // function; find_rank (MPIUtils.h) must stay in step with it or a term is placed and queried on
        // different ranks, which duplicates a row silently), and `tag_partner` the partner's join tag.
        auto push = [&](size_t r_prime,
                        uint32_t tag_partner,
                        std::span<const RowPosT> pos,
                        int phase,
                        bool rot,
                        size_t row,
                        double v_src,
                        double c0) {
            if (r_prime == my_rank) {
                res.self.push(pos, phase, tag_partner, rot, v_src);
            }
            else {
                VecZ &buf = res.queries[r_prime];
                QueryWire<NumModes>::push(buf, pos, phase, rot);
                if constexpr (CaptureValues) {
                    QueryWire<NumModes>::push_value(buf, v_src);
                }
            }
            res.sent[r_prime].push_back(SentRecord{static_cast<TermIndex>(row), static_cast<int8_t>(phase)});
            if (mask != nullptr) {
                res.sent_c0[r_prime].push_back(c0);
            }
        };

        // Row i as the scan reads it: its positions (empty span for a spilled row) and popcount.
        struct RowRead {
            RowPositions src;
            size_t pop;
        };
        // Read through the row store's chunk already resolved for i's 64-row window: every loop below
        // walks an inverted-index word, so the chunk lookup is hoisted out of the row loop once and for
        // all -- the emitting rows included.
        using RowBlock = typename OperatorIndex<NumModes>::RowBlock;
        auto read_row_in = [&](const RowBlock &block, size_t i) -> RowRead {
            const RowPositions src = ham.positions_at(OperatorIndex<NumModes>::block_row(block, i));
            if (src.inlined()) {
                return {src, src.pos.size()};
            }
            return {src, ham.popcount(i)};
        };
        /*! The emitting tail of one anticommuting row: the row read, the partner product, the structural
         *  cutoff and the record. Deliberately NOT inlined: it runs only behind the coefficient gate --
         *  about a fifth of Anti(G) on the value path -- so inlining it would put its frame, its spills
         *  and its closure reloads on the silent path too, which is the cost the protocol exists to avoid.
         *
         *  `block` is row i's 64-row window, resolved once by the caller, so the row read below costs no
         *  chunk lookup of its own. `pre` is the row when the caller has already read it (the orbital-gate
         *  loop reads it for the cosine push), nullptr otherwise. Returns whether the row emitted, i.e.
         *  whether `rot` was set: the caller owes a row that did not its share of the fused cos sweep.
         */
        auto emit_row = [&] [[gnu::noinline]] (const RowBlock &block,
                                               size_t i,
                                               const RowRead *pre,
                                               double v_src,
                                               double abs_c,
                                               bool row_rot) -> bool {
            const RowRead row = (pre != nullptr) ? *pre : read_row_in(block, i);
            auto p = emit_partner<NumModes, A>(ham,
                                               i,
                                               row.src,
                                               ectx,
                                               std::span<const uint16_t>(gen_pos),
                                               std::span<RowPosT>(pbuf),
                                               need_dense,
                                               /*need_sign=*/false);
            // Structural cutoff on the partner, unless upper_atol rescues it (CutoffContext::is_above_upper).
            const bool struct_pass = cutoff_eval.has_digest_form() ? cutoff_eval.passes_from_digest(p.k, p.paired)
                                                                   : cutoff_eval.passes_with_popcount(p.dense, p.k);
            if (!struct_pass && !over_cutoff) {
                return false; // the partner cannot be tracked and would not be minted: nobody needs to hear
            }
            const bool rot = row_rot && (struct_pass || cut_st.is_above_upper(abs_c));
            if (!rot && kAnswerSilentHits) {
                return false; // silent on the partner's cutoff instead: answered the same way
            }
            if (rot) {
                marks.set_rot(i);
                // The fused cos sweep for an emitting row: the scale lands now, while v_src is still in a
                // register. The record carries the pre-cos value, so nothing has to read this slot back.
                // The caller scales the silent rows in the same pass.
                if (sweep_on_emit != nullptr) {
                    sweep_on_emit[i] = v_src * fused_scale_cos;
                }
            }
            if (p.sign_pending) {
                p.phase_factor = A::rotation_sign_positions(ectx, row.src.pos.data(), row.src.pos.size());
            }
            const int phase = A::emit_phase(p.phase_factor, row.pop, gen_pop, p.overlap);
            double c0 = 0.0;
            // Only an E-record can mint, so only it needs the fresh partner's pre-gate coefficient.
            if (rot && mask != nullptr && digest_is_paired(p.k, p.paired)) {
                c0 = A::state_phase_positions(pbuf.data(), p.k, *mask);
            }
            // The record's identity: the partner's own fingerprint, folded off the positions the merge has
            // just written into pbuf -- one label XOR per position, and the row's key is not resident to
            // XOR instead. Its top half is the join tag the table is keyed on, and a receiver folds the
            // same number off the positions it decodes, so the two agree. k == 0 means the partner is the
            // identity, i.e. the source is G itself, and the empty fold is 0 -- no special case.
            const uint64_t fp_partner = routing::fingerprint_positions(labels, pbuf.data(), p.k);
            const uint32_t tag_partner = join_tag(fp_partner);
            // Where it goes. The XOR identity covers every flat-linear geometry; the two that are not (a
            // non-power-of-two S under linear routing, and the splitmix router) still need a number of the
            // partner's own -- the fingerprint just folded, or its dense form.
            size_t r_prime = partner_slot;
            if (multi_rank && !shift_routes) {
                if (router.is_linear()) {
                    r_prime = router.dest_from_fingerprint(fp_partner);
                }
                else {
                    assert(p.has_dense);
                    r_prime = router.dest<NumModes>(p.dense);
                }
            }
#ifndef NDEBUG
            if (p.has_dense && !routing_checked) {
                routing_checked = true;
                assert(r_prime == router.dest<NumModes>(p.dense)); // an identity
                assert(tag_partner == key_of<NumModes>(p.dense));  // and so is this
            }
#endif
            push(r_prime, tag_partner, std::span<const RowPosT>(pbuf).first(p.k), phase, rot, i, v_src, c0);
            return rot;
        };

        // Pass 1 and pass 2 stay fused over `nz`: splitting them regressed measurably, as `nz` spills L1
        // between them. `nz` is scratch-owned, both to reuse its capacity across gates and because the
        // join streams it again after the exchange (Engine.h).
        std::vector<EvenParityNzWord> &nz = scratch.nz;
        size_t n_anti = 0;
        if (skip_scan) {
            nz.clear(); // pass 1 clears it on entry; the skip must too (the vector is reused)
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
                                             n_anti);
        }
        // After pass 1, so the clear knows which words carry this gate's rows; before the emit pass, which
        // is what sets them.
        marks.begin(n, nz);
        // The self-slot buffers are sized to what this gate will EMIT, not to the fold's tally. Reserving
        // |Anti(G)| filled them to ~1.7 % of themselves in the lower_atol regime; dropping the reserve
        // outright is not the answer either, the geometric growth it left behind costing 1.4 % of
        // propagate. The previous gate's emitted count with a margin is a good enough estimate to leave
        // one allocation, is capped by the fold's own bound, and when it falls short push() doubles as it
        // always has.
        if (rank_count == 1) {
            const size_t hint = scratch.self_records_hint;
            const size_t want = std::min(n_anti, std::max(kSelfReserveFloor, hint + (hint / 4)));
            res.self.reserve(want, QueryWire<NumModes>::kReservePositionsPerQuery);
            res.sent[my_rank].reserve(want);
        }

        // Everything the per-row loops read on every anticommuting row, hoisted out of them: through a
        // closure these are reloaded per row, which is a third of the silent path's instructions.
        const CutoffContext cut = cut_st;
        const double *const coeff_ptr = coeffs.data();
        const size_t coeff_n = coeffs.size();
        // `use_coeff_checks` is fixed for the whole scan, so it rides the value as a mask rather than a
        // per-row branch: clearing the sign bit is std::abs exactly, and a zero mask is the +0.0 the
        // `false` arm produced. The branch cost three instructions on every anticommuting row.
        const uint64_t abs_mask = cut.use_coeff_checks ? ~(uint64_t{1} << 63U) : uint64_t{0};
        double *const sweep = fused_scale_coeffs;
        const double sweep_cos = fused_scale_cos;
        const bool word_aligned_cos = !only_rotate_len_k.has_value();
        // No orbital gate and no fused sweep: the whole word goes into the cosine set in one push.
        const bool build_cos_words = word_aligned_cos && sweep == nullptr;
        CosineWordBuilder cos_b;

        /*! One emit pass over the fold's words. `WordAlignedCos` (no orbital gate) decides whether the
         *  cosine set is pushed per word or per passing index, and whether the row has to be read before
         *  the coefficient gate; `CaptureValues` decides whether a silent row is skipped and swept or
         *  still sends its rot=0 record. The four combinations are four instantiations of one body rather
         *  than three hand-written loops.
         */
        auto emit_pass = [&]<bool WordAlignedCos>() {
            for (const auto &w : nz) {
                marks.set_foll_word(w.base / 64, w.foll);
                // One chunk resolution per 64 rows; on the hot arm a silent row is never read at all.
                const auto block = ham.row_block(w.base);
                if constexpr (WordAlignedCos) {
                    if (build_cos_words) {
                        cos_b.push_word(w.base, w.overlap);
                    }
                }
                for (uint64_t m = w.overlap; m != 0U; m &= m - 1) {
                    const size_t i = w.base + static_cast<size_t>(std::countr_zero(m));
                    // The hot shape reads the coefficient with no bound: the value path guarantees one per
                    // pre-gate row (asserted above). The other arms run where the picture may carry none.
                    const double c =
                        (WordAlignedCos && CaptureValues) ? coeff_ptr[i] : ((i < coeff_n) ? coeff_ptr[i] : 0.0);
                    const double abs_c = std::bit_cast<double>(std::bit_cast<uint64_t>(c) & abs_mask);
                    const double v_src = CaptureValues ? c : 0.0;
                    if constexpr (WordAlignedCos) {
                        if constexpr (CaptureValues) {
                            // E(M) factorises: this half reads the coefficient alone, the other needs the
                            // partner's digest. A record is silent iff this half already fails, so the
                            // whole partner product -- the row read, the merge, the cutoff, the encoding --
                            // is skipped for it. Exactness survives because a hit on this row is answered
                            // with its value.
                            if (rotation_coeff_gate(cut, abs_c)
                                && emit_row(block, i, nullptr, v_src, abs_c, /*row_rot=*/true)) {
                                continue;
                            }
                            if (sweep != nullptr) {
                                sweep[i] = c * sweep_cos;
                            }
                        }
                        else {
                            // Graph mode: nothing is dropped for being silent -- a below-threshold term
                            // still sends its rot=0 record, since its partner may rotate it.
                            emit_row(block, i, nullptr, v_src, abs_c, rotation_coeff_gate(cut, abs_c));
                        }
                    }
                    else {
                        // Orbital gate active: the per-index cosine push covers only orbital-passing terms,
                        // and the row it needs is the row the emit reads. A capped term still sends (with
                        // rot=0) on the graph path, since its partner may rotate it.
                        const RowRead row = read_row_in(block, i);
                        if (row.pop <= static_cast<size_t>(*only_rotate_len_k)) {
                            cos_b.push_index(i);
                        }
                        const bool row_rot =
                            rotation_coeff_gate(cut, abs_c) && rotation_pop_gate(only_rotate_len_k, row.pop);
                        if (row_rot || !CaptureValues) {
                            emit_row(block, i, &row, v_src, abs_c, row_rot);
                        }
                    }
                }
            }
        };
        if (word_aligned_cos) {
            emit_pass.template operator()<true>();
        }
        else {
            // The fused sweep is off with an orbital gate by construction.
            assert(sweep == nullptr && "the fused sweep does not run with an orbital gate");
            emit_pass.template operator()<false>();
        }
        // After the emit pass, so it is the count this gate actually staged.
        scratch.self_records_hint = res.self.size();
        res.cos_blocks.push_back(cos_b.finish());
    }
    return res;
}

} // namespace monoprop::detail
