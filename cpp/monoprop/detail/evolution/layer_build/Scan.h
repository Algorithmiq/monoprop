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
        // Block-scoped: a dense column is stored in chunks, so its words are contiguous only within one
        // fold block (InvertedIndex::dense_column_block).
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
// static part (the structural cutoff on M'=M⊕G, applied in emit). The dynamic part is two independent
// tests, kept separate because only one of them needs the row: the value path reads the coefficient half
// first and, when it fails, never reads the row at all (Scan.h's visit).
inline auto rotation_coeff_gate(const CutoffContext &ctx, double abs_c) -> bool {
    return !ctx.is_below_sin(abs_c);
}
inline auto rotation_pop_gate(std::optional<size_t> only_rotate_len_k, size_t mono_pop) -> bool {
    return !only_rotate_len_k.has_value() || mono_pop <= static_cast<size_t>(*only_rotate_len_k);
}
inline auto rotation_dynamic_gate(std::optional<size_t> only_rotate_len_k,
                                  size_t mono_pop,
                                  const CutoffContext &ctx,
                                  double abs_c) -> bool {
    return rotation_pop_gate(only_rotate_len_k, mono_pop) && rotation_coeff_gate(ctx, abs_c);
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
    bool sign_pending = false; // phase_factor not filled: the caller asked for the digest only
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
//
// `need_sign` false returns the digest with `sign_pending` set and no sign computed, for a caller whose
// send predicate reads the digest first and may drop the record: the sign is then taken from the same
// positions once the record is known to go out. The paths that build the dense partner compute the sign
// on the way there, so they honour the request only in the common inline case.
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
    // The per-slot arrays below are indexed by DESTINATION SLOT through WindowVec::at_slot, and cover only
    // the slots this generator can reach: S of the P=R*S world under linear routing, all P under
    // splitmix. See mpi::PeerPlan::window.
    mpi::SlotWindow window;
    // The wire records (QueryWire.h) per destination slot, in stream order = ascending source row. Fused
    // (value word after each record) iff capture_values.
    mpi::WindowVec<VecZ> queries;
    // Per destination slot, the SOURCE row and emit phase of each record, parallel to the records of
    // that slot (the self slot's parallel to `self`). Ascending in row, because the scan walks rows
    // ascending: the absence pass and the graph sink's out lists rely on that order.
    mpi::WindowVec<std::vector<SentRecord>> sent;
    // Parallel to `sent`: c0(μ), the pre-gate coefficient the partner would be minted with (Schrödinger:
    // state score of a fully paired μ, else 0). Only filled when a state mask is given, i.e. for the
    // fused Schrödinger picture; read by the absence pass alone.
    mpi::WindowVec<std::vector<double>> sent_c0;
    // Records addressed to this slot itself, staged as positions instead of encoded into the window's
    // self slot, and joined inline. Empty unless the window contains my_rank, i.e. unless this
    // generator's rank shift is zero.
    SelfQueryStage<NumModes> self;
};

// Classify, cut off and emit in one pass over the anticommuting terms. Every anticommuting row records
// its pivot bit (`foll`) in scratch.marks and then decides two things about its partner μ = M⊕G
// (Engine.h has the protocol):
//   E(M)    = rotation_dynamic_gate ∧ (struct_pass(μ) ∨ is_above_upper(|c|))   its `rot` bit
//   send(M) = capture_values ? E(M) : struct_pass(μ) ∨ over_cutoff_possible
// The value path sends records for emitting terms ONLY: a silent term's contribution reaches its
// partner as a round-2 response instead, so it needs neither the row, the merge nor the wire, and the
// coefficient half of E is therefore tested first. Graph mode has no coefficients and so no silent
// terms to answer, and keeps the symmetric predicate its positional in/out pairing is proved on.
// Records go to the owner of μ (routing::Router; self at R==1) in ascending source-index order, so the
// join and the miss-index assignment are deterministic. The rows themselves are NOT indexed here: the
// join is streamed over `scratch.nz` against a table of the records (BucketJoin.h), so the only per-term
// row work the scan does is the partner merge every record needs anyway.
//
// `fused_scale_coeffs` (no length cap only; must alias coeffs.data()) scales the anticommuting coeffs in
// place by `fused_scale_cos`=cos(2·build_angle), so no cosine set is built; the value a record carries
// is the PRE-cos coefficient, read before the store. Only rows that set `rot` are scaled here: a silent
// row must still hold its pre-gate coefficient when the join answers a hit on it, so its cos factor is
// folded in afterwards by scale_silent_anti_coeffs. `state_mask` (Schrödinger fused picture only)
// switches on the c0(μ) side channel in `sent_c0`. `window` must be the peer plan's window for
// `my_rank`: it is what the per-slot arrays are sized to.
template <size_t NumModes, Algebra A>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const Monomial<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const CutoffContext &cut_st,
                            const VecD &coeffs,
                            std::optional<size_t> only_rotate_len_k,
                            bool over_cutoff_possible,
                            mpi::SlotWindow window,
                            size_t my_rank,
                            const routing::Router &router,
                            GateScratch<NumModes> &scratch,
                            bool capture_values = false,
                            double *fused_scale_coeffs = nullptr,
                            double fused_scale_cos = 1.0,
                            const Monomial<NumModes> *state_mask = nullptr) -> FusedScanResult<NumModes> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t gen_pop = gen.count();
    const size_t rank_count = router.flat_world();
    const auto ectx = A::make_gen_context(gen);
    assert(window.stop() <= rank_count && window.count != 0);

    // The value path is exactly the path that answers silent hits with a response (Engine.h,
    // ContractSink::wants_responses), and so the path whose send predicate is E(M) alone.
    const bool answer_silent_hits = capture_values;

    FusedScanResult<NumModes> res;
    res.window = window;
    res.queries.reset(window);
    res.sent.reset(window);
    // Sized on the early-return paths below too, so the engine's per-slot access is always in bounds.
    if (state_mask != nullptr) {
        res.sent_c0.reset(window);
    }
    // No anticommuting words on an early return: the join then has no row to match a record against,
    // and the per-row marks stay untouched because nothing is sent either.
    scratch.nz.clear();
    scratch.join.clear_rows();

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
        // What still reads the dense partner: the splitmix router (its hash), an opaque cutoff, or a sign
        // kernel with no position form. Under linear routing with a structural cutoff: nothing.
        const bool need_dense = (rank_count != 1 && !router.is_linear()) || !cutoff_eval.has_digest_form()
                                || !A::sign_from_positions_ok(ectx);

        auto push = [&](uint64_t fp_partner,
                        const PartnerProduct<NumModes> &p,
                        std::span<const RowPosT> pos,
                        int phase,
                        bool rot,
                        size_t row,
                        double v_src,
                        double c0) {
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
                res.self.push(pos, phase, fp_partner, rot, v_src);
            }
            else {
                VecZ &buf = res.queries.at_slot(r_prime);
                QueryWire<NumModes>::push(buf, pos, phase, rot);
                if (capture_values) {
                    QueryWire<NumModes>::push_value(buf, v_src);
                }
            }
            res.sent.at_slot(r_prime).push_back(SentRecord{static_cast<TermIndex>(row), static_cast<int8_t>(phase)});
            if (state_mask != nullptr) {
                res.sent_c0.at_slot(r_prime).push_back(c0);
            }
        };

        // Row i as the scan reads it: its positions (empty span for a spilled row) and popcount.
        struct RowRead {
            RowPositions src;
            size_t pop;
        };
        auto read_row = [&](size_t i) -> RowRead {
            const RowPositions src = ham.row_positions(i);
            if (src.inlined()) {
                return {src, src.pos.size()};
            }
            return {src, ham.popcount(i)};
        };
        // One anticommuting row: its pivot bit, its key for the join, then the partner product and the
        // send predicate. `fetch_coeff` yields (v_src, |c|) and is called up front, because on the value
        // path the coefficient alone decides whether this row has to be read at all. `pre` is the row
        // when the caller has already read it (the orbital-gate loop reads it for the cosine push),
        // nullptr otherwise.
        auto visit = [&](size_t i, bool is_follower, const RowRead *pre, auto &&fetch_coeff) {
            if (is_follower) {
                marks.set_foll(i);
            }
            // Every anticommuting row is a possible hit target for an incoming record, whether or not it
            // sends one itself, so all of them are staged for the join -- off the key the store keeps per
            // row, which is why staging costs no row read.
            scratch.join.add_row_key(ham.key(i), i);
            std::optional<RowRead> lazy;
            auto row_of = [&]() -> const RowRead & {
                if (pre != nullptr) {
                    return *pre;
                }
                if (!lazy.has_value()) {
                    lazy = read_row(i);
                }
                return *lazy;
            };
            const auto [v_src, abs_c] = fetch_coeff();
            // E(ν) factorises: this half reads the row alone (its length and its coefficient), the other
            // needs the partner's digest. A record is silent iff this half already fails, so on the value
            // path the whole partner product -- the row read, the merge, the cutoff, the encoding -- is
            // skipped for it, and the sine cutoff is tested first because it is the half that needs no
            // popcount. Exactness survives because a hit on this row is answered with its coefficient.
            bool row_rot = rotation_coeff_gate(cut_st, abs_c);
            if (row_rot && only_rotate_len_k.has_value()) {
                row_rot = rotation_pop_gate(only_rotate_len_k, row_of().pop);
            }
            if (!row_rot && answer_silent_hits) {
                return;
            }
            const RowRead &row = row_of();
            auto p = emit_partner<NumModes, A>(ham,
                                               i,
                                               row.src,
                                               ectx,
                                               std::span<const uint16_t>(gen_pos),
                                               std::span<RowPosT>(pbuf),
                                               need_dense,
                                               /*need_sign=*/false);
            // Structural cutoff on the partner M⊕G, unless upper_atol rescues it (CutoffContext::is_above_upper).
            const bool struct_pass = cutoff_eval.has_digest_form() ? cutoff_eval.passes_from_digest(p.k, p.paired)
                                                                   : cutoff_eval.passes_with_popcount(p.dense, p.k);
            if (!struct_pass && !over_cutoff_possible) {
                return; // μ cannot be tracked and would not be minted: nobody needs to hear from M
            }
            const bool rot = row_rot && (struct_pass || cut_st.is_above_upper(abs_c));
            if (!rot && answer_silent_hits) {
                return; // silent on the partner's cutoff instead: answered the same way
            }
            if (rot) {
                marks.set_rot(i);
                // The fused cos sweep, for the rows it applies to: a row that sets `rot` sends a record,
                // so nothing will ask it for its pre-gate coefficient and the scale can land now, while
                // v_src is still in a register. Silent rows are swept by scale_silent_anti_coeffs.
                if (fused_scale_coeffs != nullptr) {
                    fused_scale_coeffs[i] = v_src * fused_scale_cos;
                }
            }
            if (p.sign_pending) {
                p.phase_factor = A::rotation_sign_positions(ectx, row.src.pos.data(), row.src.pos.size());
            }
            const int phase = A::emit_phase(p.phase_factor, row.pop, gen_pop, p.overlap);
            double c0 = 0.0;
            // Only an E-record can mint, so only it needs the fresh partner's pre-gate coefficient.
            if (rot && state_mask != nullptr && digest_is_paired(p.k, p.paired)) {
                c0 = A::state_phase_positions(pbuf.data(), p.k, *state_mask);
            }
            // The record's key. The row's stored key is mixed (not GF(2)-linear), so fp(μ) cannot come
            // from it by XOR; folding μ's own positions is the same map a receiver applies to the
            // positions it decodes, and it runs only for a record that is actually going out.
            const uint64_t fp_partner = p.k == 0 ? 0 : routing::fingerprint_positions(labels, pbuf.data(), p.k);
            push(fp_partner, p, std::span<const RowPosT>(pbuf).first(p.k), phase, rot, i, v_src, c0);
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
        // After pass 1, so the clear knows which words carry this gate's rows and the join's row side is
        // sized to the tally; before the emit pass, which is what fills both.
        marks.begin(n, nz);
        scratch.join.begin_rows(n_anti);
        if (rank_count == 1) {
            // A hint only; wider terms grow the buffer as needed.
            res.self.reserve(n_anti, QueryWire<NumModes>::kReservePositionsPerQuery);
            res.sent.at_slot(my_rank).reserve(n_anti);
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
            if (word_aligned_cos) {
                // No orbital gate: the whole word goes into the cosine set, unless the fused sweep is
                // scaling in place instead (visit does that, per row, once `rot` is known).
                if (fused_scale_coeffs == nullptr) {
                    cos_b.push_word(w.base, w.overlap);
                }
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    visit(i, ((w.foll >> tz) & 1U) != 0, nullptr, [&] { return derive_coeff(i); });
                }
            }
            else {
                // Orbital gate active: the per-index cosine push covers only orbital-passing terms. A
                // capped term still sends (with rot=0), since its partner may rotate it.
                for (uint64_t m = w.overlap; m; m &= m - 1) {
                    const size_t tz = static_cast<size_t>(std::countr_zero(m));
                    const size_t i = w.base + tz;
                    const RowRead row = read_row(i);
                    if (row.pop <= static_cast<size_t>(*only_rotate_len_k)) {
                        cos_b.push_index(i);
                    }
                    visit(i, ((w.foll >> tz) & 1U) != 0, &row, [&] { return derive_coeff(i); });
                }
            }
        }
        res.cos_blocks.push_back(cos_b.finish());
    }
    return res;
}

// The other half of the fused cos sweep: the gate's cosine scale for the anticommuting rows the scan
// left alone, i.e. exactly the silent ones (`!marks.rot`). Those rows must still carry their pre-gate
// coefficient while the join runs, because that value is what a response sends back to a partner that
// did rotate (Engine.h round 2) -- so this runs AFTER the join and before the halves are applied. The
// multiplication is the same `v * cos` the scan would have done, and the two row sets are disjoint and
// together exactly Anti(G), so the result is independent of which pass touched a row.
//
// Only for the fused sweep: the two-pass paths scale from a CosMask instead, which already covers the
// silent rows. `coeffs` must be the swept array, `nz` the fold's words and `marks` the scan's own.
inline auto scale_silent_anti_coeffs(std::span<const EvenParityNzWord> nz,
                                     const RowMarks &marks,
                                     double *coeffs,
                                     double cos_val) -> void {
    for (const auto &w : nz) {
        for (uint64_t m = w.overlap & ~marks.rot_word(w.base / 64); m; m &= m - 1) {
            coeffs[w.base + static_cast<size_t>(std::countr_zero(m))] *= cos_val;
        }
    }
}

} // namespace monoprop::detail
