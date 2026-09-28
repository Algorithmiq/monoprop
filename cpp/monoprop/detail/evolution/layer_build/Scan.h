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

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
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
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

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
    std::vector<CosMask> cos_blocks; // ascending, disjoint, chunk order
    // The six arrays below cover only the slots this generator can reach (mpi::PeerPlan::window) and
    // are addressed by destination slot through WindowVec::at_slot.
    mpi::WindowVec<VecZ> leader_queries;              // serialized leader queries per owner slot
    mpi::WindowVec<std::vector<size_t>> leader_src;   // parallel to leader_queries (source op idx)
    mpi::WindowVec<VecZ> follower_queries;            // serialized follower queries per owner slot
    mpi::WindowVec<std::vector<size_t>> follower_src; // parallel to follower_queries
    // Fused-contraction only (capture_values): signed pre-cos source coeff (v_src) parallel to
    // leader_src / follower_src. Empty when capture_values is false.
    mpi::WindowVec<std::vector<double>> leader_val;
    mpi::WindowVec<std::vector<double>> follower_val;
    // Self-owned queries, staged as positions and resolved inline. Order must match the self slot's
    // leader_src / follower_src. Empty unless this generator's rank shift is zero.
    SelfQueryStage<NumModes> leader_self;
    SelfQueryStage<NumModes> follower_self;
};

/*!
 * \brief Number of logical word ranges a threaded scan of `words` inverted-index words uses.
 *
 * `min(options.threads, ceil(words / kColumnBlockWords))`: a function of the word count and the budget
 * only, never of the team the runtime actually provides, so range IDs and the merged order are fixed.
 *
 * \throws std::invalid_argument if `options.threads` is not positive.
 */
inline auto scan_ranges(size_t words, parallel::Options options) -> size_t {
    if (options.threads < 1) {
        throw std::invalid_argument(
            std::format("fused_find_and_collect: the thread budget must be positive, got {}", options.threads));
    }
    return std::min(static_cast<size_t>(options.threads), logical_ranges(words, kColumnBlockWords));
}

/*!
 * \brief Words `[first, second)` of logical scan range `range` out of `ranges` over `words` words.
 *
 * Ranges are contiguous runs of whole kColumnBlockWords fold blocks, ascending with the range ID and
 * covering every word; each is nonempty when `ranges` does not exceed the block count (scan_ranges).
 */
inline auto scan_range_words(size_t range, size_t ranges, size_t words) -> std::pair<size_t, size_t> {
    assert(range < ranges && ranges <= logical_ranges(words, kColumnBlockWords));
    // Rows are uint32 TermIndex values, so words <= 2^26, blocks <= 2^16 and these products stay below 2^32.
    const size_t blocks = logical_ranges(words, kColumnBlockWords);
    const size_t lo = (range * blocks / ranges) * kColumnBlockWords;
    const size_t hi = std::min(((range + 1) * blocks / ranges) * kColumnBlockWords, words);
    return {lo, hi};
}

/*!
 * \brief Scan inverted-index words `[wlo, whi)` of `op` for terms anticommuting with `gen`, and emit their
 * partner queries and cosine words into a fresh result.
 *
 * This is the whole serial scan restricted to one word range; fused_find_and_collect runs it once over
 * every word, or once per logical range on scan workers and merges the pieces in range order. It reads only
 * frozen storage and never calls a lazy accessor: `index` must be `op.inverted_index()` and `row_parity`
 * its `row_parity_words()` (null unless the fold needs the odd-|G| correction), both prepared on the caller.
 * The tail mask comes from the index's global last word, so only the range holding that word masks it. All
 * scratch is private to the invocation or thread_local on the executing thread. With `fused_scale_coeffs`
 * it writes the coefficients of anticommuting sources in its own words only, each once, after reading the
 * pre-scale value its cutoff decisions and captured value use. It resolves no target, grows no store and
 * calls no MPI.
 *
 * The remaining parameters are fused_find_and_collect's; see there.
 */
template <size_t NumModes, Algebra A>
auto fused_find_and_collect_range(const MPOperator<NumModes> &op,
                                  const Monomial<NumModes> &gen,
                                  const CutoffEvaluator<NumModes> &cutoff_eval,
                                  const CutoffContext &cut_st,
                                  const VecD &coeffs,
                                  std::optional<size_t> only_rotate_len_k,
                                  mpi::SlotWindow window,
                                  size_t my_rank,
                                  const routing::Router &router,
                                  size_t gen_shift,
                                  const InvertedIndex<NumModes> &index,
                                  const uint64_t *row_parity,
                                  size_t wlo,
                                  size_t whi,
                                  bool capture_values,
                                  double *fused_scale_coeffs,
                                  double fused_scale_cos) -> FusedScanResult<NumModes> {
    const size_t gen_pop = gen.count();
    const size_t rank_count = router.flat_world();
    const auto ectx = A::make_gen_context(gen);

    FusedScanResult<NumModes> res;
    res.leader_queries.reset(window);
    res.leader_src.reset(window);
    res.follower_queries.reset(window);
    res.follower_src.reset(window);
    if (capture_values) {
        res.leader_val.reset(window);
        res.follower_val.reset(window);
    }

    // The anticommutation fold runs over G's inverted-index columns (Majorana: G; Pauli: J(G) =
    // pair_swap(G), so pauli_anticommutes = parity(|M ∩ J(G)|), and Pauli never needs the odd-|G|
    // correction since parity(|G ∩ J(G)|)=0). The pivot splitting each pair is a set bit of the real G
    // (gen.find_first()), not J(G) — A and A⊕G differ exactly on G's bits.
    const Monomial<NumModes> fold_gen = A::fold_generator(gen);
    const bool g_odd = A::fold_needs_odd_correction(gen);
    const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);
    const std::span<const size_t> gen_cols(gen_columns.indices.data(), gen_columns.count);
    assert(index.rows() == op.store->size() && (!g_odd || row_parity != nullptr));
    const size_t n = op.store->size();
    const size_t last_word = index.words() - 1;
    const uint64_t last_word_mask = (n % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (n % 64)) - 1);

    auto &lq = res.leader_queries;
    auto &ls = res.leader_src;
    auto &lv = res.leader_val;
    auto &fq = res.follower_queries;
    auto &fs = res.follower_src;
    auto &fv = res.follower_val;

    const OperatorIndex<NumModes> &ham = *op.store;
    using RowPosT = typename OperatorIndex<NumModes>::PosT;

    // The generator's positions, once per range: the merge's second input.
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
        // Must agree with find_rank (both go through routing::Router), or a row duplicates silently.
        size_t r_prime = my_rank;
        if (rank_count != 1) {
            r_prime = router.dest_from_shift<NumModes>(dense, my_rank, gen_shift);
            assert(r_prime == router.dest<NumModes>(dense)); // an identity, not an approximation
        }
        if (r_prime == my_rank) {
            (is_follower ? res.follower_self : res.leader_self).push(pos, phase);
        }
        else {
            QueryWire<NumModes>::push(is_follower ? fq.at_slot(r_prime) : lq.at_slot(r_prime), pos, phase);
        }
        (is_follower ? fs : ls).at_slot(r_prime).push_back(i);
        if (capture_values) {
            (is_follower ? fv : lv).at_slot(r_prime).push_back(v_src);
        }
    };

    // The dynamic gate runs before emit_term_products, so a gate-rejected term computes no products.
    // abs_c/v_src come from the caller's coeff read, not re-read.
    auto emit = [&](size_t mono_pop, size_t i, double abs_c, double v_src, bool is_follower) {
        if (!rotation_dynamic_gate(only_rotate_len_k, mono_pop, cut_st, abs_c)) {
            return;
        }
        const auto p =
            emit_term_products<NumModes, A>(ham, i, ectx, std::span<const uint16_t>(gen_pos), std::span<RowPosT>(pbuf));
        // Structural cutoff on the partner M⊕G, unless upper_atol rescues it (CutoffContext::is_above_upper).
        const bool struct_pass = cutoff_eval.passes_with_popcount(p.new_mono, p.k);
        if (!struct_pass && !cut_st.is_above_upper(abs_c)) {
            return;
        }
        const int phase = A::emit_phase(p.phase_factor, mono_pop, gen_pop, p.overlap);
        push(p.new_mono, std::span<const RowPosT>(pbuf).first(p.k), phase, i, v_src, is_follower);
    };

    // Pass 1 and pass 2 stay fused over `nz`: splitting them regressed measurably, as `nz` spills L1
    // between them. `nz` is thread_local, acquired here on the executing thread, so a scan worker reuses its
    // own capacity across gates and never shares the caller's.
    thread_local std::vector<EvenParityNzWord> nz;
    size_t n_anti = 0;
    size_t n_foll = 0;
    even_parity_scan_pass1<NumModes>(index,
                                     gen_cols,
                                     gen.find_first(),
                                     wlo,
                                     whi,
                                     last_word,
                                     last_word_mask,
                                     g_odd,
                                     row_parity,
                                     nz,
                                     n_anti,
                                     n_foll);
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
    return res;
}

// `total + add`, refusing a sum a merged scan buffer could not hold.
inline auto checked_scan_total(size_t total, size_t add, size_t max_size, const char *what) -> size_t {
    if (add > max_size || total > max_size - add) {
        throw std::length_error(std::format("fused_find_and_collect: the merged {} exceed {}", what, max_size));
    }
    return total + add;
}

// Concatenates one stream across the pieces in range order into a single buffer and releases every piece's
// buffer. Moves a piece's buffer instead of copying it when it holds the whole stream or already has room
// for the total; otherwise reserves the exact total once.
template <typename Vec, typename Pieces, typename Get>
auto merge_scan_stream(Pieces &pieces, Get &&get, const char *what) -> Vec {
    constexpr size_t npos = std::numeric_limits<size_t>::max();
    size_t total = 0;
    size_t first = npos;
    size_t nonempty = 0;
    for (size_t k = 0; k < pieces.size(); ++k) {
        const Vec &v = get(pieces[k]);
        total = checked_scan_total(total, v.size(), v.max_size(), what);
        if (!v.empty()) {
            first = (first == npos) ? k : first;
            ++nonempty;
        }
    }
    Vec out;
    if (first != npos) {
        Vec &lead = get(pieces[first]);
        size_t next = first;
        if (nonempty == 1 || lead.capacity() >= total) {
            out = std::move(lead);
            ++next;
        }
        else {
            out.reserve(total);
        }
        for (size_t k = next; k < pieces.size(); ++k) {
            const Vec &v = get(pieces[k]);
            out.insert(out.end(), v.begin(), v.end());
            Vec{}.swap(get(pieces[k])); // release now: clear() would keep the capacity
        }
    }
    // Empty pieces may still hold a reservation.
    for (auto &piece : pieces) {
        Vec{}.swap(get(piece));
    }
    return out;
}

// Appends each piece's self-stage queries in range order through SelfQueryStage::push, which rebases the
// position offsets; only the logical queries/positions are read, never a capacity tail.
template <size_t NumModes>
auto merge_self_stage(std::vector<FusedScanResult<NumModes>> &pieces,
                      SelfQueryStage<NumModes> FusedScanResult<NumModes>::*member) -> SelfQueryStage<NumModes> {
    using Stage = SelfQueryStage<NumModes>;
    using PosT = typename Stage::PosT;
    constexpr size_t npos = std::numeric_limits<size_t>::max();
    size_t queries = 0;
    size_t positions = 0;
    size_t first = npos;
    size_t nonempty = 0;
    for (size_t k = 0; k < pieces.size(); ++k) {
        const Stage &st = pieces[k].*member;
        queries = checked_scan_total(queries, st.size(), st.pos_off.max_size(), "self queries");
        positions = checked_scan_total(positions, st.positions(), st.pos_flat.max_size(), "self positions");
        if (st.size() != 0) {
            first = (first == npos) ? k : first;
            ++nonempty;
        }
    }
    Stage out;
    if (first != npos) {
        Stage &lead = pieces[first].*member;
        size_t next = first;
        if (nonempty == 1 || (lead.pos_off.size() >= queries && lead.pos_flat.size() >= positions)) {
            out = std::move(lead);
            ++next;
        }
        else {
            out.reserve(queries, 0);
            // pos_flat is sized to capacity, not filled (see SelfQueryStage), so this is the exact reservation.
            out.pos_flat.resize(positions);
        }
        for (size_t k = next; k < pieces.size(); ++k) {
            const Stage &st = pieces[k].*member;
            for (size_t q = 0; q < st.size(); ++q) {
                out.push(std::span<const PosT>(st.pos_flat.data() + st.pos_off[q], st.k_of[q]), st.phase_of[q]);
            }
            pieces[k].*member = Stage{};
        }
        assert(out.size() == queries && out.positions() == positions);
    }
    for (auto &piece : pieces) {
        piece.*member = Stage{};
    }
    return out;
}

/*!
 * \brief Merge the per-range results of one threaded scan into the result the serial scan produces.
 *
 * `pieces[r]` must be range `r`'s result over `window`, ranges ascending in source words. Each stream is
 * concatenated in range order, so every destination's leader and follower queries, sources and values keep
 * their source-order subsequence; wire records are whole words and are copied, never re-encoded. The two
 * self stages merge separately through SelfQueryStage::push. The cosine sets are moved in range order and
 * stay ascending and disjoint. Value streams exist only when `capture_values`. Every piece's buffers are
 * released as soon as they are drained.
 *
 * \throws std::length_error if a merged stream would exceed its container's max_size.
 */
template <size_t NumModes>
auto merge_scan_ranges(std::vector<FusedScanResult<NumModes>> &pieces,
                       mpi::SlotWindow window,
                       [[maybe_unused]] size_t my_rank,
                       bool capture_values) -> FusedScanResult<NumModes> {
    using Result = FusedScanResult<NumModes>;
    Result out;
    const auto merge_window = [&]<typename T>(mpi::WindowVec<T> Result::*member, const char *what) {
        mpi::WindowVec<T> &merged = out.*member;
        merged.reset(window);
        for (const auto wi : window.indices()) {
            merged[wi] = merge_scan_stream<T>(pieces, [&](Result &p) -> T & { return (p.*member)[wi]; }, what);
        }
    };
    merge_window(&Result::leader_queries, "leader queries");
    merge_window(&Result::leader_src, "leader sources");
    merge_window(&Result::follower_queries, "follower queries");
    merge_window(&Result::follower_src, "follower sources");
    if (capture_values) {
        merge_window(&Result::leader_val, "leader values");
        merge_window(&Result::follower_val, "follower values");
    }
    out.leader_self = merge_self_stage<NumModes>(pieces, &Result::leader_self);
    out.follower_self = merge_self_stage<NumModes>(pieces, &Result::follower_self);
    size_t masks = 0;
    for (const auto &piece : pieces) {
        masks = checked_scan_total(masks, piece.cos_blocks.size(), out.cos_blocks.max_size(), "cosine sets");
    }
    out.cos_blocks.reserve(masks);
    for (auto &piece : pieces) {
        for (auto &mask : piece.cos_blocks) {
            out.cos_blocks.push_back(std::move(mask));
        }
        std::vector<CosMask>{}.swap(piece.cos_blocks);
    }
#ifndef NDEBUG
    if (window.contains(my_rank)) {
        assert(out.leader_queries.at_slot(my_rank).empty() && out.follower_queries.at_slot(my_rank).empty());
        assert(out.leader_self.size() == out.leader_src.at_slot(my_rank).size());
        assert(out.follower_self.size() == out.follower_src.at_slot(my_rank).size());
        assert(!capture_values || out.leader_self.size() == out.leader_val.at_slot(my_rank).size());
        assert(!capture_values || out.follower_self.size() == out.follower_val.at_slot(my_rank).size());
    }
    else {
        assert(out.leader_self.size() == 0 && out.follower_self.size() == 0);
    }
#endif
    return out;
}

// Classify, cut off and emit in one pass over the anticommuting terms. Queries go to the owner of
// M'=M⊕G (routing::Router; self at R==1) in ascending source-index order, so resolve and index assignment are
// deterministic. `fused_scale_coeffs` (no length cap only; must alias coeffs.data()) scales every anticommuting
// coeff in place by `fused_scale_cos`=cos(2·build_angle), so no cosine set is built and a hit's stored
// value is post-cos (resolve recovers it via 1/cos).
//
// `gen_shift` must be router.rank_shift(gen) and `op` hold only terms `my_rank` owns, so the owner of
// M⊕G is rank(M) ^ gen_shift (asserted against dest()). `window` is the plan's window for `my_rank`.
//
// Threading: with a parallel-safe cutoff (CutoffEvaluator::parallel_safe) and at least two logical ranges
// (scan_ranges), the words split into ranges that scan workers traverse with fused_find_and_collect_range,
// each into a private result, and merge_scan_ranges joins them in range order: the output is identical to
// the serial scan's at every budget and team size. Otherwise the serial scan runs on the caller. Lazy
// index/parity data is prepared here, on the caller, before any worker starts. `observer` is a test seam
// (see NoRangeObserver) reporting the KernelRange::scan ranges.
template <size_t NumModes, Algebra A, class Observer = NoRangeObserver>
auto fused_find_and_collect(const MPOperator<NumModes> &op,
                            const Monomial<NumModes> &gen,
                            const CutoffEvaluator<NumModes> &cutoff_eval,
                            const CutoffContext &cut_st,
                            const VecD &coeffs,
                            std::optional<size_t> only_rotate_len_k,
                            mpi::SlotWindow window,
                            size_t my_rank,
                            const routing::Router &router,
                            size_t gen_shift,
                            bool capture_values = false,
                            double *fused_scale_coeffs = nullptr,
                            double fused_scale_cos = 1.0,
                            parallel::Options options = {},
                            const Observer &observer = {}) -> FusedScanResult<NumModes> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    (void)scan_ranges(0, options); // validates the budget before anything runs
    assert(window.stop() <= router.flat_world() && window.count != 0);

    // Sized even on the early returns below, so the fused engine's src_val_r access stays in bounds.
    const auto empty_result = [&] {
        FusedScanResult<NumModes> res;
        res.leader_queries.reset(window);
        res.leader_src.reset(window);
        res.follower_queries.reset(window);
        res.follower_src.reset(window);
        if (capture_values) {
            res.leader_val.reset(window);
            res.follower_val.reset(window);
        }
        return res;
    };

    const Monomial<NumModes> fold_gen = A::fold_generator(gen);
    const bool g_odd = A::fold_needs_odd_correction(gen);
    const auto gen_columns = build_even_parity_generator_columns<NumModes>(fold_gen);
    if (gen_columns.count == 0) {
        observer.prepare(KernelRange::scan, 0);
        return empty_result();
    }
    // Both lazy accessors may build caches, so they run here and never in a worker.
    const auto &index = op.inverted_index();
    const size_t word_count = index.words();
    if (word_count == 0) {
        observer.prepare(KernelRange::scan, 0);
        return empty_result();
    }
    const uint64_t *const row_parity = g_odd ? index.row_parity_words() : nullptr;
    // The fused sweep writes fused_scale_coeffs[i] for every anticommuting i < n, so it must be the
    // very array the reads come from and cover the full operator — a violation corrupts 1/cos recovery.
    assert(fused_scale_coeffs == nullptr || (fused_scale_coeffs == coeffs.data() && coeffs.size() >= op.store->size()));

    // The scan can be skipped only when every fold column is empty; a dense column always holds ≥1
    // posting, so any dense column makes it non-empty.
    bool fold_cols_empty = true;
    for (size_t ci = 0; ci < gen_columns.count; ++ci) {
        const size_t c = gen_columns.indices[ci];
        if (index.column_is_dense(c) || !index.sparse_column_rows(c).empty()) {
            fold_cols_empty = false;
        }
    }
    // Zero-postings early-out: no term touches a fold column ⇒ nothing anticommutes, so skip pass 1.
    // g_odd guard is load-bearing: an odd Majorana generator anticommutes with disjoint odd-weight terms.
    if (!g_odd && fold_cols_empty) {
        observer.prepare(KernelRange::scan, 0);
        auto res = empty_result();
        res.cos_blocks.push_back(CosMask{});
        return res;
    }

    const auto scan_words = [&](size_t wlo, size_t whi) {
        return fused_find_and_collect_range<NumModes, A>(op,
                                                         gen,
                                                         cutoff_eval,
                                                         cut_st,
                                                         coeffs,
                                                         only_rotate_len_k,
                                                         window,
                                                         my_rank,
                                                         router,
                                                         gen_shift,
                                                         index,
                                                         row_parity,
                                                         wlo,
                                                         whi,
                                                         capture_values,
                                                         fused_scale_coeffs,
                                                         fused_scale_cos);
    };
    // Opaque cutoffs and small scans keep the serial scan: one range on the caller, no merge.
    const size_t ranges = cutoff_eval.parallel_safe() ? scan_ranges(word_count, options) : 1;
    observer.prepare(KernelRange::scan, ranges);
    if (ranges <= 1) {
        observer.visit(KernelRange::scan, 0);
        return scan_words(0, word_count);
    }
    // One private result per logical range, never per fold block: O(ranges * window.count) metadata.
    std::vector<FusedScanResult<NumModes>> pieces(ranges);
    parallel::for_blocks(ranges, options, [&](size_t range) {
        observer.visit(KernelRange::scan, range);
        const auto [wlo, whi] = scan_range_words(range, ranges, word_count);
        pieces[range] = scan_words(wlo, whi);
    });
    return merge_scan_ranges<NumModes>(pieces, window, my_rank, capture_values);
}

} // namespace monoprop::detail
