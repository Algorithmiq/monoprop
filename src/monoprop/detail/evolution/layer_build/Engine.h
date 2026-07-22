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
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/profiling/RegionProfiler.h"

namespace monoprop::detail {

// Owns build_layer's machinery: per-rank accumulator, matched-follower set, query streams, deferred
// self-misses, and the resolve/exchange/finalize ops. combined_size = the pre-layer operator size.
template <size_t NumModes>
struct LayerBuildEngine {
    struct DeferredSelfMiss {
        Monomial<NumModes> maj;
        size_t src;
        int phase;
        double v_src = 0.0; // fused only: op_pre[src] captured at scan emit; 0 (unused) otherwise
    };
    // config (set at construction): methods need only the operator + MPI topology. Cutoffs/generator/
    // coeffs live in the build_layer orchestrator (they drive the scan/metadata), so are not held here.
    MPOperator<NumModes> &local_op; // scanned, looked up, and grown by the inserts
    mpi::Comm comm;
    size_t R;
    size_t my_rank;
    // Picture flag (fused R>1 only): selects cross-rank MISS handling — a fresh insert's coeff is 0 in
    // Heisenberg but HF-scored in Schrödinger. Unused at R==1 and in the non-fused path.
    bool schrodinger_ = false;
    // Operator basis (fused R>1 only): Pauli vs Majorana HF scoring of fresh cross-rank Schrödinger misses.
    Basis basis_ = Basis::Majorana;

    std::vector<PartnerAcc> acc;
    // Follower-matched set over [0, combined_size): caller-owned + epoch-stamped so no O(n) per-gate
    // clear (see MatchedEpochSet). Atomics-free: ≤1 writer per slot (distinct leaders → distinct found).
    MatchedEpochSet &matched;
    size_t combined_size;
    std::vector<VecZ> queries_r;
    std::vector<std::vector<size_t>> src_idx_r;
    std::vector<DeferredSelfMiss> deferred_self_misses;

    // fused contraction (set by build_layer, all ranks): when fused_ != nullptr the engine emits
    // RotationRec streams into it (no acc/LayerCore) and reads pre-cos target coeffs from op_coeffs_.
    // src_val_r is parallel to src_idx_r: the scan-captured v_src per query.
    FusedContract *fused_ = nullptr;
    const VecD *op_coeffs_ = nullptr;
    std::vector<std::vector<double>> src_val_r;
    // Fused query+value send buffer (R>1): interleaves queries_r + src_val_r into one
    // kQueryWordsFused-wide stream so a SINGLE alltoallv carries both. Reused across gates.
    std::vector<VecZ> combined_qv_;

    // Fused cos sweep (k==0): the scan already scaled every anticommuting coeff by cos(2θ), so a hit
    // partner's stored value is POST-cos; resolve recovers pre-cos v_tgt as stored·inv_cos_.
    bool fused_scale_ = false;
    double inv_cos_ = 1.0;

    LayerBuildEngine(MPOperator<NumModes> &local_op_,
                     mpi::Comm comm_,
                     size_t R_,
                     size_t my_rank_,
                     MatchedEpochSet &matched_scratch,
                     size_t combined_size_,
                     bool schrodinger = false)
        : local_op(local_op_),
          comm(comm_),
          R(R_),
          my_rank(my_rank_),
          schrodinger_(schrodinger),
          acc(R_),
          matched(matched_scratch),
          combined_size(combined_size_),
          queries_r(R_),
          src_idx_r(R_) {
        matched.begin_gate(combined_size);
    }

    // Resolves THIS rank's own query stream (queries_r[my_rank]/src_idx_r[my_rank], populated by the
    // current pass) inline; clears it so the subsequent alltoallv never sends to self.
    auto resolve_self_queries(bool is_leader_pass) -> void {
        VecZ &lq = queries_r[my_rank];
        std::vector<size_t> &ls = src_idx_r[my_rank];
        std::vector<double> *lv = fused_ ? &src_val_r[my_rank] : nullptr;
        const size_t nq = lq.empty() ? 0 : lq.size() / kQueryWords<NumModes>;
        // Append straight into the accumulator (or, fused, the record sinks), zero staging copy.
        resolve_range_(lq,
                       ls,
                       lv,
                       0,
                       nq,
                       is_leader_pass,
                       acc[my_rank].in_entries,
                       acc[my_rank].out_entries,
                       deferred_self_misses,
                       fused_ ? &fused_->hits : nullptr);
        lq.clear();
        ls.clear();
        if (lv != nullptr) {
            lv->clear();
        }
    }

    // One partner-resolution pass: resolve self-rank queries inline, then (multi-rank) alltoallv-exchange
    // cross-rank queries and fold in the answers. is_leader_pass selects the leader vs. follower half.
    auto run_exchange(bool is_leader_pass) -> void {
        {
            profiling::ScopedRegion prof_sr(profiling::Region::SelfResolve);
            resolve_self_queries(is_leader_pass);
        }
        if (R <= 1) {
            return;
        }
        profiling::ScopedRegion prof_mx(profiling::Region::MpiExchange);
        if (fused_ != nullptr) {
            // Fused R>1 exchange. Round 1: queries A→B fused with the v_src value stream (one bit-cast
            // trailing word per record), so ONE alltoallv replaces the former query+value pair.
            combined_qv_.resize(R);
            for (size_t r = 0; r < R; ++r) {
                build_fused_query_value<NumModes>(queries_r[r], src_val_r[r], combined_qv_[r]);
            }
            std::vector<std::vector<size_t>> inc_q;
            mpi::begin_alltoallv(combined_qv_, comm).wait_into(inc_q);
            // Resolver: emit half-rotations into fused_ and return one VALUE per query — target coeff for a
            // HIT, freshly-computed insert coeff for a MISS — in the SAME round (see resolve_incoming_queries_fused).
            auto resp_val = resolve_incoming_queries_fused(inc_q,
                                                           local_op,
                                                           R,
                                                           is_leader_pass,
                                                           matched,
                                                           combined_size,
                                                           *op_coeffs_,
                                                           schrodinger_,
                                                           *fused_,
                                                           fused_scale_,
                                                           inv_cos_,
                                                           basis_);
            // Round 2: value responses B→A, using the known transpose recv counts (see response_recv_counts).
            std::vector<int> resp_recv = response_recv_counts();
            std::vector<std::vector<double>> inc_rval;
            mpi::begin_alltoallv(resp_val, comm, /*skip_self=*/false, &resp_recv).wait_into(inc_rval);
            process_query_responses_fused<NumModes>(inc_rval, src_idx_r, queries_r, R, my_rank, *fused_);
            return;
        }
        // Non-fused R>1 exchange (graph build / replay).
        std::vector<std::vector<size_t>> inc_q;
        mpi::begin_alltoallv(queries_r, comm).wait_into(inc_q);
        auto resps = resolve_incoming_queries(inc_q, local_op, R, is_leader_pass, matched, combined_size, acc);
        // One TermIndex response per query, with the known transpose recv counts (see response_recv_counts).
        std::vector<int> resp_recv_counts = response_recv_counts();
        std::vector<std::vector<TermIndex>> inc_r;
        mpi::begin_alltoallv(resps, comm, /*skip_self=*/false, &resp_recv_counts).wait_into(inc_r);
        process_query_responses<NumModes>(inc_r, src_idx_r, queries_r, R, my_rank, acc);
    }

    // In-place compact the per-rank cross-rank follower query streams, dropping followers a leader
    // already matched in the leader pass (matched.is_marked(src)) so they are not re-resolved over the wire.
    auto drop_matched_cross_rank_followers() -> void {
        constexpr size_t W = kQueryWords<NumModes>;
        for (size_t r = 0; r < R; ++r) {
            if (r == my_rank) {
                continue;
            }
            VecZ &q = queries_r[r];
            std::vector<size_t> &s = src_idx_r[r];
            // Fused: the v_src value stream is parallel to the query/source streams and is alltoallv'd
            // alongside them, so it must be compacted in lockstep (nullptr in the non-fused path).
            std::vector<double> *v = (fused_ != nullptr) ? &src_val_r[r] : nullptr;
            const size_t nq = s.size();
            size_t kept = 0;
            for (size_t k = 0; k < nq; ++k) {
                if (matched.is_marked(s[k])) {
                    continue;
                }
                if (kept != k) { // slide the surviving query's W words down into the next kept slot
                    std::copy(q.begin() + static_cast<std::ptrdiff_t>(k * W),
                              q.begin() + static_cast<std::ptrdiff_t>((k + 1) * W),
                              q.begin() + static_cast<std::ptrdiff_t>(kept * W));
                }
                s[kept] = s[k];
                if (v != nullptr) {
                    (*v)[kept] = (*v)[k];
                }
                ++kept;
            }
            q.resize(kept * W);
            s.resize(kept);
            if (v != nullptr) {
                v->resize(kept);
            }
        }
    }

    // Sub-step of finish() — do not call directly. LOAD-BEARING precondition: call only AFTER both resolve
    // passes complete, else the base+k ↔ acc-slot assignment and per-miss distinctness break.
    auto insert_deferred_self_misses() -> void {
        const size_t n_miss = deferred_self_misses.size();
        if (n_miss > 0) {
            profiling::ScopedRegion prof_di(profiling::Region::DeferInsert);
            // Parallel deterministic insert (any rank count). Deferred SELF misses are pairwise-distinct
            // (maj = source⊕G, ⊕G injective) and still absent, so miss k gets base+k in leader-then-follower
            // order — byte-identical to the serial loop, no dedup, no atomics (disjoint slots/shards/index
            // words). See insert_absent_terms.
            auto key_at = [&](size_t k) -> const Monomial<NumModes> & { return deferred_self_misses[k].maj; };
            if (fused_ != nullptr) {
                // Fused: append INSERT records (v_tgt filled later, after op_coeffs is extended).
                const size_t rec_base = fused_->inserts.size();
                fused_->inserts.resize(rec_base + n_miss);
                insert_absent_terms<NumModes>(local_op, n_miss, key_at, [&](size_t k, size_t base) {
                    const auto &m = deferred_self_misses[k];
                    assign_row<NumModes>(*local_op.store, base + k, m.maj);
                    fused_->inserts[rec_base + k] =
                        RotationRec{m.src, base + k, m.v_src, /*v_tgt=*/0.0, static_cast<int32_t>(m.phase)};
                });
            }
            else {
                const size_t in_base = acc[my_rank].in_entries.size();
                const size_t out_base = acc[my_rank].out_entries.size();
                acc[my_rank].in_entries.resize(in_base + n_miss);
                acc[my_rank].out_entries.resize(out_base + n_miss);
                insert_absent_terms<NumModes>(local_op, n_miss, key_at, [&](size_t k, size_t base) {
                    const auto &m = deferred_self_misses[k];
                    assign_row<NumModes>(*local_op.store, base + k, m.maj);
                    acc[my_rank].in_entries[in_base + k] = {base + k, m.phase};
                    acc[my_rank].out_entries[out_base + k] = {m.src, m.phase};
                });
            }
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
            p.sin_send_indices.resize(P + Q);
            p.sin_recv_entries.resize(P + Q);
            for (size_t k = 0; k < P + Q; ++k) {
                if (k < P) {
                    const auto &e = a.in_entries[k];
                    p.sin_send_indices[k] = e.idx;
                    p.sin_recv_entries[Q + k] = {e.idx, e.phase};
                }
                else {
                    const size_t j = k - P;
                    const auto &e = a.out_entries[j];
                    p.sin_send_indices[k] = e.idx;
                    p.sin_recv_entries[j] = {e.idx, -e.phase};
                }
            }
        }
        return partners;
    }

    // cos is not stored on the layer: hand it over when `out_cos` is non-null (the in-build contraction
    // needs it for evolve_step); otherwise it is recomputed from the inverted index fold at replay.
    auto finish(CosMask &&cos_all, CosMask *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        insert_deferred_self_misses();
        if (fused_ != nullptr) {
            // Fused (all ranks): the LayerCore is transient, so skip assemble/build_layer_storage and return
            // nullptr. Two-pass fused (k>0 / cos==0 fallback) appends inserted endpoints into cos so the
            // immediate cos scale covers them; the fused cos sweep built no set and its apply covers inserts
            // in-place, so leave *out_cos empty there.
            if (out_cos != nullptr && !fused_scale_) {
                append_inserted_endpoints_(cos_all);
                *out_cos = std::move(cos_all);
            }
            return nullptr;
        }
        profiling::ScopedRegion prof_gather(profiling::Region::Gather);
        std::vector<CrossRankPartnerData> partners = assemble_partners();
        if (out_cos != nullptr) {
            append_inserted_endpoints_(cos_all);
            *out_cos = std::move(cos_all);
        }
        return build_layer_storage_unified(std::move(partners), my_rank);
    }

private:
    // Response counts are the TRANSPOSE of the query counts (one answer per query), so passing them as
    // known_recv_counts skips the response count-Alltoall round.
    auto response_recv_counts() const -> std::vector<int> {
        std::vector<int> counts(R);
        for (size_t r = 0; r < R; ++r) {
            counts[r] = static_cast<int>(queries_r[r].size() / kQueryWords<NumModes>);
        }
        return counts;
    }

    // Every rotation TARGET must be in cos so the gradient reverse-sweep can un-do this layer's cosine
    // scaling; only freshly INSERTED half-terms can be absent (see test_infinite_cutoff). Inserts are
    // APPENDED in [combined_size, local_op.size()), so append just that range, not every target.
    auto append_inserted_endpoints_(CosMask &cos_all) -> void {
        const size_t cos_lo = combined_size;
        const size_t cos_hi = local_op.store->size();
        CosineWordBuilder end_b;
        for (size_t idx = cos_lo; idx < cos_hi; ++idx) {
            end_b.push_index(idx);
        }
        CosMask end_words = end_b.finish();
        // Scan cos bits (< cos_lo) and inserted endpoint bits (≥ cos_lo) are disjoint, so only the seam
        // word can carry both — OR it, then append the rest, keeping blocks ascending/disjoint.
        cos_all.total_count += end_words.total_count;
        if (!cos_all.blocks.empty() && !end_words.blocks.empty()
            && end_words.blocks.front().first == cos_all.blocks.back().first) {
            cos_all.blocks.back().second |= end_words.blocks.front().second;
            cos_all.blocks.insert(cos_all.blocks.end(), end_words.blocks.begin() + 1, end_words.blocks.end());
        }
        else {
            cos_all.blocks.insert(cos_all.blocks.end(), end_words.blocks.begin(), end_words.blocks.end());
        }
    }

    // Batched: gather up to kResolveBatch surviving queries, resolve via the index's group-prefetch
    // find_batch, then emit in query order — transparent to a one-find-at-a-time loop's result.
    static constexpr size_t kResolveBatch = 64;
    // `lv` (fused only) is the per-query v_src array parallel to `ls`. `hit_sink` (fused only) receives HIT
    // RotationRecs; when non-null the acc in/out sinks are NOT written and misses carry v_src.
    auto resolve_range_(VecZ &lq,
                        std::vector<size_t> &ls,
                        std::vector<double> *lv,
                        size_t lo,
                        size_t hi,
                        bool is_leader_pass,
                        DefaultInitVector<PhasedEntry> &in_sink,
                        DefaultInitVector<PhasedEntry> &out_sink,
                        std::vector<DeferredSelfMiss> &miss_sink,
                        std::vector<RotationRec> *hit_sink) -> void {
        const bool fused = (hit_sink != nullptr);
        const size_t op_size = local_op.store->size();
        std::array<Monomial<NumModes>, kResolveBatch> keys;
        std::array<int, kResolveBatch> phases;
        std::array<size_t, kResolveBatch> srcs;
        std::array<double, kResolveBatch> vals;
        std::array<size_t, kResolveBatch> found;
        size_t q = lo;
        while (q < hi) {
            // Gather the next batch of queries that survive the follower-matched skip.
            size_t m = 0;
            for (; q < hi && m < kResolveBatch; ++q) {
                const size_t src = ls[q];
                if (!is_leader_pass && matched.is_marked(src)) {
                    continue; // follower already matched by a leader → not an independent rotation
                }
                query_read<NumModes>(lq, q, keys[m], phases[m]);
                srcs[m] = src;
                if (fused) {
                    vals[m] = (*lv)[q];
                }
                ++m;
            }
            if (m == 0) {
                break;
            }
            local_op.store->find_batch(keys.data(), m, found.data());
            for (size_t j = 0; j < m; ++j) {
                // kNotFound == kMissingIndex == size_t max, so one bound check covers both.
                if (found[j] < op_size) {
                    if (is_leader_pass) {
                        matched.mark(found[j]); // distinct leaders → distinct found → no atomics
                    }
                    if (fused) {
                        // Capture the partner's PRE-cos v_tgt: under the fused cos sweep op_coeffs_[found]
                        // was already scaled, so recover it with inv_cos_; otherwise it is still pre-cos.
                        const double v_tgt =
                            fused_scale_ ? (*op_coeffs_)[found[j]] * inv_cos_ : (*op_coeffs_)[found[j]];
                        hit_sink->push_back(
                            RotationRec{srcs[j], found[j], vals[j], v_tgt, static_cast<int32_t>(phases[j])});
                    }
                    else {
                        in_sink.push_back({found[j], phases[j]});
                        out_sink.push_back({srcs[j], phases[j]});
                    }
                }
                else {
                    miss_sink.push_back({keys[j], srcs[j], phases[j], fused ? vals[j] : 0.0});
                }
            }
        }
    }
};

// Primary-path layer builder (paper Algorithm 2): the fused scan feeds two MPI exchange passes into a
// per-rank PartnerAcc, then self-rank absent partners are inserted (load-bearing: AFTER both resolves)
// and a LayerCore is assembled. See LayerBuilder.h for the algorithm.
template <size_t NumModes>
auto build_layer(MPOperator<NumModes> &local_op,
                 const Monomial<NumModes> &gen,
                 const CutoffFn<NumModes> &cutoff_fn,
                 const std::optional<double> &atol,
                 std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                 const std::optional<double> &upper_atol,
                 const std::optional<double> &param,
                 int only_rotate_len_k,
                 MatchedEpochSet &matched_scratch,
                 mpi::Comm comm,
                 CosMask *out_cos = nullptr,
                 FusedContract *fused_contract = nullptr,
                 bool schrodinger = false,
                 VecD *fused_scale_coeffs = nullptr,
                 bool *fused_scale_out = nullptr,
                 Basis basis = Basis::Majorana) -> std::shared_ptr<LayerCore> {
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = static_cast<size_t>(mpi::size(comm));
    // Fused contraction: the caller passes a non-null sink for the ContractImmediately forward path.
    // Runs at ALL rank counts (R>1 uses the cross-rank half-rotation exchange); the sole guard is the sink.
    const bool use_fused = (fused_contract != nullptr);
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs ? local_coeffs->get() : empty_coeffs();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // Fused cos sweep: fold the per-gate cosine scale into the scan's own coefficient pass (one sweep vs
    // the eager read + CosMask + RMW-scale). k==0 only (a popcount>k hit is outside the per-index cos set,
    // so 1/cos recovery would be wrong) and cos!=0 (else recovery is impossible; two-pass fallback). cos is
    // even, so the sweep's cos(2·build_angle) matches the apply's cos(2·apply_angle) bit-for-bit.
    const double cos_build = (use_fused && param.has_value()) ? std::cos(2.0 * param.value()) : 1.0;
    const bool fused_scale =
        use_fused && only_rotate_len_k == 0 && fused_scale_coeffs != nullptr && param.has_value() && cos_build != 0.0;
    // build_layer is the single authority for this decision; report it so the fused caller drives the
    // apply from the SAME decision instead of risking a build/apply disagreement.
    if (fused_scale_out != nullptr) {
        *fused_scale_out = fused_scale;
    }
    assert(fused_scale_coeffs == nullptr || (local_coeffs && &local_coeffs->get() == fused_scale_coeffs));

    FusedScanResult fused = [&] {
        profiling::ScopedRegion prof_find(profiling::Region::Find);
        // Dispatch the scan on the basis at compile time (Pauli emit-sign/J(G) fold vs Majorana
        // interleave/hermitian phase). Every other argument is basis-agnostic.
        double *const sweep_ptr = fused_scale ? fused_scale_coeffs->data() : nullptr;
        return with_algebra<NumModes>(basis, [&]<class A>() {
            return fused_find_and_collect<NumModes, A>(local_op,
                                                       gen,
                                                       cut_eval,
                                                       cut_st,
                                                       coeffs,
                                                       only_rotate_len_k,
                                                       R,
                                                       my_rank,
                                                       /*capture_values=*/use_fused,
                                                       sweep_ptr,
                                                       cos_build);
        });
    }();

    CosMask cos_all;
    if (fused.cos_blocks.size() == 1) {
        // The serial scan produces a single cosine block set — take it wholesale.
        cos_all = std::move(fused.cos_blocks[0]);
    }
    else {
        // Cosine block sets are disjoint and ascending; concatenate in order.
        for (const auto &block : fused.cos_blocks) {
            cos_all.total_count += block.total_count;
            cos_all.blocks.insert(cos_all.blocks.end(), block.blocks.begin(), block.blocks.end());
        }
    }
    fused.cos_blocks = std::vector<CosMask>{};

    LayerBuildEngine<NumModes> eng(local_op,
                                   comm,
                                   R,
                                   my_rank,
                                   matched_scratch,
                                   /*combined_size=*/local_op.store->size(),
                                   schrodinger);
    eng.basis_ = basis;
    if (use_fused) {
        eng.fused_ = fused_contract;
        eng.op_coeffs_ = &coeffs; // SAME array the scan read (= *op_coeffs)
        eng.fused_scale_ = fused_scale;
        if (fused_scale) {
            eng.inv_cos_ = 1.0 / cos_build; // pre-cos recovery factor for hit v_tgt (see resolve_range_)
        }
    }

    eng.queries_r = std::move(fused.leader_queries);
    eng.src_idx_r = std::move(fused.leader_src);
    if (use_fused) {
        eng.src_val_r = std::move(fused.leader_val);
    }
    eng.run_exchange(/*is_leader_pass=*/true);

    eng.queries_r = std::move(fused.follower_queries);
    eng.src_idx_r = std::move(fused.follower_src);
    if (use_fused) {
        eng.src_val_r = std::move(fused.follower_val);
    }
    if (R > 1) {
        eng.drop_matched_cross_rank_followers();
    }
    eng.run_exchange(/*is_leader_pass=*/false);

    auto storage = eng.finish(std::move(cos_all), out_cos);

    // Recompute metadata rides WITH the layer so it survives every graph transform. scaled_count is the
    // POST-insert operator size: folding the inverted index truncated to it reproduces the "all
    // anticommuting" cos bit-for-bit with no stored bitmap. Fused mode has no LayerCore to stamp.
    if (storage != nullptr) {
        storage->generator_words.assign(gen.data(), gen.data() + mpi_detail::kWords<NumModes>);
        storage->scaled_count = static_cast<uint64_t>(local_op.store->size());
    }

    return storage;
}

} // namespace monoprop::detail
