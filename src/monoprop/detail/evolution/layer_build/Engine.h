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

#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Parallel.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/profiling/RegionProfiler.h"

namespace monoprop::detail {

// ─── LayerBuildEngine ─────────────────────────────────────────────────────────
// Owns the machinery for build_layer: the per-rank accumulator, the (caller-owned,
// epoch-stamped) matched-follower set, the per-rank query streams, the deferred self-misses, and the
// resolve/exchange/finalize operations. combined_size = the pre-layer operator size.
template <size_t NumModes>
struct LayerBuildEngine {
    struct DeferredSelfMiss {
        MajoranaSet<NumModes> maj;
        size_t src;
        int phase;
        double v_src = 0.0; // fused only: op_pre[src] captured at scan emit; 0 (unused) otherwise
    };
    // ── config (set at construction) ──
    // The engine's methods only need the operator + the MPI topology. The cutoffs / generator /
    // coeffs live in the free-function orchestrator (build_layer): they drive the fused
    // scan and the metadata, not the resolve/exchange/finish machinery, so they are deliberately NOT
    // held here — the struct advertises exactly the surface its methods touch.
    MPOperator<NumModes> &local_op; // scanned, looked up, and grown by the inserts
    mpi::Comm comm;
    size_t R;
    size_t my_rank;
    // Picture flag (fused R>1 only): selects cross-rank MISS handling. A fresh cross-rank insert's
    // coeff is 0 in Heisenberg (querier half is a no-op) but HF-scored non-zero in Schrödinger (querier
    // half's v_partner needs a post-extension exchange). Unused at R==1 and in the non-fused path.
    bool schrodinger_ = false;
    // Operator basis (fused R>1 only): selects Pauli vs Majorana HF scoring of fresh cross-rank
    // Schrödinger misses in the fused resolver. Set by build_layer; Majorana by default.
    Basis basis_ = Basis::Majorana;

    // ── state (grows during the build) ──
    std::vector<PartnerAcc> acc;
    // Follower-matched set over the combined index space [0, combined_size): epoch-stamped and owned
    // by the caller so no O(n) per-gate clear (see MatchedEpochSet). Atomics-free: ≤1 writer per slot
    // (distinct leaders → distinct found via injective ⊕G), leader-pass writes / follower-pass reads.
    MatchedEpochSet &matched;
    size_t combined_size;
    std::vector<VecZ> queries_r;
    std::vector<std::vector<size_t>> src_idx_r;
    std::vector<DeferredSelfMiss> deferred_self_misses;

    // ── fused contraction (set by build_layer for the ContractImmediately forward path, all ranks) ──
    // When fused_ != nullptr the engine emits RotationRec streams into it (skipping the acc /
    // in_entries / out_entries and the LayerCore) and reads pre-cos target coeffs from op_coeffs_.
    // src_val_r is parallel to src_idx_r (self-rank only at R==1): the scan-captured v_src per query.
    FusedContract *fused_ = nullptr;
    const VecD *op_coeffs_ = nullptr;
    std::vector<std::vector<double>> src_val_r;
    // Fused query+value send buffer (R>1 ContractImmediately): each gate we interleave queries_r + src_val_r
    // into one kQueryWordsFused-wide stream so a SINGLE alltoallv carries both. Reused across gates (HWM).
    std::vector<VecZ> combined_qv_;

    // Fused cos sweep (ContractImmediately k==0): the scan already multiplied every anticommuting
    // coefficient by cos(2θ) in its own pass, so a hit partner's stored value is POST-cos here; resolve
    // recovers the pre-cos v_tgt as stored·inv_cos_ (one extra rounding, ≤1 ulp — see build_layer).
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
        const size_t chunks = partition_chunk_count(nq);
        if (chunks <= 1) {
            // Serial (also the nq==0 case): append straight into the accumulator (or, fused, the record
            // sinks), zero staging copy.
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
        }
        else {
            // Probes run chunked in parallel (lock-free lookup, ≤1 writer per matched slot). The
            // order-sensitive outputs are collected per-chunk and concatenated in chunk order, so the
            // result (including deferred-miss / hit-record insertion order) matches the serial scan.
            std::vector<DefaultInitVector<PhasedEntry>> in_parts(chunks);
            std::vector<DefaultInitVector<PhasedEntry>> out_parts(chunks);
            std::vector<std::vector<DeferredSelfMiss>> miss_parts(chunks);
            std::vector<std::vector<RotationRec>> hit_parts(fused_ ? chunks : 0);
            for_each_chunk(nq, chunks, [&](size_t c, size_t lo, size_t hi) {
                resolve_range_(lq,
                               ls,
                               lv,
                               lo,
                               hi,
                               is_leader_pass,
                               in_parts[c],
                               out_parts[c],
                               miss_parts[c],
                               fused_ ? &hit_parts[c] : nullptr);
            });
            if (!fused_) {
                append_gathered_chunks(acc[my_rank].in_entries, in_parts);
                append_gathered_chunks(acc[my_rank].out_entries, out_parts);
            }
            append_gathered_chunks(deferred_self_misses, miss_parts);
            if (fused_) {
                append_gathered_chunks(fused_->hits, hit_parts);
            }
        }
        lq.clear();
        ls.clear();
        if (lv != nullptr) {
            lv->clear();
        }
    }

    // One partner-resolution pass: resolve this rank's self-rank queries inline, then (multi-rank
    // only) alltoallv-exchange the cross-rank queries and fold in the one-response-per-query answers.
    // is_leader_pass selects the leader vs. follower half of the gate's two passes.
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
            // ── Fused R>1 exchange (ContractImmediately) ──
            // Round 1: queries A→B FUSED with the v_src value stream — each query record carries its own
            // v_src as a trailing bit-cast word (build_fused_query_value), so ONE alltoallv replaces the
            // former query+value pair. Saves a full count+payload round every gate (bit-identical: the
            // value travels adjacent to its query, same routing / per-source order as the two-stream path).
            combined_qv_.resize(R);
            for (size_t r = 0; r < R; ++r) {
                build_fused_query_value<NumModes>(queries_r[r], src_val_r[r], combined_qv_[r]);
            }
            std::vector<std::vector<size_t>> inc_q;
            mpi::begin_alltoallv(combined_qv_, comm).wait_into(inc_q);
            // Resolver: emit half-rotations into fused_ and return one VALUE per incoming query — the
            // real target coeff for a HIT and the freshly-computed insert coeff for a MISS, both in the
            // SAME round. There is no NaN sentinel and no second exchange (see resolve_incoming_queries_fused).
            // inc_q now holds fused (maj,phase,v_src) records; the resolver reads v_src straight off each.
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
        // ── Non-fused R>1 exchange (graph build / replay) — unchanged ──
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

    // Sub-step of finish() — do not call directly. Precondition (LOAD-BEARING): call only AFTER both
    // resolve passes complete. Inserting earlier would corrupt the base+k ↔ acc-slot index assignment
    // established below (and the per-miss distinctness argument relies on all passes having run).
    auto insert_deferred_self_misses() -> void {
        const size_t n_miss = deferred_self_misses.size();
        if (n_miss > 0) {
            profiling::ScopedRegion prof_di(profiling::Region::DeferInsert);
            // ── Parallel deterministic insert (any rank count) ──
            // The deferred SELF misses are pairwise-distinct (each maj is source⊕G over distinct op
            // terms, ⊕G injective) and still absent (a cross-rank term inserted mid-pass is some other
            // rank's source'⊕G, source'≠source). So miss k, in deterministic leader-then-follower
            // order, is assigned base+k — byte-identical to the serial loop — with no dedup and NO
            // ATOMICS: op slots, map shards, inverted index words and acc slots are written by disjoint tasks.
            // Grow → scatter → index → resync (see insert_absent_terms). key_at reads the staged dense
            // MajoranaSet directly (no packed-row re-materialization); per_slot scatters the row into the
            // disjoint op slot base+k plus the matching per-record side entry. Side arrays are resized
            // before the insert (their base offsets don't depend on the op insert base).
            auto key_at = [&](size_t k) -> const MajoranaSet<NumModes> & { return deferred_self_misses[k].maj; };
            if (fused_ != nullptr) {
                // Fused: append INSERT records (v_tgt filled later, after op_coeffs is extended). No acc /
                // in_entries / out_entries in fused mode.
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
            // One parallel region over P+Q: k<P fills in-entry slots, k>=P fills out-entry slots.
            parallel_for_indices(P + Q, [&](size_t k) {
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
            });
        }
        return partners;
    }

    // cos is not stored on the layer. When `out_cos` is non-null the in-build contraction needs the
    // full anticommuting cos for the immediate evolve_step, so we hand it over; otherwise cos is
    // discarded and recomputed from the inverted index fold at replay (generator_words + scaled_count).
    auto finish(CosMask &&cos_all, CosMask *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        insert_deferred_self_misses();
        if (fused_ != nullptr) {
            // Fused (ContractImmediately, all ranks): the LayerCore is transient in this mode, so skip
            // assemble_partners + build_layer_storage_unified entirely and return nullptr. In the two-pass
            // fused path (k>0 / cos==0 fallback) we append the inserted endpoints into cos (see
            // append_inserted_endpoints_) so the immediate cos scale covers them, exactly as the non-fused
            // evolve_step path expects. Under the fused cos sweep the scan applied cos in place and built
            // no cosine set; the apply covers inserts via its in-place insert arm and never reads *out_cos
            // — building/moving it would be dead work on the hot per-gate path, so leave it empty.
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
    // Response counts are the TRANSPOSE of the query counts: resolve returns exactly one answer per query,
    // so recv_counts[r] == queries_r[r].size()/W. Both sides know this, so passing it as known_recv_counts
    // skips the response count-Alltoall round (W is the fixed per-query word count, so the division is exact).
    auto response_recv_counts() const -> std::vector<int> {
        std::vector<int> counts(R);
        for (size_t r = 0; r < R; ++r) {
            counts[r] = static_cast<int>(queries_r[r].size() / kQueryWords<NumModes>);
        }
        return counts;
    }

    // Every rotation TARGET must be in cos so the gradient reverse-sweep can recover its pre-layer
    // coefficient by un-doing this layer's cosine scaling. Cycle targets are already in cos from the
    // fused scan; only freshly INSERTED half-terms can be absent. Forward energy is unaffected (an
    // inserted target's coefficient is 0 when the cos pass runs). Without this the reverse sweep
    // over-scales those endpoints — see test_infinite_cutoff.
    //
    // Inserts are APPENDED, occupying [combined_size, local_op.size()), so we append just that range
    // (O(inserted)) instead of scanning every rotation target (a serial Amdahl anchor).
    auto append_inserted_endpoints_(CosMask &cos_all) -> void {
        const size_t cos_lo = combined_size;
        const size_t cos_hi = local_op.store->size();
        CosineWordBuilder end_b;
        for (size_t idx = cos_lo; idx < cos_hi; ++idx) {
            end_b.push_index(idx);
        }
        CosMask end_words = end_b.finish();
        // Scan cos bits are all < cos_lo (pre-insert indices); the freshly-inserted endpoint bits are
        // all >= cos_lo. The two sets are disjoint, so ONLY the seam word (cos_lo>>6, when cos_lo is
        // not 64-aligned) can carry bits from both — OR just that word, then append the rest. This keeps
        // blocks ascending/disjoint, the invariant the inverted index fold + replay need.
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

    // Batched: gather up to kResolveBatch surviving queries, resolve them with the index's
    // group-prefetch find_batch (which overlaps the independent DRAM misses of the probes), then emit
    // sequentially in query order. Emission order, matched marks and miss order are identical to a
    // one-find-at-a-time loop, so the batching is transparent to the result.
    static constexpr size_t kResolveBatch = 64;
    // `lv` (fused only, else nullptr) is the per-query v_src array parallel to `ls`. `hit_sink` (fused
    // only, else nullptr) receives HIT RotationRecs; when it is non-null the acc in/out sinks are NOT
    // written (no LayerCore in fused mode) and misses carry v_src.
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
        std::array<MajoranaSet<NumModes>, kResolveBatch> keys;
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
                        // Capture the partner's PRE-cos v_tgt now. Under the fused cos sweep the scan
                        // already scaled op_coeffs_[found] (found < combined_size, anticommuting ⇒ swept),
                        // so recover the pre-cos value with the inverse factor; without the sweep (k>0 /
                        // cos==0 fallback) the stored value is still pre-cos and stays so (extend only
                        // appends, the mask scale runs after build).
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

// ─── build_layer ─────────────────────────────────────────────────
// Primary-path layer builder. Implements paper Algorithm 2 and emits a graph layer directly.
// Runs the fused scan (FindAnticommuting + apply_cutoffs in one walk) to produce the compressed
// cosine blocks and cutoff-applied per-rank leader/follower query streams. During the two exchange
// passes, rotation participants accumulate into a uniform per-rank PartnerAcc (self slot = partner
// with in:=tgt, out:=src). After both passes: self-rank absent partners are inserted (load-bearing:
// AFTER both resolves), the per-rank CrossRankPartnerData is assembled, and a LayerCore is built.
template <size_t NumModes>
auto build_layer(MPOperator<NumModes> &local_op,
                 const MajoranaSet<NumModes> &gen,
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
    // Fused contraction: the caller (evolve_mode_contract_immediately_) passes a non-null sink for the
    // ContractImmediately forward path. Fused now runs at ALL rank counts (R>1 uses the cross-rank
    // half-rotation exchange in run_exchange); the sole guard is a non-null sink.
    const bool use_fused = (fused_contract != nullptr);
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs ? local_coeffs->get() : empty_coeffs();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // Fused cos sweep: the ContractImmediately mode's cos implementation (use_fused == that mode, and the
    // caller hands the picture's MUTABLE coeff vector through fused_scale_coeffs). The scan folds the
    // per-gate cosine scale into its own coefficient pass — one sweep instead of the eager read-pass +
    // CosMask round-trip + RMW-scale-pass, which restreamed every anticommuting coefficient from DRAM
    // twice per gate. k==0 only: a hit target with popcount>k is outside the k>0 per-index cos set, so
    // the 1/cos recovery in resolve would be wrong for it — only_rotate_len_k>0 keeps the two-pass eager.
    // cos(2·param)==0.0 exactly would make the recovery impossible; fall back to two-pass (unreachable
    // for real angles — no double has cosine exactly 0 — defensive only). cos is even, so the sweep's
    // cos(2·build_angle) equals the apply's cos(2·apply_angle) bit-for-bit (apply_angle = ±build_angle).
    const double cos_build = (use_fused && param.has_value()) ? std::cos(2.0 * param.value()) : 1.0;
    const bool fused_scale =
        use_fused && only_rotate_len_k == 0 && fused_scale_coeffs != nullptr && param.has_value() && cos_build != 0.0;
    // build_layer is the single authority for this decision; report it so the fused caller
    // (evolve_mode_contract_immediately_) drives the apply (skip-the-mask-scale, in-place insert arm)
    // from the SAME decision instead of recomputing it and risking a build/apply disagreement.
    if (fused_scale_out != nullptr) {
        *fused_scale_out = fused_scale;
    }
    assert(fused_scale_coeffs == nullptr || (local_coeffs && &local_coeffs->get() == fused_scale_coeffs));

    FusedScanResult fused = [&] {
        profiling::ScopedRegion prof_find(profiling::Region::Find);
        // Dispatch the scan on the basis at compile time (Pauli emit-sign kernel + J(G) fold vs the
        // Majorana interleave/hermitian phase). Every other argument — including the fused cos sweep,
        // which scales the same anticommuting set the fold finds — is basis-agnostic.
        double *const sweep_ptr = fused_scale ? fused_scale_coeffs->data() : nullptr;
        auto scan = [&]<bool IsPauli>() {
            return fused_find_and_collect<NumModes, IsPauli>(local_op,
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
        };
        if (basis == Basis::Pauli) {
            return scan.template operator()<true>();
        }
        return scan.template operator()<false>();
    }();

    CosMask cos_all;
    {
        // Per-chunk cosine blocks are disjoint and ascending; chunk-order concat (parallel for
        // large totals) reproduces the serial order exactly.
        for (const auto &block : fused.cos_blocks) {
            cos_all.total_count += block.total_count;
        }
        append_parts_in_order(cos_all.blocks, fused.cos_blocks.size(), [&](size_t c) -> auto & {
            return fused.cos_blocks[c].blocks;
        });
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

    // Recompute metadata rides WITH the layer (in its LayerCore), so it survives every graph transform
    // (slice/union/consume/Schrödinger-prepend). scaled_count is the POST-insert operator size (after
    // finish() ran this layer's partner inserts): the stored cos is "all anticommuting", so folding the
    // inverted index truncated to scaled_count reproduces it bit-for-bit in both pictures with no stored bitmap.
    // Fused mode returns no LayerCore (the layer is transient), so there is nothing to stamp.
    if (storage != nullptr) {
        storage->generator_words.assign(gen.data(), gen.data() + mpi_detail::kWords<NumModes>);
        storage->scaled_count = static_cast<uint64_t>(local_op.store->size());
    }

    return storage;
}

} // namespace monoprop::detail
