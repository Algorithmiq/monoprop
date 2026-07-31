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
#include <limits>
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

namespace monoprop::detail {

// Every rotation target must be in cos so the gradient reverse-sweep can un-do this layer's cosine
// scaling; only freshly inserted half-terms can be absent (see tests/test_infinite_cutoff.py), and those
// sit in [combined_size, op.size()). Scan cos bits and inserted endpoint bits are disjoint, so only the
// seam word can carry both — bitwise-or that one, append the rest, keeping blocks ascending/disjoint.
template <size_t NumModes>
inline auto append_inserted_endpoints(CosMask &cos_all, size_t combined_size, const MPOperator<NumModes> &op) -> void {
    const size_t cos_lo = combined_size;
    const size_t cos_hi = op.store->size();
    CosineWordBuilder end_b;
    for (size_t idx = cos_lo; idx < cos_hi; ++idx) {
        end_b.push_index(idx);
    }
    CosMask end_words = end_b.finish();
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

// A sink owns the divergent state and supplies the emission surfaces — self-resolve, cross-rank
// resolve/process, deferred self-insert — plus finalize. Each monomorphizes: no run-time fused/graph branch.

// Graph-build sink: accumulates the per-rank PartnerAcc endpoints and assembles a LayerCore at finalize.
// wants_values=false — the scan captures no coeffs and every rotation records only (index, phase).
template <size_t NumModes>
struct GraphSink {
    static constexpr bool wants_values = false;
    static constexpr size_t kStride = kQueryWords<NumModes>;
    using Response = TermIndex;
    static auto init_response() -> Response { return std::numeric_limits<TermIndex>::max(); }

    size_t R;
    size_t my_rank;
    std::vector<PartnerAcc> acc;
    size_t def_in_base_ = 0; // deferred self-miss bases into acc[my_rank]
    size_t def_out_base_ = 0;
    std::vector<size_t> in_base_; // cross-rank per-rank base into acc[s].in_entries (set in prepare)

    GraphSink(size_t R_, size_t my_rank_) : R(R_), my_rank(my_rank_), acc(R_) {}

    auto self_hit(size_t src, size_t found, int phase, double /*v_src*/) -> void {
        acc[my_rank].in_entries.push_back({found, phase});
        acc[my_rank].out_entries.push_back({src, phase});
    }
    auto prepare_deferred(size_t n_miss) -> void {
        def_in_base_ = acc[my_rank].in_entries.size();
        def_out_base_ = acc[my_rank].out_entries.size();
        acc[my_rank].in_entries.resize(def_in_base_ + n_miss);
        acc[my_rank].out_entries.resize(def_out_base_ + n_miss);
    }
    auto emit_deferred(size_t k, size_t idx, size_t src, int phase, double /*v_src*/) -> void {
        acc[my_rank].in_entries[def_in_base_ + k] = {idx, phase};
        acc[my_rank].out_entries[def_out_base_ + k] = {src, phase};
    }

    // Cross-rank (R>1). Send buffer = the plain query stream (no value fusion). The exchange is positional:
    // responses[s][q] must answer incoming[s][q], one resolution per query.
    auto send_buffer(std::vector<VecZ> &queries,
                     std::vector<std::vector<double>> & /*vals*/,
                     std::vector<VecZ> & /*scratch*/) -> std::vector<VecZ> & {
        return queries;
    }
    auto prepare(const IncomingProbe<NumModes> & /*pr*/,
                 size_t rank_count,
                 MPOperator<NumModes> & /*op*/,
                 const std::vector<std::vector<Response>> &responses) -> void {
        in_base_.assign(rank_count, 0);
        for (size_t s = 0; s < rank_count; ++s) {
            in_base_[s] = acc[s].in_entries.size();
            acc[s].in_entries.resize(in_base_[s] + responses[s].size());
        }
    }
    auto on_resolved(size_t g,
                     size_t s,
                     size_t q,
                     size_t ip,
                     const IncomingProbe<NumModes> &pr,
                     const std::vector<VecZ> & /*incoming*/) -> Response {
        acc[s].in_entries[in_base_[s] + q] = {ip, pr.phase_of[g]};
        return static_cast<TermIndex>(ip);
    }
    auto process_reserve(const std::vector<std::vector<Response>> & /*inc_r*/,
                         size_t /*rank_count*/,
                         size_t /*my_rank*/) -> void {}
    auto on_response_block(size_t r,
                           const std::vector<Response> &resp,
                           const std::vector<size_t> &srcs,
                           const VecZ &qbuf) -> void {
        auto &out = acc[r].out_entries;
        const size_t base = out.size();
        const size_t nq = resp.size();
        out.resize(base + nq);
        for (size_t q = 0; q < nq; ++q) {
            assert(resp[q] != std::numeric_limits<TermIndex>::max() && "resolver must insert absent cross-rank terms");
            out[base + q] = {srcs[q], query_phase<NumModes>(qbuf, q)};
        }
    }

    // Drains the per-rank accumulators into the LayerCore's sin_send/sin_recv lists (layout derivation:
    // see cross_rank_sin_recv_index). cos covers all anticommuting indices, endpoints included, since the
    // sin_recv apply only adds the sine term.
    auto finalize(CosMask &&cos_all, CosMask *out_cos, size_t combined_size, MPOperator<NumModes> &op)
        -> std::shared_ptr<LayerCore> {
        std::vector<CrossRankPartnerData> partners(R);
        for (size_t r = 0; r < R; ++r) {
            const auto &a = acc[r];
            auto &p = partners[r];
            const size_t P = a.in_entries.size();
            const size_t Q = a.out_entries.size();
            if (P + Q == 0) {
                continue;
            }
            p.in_count = P; // boundary for deriving the sin_recv index list from sin_send (not stored)
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
        if (out_cos != nullptr) {
            append_inserted_endpoints<NumModes>(cos_all, combined_size, op);
            *out_cos = std::move(cos_all);
        }
        return build_layer_storage_unified(std::move(partners), my_rank);
    }
};

// Fused ContractImmediately sink: applies each resolved rotation directly to op_coeffs via the
// FusedContract record streams (no LayerCore — finalize returns nullptr). wants_values=true: the scan
// captures the signed pre-cos v_src, and resolve reads v_tgt from op_coeffs (·inv_cos under the cos sweep).
template <size_t NumModes>
struct ContractSink {
    static constexpr bool wants_values = true;
    static constexpr size_t kStride = kQueryWordsFused<NumModes>;
    using Response = double;
    static auto init_response() -> Response { return 0.0; }

    size_t R;
    size_t my_rank;
    FusedContract &fc;
    const VecD &op_coeffs; // the very array the scan read, not a copy
    bool fused_scale;      // fused cos sweep active: hit v_tgt recovered as stored·inv_cos
    double inv_cos;
    bool schrodinger;                 // fresh cross-rank miss coeff: 0 (Heisenberg) vs state-scored (Schrödinger)
    Basis basis;                      // Pauli vs Majorana state scoring of fresh cross-rank Schrödinger misses
    size_t def_base_ = 0;             // deferred self-insert base into fc.inserts
    size_t cross_base_ = 0;           // cross-rank resolver-half base into fc.cross_half
    Monomial<NumModes> state_mask_{}; // Schrödinger fresh-insert scoring mask (empty in Heisenberg)

    // No constructor on purpose: as an aggregate the call site names each field, so the two adjacent
    // bools cannot be swapped silently. GraphSink keeps its ctor because it sizes `acc` from R.

    // Self-resolve hit (both endpoints local). always_inline: called once per surviving rotation in the
    // R=1 hot loop, where a real call is a measurable regression on the Pauli benches.
    [[gnu::always_inline]] auto self_hit(size_t src, size_t found, int phase, double v_src) -> void {
        const double v_tgt = fused_scale ? op_coeffs[found] * inv_cos : op_coeffs[found];
        fc.hits.push_back(RotationRec{src, found, v_src, v_tgt, static_cast<int32_t>(phase)});
    }
    // Deferred self-miss insert: v_tgt filled later (after op_coeffs is extended by the apply).
    auto prepare_deferred(size_t n_miss) -> void {
        def_base_ = fc.inserts.size();
        fc.inserts.resize(def_base_ + n_miss);
    }
    [[gnu::always_inline]] auto emit_deferred(size_t k, size_t idx, size_t src, int phase, double v_src) -> void {
        fc.inserts[def_base_ + k] = RotationRec{src, idx, v_src, /*v_tgt=*/0.0, static_cast<int32_t>(phase)};
    }

    // Cross-rank (R>1). Send buffer = queries interleaved with their v_src stream into `scratch`
    // (combined_qv_), so one alltoallv carries query + value.
    auto send_buffer(std::vector<VecZ> &queries, std::vector<std::vector<double>> &vals, std::vector<VecZ> &scratch)
        -> std::vector<VecZ> & {
        scratch.resize(queries.size());
        for (size_t r = 0; r < queries.size(); ++r) {
            build_fused_query_value<NumModes>(queries[r], vals[r], scratch[r]);
        }
        return scratch;
    }
    auto prepare(const IncomingProbe<NumModes> &pr,
                 size_t /*rank_count*/,
                 MPOperator<NumModes> &op,
                 const std::vector<std::vector<Response>> & /*responses*/) -> void {
        state_mask_ = schrodinger ? initial_state_mask<NumModes>(op.initial_state) : Monomial<NumModes>{};
        cross_base_ = fc.cross_half.size();
        fc.cross_half.resize(cross_base_ + pr.nq_total);
    }
    auto on_resolved(size_t g,
                     size_t s,
                     size_t q,
                     size_t ip,
                     const IncomingProbe<NumModes> &pr,
                     const std::vector<VecZ> &incoming) -> Response {
        double v_tgt;
        if (ip < pr.base) {
            v_tgt = fused_scale ? op_coeffs[ip] * inv_cos : op_coeffs[ip];
        }
        else if (schrodinger) {
            v_tgt =
                is_paired<NumModes>(pr.mono[g]) ? algebra_state_phase<NumModes>(basis, pr.mono[g], state_mask_) : 0.0;
        }
        else {
            v_tgt = 0.0; // Heisenberg fresh insert
        }
        fc.cross_half[cross_base_ + g] = HalfRotationRec{ip,
                                                         query_value<NumModes>(incoming[s], q),
                                                         static_cast<int32_t>(pr.phase_of[g]),
                                                         /*is_insert=*/ip >= pr.base};
        return v_tgt;
    }
    auto process_reserve(const std::vector<std::vector<Response>> &inc_r, size_t rank_count, size_t my_rank_) -> void {
        size_t incoming = 0;
        for (size_t r = 0; r < rank_count; ++r) {
            if (r != my_rank_) {
                incoming += inc_r[r].size();
            }
        }
        fc.cross_half.reserve(fc.cross_half.size() + incoming);
    }
    // A querier half always writes a pre-gate term the cos sweep already covered ⇒ is_insert=false.
    auto on_response_block(size_t /*r*/,
                           const std::vector<Response> &rval,
                           const std::vector<size_t> &srcs,
                           const VecZ &qbuf) -> void {
        const size_t nq = rval.size();
        for (size_t q = 0; q < nq; ++q) {
            const auto nphase = static_cast<int32_t>(-query_phase<NumModes>(qbuf, q));
            fc.cross_half.push_back(HalfRotationRec{srcs[q], rval[q], nphase, /*is_insert=*/false});
        }
    }

    // No LayerCore in the fused path → nullptr. Two-pass fused (k>0 / cos==0 fallback) appends inserted
    // endpoints so the immediate cos scale covers them; the fused cos sweep covers them in-place instead.
    auto finalize(CosMask &&cos_all, CosMask *out_cos, size_t combined_size, MPOperator<NumModes> &op)
        -> std::shared_ptr<LayerCore> {
        if (out_cos != nullptr && !fused_scale) {
            append_inserted_endpoints<NumModes>(cos_all, combined_size, op);
            *out_cos = std::move(cos_all);
        }
        return nullptr;
    }
};

// Owns build_layer's machinery over a compile-time Sink policy. combined_size = the pre-layer operator size.
template <size_t NumModes, class Sink>
struct LayerBuildEngine {
    struct DeferredSelfMiss {
        Monomial<NumModes> mono;
        size_t src;
        int phase;
        double v_src = 0.0; // ContractSink only: op_pre[src] captured at scan emit; 0 for GraphSink
    };
    MPOperator<NumModes> &local_op; // scanned, looked up, and grown by the inserts
    mpi::Comm comm;
    size_t R;
    size_t my_rank;
    // Follower-matched set over [0, combined_size), caller-owned (see MatchedEpochSet). Distinct leaders
    // → distinct found, so each slot is marked once.
    MatchedEpochSet &matched;
    size_t combined_size;
    std::vector<VecZ> queries_r;
    std::vector<std::vector<size_t>> src_idx_r;
    std::vector<DeferredSelfMiss> deferred_self_misses;
    // Scan-captured v_src per query (ContractSink only via Sink::wants_values; empty for GraphSink).
    std::vector<std::vector<double>> src_val_r;
    // Fused query+value send scratch (ContractSink, R>1): shared by a gate's two exchange passes.
    std::vector<VecZ> combined_qv_;
    Sink sink;

    LayerBuildEngine(MPOperator<NumModes> &local_op_,
                     mpi::Comm comm_,
                     size_t R_,
                     size_t my_rank_,
                     MatchedEpochSet &matched_scratch,
                     size_t combined_size_,
                     Sink &&sink_)
        : local_op(local_op_),
          comm(comm_),
          R(R_),
          my_rank(my_rank_),
          matched(matched_scratch),
          combined_size(combined_size_),
          queries_r(R_),
          src_idx_r(R_),
          sink(std::move(sink_)) {
        matched.begin_gate(combined_size);
    }

    // Resolve this rank's own query stream inline, then clear it so the alltoallv never sends to self.
    auto resolve_self_queries(bool is_leader_pass) -> void {
        VecZ &lq = queries_r[my_rank];
        std::vector<size_t> &ls = src_idx_r[my_rank];
        std::vector<double> *lv = nullptr;
        if constexpr (Sink::wants_values) {
            lv = &src_val_r[my_rank];
        }
        const size_t nq = lq.empty() ? 0 : lq.size() / kQueryWords<NumModes>;
        resolve_range_(lq, ls, lv, 0, nq, is_leader_pass);
        lq.clear();
        ls.clear();
        if constexpr (Sink::wants_values) {
            src_val_r[my_rank].clear();
        }
    }

    // One partner-resolution pass over the given query streams, which it takes ownership of. Round 1
    // carries the queries (the sink may fuse the v_src stream into them) and the resolver inserts absent
    // partners in that same round; round 2 returns the answers. Taking the streams here rather than having
    // the caller assign the members first is what makes the two-pass protocol unmissable — the follower
    // pass must also drop the queries a leader already matched, and that only holds once the leader pass
    // has run.
    auto run_exchange(bool is_leader_pass,
                      std::vector<VecZ> &&queries,
                      std::vector<std::vector<size_t>> &&src_idx,
                      std::vector<std::vector<double>> &&src_val) -> void {
        queries_r = std::move(queries);
        src_idx_r = std::move(src_idx);
        // src_val is empty unless Sink::wants_values, so the move is a no-op under GraphSink.
        src_val_r = std::move(src_val);
        if (!is_leader_pass && R > 1) {
            drop_matched_cross_rank_followers();
        }
        resolve_self_queries(is_leader_pass);
        if (R <= 1) {
            return;
        }
        std::vector<VecZ> &send = sink.send_buffer(queries_r, src_val_r, combined_qv_);
        std::vector<std::vector<size_t>> inc_q;
        mpi::begin_alltoallv(send, comm).wait_into(inc_q);
        auto resp = resolve_incoming<NumModes>(inc_q, local_op, R, is_leader_pass, matched, combined_size, sink);
        std::vector<int> resp_recv = response_recv_counts();
        std::vector<std::vector<typename Sink::Response>> inc_r;
        // The answers travel the query exchange's legs backwards, one per query, so the hybrid transport
        // reuses that exchange's offset tables; nothing may collectively intervene between the two calls.
        // Sink::kStride is the query leg's words per query, the ratio between the two legs' counts.
        mpi::begin_alltoallv(resp,
                             comm,
                             /*skip_self=*/false,
                             &resp_recv,
                             /*reverse_of_previous=*/true,
                             /*forward_stride=*/static_cast<int>(Sink::kStride))
            .wait_into(inc_r);
        process_responses<NumModes>(inc_r, src_idx_r, queries_r, R, my_rank, sink);
    }

    // Followers a leader already matched must not be re-resolved over the wire, so compact them out.
    auto drop_matched_cross_rank_followers() -> void {
        constexpr size_t W = kQueryWords<NumModes>;
        for (size_t r = 0; r < R; ++r) {
            if (r == my_rank) {
                continue;
            }
            VecZ &q = queries_r[r];
            std::vector<size_t> &s = src_idx_r[r];
            // Fused: the v_src stream is parallel to the query/source streams, so compact it in lockstep.
            std::vector<double> *v = nullptr;
            if constexpr (Sink::wants_values) {
                v = &src_val_r[r];
            }
            const size_t nq = s.size();
            size_t kept = 0;
            for (size_t k = 0; k < nq; ++k) {
                if (matched.is_marked(s[k])) {
                    continue;
                }
                if (kept != k) {
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

    // Sub-step of finish() — do not call directly. Precondition: call only after both resolve passes
    // complete, else the base+k ↔ record-slot assignment and per-miss distinctness break. Deferred self
    // misses are pairwise-distinct (mono = source⊕G, ⊕G injective) and still absent, so miss k gets
    // base+k in leader-then-follower order. See insert_absent_terms.
    auto insert_deferred_self_misses() -> void {
        const size_t n_miss = deferred_self_misses.size();
        if (n_miss == 0) {
            return;
        }
        auto key_at = [&](size_t k) -> const Monomial<NumModes> & { return deferred_self_misses[k].mono; };
        sink.prepare_deferred(n_miss);
        insert_absent_terms<NumModes>(local_op, n_miss, key_at, [&](size_t k, size_t base) {
            const auto &m = deferred_self_misses[k];
            assign_row<NumModes>(*local_op.store, base + k, m.mono);
            sink.emit_deferred(k, base + k, m.src, m.phase, m.v_src);
        });
    }

    auto finish(CosMask &&cos_all, CosMask *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        insert_deferred_self_misses();
        return sink.finalize(std::move(cos_all), out_cos, combined_size, local_op);
    }

private:
    // Response counts are the transpose of the query counts (one answer per query), so passing them as
    // known_recv_counts skips the response count-Alltoall round.
    auto response_recv_counts() const -> std::vector<int> {
        std::vector<int> counts(R);
        for (size_t r = 0; r < R; ++r) {
            counts[r] = static_cast<int>(queries_r[r].size() / kQueryWords<NumModes>);
        }
        return counts;
    }

    // Batched self-resolve over the index's group-prefetch find_batch; hits/misses are emitted to the sink
    // in query order. `lv` is the per-query v_src array parallel to `ls` (read only when Sink::wants_values).
    static constexpr size_t kResolveBatch = 64;
    auto resolve_range_(VecZ &lq,
                        std::vector<size_t> &ls,
                        [[maybe_unused]] std::vector<double> *lv,
                        size_t lo,
                        size_t hi,
                        bool is_leader_pass) -> void {
        const size_t op_size = local_op.store->size();
        std::array<Monomial<NumModes>, kResolveBatch> keys;
        std::array<int, kResolveBatch> phases;
        std::array<size_t, kResolveBatch> srcs;
        std::array<double, kResolveBatch> vals;
        std::array<size_t, kResolveBatch> found;
        size_t q = lo;
        while (q < hi) {
            size_t m = 0;
            for (; q < hi && m < kResolveBatch; ++q) {
                const size_t src = ls[q];
                if (!is_leader_pass && matched.is_marked(src)) {
                    continue; // follower already matched by a leader → not an independent rotation
                }
                query_read<NumModes>(lq, q, keys[m], phases[m]);
                srcs[m] = src;
                if constexpr (Sink::wants_values) {
                    vals[m] = (*lv)[q];
                }
                ++m;
            }
            if (m == 0) {
                break;
            }
            local_op.store->find_batch(keys.data(), m, found.data());
            for (size_t j = 0; j < m; ++j) {
                double v_src = 0.0;
                if constexpr (Sink::wants_values) {
                    v_src = vals[j];
                }
                // kNotFound == kMissingIndex == size_t max, so one bound check covers both.
                if (found[j] < op_size) {
                    if (is_leader_pass) {
                        matched.mark(found[j]);
                    }
                    sink.self_hit(srcs[j], found[j], phases[j], v_src);
                }
                else {
                    deferred_self_misses.push_back({keys[j], srcs[j], phases[j], v_src});
                }
            }
        }
    }
};

// Primary-path layer builder: one fused scan, then two resolve passes into the chosen sink. See LayerBuilder.h.
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
    // Fused contraction runs at all rank counts (R>1 via the cross-rank half-rotation exchange).
    const bool use_fused = (fused_contract != nullptr);
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs ? local_coeffs->get() : empty_coeffs();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // Fused cos sweep: fold the per-gate cosine scale into the scan's own coefficient pass. k==0 only (a
    // popcount>k hit is outside the per-index cos set, so 1/cos recovery would be wrong) and cos!=0 (else
    // recovery is impossible; two-pass fallback). cos is even, so the sweep's cos(2·build_angle) matches
    // the apply's cos(2·apply_angle) bit-for-bit.
    const double cos_build = (use_fused && param.has_value()) ? std::cos(2.0 * param.value()) : 1.0;
    const bool fused_scale =
        use_fused && only_rotate_len_k == 0 && fused_scale_coeffs != nullptr && param.has_value() && cos_build != 0.0;
    // build_layer is the single authority for this decision; the fused caller must drive its apply from it.
    if (fused_scale_out != nullptr) {
        *fused_scale_out = fused_scale;
    }
    assert(fused_scale_coeffs == nullptr || (local_coeffs && &local_coeffs->get() == fused_scale_coeffs));

    FusedScanResult fused = [&] {
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

    auto run = [&]<class Sink>(Sink sink) -> std::shared_ptr<LayerCore> {
        LayerBuildEngine<NumModes, Sink> eng(local_op,
                                             comm,
                                             R,
                                             my_rank,
                                             matched_scratch,
                                             /*combined_size=*/local_op.store->size(),
                                             std::move(sink));
        eng.run_exchange(/*is_leader_pass=*/true,
                         std::move(fused.leader_queries),
                         std::move(fused.leader_src),
                         std::move(fused.leader_val));
        eng.run_exchange(/*is_leader_pass=*/false,
                         std::move(fused.follower_queries),
                         std::move(fused.follower_src),
                         std::move(fused.follower_val));

        return eng.finish(std::move(cos_all), out_cos);
    };

    std::shared_ptr<LayerCore> storage;
    if (use_fused) {
        const double inv_cos = fused_scale ? 1.0 / cos_build : 1.0; // pre-cos recovery factor for hit v_tgt
        storage = run(ContractSink<NumModes>{.R = R,
                                             .my_rank = my_rank,
                                             .fc = *fused_contract,
                                             .op_coeffs = coeffs,
                                             .fused_scale = fused_scale,
                                             .inv_cos = inv_cos,
                                             .schrodinger = schrodinger,
                                             .basis = basis});
    }
    else {
        storage = run(GraphSink<NumModes>{R, my_rank});
    }

    // Recompute metadata rides with the layer so it survives every graph transform. scaled_count is the
    // post-insert operator size: the fold truncated to it reproduces the "all anticommuting" cos
    // bit-for-bit with no stored bitmap. Fused mode has no LayerCore to stamp.
    if (storage != nullptr) {
        storage->generator_words.assign(gen.data(), gen.data() + mpi_detail::kWords<NumModes>);
        storage->scaled_count = static_cast<uint64_t>(local_op.store->size());
    }

    return storage;
}

} // namespace monoprop::detail
