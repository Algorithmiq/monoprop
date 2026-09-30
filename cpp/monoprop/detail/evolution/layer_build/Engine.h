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
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"
#include "monoprop/detail/parallel/Options.h"

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
    // on_resolved for distinct queries may run concurrently after prepare(): each writes only its own
    // acc[slot].in_entries element, which prepare() sized. Certifies nothing about the other methods.
    static constexpr bool parallel_resolve = true;
    [[nodiscard]] auto incoming_form() const -> QueryForm { return QueryForm::Plain; }
    [[nodiscard]] auto querier_form() const -> QueryForm { return QueryForm::Plain; }
    using Response = TermIndex;
    static auto init_response() -> Response { return std::numeric_limits<TermIndex>::max(); }

    size_t R;
    size_t my_rank;
    std::vector<PartnerAcc> acc;
    size_t def_in_base_ = 0; // deferred self-miss bases into acc[my_rank]
    size_t def_out_base_ = 0;
    // Per-window-slot base into acc[slot].in_entries (set in prepare).
    mpi::WindowVec<size_t> in_base_;

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
    auto send_buffer(mpi::WindowVec<VecZ> &queries,
                     mpi::WindowVec<std::vector<double>> & /*vals*/,
                     mpi::WindowVec<VecZ> & /*scratch*/) -> mpi::WindowVec<VecZ> & {
        return queries;
    }
    // `acc` stays flat [P] for build_layer_storage_unified, so window indices map back to slots here.
    auto prepare(const IncomingProbe<NumModes> & /*pr*/,
                 MPOperator<NumModes> & /*op*/,
                 const mpi::WindowVec<std::vector<Response>> &responses) -> void {
        const mpi::SlotWindow w = responses.window();
        in_base_.reset(w);
        for (const auto wi : w.indices()) {
            PartnerAcc &a = acc[w.slot(wi)];
            in_base_[wi] = a.in_entries.size();
            a.in_entries.resize(in_base_[wi] + responses[wi].size());
        }
    }
    auto on_resolved(size_t g,
                     mpi::WindowIndex s,
                     size_t q,
                     size_t ip,
                     const IncomingProbe<NumModes> &pr,
                     typename QueryWire<NumModes>::WireView /*stream*/) -> Response {
        acc[pr.window.slot(s)].in_entries[in_base_[s] + q] = {ip, pr.phase_of[g]};
        return static_cast<TermIndex>(ip);
    }
    template <class Responses>
    auto process_reserve(const mpi::WindowVec<Responses> & /*inc_r*/, size_t /*my_rank*/) -> void {}
    // `resp`, `srcs` and `qbuf` are views of the resolver's published answers and this owner's own sources and
    // plain queries; all three stay unchanged until this returns.
    auto on_response_block(size_t r,
                           std::span<const Response> resp,
                           std::span<const size_t> srcs,
                           typename QueryWire<NumModes>::WireView qbuf) -> void {
        auto &out = acc[r].out_entries;
        const size_t base = out.size();
        const size_t nq = resp.size();
        const QueryForm form = querier_form();
        out.resize(base + nq);
        size_t off = 0;
        for (size_t q = 0; q < nq; ++q) {
            assert(resp[q] != std::numeric_limits<TermIndex>::max() && "resolver must insert absent cross-rank terms");
            out[base + q] = {srcs[q], QueryWire<NumModes>::phase_at(qbuf, off)};
            off = QueryWire<NumModes>::next_off(qbuf, form, off);
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
        return build_layer_storage_unified(partners, my_rank);
    }
};

// Fused ContractImmediately sink: applies each resolved rotation directly to op_coeffs via the
// FusedContract record streams (no LayerCore — finalize returns nullptr). wants_values=true: the scan
// captures the signed pre-cos v_src, and resolve reads v_tgt from op_coeffs (·inv_cos under the cos sweep).
template <size_t NumModes>
struct ContractSink {
    static constexpr bool wants_values = true;
    // on_resolved for distinct queries may run concurrently after prepare() in Heisenberg: each writes only its
    // own cross_half element and reads the pre-insert coefficients. Schrodinger scoring stays serial, see
    // parallel_resolve_enabled(). Certifies nothing about the other methods (on_response_block appends).
    static constexpr bool parallel_resolve = true;
    // This rank receives fused records, but on_response_block is handed its own plain queries_r;
    // reading the wrong form there decodes a neighbouring record's phase, i.e. a coefficient sign flip.
    [[nodiscard]] auto incoming_form() const -> QueryForm { return QueryForm::Fused; }
    [[nodiscard]] auto querier_form() const -> QueryForm { return QueryForm::Plain; }
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

    // Self-resolve hit: both endpoints are local.
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

    //! The Schrodinger fresh-insert arm scores the state through mono_at; it keeps the serial scatter.
    [[nodiscard]] auto parallel_resolve_enabled() const -> bool { return !schrodinger; }

    // Cross-rank (R>1). Send buffer = queries interleaved with their v_src stream into `scratch`
    // (combined_qv_), so one alltoallv carries query + value.
    auto send_buffer(mpi::WindowVec<VecZ> &queries,
                     mpi::WindowVec<std::vector<double>> &vals,
                     mpi::WindowVec<VecZ> &scratch) -> mpi::WindowVec<VecZ> & {
        const mpi::SlotWindow w = queries.window();
        scratch.reset(w);
        for (const auto wi : w.indices()) {
            QueryWire<NumModes>::build_fused(queries[wi], vals[wi], scratch[wi]);
        }
        return scratch;
    }
    auto prepare(const IncomingProbe<NumModes> &pr,
                 MPOperator<NumModes> &op,
                 const mpi::WindowVec<std::vector<Response>> & /*responses*/) -> void {
        state_mask_ = schrodinger ? initial_state_mask<NumModes>(op.initial_state) : Monomial<NumModes>{};
        cross_base_ = fc.cross_half.size();
        fc.cross_half.resize(cross_base_ + pr.nq_total);
    }
    // `stream` is sender s's fused query stream, a view the caller keeps unchanged until the scatter returns.
    auto on_resolved(size_t g,
                     mpi::WindowIndex /*s*/,
                     size_t /*q*/,
                     size_t ip,
                     const IncomingProbe<NumModes> &pr,
                     typename QueryWire<NumModes>::WireView stream) -> Response {
        double v_tgt;
        if (ip < pr.base) {
            v_tgt = fused_scale ? op_coeffs[ip] * inv_cos : op_coeffs[ip];
        }
        else if (schrodinger) {
            v_tgt = pr.is_paired_at(g) ? algebra_state_phase<NumModes>(basis, pr.mono_at(g), state_mask_) : 0.0;
        }
        else {
            v_tgt = 0.0; // Heisenberg fresh insert
        }
        fc.cross_half[cross_base_ + g] = HalfRotationRec{ip,
                                                         QueryWire<NumModes>::value_at(stream, pr.off_of[g]),
                                                         static_cast<int32_t>(pr.phase_of[g]),
                                                         /*is_insert=*/ip >= pr.base};
        return v_tgt;
    }
    template <class Responses>
    auto process_reserve(const mpi::WindowVec<Responses> &inc_r, size_t my_rank_) -> void {
        const mpi::SlotWindow w = inc_r.window();
        size_t incoming = 0;
        for (const auto wi : w.indices()) {
            if (w.slot(wi) != my_rank_) {
                incoming += inc_r[wi].size();
            }
        }
        fc.cross_half.reserve(fc.cross_half.size() + incoming);
    }
    // A querier half always writes a pre-gate term the cos sweep already covered ⇒ is_insert=false.
    auto on_response_block(size_t /*r*/,
                           std::span<const Response> rval,
                           std::span<const size_t> srcs,
                           typename QueryWire<NumModes>::WireView qbuf) -> void {
        const size_t nq = rval.size();
        const QueryForm form = querier_form();
        size_t off = 0;
        for (size_t q = 0; q < nq; ++q) {
            const auto nphase = static_cast<int32_t>(-QueryWire<NumModes>::phase_at(qbuf, off));
            fc.cross_half.push_back(HalfRotationRec{srcs[q], rval[q], nphase, /*is_insert=*/false});
            off = QueryWire<NumModes>::next_off(qbuf, form, off);
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

/*
 * Owns one layer's partner resolution over a compile-time Sink policy for one owner: the flat routing slot
 * `my_rank` out of `R` (a physical rank of the legacy runtime, or rank * T + shard of a sharded one), its store, its
 * matched marks and its sink. combined_size = the pre-layer operator size, the source boundary.
 *
 * Each pass (leaders, then followers) is three owner-local phases with a handoff between them:
 *
 *   prepare_exchange   adopt the traversal's outgoing streams, drop followers a leader already matched, resolve
 *                      same-owner queries against the frozen store (their misses are deferred), and expose the
 *                      outgoing payload.
 *   resolve_published  decode, probe and answer the queries other owners published to this owner, in ascending
 *                      sender order, then insert their misses and publish them in the index.
 *   consume_published  fold the answers to this owner's own queries into its sink, in window and query order.
 *
 * finish() then inserts the deferred same-owner misses (leaders, then followers) and finalizes the sink. The phases
 * perform no MPI, no barrier and no nested worksharing beyond the serial-or-budgeted kernels `options` selects; the
 * caller owns every handoff. run_exchange() is the legacy transport adapter over the same phases.
 *
 * Handoff lifetimes: the payload prepare_exchange returns is read by its destinations until they have resolved it,
 * and consume_published still reads this owner's own sources and plain queries, so neither may change until this
 * owner consumed its answers and every destination resolved the payload. The next prepare_exchange replaces them.
 *
 * `Observer` is the test seam of the threaded kernels (see NoRangeObserver). It may also provide
 * `exchange_prepared(leaders, payload, sources, values)`, `exchange_resolved(leaders, responses)` and
 * `exchange_consumed(leaders, responses)`, which the phases call with what they produced or consumed; production
 * observers provide none of them.
 */
template <size_t NumModes, typename Sink, class Observer = NoRangeObserver>
struct LayerBuildEngine {
    using RowPosT = typename OperatorIndex<NumModes>::PosT;
    using Response = typename Sink::Response; //!< One answer per resolved query.

    // A miss keeps its decoded positions (pos_at indexes deferred_pos_flat_) and the probe's hash.
    struct DeferredSelfMiss {
        size_t pos_at;
        uint32_t k;
        uint32_t hash;
        size_t src;
        int phase;
        double v_src = 0.0; // ContractSink only: op_pre[src] captured at scan emit; 0 for GraphSink
    };
    MPOperator<NumModes> &local_op; // scanned, looked up, and grown by the inserts
    size_t R;                       // flat owners in the routing world
    size_t my_rank;                 // this owner's flat routing slot
    // Follower-matched set over [0, combined_size), caller-owned (see MatchedEpochSet). Distinct leaders
    // → distinct found, so each slot is marked once.
    MatchedEpochSet &matched;
    size_t combined_size;
    // The traversal's window for my_rank: every per-slot array below is sized to it.
    mpi::SlotWindow window;
    mpi::WindowVec<VecZ> queries_r;
    mpi::WindowVec<std::vector<size_t>> src_idx_r;
    std::vector<DeferredSelfMiss> deferred_self_misses;
    // Deferred-miss positions, concatenated in miss order; parallel to deferred_self_misses.
    std::vector<RowPosT> deferred_pos_flat_;
    // This pass's self-owned queries as positions, straight from the scan: never encoded, so the resolve
    // below has nothing to decode. Parallel to src_idx_r's self slot.
    SelfQueryStage<NumModes> self_stage_;
    // Scan-captured v_src per query (ContractSink only via Sink::wants_values; empty for GraphSink).
    mpi::WindowVec<std::vector<double>> src_val_r;
    // Fused query+value send scratch (ContractSink, R>1): shared by a gate's two exchange passes.
    mpi::WindowVec<VecZ> combined_qv_;
    Sink sink;
    // Budget forwarded to the within-owner kernels; a sharded caller passes the serial default.
    parallel::Options options;
    Observer observer;

    LayerBuildEngine(MPOperator<NumModes> &local_op_,
                     size_t R_,
                     size_t my_rank_,
                     MatchedEpochSet &matched_scratch,
                     size_t combined_size_,
                     Sink &&sink_,
                     mpi::SlotWindow window_,
                     parallel::Options options_ = {},
                     const Observer &observer_ = {})
        : local_op(local_op_),
          R(R_),
          my_rank(my_rank_),
          matched(matched_scratch),
          combined_size(combined_size_),
          window(window_),
          sink(std::move(sink_)),
          options(options_),
          observer(observer_) {
        assert(window.stop() <= R && window.count != 0);
        queries_r.reset(window);
        src_idx_r.reset(window);
        matched.begin_gate(combined_size);
    }

    // Resolve this rank's own query stream inline, then clear it so the alltoallv never sends to self.
    // Self is in the window only when this generator's rank shift is zero.
    auto resolve_self_queries(bool is_leader_pass) -> void {
        if (!window.contains(my_rank)) {
            assert(self_stage_.size() == 0 && "a self-owned partner outside this generator's peer window");
            self_stage_.clear();
            return;
        }
        std::vector<size_t> &ls = src_idx_r.at_slot(my_rank);
        std::vector<double> *lv = nullptr;
        if constexpr (Sink::wants_values) {
            lv = &src_val_r.at_slot(my_rank);
        }
        // The scan routes a self-owned partner to the stage, never to the wire buffer.
        resolve_range_(ls, lv, is_leader_pass);
        self_stage_.clear();
        ls.clear();
        if constexpr (Sink::wants_values) {
            src_val_r.at_slot(my_rank).clear();
        }
    }

    /*!
     * \brief Adopt one pass's outgoing streams, resolve its same-owner queries and expose the outgoing payload.
     *
     * Taking the streams here rather than having the caller assign the members first is what makes the two-pass
     * protocol unmissable: the follower pass must drop the queries a leader already matched, which holds only once
     * this owner has resolved every leader query addressed to it. Follower filtering is stable, so kept queries keep
     * their order. Same-owner queries are probed against the frozen store; their misses are deferred to finish().
     *
     * \param leaders    Whether this is the leader pass.
     * \param queries    Plain query streams per window slot, from the traversal; the self slot is empty.
     * \param sources    Source rows parallel to `queries`, the self slot parallel to `self_stage`.
     * \param values     Captured source values parallel to `sources` (ContractSink), or empty.
     * \param self_stage Same-owner queries as unencoded positions.
     * \return The payload to publish, one block per destination slot: the plain queries (GraphSink) or the fused
     *         query/value stream (ContractSink). It stays valid and unchanged until the next prepare_exchange()
     *         or the engine's destruction; the self slot is empty. With a single owner nothing is published.
     */
    auto prepare_exchange(bool leaders,
                          mpi::WindowVec<VecZ> &&queries,
                          mpi::WindowVec<std::vector<size_t>> &&sources,
                          mpi::WindowVec<std::vector<double>> &&values,
                          SelfQueryStage<NumModes> &&self_stage) -> const mpi::WindowVec<VecZ> & {
        assert(queries.window() == window && sources.window() == window);
        assert(values.size() == 0 || values.window() == window); // empty under GraphSink
        leaders_pass_ = leaders;
        queries_r = std::move(queries);
        src_idx_r = std::move(sources);
        self_stage_ = std::move(self_stage);
        // values is empty unless Sink::wants_values, so the move is a no-op under GraphSink.
        src_val_r = std::move(values);
        if (!leaders && R > 1) {
            drop_matched_cross_rank_followers();
        }
        resolve_self_queries(leaders);
        const mpi::WindowVec<VecZ> *payload = &queries_r;
        if (R > 1) {
            payload = &sink.send_buffer(queries_r, src_val_r, combined_qv_);
        }
        if constexpr (requires { observer.exchange_prepared(leaders, *payload, src_idx_r, src_val_r); }) {
            observer.exchange_prepared(leaders, *payload, src_idx_r, src_val_r);
        }
        return *payload;
    }

    /*!
     * \brief Resolve the queries other owners published to this owner, then insert and publish its misses.
     *
     * Senders are consumed in ascending window order, each in its own query order, so row IDs do not depend on when a
     * sender produced its block. Every stream is validated before any is decoded; probes read the frozen store; the
     * scatter reads pre-insert coefficients; misses are then inserted and indexed on this owner only.
     *
     * \param incoming One read-only view per window slot of this owner's window: the block the slot's owner
     *                 published for this owner. The viewed buffers must stay unchanged until this returns.
     * \param leaders  Whether this is the leader pass; leader hits mark their target rows as matched.
     * \return One answer per received query, per sender slot, in query order.
     * \throws std::invalid_argument if `incoming` does not cover this owner's window, before anything is read.
     * \throws As resolve_incoming(); a failure after insertion began leaves the store partly grown.
     */
    auto resolve_published(const mpi::WindowVec<std::span<const size_t>> &incoming, bool leaders)
        -> mpi::WindowVec<std::vector<Response>> {
        if (incoming.window() != window) {
            throw std::invalid_argument(std::format("LayerBuildEngine::resolve_published: incoming slots [{}, {}) do "
                                                    "not match the owner's window [{}, {})",
                                                    incoming.window().base,
                                                    incoming.window().stop(),
                                                    window.base,
                                                    window.stop()));
        }
        auto responses =
            resolve_incoming<NumModes>(incoming, local_op, leaders, matched, combined_size, sink, options, observer);
        if constexpr (requires { observer.exchange_resolved(leaders, responses); }) {
            observer.exchange_resolved(leaders, responses);
        }
        return responses;
    }

    /*!
     * \brief Fold the answers to this owner's published queries into its sink.
     *
     * Reads this owner's own sources and plain queries, which prepare_exchange() retained, so each answer meets the
     * query it answers. The self slot is skipped: prepare_exchange() resolved it.
     *
     * \param responses One read-only view per window slot: the answers the slot's owner published for this owner's
     *                  queries, in query order. The viewed buffers must stay unchanged until this returns.
     * \throws std::invalid_argument if `responses` does not cover this owner's window, and std::length_error if a
     *         block does not hold exactly one answer per query; both before any record is written.
     */
    auto consume_published(const mpi::WindowVec<std::span<const Response>> &responses) -> void {
        if (responses.window() != window) {
            throw std::invalid_argument(std::format("LayerBuildEngine::consume_published: response slots [{}, {}) do "
                                                    "not match the owner's window [{}, {})",
                                                    responses.window().base,
                                                    responses.window().stop(),
                                                    window.base,
                                                    window.stop()));
        }
        for (const auto wi : window.indices()) {
            if (window.slot(wi) != my_rank && responses[wi].size() != src_idx_r[wi].size()) {
                throw std::length_error(std::format("LayerBuildEngine::consume_published: slot {} answered {} of {} "
                                                    "queries",
                                                    window.slot(wi),
                                                    responses[wi].size(),
                                                    src_idx_r[wi].size()));
            }
        }
        if constexpr (requires { observer.exchange_consumed(leaders_pass_, responses); }) {
            observer.exchange_consumed(leaders_pass_, responses);
        }
        process_responses<NumModes>(responses, src_idx_r, queries_r, my_rank, sink);
    }

    /*!
     * \brief Legacy transport adapter: one pass over the same phases, with MPI or in-process communicator rounds.
     *
     * Round 1 carries the payload, and the resolver inserts absent partners in that same round; round 2 returns the
     * answers. Each round completes inside its handle's lifetime and under the distributed guard, so a failure after
     * posting never reaches a handle's draining destructor while a peer is still inside the round.
     *
     * \param comm The communicator whose flat slots are the routing owners.
     * \param plan The generator's peer plan; its window is this engine's window.
     */
    auto run_exchange(mpi::Comm comm,
                      mpi::PeerPlan plan,
                      bool is_leader_pass,
                      mpi::WindowVec<VecZ> &&queries,
                      mpi::WindowVec<std::vector<size_t>> &&src_idx,
                      mpi::WindowVec<std::vector<double>> &&src_val,
                      SelfQueryStage<NumModes> &&self_stage) -> void {
        const mpi::WindowVec<VecZ> &send = prepare_exchange(is_leader_pass,
                                                            std::move(queries),
                                                            std::move(src_idx),
                                                            std::move(src_val),
                                                            std::move(self_stage));
        if (R <= 1) {
            return;
        }
        mpi::WindowVec<VecZ> inc_q;
        auto query_round = mpi::begin_alltoallv(send, comm, /*skip_self=*/false, /*known_recv_counts=*/nullptr, plan);
        mpi::guard_distributed(comm, [&] { query_round.wait_into(inc_q); });
        // The peers may already be inside the response round, so a failure here (a malformed stream, a
        // decode/probe/scatter worker, publication) aborts rather than unwinds.
        auto resp =
            mpi::guard_distributed(comm, [&] { return resolve_published(mpi::views_of(inc_q), is_leader_pass); });
        std::vector<int> resp_recv = response_recv_counts();
        mpi::WindowVec<std::vector<Response>> inc_r;
        // Answers retrace the queries over an XOR involution, so the same plan holds.
        auto response_round = mpi::begin_alltoallv(resp, comm, /*skip_self=*/false, &resp_recv, plan);
        mpi::guard_distributed(comm, [&] { response_round.wait_into(inc_r); });
        consume_published(mpi::views_of(inc_r));
    }

    // Followers a leader already matched must not be re-resolved over the wire, so compact them out.
    auto drop_matched_cross_rank_followers() -> void {
        using QW = QueryWire<NumModes>;
        const QueryForm form = sink.querier_form();
        for (const auto wi : window.indices()) {
            if (window.slot(wi) == my_rank) {
                continue;
            }
            VecZ &q = queries_r[wi];
            std::vector<size_t> &s = src_idx_r[wi];
            // Fused: the v_src stream is parallel to the query/source streams, so compact it in lockstep.
            std::vector<double> *v = nullptr;
            if constexpr (Sink::wants_values) {
                v = &src_val_r[wi];
            }
            const size_t nq = s.size();
            size_t kept = 0;
            // Two cursors, since a dropped query has no fixed width; order is the accumulation order.
            size_t src_off = 0;
            size_t dst_off = 0;
            for (size_t qi = 0; qi < nq; ++qi) {
                const size_t next = QW::next_off(q, form, src_off);
                if (!matched.is_marked(s[qi])) {
                    dst_off += QW::move_query(q, form, src_off, dst_off);
                    s[kept] = s[qi];
                    if (v != nullptr) {
                        (*v)[kept] = (*v)[qi];
                    }
                    ++kept;
                }
                src_off = next;
            }
            q.resize(dst_off);
            s.resize(kept);
            if (v != nullptr) {
                v->resize(kept);
            }
        }
    }

    // Sub-step of finish() — call only after both resolve passes complete, or the base+k assignment and
    // per-miss distinctness break. Misses are pairwise-distinct (mono = source⊕G, ⊕G injective), so miss
    // k gets base+k in leader-then-follower order.
    auto insert_deferred_self_misses() -> void {
        const size_t n_miss = deferred_self_misses.size();
        if (n_miss == 0) {
            return;
        }
        sink.prepare_deferred(n_miss);
        // Grow the rows, write each miss's positions, insert; same base+k ordering as insert_absent_terms.
        // Kept as the dense reference sparse_resolve_tests.cpp differentially tests this path against.
        const size_t base = local_op.store->grow_rows_geometric(n_miss);
        for (size_t k = 0; k < n_miss; ++k) {
            const auto &m = deferred_self_misses[k];
            local_op.store->set_positions(base + k,
                                          std::span<const RowPosT>(deferred_pos_flat_).subspan(m.pos_at, m.k));
            sink.emit_deferred(k, base + k, m.src, m.phase, m.v_src);
        }
        local_op.store
            ->bulk_insert_hashed(n_miss, base, [&](size_t j) { return deferred_self_misses[j].hash; }, options);
        local_op.reindex_after_growth(base, n_miss);
    }

    auto finish(CosMask &&cos_all, CosMask *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        insert_deferred_self_misses();
        return sink.finalize(std::move(cos_all), out_cos, combined_size, local_op);
    }

private:
    // Response counts are the transpose of the query counts (one answer per query), so passing them as
    // known_recv_counts skips the response count-Alltoall round.
    // Flat [P], as begin_alltoallv's known_recv_counts is; only the window's slots are non-zero.
    auto response_recv_counts() const -> std::vector<int> {
        std::vector<int> counts(R, 0);
        for (const auto wi : window.indices()) {
            // One response per query, and src_idx_r's block holds one source per query: no walk, no division.
            counts[window.slot(wi)] = static_cast<int>(src_idx_r[wi].size());
        }
        return counts;
    }

    bool leaders_pass_ = true; // the pass the retained streams belong to, as prepare_exchange() adopted them

    // Self-resolve scratch for one window of at most kSelfProbeWindow queries, reused across windows and both
    // passes: gathered offsets (absolute into the stage), counts, results, hashes and the emission payload.
    std::vector<size_t> self_pos_off_;
    std::vector<uint32_t> self_k_of_;
    std::vector<size_t> self_found_;
    std::vector<uint32_t> self_hash_;
    std::vector<int> self_phase_;
    std::vector<size_t> self_src_;
    std::vector<double> self_val_; // ContractSink only

    // Self-resolve in bounded windows. For each window, on the caller: stable follower filtering (leaders skip
    // it) and a gather into the scratch; then probe_frozen_positions (KernelRange::self_probe) on workers over
    // at most 16 blocks; then, on the caller, hit/miss emission in original order. Nothing is inserted between
    // windows or passes: misses keep their positions and hashes for insert_deferred_self_misses. `lv` is the
    // per-query v_src array parallel to `ls` (read only when Sink::wants_values).
    auto resolve_range_(std::vector<size_t> &ls, [[maybe_unused]] std::vector<double> *lv, bool is_leader_pass)
        -> void {
        const size_t op_size = local_op.store->size();
        const size_t hi = self_stage_.size();
        const size_t cap = std::min(hi, kSelfProbeWindow);
        if (self_pos_off_.size() < cap) {
            self_pos_off_.resize(cap);
            self_k_of_.resize(cap);
            self_found_.resize(cap);
            self_hash_.resize(cap);
            self_phase_.resize(cap);
            self_src_.resize(cap);
            if constexpr (Sink::wants_values) {
                self_val_.resize(cap);
            }
        }
        const std::span<const RowPosT> stage_pos(self_stage_.pos_flat.data(), self_stage_.positions());
        size_t q = 0;
        while (q < hi) {
            size_t m = 0;
            for (; q < hi && m < kSelfProbeWindow; ++q) {
                const size_t src = ls[q];
                if (!is_leader_pass && matched.is_marked(src)) {
                    continue; // follower already matched by a leader → not an independent rotation
                }
                self_pos_off_[m] = self_stage_.pos_off[q];
                self_k_of_[m] = self_stage_.k_of[q];
                self_phase_[m] = self_stage_.phase_of[q];
                self_src_[m] = src;
                if constexpr (Sink::wants_values) {
                    self_val_[m] = (*lv)[q];
                }
                ++m;
            }
            if (m == 0) {
                break;
            }
            // The hashes come back because a miss needs one at insert, folded from these same positions.
            probe_frozen_positions<NumModes>(*local_op.store,
                                             stage_pos,
                                             std::span<const size_t>(self_pos_off_).first(m),
                                             std::span<const uint32_t>(self_k_of_).first(m),
                                             std::span<size_t>(self_found_).first(m),
                                             std::span<uint32_t>(self_hash_).first(m),
                                             options,
                                             observer,
                                             KernelRange::self_probe);
            for (size_t j = 0; j < m; ++j) {
                double v_src = 0.0;
                if constexpr (Sink::wants_values) {
                    v_src = self_val_[j];
                }
                const size_t found = self_found_[j];
                // kNotFound == kMissingIndex == size_t max, so one bound check covers both.
                if (found < op_size) {
                    // Freshly inserted partners (found >= combined_size) skip the mark: combined_size bounds it.
                    if (is_leader_pass && found < combined_size) {
                        matched.mark(found);
                    }
                    sink.self_hit(self_src_[j], found, self_phase_[j], v_src);
                }
                else {
                    // The stage dies with this pass and the misses are flushed after both, so copy now.
                    const size_t at = deferred_pos_flat_.size();
                    const auto *const first = self_stage_.pos_flat.data() + self_pos_off_[j];
                    deferred_pos_flat_.insert(deferred_pos_flat_.end(), first, first + self_k_of_[j]);
                    deferred_self_misses.push_back(
                        {at, self_k_of_[j], self_hash_[j], self_src_[j], self_phase_[j], v_src});
                }
            }
        }
    }
};

static inline auto empty_coeffs() -> const VecD & {
    static const VecD coeffs;
    return coeffs;
}

/*!
 * \brief One owner's traversal of one gate: its outgoing streams, cosine set and the gate-wide decisions.
 *
 * Produced by scan_gate() and consumed by a LayerBuildEngine: the streams move into its two passes, the cosine set
 * into finish().
 */
template <size_t NumModes>
struct GateScan {
    FusedScanResult<NumModes> streams; //!< Leader/follower queries, sources, values and self stages.
    CosMask cos_all;                   //!< Anticommuting rows, ascending and disjoint; empty for the fused sweep.
    mpi::PeerPlan plan;                //!< The generator's peer plan (legacy transport only).
    mpi::SlotWindow window;            //!< The owner's destination window: every per-slot stream is sized to it.
    size_t combined_size = 0;          //!< Pre-layer store size: the source boundary and the matched-set bound.
    bool identity = false;             //!< Identity generator: nothing anticommutes and nothing is exchanged.
    bool fused_scale = false;          //!< The fused cosine sweep scaled the coefficients during the traversal.
    double inv_cos = 1.0;              //!< Pre-cosine recovery factor for hit targets under the fused sweep.
};

/*!
 * \brief Owner traversal and source capture for one gate: the partner queries of every anticommuting row.
 *
 * The first phase of build_layer(), extracted so that a sharded caller can run it per owner. Reads `op` and its lazy
 * caches, which it may build, and under the fused sweep writes `*fused_scale_coeffs` (captured values are taken
 * before each overwrite); it grows no store and performs no MPI. With a serial `options` it runs entirely on the
 * calling thread. An opaque cutoff predicate is evaluated here, on whichever thread calls it.
 *
 * \param router    Flat-owner routing; `my_rank` is this owner's flat slot and `op` holds only rows it owns.
 * \param use_fused Whether the caller contracts immediately (ContractSink): values are captured and the fused
 *                  cosine sweep may run.
 * \return The streams, cosine set and decisions; `fused_scale` is the single authority for the apply.
 */
template <size_t NumModes, class Observer = NoRangeObserver>
auto scan_gate(MPOperator<NumModes> &local_op,
               const Monomial<NumModes> &gen,
               const CutoffFn<NumModes> &cutoff_fn,
               const std::optional<double> &atol,
               std::optional<std::reference_wrapper<const VecD>> local_coeffs,
               const std::optional<double> &upper_atol,
               const std::optional<double> &param,
               std::optional<size_t> only_rotate_len_k,
               const routing::Router &router,
               size_t my_rank,
               bool use_fused,
               VecD *fused_scale_coeffs,
               Basis basis,
               parallel::Options options = {},
               const Observer &observer = {}) -> GateScan<NumModes> {
    GateScan<NumModes> scan;
    // Under linear routing every query for this generator goes to rank my_rank ^ rank_shift(gen).
    const size_t gen_shift = router.rank_shift<NumModes>(gen);
    scan.plan = mpi::PeerPlan{.sparse = router.is_linear(), .shift = static_cast<int>(gen_shift)};
    // S of the P slots under linear routing, all P otherwise; every per-slot array is sized to it.
    scan.window = scan.plan.window(my_rank, router.ranks(), router.partitions());
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs.value_or(empty_coeffs()).get();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // Fused cos sweep: fold the per-gate cosine scale into the scan's own coefficient pass. No length cap only (a
    // popcount>k hit is outside the per-index cos set, so 1/cos recovery would be wrong) and cos!=0 (else
    // recovery is impossible; two-pass fallback). cos is even, so the sweep's cos(2·build_angle) matches
    // the apply's cos(2·apply_angle) bit-for-bit.
    const double cos_build = (use_fused && param.has_value()) ? std::cos(2.0 * param.value()) : 1.0;
    scan.fused_scale = use_fused && !only_rotate_len_k.has_value() && fused_scale_coeffs != nullptr && param.has_value()
                       && cos_build != 0.0;
    scan.inv_cos = scan.fused_scale ? 1.0 / cos_build : 1.0;
    assert(fused_scale_coeffs == nullptr || (local_coeffs && &local_coeffs->get() == fused_scale_coeffs));

    // An identity generator anticommutes with nothing, so its exchange would be empty collectives. The
    // generator list is replicated, so every owner skips it together.
    scan.identity = !gen.any();
    if (!scan.identity) {
        double *const sweep_ptr = scan.fused_scale ? fused_scale_coeffs->data() : nullptr;
        auto &fused = scan.streams;
        fused = with_algebra<NumModes>(basis, [&]<typename A>() {
            return fused_find_and_collect<NumModes, A, Observer>(local_op,
                                                                 gen,
                                                                 cut_eval,
                                                                 cut_st,
                                                                 coeffs,
                                                                 only_rotate_len_k,
                                                                 scan.window,
                                                                 my_rank,
                                                                 router,
                                                                 gen_shift,
                                                                 /*capture_values=*/use_fused,
                                                                 sweep_ptr,
                                                                 cos_build,
                                                                 options,
                                                                 observer);
        });
        auto &cos_all = scan.cos_all;
        if (fused.cos_blocks.size() == 1) {
            // The serial scan produces a single cosine block set — take it wholesale.
            cos_all = std::move(fused.cos_blocks[0]);
        }
        else {
            // A threaded scan produces one set per word range, in range order: the ranges own disjoint
            // ascending words, so concatenating in order keeps the blocks ascending and disjoint.
            size_t blocks = 0;
            for (const auto &block : fused.cos_blocks) {
                blocks += block.blocks.size();
            }
            cos_all.blocks.reserve(blocks);
            for (auto &block : fused.cos_blocks) {
                assert(block.blocks.empty() || cos_all.blocks.empty()
                       || cos_all.blocks.back().first < block.blocks.front().first);
                cos_all.total_count += block.total_count;
                cos_all.blocks.insert(cos_all.blocks.end(), block.blocks.begin(), block.blocks.end());
                CosMask{}.blocks.swap(block.blocks); // release each range's copy as soon as it is appended
            }
        }
        fused.cos_blocks = std::vector<CosMask>{};
    }
    scan.combined_size = local_op.store->size();
    return scan;
}

/*!
 * \brief Stamp a finished layer's recompute metadata: its generator words and the post-insertion store size.
 *
 * scaled_count is the store size after this layer's inserts, not the pre-layer source boundary: the fold truncated
 * to it reproduces the "all anticommuting" cos bit-for-bit with no stored bitmap. Fused mode has no LayerCore.
 */
template <size_t NumModes>
auto stamp_layer_metadata(LayerCore &storage, const Monomial<NumModes> &gen, const MPOperator<NumModes> &local_op)
    -> void {
    storage.generator_words.assign(gen.data(), gen.data() + mpi_detail::kWords<NumModes>);
    storage.scaled_count = static_cast<uint64_t>(local_op.store->size());
}

// Primary-path layer builder: one fused scan, then two resolve passes into the chosen sink. See LayerBuilder.h.
// The scan (fused_find_and_collect), the incoming decode and frozen probes, the self probe and a certified
// sink's scatter may run on workers; validation, missing-ID assignment, insertion, publication, inverted-index
// maintenance and graph packing stay on the caller. `observer` is the test seam of those phases (NoRangeObserver).
template <size_t NumModes, class Observer = NoRangeObserver>
auto build_layer(MPOperator<NumModes> &local_op,
                 const Monomial<NumModes> &gen,
                 const CutoffFn<NumModes> &cutoff_fn,
                 const std::optional<double> &atol,
                 std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                 const std::optional<double> &upper_atol,
                 const std::optional<double> &param,
                 std::optional<size_t> only_rotate_len_k,
                 MatchedEpochSet &matched_scratch,
                 mpi::Comm comm,
                 CosMask *out_cos = nullptr,
                 FusedContract *fused_contract = nullptr,
                 bool schrodinger = false,
                 VecD *fused_scale_coeffs = nullptr,
                 bool *fused_scale_out = nullptr,
                 Basis basis = Basis::Majorana,
                 parallel::Options options = {},
                 const Observer &observer = {}) -> std::shared_ptr<LayerCore> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = static_cast<size_t>(mpi::size(comm));
    // R is the flat world (ranks x partitions).
    const routing::Router router = router_for<NumModes>(comm);
    assert(router.flat_world() == R);
    // Fused contraction runs at all rank counts (R>1 via the cross-rank half-rotation exchange).
    const bool use_fused = (fused_contract != nullptr);
    auto scan = scan_gate<NumModes, Observer>(local_op,
                                              gen,
                                              cutoff_fn,
                                              atol,
                                              local_coeffs,
                                              upper_atol,
                                              param,
                                              only_rotate_len_k,
                                              router,
                                              my_rank,
                                              use_fused,
                                              fused_scale_coeffs,
                                              basis,
                                              options,
                                              observer);
    // scan_gate is the single authority for this decision; the fused caller must drive its apply from it.
    if (fused_scale_out != nullptr) {
        *fused_scale_out = scan.fused_scale;
    }

    auto run = [&]<typename Sink>(Sink sink) -> std::shared_ptr<LayerCore> {
        LayerBuildEngine<NumModes, Sink, Observer> eng(local_op,
                                                       R,
                                                       my_rank,
                                                       matched_scratch,
                                                       scan.combined_size,
                                                       std::move(sink),
                                                       scan.window,
                                                       options,
                                                       observer);
        if (!scan.identity) {
            auto &fused = scan.streams;
            eng.run_exchange(comm,
                             scan.plan,
                             /*is_leader_pass=*/true,
                             std::move(fused.leader_queries),
                             std::move(fused.leader_src),
                             std::move(fused.leader_val),
                             std::move(fused.leader_self));
            eng.run_exchange(comm,
                             scan.plan,
                             /*is_leader_pass=*/false,
                             std::move(fused.follower_queries),
                             std::move(fused.follower_src),
                             std::move(fused.follower_val),
                             std::move(fused.follower_self));
        }

        return eng.finish(std::move(scan.cos_all), out_cos);
    };

    std::shared_ptr<LayerCore> storage;
    if (use_fused) {
        const auto &coeffs = local_coeffs.value_or(empty_coeffs()).get();
        storage = run(ContractSink<NumModes>{.R = R,
                                             .my_rank = my_rank,
                                             .fc = *fused_contract,
                                             .op_coeffs = coeffs,
                                             .fused_scale = scan.fused_scale,
                                             .inv_cos = scan.inv_cos,
                                             .schrodinger = schrodinger,
                                             .basis = basis});
    }
    else {
        storage = run(GraphSink<NumModes>{R, my_rank});
    }

    // Recompute metadata rides with the layer so it survives every graph transform.
    if (storage != nullptr) {
        stamp_layer_metadata<NumModes>(*storage, gen, local_op);
    }

    return storage;
}

} // namespace monoprop::detail
