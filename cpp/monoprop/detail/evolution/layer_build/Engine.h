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
#include <span>
#include <utility>
#include <vector>

#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/BucketJoin.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/PairExchange.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

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

// The gate exchange. Per gate G with parameter θ (sin = sin 2θ, cos = cos 2θ), every tracked term ν
// anticommuting with G has a partner μ = ν ⊕ G owned by one flat slot, and the pair (ν, μ) rotates iff
// E(ν) ∨ E(μ), with both adds using both PRE-cos values:
//     c_ν += sin·(−φ_ν)·v_μ,   c_μ += sin·φ_ν·v_ν,   φ_ν = A::emit_phase(ν, G) = −φ_μ.
// (φ_μ = −φ_ν is forced by (i M_G)² = −1 and pinned by evolution_detail_tests.) The value path is one
// and a half rounds: only an EMITTING ν sends a record (key μ, φ_ν, rot = E(ν), [v_ν]) to owner(μ), so
// round 1 is O(|emitted|) rather than O(|Anti(G)|); after that exchange each slot joins what it received
// against its own anticommuting rows (BucketJoin.h):
//   • hit  → μ tracked here: apply +φ_rec·v_rec onto μ iff rot_rec ∨ E(μ). If E(μ) too, μ's own record
//            supplies ν's half symmetrically and nothing more is needed. If μ is SILENT it sent no
//            record, so this slot stages a response (the record's position in the sender's stream, plus
//            μ's pre-gate coefficient) back to ν's slot -- the half round. Values stay pre-cos on both
//            sides: the scan streams every silent row's pre-cos coefficient to `scratch.pre_cos` as it
//            sweeps (Scan.h, GateScratch.h SilentIndex), so no ×1/cos recovery of a stored value is
//            needed anywhere.
//   • miss → μ absent everywhere: the record mints μ at base + j (join order) with the same half.
//   • round 2 delivers the responses; each applies ν's half, −φ_ν·v_μ, and marks ν answered.
//   • absence pass over the records ν sent: only the owner of μ can send key ν or answer it, so
//            ¬received[ν] ∧ ¬answered[ν] ⟺ μ absent. μ was then minted from ν's record and ν's own
//            half is −φ_ν·c0(μ), c0 being the fresh term's pre-gate coefficient the sender computed.
// Per pair that is the same two adds with the same pre-cos values whichever branch supplies them, so the
// three cases are exact and not an approximation of the symmetric protocol.
// Δ = 0 (self peer) is the same rule with the records staged as positions instead of encoded, and a
// silent hit applied at once instead of answered. Mint indices are assigned in a fixed order -- self
// stage first, then incoming sources ascending, each in stream order -- so a run is bit-identical at
// fixed (R, S).
//
// Graph mode has no coefficients, so no term is silent and no response can be needed: it keeps the
// symmetric one-round predicate (every anticommuting term whose partner is structurally admissible
// sends, `rot` says whether its own side rotates) and the four ordered lists its positional replay is
// proved on. `Sink::wants_responses` is the compile-time switch between the two.
//
// Graph mode records the same rotations as (index, phase) endpoints per peer slot, in an order replay can
// pair positionally; GraphSink below has the rules. A sink owns the divergent state and supplies the
// four surfaces Resolve.h lists plus finalize. Each monomorphizes: no run-time fused/graph branch.

// Graph-build sink: accumulates the per-slot PartnerAcc endpoints and assembles a LayerCore at finalize.
// wants_values=false — the scan captures no coeffs and every rotation records only (index, phase).
//
// Roles in a pair are decided by the pivot bit (`RowMarks::foll`): ν and μ differ exactly on G's bits
// and the pivot is one of them, so exactly one endpoint of a tracked pair carries it. Per peer slot q,
// with the join's arrival order = q's stream order = ascending row at q:
//   in_pairs        hits on my follower rows whose pair rotates          (arrival order) (row(μ), φ_rec)
//   in_mints        rot=1 misses                                          (arrival order) (base+j, φ_rec)
//   out_pairs       my leaders with received ∧ (rot ∨ partner_rot)       (ascending row) (row, φ_own)
//   out_unanswered  my rot ∧ ¬received                                    (ascending row) (row, φ_own)
// Replay's positional pairing needs my out[j] and q's in[j] to be the same pair: my out_pairs are q's
// in_pairs (q's followers whose leader is at p), both in ascending-leader order; my out_unanswered are
// q's in_mints (my E-records that missed at q), both in my stream order. The in-pair takes the leader's
// φ, the out-pair its own, so sin_recv's (out, −φ) / (in, +φ) split (finalize) reproduces the two adds.
template <size_t NumModes>
struct GraphSink {
    static constexpr bool wants_values = false;
    static constexpr bool wants_responses = false;
    [[nodiscard]] auto incoming_form() const -> QueryForm { return QueryForm::Plain; }

    size_t R;
    size_t my_rank;
    std::vector<PartnerAcc> acc; // flat [P]: finalize hands it to build_layer_storage_unified, which is P-shaped

    GraphSink(size_t R_, size_t my_rank_) : R(R_), my_rank(my_rank_), acc(R_) {}

    auto hit(size_t slot, size_t row, double /*v*/, int phase, bool foll) -> void {
        if (foll) {
            acc[slot].in_pairs.push_back({row, phase});
        }
    }
    auto mint(size_t slot, size_t idx, double /*v*/, int phase) -> void { acc[slot].in_mints.push_back({idx, phase}); }
    auto out_pair(size_t slot, size_t row, int phase) -> void { acc[slot].out_pairs.push_back({row, phase}); }
    auto out_unanswered(size_t slot, size_t row, double /*c0*/, int phase) -> void {
        acc[slot].out_unanswered.push_back({row, phase});
    }
    // Nothing to size: the four lists are per peer slot, and a bound over the whole gate says nothing
    // about how the halves split between the slots.
    auto reserve_halves(size_t /*upper_bound*/) -> void {}

    // Drains the per-slot accumulators into the LayerCore's sin_send/sin_recv lists (layout derivation:
    // see cross_rank_sin_recv_index): sin_send = [in…, out…], sin_recv = [(out, −φ)…, (in, +φ)…]. cos
    // covers all anticommuting indices, endpoints included, since the sin_recv apply only adds the sine
    // term.
    auto finalize(CosMask &&cos_all, CosMask *out_cos, size_t combined_size, MPOperator<NumModes> &op)
        -> std::shared_ptr<LayerCore> {
        std::vector<CrossRankPartnerData> partners(R);
        for (size_t r = 0; r < R; ++r) {
            const auto &a = acc[r];
            auto &p = partners[r];
            const size_t P = a.in_count();
            const size_t Q = a.out_count();
            if (P + Q == 0) {
                continue;
            }
            p.in_count = P; // boundary for deriving the sin_recv index list from sin_send (not stored)
            p.sin_send_indices.resize(P + Q);
            p.sin_recv_entries.resize(P + Q);
            size_t k = 0;
            for (const auto *list : {&a.in_pairs, &a.in_mints}) {
                for (const auto &e : *list) {
                    p.sin_send_indices[k] = e.idx;
                    p.sin_recv_entries[Q + k] = {e.idx, e.phase};
                    ++k;
                }
            }
            size_t j = 0;
            for (const auto *list : {&a.out_pairs, &a.out_unanswered}) {
                for (const auto &e : *list) {
                    p.sin_send_indices[P + j] = e.idx;
                    p.sin_recv_entries[j] = {e.idx, -e.phase};
                    ++j;
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

// Fused ContractImmediately sink: every half-rotation this slot owns goes straight into the FusedContract
// (no LayerCore — finalize returns nullptr), applied to op_coeffs by apply_fused_contract. wants_values=true:
// records carry the sender's pre-cos coefficient, which is the whole value the apply needs.
template <size_t NumModes>
struct ContractSink {
    static constexpr bool wants_values = true;
    static constexpr bool wants_responses = true;
    [[nodiscard]] auto incoming_form() const -> QueryForm { return QueryForm::Fused; }

    FusedContract &fc;
    bool fused_scale; // the fused cos sweep ran: inserted endpoints fold cos in at the apply, not here
    // Where a response's payload comes from. Without the fused sweep the scan leaves every coefficient
    // pre-gate, so the picture's own array answers. With it, every anticommuting row is already scaled
    // and the pre-cos value of a silent one is in `pre_cos`, at the slot `silent` names -- the same
    // double the scan read, so the answer is bit-identical either way.
    const double *pre_gate_coeffs = nullptr;
    const SilentIndex *silent = nullptr;
    const double *pre_cos = nullptr;

    auto hit(size_t /*slot*/, size_t row, double v, int phase, bool /*foll*/) -> void {
        push_half_(HalfRotationRec{row_of_(row), static_cast<int8_t>(phase), /*is_insert=*/false, v});
    }
    [[nodiscard]] auto silent_value(size_t row) const -> double {
        return (pre_cos != nullptr) ? pre_cos[silent->ordinal(row)] : pre_gate_coeffs[row];
    }
    // ν's half of an asymmetric pair, from the response its partner sent (or, on the self slot, from the
    // silent hit directly): −φ_ν · v_μ, the same add the symmetric case gets off μ's own record.
    auto answer(size_t /*slot*/, size_t row, double v, int phase) -> void {
        push_half_(HalfRotationRec{row_of_(row), static_cast<int8_t>(-phase), /*is_insert=*/false, v});
    }
    auto mint(size_t /*slot*/, size_t idx, double v, int phase) -> void {
        push_half_(HalfRotationRec{row_of_(idx), static_cast<int8_t>(phase), /*is_insert=*/true, v});
    }
    auto out_pair(size_t /*slot*/, size_t /*row*/, int /*phase*/) -> void {}
    // Pushed in Heisenberg too, where c0 = 0, so the branch is one rule rather than two. Adding ±0.0
    // cannot change a nonzero coefficient; its only effect is on a coefficient that is exactly -0.0,
    // which the add can turn into +0.0.
    auto out_unanswered(size_t /*slot*/, size_t row, double c0, int phase) -> void {
        push_half_(HalfRotationRec{row_of_(row), static_cast<int8_t>(-phase), /*is_insert=*/false, c0});
    }
    // One allocation per gate instead of a doubling chain: `fc` is built fresh for each gate, so every
    // half used to move once per reallocation. The bound is the caller's (Engine.h exchange_and_join).
    auto reserve_halves(size_t upper_bound) -> void { fc.halves.reserve(upper_bound); }

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

private:
    // Every slot a half names is a row of this rank's own operator, and those are TermIndex-wide
    // throughout the engine (the join's own entries carry them as uint32 too).
    static auto row_of_(size_t row) -> TermIndex {
        assert(row <= std::numeric_limits<TermIndex>::max() && "a half named a slot past the TermIndex ceiling");
        return static_cast<TermIndex>(row);
    }
    // Holds the reserve honest: a bound that ever came out short would silently reintroduce the doubling
    // chain rather than fail, so it is asserted at the only place that can notice.
    auto push_half_(HalfRotationRec rec) -> void {
        assert(fc.halves.size() < fc.halves.capacity() && "more halves than reserve_halves() was sized for");
        fc.halves.push_back(rec);
    }
};

// Capacity, not size, throughout the overloads below: what a buffer costs the process is what it
// reserved, and several of these are deliberately sized to a bound and filled short (SelfQueryStage
// says so of itself).

template <typename Vector>
[[nodiscard]] inline auto reserved_bytes(const Vector &v) -> size_t {
    return v.capacity() * sizeof(typename Vector::value_type);
}

template <typename T>
[[nodiscard]] inline auto reserved_bytes(const mpi::WindowVec<std::vector<T>> &wv) -> size_t {
    size_t bytes = wv.size() * sizeof(std::vector<T>);
    for (const auto &slot : wv) {
        bytes += reserved_bytes(slot);
    }
    return bytes;
}

template <size_t NumModes>
[[nodiscard]] inline auto reserved_bytes(const SelfQueryStage<NumModes> &stage) -> size_t {
    return reserved_bytes(stage.pos_flat) + reserved_bytes(stage.pos_off) + reserved_bytes(stage.k_of)
           + reserved_bytes(stage.phase_of) + reserved_bytes(stage.rot_of) + reserved_bytes(stage.val_of)
           + reserved_bytes(stage.fp_of);
}

template <size_t NumModes>
[[nodiscard]] inline auto reserved_bytes(const IncomingRecords<NumModes> &records) -> size_t {
    return reserved_bytes(records.goff) + reserved_bytes(records.phase_of) + reserved_bytes(records.rot_of)
           + reserved_bytes(records.val_of) + reserved_bytes(records.pos_flat) + reserved_bytes(records.pos_off)
           + reserved_bytes(records.k_of) + reserved_bytes(records.fp_of);
}

template <size_t NumModes>
[[nodiscard]] inline auto reserved_bytes(const MissStage<NumModes> &misses) -> size_t {
    return reserved_bytes(misses.pos_flat) + reserved_bytes(misses.pos_off) + reserved_bytes(misses.k_of);
}

// Owns build_layer's machinery over a compile-time Sink policy. combined_size = the pre-layer operator size.
template <size_t NumModes, typename Sink>
struct LayerBuildEngine {
    using RowPosT = typename OperatorIndex<NumModes>::PosT;

    MPOperator<NumModes> &local_op; // scanned, looked up, and grown by the inserts
    mpi::Comm comm;
    size_t R;
    size_t my_rank;
    // This gate's join (its row side staged by the scan) and the other per-gate scratch,
    // propagator-owned so capacity survives the gate; the protocol's per-row state is in
    // scratch.marks (GateScratch.h).
    GateScratch<NumModes> &scratch;
    size_t combined_size;
    // The destination slots this generator can reach: `plan`'s window for my_rank. Every per-slot array
    // is sized to it, so a flat slot only ever enters through WindowVec::at_slot.
    mpi::SlotWindow window;
    // Which destination ranks this gate's records can reach. Dense unless the router is GF(2)-linear;
    // see mpi::PeerPlan. Derived once per layer in build_layer, never per record.
    mpi::PeerPlan plan;
    Sink sink;
    MissStage<NumModes> misses; // this gate's mints, in join order
    bool pair_path = false;     // this gate's exchange is one rank pair: mpi::pair_exchange carries it
    int pair_shift = 0;         // the rank shift it names; 0 is the in-rank exchange

    LayerBuildEngine(MPOperator<NumModes> &local_op_,
                     mpi::Comm comm_,
                     size_t R_,
                     size_t my_rank_,
                     GateScratch<NumModes> &scratch_,
                     size_t combined_size_,
                     Sink &&sink_,
                     mpi::PeerPlan plan_ = {}) // dense by default: the tests build the engine directly
        : local_op(local_op_),
          comm(comm_),
          R(R_),
          my_rank(my_rank_),
          scratch(scratch_),
          combined_size(combined_size_),
          plan(plan_),
          sink(std::move(sink_)) {
        const auto geom = mpi::geometry(comm);
        window = plan.window(my_rank, static_cast<size_t>(geom.ranks), static_cast<size_t>(geom.partitions));
        assert(window.stop() <= R && window.count != 0);
        // pair_exchange addresses ONE rank pair, so it covers exactly the gates whose window is one
        // rank's partitions: linear routing (the peer rank is my_rank ^ shift) and every single-rank
        // world (the peer is this rank, shift 0). Dense routing over several ranks reaches a window of
        // ranks x partitions, which only the collective can carry.
        pair_path = R > 1 && window.count == static_cast<size_t>(geom.partitions);
        pair_shift = plan.sparse ? plan.shift : 0;
        assert((!pair_path || plan.sparse || geom.ranks == 1) && "a dense multi-rank gate cannot pair");
    }

    // The round and a half: exchange the scan's records, match every record against this slot's
    // anticommuting rows in ONE bucketed join, apply the receiver rule to the self stage and then the
    // incoming records in that fixed order, exchange the responses the silent hits staged, insert the
    // misses while that is in flight, apply the responses, walk the sent records. Taking the scan result
    // by value is what makes the sequence unmissable -- nothing else can read the records once they are
    // here.
    //
    // The self stage is joined after the wait, not before it: both sides of the query space must be in
    // the same join, since the rows are streamed against it exactly once. That costs the small overlap
    // the self stage used to give, and only when a slot has both self-owned and remote partners for one
    // generator -- under linear routing a generator's shift is either zero (all self, empty exchange) or
    // not (empty self stage).
    auto exchange_and_join(FusedScanResult<NumModes> &&scan) -> void {
        // The scan sized its arrays to the same plan, so the two windows must agree exactly -- a mismatch
        // would re-base every slot against the wrong base.
        assert(scan.window.base == window.base && scan.window.count == window.count);
        RowMarks &marks = scratch.marks;
        BucketJoin<NumModes> &join = scratch.join;
        const size_t base = local_op.store->size();
        misses.clear();

        // Self is inside the window only when this generator's rank shift is zero; otherwise the window
        // names another rank outright and the scan cannot have staged a self-owned partner. The self block
        // of the wire array is empty either way (staged, never encoded), so the exchange never sends to
        // self.
        if (window.contains(my_rank)) {
            assert(scan.queries.at_slot(my_rank).empty() && "self-owned records are staged, never encoded");
        }
        else {
            assert(scan.self.size() == 0 && "a self-owned partner outside this generator's peer window");
        }
        std::optional<mpi::PendingAlltoallv<size_t>> pending;
        if (R > 1 && !pair_path) {
            pending.emplace(
                mpi::begin_alltoallv(scan.queries, comm, /*skip_self=*/false, /*known_recv_counts=*/nullptr, plan));
        }
        mpi::WindowVec<VecZ> incoming;
        IncomingRecords<NumModes> pr;
        // The queries live in the scratch pool from here on, not in `scan`: the peers of a pair exchange
        // read this buffer in place until this partition's NEXT call returns, which is past the end of
        // `scan`. The move keeps every slot's storage where it is, so nothing is copied to say so.
        // The scan filled the scratch's buffer; put it back before anything can publish it, so the
        // records outlive the gate whatever `scan` does. The parity moves on at the end of the gate.
        mpi::WindowVec<VecZ> &queries = scratch.wire_queries();
        queries = std::move(scan.queries);
        if (pair_path) {
            pr = decode_incoming_records<NumModes>(mpi::pair_exchange(comm, pair_shift, publish_(queries)).from,
                                                   window,
                                                   sink.incoming_form());
        }
        else if (pending.has_value()) {
            pending->wait_into(incoming);
            pr = decode_incoming_records<NumModes>(slot_streams(incoming, scratch.slot_views),
                                                   incoming.window(),
                                                   sink.incoming_form());
        }
        // The response staging indexes the response buffers by the SOURCE window's index, so the two
        // windows must be the one window -- a mismatch would answer the wrong peer.
        assert((pr.nq_total == 0 || (pr.window.base == window.base && pr.window.count == window.count))
               && "the delivered records came from a window this gate cannot answer");

        // Query order IS mint order: the self stage first, then the incoming sources in ascending window
        // order, each in its sender's stream order.
        const size_t n_self = scan.self.size();
        join.begin_queries(n_self + pr.nq_total);
        for (size_t q = 0; q < n_self; ++q) {
            join.add_query(q, scan.self.fp_of[q]);
        }
        for (size_t g = 0; g < pr.nq_total; ++g) {
            join.add_query(n_self + g, pr.fp_of[g]);
        }
        // The row side is built now, not by the scan: every record this gate must answer is staged, so
        // the join can reject the anticommuting rows no record can name before they cost anything.
        join.stage_rows(*local_op.store, scratch.nz);
        join.run(*local_op.store, [&](size_t q) -> std::span<const RowPosT> {
            return (q < n_self) ? scan.self.positions_at(q) : pr.positions_at(q - n_self);
        });

        // Both output buffers of the resolve phase are sized here, before the first push, because the
        // join has just settled the two counts they depend on. Every sink call is keyed on either a
        // distinct query -- a hit pushes at most one half, a miss mints at most one -- or a distinct
        // record this slot sent, since a record is answered (round 2, or a self silent hit) or absent
        // but never both, and a row sends exactly one record because it has exactly one partner. So
        // |Q| + |sent| bounds the gate's halves, and |Q| - |hits| bounds its mints exactly.
        size_t n_sent = 0;
        for (size_t k = 0; k < window.count; ++k) {
            n_sent += scan.sent[mpi::WindowIndex{k}].size();
        }
        sink.reserve_halves(join.queries() + n_sent);
        misses.reserve(join.queries() - join.hits());

        // Round 2's send buffer is claimed before the first push for the same lifetime reason, and it
        // is the OTHER slot of the pool: round 1's is still published to the peers.
        // Round 2's buffer has no twin: only the two-call fused sink stages responses, and its second
        // call is bounded by the NEXT gate's first, so one buffer is already outlived. A sink that
        // answers nothing keeps an empty local rather than naming the shared one, so the gate stamp
        // never charges it what an earlier fused call left there.
        mpi::WindowVec<VecZ> responses_none;
        mpi::WindowVec<VecZ> &responses = Sink::wants_responses ? scratch.wire_r : responses_none;
        if constexpr (Sink::wants_responses) {
            reuse_wire(responses, window);
        }
        size_t answered_self = 0;
        if (n_self != 0) {
            answered_self = join_self<NumModes>(scan.self,
                                                join,
                                                /*q_base=*/0,
                                                marks,
                                                my_rank,
                                                base,
                                                misses,
                                                sink,
                                                std::span<const SentRecord>(scan.sent.at_slot(my_rank)));
        }
        join_incoming<NumModes>(pr, join, /*q_base=*/n_self, marks, base, misses, sink, responses);

        // Round 1 at its widest: the scan's arrays, the delivered records and the staged responses.
        stamp_gate_buffers_(scan, queries, incoming, pr, responses, /*extra=*/0);

        // Round 2 is the same verb with the response counts, posted before the inserts so the store's
        // growth overlaps the peers' answers. Its window is round 1's: the rank shift is an involution,
        // so the slot a record came from is a slot this one can send to.
        std::optional<mpi::PendingAlltoallv<size_t>> answering;
        if constexpr (Sink::wants_responses) {
            if (R > 1 && !pair_path) {
                answering.emplace(
                    mpi::begin_alltoallv(responses, comm, /*skip_self=*/false, /*known_recv_counts=*/nullptr, plan));
            }
        }
        // The inserts move ahead of the pair exchange rather than under it: pair_exchange is synchronous,
        // so there is no post to overlap, and the store's growth is independent of the answers either way
        // (mint indices were assigned against `base` before this line). Their order against
        // apply_responses is what the result depends on, and that is unchanged.
        insert_misses<NumModes>(local_op, misses, base);
        if constexpr (Sink::wants_responses) {
            if (pair_path) {
                const auto answers = mpi::pair_exchange(comm, pair_shift, publish_(responses)).from;
                // Round 2 at its widest: round 1's buffers are all still in scope under the answers.
                stamp_gate_buffers_(scan, queries, incoming, pr, responses, /*extra=*/0);
                apply_responses<NumModes>(marks, scan.sent, answers, window, sink);
            }
            else if (answering.has_value()) {
                mpi::WindowVec<VecZ> answers;
                answering->wait_into(answers);
                stamp_gate_buffers_(scan, queries, incoming, pr, responses, reserved_bytes(answers));
                apply_responses<NumModes>(marks,
                                          scan.sent,
                                          slot_streams(answers, scratch.slot_views),
                                          answers.window(),
                                          sink);
            }
        }
        absence_pass<NumModes>(marks, scan.sent, scan.sent_c0, sink);
        // Past every read of this gate's queries, so the next gate's scan takes the OTHER buffer and
        // this one stays intact for the peers that are still holding views into it.
        ++scratch.wire_gate;

        scratch.counters.gates += 1;
        scratch.counters.records += n_sent;
        if constexpr (Sink::wants_responses) {
            scratch.counters.responses += answered_self;
            for (size_t k = 0; k < window.count; ++k) {
                scratch.counters.responses += responses[mpi::WindowIndex{k}].size() / kResponseWords;
            }
        }
    }

    /*! @brief The gate's S send descriptors, for pair_exchange.
     *
     *  The verb copies the outer array before its barrier, so one reusable array serves both rounds.
     *  The BUFFERS it names are the ones the lifetime rule binds, and those are the scratch pool's.
     */
    auto publish_(const mpi::WindowVec<VecZ> &buf) -> mpi::SubStreams {
        scratch.wire_spans.resize(window.count);
        for (size_t k = 0; k < window.count; ++k) {
            scratch.wire_spans[k] = std::span<const size_t>(buf[mpi::WindowIndex{k}]);
        }
        return mpi::SubStreams{scratch.wire_spans};
    }

    /*! @brief Stamps the gate's per-gate buffers at one instant into the call's high-water mark.
     *
     *  @param extra Round 2's answers, which are live only at the second sample.
     *
     *  Two kinds of buffer are counted, and neither can be read back once the gate is over. The
     *  exchange buffers -- what the gate puts on the wire, what it receives, the responses it stages
     *  and the answers it applies -- die with the gate outright. The scan's own three -- the pre-cos
     *  stream, the silent index, and the join's query-tag filter -- are sized per gate and released or
     *  overwritten under the next one, so `gate_scratch_bytes` at rest reports the LAST gate's, not the
     *  widest gate's. On a wide gate the two together are the largest thing the call adds to RSS, which
     *  is the whole reason the field exists.
     *
     *  The rest of GateScratch stays out: `join`'s buckets, `marks` and `nz` only ever grow, so their
     *  resting figure in `gate_scratch_bytes` already IS their peak and counting them here would say
     *  nothing new. scan.cos_blocks stays out too -- it is moved into the layer rather than freed, so
     *  the layer storage already prices it.
     */
    auto stamp_gate_buffers_(const FusedScanResult<NumModes> &scan,
                             const mpi::WindowVec<VecZ> &queries,
                             const mpi::WindowVec<VecZ> &incoming,
                             const IncomingRecords<NumModes> &pr,
                             const mpi::WindowVec<VecZ> &responses,
                             size_t extra) -> void {
        const size_t bytes = extra + reserved_bytes(queries) + reserved_bytes(scan.sent) + reserved_bytes(scan.sent_c0)
                             + reserved_bytes(scan.self) + reserved_bytes(incoming) + reserved_bytes(pr)
                             + reserved_bytes(responses) + reserved_bytes(misses) + reserved_bytes(scratch.pre_cos)
                             + scratch.silent.memory_bytes() + scratch.join.filter_bytes();
        scratch.buffers_hwm_bytes = std::max(scratch.buffers_hwm_bytes, bytes);
    }

    auto finish(CosMask &&cos_all, CosMask *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        return sink.finalize(std::move(cos_all), out_cos, combined_size, local_op);
    }
};

static inline auto empty_coeffs() -> const VecD & {
    static const VecD coeffs;
    return coeffs;
}

// Primary-path layer builder: one fused scan, one exchange, one join into the chosen sink. See LayerBuilder.h.
// `over_cutoff_possible` is the caller's per-call flag (MonomialPropagator::run_gate_loop_): true when a
// tracked term may fail the structural cutoff, which widens the scan's send predicate to every
// anticommuting term.
template <size_t NumModes>
auto build_layer(MPOperator<NumModes> &local_op,
                 const Monomial<NumModes> &gen,
                 const CutoffFn<NumModes> &cutoff_fn,
                 const std::optional<double> &atol,
                 std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                 const std::optional<double> &upper_atol,
                 const std::optional<double> &param,
                 std::optional<size_t> only_rotate_len_k,
                 bool over_cutoff_possible,
                 GateScratch<NumModes> &scratch,
                 mpi::Comm comm,
                 CosMask *out_cos = nullptr,
                 FusedContract *fused_contract = nullptr,
                 bool schrodinger = false,
                 VecD *fused_scale_coeffs = nullptr,
                 bool *fused_scale_out = nullptr,
                 Basis basis = Basis::Majorana) -> std::shared_ptr<LayerCore> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = static_cast<size_t>(mpi::size(comm));
    // R is the FLAT world (ranks x partitions); the router is what splits it back into the two levels.
    const routing::Router router = router_for<NumModes>(comm);
    assert(router.flat_world() == R);
    // Under linear routing every record for THIS generator lands on the rank this rank's own index XOR
    // rank_shift(gen), so the exchange knows its peer before it starts. Dense otherwise, which is
    // today's collective.
    const size_t gen_shift = router.rank_shift<NumModes>(gen);
    const auto plan = mpi::PeerPlan{.sparse = router.is_linear(), .shift = static_cast<int>(gen_shift)};
    // The reachable slots, once per generator: S of the P=R*S world under linear routing, all P otherwise.
    // Every per-slot structure from the scan to the wire is sized to this run.
    const mpi::SlotWindow scan_window = plan.window(my_rank, router.ranks(), router.partitions());
    // Fused contraction runs at all rank counts (R>1 via the cross-rank half-rotation exchange).
    const bool use_fused = (fused_contract != nullptr);
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs.value_or(empty_coeffs()).get();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // Fused cos sweep: fold the per-gate cosine scale into the scan's own coefficient pass. No length cap
    // only (a popcount>k term is outside the per-index cos set) and cos!=0 (else the two-pass fallback).
    // cos is even, so the sweep's cos(2·build_angle) matches the apply's cos(2·apply_angle) bit-for-bit.
    const double cos_build = (use_fused && param.has_value()) ? std::cos(2.0 * param.value()) : 1.0;
    const bool fused_scale = use_fused && !only_rotate_len_k.has_value() && fused_scale_coeffs != nullptr
                             && param.has_value() && cos_build != 0.0;
    // build_layer is the single authority for this decision; the fused caller must drive its apply from it.
    if (fused_scale_out != nullptr) {
        *fused_scale_out = fused_scale;
    }
    assert(fused_scale_coeffs == nullptr || (local_coeffs && &local_coeffs->get() == fused_scale_coeffs));

    // An identity generator anticommutes with nothing: the scan returns on its empty fold-column set
    // with no record, no cosine block and no coefficient swept, and the exchange would carry no payload.
    // The generator list is replicated, so skipping needs no agreement. (A zero chemical potential alone
    // contributes 60 of the 60-site Hubbard's 476 generators per Trotter layer.) No gate is merged: a
    // no-op gate is simply not exchanged for.
    const bool identity_gen = !gen.any();

    FusedScanResult<NumModes> fused;
    CosMask cos_all;
    // An identity generator skips the scan, so the row side it would have staged is emptied here.
    scratch.nz.clear();
    scratch.join.clear_rows();
    if (!identity_gen) {
        double *const sweep_ptr = fused_scale ? fused_scale_coeffs->data() : nullptr;
        // The fresh partner's pre-gate coefficient is state-scored only in the Schrödinger picture, and
        // only the fused path needs it on the sender side (graph replay reads it off the extended vector).
        std::optional<Monomial<NumModes>> state_mask;
        if (use_fused && schrodinger) {
            state_mask = initial_state_mask<NumModes>(local_op.initial_state);
        }
        fused = with_algebra<NumModes>(basis, [&]<typename A>() {
            return fused_find_and_collect<NumModes, A>(local_op,
                                                       gen,
                                                       cut_eval,
                                                       cut_st,
                                                       coeffs,
                                                       only_rotate_len_k,
                                                       over_cutoff_possible,
                                                       scan_window,
                                                       my_rank,
                                                       router,
                                                       scratch,
                                                       /*capture_values=*/use_fused,
                                                       sweep_ptr,
                                                       cos_build,
                                                       state_mask ? &*state_mask : nullptr);
        });
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
    }

    auto run = [&]<typename Sink>(Sink sink) -> std::shared_ptr<LayerCore> {
        LayerBuildEngine<NumModes, Sink> eng(local_op,
                                             comm,
                                             R,
                                             my_rank,
                                             scratch,
                                             /*combined_size=*/local_op.store->size(),
                                             std::move(sink),
                                             plan);
        if (!identity_gen) {
            eng.exchange_and_join(std::move(fused));
        }
        return eng.finish(std::move(cos_all), out_cos);
    };

    std::shared_ptr<LayerCore> storage;
    if (use_fused) {
        const bool swept = fused_scale && !identity_gen;
        storage = run(ContractSink<NumModes>{.fc = *fused_contract,
                                             .fused_scale = fused_scale,
                                             .pre_gate_coeffs = coeffs.data(),
                                             .silent = swept ? &scratch.silent : nullptr,
                                             .pre_cos = swept ? scratch.pre_cos.data() : nullptr});
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
