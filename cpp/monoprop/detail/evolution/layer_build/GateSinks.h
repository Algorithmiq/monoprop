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

// The two sinks the gate exchange resolves into. A sink owns the divergent state and supplies the
// surfaces Resolve.h lists plus finalize; each monomorphizes, so there is no run-time fused/graph branch.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
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

/*! @brief Graph-build sink: collects the per-slot PartnerAcc endpoints and assembles a LayerCore.
 *
 *  wants_values=false -- the scan captures no coefficients and every rotation records only
 *  (index, phase), so no term is silent and no response can be needed: the symmetric send predicate and
 *  the four ordered lists its positional replay is proved on both stand.
 *
 *  Roles in a pair are decided by the pivot bit (`RowMarks::foll`): a term and its partner differ exactly
 *  on G's bits and the pivot is one of them, so exactly one endpoint of a tracked pair carries it. Per
 *  peer slot q, with the join's arrival order = q's stream order = ascending row at q:
 *    in_pairs        hits on my follower rows whose pair rotates    (arrival order) (row(partner), phi_rec)
 *    in_mints        rot=1 misses                                   (arrival order) (base+j, phi_rec)
 *    out_pairs       my leaders with received and (rot or partner_rot)  (ascending row) (row, phi_own)
 *    out_unanswered  my rot and not received                        (ascending row) (row, phi_own)
 *  Replay's positional pairing needs my out[j] and q's in[j] to be the same pair: my out_pairs are q's
 *  in_pairs (q's followers whose leader is at p), both in ascending-leader order; my out_unanswered are
 *  q's in_mints (my E-records that missed at q), both in my stream order. The in-pair takes the leader's
 *  phase, the out-pair its own, so sin_recv's (out, -phi) / (in, +phi) split reproduces the two adds.
 */
template <size_t NumModes>
struct GraphSink {
    static constexpr bool wants_values = false;
    static constexpr bool wants_responses = false;
    [[nodiscard]] auto incoming_form() const -> QueryForm { return QueryForm::Plain; }

    size_t R;
    size_t my_rank;
    std::vector<PartnerAcc> acc; // flat [R]: finalize hands it to build_layer_storage_unified

    GraphSink(size_t R_, size_t my_rank_) : R(R_), my_rank(my_rank_), acc(R_) {}

    auto hit(size_t slot, size_t row, double /*v*/, int phase, bool foll, bool /*own_rot*/) -> void {
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

    /*! @brief Drains the per-slot accumulators into the LayerCore's sin_send/sin_recv lists.
     *
     *  Layout derivation: see cross_rank_sin_recv_index. sin_send = [in..., out...],
     *  sin_recv = [(out, -phi)..., (in, +phi)...]. cos covers all anticommuting indices, endpoints
     *  included, since the sin_recv apply only adds the sine term.
     */
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

/*! @brief Fused ContractImmediately sink: every half-rotation this slot owns goes straight into the
 *  FusedContract (no LayerCore -- finalize returns nullptr), applied to op_coeffs by
 *  apply_fused_contract.
 *
 *  wants_values=true: records carry the sender's pre-cos coefficient, which is the whole value the apply
 *  needs.
 */
template <size_t NumModes>
struct ContractSink {
    static constexpr bool wants_values = true;
    static constexpr bool wants_responses = true;
    //! A mutual pair's two halves from the leader's record alone (Resolve.h join_self): the follower's
    //! half is silent_value(follower), which is what the follower's own record would have delivered.
    static constexpr bool pairs_once = true;
    [[nodiscard]] auto incoming_form() const -> QueryForm { return QueryForm::Fused; }

    FusedContract &fc;
    bool fused_scale;       // the fused cos sweep ran: inserted endpoints fold cos in at the apply, not here
    const VecD &op_coeffs;  // the very array the scan read, not a copy: under the sweep every
                            // anticommuting slot in it already holds fl(c * cos)
    double cos_build = 1.0; // cos(2*theta), the factor the sweep applied
    double inv_cos = 1.0;   // fl(1/cos_build) under the sweep, 1.0 without it
    // Does an absent partner's half carry a value at all? Only the Schrodinger picture scores a fresh
    // term's pre-gate coefficient (Scan.h `state_mask`); in Heisenberg `c0` is the literal 0.0 that
    // absence_pass substitutes. Equal to absence_pass's own `has_c0` by construction: `sent_c0` is
    // windowed exactly when the scan was given a state mask.
    bool absences_carry_c0 = false;

    /*! @brief One rotation half onto tracked row `row`, from the record that found it.
     *
     *  ONE endpoint of a rotating pair reads a partner value RECOVERED from its swept slot,
     *  stored*(1/cos); the other reads the exact pre-cos double the record carries. Which is which is
     *  not a free choice: the two-pass protocol built one rotation record per pair from the leader's
     *  query, applying c[leader] from the follower's stored*(1/cos) while c[follower] took the leader's
     *  value straight out of the scan's register. Reproducing that here means recovering exactly when
     *  the half lands on a leader that EMITTED. `foll` is complementary across a pair (mu = nu ^ G flips
     *  it), so the row's own pivot bit names the sender's side too and nothing extra crosses the wire.
     */
    auto hit(size_t /*slot*/, size_t row, double v, int phase, bool foll, bool own_rot) -> void {
        const double v_partner = (fused_scale && own_rot && !foll) ? recovered_(v) : v;
        push_half_(HalfRotationRec{row_of_(row), static_cast<int8_t>(phase), /*is_insert=*/false, v_partner});
    }
    //! A silent row is always the endpoint the sender's query FOUND, so its value is always recovered --
    //! and here it is read straight off its own swept slot, the expression's original form.
    [[nodiscard]] auto silent_value(size_t row) const -> double {
        return fused_scale ? op_coeffs[row] * inv_cos : op_coeffs[row];
    }
    // The sender's half of an asymmetric pair, from the response its partner sent (or, on the self slot,
    // from the silent hit directly): -phi * v, the same add the symmetric case gets off the partner's
    // own record.
    auto answer(size_t /*slot*/, size_t row, double v, int phase) -> void {
        push_half_(HalfRotationRec{row_of_(row), static_cast<int8_t>(-phase), /*is_insert=*/false, v});
    }
    auto mint(size_t /*slot*/, size_t idx, double v, int phase) -> void {
        push_half_(HalfRotationRec{row_of_(idx), static_cast<int8_t>(phase), /*is_insert=*/true, v});
    }
    auto out_pair(size_t /*slot*/, size_t /*row*/, int /*phase*/) -> void {}
    // Skipped in Heisenberg, where c0 is exactly 0.0: the half's add is `c += sin*(-phase)*0.0`, which
    // cannot change any coefficient's VALUE -- its one and only effect is that a coefficient sitting at
    // -0.0 stays there instead of being turned into +0.0, and -0.0 == +0.0. Keeping it costs a 16-byte
    // half and one random-access `+=` per mint of every gate.
    auto out_unanswered(size_t /*slot*/, size_t row, double c0, int phase) -> void {
        if (!absences_carry_c0) {
            return;
        }
        push_half_(HalfRotationRec{row_of_(row), static_cast<int8_t>(-phase), /*is_insert=*/false, c0});
    }
    // One allocation per gate instead of a doubling chain: `fc` is built fresh for each gate, so every
    // half used to move once per reallocation. The bound is the caller's (Engine.h exchange_and_join).
    auto reserve_halves(size_t upper_bound) -> void { fc.halves.reserve(upper_bound); }

    // No LayerCore in the fused path -> nullptr. Two-pass fused (k>0 / cos==0 fallback) appends inserted
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
    // fl(fl(v * cos) * fl(1/cos)): the double the partner's own swept slot holds, reached from the
    // record's pre-cos value because that partner's row is not on this rank. The sweep stored fl(v * cos)
    // there exactly, so this is the same pair of roundings a load and a multiply of that slot would give.
    [[nodiscard]] auto recovered_(double v) const -> double { return (v * cos_build) * inv_cos; }

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

} // namespace monoprop::detail
