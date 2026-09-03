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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::detail {

// Sentinel for "no index resolved yet" in the resolve slots. Deliberately equal to
// OperatorIndex::kNotFound, so one `>= size` bound check covers a miss and an unresolved slot alike.
inline constexpr size_t kMissingIndex = std::numeric_limits<size_t>::max();

// One nonzero-overlap word of the anticommutation fold, produced by the scan's pass 1 and read again by
// the join's streaming pass and the per-gate mark clear. `overlap` bit t set ⟺ term (base+t)
// anticommutes with G; `foll` = overlap & pivot column = the followers (leaders are `overlap ^ foll`).
// `base` is a multiple of 64: it is word wi's first row, so base/64 indexes a row-bitset word directly.
struct EvenParityNzWord {
    size_t base;
    uint64_t overlap;
    uint64_t foll;
};

// One record this slot sent for a gate, in stream order (ascending source row): the source row and the
// emit phase φ of that source. The absence pass walks these, so the row is carried rather than looked
// up -- there are no per-gate ordinals to index a phase array by.
struct SentRecord {
    TermIndex row;
    int8_t phase; // ternary, as A::emit_phase returns it
};

// Round 2 of the gate protocol (Engine.h): the answer a slot owes a record that hit one of its silent
// rows. `sent_idx` is that record's position in the sender's own stream to this slot, which is what
// lets the answer name a row without carrying one -- the sender maps it back through its `sent` list.
// Two words, so the response round reuses the records' transport verb and buffer type.
inline constexpr size_t kResponseWords = 2;

// A wire buffer's smallest interesting slot, so the release rule below leaves tiny slots alone.
inline constexpr size_t kWireFloorWords = 64;

/*!
 * @brief Re-window a wire buffer for a new gate, keeping the slot storage the last gate paid for.
 *
 * `WindowVec::reset` assigns an empty vector into each existing element, which CLEARS a slot but
 * keeps its capacity -- so a buffer reused across gates never allocates again, and never gives a
 * byte back either. That is the whole cost model of a buffer that must outlive its gate: the
 * pair_exchange lifetime rule (PairExchange.h) forbids freeing it at the end of the gate that filled
 * it, so without a rule one wide gate pins its peak for the rest of the call.
 *
 * The rule is the engine's usual one (BucketJoin::begin_rows): a slot whose capacity has run past 4x
 * what it last held gives the storage back and re-earns it. The comparison uses the sizes still in
 * place, so it must run BEFORE the reset that clears them.
 */
inline auto reuse_wire(mpi::WindowVec<VecZ> &wire, mpi::SlotWindow window) -> void {
    for (VecZ &slot : wire) {
        if (slot.capacity() > 4 * std::max(slot.size(), kWireFloorWords)) {
            slot = VecZ{};
        }
    }
    wire.reset(window);
}

// COMMPROF's tallies (MonomialPropagator::run_gate_loop_ prints them): the wire volume of one call's
// gates, as this slot SENT it. Accumulated across the gates of a call, not reset per gate.
struct ExchangeCounters {
    size_t gates = 0;     // gates that exchanged, i.e. excluding the identity ones
    size_t records = 0;   // round-1 records pushed, self-staged included
    size_t responses = 0; // round-2 responses staged, the self slot's immediate ones included
};

// A trivial aggregate on purpose — not std::pair — so DefaultInitVector can skip the zero-fill and lower
// the gather to memmove.
struct PhasedEntry {
    size_t idx; // local target index for in_entries, local source index for out_entries
    int phase;
};

// One peer slot's rotation endpoints as the graph sink collects them (Engine.h GraphSink has the rules),
// drained into the LayerCore's sin_send/sin_recv lists by GraphSink::finalize as
// in = [in_pairs, in_mints] and out = [out_pairs, out_unanswered]. Replay pairs my out[j] with the
// peer's in[j] positionally, which the four lists' orders guarantee: in_pairs/in_mints are in the
// peer's stream order (ascending peer row), out_pairs/out_unanswered in ascending own row.
struct PartnerAcc {
    std::vector<PhasedEntry> in_pairs;       // my follower endpoints of pairs with a tracked partner
    std::vector<PhasedEntry> in_mints;       // partners this slot minted from a peer's E-record
    std::vector<PhasedEntry> out_pairs;      // my leader endpoints of pairs with a tracked partner
    std::vector<PhasedEntry> out_unanswered; // my E-records whose key missed at the peer (it minted)

    [[nodiscard]] auto in_count() const -> size_t { return in_pairs.size() + in_mints.size(); }
    [[nodiscard]] auto out_count() const -> size_t { return out_pairs.size() + out_unanswered.size(); }
};

// One half-rotation (Engine.h has the protocol): the add this slot owns of a rotation between ν and
// μ = ν ⊕ G, applied as op[local_idx] += sin·phase_signed·v_partner with v_partner the partner's pre-cos
// coefficient off the wire. Each slot is touched by exactly one add per gate (a term has one partner),
// so the order of the records is irrelevant to the result.
// Two words rather than three in the default build: the index is a row, so it is TermIndex-wide like every
// other row in the engine, and the phase is ternary. One gate's halves are pushed and then drained as two
// sequential streams over a buffer the join sizes exactly, so a third off the record is a third off the
// memory traffic of both passes.
struct HalfRotationRec {
    TermIndex local_idx = 0; // the slot this rank owns
    int8_t phase_signed = 0; // +φ_rec for a hit or mint, −φ_own for an absent partner: pre-signed so the
                             // apply never negates
    // A mint writes a slot inserted this gate (after the fused cos sweep), so the apply folds the gate's
    // cos in (c = cos·c + sin·φ·v) instead of a plain add. False for every pre-gate term.
    bool is_insert = false;
    double v_partner = 0.0; // partner's pre-cos coefficient: v_rec for a hit or mint, the answering row's
                            // own coefficient for a response, c0(μ) for an absence
};
// Only the default 32-bit TermIndex can hold the layout; monoprop_WIDE_TERM_INDEX pays a word for it.
static_assert(sizeof(TermIndex) != sizeof(uint32_t) || sizeof(HalfRotationRec) == 16,
              "the apply streams these, so the packing is the point");

// Sink threaded through build_layer's fused branch (ContractImmediately); a non-null FusedContract*
// selects the fused path. Drained by apply_fused_contract.
struct FusedContract {
    std::vector<HalfRotationRec> halves;
};

// bit_cast, not a conversion, so v_src arrives over the wire bit-identical.
static_assert(sizeof(size_t) == sizeof(double), "fused query value word assumes 64-bit VecZ element");
inline auto encode_value(double v) -> size_t {
    return std::bit_cast<size_t>(v);
}
inline auto decode_value(size_t word) -> double {
    return std::bit_cast<double>(word);
}

//! Appends one round-2 response, `kResponseWords` words wide.
inline auto push_response(VecZ &buf, size_t sent_idx, double v) -> void {
    buf.push_back(sent_idx);
    buf.push_back(encode_value(v));
}

} // namespace monoprop::detail
