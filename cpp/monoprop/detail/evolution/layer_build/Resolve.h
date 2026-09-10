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

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/TableJoin.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowKey.h"

namespace monoprop::detail {

// The join side of the gate exchange (Engine.h states the protocol). Everything here is
// picture-independent; what a hit, a mint, a response or an absence records is supplied by a
// compile-time sink with these surfaces:
//   hit(slot, row, v, phase, foll, own_rot)  a record hit tracked row `row` and the pair rotates.
//                                       `foll` and `own_rot` are that row's own pivot bit and E(row):
//                                       the value picture needs both to tell which endpoint of the pair
//                                       this half is (GateSinks.h ContractSink::hit says why).
//   mint(slot, idx, v, phase)           a rot=1 record missed: its key is inserted at `idx`
//   out_pair(slot, row, phase)          a leader of this slot whose partner is tracked and rotates
//   out_unanswered(slot, row, c0, phase) an E-record of this slot whose key missed at `slot`
// and, iff `Sink::wants_responses`, the round-2 pair:
//   silent_value(row) -> double         row's pre-gate value, the payload of a response
//   answer(slot, row, v, phase)         a response arrived for this slot's row: its half is -phase*v
// plus, once per gate before any of them:
//   reserve_halves(n)                   at most n of the recording calls above will follow this gate
// and, iff the sink declares `static constexpr bool pairs_once`, join_self may settle a mutual pair from
// the leader's record alone, pushing the follower's half through answer() (see join_self).
// `slot` is always the flat slot of the peer the record came from / went to.
//
// How a partner is found is entirely TableJoin's business: the join reads `join.hit(q)` and nothing
// else, so replacing the matching structure leaves the receiver rule, the mint order and the absence
// pass untouched.

// One read-only record stream per sender slot, ascending. Every producer of a gate's incoming records
// reaches the decode through it: today the collective's receive buffer, whose slots are viewed in place.
using WireStreams = std::span<const std::span<const size_t>>;

//! Views of a per-slot wire buffer's slots, ascending, into caller-owned storage.
inline auto slot_streams(const std::vector<VecZ> &wire, std::vector<std::span<const size_t>> &into) -> WireStreams {
    into.resize(wire.size());
    for (size_t s = 0; s < wire.size(); ++s) {
        into[s] = std::span<const size_t>(wire[s]);
    }
    return {into};
}

/*! @brief Decodes every incoming record into @a pr: key positions, phase, rot bit, value, and the key's
 *  join tag folded off those positions.
 *
 *  The tag is the same label XOR the sender's own key took, kept as the 32-bit tag, so the wire carries
 *  neither and sender and receiver agree by using one map. Nothing is looked up here -- the keys go into
 *  the gate's TableJoin, which probes them against the operator's term table.
 *
 *  Each record's header is decoded ONCE and reused for its width, its value word and its fields, and the
 *  tag is folded inside the same loop, while the positions this record owns are still in cache.
 */
template <size_t NumModes>
auto decode_incoming_records(WireStreams incoming, QueryForm form, IncomingRecords<NumModes> &pr) -> void {
    using QW = QueryWire<NumModes>;
    using PosT = typename IncomingRecords<NumModes>::PosT;
    const size_t senders = incoming.size();

    pr.goff.assign(senders + 1, 0);
    for (size_t s = 0; s < senders; ++s) {
        pr.goff[s + 1] = pr.goff[s] + QW::count_queries(incoming[s], form);
    }
    pr.nq_total = pr.goff[senders];
    if (pr.nq_total == 0) {
        return;
    }

    pr.phase_of.resize(pr.nq_total);
    pr.rot_of.resize(pr.nq_total);
    if (form == QueryForm::Fused) {
        pr.val_of.resize(pr.nq_total);
    }
    pr.pos_off.resize(pr.nq_total);
    pr.k_of.resize(pr.nq_total);
    pr.tag_of.resize(pr.nq_total);
    pr.pos_flat.clear();
    // A hint only, so this stays one allocation for the common case.
    pr.pos_flat.reserve(pr.nq_total * QW::kReservePositionsPerQuery);
    const uint64_t *const labels = routing::linear_basis<2 * NumModes>().data();
    for (size_t s = 0; s < senders; ++s) {
        const std::span<const size_t> buf = incoming[s];
        size_t off = 0;
        for (size_t g = pr.goff[s]; g < pr.goff[s + 1]; ++g) {
            const auto h = QW::header_at(buf, off);
            const size_t at = pr.pos_flat.size();
            pr.pos_flat.resize(at + h.k); // default-init grow: read_query writes every element
            pr.pos_off[g] = at;
            pr.k_of[g] = static_cast<uint16_t>(h.k);
            if (form == QueryForm::Fused) {
                pr.val_of[g] = QW::value_at(buf, off, h);
            }
            const auto d = QW::read_query(buf, form, off, h, std::span<PosT>(pr.pos_flat).subspan(at, h.k));
            pr.phase_of[g] = static_cast<int8_t>(d.phase);
            pr.rot_of[g] = static_cast<uint8_t>(d.rot);
            pr.tag_of[g] = join_tag(routing::fingerprint_positions(labels, pr.pos_flat.data() + at, h.k));
            off = d.next;
        }
    }
}

/*! @brief Bulk insert of the misses into rows [base, base + n).
 *
 *  insert_absent_terms' steps without its dense round-trip, on the same ordering contract. @a base must
 *  be the size the join assigned indices against.
 */
template <size_t NumModes>
auto insert_misses(MPOperator<NumModes> &op, const MissStage<NumModes> &misses, size_t base) -> void {
    const size_t n = misses.size();
    if (n == 0) {
        return;
    }
    [[maybe_unused]] const size_t grown_at = op.store->grow_rows_geometric(n);
    assert(grown_at == base && "misses were indexed against a size the store no longer has");
    for (size_t j = 0; j < n; ++j) {
        op.store->set_positions(base + j, misses.positions_at(j));
    }
    op.reindex_after_growth(base, n);
}

/*! @brief The receiver rule for one record (key = the partner, from the owner of its source at @a slot).
 *
 *  hit  -> the partner is tracked: mark received (and its rot); the pair rotates iff rot_rec or rot_own,
 *          and then this slot's half is +phase_rec * v_rec onto the row.
 *  miss -> the partner is absent everywhere; a rot=1 record mints it at base + j with the same half. A
 *          rot=0 record that misses is dropped: neither side rotates, so there is nothing to mint.
 *
 *  @return Whether the sender still owes its own half a value, i.e. true iff the pair rotates on the
 *  record's `rot` alone -- the partner is tracked but silent, which is the case round 2 answers
 *  (Engine.h). Always false unless the sink asked for responses; graph mode's silent rows send records
 *  of their own instead.
 */
template <size_t NumModes, typename Sink>
[[gnu::always_inline]] inline auto join_record(RowMarks &marks,
                                               size_t slot,
                                               size_t row,
                                               std::span<const typename OperatorIndex<NumModes>::PosT> pos,
                                               int phase,
                                               bool rot,
                                               double v,
                                               size_t base,
                                               MissStage<NumModes> &misses,
                                               Sink &sink) -> bool {
    if (row != TableJoin<NumModes>::kMissing) {
        marks.set_received(row);
        if (rot) {
            marks.set_partner_rot(row);
        }
        const bool own_rot = marks.rot(row);
        if (rot || own_rot) {
            sink.hit(slot, row, v, phase, marks.foll(row), own_rot);
        }
        return Sink::wants_responses && rot && !own_rot;
    }
    if (!rot) {
        return false;
    }
    const size_t idx = base + misses.size();
    misses.push(pos);
    sink.mint(slot, idx, v, phase);
    return false;
}

//! Whether `Sink` settles mutual pairs from the leader's record alone (join_self). False unless declared.
template <typename Sink>
[[nodiscard]] consteval auto sink_pairs_once() -> bool {
    if constexpr (requires { Sink::pairs_once; }) {
        static_assert(!Sink::pairs_once || Sink::wants_responses,
                      "pair-once reads the follower's swept slot through silent_value/answer");
        return Sink::pairs_once;
    }
    else {
        return false;
    }
}

/*! @brief The records this slot addressed to itself, in stream order.
 *
 *  They occupy the join's query indices [q_base, q_base + stage.size()), which is the front of the query
 *  space: the self stage is joined first, so its misses take the first mint indices.
 *
 *  `sent_self` is the parallel record of what those queries were sent for (the source row and its phase),
 *  which is why a silent hit needs no response here: both halves are on this slot, so the sender's is
 *  applied at once instead of travelling. `answered` is still marked, so the absence pass reads the pair
 *  alike however the answer arrived.
 *
 *  PAIR-ONCE (`pair_once`, a sink that declares `pairs_once`). A mutual pair -- both endpoints rotating,
 *  so each one's record hits the other's source -- is settled by the LEADER's record when that record is
 *  resolved first: it pushes its own half and, in place of the follower's record, the follower's half
 *  {source, -phase, silent_value(row)}. That is the very half the follower's record would have
 *  delivered, because phase_foll = -phase_lead (Engine.h) and the follower's record would have recovered
 *  fl(fl(v*cos)*inv_cos) = op_coeffs[row]*inv_cos = silent_value(row) from the same swept slot. The
 *  follower's record is then recognised by `received(source) && foll(source)` -- only its partner's
 *  record can set `received` on its source, keys being injective per gate -- and is skipped here and at
 *  the probe (TableJoin::run's skip, on `matched`). In the follower-first order both records take the
 *  ordinary arm, so the result is the same and only the saving is lost. Mint order is untouched: a
 *  skipped record counts as a hit, and hits never mint.
 *
 *  @return How many self records were answered without the wire.
 */
template <size_t NumModes, typename Sink>
auto join_self(const SelfQueryStage<NumModes> &stage,
               const TableJoin<NumModes> &join,
               size_t q_base,
               RowMarks &marks,
               size_t my_rank,
               size_t base,
               MissStage<NumModes> &misses,
               Sink &sink,
               std::span<const SentRecord> sent_self,
               bool pair_once = false) -> size_t {
    assert(!Sink::wants_responses || sent_self.size() == stage.size());
    assert((!pair_once || sink_pairs_once<Sink>()) && "pair-once needs a sink that declares it");
    size_t answered = 0;
    for (size_t q = 0; q < stage.size(); ++q) {
        [[maybe_unused]] size_t src = 0;
        [[maybe_unused]] bool src_received = false;
        if constexpr (sink_pairs_once<Sink>()) {
            if (pair_once) {
                src = sent_self[q].row;
                src_received = marks.received(src);
                // A skipped probe implies the leader's record settled this pair before this one was reached.
                assert(!join.skipped(q_base + q) || (src_received && marks.foll(src)));
                if (src_received && marks.foll(src)) {
                    continue; // the follower's record of a settled pair: both halves are already pushed
                }
            }
        }
        // One read of the join's outcome per query, whichever arm takes it; a skipped query has none.
        const size_t row = join.hit(q_base + q);
        if constexpr (sink_pairs_once<Sink>()) {
            if (pair_once && !src_received && row != TableJoin<NumModes>::kMissing && marks.rot(row)
                && marks.foll(row)) {
                // The leader's record onto a rotating follower whose own record is still to come.
                marks.set_received(row);
                marks.set_partner_rot(row);
                marks.set_received(src);
                marks.set_partner_rot(src);
                const int phase = static_cast<int>(stage.phase_of[q]);
                sink.hit(my_rank, row, stage.value_at(q), phase, /*foll=*/true, /*own_rot=*/true);
                sink.answer(my_rank, src, sink.silent_value(row), phase);
                continue;
            }
        }
        const bool answer = join_record<NumModes>(marks,
                                                  my_rank,
                                                  row,
                                                  stage.positions_at(q),
                                                  static_cast<int>(stage.phase_of[q]),
                                                  stage.rot_of[q] != 0,
                                                  stage.value_at(q),
                                                  base,
                                                  misses,
                                                  sink);
        if constexpr (Sink::wants_responses) {
            if (answer) {
                const SentRecord &s = sent_self[q];
                marks.set_answered(s.row);
                sink.answer(my_rank, s.row, sink.silent_value(row), static_cast<int>(s.phase));
                ++answered;
            }
        }
    }
    return answered;
}

/*! @brief The records the exchange delivered: sources in ascending slot order, each in its stream order.
 *
 *  That order is what pairs a peer's in-mints with this slot's out-unanswered positionally in graph mode.
 *  A hit on a silent row stages a response to that record's sender, naming the record by its position in
 *  the sender's own stream to this slot -- the one index both sides can compute without agreeing on
 *  anything but the order they already agree on.
 *
 *  @return How many responses were staged, so COMMPROF needs no second pass over the buffers.
 */
template <size_t NumModes, typename Sink>
auto join_incoming(const IncomingRecords<NumModes> &pr,
                   const TableJoin<NumModes> &join,
                   size_t q_base,
                   RowMarks &marks,
                   size_t base,
                   MissStage<NumModes> &misses,
                   Sink &sink,
                   std::vector<VecZ> &responses) -> size_t {
    size_t staged = 0;
    for (size_t s = 0; s + 1 < pr.goff.size(); ++s) {
        for (size_t g = pr.goff[s]; g < pr.goff[s + 1]; ++g) {
            const size_t row = join.hit(q_base + g);
            // The value column exists only for the fused form, and that is a property of the sink, so
            // the branch value_at() would cost per record is settled at compile time here.
            double v = 0.0;
            if constexpr (Sink::wants_values) {
                v = pr.val_of[g];
            }
            const bool answer = join_record<NumModes>(marks,
                                                      s,
                                                      row,
                                                      pr.positions_at(g),
                                                      static_cast<int>(pr.phase_of[g]),
                                                      pr.rot_of[g] != 0,
                                                      v,
                                                      base,
                                                      misses,
                                                      sink);
            if constexpr (Sink::wants_responses) {
                if (answer) {
                    push_response(responses[s], g - pr.goff[s], sink.silent_value(row));
                    ++staged;
                }
            }
        }
    }
    return staged;
}

/*! @brief Round 2's arrival: response j from slot s names one of the records THIS slot sent to s.
 *
 *  It names it by its position in that stream, and carries the answering row's pre-gate coefficient. The
 *  sender's half is -phase * v, the phase being the one it kept in `sent` -- the record itself carried no
 *  row, and none had to.
 */
template <size_t NumModes, typename Sink>
auto apply_responses(RowMarks &marks,
                     const std::vector<std::vector<SentRecord>> &sent,
                     WireStreams responses,
                     Sink &sink) -> void {
    assert(responses.size() == sent.size() && "one answer stream per slot");
    for (size_t s = 0; s < responses.size(); ++s) {
        const std::span<const size_t> buf = responses[s];
        const std::vector<SentRecord> &records = sent[s];
        for (size_t off = 0; off + kResponseWords <= buf.size(); off += kResponseWords) {
            const size_t idx = buf[off];
            assert(idx < records.size() && "a response named a record this slot never sent to that peer");
            const SentRecord &r = records[idx];
            marks.set_answered(r.row);
            sink.answer(s, r.row, decode_value(buf[off + 1]), static_cast<int>(r.phase));
        }
    }
}

/*! @brief The walk over this slot's own records after the join and after round 2, per destination slot in
 *  stream order (ascending ordinal).
 *
 *  A record gets exactly one of three answers, and they are what distinguishes an absent partner from a
 *  tracked one:
 *   received  the partner sent a record of its own, so it is tracked and rotates; both halves are applied.
 *   answered  the partner is tracked but silent, and sent its coefficient back; the sender's half came
 *             from that.
 *   neither   nobody could send this key but the partner's owner, and it does whenever the partner is
 *             tracked, so the partner is absent -- minted by its owner from this very record -- and the
 *             sender's own half (-phase * c0) is supplied here. That is the absence proper.
 *
 *  A leader whose partner is tracked and whose pair rotates is reported as the out-side of that pair; the
 *  graph sink pairs it positionally with the peer's in-pair, the fused sink ignores it (its half arrived
 *  with the partner's record).
 */
template <size_t NumModes, typename Sink>
auto absence_pass(const RowMarks &marks,
                  const std::vector<std::vector<SentRecord>> &sent,
                  const std::vector<std::vector<double>> &sent_c0,
                  Sink &sink) -> void {
    const bool has_c0 = sent_c0.size() == sent.size();
    for (size_t s = 0; s < sent.size(); ++s) {
        const std::vector<SentRecord> &records = sent[s];
        for (size_t j = 0; j < records.size(); ++j) {
            const size_t row = static_cast<size_t>(records[j].row);
            const int phase = static_cast<int>(records[j].phase);
            const bool received = marks.received(row);
            if (marks.rot(row) && !received && !marks.answered(row)) {
                sink.out_unanswered(s, row, has_c0 ? sent_c0[s][j] : 0.0, phase);
            }
            else if (received && !marks.foll(row) && (marks.rot(row) || marks.partner_rot(row))) {
                sink.out_pair(s, row, phase);
            }
        }
    }
}

} // namespace monoprop::detail
