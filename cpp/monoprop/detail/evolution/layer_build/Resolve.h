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
#include <limits>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/BucketJoin.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop::detail {

// The join side of the gate exchange (Engine.h states the protocol). Everything here is
// picture-independent; what a hit, a mint or an absence records is supplied by a compile-time sink with
// four surfaces:
//   hit(slot, row, v, phase, foll)      a record hit tracked row `row` and the pair rotates
//   mint(slot, idx, v, phase)           a rot=1 record missed: its key is inserted at `idx`
//   out_pair(slot, row, phase)          a leader of this slot whose partner is tracked and rotates
//   out_unanswered(slot, row, c0, phase) an E-record of this slot whose key missed at `slot`
// `slot` is always the flat slot of the peer the record came from / went to.
//
// How a partner is found is entirely BucketJoin's business: the join reads `join.hit(q)` and nothing
// else, so replacing the matching structure leaves the receiver rule, the mint order and the absence
// pass untouched.

// The records one exchange delivered, decoded. Record g belongs to the sender at window index k iff
// goff[k] <= g < goff[k+1]; within a sender they are in the sender's stream order.
template <size_t NumModes>
struct IncomingRecords {
    // The operator store's position width, not the wire's: these positions exist to become rows.
    using PosT = typename OperatorIndex<NumModes>::PosT;

    mpi::SlotWindow window;
    std::vector<size_t> goff;           // window.count+1 flat offsets: g = goff[k] + q
    DefaultInitVector<int8_t> phase_of; // g → record phase
    DefaultInitVector<uint8_t> rot_of;  // g → record rot bit
    DefaultInitVector<double> val_of;   // g → record value; sized only for the Fused form
    size_t nq_total = 0;

    // The keys as they arrived, flat: record g owns pos_flat[pos_off[g] .. pos_off[g] + k_of[g]).
    DefaultInitVector<PosT> pos_flat;
    DefaultInitVector<size_t> pos_off;
    DefaultInitVector<uint32_t> k_of;
    // g → the key's routing fingerprint, recomputed from the positions: the join's key for this record.
    DefaultInitVector<uint64_t> fp_of;

    //! Record g's ascending positions, as a view into pos_flat.
    [[nodiscard]] auto positions_at(size_t g) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[g], k_of[g]);
    }
    [[nodiscard]] auto value_at(size_t g) const -> double { return val_of.empty() ? 0.0 : val_of[g]; }
};

// Decode every incoming record: its key positions, phase, rot bit, value, and the key's fingerprint
// recomputed off those positions (so the wire carries none, and sender and receiver agree by using one
// function). Nothing is looked up here -- the keys go into the gate's BucketJoin, which matches them
// against the rows the scan staged, because a tracked partner of an anticommuting term is itself
// anticommuting (BucketJoin.h).
template <size_t NumModes>
auto decode_incoming_records(const mpi::WindowVec<VecZ> &incoming, // serialized, one VecZ per sender slot
                             QueryForm form) -> IncomingRecords<NumModes> {
    using QW = QueryWire<NumModes>;
    using PosT = typename IncomingRecords<NumModes>::PosT;
    IncomingRecords<NumModes> pr;
    pr.window = incoming.window();
    const size_t senders = pr.window.count;

    pr.goff.assign(senders + 1, 0);
    for (size_t k = 0; k < senders; ++k) {
        pr.goff[k + 1] = pr.goff[k] + QW::count_queries(incoming[mpi::WindowIndex{k}], form);
    }
    pr.nq_total = pr.goff[senders];
    if (pr.nq_total == 0) {
        return pr;
    }

    pr.phase_of.resize(pr.nq_total);
    pr.rot_of.resize(pr.nq_total);
    if (form == QueryForm::Fused) {
        pr.val_of.resize(pr.nq_total);
    }
    pr.pos_off.resize(pr.nq_total);
    pr.k_of.resize(pr.nq_total);
    pr.fp_of.resize(pr.nq_total);
    pr.pos_flat.clear();
    // A hint only, so this stays one allocation for the common case.
    pr.pos_flat.reserve(pr.nq_total * QW::kReservePositionsPerQuery);
    for (size_t si = 0; si < senders; ++si) {
        const VecZ &buf = incoming[mpi::WindowIndex{si}];
        size_t off = 0;
        for (size_t g = pr.goff[si]; g < pr.goff[si + 1]; ++g) {
            const size_t k = QW::k_at(buf, off);
            const size_t at = pr.pos_flat.size();
            pr.pos_flat.resize(at + k); // default-init grow: read_query writes every element
            pr.pos_off[g] = at;
            pr.k_of[g] = static_cast<uint32_t>(k);
            if (form == QueryForm::Fused) {
                pr.val_of[g] = QW::value_at(buf, off);
            }
            const auto d = QW::read_query(buf, form, off, std::span<PosT>(pr.pos_flat).subspan(at, k));
            pr.phase_of[g] = static_cast<int8_t>(d.phase);
            pr.rot_of[g] = static_cast<uint8_t>(d.rot);
            off = d.next;
        }
    }
    const uint64_t *const labels = routing::linear_basis<2 * NumModes>().data();
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const auto pos = pr.positions_at(g);
        pr.fp_of[g] = routing::fingerprint_positions(labels, pos.data(), pos.size());
    }
    return pr;
}

// The keys that missed, in join order: miss j becomes row base + j. Keys are source⊕G over globally
// distinct sources, ⊕G injective ⇒ keys pairwise distinct ⇒ misses distinct and absent, whichever slot
// or record they came from.
template <size_t NumModes>
struct MissStage {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    DefaultInitVector<PosT> pos_flat;
    std::vector<size_t> pos_off;
    std::vector<uint32_t> k_of;

    [[nodiscard]] auto size() const -> size_t { return k_of.size(); }
    auto clear() -> void {
        pos_flat.clear();
        pos_off.clear();
        k_of.clear();
    }
    auto push(std::span<const PosT> pos) -> void {
        pos_off.push_back(pos_flat.size());
        k_of.push_back(static_cast<uint32_t>(pos.size()));
        pos_flat.insert(pos_flat.end(), pos.begin(), pos.end());
    }
    [[nodiscard]] auto positions_at(size_t j) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[j], k_of[j]);
    }
};

// Bulk insert of the misses into rows [base, base+n): insert_absent_terms' steps without its dense
// round-trip, on the same ordering contract. `base` must be the size the join assigned indices against.
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

// The receiver rule for one record (key = μ, from the owner of ν = μ⊕G at `slot`) against the table:
//   hit  → the partner is tracked: mark received (and its rot); the pair rotates iff rot_rec ∨ rot_own,
//          and then this slot's half is +φ_rec · v_rec onto row(μ).
//   miss → μ is absent everywhere; a rot=1 record mints it at base + j with the same half. A rot=0
//          record that misses is dropped: neither side rotates, so there is nothing to mint.
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
                                               Sink &sink) -> void {
    if (row != BucketJoin<NumModes>::kMissing) {
        marks.set_received(row);
        if (rot) {
            marks.set_partner_rot(row);
        }
        if (rot || marks.rot(row)) {
            sink.hit(slot, row, v, phase, marks.foll(row));
        }
        return;
    }
    if (!rot) {
        return;
    }
    const size_t idx = base + misses.size();
    misses.push(pos);
    sink.mint(slot, idx, v, phase);
}

// The records this slot addressed to itself, in stream order. They occupy the join's query indices
// [q_base, q_base + stage.size()), which for the one-round protocol is the front of the query space:
// the self stage is joined first, so its misses take the first mint indices.
template <size_t NumModes, typename Sink>
auto join_self(const SelfQueryStage<NumModes> &stage,
               const BucketJoin<NumModes> &join,
               size_t q_base,
               RowMarks &marks,
               size_t my_rank,
               size_t base,
               MissStage<NumModes> &misses,
               Sink &sink) -> void {
    for (size_t q = 0; q < stage.size(); ++q) {
        join_record<NumModes>(marks,
                              my_rank,
                              join.hit(q_base + q),
                              stage.positions_at(q),
                              static_cast<int>(stage.phase_of[q]),
                              stage.rot_of[q] != 0,
                              stage.val_of[q],
                              base,
                              misses,
                              sink);
    }
}

// The records the exchange delivered: sources in ascending window order, each in its stream order. That
// order is what pairs a peer's in-mints with this slot's out-unanswered positionally in graph mode.
template <size_t NumModes, typename Sink>
auto join_incoming(const IncomingRecords<NumModes> &pr,
                   const BucketJoin<NumModes> &join,
                   size_t q_base,
                   RowMarks &marks,
                   size_t base,
                   MissStage<NumModes> &misses,
                   Sink &sink) -> void {
    for (size_t k = 0; k < pr.window.count; ++k) {
        const size_t slot = pr.window.slot(mpi::WindowIndex{k});
        for (size_t g = pr.goff[k]; g < pr.goff[k + 1]; ++g) {
            join_record<NumModes>(marks,
                                  slot,
                                  join.hit(q_base + g),
                                  pr.positions_at(g),
                                  static_cast<int>(pr.phase_of[g]),
                                  pr.rot_of[g] != 0,
                                  pr.value_at(g),
                                  base,
                                  misses,
                                  sink);
        }
    }
}

// The walk over this slot's own records after the join, per destination slot in stream order (ascending
// ordinal). Only the owner of μ can send key ν, and it does whenever μ is tracked, so
//     ¬received[ord(ν)]  ⟺  μ is absent.
// An E-record whose partner is absent was minted by its owner from that record; the sender's own half
// (−φ_ν · c0(μ)) is supplied here, which is the absence proper. A leader whose partner is tracked and
// whose pair rotates is reported as the out-side of that pair; the graph sink pairs it positionally with
// the peer's in-pair, the fused sink ignores it (its half arrived with the partner's record).
template <size_t NumModes, typename Sink>
auto absence_pass(const RowMarks &marks,
                  const mpi::WindowVec<std::vector<SentRecord>> &sent,
                  const mpi::WindowVec<std::vector<double>> &sent_c0,
                  Sink &sink) -> void {
    const mpi::SlotWindow w = sent.window();
    const bool has_c0 = sent_c0.size() == sent.size();
    for (size_t k = 0; k < w.count; ++k) {
        const mpi::WindowIndex wi{k};
        const size_t slot = w.slot(wi);
        const std::vector<SentRecord> &records = sent[wi];
        for (size_t j = 0; j < records.size(); ++j) {
            const size_t row = static_cast<size_t>(records[j].row);
            const int phase = static_cast<int>(records[j].phase);
            const bool received = marks.received(row);
            if (marks.rot(row) && !received) {
                sink.out_unanswered(slot, row, has_c0 ? sent_c0[wi][j] : 0.0, phase);
            }
            else if (received && !marks.foll(row) && (marks.rot(row) || marks.partner_rot(row))) {
                sink.out_pair(slot, row, phase);
            }
        }
    }
}

} // namespace monoprop::detail
