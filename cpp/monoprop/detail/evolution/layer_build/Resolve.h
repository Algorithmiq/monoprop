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
#include "monoprop/detail/evolution/layer_build/AntiTable.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
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

// The records one exchange delivered, decoded and probed. Record g belongs to the sender at window index
// k iff goff[k] <= g < goff[k+1]; within a sender they are in the sender's stream order.
template <size_t NumModes>
struct IncomingProbe {
    // The operator store's position width, not the wire's: these positions exist to become rows.
    using PosT = typename OperatorIndex<NumModes>::PosT;
    using Ordinal = typename AntiTable<NumModes>::Ordinal;

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
    // g → the key's ordinal in the receiver's AntiTable (AntiTable::kNone for a miss).
    DefaultInitVector<Ordinal> ord_of;

    //! Record g's ascending positions, as a view into pos_flat.
    [[nodiscard]] auto positions_at(size_t g) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[g], k_of[g]);
    }
    [[nodiscard]] auto value_at(size_t g) const -> double { return val_of.empty() ? 0.0 : val_of[g]; }
};

// Decode every incoming record and look its key up in the receiver's own AntiTable for this gate -- the
// fingerprint recomputed off the decoded positions, so the wire carries none -- because a tracked partner
// of an anticommuting term is itself anticommuting (AntiTable.h). Read-only on the operator.
//
// This is the join's one lookup seam: ord_of is the only thing join_incoming reads that depends on HOW
// a partner is found, and join_self's table.probe is the same seam for the staged records. A different
// structure (e.g. a table over the received records probed by one streaming pass over the anticommuting
// ordinals) replaces this loop and leaves the join, the mint order and the absence pass untouched.
template <size_t NumModes>
auto probe_incoming_queries(const mpi::WindowVec<VecZ> &incoming, // serialized, one VecZ per sender slot
                            const MPOperator<NumModes> &op,
                            QueryForm form,
                            const AntiTable<NumModes> &table) -> IncomingProbe<NumModes> {
    using QW = QueryWire<NumModes>;
    using PosT = typename IncomingProbe<NumModes>::PosT;
    IncomingProbe<NumModes> pr;
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
    pr.ord_of.resize(pr.nq_total);
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
        pr.ord_of[g] = table.probe(*op.store, routing::fingerprint_positions(labels, pos.data(), pos.size()), pos);
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
[[gnu::always_inline]] inline auto join_record(AntiTable<NumModes> &table,
                                               size_t slot,
                                               typename AntiTable<NumModes>::Ordinal ord,
                                               std::span<const typename OperatorIndex<NumModes>::PosT> pos,
                                               int phase,
                                               bool rot,
                                               double v,
                                               size_t base,
                                               MissStage<NumModes> &misses,
                                               Sink &sink) -> void {
    if (ord != AntiTable<NumModes>::kNone) {
        table.set_received(ord);
        if (rot) {
            table.set_partner_rot(ord);
        }
        if (rot || table.rot(ord)) {
            sink.hit(slot, table.row_of(ord), v, phase, table.foll(ord));
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

// The records this slot addressed to itself, in stream order.
template <size_t NumModes, typename Sink>
auto join_self(const SelfQueryStage<NumModes> &stage,
               AntiTable<NumModes> &table,
               const OperatorIndex<NumModes> &store,
               size_t my_rank,
               size_t base,
               MissStage<NumModes> &misses,
               Sink &sink) -> void {
    for (size_t q = 0; q < stage.size(); ++q) {
        const auto pos = stage.positions_at(q);
        const auto ord = table.probe(store, stage.fp_of[q], pos);
        join_record<NumModes>(table,
                              my_rank,
                              ord,
                              pos,
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
auto join_incoming(const IncomingProbe<NumModes> &pr,
                   AntiTable<NumModes> &table,
                   size_t base,
                   MissStage<NumModes> &misses,
                   Sink &sink) -> void {
    for (size_t k = 0; k < pr.window.count; ++k) {
        const size_t slot = pr.window.slot(mpi::WindowIndex{k});
        for (size_t g = pr.goff[k]; g < pr.goff[k + 1]; ++g) {
            join_record<NumModes>(table,
                                  slot,
                                  pr.ord_of[g],
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
auto absence_pass(const AntiTable<NumModes> &table,
                  const mpi::WindowVec<std::vector<uint32_t>> &sent,
                  const mpi::WindowVec<std::vector<double>> &sent_c0,
                  Sink &sink) -> void {
    const mpi::SlotWindow w = sent.window();
    const bool has_c0 = sent_c0.size() == sent.size();
    for (size_t k = 0; k < w.count; ++k) {
        const mpi::WindowIndex wi{k};
        const size_t slot = w.slot(wi);
        const std::vector<uint32_t> &ords = sent[wi];
        for (size_t j = 0; j < ords.size(); ++j) {
            const auto ord = ords[j];
            const bool received = table.received(ord);
            if (table.rot(ord) && !received) {
                sink.out_unanswered(slot, table.row_of(ord), has_c0 ? sent_c0[wi][j] : 0.0, table.phase(ord));
            }
            else if (received && !table.foll(ord) && (table.rot(ord) || table.partner_rot(ord))) {
                sink.out_pair(slot, table.row_of(ord), table.phase(ord));
            }
        }
    }
}

} // namespace monoprop::detail
