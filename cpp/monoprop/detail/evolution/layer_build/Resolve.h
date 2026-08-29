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
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop::detail {

// Picture-independent probe/insert machinery shared by resolve_incoming and its callers. Each miss takes
// the next index base+j in (sender,record) order, so the assignment (and multi-rank bit-exactness) cannot
// drift between resolvers. Queries are source⊕G over globally-distinct sources, ⊕G injective ⇒ queries
// pairwise distinct ⇒ misses distinct and absent.
template <size_t NumModes>
struct IncomingProbe {
    // The operator store's position width, not the wire's: these positions exist to become rows.
    using PosT = typename OperatorIndex<NumModes>::PosT;

    // The slots `incoming` covers; senders are named by their index into it, never by a flat slot.
    mpi::SlotWindow window;
    std::vector<size_t> goff;              // window.count+1 flat offsets: g = goff[k] + q
    DefaultInitVector<uint32_t> sender_wi; // g → sender's WINDOW index (see sender_index/sender_slot)
    DefaultInitVector<int> phase_of;       // g → query phase
    // g → word offset of that query inside its sender's buffer; a query ordinal names no position.
    DefaultInitVector<size_t> off_of;
    DefaultInitVector<size_t> idx_of; // g → resolved index (hit: < base; miss: base+j)
    std::vector<TermIndex> miss_g;    // j → the g that became miss j (Phase 4 reads the key of miss_g[j])
    size_t base = 0;                  // op size before the miss inserts (the miss-index base)
    size_t nq_total = 0;

    // The queries as they arrived, flat: query g owns pos_flat[pos_off[g] .. pos_off[g] + k_of[g]).
    DefaultInitVector<PosT> pos_flat;
    DefaultInitVector<size_t> pos_off;
    DefaultInitVector<uint32_t> k_of;
    // g → fold_hash of the query key, folded by the probe and reused by the insert.
    DefaultInitVector<uint32_t> hash_of;

    //! Query g's ascending positions, as a view into pos_flat.
    [[nodiscard]] auto positions_at(size_t g) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[g], k_of[g]);
    }

    //! The two ways to name query g's sender. sender_wi is re-based, so nothing else reads it.
    [[nodiscard]] auto sender_index(size_t g) const -> mpi::WindowIndex { return mpi::WindowIndex{sender_wi[g]}; }
    [[nodiscard]] auto sender_slot(size_t g) const -> size_t { return window.slot(sender_index(g)); }

    //! Builds a dense bitset; only the fully paired minority of callers needs one.
    [[nodiscard]] auto mono_at(size_t g) const -> Monomial<NumModes> {
        Monomial<NumModes> m;
        for (const PosT q : positions_at(g)) {
            m.set(static_cast<size_t>(q));
        }
        return m;
    }

    //! Every mode of query g carries both Majoranas, read off the positions.
    [[nodiscard]] auto is_paired_at(size_t g) const -> bool {
        const auto pos = positions_at(g);
        return pos.size() == 2 * QueryWire<NumModes>::pair_count(pos);
    }
};

// Read-only phases 1-2 of the exchange: `form` says whether the incoming records are fused
// (ContractSink) or plain (GraphSink). The caller runs phase 3, then insert_incoming_misses.
template <size_t NumModes>
auto probe_incoming_queries(const mpi::WindowVec<VecZ> &incoming, // serialized, one VecZ per sender slot
                            MPOperator<NumModes> &op,
                            QueryForm form) -> IncomingProbe<NumModes> {
    using QW = QueryWire<NumModes>;
    using PosT = typename IncomingProbe<NumModes>::PosT;
    IncomingProbe<NumModes> pr;
    pr.window = incoming.window();
    const size_t senders = pr.window.count;

    pr.goff.assign(senders + 1, 0);
    for (size_t k = 0; k < senders; ++k) {
        const size_t nq = QW::count_queries(incoming[mpi::WindowIndex{k}], form);
        pr.goff[k + 1] = pr.goff[k] + nq;
    }
    pr.nq_total = pr.goff[senders];
    if (pr.nq_total == 0) {
        return pr;
    }

    pr.sender_wi.resize(pr.nq_total);
    for (size_t k = 0; k < senders; ++k) {
        std::fill(pr.sender_wi.begin() + static_cast<std::ptrdiff_t>(pr.goff[k]),
                  pr.sender_wi.begin() + static_cast<std::ptrdiff_t>(pr.goff[k + 1]),
                  static_cast<uint32_t>(k));
    }

    // Phase 1 (read-only): deserialize, then probe with the group-prefetch batch find. One walk per sender.
    pr.phase_of.resize(pr.nq_total);
    pr.off_of.resize(pr.nq_total);
    pr.idx_of.resize(pr.nq_total);
    pr.pos_off.resize(pr.nq_total);
    pr.k_of.resize(pr.nq_total);
    pr.hash_of.resize(pr.nq_total);
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
            pr.off_of[g] = off;
            const auto d = QW::read_query(buf, form, off, std::span<PosT>(pr.pos_flat).subspan(at, k));
            pr.phase_of[g] = d.phase;
            off = d.next;
        }
    }
    {
        const size_t op_size = op.store->size();
        // The vectors are sized to capacity, not to nq_total, so every span is trimmed explicitly.
        op.store->find_batch_positions(std::span<const PosT>(pr.pos_flat),
                                       std::span<const size_t>(pr.pos_off).first(pr.nq_total),
                                       std::span<const uint32_t>(pr.k_of).first(pr.nq_total),
                                       std::span<size_t>(pr.idx_of).first(pr.nq_total),
                                       std::span<uint32_t>(pr.hash_of).first(pr.nq_total));
        for (size_t g = 0; g < pr.nq_total; ++g) {
            if (pr.idx_of[g] >= op_size) { // kNotFound is size_t max → also lands here
                pr.idx_of[g] = kMissingIndex;
            }
        }
    }

    // Phase 2 ((sender,query) prefix order): each miss takes the next index base+j.
    pr.base = op.store->size();
    for (size_t g = 0; g < pr.nq_total; ++g) {
        if (pr.idx_of[g] == kMissingIndex) {
            pr.idx_of[g] = pr.base + pr.miss_g.size();
            pr.miss_g.push_back(static_cast<TermIndex>(g));
        }
    }
    return pr;
}

// Phase 4 (bulk insert of the distinct absent terms) into op slots [base, base+n_miss). Call after the
// caller's Phase-3 scatter, which reads pre-insert op_coeffs for hits and needs base == op.size().
template <size_t NumModes>
auto insert_incoming_misses(MPOperator<NumModes> &op, const IncomingProbe<NumModes> &pr) -> void {
    const size_t n_miss = pr.miss_g.size();
    if (n_miss == 0) {
        return;
    }
    // insert_absent_terms' three steps without its two dense round-trips, and on the same ordering
    // contract, which is what matters: slot j lands at base+j, in miss order = (sender, record) order.
    const size_t base = op.store->grow_rows_geometric(n_miss);
    for (size_t j = 0; j < n_miss; ++j) {
        const size_t g = pr.miss_g[j];
        op.store->set_positions(base + j, pr.positions_at(g));
    }
    op.store->bulk_insert_hashed(n_miss, base, [&](size_t j) { return pr.hash_of[pr.miss_g[j]]; });
    op.reindex_after_growth(base, n_miss);
}

// resolve_incoming / process_responses are the picture-independent cross-rank exchange skeletons; what
// each resolved query records and answers with is supplied by a compile-time sink. The two concrete sinks
// (GraphSink / ContractSink) live in Engine.h, each also carrying the self-resolve + finalize policy.

// Resolver rank (any cross-rank sink): for each query from sender s, look up M' locally; found → answer
// with its index/value, absent → insert it in the same round (the resolver is the sole inserter of
// cross-rank absent terms). Matched-follower marks stay here so both sinks mark byte-identically.
template <size_t NumModes, typename Sink>
auto resolve_incoming(const mpi::WindowVec<VecZ> &incoming, // serialized, one VecZ per sender slot
                      MPOperator<NumModes> &op,
                      bool is_leader_pass,
                      MatchedEpochSet &matched,
                      size_t combined_size, // pre-layer op size: bounds the matched set
                      Sink &sink) -> mpi::WindowVec<std::vector<typename Sink::Response>> {
    using Resp = typename Sink::Response;
    const IncomingProbe<NumModes> pr = probe_incoming_queries<NumModes>(incoming, op, sink.incoming_form());
    // The response window is the query window: the pairing is an XOR involution, so a rank answers
    // exactly the slots it queried.
    mpi::WindowVec<std::vector<Resp>> responses(pr.window);
    for (size_t k = 0; k < pr.window.count; ++k) {
        responses[mpi::WindowIndex{k}].assign(pr.goff[k + 1] - pr.goff[k], Sink::init_response());
    }
    if (pr.nq_total == 0) {
        return responses;
    }

    // Phase 3 (scatter): responses + sink records + matched-follower marks. Freshly inserted partners
    // (ip ≥ combined_size) skip the mark.
    sink.prepare(pr, op, responses);
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const mpi::WindowIndex s = pr.sender_index(g);
        const size_t q = g - pr.goff[s.value];
        const size_t ip = pr.idx_of[g];
        responses[s][q] = sink.on_resolved(g, s, q, ip, pr, incoming);
        if (is_leader_pass && ip < combined_size) {
            matched.mark(ip);
        }
    }

    insert_incoming_misses<NumModes>(op, pr);
    return responses;
}

// Querier rank (any cross-rank sink): fold each resolver response into a querier-side record. The self/
// local slot was already resolved inline, so it is skipped here (and is in the window only when this
// generator's rank shift is zero). inc_r[k][q] answers query q sent to the window's k-th slot.
template <size_t NumModes, typename Sink>
auto process_responses(const mpi::WindowVec<std::vector<typename Sink::Response>> &inc_r,
                       const mpi::WindowVec<std::vector<size_t>> &src_idx,
                       const mpi::WindowVec<VecZ> &queries, // serialized query buffers (for phase recovery)
                       size_t my_rank,
                       Sink &sink) -> void {
    const mpi::SlotWindow w = inc_r.window();
    assert(src_idx.window().base == w.base && src_idx.window().count == w.count);
    assert(queries.window().base == w.base && queries.window().count == w.count);
    sink.process_reserve(inc_r, my_rank);
    for (size_t k = 0; k < w.count; ++k) {
        const mpi::WindowIndex wi{k};
        const size_t r = w.slot(wi);
        if (r == my_rank) {
            continue;
        }
        sink.on_response_block(r, inc_r[wi], src_idx[wi], queries[wi]);
    }
}

} // namespace monoprop::detail
