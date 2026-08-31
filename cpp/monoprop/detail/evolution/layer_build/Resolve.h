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
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
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

    std::vector<size_t> goff;              // rank_count+1 flat offsets: g = goff[s] + q
    DefaultInitVector<uint32_t> sender_of; // g → sender rank
    DefaultInitVector<int> phase_of;       // g → query phase
    // g → word offset of that query inside incoming[sender_of[g]]; a query ordinal names no position.
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

    // Builds a dense bitset; only the fully paired minority of callers needs one.
    [[nodiscard]] auto mono_at(size_t g) const -> Monomial<NumModes> {
        Monomial<NumModes> m;
        const PosT *p = pos_flat.data() + pos_off[g];
        for (size_t j = 0; j < k_of[g]; ++j) {
            m.set(static_cast<size_t>(p[j]));
        }
        return m;
    }

    // Every mode of query g carries both Majoranas, read off the positions.
    [[nodiscard]] auto is_paired_at(size_t g) const -> bool {
        const PosT *p = pos_flat.data() + pos_off[g];
        const size_t k = k_of[g];
        return k == 2 * QueryWire<NumModes>::pair_count(p, k);
    }
};

// Read-only phases 1-2 of the exchange: `form` says whether the incoming records are fused
// (ContractSink) or plain (GraphSink). The caller runs phase 3, then insert_incoming_misses.
template <size_t NumModes>
auto probe_incoming_queries(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                            MPOperator<NumModes> &op,
                            size_t rank_count,
                            QueryForm form) -> IncomingProbe<NumModes> {
    using QW = QueryWire<NumModes>;
    IncomingProbe<NumModes> pr;

    pr.goff.assign(rank_count + 1, 0);
    for (size_t s = 0; s < rank_count; ++s) {
        const size_t nq = QW::count_queries(incoming[s], form);
        pr.goff[s + 1] = pr.goff[s] + nq;
    }
    pr.nq_total = pr.goff[rank_count];
    if (pr.nq_total == 0) {
        return pr;
    }

    pr.sender_of.resize(pr.nq_total);
    for (size_t s = 0; s < rank_count; ++s) {
        std::fill(pr.sender_of.begin() + static_cast<std::ptrdiff_t>(pr.goff[s]),
                  pr.sender_of.begin() + static_cast<std::ptrdiff_t>(pr.goff[s + 1]),
                  static_cast<uint32_t>(s));
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
    for (size_t s = 0; s < rank_count; ++s) {
        size_t off = 0;
        for (size_t g = pr.goff[s]; g < pr.goff[s + 1]; ++g) {
            int ph = 0;
            const size_t k = QW::k_at(incoming[s], off);
            const size_t at = pr.pos_flat.size();
            pr.pos_flat.resize(at + k); // default-init grow: read_query writes every element
            pr.pos_off[g] = at;
            pr.k_of[g] = static_cast<uint32_t>(k);
            pr.off_of[g] = off;
            off = QW::read_query(incoming[s], form, off, pr.pos_flat.data() + at, ph);
            pr.phase_of[g] = ph;
        }
        assert(off == incoming[s].size() && "the query walk did not consume the sender's whole buffer");
    }
    {
        const size_t op_size = op.store->size();
        op.store->find_batch_positions(pr.pos_flat.data(),
                                       pr.pos_off.data(),
                                       pr.k_of.data(),
                                       pr.nq_total,
                                       pr.idx_of.data(),
                                       pr.hash_of.data());
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
        op.store->set_positions(base + j, pr.pos_flat.data() + pr.pos_off[g], pr.k_of[g]);
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
auto resolve_incoming(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                      MPOperator<NumModes> &op,
                      size_t rank_count,
                      bool is_leader_pass,
                      MatchedEpochSet &matched,
                      size_t combined_size, // pre-layer op size: bounds the matched set
                      Sink &sink) -> std::vector<std::vector<typename Sink::Response>> {
    using Resp = typename Sink::Response;
    const IncomingProbe<NumModes> pr = probe_incoming_queries<NumModes>(incoming, op, rank_count, sink.incoming_form());
    std::vector<std::vector<Resp>> responses(rank_count);
    for (size_t s = 0; s < rank_count; ++s) {
        responses[s].assign(pr.goff[s + 1] - pr.goff[s], Sink::init_response());
    }
    if (pr.nq_total == 0) {
        return responses;
    }

    // Phase 3 (scatter): responses + sink records + matched-follower marks. Freshly inserted partners
    // (ip ≥ combined_size) skip the mark.
    sink.prepare(pr, rank_count, op, responses);
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t s = pr.sender_of[g];
        const size_t q = g - pr.goff[s];
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
// local rank was already resolved inline, so it is skipped here. inc_r[r][q] answers query q from rank r.
template <size_t NumModes, typename Sink>
auto process_responses(const std::vector<std::vector<typename Sink::Response>> &inc_r,
                       const std::vector<std::vector<size_t>> &src_idx,
                       const std::vector<VecZ> &queries, // serialized query buffers (for phase recovery)
                       size_t rank_count,
                       size_t my_rank,
                       Sink &sink) -> void {
    sink.process_reserve(inc_r, rank_count, my_rank);
    for (size_t r = 0; r < rank_count; ++r) {
        if (r == my_rank) {
            continue;
        }
        sink.on_response_block(r, inc_r[r], src_idx[r], queries[r]);
    }
}

} // namespace monoprop::detail
