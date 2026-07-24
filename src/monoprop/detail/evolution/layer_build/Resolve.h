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

#include <cstddef>
#include <limits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h" // is_paired / get_hf_mask (common) + algebra_hf_phase (fresh Schrödinger miss coeff)
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop::detail {

// resolve_incoming and its callers share this picture-independent probe/insert machinery: deserialize
// every incoming record, batch-find it, and assign each miss the next index base+j in a serial
// (sender,record)-order prefix — so the deterministic assignment (and multi-rank bit-exactness) cannot
// drift between resolvers. The per-query scatter (Phase 3) is supplied by the cross-rank sink.
// PARALLELISM (load-bearing): queries are source⊕G for globally-distinct sources and ⊕G is injective ⇒
// queries pairwise distinct ⇒ misses distinct and absent, so miss j gets base+j like a serial loop.
template <size_t NumModes>
struct IncomingProbe {
    std::vector<size_t> goff;                  // rank_count+1 flat offsets: g = goff[s] + q
    DefaultInitVector<uint32_t> sender_of;     // g → sender rank
    DefaultInitVector<Monomial<NumModes>> maj; // g → deserialized query monomial
    DefaultInitVector<int> phase_of;           // g → query phase
    DefaultInitVector<size_t> idx_of;          // g → resolved index (HIT: < base; MISS: base+j)
    std::vector<TermIndex> miss_g;             // j → the g that became miss j (Phase 4 reads maj[miss_g[j]])
    size_t base = 0;                           // op size before the miss inserts (the miss-index base)
    size_t nq_total = 0;
};

// Phases 1-2 for QUERY records: deserialize + batch-find every incoming record and assign miss indices
// (read-only w.r.t. operator contents). QW = per-record stride: the plain query width, or kQueryWordsFused
// for the fused resolver (trailing v_src word, read by the sink not here). The caller runs Phase-3, then
// insert_incoming_misses.
template <size_t NumModes, size_t QW = kQueryWords<NumModes>>
auto probe_incoming_queries(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                            MPOperator<NumModes> &op,
                            size_t rank_count) -> IncomingProbe<NumModes> {
    constexpr size_t W = QW;
    IncomingProbe<NumModes> pr;

    // Per-sender query counts and flat (sender-major, query-minor) offsets: g = goff[s] + q.
    pr.goff.assign(rank_count + 1, 0);
    for (size_t s = 0; s < rank_count; ++s) {
        const size_t nq = incoming[s].empty() ? 0 : incoming[s].size() / W;
        pr.goff[s + 1] = pr.goff[s] + nq;
    }
    pr.nq_total = pr.goff[rank_count];
    if (pr.nq_total == 0) {
        return pr;
    }

    // Deterministic PARALLEL resolve (see PARALLELISM above): probes run lock-free (table not mutated);
    // only the miss-rank prefix (Phase 2) is serial.
    pr.sender_of.resize(pr.nq_total);
    for (size_t s = 0; s < rank_count; ++s) {
        std::fill(pr.sender_of.begin() + static_cast<std::ptrdiff_t>(pr.goff[s]),
                  pr.sender_of.begin() + static_cast<std::ptrdiff_t>(pr.goff[s + 1]),
                  static_cast<uint32_t>(s));
    }

    // Phase 1 (parallel, read-only): deserialize, then probe with the group-prefetch batch find
    // (chunked so each task pipelines its own probes; the table is not mutated during this phase).
    pr.maj.resize(pr.nq_total);
    pr.phase_of.resize(pr.nq_total);
    pr.idx_of.resize(pr.nq_total);
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t s = pr.sender_of[g];
        const size_t q = g - pr.goff[s];
        Monomial<NumModes> m;
        int ph = 0;
        query_read<NumModes, QW>(incoming[s], q, m, ph);
        pr.maj[g] = m;
        pr.phase_of[g] = ph;
    }
    {
        const size_t op_size = op.store->size();
        op.store->find_batch(pr.maj.data(), pr.nq_total, pr.idx_of.data());
        for (size_t g = 0; g < pr.nq_total; ++g) {
            if (pr.idx_of[g] >= op_size) { // kNotFound is size_t max → also lands here
                pr.idx_of[g] = kMissingIndex;
            }
        }
    }

    // Phase 2 (serial prefix, (sender,query) order): each miss takes the next index base+j. miss_g[j]
    // records which query g became miss j, so Phase 4 reads the deserialized maj[miss_g[j]] directly.
    pr.base = op.store->size(); // LOCAL insert base into the op being mutated
    for (size_t g = 0; g < pr.nq_total; ++g) {
        if (pr.idx_of[g] == kMissingIndex) {
            pr.idx_of[g] = pr.base + pr.miss_g.size();
            pr.miss_g.push_back(static_cast<TermIndex>(g));
        }
    }
    return pr;
}

// Phase 4 (parallel bulk insert of the distinct absent terms): scatter majs into disjoint op slots
// [base, base+n_miss), insert keys into disjoint map shards, resync the inverted index — atomics-free.
// Call AFTER the caller's Phase-3 scatter, which reads pre-insert op_coeffs for hits and needs base == op.size().
template <size_t NumModes>
auto insert_incoming_misses(MPOperator<NumModes> &op, const IncomingProbe<NumModes> &pr) -> void {
    const size_t n_miss = pr.miss_g.size();
    if (n_miss == 0) {
        return;
    }
    // See insert_absent_terms. pr.base (captured at Phase 2) still equals op.size() here since no insert has
    // run; one writer per miss slot base+j, the staged maj read straight from the deserialization buffer.
    insert_absent_terms<NumModes>(
        op,
        n_miss,
        [&](size_t j) -> const Monomial<NumModes> & { return pr.maj[pr.miss_g[j]]; },
        [&](size_t j, size_t base) { assign_row<NumModes>(*op.store, base + j, pr.maj[pr.miss_g[j]]); });
}

// resolve_incoming / process_responses are the picture-independent cross-rank exchange skeletons. The
// per-query divergence — what each resolved query records and what value it answers with — is supplied by
// a compile-time cross-rank SINK. The two concrete sinks (GraphSink / ContractSink) live with the engine
// (Engine.h): each also carries the self-resolve + finalize policy, so one control flow serves both the
// graph-build and fused ContractImmediately paths. A sink must provide:
//   static constexpr size_t kStride;           // query record stride (kQueryWords / kQueryWordsFused)
//   using Response; static Response init_response();
//   send_buffer(queries, vals, scratch) -> std::vector<VecZ>&      // round-1 payload (plain / value-fused)
//   prepare(pr, rank_count, op, responses)                         // size the resolver-side records
//   on_resolved(g, s, q, ip, pr, incoming) -> Response            // record + answer one query
//   process_reserve(inc_r, rank_count, my_rank)                   // reserve the querier-side records
//   on_response_block(r, resp, srcs, qbuf)                        // fold one rank's answers back

// Resolver rank (any cross-rank sink): for each query from sender s, look up M' locally; found → answer
// with its index/value, absent → INSERT it in the SAME round (the resolver is the sole inserter of
// cross-rank absent terms). The sink's on_resolved supplies the per-query scatter + response; the
// matched-follower marks (leader pass) stay here so both pictures mark byte-identically.
template <size_t NumModes, class Sink>
auto resolve_incoming(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                      MPOperator<NumModes> &op,
                      size_t rank_count,
                      bool is_leader_pass,
                      MatchedEpochSet &matched,
                      size_t combined_size, // pre-layer op size: bounds the matched set
                      Sink &sink) -> std::vector<std::vector<typename Sink::Response>> {
    using Resp = typename Sink::Response;
    const IncomingProbe<NumModes> pr = probe_incoming_queries<NumModes, Sink::kStride>(incoming, op, rank_count);
    std::vector<std::vector<Resp>> responses(rank_count);
    for (size_t s = 0; s < rank_count; ++s) {
        responses[s].assign(pr.goff[s + 1] - pr.goff[s], Sink::init_response());
    }
    if (pr.nq_total == 0) {
        return responses;
    }

    // Phase 3 (parallel scatter): responses + sink records + matched-follower marks. Found indices are
    // distinct (≤1 writer per slot); freshly inserted partners (ip ≥ combined_size) skip the mark.
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
template <size_t NumModes, class Sink>
auto process_responses(const std::vector<std::vector<typename Sink::Response>> &inc_r,
                       const std::vector<std::vector<size_t>> &src_idx,
                       const std::vector<VecZ> &queries, // serialized query buffers (for phase recovery)
                       size_t rank_count,
                       size_t my_rank,
                       Sink &sink) -> void {
    sink.process_reserve(inc_r, rank_count, my_rank);
    for (size_t r = 0; r < rank_count; ++r) {
        if (r == my_rank) {
            continue; // local already handled inline
        }
        sink.on_response_block(r, inc_r[r], src_idx[r], queries[r]);
    }
}

} // namespace monoprop::detail
