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
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop::detail {

// Picture-independent probe/insert machinery shared by resolve_incoming and its callers. Each miss takes
// the next index base+j in (sender,record) order, so the assignment (and multi-rank bit-exactness) cannot
// drift between resolvers. Queries are source⊕G over globally-distinct sources, ⊕G injective ⇒ queries
// pairwise distinct ⇒ misses distinct and absent.
// Key is whichever form the store being probed is keyed by (QueryKeysFor): a monomial for the dense store,
// a row-or-escape key for the support form. Named `Key` rather than fixed because the whole point of the
// two record forms is that a resolve never converts one into the other.
template <typename Key>
struct IncomingProbeT {
    std::vector<size_t> goff;              // rank_count+1 flat offsets: g = goff[s] + q
    DefaultInitVector<uint32_t> sender_of; // g → sender rank
    // Not owned: a view over the thread_local key batch probe_incoming_queries fills, which is why only
    // one IncomingProbe may be live per thread at a time (see the note there). Every element is
    // full-width and fully overwritten before anything reads it.
    std::span<const Key> mono;        // g → deserialized query key
    DefaultInitVector<int> phase_of;  // g → query phase
    DefaultInitVector<size_t> idx_of; // g → resolved index (hit: < base; miss: base+j)
    std::vector<TermIndex> miss_g;    // j → the g that became miss j (Phase 4 reads mono[miss_g[j]])
    size_t base = 0;                  // op size before the miss inserts (the miss-index base)
    size_t nq_total = 0;
};

// The probe over a store, spelled once so callers need not name the key form.
template <typename Store>
using IncomingProbeFor = IncomingProbeT<typename QueryKeysFor<Store>::type::key_type>;

// Phases 1-2, read-only w.r.t. operator contents. query_stride is the per-record width: the plain query
// width, or the fused one for the fused resolver. The caller runs Phase 3, then insert_incoming_misses.
// No width parameter: the monomials are built at `op.num_bits()`, which is the width of the rows they are
// probed against, so the probe monomials and those rows provably share a width. The stride stays an
// ordinary argument.
// inline is load-bearing: this is a plain function defined in a header, so without it every TU that
// includes this emits its own definition and the link fails.
template <typename Store>
inline auto probe_incoming_queries(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                                   MPOperator &op,
                                   Store &store,
                                   size_t rank_count,
                                   size_t query_stride,
                                   size_t record_capacity) -> IncomingProbeFor<Store> {
    using Keys = typename QueryKeysFor<Store>::type;
    IncomingProbeFor<Store> pr;

    pr.goff.assign(rank_count + 1, 0);
    for (size_t s = 0; s < rank_count; ++s) {
        // Off the buffer's own header, not its size: a support-form buffer carries an escape tail after its
        // records, so size/stride is not the record count.
        pr.goff[s + 1] = pr.goff[s] + query_record_count(incoming[s]);
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

    // Phase 1 (read-only): deserialize, then probe with the group-prefetch batch find.
    //
    // The keys come from a batch that outlives the call, grow-only, rather than being constructed per
    // resolve. Constructing them was measurable: a dense element is 72 bytes whose constructor zeroes all
    // eight inline words whatever the operator's real width, and above that width it allocates -- so a
    // layer paid nq_total of those, every layer, only to overwrite every word immediately (the record
    // reader overwrites whole, so no stale bit from a previous layer can survive). Reusing the buffer is
    // worth -64% to -72% of this phase at 2-8 words and -41% at 16.
    //
    // thread_local, matching the scan's `nz`: each partition master gets its own and reuses its
    // capacity across layers. Two consequences, both load-bearing:
    //   - At most one IncomingProbe may be live per thread. Only resolve_incoming builds one, holds it
    //     to the end of the call, and calls nothing that builds another; the self-resolve path has its
    //     own separate batch (Engine.h's keys_).
    //   - The extent is part of the batch's state, which is why configure() is called every time rather
    //     than once: a thread servicing two propagators of different storage widths must not reuse
    //     elements sized for the other, or the reader would write a wide record into a narrow element.
    // It holds the largest layer's worth of keys until the thread exits, where before it was
    // released after every layer. Peak RSS is unchanged -- the peak was always reached *during* a
    // layer -- but the resting footprint after one is not; the same trade `nz` and `keys_` already make.
    thread_local Keys scratch;
    scratch.configure(op.num_bits(), record_capacity);
    scratch.ensure(pr.nq_total);
    scratch.begin_batch();
    pr.mono = std::span<const typename Keys::key_type>(scratch.data(), pr.nq_total);
    pr.phase_of.resize(pr.nq_total);
    pr.idx_of.resize(pr.nq_total);
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t s = pr.sender_of[g];
        const size_t q = g - pr.goff[s];
        pr.phase_of[g] = scratch.read_record(incoming[s], q, query_stride, g);
    }
    {
        const size_t op_size = store.size();
        store.find_batch(pr.mono.data(), pr.nq_total, pr.idx_of.data());
        for (size_t g = 0; g < pr.nq_total; ++g) {
            if (pr.idx_of[g] >= op_size) { // kNotFound is size_t max → also lands here
                pr.idx_of[g] = kMissingIndex;
            }
        }
    }

    // Phase 2 ((sender,query) prefix order): each miss takes the next index base+j.
    pr.base = store.size();
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
// op/pr are deduced from their argument types (MPOperator/IncomingProbe).
auto insert_incoming_misses(auto &op, auto &store, const auto &pr) -> void {
    const size_t n_miss = pr.miss_g.size();
    if (n_miss == 0) {
        return;
    }
    insert_absent_terms(
        op,
        store,
        n_miss,
        [&](size_t j) -> decltype(auto) { return pr.mono[pr.miss_g[j]]; },
        [&](size_t j, size_t base) { assign_row(store, base + j, pr.mono[pr.miss_g[j]]); });
}

// resolve_incoming / process_responses are the picture-independent cross-rank exchange skeletons; what
// each resolved query records and answers with is supplied by a compile-time sink. The two concrete sinks
// (GraphSink / ContractSink) live in Engine.h, each also carrying the self-resolve + finalize policy.

// Resolver rank (any cross-rank sink): for each query from sender s, look up M' locally; found → answer
// with its index/value, absent → insert it in the same round (the resolver is the sole inserter of
// cross-rank absent terms). Matched-follower marks stay here so both sinks mark byte-identically.
// op is deduced (MPOperator, an argument type); Sink stays named -- it is referenced by name below
// (typename Sink::Response). No width is named anywhere here any more: the record stride is a member of
// the sink and the monomial width comes off the operator.
template <typename Sink>
auto resolve_incoming(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                      auto &op,
                      auto &store,
                      size_t rank_count,
                      bool is_leader_pass,
                      MatchedEpochSet &matched,
                      size_t combined_size, // pre-layer op size: bounds the matched set
                      Sink &sink) -> std::vector<std::vector<typename Sink::Response>> {
    using Resp = typename Sink::Response;
    const auto pr = probe_incoming_queries(incoming, op, store, rank_count, sink.stride(), sink.capacity());
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

    insert_incoming_misses(op, store, pr);
    return responses;
}

// Querier rank (any cross-rank sink): fold each resolver response into a querier-side record. The self/
// local rank was already resolved inline, so it is skipped here. inc_r[r][q] answers query q from rank r.
// Unlike resolve_incoming, nothing here touches the operator, so no argument needs deducing at all --
// Sink stays named for the same reason as above.
template <typename Sink>
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
