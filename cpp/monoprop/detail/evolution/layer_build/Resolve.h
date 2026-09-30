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
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

namespace monoprop::detail {

//! Queries per logical block of the frozen probe and the incoming decode: a function of the query count only.
inline constexpr size_t kProbeBlockQueries = 256;
//! Queries per logical block of the incoming on_resolved scatter.
inline constexpr size_t kResolveScatterBlock = 1024;
//! Most self queries probed per window; bounds the gathered scratch (at most 16 probe blocks per window).
inline constexpr size_t kSelfProbeWindow = 4096;

/*!
 * \brief Look up `out.size()` position-list queries in a frozen store, on up to `options.threads` workers.
 *
 * Query `q` is `pos_flat[pos_off[q], pos_off[q] + k_of[q])`; offsets are absolute into the one shared buffer.
 * The queries split into logical blocks of kProbeBlockQueries, each one OperatorIndex::find_batch_positions
 * call over its slice of `pos_off`, `k_of`, `out` and `hash_out`, so the batched pipeline, the exact collision
 * check and the cached hashes are the serial ones and the results are identical at every budget and team
 * size. `out[q]` is the row ID or OperatorIndex::kNotFound; `hash_out`, when not empty, receives each query's
 * fold hash. A single block runs on the caller.
 *
 * The store must stay frozen -- no row, overflow, table or index mutation -- until this returns; it returns
 * only after every worker has joined.
 *
 * \param observer Test seam; see NoRangeObserver. \param kind The KernelRange it reports.
 * \throws std::invalid_argument unless `pos_off`, `k_of` and `out` have equal lengths, `hash_out` is empty or of
 *         that length, and every query's span lies inside `pos_flat`; all checked before any block runs.
 */
template <size_t NumModes, class Observer = NoRangeObserver>
auto probe_frozen_positions(const OperatorIndex<NumModes> &store,
                            std::span<const typename OperatorIndex<NumModes>::PosT> pos_flat,
                            std::span<const size_t> pos_off,
                            std::span<const uint32_t> k_of,
                            std::span<size_t> out,
                            std::span<uint32_t> hash_out,
                            parallel::Options options = {},
                            const Observer &observer = {},
                            KernelRange kind = KernelRange::incoming_probe) -> void {
    const size_t n = out.size();
    if (pos_off.size() != n || k_of.size() != n || (!hash_out.empty() && hash_out.size() != n)) {
        throw std::invalid_argument(std::format("probe_frozen_positions: {} offsets, {} counts, {} outputs and {} "
                                                "hashes; the spans must have equal lengths (hashes may be empty)",
                                                pos_off.size(),
                                                k_of.size(),
                                                n,
                                                hash_out.size()));
    }
    for (size_t q = 0; q < n; ++q) {
        if (pos_off[q] > pos_flat.size() || k_of[q] > pos_flat.size() - pos_off[q]) {
            throw std::invalid_argument(std::format("probe_frozen_positions: query {} spans [{}, +{}) outside the "
                                                    "{}-position buffer",
                                                    q,
                                                    pos_off[q],
                                                    k_of[q],
                                                    pos_flat.size()));
        }
    }
    const size_t blocks = logical_ranges(n, kProbeBlockQueries);
    observer.prepare(kind, blocks);
    // Each block writes only its own slice of `out` and `hash_out`; the store and pos_flat are read-only.
    parallel::for_blocks(blocks, options, [&](size_t block) {
        observer.visit(kind, block);
        const size_t lo = block * kProbeBlockQueries;
        const size_t m = std::min(kProbeBlockQueries, n - lo);
        store.find_batch_positions(pos_flat,
                                   pos_off.subspan(lo, m),
                                   k_of.subspan(lo, m),
                                   out.subspan(lo, m),
                                   hash_out.empty() ? std::span<uint32_t>{} : hash_out.subspan(lo, m));
    });
}

// Picture-independent probe/insert machinery shared by resolve_incoming and its callers. Each miss takes
// the next index base+j in (sender,record) order, so the assignment (and multi-rank bit-exactness) cannot
// drift between resolvers. Queries are source⊕G over globally-distinct sources, ⊕G injective ⇒ queries
// pairwise distinct ⇒ misses distinct and absent.
template <size_t NumModes>
struct IncomingProbe {
    // The operator store's position width, not the wire's: these positions exist to become rows.
    using PosT = typename OperatorIndex<NumModes>::PosT;

    // The slots `incoming` covers; senders are named by WindowIndex into it.
    mpi::SlotWindow window;
    std::vector<size_t> goff;              // window.count+1 flat offsets: g = goff[k] + q
    DefaultInitVector<uint32_t> sender_wi; // g → sender's window index
    DefaultInitVector<int> phase_of;       // g → query phase
    // g → word offset of that query inside its sender's buffer; a query ordinal names no position.
    DefaultInitVector<size_t> off_of;
    DefaultInitVector<size_t> idx_of; // g → resolved index (hit: < base; miss: base+j)
    // j → the query ordinal g that became miss j (Phase 4 reads its key). Ordinals, not TermIndex values:
    // their range is the query count, not the store's.
    std::vector<size_t> miss_g;
    size_t base = 0; // op size before the miss inserts (the miss-index base)
    size_t nq_total = 0;

    // The queries as they arrived, flat: query g owns pos_flat[pos_off[g] .. pos_off[g] + k_of[g]). Every
    // per-query array holds exactly nq_total elements, pos_flat exactly the decoded positions.
    DefaultInitVector<PosT> pos_flat;
    DefaultInitVector<size_t> pos_off;
    DefaultInitVector<uint32_t> k_of;
    // g → fold_hash of the query key, folded by the probe and reused by the insert.
    DefaultInitVector<uint32_t> hash_of;

    //! Query g's ascending positions, as a view into pos_flat.
    [[nodiscard]] auto positions_at(size_t g) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[g], k_of[g]);
    }

    //! Query g's sender in `window`; window.slot() gives the flat slot.
    [[nodiscard]] auto sender_index(size_t g) const -> mpi::WindowIndex { return mpi::WindowIndex{sender_wi[g]}; }

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

namespace resolve_detail {

// Refuses a length the vector type cannot hold, before any allocation.
template <typename Vec>
auto check_fits(const Vec &v, size_t n, const char *what) -> void {
    if (n > v.max_size()) {
        throw std::length_error(std::format("probe_incoming_queries: {} {} exceed the vector limit", n, what));
    }
}

} // namespace resolve_detail

/*!
 * \brief Read-only phases 1-2 of the exchange: validate, decode and probe the received queries, then assign
 * each miss its predicted row ID.
 *
 * `incoming` holds one serialized stream per sender window slot: owning `VecZ` blocks as a transport unpacked them, or
 * read-only views of buffers another owner published (QueryWire::WireView); the streams are only read and must stay
 * unchanged until this returns. `form` says whether the records are fused (ContractSink) or plain (GraphSink). In
 * order:
 *   1. On the caller, every sender stream is walked in window order with QueryWire::checked_extent, so a
 *      malformed header, a noncanonical escape or a truncated record is rejected before any payload is read.
 *      The query and position prefixes, sender narrowing and vector limits are checked, every per-query array
 *      is allocated once at its exact size, and the offsets (absolute into `pos_flat`), counts and senders are
 *      filled in canonical sender-window/query order.
 *   2. Workers decode disjoint blocks of records into their prepared position spans; read_positions rejects
 *      positions outside the term's width, and each record must end exactly at its prepared boundary.
 *   3. After that join, probe_frozen_positions looks every query up in the frozen store, keeping its hash.
 *   4. After that join, the caller checks that base + misses stays below the row-ID ceiling, then assigns miss
 *      j the ID base + j in ascending sender-window/query order.
 * A failure in any phase leaves `op` untouched. The caller runs phase 3 (the scatter), then
 * insert_incoming_misses.
 *
 * \param observer Test seam reporting KernelRange::decode and KernelRange::incoming_probe; see NoRangeObserver.
 * \throws MalformedQueryStream for a malformed stream (from the caller before decoding, or from a decode worker
 *         for invalid positions, rethrown after the join and before probing).
 * \throws TermIndexCeilingReached if the misses would pass the row-ID ceiling.
 */
template <size_t NumModes, class Stream, class Observer = NoRangeObserver>
    requires std::convertible_to<const Stream &, typename QueryWire<NumModes>::WireView>
auto probe_incoming_queries(const mpi::WindowVec<Stream> &incoming, // serialized, one stream per sender slot
                            MPOperator<NumModes> &op,
                            QueryForm form,
                            parallel::Options options = {},
                            const Observer &observer = {}) -> IncomingProbe<NumModes> {
    using QW = QueryWire<NumModes>;
    using PosT = typename IncomingProbe<NumModes>::PosT;
    IncomingProbe<NumModes> pr;
    pr.window = incoming.window();
    pr.base = op.store->size();
    const size_t senders = pr.window.count;
    if (senders > size_t{std::numeric_limits<uint32_t>::max()} + 1U) {
        throw std::length_error(
            std::format("probe_incoming_queries: {} senders exceed the uint32 sender index", senders));
    }

    // 1a. Validate every stream and count its queries and positions, before any decode.
    resolve_detail::check_fits(pr.goff, senders + 1, "sender offsets");
    pr.goff.assign(senders + 1, 0);
    size_t positions = 0;
    for (size_t k = 0; k < senders; ++k) {
        const typename QW::WireView buf = incoming[mpi::WindowIndex{k}];
        size_t nq = 0;
        for (size_t off = 0; off < buf.size();) {
            const auto ext = QW::checked_extent(buf, form, off);
            if (ext.k > std::numeric_limits<size_t>::max() - positions) {
                throw std::length_error("probe_incoming_queries: the decoded position count overflows");
            }
            positions += ext.k;
            off += ext.words; // checked_extent guarantees off + words <= buf.size()
            ++nq;
        }
        // nq <= buf.size() words, but the running sum over senders is still checked.
        if (nq > std::numeric_limits<size_t>::max() - pr.goff[k]) {
            throw std::length_error("probe_incoming_queries: the query count overflows");
        }
        pr.goff[k + 1] = pr.goff[k] + nq;
    }
    pr.nq_total = pr.goff[senders];
    const size_t n = pr.nq_total;

    // 1b. Allocate every output once, at its exact size; workers never resize.
    resolve_detail::check_fits(pr.sender_wi, n, "queries");
    resolve_detail::check_fits(pr.phase_of, n, "queries");
    resolve_detail::check_fits(pr.off_of, n, "queries");
    resolve_detail::check_fits(pr.idx_of, n, "queries");
    resolve_detail::check_fits(pr.pos_off, n, "queries");
    resolve_detail::check_fits(pr.k_of, n, "queries");
    resolve_detail::check_fits(pr.hash_of, n, "queries");
    resolve_detail::check_fits(pr.pos_flat, positions, "positions");
    pr.sender_wi.resize(n);
    pr.phase_of.resize(n);
    pr.off_of.resize(n);
    pr.idx_of.resize(n);
    pr.pos_off.resize(n);
    pr.k_of.resize(n);
    pr.hash_of.resize(n);
    pr.pos_flat.resize(positions);

    // 1c. Canonical sender-window/query order: record offsets, counts and absolute position offsets. The headers
    // were validated above, so reading them again cannot leave the buffer.
    size_t pos_at = 0;
    for (size_t k = 0; k < senders; ++k) {
        const typename QW::WireView buf = incoming[mpi::WindowIndex{k}];
        size_t off = 0;
        for (size_t g = pr.goff[k]; g < pr.goff[k + 1]; ++g) {
            const auto h = QW::header_at(buf, off);
            pr.sender_wi[g] = static_cast<uint32_t>(k);
            pr.off_of[g] = off;
            pr.k_of[g] = static_cast<uint32_t>(h.k); // k <= kMaxPositions <= 65535
            pr.pos_off[g] = pos_at;
            pos_at += h.k;
            off += QW::words_of_header(h) + (form == QueryForm::Fused ? 1U : 0U);
        }
    }
    assert(pos_at == positions);

    // 2. Decode disjoint records into their prepared spans.
    const size_t blocks = logical_ranges(n, kProbeBlockQueries);
    observer.prepare(KernelRange::decode, blocks);
    parallel::for_blocks(blocks, options, [&](size_t block) {
        observer.visit(KernelRange::decode, block);
        const size_t hi = std::min(n, (block + 1) * kProbeBlockQueries);
        for (size_t g = block * kProbeBlockQueries; g < hi; ++g) {
            const size_t k = pr.sender_wi[g];
            const typename QW::WireView buf = incoming[mpi::WindowIndex{k}];
            const auto d = QW::read_query(buf,
                                          form,
                                          pr.off_of[g],
                                          std::span<PosT>(pr.pos_flat).subspan(pr.pos_off[g], pr.k_of[g]));
            const size_t boundary = (g + 1 < pr.goff[k + 1]) ? pr.off_of[g + 1] : buf.size();
            if (d.next != boundary) {
                throw MalformedQueryStream(
                    std::format("probe_incoming_queries: query {} ends at word {}, not at its prepared boundary {}",
                                g,
                                d.next,
                                boundary));
            }
            pr.phase_of[g] = d.phase;
        }
    });

    // 3. Frozen probe: nothing mutates op until insert_incoming_misses.
    probe_frozen_positions<NumModes>(*op.store,
                                     std::span<const PosT>(pr.pos_flat),
                                     std::span<const size_t>(pr.pos_off),
                                     std::span<const uint32_t>(pr.k_of),
                                     std::span<size_t>(pr.idx_of),
                                     std::span<uint32_t>(pr.hash_of),
                                     options,
                                     observer,
                                     KernelRange::incoming_probe);

    // 4. Phase 2 ((sender,query) prefix order): each miss takes the next index base+j. The row-ID limit is
    // checked before any predicted ID is written.
    size_t n_miss = 0;
    for (size_t g = 0; g < n; ++g) {
        n_miss += static_cast<size_t>(pr.idx_of[g] >= pr.base); // kNotFound is size_t max, so it lands here too
    }
    if (n_miss > OperatorIndex<NumModes>::kIndexCeiling - pr.base) {
        throw TermIndexCeilingReached(std::format("probe_incoming_queries: {} new terms on a store of {} would pass "
                                                  "the 2^32 TermIndex ceiling",
                                                  n_miss,
                                                  pr.base));
    }
    pr.miss_g.resize(n_miss);
    size_t j = 0;
    for (size_t g = 0; g < n; ++g) {
        if (pr.idx_of[g] >= pr.base) {
            pr.idx_of[g] = pr.base + j;
            pr.miss_g[j++] = g;
        }
    }
    return pr;
}

// Phase 4 (bulk insert of the distinct absent terms) into op slots [base, base+n_miss). Call after the
// caller's Phase-3 scatter, which reads pre-insert op_coeffs for hits and needs base == op.size(). Publication
// is single-writer: grow, set_positions, bulk_insert_hashed with the probe's cached hashes (`options` is its
// publication seam, which the packed index ignores), then the inverted-index maintenance.
template <size_t NumModes>
auto insert_incoming_misses(MPOperator<NumModes> &op, const IncomingProbe<NumModes> &pr, parallel::Options options = {})
    -> void {
    const size_t n_miss = pr.miss_g.size();
    if (n_miss == 0) {
        return;
    }
    // insert_absent_terms' three steps without its two dense round-trips, and on the same ordering
    // contract, which is what matters: slot j lands at base+j, in miss order = (sender, record) order.
    const size_t base = op.store->grow_rows_geometric(n_miss);
    assert(base == pr.base);
    for (size_t j = 0; j < n_miss; ++j) {
        const size_t g = pr.miss_g[j];
        op.store->set_positions(base + j, pr.positions_at(g));
    }
    op.store->bulk_insert_hashed(n_miss, base, [&](size_t j) { return pr.hash_of[pr.miss_g[j]]; }, options);
    op.reindex_after_growth(base, n_miss);
}

/*!
 * \brief Whether the leader hits of `pr` (rows below `combined_size`) are pairwise distinct, so their matched
 * marks are writes to distinct elements.
 *
 * Holds in production because queries are source XOR G over globally distinct sources; the parallel scatter
 * relies on it and debug builds assert it. O(hits log hits); for assertions and tests only.
 */
template <size_t NumModes>
auto leader_marks_unique(const IncomingProbe<NumModes> &pr, size_t combined_size) -> bool {
    std::vector<size_t> hits;
    for (size_t g = 0; g < pr.nq_total; ++g) {
        if (pr.idx_of[g] < combined_size) {
            hits.push_back(pr.idx_of[g]);
        }
    }
    std::ranges::sort(hits);
    return std::ranges::adjacent_find(hits) == hits.end();
}

/*!
 * \brief Whether `sink` certifies that its on_resolved calls for distinct queries may run concurrently.
 *
 * A sink opts in with `static constexpr bool parallel_resolve = true`, which certifies only independent
 * on_resolved writes after prepare(), never its other methods. A sink may further veto it at run time with
 * `parallel_resolve_enabled()`. A missing or false capability keeps the serial scatter.
 */
template <typename Sink>
auto sink_resolves_in_parallel(const Sink &sink) -> bool {
    if constexpr (requires { Sink::parallel_resolve; }) {
        if constexpr (static_cast<bool>(Sink::parallel_resolve)) {
            if constexpr (requires {
                              { sink.parallel_resolve_enabled() } -> std::convertible_to<bool>;
                          }) {
                return sink.parallel_resolve_enabled();
            }
            else {
                return true;
            }
        }
    }
    (void)sink;
    return false;
}

// resolve_incoming / process_responses are the picture-independent cross-rank exchange skeletons; what
// each resolved query records and answers with is supplied by a compile-time sink. The two concrete sinks
// (GraphSink / ContractSink) live in Engine.h, each also carrying the self-resolve + finalize policy.

// Resolver rank (any cross-rank sink): for each query from sender s, look up M' locally; found → answer
// with its index/value, absent → insert it in the same round (the resolver is the sole inserter of
// cross-rank absent terms). Matched-follower marks stay here so both sinks mark byte-identically.
// Threading: decode and probe as probe_incoming_queries; the scatter runs on workers only for a sink that
// certifies it (sink_resolves_in_parallel), otherwise on the caller; publication stays on the caller, after
// the scatter, which reads pre-insert coefficients. `observer` is the test seam (see NoRangeObserver).
//
// `incoming` is as for probe_incoming_queries: owning blocks or read-only views of published buffers, unchanged until
// this returns. The sink's on_resolved receives the sender's stream as a QueryWire::WireView.
template <size_t NumModes, class Stream, typename Sink, class Observer = NoRangeObserver>
    requires std::convertible_to<const Stream &, typename QueryWire<NumModes>::WireView>
auto resolve_incoming(const mpi::WindowVec<Stream> &incoming, // serialized, one stream per sender slot
                      MPOperator<NumModes> &op,
                      bool is_leader_pass,
                      MatchedEpochSet &matched,
                      size_t combined_size, // pre-layer op size: bounds the matched set
                      Sink &sink,
                      parallel::Options options = {},
                      const Observer &observer = {}) -> mpi::WindowVec<std::vector<typename Sink::Response>> {
    using Resp = typename Sink::Response;
    const IncomingProbe<NumModes> pr =
        probe_incoming_queries<NumModes>(incoming, op, sink.incoming_form(), options, observer);
    // The pairing is an XOR involution, so the response window is the query window.
    mpi::WindowVec<std::vector<Resp>> responses(pr.window);
    for (const auto wi : pr.window.indices()) {
        responses[wi].assign(pr.goff[wi.value + 1] - pr.goff[wi.value], Sink::init_response());
    }
    if (pr.nq_total == 0) {
        observer.prepare(KernelRange::scatter, 0);
        return responses;
    }

    // Phase 3 (scatter): responses + sink records + matched-follower marks. Freshly inserted partners
    // (ip ≥ combined_size) skip the mark. prepare() sizes every sink container on the caller first.
    sink.prepare(pr, op, responses);
    const size_t n = pr.nq_total;
    const size_t blocks = logical_ranges(n, kResolveScatterBlock);
    const auto scatter = [&](size_t block) {
        observer.visit(KernelRange::scatter, block);
        const size_t hi = std::min(n, (block + 1) * kResolveScatterBlock);
        for (size_t g = block * kResolveScatterBlock; g < hi; ++g) {
            const mpi::WindowIndex s = pr.sender_index(g);
            const size_t q = g - pr.goff[s.value];
            const size_t ip = pr.idx_of[g];
            responses[s][q] = sink.on_resolved(g, s, q, ip, pr, typename QueryWire<NumModes>::WireView(incoming[s]));
            if (is_leader_pass && ip < combined_size) {
                matched.mark(ip);
            }
        }
    };
    observer.prepare(KernelRange::scatter, blocks);
    if (sink_resolves_in_parallel(sink)) {
        // Each query writes its own response and its own sink record; a leader hit marks its own row, distinct
        // across queries (leader_marks_unique), and misses (ip >= base >= combined_size) mark nothing. The
        // marks are distinct uint16 elements, so no synchronization; never paper over a duplicate with atomics.
        assert(!is_leader_pass || leader_marks_unique(pr, combined_size));
        parallel::for_blocks(blocks, options, scatter);
    }
    else {
        for (size_t block = 0; block < blocks; ++block) {
            scatter(block);
        }
    }

    insert_incoming_misses<NumModes>(op, pr, options);
    return responses;
}

// Querier rank (any cross-rank sink): fold each resolver response into a querier-side record. The self/
// local slot was already resolved inline, so it is skipped here. inc_r[k][q] answers query q sent to
// the window's k-th slot. `inc_r` holds owning response blocks or read-only views of blocks the resolver
// published; either way the sink receives each block, its sources and its plain query stream as views, and
// records them in window order, preserving each block's query order.
template <size_t NumModes, typename Sink, class Responses>
    requires std::convertible_to<const Responses &, std::span<const typename Sink::Response>>
auto process_responses(const mpi::WindowVec<Responses> &inc_r,
                       const mpi::WindowVec<std::vector<size_t>> &src_idx,
                       const mpi::WindowVec<VecZ> &queries, // serialized query buffers (for phase recovery)
                       size_t my_rank,
                       Sink &sink) -> void {
    const mpi::SlotWindow w = inc_r.window();
    assert(src_idx.window() == w && queries.window() == w);
    sink.process_reserve(inc_r, my_rank);
    for (const auto wi : w.indices()) {
        const size_t r = w.slot(wi);
        if (r == my_rank) {
            continue;
        }
        sink.on_response_block(r,
                               std::span<const typename Sink::Response>(inc_r[wi]),
                               std::span<const size_t>(src_idx[wi]),
                               typename QueryWire<NumModes>::WireView(queries[wi]));
    }
}

} // namespace monoprop::detail
