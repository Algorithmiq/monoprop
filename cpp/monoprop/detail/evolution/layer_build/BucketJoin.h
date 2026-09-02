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

// The per-gate partner join as a bucketed (radix) join of two streams.
//
// The algebra is unchanged: if M anticommutes with G then so does M ^ G (Majorana: |M'||G| - |M' ∩ G| ≡
// |M||G| - |M ∩ G| mod 2 using |G|² ≡ |G|; Pauli: ω(M ^ G, G) = ω(M, G)), so a tracked partner is one of
// the rows the fold already returned and the operator needs no persistent hash table. What changed is
// that NEITHER side of the join is the random-access side.
//
// Under the exact one-round protocol a term below `lower_atol` whose partner passes the structural cutoff
// still sends a rot=0 value record, so the records a slot must answer number |Q| ≈ 0.85 |Anti(G)| in that
// regime -- a table over the records is as large as a table over Anti(G), and either way one side pays a
// random insert or probe into a structure far past L2. So both sides are written sequentially into
// buckets of the mixed key's top bits (~4 K entries each) and each bucket is joined inside L1, with a
// tiny open-addressing table over the smaller of the two.
//
// Two things stay load-bearing. The bucket and the compare tag come from the MIXED fingerprint (the
// fingerprint is GF(2)-linear, so keys differing by the per-gate constant fp(G) would otherwise share a
// bucket wholesale). And every tag match is confirmed against the query's positions -- the fingerprint
// maps 2*NumModes bits onto 64, so a false match would silently merge two distinct terms.

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

template <size_t NumModes>
class BucketJoin {
public:
    using PosT = typename OperatorIndex<NumModes>::PosT;

    //! "No row matched this query". Valid rows are below the TermIndex ceiling, so the sentinel is free.
    static constexpr size_t kMissing = static_cast<size_t>(std::numeric_limits<TermIndex>::max());

    // The anticommuting side, staged by the scan while it has each row in hand (Scan.h): the key is
    // folded once there, off positions the merge has just read, rather than in a second pass over the
    // rows. `n_anti` is pass 1's tally, so this is where the 4× release rule applies -- one gate with a
    // very large anticommuting set (a single-qubit Pauli generator) must not pin its footprint.
    auto begin_rows(size_t n_anti) -> void {
        if (rows_.capacity() > 4 * std::max<size_t>(n_anti, kMinSlots)) {
            rows_ = std::vector<Entry>{};
            row_buckets_ = DefaultInitVector<Entry>{};
        }
        rows_.clear();
        rows_.reserve(n_anti);
    }

    // Resets the row side without releasing anything: the paths that stage no rows at all (an identity
    // gate, or a fold whose anticommuting set is empty) are the majority by count, and begin_rows(0) would
    // hand their buffers back on every one of them.
    auto clear_rows() -> void { rows_.clear(); }

    //! One anticommuting row under its own fingerprint. Order is irrelevant: the buckets reorder anyway.
    auto add_row(uint64_t fp, size_t row) -> void { rows_.push_back(entry_of(fp, row)); }

    // The same, from the key the store already holds for that row (OperatorIndex::key): the scan stages
    // its whole anticommuting set this way, so pass 1 never touches a row.
    auto add_row_key(uint32_t key, size_t row) -> void {
        rows_.push_back(Entry{.tag = key, .v = static_cast<uint32_t>(row)});
    }

    // The record side, in Q order (self-staged then incoming, as the join applies them). Sizes the hit
    // slots, which are the join's only output.
    auto begin_queries(size_t n_queries) -> void {
        if (queries_.capacity() > 4 * std::max<size_t>(n_queries, kMinSlots)) {
            queries_ = std::vector<Entry>{};
            query_buckets_ = DefaultInitVector<Entry>{};
            hit_ = std::vector<TermIndex>{};
        }
        queries_.clear();
        queries_.reserve(n_queries);
        hit_.assign(n_queries, kMissingRow);
    }

    // Enters query `q` under the fingerprint of the monomial it is looking for (fp(M) ^ fp(G) at the
    // sender, recomputed from the decoded positions at a receiver -- one function, so the two agree).
    auto add_query(size_t q, uint64_t fp) -> void {
        assert(q < hit_.size() && "query index outside the count begin_queries() was sized for");
        queries_.push_back(entry_of(fp, q));
    }

    // Buckets both sides and joins them bucket by bucket, filling hit(). `pos_of(q)` must return query
    // q's ascending positions.
    //
    // At most one query can match a row: keys are M⊕G over globally distinct sources and ⊕G is
    // injective, so no two queries of one gate name the same monomial.
    template <typename PosOf>
    auto run(const OperatorIndex<NumModes> &store, PosOf &&pos_of) -> void {
        if (rows_.empty() || queries_.empty()) {
            return; // nothing to answer, or nothing to answer with
        }
        const size_t big = std::max(rows_.size(), queries_.size());
        // ~4 K entries per bucket: the two sides of one bucket plus its table stay inside the core's own
        // caches. Signed arithmetic because bit_width < 12 on the small gates that dominate by count.
        const auto want = static_cast<int>(std::bit_width(big)) - 12;
        const size_t bits =
            static_cast<size_t>(std::clamp(want, static_cast<int>(kMinBucketBits), static_cast<int>(kMaxBucketBits)));
        const size_t buckets = size_t{1} << bits;
        const size_t shift = 32 - bits;
        bucketize_(rows_, row_buckets_, row_offsets_, buckets, shift);
        bucketize_(queries_, query_buckets_, query_offsets_, buckets, shift);
        for (size_t b = 0; b < buckets; ++b) {
            const std::span<const Entry> rows(row_buckets_.data() + row_offsets_[b],
                                              row_offsets_[b + 1] - row_offsets_[b]);
            const std::span<const Entry> queries(query_buckets_.data() + query_offsets_[b],
                                                 query_offsets_[b + 1] - query_offsets_[b]);
            if (rows.empty() || queries.empty()) {
                continue;
            }
            join_bucket_(store, rows, queries, pos_of);
        }
    }

    //! The row query `q` matched, or kMissing: the partner is absent from this slot.
    [[nodiscard]] auto hit(size_t q) const -> size_t {
        const TermIndex row = hit_[q];
        return (row == kMissingRow) ? kMissing : static_cast<size_t>(row);
    }

    [[nodiscard]] auto rows() const -> size_t { return rows_.size(); }
    [[nodiscard]] auto queries() const -> size_t { return queries_.size(); }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return ((rows_.capacity() + row_buckets_.capacity() + queries_.capacity() + query_buckets_.capacity()
                 + table_.capacity())
                * sizeof(Entry))
               + (hit_.capacity() * sizeof(TermIndex))
               + ((row_offsets_.capacity() + query_offsets_.capacity() + fill_.capacity()) * sizeof(uint32_t));
    }

private:
    // 8 bytes, so a bucket of a few thousand entries and its table stay L1-resident. `v` is a row on the
    // anticommuting side and a query index on the record side; kNoValue marks an empty table slot.
    struct Entry {
        uint32_t tag; // the mixed key's high 32 bits: top `bits` choose the bucket, all 32 are compared
        uint32_t v;
    };

    static constexpr uint32_t kNoValue = std::numeric_limits<uint32_t>::max();
    static constexpr TermIndex kMissingRow = std::numeric_limits<TermIndex>::max();
    static constexpr size_t kMinSlots = 16;
    static constexpr size_t kMinBucketBits = 4;
    static constexpr size_t kMaxBucketBits = 10;

    // One definition of the tag, in the store (OperatorIndex::join_tag), so a row's stored key and a
    // query's tag cannot drift apart.
    static auto entry_of(uint64_t fp, size_t v) -> Entry {
        return Entry{.tag = OperatorIndex<NumModes>::join_tag(fp), .v = static_cast<uint32_t>(v)};
    }

    // Counting sort of `in` into `out` by the tag's top bits, with `offsets` left as the bucket bounds
    // (buckets+1 entries). One sequential read and one scattered-but-few-streams write per side.
    auto bucketize_(const std::vector<Entry> &in,
                    DefaultInitVector<Entry> &out,
                    std::vector<uint32_t> &offsets,
                    size_t buckets,
                    size_t shift) -> void {
        offsets.assign(buckets + 1, 0);
        for (const Entry e : in) {
            ++offsets[(e.tag >> shift) + 1];
        }
        for (size_t b = 0; b < buckets; ++b) {
            offsets[b + 1] += offsets[b];
        }
        fill_.assign(offsets.begin(), offsets.end() - 1);
        out.resize(in.size());
        for (const Entry e : in) {
            out[fill_[e.tag >> shift]++] = e;
        }
    }

    // One bucket: the smaller side goes into a linear-probe table, the larger streams against it. The
    // home slot takes the tag's LOW bits, which the bucket split has not spent.
    template <typename PosOf>
    auto join_bucket_(const OperatorIndex<NumModes> &store,
                      std::span<const Entry> rows,
                      std::span<const Entry> queries,
                      PosOf &pos_of) -> void {
        const bool table_holds_queries = queries.size() <= rows.size();
        const std::span<const Entry> tabled = table_holds_queries ? queries : rows;
        const std::span<const Entry> streamed = table_holds_queries ? rows : queries;
        const size_t slots = std::bit_ceil(std::max<size_t>(kMinSlots, ((tabled.size() * 10) / 7) + 1));
        const size_t mask = slots - 1;
        if (table_.size() < slots) {
            table_.resize(slots);
        }
        std::fill_n(table_.begin(), slots, Entry{.tag = 0, .v = kNoValue});
        for (const Entry e : tabled) {
            size_t s = e.tag & mask;
            while (table_[s].v != kNoValue) {
                s = (s + 1) & mask;
            }
            table_[s] = e;
        }
        for (const Entry e : streamed) {
            for (size_t s = e.tag & mask;; s = (s + 1) & mask) {
                const Entry t = table_[s];
                if (t.v == kNoValue) {
                    break;
                }
                if (t.tag != e.tag) {
                    continue;
                }
                const size_t row = table_holds_queries ? e.v : t.v;
                const size_t q = table_holds_queries ? t.v : e.v;
                if (store.row_eq_positions(row, pos_of(q))) {
                    assert(hit_[q] == kMissingRow && "two rows confirmed the same query key");
                    hit_[q] = static_cast<TermIndex>(row);
                    break;
                }
            }
        }
    }

    std::vector<Entry> rows_;              // staged anticommuting rows, scan order
    std::vector<Entry> queries_;           // staged records, Q order
    DefaultInitVector<Entry> row_buckets_; // the same, bucketed
    DefaultInitVector<Entry> query_buckets_;
    DefaultInitVector<Entry> table_; // one bucket's probe table, reused across buckets
    std::vector<TermIndex> hit_;     // query -> matching row, kMissingRow until a row confirms it
    std::vector<uint32_t> row_offsets_;
    std::vector<uint32_t> query_offsets_;
    std::vector<uint32_t> fill_; // bucketize_'s write cursors
};

} // namespace monoprop::detail
