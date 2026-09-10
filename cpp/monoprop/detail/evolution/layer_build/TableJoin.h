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

// The per-gate partner join as one batched probe of the operator's persistent term table.
//
// A record names its partner M^G by the partner's 4-byte join key (key(M) ^ key(G) at the sender, folded
// off the decoded positions at a receiver) and its positions. The table (TermTable.h) maps that key to
// the one row holding it, if any, so a gate's join is |Q| probes -- O(records) -- and never touches the
// anticommuting rows no record names. At most one query can match a row: keys are M^G over globally
// distinct sources and ^G is injective, so no two queries of one gate name the same monomial.
//
// A self record's probe may be SKIPPED by the caller's rule (Resolve.h join_self settles a mutual pair
// from the leader's record, leaving the follower's record nothing to find). This class only records the
// outcome, so the resolve can tell a skipped query from a miss.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/TermTable.h"

namespace monoprop::detail {

template <size_t NumModes>
class TableJoin {
public:
    using PosT = typename OperatorIndex<NumModes>::PosT;

    //! "No row matched this query". Valid rows are below the TermIndex ceiling, so the sentinel is free.
    static constexpr size_t kMissing = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    static constexpr TermIndex kMissingRow = std::numeric_limits<TermIndex>::max();
    /*! @brief A query whose probe the caller skipped (pair-once). One below the missing sentinel: only
     *  rows below the gate's insert base are ever stored here, so the engine enables the skip only while
     *  that base is below it.
     */
    static constexpr TermIndex kSkippedRow = kMissingRow - 1;

    /*! @brief Sizes the hit slots, the join's only output, for a gate of `n_queries` records.
     *
     *  The order is the resolve's: the self-staged records first, then the delivered ones. A gate with a
     *  very large record count must not pin its footprint, hence the 4x release rule.
     */
    auto begin_queries(size_t n_queries) -> void {
        if (hit_.capacity() > 4 * std::max<size_t>(n_queries, kMinSlots)) {
            hit_ = std::vector<TermIndex>{};
        }
        hit_.assign(n_queries, kMissingRow);
        hits_ = 0;
        skipped_ = 0;
    }

    /*! @brief Probes every query against @a table, filling hit().
     *
     *  `key_of(q)` / `pos_of(q)` are query q's join key and ascending positions; `skip(q)` and
     *  `on_hit(q, row)` are TermTable::find_batch's (the pair-once rule and the probe-phase mark).
     *  @pre table.rows() == store.size(): the table covers every row a record could name.
     */
    template <typename KeyOf, typename PosOf, typename Skip, typename OnHit>
    auto run(const TermTable &table,
             const OperatorIndex<NumModes> &store,
             KeyOf &&key_of,
             PosOf &&pos_of,
             Skip &&skip,
             OnHit &&on_hit) -> void {
        assert(table.rows() == store.size() && "the term table must cover the whole store before a gate probes it");
        const auto stats = table.find_batch(store,
                                            hit_.size(),
                                            std::forward<KeyOf>(key_of),
                                            std::forward<PosOf>(pos_of),
                                            std::forward<Skip>(skip),
                                            std::forward<OnHit>(on_hit),
                                            std::span<TermIndex>(hit_),
                                            kMissingRow,
                                            kSkippedRow);
        // A skipped query is a settled one: its partner is tracked, so it can never mint. Counting it as
        // a hit keeps queries() - hits() the exact mint bound the resolve sizes by.
        hits_ = stats.hits + stats.skipped;
        skipped_ = stats.skipped;
    }

    //! The row query `q` matched, or kMissing: the partner is absent from this slot. @pre !skipped(q).
    [[nodiscard]] auto hit(size_t q) const -> size_t {
        const TermIndex row = hit_[q];
        assert(row != kSkippedRow && "a skipped query has no row; test skipped() first");
        return (row == kMissingRow) ? kMissing : static_cast<size_t>(row);
    }
    //! Whether query `q`'s probe was skipped under the caller's rule.
    [[nodiscard]] auto skipped(size_t q) const -> bool { return hit_[q] == kSkippedRow; }

    [[nodiscard]] auto queries() const -> size_t { return hit_.size(); }
    //! Queries settled without a mint -- confirmed or skipped -- so queries() - hits() is the mint count.
    [[nodiscard]] auto hits() const -> size_t { return hits_; }
    //! Of hits(), the probes the pair-once rule skipped.
    [[nodiscard]] auto skipped_count() const -> size_t { return skipped_; }

    [[nodiscard]] auto memory_bytes() const -> size_t { return hit_.capacity() * sizeof(TermIndex); }

private:
    static constexpr size_t kMinSlots = 16;

    std::vector<TermIndex> hit_; // query -> matching row, kMissingRow for a miss, kSkippedRow for a skip
    size_t hits_ = 0;            // confirmed + skipped this gate: what the resolve phase sizes by
    size_t skipped_ = 0;
};

} // namespace monoprop::detail
