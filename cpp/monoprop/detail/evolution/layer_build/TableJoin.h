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
    }

    /*! @brief Probes every query against @a table, filling hit().
     *
     *  `key_of(q)` / `pos_of(q)` are query q's join key and ascending positions; `on_hit(q, row)` is
     *  TermTable::find_batch's, run at every confirm in query order within a group (the probe-phase mark).
     *  @pre table.rows() == store.size(): the table covers every row a record could name.
     */
    template <typename KeyOf, typename PosOf, typename OnHit>
    auto run(const TermTable &table,
             const OperatorIndex<NumModes> &store,
             KeyOf &&key_of,
             PosOf &&pos_of,
             OnHit &&on_hit) -> void {
        assert(table.rows() == store.size() && "the term table must cover the whole store before a gate probes it");
        hits_ = table.find_batch(store,
                                 hit_.size(),
                                 std::forward<KeyOf>(key_of),
                                 std::forward<PosOf>(pos_of),
                                 std::forward<OnHit>(on_hit),
                                 std::span<TermIndex>(hit_),
                                 kMissingRow);
    }

    //! The row query `q` matched, or kMissing: the partner is absent from this slot.
    [[nodiscard]] auto hit(size_t q) const -> size_t {
        const TermIndex row = hit_[q];
        return (row == kMissingRow) ? kMissing : static_cast<size_t>(row);
    }

    [[nodiscard]] auto queries() const -> size_t { return hit_.size(); }
    //! Queries that confirmed a row, so queries() - hits() is the gate's exact mint count.
    [[nodiscard]] auto hits() const -> size_t { return hits_; }

    [[nodiscard]] auto memory_bytes() const -> size_t { return hit_.capacity() * sizeof(TermIndex); }

private:
    static constexpr size_t kMinSlots = 16;

    std::vector<TermIndex> hit_; // query -> matching row, kMissingRow for a miss
    size_t hits_ = 0;            // rows confirmed this gate: what the resolve phase sizes by
};

} // namespace monoprop::detail
