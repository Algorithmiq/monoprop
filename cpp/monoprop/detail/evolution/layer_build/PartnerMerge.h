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

// M⊕G as ascending positions. A slot in both M and G cancels (m_p m_p = 1), so the partner is the
// symmetric difference of two ascending position lists, and one merge yields its positions, `overlap`
// and `d` (modes carrying BOTH Majoranas) together -- the (k, d) digest the structural cutoff wants,
// with no second sweep over the dense form and no walk back out of it.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

// Both inputs must be strictly ascending. The output is their symmetric difference, so it is bounded
// by the universe the positions are drawn from -- 2*NumModes here -- and ka + kb is only the bound that
// ignores cancellation. Returns the merged count. GenT is separate from PosT because the generator's
// positions are the wire's width, not the store's.
template <typename PosT, typename GenT>
[[gnu::always_inline]] inline auto merge_partner_positions(const PosT *a,
                                                           size_t ka,
                                                           const GenT *b,
                                                           size_t kb,
                                                           PosT *out,
                                                           size_t &overlap_out,
                                                           size_t &d_out) noexcept -> size_t {
    size_t i = 0;
    size_t j = 0;
    size_t n = 0;
    size_t overlap = 0;
    size_t d = 0;
    // Seeded ODD, so the (prev % 2 == 0) test cannot fire on the first emit and the loops need no
    // n != 0 guard; 1 is not a reachable `prev + 1` either, since prev would have to be 0 and even.
    size_t prev = 1;
    // Ascending output, so a doubly-occupied mode is an even position immediately followed by its
    // successor -- the same count paired_mode_count folds out of the bitset. Written out three times
    // rather than through a lambda: callgrind measured 15,279,191 CALLS to that lambda at 18
    // instructions each (275.1M, a third of this port's whole delta), because GCC declined to inline a
    // closure capturing five locals by reference into three call sites.
    while (i < ka && j < kb) {
        const size_t pa = static_cast<size_t>(a[i]);
        const size_t pb = static_cast<size_t>(b[j]);
        if (pa == pb) {
            ++overlap;
            ++i;
            ++j;
            continue;
        }
        const size_t p = pa < pb ? pa : pb;
        i += static_cast<size_t>(pa < pb);
        j += static_cast<size_t>(pb < pa);
        d += static_cast<size_t>((prev % 2 == 0) && p == prev + 1);
        out[n++] = static_cast<PosT>(p);
        prev = p;
    }
    for (; i < ka; ++i) {
        const size_t p = static_cast<size_t>(a[i]);
        d += static_cast<size_t>((prev % 2 == 0) && p == prev + 1);
        out[n++] = static_cast<PosT>(p);
        prev = p;
    }
    for (; j < kb; ++j) {
        const size_t p = static_cast<size_t>(b[j]);
        d += static_cast<size_t>((prev % 2 == 0) && p == prev + 1);
        out[n++] = static_cast<PosT>(p);
        prev = p;
    }
    overlap_out = overlap;
    d_out = d;
    return n;
}

// Self-owned queries never reach a wire, so they are staged as positions rather than encoded records:
// OperatorIndex's find_batch_positions and set_positions both take exactly this shape, so the resolve
// path consumes the stage with no transformation and the codec is not on the self leg at all.
template <size_t NumModes>
struct SelfQueryStage {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    // SIZED, not filled: the vectors carry the capacity and n_/pos_n_ carry the logical length, so a
    // push writes rather than appends. Read them through size() and the data pointers only.
    //
    // DefaultInitVector, the allocator Resolve.h already uses for exactly this: a plain vector's resize
    // VALUE-initialises, so sizing pos_flat ahead would memset every byte a push is about to overwrite
    // -- trading the append cost for a per-gate zero-fill instead of removing it.
    DefaultInitVector<PosT> pos_flat;  // ascending positions, concatenated in push order
    DefaultInitVector<size_t> pos_off; // query -> absolute offset into pos_flat
    DefaultInitVector<uint32_t> k_of;
    DefaultInitVector<int8_t> phase_of; // emit_phase is ternary, so a byte is the whole range

    [[nodiscard]] auto size() const -> size_t { return n_; }
    [[nodiscard]] auto positions() const -> size_t { return pos_n_; }

    auto clear() -> void {
        n_ = 0;
        pos_n_ = 0;
    }

    auto reserve(size_t n_queries, size_t positions_per_query) -> void {
        if (pos_off.size() < n_queries) {
            pos_off.resize(n_queries);
            k_of.resize(n_queries);
            phase_of.resize(n_queries);
        }
        if (pos_flat.size() < n_queries * positions_per_query) {
            pos_flat.resize(n_queries * positions_per_query);
        }
    }

    // Four preallocated writes, not four container appends. Callgrind on the pauli cell put
    // vector<uint8_t>::_M_range_insert at 110.6M instructions and vector<uint32_t>::emplace_back at
    // 52.2M -- 18% of this port's whole instruction delta -- for a push whose capacity is already
    // reserved. `insert` cannot know that, so it re-derives the grow path per query; grow_() is the
    // one place that checks, and it runs once per capacity doubling instead of once per push.
    auto push(const PosT *pos, size_t k, int phase) -> void {
        assert(phase >= -1 && phase <= 1 && "emit_phase is ternary: rotation_sign, or REAL_PARTS entry");
        const size_t n = n_;
        const size_t at = pos_n_;
        if (n == pos_off.size() || at + k > pos_flat.size()) {
            grow_(k);
        }
        pos_off[n] = at;
        // An explicit loop, not std::copy_n: k averages ~5 bytes here and copy_n compiles to a memcpy
        // CALL, which callgrind counted 2.54M extra times for a copy smaller than its own prologue.
        PosT *dst = pos_flat.data() + at;
        for (size_t j = 0; j < k; ++j) {
            dst[j] = pos[j];
        }
        k_of[n] = static_cast<uint32_t>(k);
        phase_of[n] = static_cast<int8_t>(phase);
        n_ = n + 1;
        pos_n_ = at + k;
    }

private:
    size_t n_ = 0;     // queries pushed
    size_t pos_n_ = 0; // positions pushed

    // Amortised doubling, and pos_flat grows by the larger of a double and what this push needs, so a
    // single wide term cannot leave it short.
    [[gnu::noinline]] auto grow_(size_t k) -> void {
        if (n_ == pos_off.size()) {
            const size_t want = (pos_off.size() * 2) + 64;
            pos_off.resize(want);
            k_of.resize(want);
            phase_of.resize(want);
        }
        if (pos_n_ + k > pos_flat.size()) {
            pos_flat.resize(std::max((pos_flat.size() * 2) + 256, pos_n_ + k));
        }
    }
};

} // namespace monoprop::detail
