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

// M⊕G as ascending positions: a slot in both M and G cancels, so the partner is the symmetric
// difference of two ascending position lists, and one merge yields the positions and overlap together.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

// Writes the symmetric difference of a and b to out and returns its length; overlap_out gets the
// shared-position count. a and b must be ascending, or the result is silently wrong.
template <typename PosT, typename GenT>
[[gnu::always_inline]] inline auto merge_partner_positions(const PosT *a,
                                                           size_t ka,
                                                           const GenT *b,
                                                           size_t kb,
                                                           PosT *out,
                                                           size_t &overlap_out) noexcept -> size_t {
    size_t i = 0;
    size_t j = 0;
    size_t n = 0;
    size_t overlap = 0;
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
        out[n++] = static_cast<PosT>(p);
    }
    for (; i < ka; ++i) {
        out[n++] = static_cast<PosT>(a[i]);
    }
    for (; j < kb; ++j) {
        out[n++] = static_cast<PosT>(b[j]);
    }
    overlap_out = overlap;
    return n;
}

// Stages self-owned query positions for direct use by OperatorIndex's find_batch_positions and
// set_positions, with no encoding step.
template <size_t NumModes>
struct SelfQueryStage {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    // Sized to capacity, not filled: the logical length is size()/positions(), not the vectors' own size().
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

    // Appends one query's positions and its (offset, k, phase) record; grows only when capacity runs out.
    auto push(const PosT *pos, size_t k, int phase) -> void {
        assert(phase >= -1 && phase <= 1 && "emit_phase is ternary: rotation_sign, or REAL_PARTS entry");
        const size_t n = n_;
        const size_t at = pos_n_;
        if (n == pos_off.size() || at + k > pos_flat.size()) {
            grow_(k);
        }
        pos_off[n] = at;
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

    // Doubles capacity, but grows pos_flat by at least what this push needs, so a wide term can't leave it short.
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
