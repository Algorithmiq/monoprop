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
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

/*! @brief The symmetric difference's length, the count of cancelled positions, and its paired modes. */
struct MergedPartner {
    size_t count;   //!< positions written to out
    size_t overlap; //!< positions present in both inputs, which therefore cancelled
    size_t paired;  //!< modes carrying BOTH of their positions in the output: the `d` of the (k, d) digest
};

/*! @brief Writes the symmetric difference of `a` and `b` to `out`.
 *
 *  Both inputs must be strictly ascending, or the result is silently wrong; `out` needs room for
 *  `a.size() + b.size()`. One pass yields the positions, the overlap and the paired-mode count together,
 *  so the structural cutoff (length_keeps / support_keeps) never reads a bitset. Mode m owns positions
 *  (2m, 2m+1), so in ascending output a paired mode is an even position immediately followed by its
 *  successor. Written out three times rather than through a lambda: on an earlier port GCC declined to
 *  inline a five-capture closure into three call sites and the calls alone cost a third of the kernel.
 */
template <std::ranges::contiguous_range Row, std::ranges::contiguous_range Gen, std::ranges::contiguous_range Out>
[[gnu::always_inline]] inline auto merge_partner_positions(const Row &a, const Gen &b, Out &&out) noexcept
    -> MergedPartner {
    using PosT = std::ranges::range_value_t<Out>;
    const size_t ka = std::ranges::size(a);
    const size_t kb = std::ranges::size(b);
    size_t i = 0;
    size_t j = 0;
    size_t n = 0;
    size_t overlap = 0;
    size_t paired = 0;
    // Seeded ODD so the (prev even && p == prev + 1) test cannot fire on the first emit: 1 is not a
    // reachable prev + 1 either, since prev would have to be 0 and even.
    size_t prev = 1;
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
        paired += static_cast<size_t>((prev % 2 == 0) && p == prev + 1);
        out[n++] = static_cast<PosT>(p);
        prev = p;
    }
    for (; i < ka; ++i) {
        const size_t p = static_cast<size_t>(a[i]);
        paired += static_cast<size_t>((prev % 2 == 0) && p == prev + 1);
        out[n++] = static_cast<PosT>(p);
        prev = p;
    }
    for (; j < kb; ++j) {
        const size_t p = static_cast<size_t>(b[j]);
        paired += static_cast<size_t>((prev % 2 == 0) && p == prev + 1);
        out[n++] = static_cast<PosT>(p);
        prev = p;
    }
    return {n, overlap, paired};
}

/*! @brief Stages the records addressed to this slot itself as positions, for direct use by the
 *  AntiTable probe and OperatorIndex::set_positions, with no encoding step.
 *
 *  Carries exactly the fields a wire record does (QueryWire.h): key positions, phase, rot and, for
 *  the fused form, the sender's value.
 */
template <size_t NumModes>
struct SelfQueryStage {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    //! Sized to capacity, not filled: the logical length is size()/positions(), not the vectors' own size().
    DefaultInitVector<PosT> pos_flat;   //!< ascending positions, concatenated in push order
    DefaultInitVector<size_t> pos_off;  //!< query -> absolute offset into pos_flat
    DefaultInitVector<uint32_t> k_of;   //!< positions per query
    DefaultInitVector<int8_t> phase_of; //!< emit_phase is ternary, so a byte is the whole range
    DefaultInitVector<uint8_t> rot_of;  //!< the sender's rotation bit
    DefaultInitVector<double> val_of;   //!< the sender's pre-cos coefficient; written only when values are captured
    DefaultInitVector<uint64_t> fp_of;  //!< routing fingerprint of the query (fp(M) ^ fp(G)): the AntiTable key

    [[nodiscard]] auto size() const -> size_t { return n_; }
    [[nodiscard]] auto positions() const -> size_t { return pos_n_; }
    [[nodiscard]] auto positions_at(size_t q) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[q], k_of[q]);
    }

    auto clear() -> void {
        n_ = 0;
        pos_n_ = 0;
    }

    auto reserve(size_t n_queries, size_t positions_per_query) -> void {
        if (pos_off.size() < n_queries) {
            pos_off.resize(n_queries);
            k_of.resize(n_queries);
            phase_of.resize(n_queries);
            rot_of.resize(n_queries);
            val_of.resize(n_queries);
            fp_of.resize(n_queries);
        }
        if (pos_flat.size() < n_queries * positions_per_query) {
            pos_flat.resize(n_queries * positions_per_query);
        }
    }

    //! Appends one record; grows only when capacity runs out.
    template <std::ranges::contiguous_range Pos>
    auto push(const Pos &pos, int phase, uint64_t fp, bool rot = false, double val = 0.0) -> void {
        assert(phase >= -1 && phase <= 1 && "emit_phase is ternary: rotation_sign, or REAL_PARTS entry");
        const size_t k = std::ranges::size(pos);
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
        rot_of[n] = static_cast<uint8_t>(rot);
        val_of[n] = val;
        fp_of[n] = fp;
        n_ = n + 1;
        pos_n_ = at + k;
    }

private:
    size_t n_ = 0;     //!< queries pushed
    size_t pos_n_ = 0; //!< positions pushed

    //! Doubles capacity, but grows pos_flat by at least what this push needs, so a wide term can't leave it short.
    [[gnu::noinline]] auto grow_(size_t k) -> void {
        if (n_ == pos_off.size()) {
            const size_t want = (pos_off.size() * 2) + 64;
            pos_off.resize(want);
            k_of.resize(want);
            phase_of.resize(want);
            rot_of.resize(want);
            val_of.resize(want);
            fp_of.resize(want);
        }
        if (pos_n_ + k > pos_flat.size()) {
            pos_flat.resize(std::max((pos_flat.size() * 2) + 256, pos_n_ + k));
        }
    }
};

} // namespace monoprop::detail
