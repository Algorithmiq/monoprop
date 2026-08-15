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

// The sparse form of a monomial -- an ascending list of set-bit positions, which is what
// OperatorIndex already stores -- and the one product kernel the layer build needs over it.
//
// Why this exists: the emit path held a position list, expanded it into a dense Bitset, ran four
// dense-width operations on it, and threw it away. xor_gen does the whole thing in one merge over
// max(k, g) elements and returns every scalar the caller needed from those four operations.
//
// Bit conventions (see AlgebraCommon.h:49 and PauliAlgebra.h:17-18): positions are PHYSICAL, LSb0,
// ascending. Mode m owns the adjacent pair (2m, 2m+1), so a pair never straddles a word and "both
// bits of a mode are set" is "an even position immediately followed by its successor".

#include <cstddef>
#include <cstdint>

namespace monoprop {

// The scalars the emit path reads off one product. `k`, `d` describe M^G; `overlap` and `sign`
// describe the pair (M, G) and are what the phase needs.
struct XorDigest {
    size_t k = 0;       // popcount(M ^ G) -- equals the number of positions written to `out`
    size_t d = 0;       // modes with BOTH bits set in M ^ G; xor_sum = k - 2d, or_sum = k - d
    size_t overlap = 0; // |M & G|
    int sign = 1;       // Majorana interleave sign (-1)^{sum_g rank_M(g)}; see MajoranaAlgebra.h:72
};

/*!
 * \brief Symmetric difference of an ascending position list with a sparse generator, plus the
 *        popcount, pair count, overlap and interleave sign, in one pass.
 *
 * `out` must have room for `k + g` positions. Both inputs must be strictly ascending; the output is
 * strictly ascending, so it is directly comparable and hashable as a canonical key.
 *
 * The sign is accumulated rather than computed: when the merge consumes generator position g_j it
 * has already consumed exactly rank_M(g_j) of M's positions, which is the quantity
 * interleave_phase() sums. Majorana only -- PauliAlgebra::rotation_sign needs both operands
 * (PauliAlgebra.h:122-140) and is scored by the caller.
 */
template <typename PosT>
[[gnu::always_inline]] inline auto xor_gen(const PosT *mp, size_t k, const PosT *gp, size_t g, PosT *out) noexcept
    -> XorDigest {
    XorDigest r;
    size_t i = 0;
    size_t j = 0;
    size_t n = 0;
    size_t rank_sum = 0;
    // Sentinel below every valid position, so the first emitted entry never reads uninitialized
    // state and can never be mistaken for the upper half of a pair.
    int32_t prev = -2;

    const auto emit = [&](PosT p) {
        // A mode contributes 2 positions iff its even bit is immediately followed by its odd one.
        if (static_cast<int32_t>(p) == prev + 1 && (prev & 1) == 0) {
            ++r.d;
        }
        prev = static_cast<int32_t>(p);
        out[n++] = p;
    };

    while (i < k && j < g) {
        if (mp[i] < gp[j]) {
            emit(mp[i]);
            ++i;
        }
        else if (gp[j] < mp[i]) {
            rank_sum += i; // exactly the M positions strictly below gp[j]
            emit(gp[j]);
            ++j;
        }
        else {
            // Shared position: cancels out of the product, and rank_M(gp[j]) is still i.
            rank_sum += i;
            ++r.overlap;
            ++i;
            ++j;
        }
    }
    while (i < k) {
        emit(mp[i]);
        ++i;
    }
    while (j < g) {
        rank_sum += k; // M is exhausted, so every remaining generator bit sits above all of it
        emit(gp[j]);
        ++j;
    }

    r.k = n;
    r.sign = ((rank_sum & 1U) != 0U) ? -1 : 1;
    return r;
}

/*!
 * \brief Pair digest of an ascending position list: popcount and the number of fully occupied modes.
 *
 * The reference form of what xor_gen accumulates, for tests and for cold paths.
 */
template <typename PosT>
[[nodiscard]] inline auto pair_digest(const PosT *p, size_t k) noexcept -> XorDigest {
    XorDigest r;
    r.k = k;
    for (size_t t = 0; t + 1 < k; ++t) {
        if ((p[t] & 1) == 0 && p[t + 1] == p[t] + 1) {
            ++r.d;
        }
    }
    return r;
}

// The structural cutoffs, as integer comparisons. Both keep a fully paired monomial unconditionally
// (AlgebraCommon.h:157-159): those are the only terms contributing to an expectation value against a
// product reference state. xor_sum = k - 2d, popcount_sum = k, or_sum = k - d.
[[nodiscard]] inline constexpr auto is_paired(size_t k, size_t d) noexcept -> bool {
    return k == 2 * d;
}
[[nodiscard]] inline constexpr auto length_keeps(size_t k, size_t d, size_t cutoff) noexcept -> bool {
    return k == 2 * d || k <= cutoff;
}
[[nodiscard]] inline constexpr auto support_keeps(size_t k, size_t d, size_t cutoff) noexcept -> bool {
    return k == 2 * d || k - d <= cutoff;
}

} // namespace monoprop
