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

// The structural algebra on a sparse row's `codes` word, one function per dense counterpart in
// AlgebraCommon.h / PauliAlgebra.h / MajoranaAlgebra.h. Each is exact, not approximate: agreement with
// the dense version over the tests/data fixtures and randomized rows is asserted in
// cpp/tests/codes_algebra_tests.cpp, and that test is the gate on ever making these the default.
//
// Why any of this is possible in one word: a mode's two physical positions 2m, 2m+1 become the 2-bit
// field of slot j, so quantities the dense form derives from a per-word masked shift chain over the
// whole register become popcounts of two masks of a single word, independent of the storage width.
// With n = popcount(row_occupied_bits(codes)) and d = popcount(row_paired_bits(codes)):
//
//   or_sum (support/Pauli weight) = n        popcount_sum (length) = n + d      xor_sum = n - d
//   is_paired  <=>  d == n                   pair_swap = swap the two bits of every field
//   Y letters  =  fields equal to 0b01
//
// Nothing here reads a mode lane except codes_interleave_phase, which is inherently a two-row
// operation, and codes_cutoff_sums when a logical width narrower than the storage width makes some
// modes inactive.
//
// Names are prefixed rather than overloading the dense functions: while both representations are live
// a call site should say which one it means, and overload resolution between `MonomialLike auto` and
// SparseRow would decide that silently.

#include <bit>
#include <cassert>
#include <cstddef>

#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/detail/operator/SparseRowStore.h"

namespace monoprop::detail {

// Set bits of `codes` in slots strictly below `slot`. The dense counterpart is the prefix popcount an
// interleave scan maintains word by word.
[[nodiscard]] inline auto codes_popcount_below(RowCodes codes, size_t slot) noexcept -> size_t {
    // A shift by 2*kRowMaxSlots would be undefined, and "below every slot" is the whole word anyway.
    if (slot >= kRowMaxSlots) {
        return static_cast<size_t>(std::popcount(codes));
    }
    return static_cast<size_t>(std::popcount(codes & ((RowCodes{1} << (2 * slot)) - 1)));
}

// or_sum / popcount_sum / xor_sum for a row whose every slot is inside the active window. Two popcounts
// and no reference to the storage width, where the dense cutoff_sums runs a masked shift chain per word
// and needs CutoffMasks to avoid rederiving the masks per term.
[[nodiscard]] inline auto codes_cutoff_sums(RowCodes codes) noexcept -> CutoffSums {
    const auto n = static_cast<size_t>(std::popcount(row_occupied_bits(codes)));
    const auto d = static_cast<size_t>(std::popcount(row_paired_bits(codes)));
    return {n - d, n + d, n};
}

// The same, restricted to the active window. inactive_mode_prefix is the count of leading *physical*
// modes the logical width excludes -- storage_num_modes - logical_num_modes, half of
// CutoffMasks::active_bit_offset -- and the dense form applies it as `mono >> active_bit_offset`.
//
// The inactive modes are exactly the low ones, so they are a prefix of the ascending slots and drop out
// with one shift. A propagator's rows never carry them (a term is built from logical indices, which map
// into the window), so the common case is the zero-prefix early exit; the general path exists because
// the dense function it must agree with accepts such a monomial.
[[nodiscard]] inline auto codes_cutoff_sums(const SparseRow &row, size_t inactive_mode_prefix) noexcept -> CutoffSums {
    if (inactive_mode_prefix == 0) {
        return codes_cutoff_sums(row.codes);
    }
    const size_t n = row.num_slots();
    size_t inactive_slots = 0;
    while (inactive_slots < n && row.mode(inactive_slots) < inactive_mode_prefix) {
        ++inactive_slots;
    }
    if (inactive_slots >= kRowMaxSlots) {
        return {0, 0, 0};
    }
    return codes_cutoff_sums(row.codes >> (2 * inactive_slots));
}

// Both cutoffs keep a fully paired row unconditionally, exactly as the dense ones do: those are the
// only terms contributing to an expectation value against a product reference state.
[[nodiscard]] inline auto codes_length_cutoff(const SparseRow &row,
                                              unsigned int cutoff,
                                              size_t inactive_mode_prefix) noexcept -> bool {
    const auto sums = codes_cutoff_sums(row, inactive_mode_prefix);
    return sums.xor_sum == 0 || sums.popcount_sum <= cutoff;
}

[[nodiscard]] inline auto codes_support_cutoff(const SparseRow &row,
                                               unsigned int cutoff,
                                               size_t inactive_mode_prefix) noexcept -> bool {
    const auto sums = codes_cutoff_sums(row, inactive_mode_prefix);
    return sums.xor_sum == 0 || sums.or_sum <= cutoff;
}

// The counterparts of CutoffEvaluator::passes_with_popcount, one per concrete cutoff functor. Same
// shortcut and same reasoning: the predicate is `xor_sum == 0 || measure <= cutoff`, so a popcount
// already at or below the bound proves keep without reading the row at all (or_sum <= popcount_sum makes
// that sound for the support cutoff too). `popcount_sum` is the whole-register count, which can only
// exceed the active-window one, so the shortcut stays conservative when a logical width is narrower than
// the storage width.
//
// There is no evaluator argument: which cutoff a propagator has is fixed for its lifetime, so the caller
// resolves it once per gate rather than re-branching per term.
[[nodiscard]] inline auto codes_length_passes_with_popcount(const SparseRow &row,
                                                            unsigned int cutoff,
                                                            size_t popcount_sum,
                                                            size_t inactive_mode_prefix) noexcept -> bool {
    return popcount_sum <= cutoff || codes_length_cutoff(row, cutoff, inactive_mode_prefix);
}

[[nodiscard]] inline auto codes_support_passes_with_popcount(const SparseRow &row,
                                                             unsigned int cutoff,
                                                             size_t popcount_sum,
                                                             size_t inactive_mode_prefix) noexcept -> bool {
    return popcount_sum <= cutoff || codes_support_cutoff(row, cutoff, inactive_mode_prefix);
}

// Every occupied mode holds both of its positions, i.e. every field is 0b11. Unoccupied modes are not
// slots at all and are trivially paired, which is why this needs no window argument -- and matches the
// dense is_paired, which likewise checks the whole register.
[[nodiscard]] inline auto codes_is_paired(RowCodes codes) noexcept -> bool {
    return row_paired_bits(codes) == row_occupied_bits(codes);
}

// The pair-swap involution J: swap the two physical bits of every mode (u <-> v). Occupancy is
// preserved -- 0b01 <-> 0b10 and 0b11 is fixed -- so the mode lanes are untouched and the row's whole
// transform is this one word operation, against a per-word masked shift-and-or on the dense side.
[[nodiscard]] constexpr auto codes_pair_swap(RowCodes codes) noexcept -> RowCodes {
    return ((codes & kRowLoBits) << 1) | ((codes >> 1) & kRowLoBits);
}

// Y letters: v=1, u=0 under the JW image, so the field is exactly 0b01.
[[nodiscard]] inline auto codes_pauli_y_count(RowCodes codes) noexcept -> size_t {
    const RowCodes v = codes & kRowLoBits;
    const RowCodes u = (codes >> 1) & kRowLoBits;
    return static_cast<size_t>(std::popcount(v & ~u));
}

// Whether two Pauli strings anticommute: the symplectic inner product x_P.z_G + z_P.x_G mod 2, which
// dense-side is p.parity_and(pair_swap(g)). Sparse-side the AND is over the modes both rows occupy, so
// it is a merge of the two ascending lane arrays.
[[nodiscard]] inline auto codes_pauli_anticommutes(const SparseRow &p, const SparseRow &g) noexcept -> bool {
    const size_t np = p.num_slots();
    const size_t ng = g.num_slots();
    unsigned int parity = 0;
    size_t i = 0;
    for (size_t k = 0; k < ng; ++k) {
        const size_t g_mode = g.mode(k);
        while (i < np && p.mode(i) < g_mode) {
            ++i;
        }
        if (i < np && p.mode(i) == g_mode) {
            // popcount of the pair-swapped generator field ANDed with p's, both 2 bits wide.
            const unsigned int swapped = ((g.code(k) & 1U) << 1) | ((g.code(k) >> 1) & 1U);
            parity ^= static_cast<unsigned int>(std::popcount(p.code(i) & swapped)) & 1U;
        }
    }
    return parity != 0;
}

// Ordering sign (-1)^S of maj.gen, S = #{set bits of maj strictly below each set bit of gen} mod 2 over
// physical bit positions. Slots ascend in the mode and a mode's low position is 2*mode, so ascending
// slots are ascending positions and one merge walk over the two rows suffices -- O(slots), where the
// dense form is either a prefix-XOR scan over every word or a per-layer full-width mask W plus a
// parity_and per term. The mask has no sparse counterpart worth building: W is dense by construction
// (roughly half the register), so this replaces it with the direct walk instead.
[[nodiscard]] inline auto codes_interleave_phase(const SparseRow &maj, const SparseRow &gen) noexcept -> int {
    const size_t nm = maj.num_slots();
    const size_t ng = gen.num_slots();
    unsigned int parity = 0;
    size_t below_slots = 0; // maj slots at modes strictly below the current generator mode
    for (size_t k = 0; k < ng; ++k) {
        const size_t g_mode = gen.mode(k);
        // Monotone across k, since generator modes ascend: the whole walk is one pass over each row.
        while (below_slots < nm && maj.mode(below_slots) < g_mode) {
            ++below_slots;
        }
        const size_t below = codes_popcount_below(maj.codes, below_slots);
        const unsigned int g_code = gen.code(k);
        // maj's bits at g_mode itself, if it occupies it: position 2*g_mode is below 2*g_mode+1 and so
        // counts for the generator's high bit only.
        const unsigned int m_code = (below_slots < nm && maj.mode(below_slots) == g_mode) ? maj.code(below_slots) : 0U;
        if ((g_code & 1U) != 0U) {
            parity ^= static_cast<unsigned int>(below) & 1U;
        }
        if ((g_code & 2U) != 0U) {
            parity ^= static_cast<unsigned int>(below + (m_code & 1U)) & 1U;
        }
    }
    return parity == 0 ? 1 : -1;
}

// Pauli's per-term rotation sign, the counterpart of pauli_rotation_sign. Same exponent, one merge walk
// instead of a masked pass over the generator's nonzero words:
//   e = g_y + sum(y_mono - y_new) + 2 * sum(v_mono & x_gen),  sign = (e mod 4 == 1 ? -1 : +1)
// where per mode the code's low bit is v (physical position 2*mode) and its high bit is u, so a Y letter
// is the field 0b01 and x = u ^ v is one bit.
//
// The dense version restricts its sums to the words the generator occupies, on the grounds that
// elsewhere the two Y counts cancel and x_gen is zero. Per *mode* that argument is exact and tighter: a
// mode the generator misses has new_mono's field equal to mono's, so the Y terms cancel, and x_gen = 0
// kills the cross term. So this walks the generator's slots and reads mono's field at each, which also
// means new_mono never has to exist -- the sign comes out of the same merge the toggle does.
[[nodiscard]] inline auto codes_pauli_rotation_sign(const SparseRow &mono, const SparseRow &gen) noexcept -> int {
    auto delta = static_cast<long>(codes_pauli_y_count(gen.codes));
    long cross = 0;
    const size_t nm = mono.num_slots();
    const size_t ng = gen.num_slots();
    size_t i = 0;
    for (size_t k = 0; k < ng; ++k) {
        const size_t g_mode = gen.mode(k);
        while (i < nm && mono.mode(i) < g_mode) {
            ++i;
        }
        const unsigned int a = (i < nm && mono.mode(i) == g_mode) ? mono.code(i) : 0U;
        const unsigned int b = gen.code(k);
        delta += (a == 0b01U) ? 1 : 0;
        delta -= ((a ^ b) == 0b01U) ? 1 : 0;
        cross += ((a & 1U) != 0U && ((b ^ (b >> 1)) & 1U) != 0U) ? 1 : 0;
    }
    return mod4(delta + (2 * cross)) == 1 ? -1 : 1;
}

// The product row of a term and a generator, and the overlap the emit phase needs. `codes` and
// `num_slots` describe the row written into `out_lanes`.
struct SparseProduct {
    RowCodes codes = 0;
    size_t num_slots = 0;
    size_t overlap = 0; // popcount(mono & gen); 0 and meaningless when overflowed
    bool overflowed = false;
};

// mono (+) gen: per mode the fields XOR, a mode whose field cancels to zero disappears, and the overlap
// is the popcount of the fields' AND. This is the dense fused_xor_into in support form, and it is the
// operation the whole representation exists for -- one merge over two ascending lane arrays, O(slots),
// against a pass over every storage word.
//
// out_lanes must hold `capacity` lanes and capacity must not exceed kRowMaxSlots (a codes word's worth).
// The product can occupy more modes than either input: up to mono's slots plus the generator's, so a
// scratch row needs CutoffEvaluator::max_mode_bound() + the generator's locality, not just the bound.
// When even that is not enough the result is reported as overflowed rather than truncated -- a truncated
// mode list keeps a plausible-looking codes word, which is exactly how the Stage 3 bench measured a
// capacity bug as if it were a speedup. On overflow the caller must fall back to the dense product;
// `overlap` is partial and is deliberately not returned.
[[nodiscard]] inline auto sparse_toggle(const SparseRow &mono,
                                        const SparseRow &gen,
                                        RowMode *out_lanes,
                                        size_t capacity) noexcept -> SparseProduct {
    assert(capacity <= kRowMaxSlots && "sparse_toggle capacity exceeds one codes word");
    const size_t nm = mono.num_slots();
    const size_t ng = gen.num_slots();
    SparseProduct result;
    size_t used = 0;
    bool over = false;

    const auto emit = [&](size_t mode, unsigned int code) {
        if (used == capacity) {
            over = true;
            return;
        }
        out_lanes[used] = static_cast<RowMode>(mode);
        result.codes |= static_cast<RowCodes>(code) << (2 * used);
        ++used;
    };

    size_t i = 0;
    size_t j = 0;
    while (!over && i < nm && j < ng) {
        const size_t m_mode = mono.mode(i);
        const size_t g_mode = gen.mode(j);
        if (m_mode < g_mode) {
            emit(m_mode, mono.code(i));
            ++i;
        }
        else if (g_mode < m_mode) {
            emit(g_mode, gen.code(j));
            ++j;
        }
        else {
            const unsigned int a = mono.code(i);
            const unsigned int b = gen.code(j);
            result.overlap += static_cast<size_t>(std::popcount(a & b));
            if (const unsigned int c = a ^ b; c != 0U) {
                emit(m_mode, c);
            }
            ++i;
            ++j;
        }
    }
    while (!over && i < nm) {
        emit(mono.mode(i), mono.code(i));
        ++i;
    }
    while (!over && j < ng) {
        emit(gen.mode(j), gen.code(j));
        ++j;
    }
    if (over) {
        return SparseProduct{0, 0, 0, true};
    }
    result.num_slots = used;
    return result;
}

} // namespace monoprop::detail
