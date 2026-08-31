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

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <format>
#include <optional>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop {
// The per-mode sums the structural cutoffs measure, over the active modes only.
struct CutoffSums {
    size_t xor_sum;      // modes with exactly one of their two Majoranas set; 0 == fully paired
    size_t popcount_sum; // Majorana operators present -- the length measure
    size_t or_sum;       // modes with either Majorana present -- the support measure (JW Pauli weight)
};

// Everything cutoff_sums would otherwise derive for every term. It depends only on the storage width
// and the logical width, both fixed for a propagator's lifetime, so it is built once and carried by the
// cutoff functors below.
//
// This exists for a measured reason. With a compile-time width these were constant expressions; with a
// runtime width, rederiving them per term costs ~2.3 ns/term at one word -- the whole single-word
// kernel is only about that -- which measured as +16% on a 32-mode in-place propagation and +2-3% on
// the 120/127-qubit models. Note what did *not* work: routing the multi-word loop through
// detail::with_nwords, the way Bitset's own word ops do, is 50-66% *slower* than this in a plain -O3
// build (fine under -march=native), so the per-word loop below stays an ordinary runtime loop.
struct CutoffMasks {
    uint64_t active = 0;      // single-word arm: valid bits AND the active window
    uint64_t even_active = 0; // single-word arm: the even-bit pattern AND `active`
    size_t active_bit_offset = 0;
    size_t num_bits = 0;                     // the width these were built for; only checked in assertions
    size_t first_active_word = 0;            // multi-word arm: words below this one are entirely inactive
    uint64_t first_word_mask = ~uint64_t{0}; // multi-word arm: the active bits within that word

    // Derived from num_bits/active_bit_offset rather than stored: each is a one-line fact about the
    // other two fields, read at 3 call sites total.
    [[nodiscard]] auto single_word() const -> bool { return num_bits <= 64; }
    [[nodiscard]] auto whole_register() const -> bool { return active_bit_offset == 0; }

    [[nodiscard]] static auto make(size_t num_bits, size_t logical_num_modes) -> CutoffMasks {
        CutoffMasks m;
        const size_t num_modes = num_bits / 2;
        m.num_bits = num_bits;
        m.active_bit_offset = 2 * (num_modes - logical_num_modes);
        if (m.single_word()) {
            // The even-bit pattern spelled as a literal rather than via even_bits(): identical value,
            // plain integer arithmetic, and no Bitset to construct.
            const uint64_t valid = num_bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << num_bits) - 1);
            m.active = m.active_bit_offset == 0 ? valid : (valid & ~((uint64_t{1} << m.active_bit_offset) - 1));
            m.even_active = (0x5555555555555555ULL & valid) & m.active;
        }
        else {
            // The same "mask off the inactive low bits" the single-word arm above does, split into the
            // word to start at and the mask for that word. The active window is the *high* end of the
            // register, so every word below first_active_word contributes nothing to any of the three
            // sums and is skipped rather than masked.
            m.first_active_word = m.active_bit_offset / Bitset::word_width;
            const size_t within = m.active_bit_offset % Bitset::word_width;
            m.first_word_mask = within == 0 ? ~uint64_t{0} : ~((uint64_t{1} << within) - 1);
        }
        return m;
    }
};

[[gnu::always_inline]] inline auto cutoff_sums(const MonomialLike auto &mono, const CutoffMasks &masks) -> CutoffSums {
    assert(mono.size() == masks.num_bits && "cutoff masks built for a different width");

    // A runtime branch, not `if constexpr`: the width is data now. Still a branch rather than folded
    // into the general loop because this arm skips the mask lookup and the loop entirely, and one word
    // covers every model up to 32 modes.
    if (masks.single_word()) {
        const uint64_t active_word = mono.word(0) & masks.active;
        const uint64_t first_pair = active_word & masks.even_active;
        const uint64_t second_pair = (active_word >> 1) & masks.even_active;
        return {static_cast<size_t>(std::popcount(first_pair ^ second_pair)),
                static_cast<size_t>(std::popcount(active_word)),
                static_cast<size_t>(std::popcount(first_pair | second_pair))};
    }
    // One pass over the words, with no Bitset temporaries. The `active & mask` / `(active >> 1) & mask`
    // / `^` / `|` / `>>` chain this replaces built five of them per term, and since Stage 2b each is a
    // full runtime-width object construction rather than a trivially copyable value.
    //
    // (word >> 1) & even_mask equals ((bits >> 1) & mask).word(w): a full-width shift carries the low
    // bit of word w+1 into bit 63 of word w, which is an odd position and so masked off regardless.
    // The same within-word-pairs argument pair_swap() and pauli_uv() already rely on.
    //
    // A narrower active window is masked, not shifted -- `mono >> active_bit_offset` would copy the
    // whole monomial per term (and allocate, past kInlineWords), and it is the *common* case: the
    // storage width rounds up to a whole 32-mode block, so the offset is non-zero for any logical
    // width that is not a multiple of 32. Masking gives the same three sums because the offset is even,
    // so a mode's two bits are dropped or kept together and every surviving mode keeps its parity --
    // exactly the argument the single-word arm above already rests on. word_mask applies to
    // first_active_word only; the assignment in the loop is a register move rather than a branch.
    const auto &mask = cached_even_bits<LSb0>(masks.num_bits);
    size_t xor_sum = 0;
    size_t popcount_sum = 0;
    size_t or_sum = 0;
    const size_t nw = mono.num_words();
    uint64_t word_mask = masks.first_word_mask;
    for (size_t w = masks.first_active_word; w < nw; ++w) {
        const uint64_t word = mono.word(w) & word_mask;
        word_mask = ~uint64_t{0};
        const uint64_t m = mask.word(w);
        const uint64_t first_pair = word & m;
        const uint64_t second_pair = (word >> 1) & m;
        xor_sum += static_cast<size_t>(std::popcount(first_pair ^ second_pair));
        popcount_sum += static_cast<size_t>(std::popcount(word));
        or_sum += static_cast<size_t>(std::popcount(first_pair | second_pair));
    }
    return {xor_sum, popcount_sum, or_sum};
}

// Cold-path form: derives the masks per call. Every per-term caller goes through a cutoff functor,
// which holds them.
[[gnu::always_inline]] inline auto cutoff_sums(const MonomialLike auto &mono, size_t logical_num_modes) -> CutoffSums {
    return cutoff_sums(mono, CutoffMasks::make(mono.size(), logical_num_modes));
}

// Both cutoffs below keep a fully paired monomial (xor_sum == 0) unconditionally: those are the only
// terms contributing to an expectation value against a product reference state, so bounding them by
// length or support would discard signal.

auto length_cutoff(const MonomialLike auto &mono, unsigned int cutoff, const CutoffMasks &masks) -> bool {
    const auto sums = cutoff_sums(mono, masks);
    return sums.xor_sum == 0 || sums.popcount_sum <= cutoff;
}

auto length_cutoff(const MonomialLike auto &mono, unsigned int cutoff, size_t logical_num_modes) -> bool {
    return length_cutoff(mono, cutoff, CutoffMasks::make(mono.size(), logical_num_modes));
}

// Whole-register overload: every mode is active. Reads the width off the instance -- a qualified
// decltype(mono)::size() would be ill-formed for a plain Bitset.
auto length_cutoff(const MonomialLike auto &mono, unsigned int cutoff) -> bool {
    return length_cutoff(mono, cutoff, mono.size() / 2);
}

auto support_cutoff(const MonomialLike auto &mono, unsigned int cutoff, const CutoffMasks &masks) -> bool {
    const auto sums = cutoff_sums(mono, masks);
    return sums.xor_sum == 0 || sums.or_sum <= cutoff;
}

auto support_cutoff(const MonomialLike auto &mono, unsigned int cutoff, size_t logical_num_modes) -> bool {
    return support_cutoff(mono, cutoff, CutoffMasks::make(mono.size(), logical_num_modes));
}

auto support_cutoff(const MonomialLike auto &mono, unsigned int cutoff) -> bool {
    return support_cutoff(mono, cutoff, mono.size() / 2);
}

namespace detail {

// Both hold their masks, so the per-term call does no width arithmetic. Real constructors rather than
// aggregate initialization, deliberately: the width used to arrive free from NumModes, and both
// remaining ways to get it wrong are silent. A logical width of 0 makes cutoff_sums' active window
// empty, so xor_sum is 0 and the cutoff keeps *everything*; masks that disagree with
// logical_num_modes do the same kind of damage. A constructor makes both unrepresentable -- with an
// aggregate, a designated initializer that omitted either field would just zero it.
struct LengthCutoff {
    unsigned int cutoff;
    size_t logical_num_modes;
    CutoffMasks masks;

    LengthCutoff(unsigned int cutoff_, size_t logical_num_modes_, size_t num_bits)
        : cutoff(cutoff_),
          logical_num_modes(logical_num_modes_),
          masks(CutoffMasks::make(num_bits, logical_num_modes_)) {}

    auto operator()(const Bitset &mono) const -> bool { return length_cutoff(mono, cutoff, masks); }
};

struct SupportCutoff {
    unsigned int cutoff;
    size_t logical_num_modes;
    CutoffMasks masks;

    SupportCutoff(unsigned int cutoff_, size_t logical_num_modes_, size_t num_bits)
        : cutoff(cutoff_),
          logical_num_modes(logical_num_modes_),
          masks(CutoffMasks::make(num_bits, logical_num_modes_)) {}

    auto operator()(const Bitset &mono) const -> bool { return support_cutoff(mono, cutoff, masks); }
};

class CutoffEvaluator {
public:
    // The two target<>() probes recover the concrete functor behind the type-erased CutoffFn so the hot
    // paths below can call it directly and read its cutoff. std::function::target<T>() matches only on
    // the *exact* stored type, so anything that wraps the functor -- a lambda, a different width, a
    // structurally identical copy of the type -- yields nullptr and silently falls back to calling
    // through the std::function. That is correct but slower, and invisible: no test and no bit-identity
    // check can see it. cutoff_function() asserts the handshake at the point where the type is chosen.
    explicit CutoffEvaluator(const CutoffFn &cutoff_fn)
        : cutoff_fn_(cutoff_fn),
          length_cutoff_(cutoff_fn.target<LengthCutoff>()),
          support_cutoff_(cutoff_fn.target<SupportCutoff>()) {}

    auto length_cutoff() const -> const LengthCutoff * { return length_cutoff_; }

    auto support_cutoff() const -> const SupportCutoff * { return support_cutoff_; }

    auto operator()(const Bitset &mono) const -> bool {
        if (length_cutoff_ != nullptr) {
            return (*length_cutoff_)(mono);
        }
        if (support_cutoff_ != nullptr) {
            return (*support_cutoff_)(mono);
        }
        return cutoff_fn_(mono);
    }

    // Fast path when popcount(mono) is known: the predicate is `xor_sum==0 || (popcount/or_sum)<=cutoff`,
    // so popcount<=cutoff alone proves keep without reading the bitset (or_sum<=popcount makes support safe).
    auto passes_with_popcount(const Bitset &mono, size_t popcount_sum) const -> bool {
        if (length_cutoff_ != nullptr) {
            if (popcount_sum <= length_cutoff_->cutoff) {
                return true;
            }
            return (*length_cutoff_)(mono);
        }
        if (support_cutoff_ != nullptr) {
            if (popcount_sum <= support_cutoff_->cutoff) {
                return true;
            }
            return (*support_cutoff_)(mono);
        }
        return cutoff_fn_(mono);
    }

    // Upper bound on the set bits (physical slots) a surviving term can carry, so the store can size
    // its packed inline rows. A length cutoff counts set bits directly; a support cutoff counts
    // modes/qubits, each spanning two slots, hence the x2.
    auto max_slot_bound() const -> std::optional<size_t> {
        if (length_cutoff_ != nullptr) {
            return length_cutoff_->cutoff;
        }
        if (support_cutoff_ != nullptr) {
            return 2 * support_cutoff_->cutoff;
        }
        return std::nullopt;
    }

    // The same bound counted in modes/qubits rather than slots, which is what a store keyed by mode
    // (SparseRowStore) sizes its rows from. It is `cutoff` for both kinds and not max_slot_bound()/2:
    // a support cutoff admits `cutoff` modes by definition, and a length cutoff of `cutoff` slots is
    // worst-case `cutoff` singly-occupied modes -- halving would truncate that row.
    //
    // This bounds a *stored* row. A row being toggled in place transiently exceeds it, by as many
    // modes as the generator touches, so a scratch row needs max_mode_bound() + generator locality.
    auto max_mode_bound() const -> std::optional<size_t> {
        if (length_cutoff_ != nullptr) {
            return length_cutoff_->cutoff;
        }
        if (support_cutoff_ != nullptr) {
            return support_cutoff_->cutoff;
        }
        return std::nullopt;
    }

private:
    const CutoffFn &cutoff_fn_;
    const LengthCutoff *length_cutoff_;
    const SupportCutoff *support_cutoff_;
};

} // namespace detail

} // namespace monoprop
