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
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"

namespace monoprop {

// A Majorana/Pauli index at or past the width of the system it is being applied to.
class AlgebraIndexOutOfRange : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A coefficient with no real encoding under the algebra model: non-Hermitian for Majorana products,
// non-real for Pauli strings.
class NonEncodableCoefficient : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Unchecked: `num_bits - 1 - bit_loc` underflows for an out-of-range index and Bitset::set is
// noexcept, so the result is an out-of-bounds write. Use indices_to_bitset_checked() for user input.
inline auto indices_to_bitset(const VecZ &arr, size_t num_bits) -> Bitset {
    Bitset bs(num_bits);
    for (const auto &bit_loc : arr) {
        bs.set(num_bits - 1 - bit_loc); // MSb0 convention: index 0 maps to the top bit
    }
    return bs;
}

// The two bounds are different quantities and neither implies the other, so they are separate
// arguments: max_index is the *logical* width (2 * logical_num_modes), which is what a caller's
// indices must fall inside, while num_bits is the *storage* width the result is built at. A
// propagator running fewer modes than its storage holds must still reject indices outside its own
// system, and storage rounds up.
inline auto indices_to_bitset_checked(const VecZ &arr, size_t max_index, size_t num_bits) -> Bitset {
    for (const auto &bit_loc : arr) {
        if (bit_loc >= max_index) {
            throw AlgebraIndexOutOfRange(
                std::format("Majorana/Pauli index {} is out of range; must be less than {}.", bit_loc, max_index));
        }
    }
    return indices_to_bitset(arr, num_bits);
}

// O(popcount) via find_first/find_next rather than an O(num_bits) scan.
auto bitset_to_indices(const MonomialLike auto &bs) -> VecZ {
    const auto pop = bs.count();
    VecZ indices(pop);
    size_t idx = pop;
    for (size_t pos = bs.find_first(); pos < bs.size(); pos = bs.find_next(pos)) {
        indices[--idx] = bs.size() - 1 - pos;
    }
    return indices;
}

auto is_paired(const MonomialLike auto &mono, const auto &even_mask) -> bool {
    // Paired = each mode's even bit and its odd partner agree (both set or both clear). Word loop
    // rather than `(mono & m) ^ ((mono >> 1) & m)`, which built three runtime-width temporaries per
    // call; (word >> 1) & m is within-word for the same reason as in cutoff_sums().
    const size_t nw = mono.num_words();
    for (size_t w = 0; w < nw; ++w) {
        const uint64_t word = mono.word(w);
        const uint64_t m = even_mask.word(w);
        if (((word & m) ^ ((word >> 1) & m)) != 0) {
            return false;
        }
    }
    return true;
}

auto is_paired(const MonomialLike auto &mono) -> bool {
    return is_paired(mono, cached_even_bits<LSb0>(mono.size()));
}

// `mono` is an index list, not a monomial, so there is no argument to deduce a width from -- hence the
// explicit num_bits, which sizes the bitset this builds.
inline auto is_paired(const VecZ &mono, size_t num_bits) -> bool {
    return is_paired(indices_to_bitset(mono, num_bits));
}

// Rows carries no structural width of its own (unlike a MonomialLike argument), so the width is
// explicit here too. It must be the width of the rows themselves: the mask is compared against them
// pairwise, and a mismatch trips Bitset's width assertions.
template <typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op, size_t num_bits) -> VecZ {
    VecZ result;
    // Memoized rather than rebuilt: the reference stays valid across the loop because is_paired's
    // two-argument form takes the mask and so never re-enters the cache with another width.
    const auto &mask = cached_even_bits<LSb0>(num_bits);
    for (const auto index : inds) {
        const auto &op_row = materialize_row(op, index);
        if (is_paired(op_row, mask)) {
            result.push_back(index);
        }
    }
    return result;
}

// Occupation mask of the initial product state: the even index 2*i of each listed mode (Majorana) or
// qubit (Pauli) that starts in state 1. Both algebras read the same mask and differ only in the phase
// they score against it (majorana_state_phase / pauli_state_phase).
inline auto initial_state_mask(const VecZ &initial_state, size_t num_bits) -> Bitset {
    VecZ bits;
    bits.reserve(initial_state.size());
    for (const auto &mode : initial_state) {
        bits.push_back(2 * mode);
    }
    return indices_to_bitset(bits, num_bits);
}

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
    size_t num_bits = 0; // the width these were built for; only checked in assertions
    bool single_word = false;
    bool whole_register = false;

    [[nodiscard]] static auto make(size_t num_bits, size_t logical_num_modes) -> CutoffMasks {
        CutoffMasks m;
        const size_t num_modes = num_bits / 2;
        m.num_bits = num_bits;
        m.active_bit_offset = 2 * (num_modes - logical_num_modes);
        m.single_word = num_bits <= 64;
        m.whole_register = logical_num_modes == num_modes;
        if (m.single_word) {
            // The even-bit pattern spelled as a literal rather than via even_bits(): identical value,
            // plain integer arithmetic, and no Bitset to construct.
            const uint64_t valid = num_bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << num_bits) - 1);
            m.active = m.active_bit_offset == 0 ? valid : (valid & ~((uint64_t{1} << m.active_bit_offset) - 1));
            m.even_active = (0x5555555555555555ULL & valid) & m.active;
        }
        return m;
    }
};

[[gnu::always_inline]] inline auto cutoff_sums(const MonomialLike auto &mono, const CutoffMasks &masks) -> CutoffSums {
    assert(mono.size() == masks.num_bits && "cutoff masks built for a different width");

    // A runtime branch, not `if constexpr`: the width is data now. Still a branch rather than folded
    // into the general loop because this arm skips the mask lookup and the loop entirely, and one word
    // covers every model up to 32 modes.
    if (masks.single_word) {
        const uint64_t active_word = mono.word(0) & masks.active;
        const uint64_t first_pair = active_word & masks.even_active;
        const uint64_t second_pair = (active_word >> 1) & masks.even_active;
        return {static_cast<size_t>(std::popcount(first_pair ^ second_pair)),
                static_cast<size_t>(std::popcount(active_word)),
                static_cast<size_t>(std::popcount(first_pair | second_pair))};
    }
    const size_t num_bits = masks.num_bits;
    const size_t active_bit_offset = masks.active_bit_offset;

    // One pass over the words, with no Bitset temporaries. The `active & mask` / `(active >> 1) & mask`
    // / `^` / `|` / `>>` chain this replaces built five of them per term, and since Stage 2b each is a
    // full runtime-width object construction rather than a trivially copyable value.
    //
    // (word >> 1) & even_mask equals ((bits >> 1) & mask).word(w): a full-width shift carries the low
    // bit of word w+1 into bit 63 of word w, which is an odd position and so masked off regardless.
    // The same within-word-pairs argument pair_swap() and pauli_uv() already rely on.
    const auto &mask = cached_even_bits<LSb0>(num_bits);
    const auto accumulate = [&mask](const auto &bits) -> CutoffSums {
        size_t xor_sum = 0;
        size_t popcount_sum = 0;
        size_t or_sum = 0;
        const size_t nw = bits.num_words();
        for (size_t w = 0; w < nw; ++w) {
            const uint64_t word = bits.word(w);
            const uint64_t m = mask.word(w);
            const uint64_t first_pair = word & m;
            const uint64_t second_pair = (word >> 1) & m;
            xor_sum += static_cast<size_t>(std::popcount(first_pair ^ second_pair));
            popcount_sum += static_cast<size_t>(std::popcount(word));
            or_sum += static_cast<size_t>(std::popcount(first_pair | second_pair));
        }
        return {xor_sum, popcount_sum, or_sum};
    };

    // The shift is the uncommon case (logical_num_modes < num_modes); keep its temporary out of the
    // path that does not need one.
    if (masks.whole_register) {
        return accumulate(mono);
    }
    return accumulate(mono >> active_bit_offset);
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
    const auto sums = cutoff_sums(mono, logical_num_modes);
    return sums.xor_sum == 0 || sums.popcount_sum <= cutoff;
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
    const auto sums = cutoff_sums(mono, logical_num_modes);
    return sums.xor_sum == 0 || sums.or_sum <= cutoff;
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
