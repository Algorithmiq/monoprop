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

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "monoprop/TypeAliases.h"

// Logical mode count at or above which the sparse rows are the cheaper backend. Build-time and not a
// runtime knob because what moves the crossover is the target ISA, which is fixed when the translation
// unit is compiled: dense costs one pass per storage word and sparse is flat in the width, so without a
// vector popcount the dense pass degrades an order of magnitude sooner. Set from CMake off
// monoprop_ENABLE_ARCH_FLAGS (see the top-level CMakeLists for the measured values); the fallback here
// is the conservative baseline-x86-64 one, so a consumer compiling these headers without the project's
// definitions gets the value that suits the wheels rather than the developer's machine.
#if !defined(monoprop_SPARSE_ROW_MIN_MODES)
#define monoprop_SPARSE_ROW_MIN_MODES 96
#endif

namespace monoprop::detail {

// The requested width needs mode indices wider than ModeT can hold, or more slots than a codes word
// has room for. Thrown rather than asserted, for the same reason OperatorIndexWidthUnsupported is:
// with the compile-time mode ceiling gone, the width is user data.
class SparseRowStoreUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Operator-term store in support form: each row is a fixed-width list of the *modes* it occupies plus
// one word holding two bits per occupied mode. It is the third backend behind the four TypeAliases.h
// row accessors, alongside std::vector<Bitset> and OperatorIndex, and agrees with both through them
// (cpp/tests/row_accessor_tests.cpp).
//
// Layout, structure-of-arrays: modes_ is `slots_per_row_` ModeT lanes per row, ascending, padded with
// kPadLane; codes_ is one CodesT per row. The two live in separate arrays on purpose -- the cutoff and
// pairing algebra reads only codes_, so evaluating it over a run of rows is a sequential walk that
// never touches a mode list.
//
// codes_ packs slot j (the j-th occupied mode, ascending) into bits 2j and 2j+1: bit 2j marks physical
// position 2*mode, bit 2j+1 marks 2*mode+1. The whole cutoff algebra follows from that one word --
// with occupied = (codes | codes>>1) & 0x5555..., paired = codes & (codes>>1) & 0x5555...,
// n = popcount(occupied) and d = popcount(paired) give or_sum = n, popcount_sum = n + d and
// xor_sum = n - d, independent of the storage width. popcount() below is the first consumer; the rest
// arrives with the algebra port.
//
// Rows wider than slots_per_row_ spill losslessly to a side map. They are not a corner case to be
// ruled out by sizing: a fully-paired term escapes the cutoff (xor_sum == 0 is kept unconditionally),
// so support is genuinely unbounded no matter what the cutoff is. They are rare -- ~0.07% of rows on
// production models, per MPOperator.h -- and an all-0b11 row needs only its mode list, so a second
// cheap row kind is available if that ever stops being true.
//
// No hash index: this is the row half only. Keying sparse rows is a separate question (a padded row
// makes equality a memcmp and the hash a fixed-width mix, which changes the hash and so forfeits
// bit-identity), and it is answered where the default flips rather than here, since a table built now
// would duplicate OperatorIndex's with no consumer.
//
// Single-writer, like OperatorIndex: one partition, one thread; parallelism is cross-partition.
class SparseRowStore {
public:
    using value_type = Bitset;
    using key_type = Bitset;
    using mapped_type = size_t;
    using ModeT = uint16_t;
    using CodesT = uint64_t;

    // Two bits per slot in one CodesT. A wider row is representable in modes_ but would put the algebra
    // back on a multi-word loop, which is the whole cost the support form removes.
    static constexpr size_t kMaxSlots = 32;
    static constexpr size_t kDefaultSlots = 8;

    // The top two ModeT values are markers, so a valid mode index is at most kPadLane - 2. kPadLane
    // fills the unused lanes of a short row (fixed, so two equal rows have equal lanes); kOverflowLane
    // sits in lane 0 of a spilled row, where it cannot be confused with the empty row's kPadLane.
    static constexpr ModeT kPadLane = std::numeric_limits<ModeT>::max();
    static constexpr ModeT kOverflowLane = static_cast<ModeT>(kPadLane - 1);
    static constexpr size_t kMaxModes = static_cast<size_t>(kOverflowLane); // exclusive bound

    static constexpr uint64_t kLoBits = 0x5555555555555555ULL; // bit 2j of every slot

    // The Stage 3 crossover, as a predicate rather than a bare number so the rule has one home.
    static constexpr size_t kMinModes = monoprop_SPARSE_ROW_MIN_MODES;
    [[nodiscard]] static constexpr auto preferred_for_modes(size_t num_modes) noexcept -> bool {
        return num_modes >= kMinModes;
    }

    // num_bits is the storage bit width of every monomial this store will hold, exactly as for
    // OperatorIndex: row() reconstructs at that width, and a wrong one changes num_words() and with it
    // the hash, the probe order and MPI owner routing. slots_per_row is the per-row mode capacity --
    // any value is correct, since over-long rows spill; size it from CutoffEvaluator::max_mode_bound().
    explicit SparseRowStore(size_t num_bits, size_t slots_per_row = kDefaultSlots)
        : num_bits_(num_bits),
          slots_per_row_(std::clamp<size_t>(slots_per_row, 1, kMaxSlots)) {
        if (((num_bits + 1) / 2) > kMaxModes) {
            throw SparseRowStoreUnsupported(
                std::format("SparseRowStore supports at most {} modes ({} bits); got {} bits ({} modes).",
                            kMaxModes,
                            2 * kMaxModes,
                            num_bits,
                            (num_bits + 1) / 2));
        }
    }

    SparseRowStore(const SparseRowStore &) = delete;
    SparseRowStore &operator=(const SparseRowStore &) = delete;
    SparseRowStore(SparseRowStore &&) = delete;
    SparseRowStore &operator=(SparseRowStore &&) = delete;

    [[nodiscard]] auto num_bits() const noexcept -> size_t { return num_bits_; }
    [[nodiscard]] auto slots_per_row() const noexcept -> size_t { return slots_per_row_; }
    [[nodiscard]] auto size() const noexcept -> size_t { return size_; }

    // Called only on an idle store, so it needs no synchronization.
    [[nodiscard]] auto clone() const -> std::unique_ptr<SparseRowStore> {
        auto out = std::make_unique<SparseRowStore>(num_bits_, slots_per_row_);
        out->modes_ = modes_;
        out->codes_ = codes_;
        out->size_ = size_;
        out->overflow_ = overflow_;
        return out;
    }

    auto reserve(size_t n) -> void {
        modes_.reserve(n * slots_per_row_);
        codes_.reserve(n);
    }

    // Returns the pre-growth size (the caller's insert base). Growth is geometric (1.5x), never
    // exact-fit: an exact fit would realloc the whole operator every layer.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve(std::max(base + n, cap + (cap / 2) + 1));
        }
        // Default-init grow, not a zeroing resize: every freshly grown row is overwritten by set()
        // before any read, so a tail zero-fill would be wasted bandwidth.
        modes_.resize((base + n) * slots_per_row_);
        codes_.resize(base + n);
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so nothing in the row is pre-read; a
    // stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        ModeT *lanes = &modes_[i * slots_per_row_];
        CodesT codes = 0;
        size_t used = 0;
        // Positions arrive ascending, so a mode's two positions are adjacent and a new mode is always
        // the next slot -- no search, and the lanes come out sorted for free.
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            const size_t mode = b >> 1;
            if (used == 0 || static_cast<size_t>(lanes[used - 1]) != mode) {
                if (used == slots_per_row_) {
                    lanes[0] = kOverflowLane;
                    codes_[i] = 0;
                    overflow_[i] = mono;
                    return;
                }
                lanes[used++] = static_cast<ModeT>(mode);
            }
            codes |= CodesT{1} << ((2 * (used - 1)) + (b & 1U));
        }
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
        for (size_t j = used; j < slots_per_row_; ++j) {
            lanes[j] = kPadLane;
        }
        codes_[i] = codes;
    }

    [[nodiscard]] auto row(size_t i) const -> value_type {
        if (spilled(i)) {
            return overflow_.at(i);
        }
        value_type mono(num_bits_);
        for_each_slot(i, [&](size_t mode, unsigned int code) {
            if ((code & 1U) != 0U) {
                mono.set(2 * mode);
            }
            if ((code & 2U) != 0U) {
                mono.set((2 * mode) + 1);
            }
        });
        return mono;
    }

    // Ascending, matching the dense backends: slots are stored ascending in the mode, and within a mode
    // position 2*mode precedes 2*mode+1.
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        if (spilled(i)) {
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        for_each_slot(i, [&](size_t mode, unsigned int code) {
            if ((code & 1U) != 0U) {
                fn(2 * mode);
            }
            if ((code & 2U) != 0U) {
                fn((2 * mode) + 1);
            }
        });
    }

    // Visits (mode, code) per occupied slot, ascending in the mode; code is the 2-bit field, so 0b01 is
    // position 2*mode alone, 0b10 is 2*mode+1 alone and 0b11 is the paired mode. Row i must not be
    // spilled -- a spilled row has no slots, and its lane 0 marker would read as a mode.
    template <typename Fn>
    auto for_each_slot(size_t i, Fn &&fn) const -> void {
        assert(!spilled(i) && "SparseRowStore::for_each_slot on a spilled row");
        const ModeT *lanes = &modes_[i * slots_per_row_];
        const CodesT codes = codes_[i];
        for (size_t j = 0; j < slots_per_row_ && lanes[j] != kPadLane; ++j) {
            fn(static_cast<size_t>(lanes[j]), static_cast<unsigned int>((codes >> (2 * j)) & 0b11U));
        }
    }

    // The row's codes word. Meaningless for a spilled row -- ask spilled(i) first; the algebra port
    // will need the same guard, which is why the spill is kept rare rather than made general.
    [[nodiscard]] auto codes(size_t i) const -> CodesT { return codes_[i]; }

    [[nodiscard]] auto spilled(size_t i) const -> bool { return modes_[i * slots_per_row_] == kOverflowLane; }

    // Occupied modes -- the support measure, or_sum.
    [[nodiscard]] auto slot_count(size_t i) const -> size_t {
        if (spilled(i)) {
            return occupied_modes(overflow_.at(i));
        }
        return static_cast<size_t>(std::popcount(occupied_bits(codes_[i])));
    }

    // Set bits -- the length measure, popcount_sum = n + d straight off the codes word.
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        if (spilled(i)) {
            return overflow_.at(i).count();
        }
        const CodesT codes = codes_[i];
        return static_cast<size_t>(std::popcount(occupied_bits(codes)))
               + static_cast<size_t>(std::popcount(paired_bits(codes)));
    }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = modes_.capacity() * sizeof(ModeT);
        total += codes_.capacity() * sizeof(CodesT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    // Diagnostic: the part of memory_bytes() that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        const size_t lanes = modes_.capacity() - std::min(modes_.capacity(), size_ * slots_per_row_);
        const size_t words = codes_.capacity() - std::min(codes_.capacity(), size_);
        return (lanes * sizeof(ModeT)) + (words * sizeof(CodesT));
    }

    // Slot count for a cutoff bound in modes (CutoffEvaluator::max_mode_bound()), clamped to what one
    // codes word holds. A bound above kMaxSlots is not an error: the rows that exceed it spill.
    [[nodiscard]] static auto slots_for_bound(size_t mode_bound) noexcept -> size_t {
        return std::clamp<size_t>(mode_bound, 1, kMaxSlots);
    }

    // Bit 2j of each slot: set iff slot j is occupied at all. popcount is n.
    [[nodiscard]] static constexpr auto occupied_bits(CodesT codes) noexcept -> CodesT {
        return (codes | (codes >> 1)) & kLoBits;
    }
    // Bit 2j of each slot: set iff slot j holds both of its positions. popcount is d.
    [[nodiscard]] static constexpr auto paired_bits(CodesT codes) noexcept -> CodesT {
        return codes & (codes >> 1) & kLoBits;
    }

private:
    // Spilled rows have no codes word, so their support is counted the dense way. Adjacent set bits do
    // not straddle a word: a mode's two positions are 2m and 2m+1.
    [[nodiscard]] static auto occupied_modes(const value_type &mono) -> size_t {
        size_t n = 0;
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            if ((b & 1U) == 0U || !mono.test(b - 1)) {
                ++n;
            }
        }
        return n;
    }

    [[nodiscard]] auto capacity() const -> size_t { return codes_.capacity(); }

    DefaultInitVector<ModeT> modes_ = {};
    DefaultInitVector<CodesT> codes_ = {};
    size_t num_bits_ = 0;
    size_t size_ = 0;
    size_t slots_per_row_ = kDefaultSlots;
    // Lossless side-map for rows occupying more than slots_per_row_ modes.
    std::unordered_map<size_t, value_type> overflow_ = {};
};

} // namespace monoprop::detail
