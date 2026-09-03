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

// Per-partition scratch the layer build reuses from gate to gate. Nothing here carries state between
// gates -- every member is re-initialised by the gate that uses it -- so it is owned by the propagator
// only to keep its capacity, and a copied propagator starts with an empty one. `counters` is the one
// exception: it accumulates over a call, and its owner resets it (MonomialPropagator::run_gate_loop_).

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "monoprop/detail/evolution/layer_build/BucketJoin.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

// The protocol's per-row state for one gate (Engine.h has the rules), as five bitsets over row indices.
// The scan sets `rot` (= E(ν), the term's own emission predicate) and `foll` (the pivot bit: which
// endpoint of a pair the row is); the join sets `received` (the partner's record arrived, i.e. the
// partner is tracked) and `partner_rot` (that record's rot bit); round 2 sets `answered` (the partner
// is tracked but silent, and sent its coefficient back instead of a record).
//
// Only rows of Anti(G) are ever set or read, and Anti(G) is exactly the set bits of the gate's `nz`
// words, so a gate clears its state by zeroing one word per nz word rather than the whole operator's
// worth of bits. Rows inserted by the gate's own mints are past the pre-gate size and carry no state.
class RowMarks {
public:
    // Sizes the bitsets to `n_rows` and clears the words `nz` names. Must run after the fold's pass 1
    // (which produces `nz`) and before the emit pass sets a bit.
    auto begin(size_t n_rows, std::span<const EvenParityNzWord> nz) -> void {
        const size_t words = (n_rows + 63) / 64;
        for (auto *bits : arrays_()) {
            if (bits->size() < words) {
                bits->resize(words, 0);
            }
        }
        for (const auto &w : nz) {
            const size_t wi = w.base / 64;
            for (auto *bits : arrays_()) {
                (*bits)[wi] = 0;
            }
        }
    }

    auto set_rot(size_t row) -> void { set_(rot_, row); }
    auto set_foll(size_t row) -> void { set_(foll_, row); }
    // The whole pivot word at once: `nz` already carries the followers of word `word` as `foll`, and
    // begin() has just zeroed it, so the scan sets them in one store instead of one per anticommuting row.
    auto set_foll_word(size_t word, uint64_t bits) -> void { foll_[word] = bits; }
    auto set_received(size_t row) -> void { set_(received_, row); }
    auto set_partner_rot(size_t row) -> void { set_(partner_rot_, row); }
    auto set_answered(size_t row) -> void { set_(answered_, row); }
    [[nodiscard]] auto rot(size_t row) const -> bool { return get_(rot_, row); }
    [[nodiscard]] auto foll(size_t row) const -> bool { return get_(foll_, row); }
    [[nodiscard]] auto received(size_t row) const -> bool { return get_(received_, row); }
    [[nodiscard]] auto partner_rot(size_t row) const -> bool { return get_(partner_rot_, row); }
    [[nodiscard]] auto answered(size_t row) const -> bool { return get_(answered_, row); }

    // One whole `rot` word, for a pass that walks the fold's `nz` words rather than single rows:
    // SilentIndex needs the complement of this over one word's anticommuting rows.
    [[nodiscard]] auto rot_word(size_t word) const -> uint64_t { return rot_[word]; }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t bytes = 0;
        for (const auto *bits : arrays_()) {
            bytes += bits->capacity() * sizeof(uint64_t);
        }
        return bytes;
    }

private:
    static auto set_(std::vector<uint64_t> &bits, size_t row) -> void { bits[row >> 6U] |= uint64_t{1} << (row & 63U); }
    static auto get_(const std::vector<uint64_t> &bits, size_t row) -> bool {
        return ((bits[row >> 6U] >> (row & 63U)) & 1U) != 0;
    }
    auto arrays_() -> std::array<std::vector<uint64_t> *, 5> {
        return {&rot_, &foll_, &received_, &partner_rot_, &answered_};
    }
    auto arrays_() const -> std::array<const std::vector<uint64_t> *, 5> {
        return {&rot_, &foll_, &received_, &partner_rot_, &answered_};
    }

    std::vector<uint64_t> rot_;
    std::vector<uint64_t> foll_;
    std::vector<uint64_t> received_;
    std::vector<uint64_t> partner_rot_;
    std::vector<uint64_t> answered_;
};

// Where a silent anticommuting row's pre-cos coefficient sits in the scan's `pre_cos` stream.
//
// The fused sweep scales EVERY anticommuting row while the scan still holds its value in a register, so
// by the time the join runs, `coeffs` no longer carries the pre-gate value a round-2 response owes the
// partner that rotated the row. The scan therefore streams that value to `GateScratch::pre_cos` as it
// goes -- ascending nz word, ascending bit within the word -- which makes a silent row's slot exactly its
// rank among the gate's silent rows. This recovers that rank in O(1) from a per-word prefix count and a
// popcount below the row's bit. Silent is exactly "anticommuting and not `rot`", and `rot` is final once
// the emit pass is done, which is when build() runs.
class SilentIndex {
public:
    // Fills the per-word silent masks and prefix counts; returns the gate's silent-row count, which is
    // how many values the scan streamed. Only the words `nz` names are written, and only they are read.
    auto build(std::span<const EvenParityNzWord> nz, const RowMarks &marks, size_t n_rows) -> size_t {
        const size_t words = (n_rows + 63) / 64;
        if (silent_.size() < words) {
            silent_.resize(words, 0);
            prefix_.resize(words, 0);
        }
        size_t total = 0;
        for (const auto &w : nz) {
            const size_t wi = w.base / 64;
            silent_[wi] = w.overlap & ~marks.rot_word(wi);
            prefix_[wi] = static_cast<uint32_t>(total);
            total += static_cast<size_t>(std::popcount(silent_[wi]));
        }
        return total;
    }

    //! Row `row`'s index in the pre-cos stream. Defined only for a silent anticommuting row of this gate.
    [[nodiscard]] auto ordinal(size_t row) const -> size_t {
        const size_t wi = row >> 6U;
        assert(((silent_[wi] >> (row & 63U)) & 1U) != 0 && "ordinal() of a row this gate did not leave silent");
        const uint64_t below = silent_[wi] & ((uint64_t{1} << (row & 63U)) - 1);
        return static_cast<size_t>(prefix_[wi]) + static_cast<size_t>(std::popcount(below));
    }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return (silent_.capacity() * sizeof(uint64_t)) + (prefix_.capacity() * sizeof(uint32_t));
    }

private:
    std::vector<uint64_t> silent_; // per row-word: anticommuting and not rot
    std::vector<uint32_t> prefix_; // per row-word: silent rows in the words before it, this gate
};

template <size_t NumModes>
struct GateScratch {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    BucketJoin<NumModes> join;        // this gate's anticommuting rows joined against its records
    RowMarks marks;                   // the protocol's per-row bits for this gate's anticommuting rows
    SilentIndex silent;               // silent row -> its slot in `pre_cos`; built once the emit pass is done
    std::vector<EvenParityNzWord> nz; // the fold's nonzero words: Anti(G), read by the emit pass and the join
    std::vector<PosT> partner;        // one partner's positions; sized 2*NumModes, the partner's upper bound
    std::vector<uint16_t> gen;        // the generator's ascending positions, the merge's second input
    // The pre-cos coefficient of each silent anticommuting row, in scan order: what a round-2 response
    // sends back. Only the fused sweep fills it; sized to the fold's tally under the same 4× release rule.
    DefaultInitVector<double> pre_cos;
    ExchangeCounters counters; // COMMPROF's per-call wire volume; reset by the caller, not per gate

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return join.memory_bytes() + marks.memory_bytes() + silent.memory_bytes()
               + (nz.capacity() * sizeof(EvenParityNzWord)) + (partner.capacity() * sizeof(PosT))
               + (gen.capacity() * sizeof(uint16_t)) + (pre_cos.capacity() * sizeof(double));
    }
};

} // namespace monoprop::detail
