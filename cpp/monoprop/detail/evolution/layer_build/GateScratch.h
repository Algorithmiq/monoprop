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
#include "monoprop/detail/mpi/Comm.h"
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
    // Sizes the bitsets to `n_rows`, rebinds the five bases, and clears the words `nz` names. Must run
    // after the fold's pass 1 (which produces `nz`) and before the emit pass sets a bit -- and it is the
    // only place the vectors can move, which is what makes the bases good for the whole gate.
    auto begin(size_t n_rows, std::span<const EvenParityNzWord> nz) -> void {
        const size_t words = (n_rows + 63) / 64;
        for (auto *bits : arrays_()) {
            if (bits->size() < words) {
                bits->resize(words, 0);
            }
        }
        bind_();
        // The five bases in registers for the whole clear: `arrays_()` used to rebuild its pointer array
        // inside this loop, once per anticommuting word.
        uint64_t *const rot = rot_;
        uint64_t *const foll = foll_;
        uint64_t *const received = received_;
        uint64_t *const partner_rot = partner_rot_;
        uint64_t *const answered = answered_;
        for (const auto &w : nz) {
            const size_t wi = w.base / 64;
            rot[wi] = 0;
            foll[wi] = 0;
            received[wi] = 0;
            partner_rot[wi] = 0;
            answered[wi] = 0;
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
    static auto set_(uint64_t *bits, size_t row) -> void { bits[row >> 6U] |= uint64_t{1} << (row & 63U); }
    static auto get_(const uint64_t *bits, size_t row) -> bool { return ((bits[row >> 6U] >> (row & 63U)) & 1U) != 0; }
    auto bind_() -> void {
        rot_ = rot_words_.data();
        foll_ = foll_words_.data();
        received_ = received_words_.data();
        partner_rot_ = partner_rot_words_.data();
        answered_ = answered_words_.data();
    }
    auto arrays_() -> std::array<std::vector<uint64_t> *, 5> {
        return {&rot_words_, &foll_words_, &received_words_, &partner_rot_words_, &answered_words_};
    }
    auto arrays_() const -> std::array<const std::vector<uint64_t> *, 5> {
        return {&rot_words_, &foll_words_, &received_words_, &partner_rot_words_, &answered_words_};
    }

    std::vector<uint64_t> rot_words_;
    std::vector<uint64_t> foll_words_;
    std::vector<uint64_t> received_words_;
    std::vector<uint64_t> partner_rot_words_;
    std::vector<uint64_t> answered_words_;
    // The bitsets' bases, rebound by begin() and valid for the gate it opened. A single load off `this`
    // instead of a load of the owning vector's data pointer, on every mark this gate sets or reads;
    // null until the first begin(), which every path that touches a mark runs first. A copy of a
    // GateScratch would carry the source's bases, so the propagator's copy constructor default-builds
    // its scratch rather than copying it (MonomialPropagator.inl).
    uint64_t *rot_ = nullptr;
    uint64_t *foll_ = nullptr;
    uint64_t *received_ = nullptr;
    uint64_t *partner_rot_ = nullptr;
    uint64_t *answered_ = nullptr;
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
        // Both bases out of the loop for the same reason RowMarks holds its own: the two stores are
        // otherwise assumed to reach the vectors' own data pointers.
        uint64_t *const silent = silent_.data();
        uint32_t *const prefix = prefix_.data();
        size_t total = 0;
        for (const auto &w : nz) {
            const size_t wi = w.base / 64;
            const uint64_t word = w.overlap & ~marks.rot_word(wi);
            silent[wi] = word;
            prefix[wi] = static_cast<uint32_t>(total);
            total += static_cast<size_t>(std::popcount(word));
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

namespace gate_scratch_detail {
inline auto wire_slot_bytes(const mpi::WindowVec<VecZ> &wire) -> size_t {
    size_t bytes = wire.size() * sizeof(VecZ);
    for (const VecZ &slot : wire) {
        bytes += slot.capacity() * sizeof(size_t);
    }
    return bytes;
}
} // namespace gate_scratch_detail

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
    // How many records the previous gate of this partition staged for itself: what the next gate's
    // self-slot reserve is sized from (Scan.h). A hint only -- wrong in either direction it costs at most
    // a few pushes their geometric grow -- which is why it may cross gates although nothing else here does.
    size_t self_records_hint = 0;
    // The send buffers pair_exchange's lifetime rule needs. Peers read a published buffer IN PLACE,
    // and the first moment every peer is proved done with it is the return of this partition's next
    // call, so a gate's records must outlive the gate -- the second reason (with `self_records_hint`)
    // something here crosses a gate boundary.
    //
    // ONE buffer per round, not a shared pool alternated per call: sharing lets a round-2 staging
    // inherit the capacity a round-1 gate left in the same slot (`reset` clears a slot but keeps its
    // storage), which made the response buffer as wide as the query buffer and cost 4.4 GiB at the
    // 8x16 ladder rung. Two query buffers because a graph-sink gate makes ONE call, so its queries are
    // still being read while the next gate's scan is already writing; the response buffer needs no
    // twin, since only the two-call fused sink stages responses at all.
    std::array<mpi::WindowVec<VecZ>, 2> wire_q;
    mpi::WindowVec<VecZ> wire_r;
    size_t wire_gate = 0; // parity of `wire_q`; bumped once per gate that exchanges
    // `wire_spans` is the outer descriptor array, which the verb copies before its barrier and the
    // caller may reuse on return.
    std::vector<std::span<const size_t>> wire_spans;
    // Slot views of an alltoallv result, so the collective path reaches the same decode surface.
    std::vector<std::span<const size_t>> slot_views;
    ExchangeCounters counters; // COMMPROF's per-call wire volume; reset by the caller, not per gate
    // Widest instant of the gate's per-gate buffers over the call, in bytes. Those buffers die with the
    // gate or are resized under it, so no resting field can name their peak; reset by the caller
    // alongside `counters`.
    size_t buffers_hwm_bytes{0uz};

    //! This gate's round-1 buffer. The scan takes it, the engine puts it back and bumps the parity.
    [[nodiscard]] auto wire_queries() -> mpi::WindowVec<VecZ> & { return wire_q[wire_gate & 1U]; }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return join.memory_bytes() + marks.memory_bytes() + silent.memory_bytes()
               + (nz.capacity() * sizeof(EvenParityNzWord)) + (partner.capacity() * sizeof(PosT))
               + (gen.capacity() * sizeof(uint16_t)) + (pre_cos.capacity() * sizeof(double)) + wire_bytes();
    }

    //! The wire buffers, which outlive their gate and so are NOT covered by the gate-buffer stamp.
    [[nodiscard]] auto wire_bytes() const -> size_t {
        using gate_scratch_detail::wire_slot_bytes;
        return wire_slot_bytes(wire_q[0]) + wire_slot_bytes(wire_q[1]) + wire_slot_bytes(wire_r);
    }
};

} // namespace monoprop::detail
