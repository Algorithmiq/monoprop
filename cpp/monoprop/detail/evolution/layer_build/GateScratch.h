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
    auto set_received(size_t row) -> void { set_(received_, row); }
    auto set_partner_rot(size_t row) -> void { set_(partner_rot_, row); }
    auto set_answered(size_t row) -> void { set_(answered_, row); }
    [[nodiscard]] auto rot(size_t row) const -> bool { return get_(rot_, row); }
    [[nodiscard]] auto foll(size_t row) const -> bool { return get_(foll_, row); }
    [[nodiscard]] auto received(size_t row) const -> bool { return get_(received_, row); }
    [[nodiscard]] auto partner_rot(size_t row) const -> bool { return get_(partner_rot_, row); }
    [[nodiscard]] auto answered(size_t row) const -> bool { return get_(answered_, row); }

    // One whole `rot` word, for a pass that walks the fold's `nz` words rather than single rows: the
    // deferred cos sweep (Scan.h) needs the complement of this over one word's anticommuting rows.
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

template <size_t NumModes>
struct GateScratch {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    BucketJoin<NumModes> join;        // this gate's anticommuting rows joined against its records
    RowMarks marks;                   // the protocol's per-row bits for this gate's anticommuting rows
    std::vector<EvenParityNzWord> nz; // the fold's nonzero words: Anti(G), read by the emit pass and the join
    std::vector<PosT> partner;        // one partner's positions; sized 2*NumModes, the partner's upper bound
    std::vector<uint16_t> gen;        // the generator's ascending positions, the merge's second input
    ExchangeCounters counters;        // COMMPROF's per-call wire volume; reset by the caller, not per gate
    // Widest instant of the per-gate transient buffers over the call, in bytes. Those buffers die with
    // the gate, so no resting field can name them; reset by the caller alongside `counters`.
    size_t buffers_hwm_bytes{0uz};

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return join.memory_bytes() + marks.memory_bytes() + (nz.capacity() * sizeof(EvenParityNzWord))
               + (partner.capacity() * sizeof(PosT)) + (gen.capacity() * sizeof(uint16_t));
    }
};

} // namespace monoprop::detail
