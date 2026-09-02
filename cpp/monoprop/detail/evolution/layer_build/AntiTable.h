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

// The per-gate partner index. If M anticommutes with G then so does M ^ G (Majorana: |M'||G| - |M' ∩ G| ≡
// |M||G| - |M ∩ G| mod 2 using |G|² ≡ |G|; Pauli: ω(M ^ G, G) = ω(M, G)), so the partner of ANY term the
// gate touches, if it is tracked at all, is one of the terms the fold already returned. Membership and
// index of a partner are therefore decided inside Anti(G) -- a set of ~K/N rows the scan has in hand --
// and the operator needs no persistent hash table. This is that set as an open-addressing table keyed by
// the routing fingerprint (Routing.h), built once per gate over EVERY row the fold marks (not only the
// emitted ones: a term below the coefficient threshold, above the rotation length cap, or with a zero
// coefficient is still somebody's partner) and discarded when the gate is done.
//
// The fingerprint is GF(2)-linear and not injective, so two things are load-bearing: the probe key is the
// MIXED fingerprint (a raw linear hash fed to linear probing is only 2-independent, and pairs differing by
// the constant fp(G) would cluster), and every key match is confirmed against the row's positions -- a
// false match would silently merge two distinct terms.

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

template <size_t NumModes>
struct AntiTable {
    using PosT = typename OperatorIndex<NumModes>::PosT;
    using Ordinal = uint32_t;
    static constexpr Ordinal kNone = std::numeric_limits<Ordinal>::max();

    // Sizes the table for `n_anti` rows and clears every per-ordinal bit. Capacity is retained across
    // gates, but a table more than four times too large for this gate is released first: one gate with a
    // very large anticommuting set (a single-qubit Pauli generator) must not pin its footprint for the
    // whole run.
    auto begin(size_t n_anti) -> void {
        const size_t slots = std::bit_ceil(std::max<size_t>(kMinSlots, (n_anti * 10 / 7) + 1));
        const size_t words = (n_anti + 63) / 64;
        if (ords_.capacity() > 4 * slots) {
            ords_ = std::vector<Ordinal>{};
            keys_ = std::vector<uint64_t>{};
            rows_ = std::vector<TermIndex>{};
            phase_ = std::vector<int8_t>{};
            for (auto *b : bit_arrays_()) {
                *b = std::vector<uint64_t>{};
            }
        }
        rows_.clear();
        ords_.assign(slots, kNone);
        keys_.resize(slots);
        mask_ = slots - 1;
        phase_.resize(n_anti);
        for (auto *b : bit_arrays_()) {
            b->assign(words, 0);
        }
        n_words_ = words;
    }

    // Rows must arrive in strictly ascending order (the fold walks them that way): ordinal_of_row relies
    // on it, and so does the graph sink, whose out lists must be in ascending row order. Returns the row's
    // ordinal.
    auto add(TermIndex row, uint64_t fp) -> Ordinal {
        assert(rows_.empty() || rows_.back() < row);
        const auto ord = static_cast<Ordinal>(rows_.size());
        rows_.push_back(row);
        const uint64_t key = routing::mix64(fp);
        size_t s = static_cast<size_t>(key) & mask_;
        while (ords_[s] != kNone) {
            s = (s + 1) & mask_;
        }
        ords_[s] = ord;
        keys_[s] = key;
        // The per-ordinal arrays were sized from begin(n_anti); a caller that under-counted grows them here.
        if (rows_.size() > phase_.size()) {
            phase_.push_back(0);
        }
        if ((rows_.size() + 63) / 64 > n_words_) {
            for (auto *b : bit_arrays_()) {
                b->push_back(0);
            }
            ++n_words_;
        }
        return ord;
    }

    // The ordinal of the tracked term whose positions are exactly `pos`, or kNone. `fp` must be that
    // term's fingerprint (the caller has it as fp(source) ^ fp(G), or recomputes it off the wire).
    [[nodiscard]] auto probe(const OperatorIndex<NumModes> &store, uint64_t fp, std::span<const PosT> pos) const
        -> Ordinal {
        const uint64_t key = routing::mix64(fp);
        for (size_t s = static_cast<size_t>(key) & mask_;; s = (s + 1) & mask_) {
            const Ordinal o = ords_[s];
            if (o == kNone) {
                return kNone;
            }
            if (keys_[s] == key && store.row_eq_positions(static_cast<size_t>(rows_[o]), pos)) {
                return o;
            }
        }
    }

    [[nodiscard]] auto size() const -> size_t { return rows_.size(); }
    [[nodiscard]] auto row_of(Ordinal ord) const -> size_t { return static_cast<size_t>(rows_[ord]); }
    [[nodiscard]] auto rows() const -> std::span<const TermIndex> { return rows_; }

    // The ordinal of a row known to be in the table (the scan's own anticommuting rows), by binary search
    // over the ascending row list. kNone if the row is not in the table.
    [[nodiscard]] auto ordinal_of_row(size_t row) const -> Ordinal {
        const auto it = std::ranges::lower_bound(rows_, static_cast<TermIndex>(row));
        if (it == rows_.end() || static_cast<size_t>(*it) != row) {
            return kNone;
        }
        return static_cast<Ordinal>(it - rows_.begin());
    }

    // The protocol's per-ordinal state (Engine.h), all cleared by begin(). The scan sets `rot` (= E(ν),
    // the term's own emission predicate), `foll` (the pivot bit: which endpoint of a pair it is) and
    // `phase` (φ_ν, meaningful only when a record was sent); the join sets `received` (the partner's
    // record arrived, i.e. the partner is tracked) and `partner_rot` (that record's rot bit).
    auto set_rot(Ordinal ord) -> void { set_(rot_, ord); }
    auto set_foll(Ordinal ord) -> void { set_(foll_, ord); }
    auto set_received(Ordinal ord) -> void { set_(received_, ord); }
    auto set_partner_rot(Ordinal ord) -> void { set_(partner_rot_, ord); }
    auto set_phase(Ordinal ord, int phase) -> void { phase_[ord] = static_cast<int8_t>(phase); }
    [[nodiscard]] auto rot(Ordinal ord) const -> bool { return get_(rot_, ord); }
    [[nodiscard]] auto foll(Ordinal ord) const -> bool { return get_(foll_, ord); }
    [[nodiscard]] auto received(Ordinal ord) const -> bool { return get_(received_, ord); }
    [[nodiscard]] auto partner_rot(Ordinal ord) const -> bool { return get_(partner_rot_, ord); }
    [[nodiscard]] auto phase(Ordinal ord) const -> int { return static_cast<int>(phase_[ord]); }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t bits = 0;
        for (const auto *b : bit_arrays_()) {
            bits += b->capacity() * sizeof(uint64_t);
        }
        return (rows_.capacity() * sizeof(TermIndex)) + (ords_.capacity() * sizeof(Ordinal))
               + (keys_.capacity() * sizeof(uint64_t)) + (phase_.capacity() * sizeof(int8_t)) + bits;
    }

private:
    static constexpr size_t kMinSlots = 16;

    static auto set_(std::vector<uint64_t> &bits, Ordinal ord) -> void { bits[ord >> 6] |= uint64_t{1} << (ord & 63U); }
    static auto get_(const std::vector<uint64_t> &bits, Ordinal ord) -> bool {
        return ((bits[ord >> 6] >> (ord & 63U)) & 1U) != 0;
    }
    auto bit_arrays_() -> std::array<std::vector<uint64_t> *, 4> { return {&rot_, &foll_, &received_, &partner_rot_}; }
    auto bit_arrays_() const -> std::array<const std::vector<uint64_t> *, 4> {
        return {&rot_, &foll_, &received_, &partner_rot_};
    }

    std::vector<TermIndex> rows_; // ordinal -> row, strictly ascending
    std::vector<Ordinal> ords_;   // slot -> ordinal, kNone when empty
    std::vector<uint64_t> keys_;  // slot -> mixed fingerprint, valid iff the slot is occupied
    std::vector<int8_t> phase_;   // ordinal -> φ, ternary
    std::vector<uint64_t> rot_;   // ordinal bits, one array each
    std::vector<uint64_t> foll_;
    std::vector<uint64_t> received_;
    std::vector<uint64_t> partner_rot_;
    size_t mask_ = 0;
    size_t n_words_ = 0;
};

} // namespace monoprop::detail
