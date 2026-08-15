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
#include <cassert>
#include <cstddef>

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/CompactQuery.h"

namespace monoprop::detail {

// The one interface every site that walks a query buffer is written against.
//
// This file used to dispatch between two record formats. It no longer does: CompactQuery is the only
// format, measured never slower than the dense record it replaced and up to 1.63x faster end to end,
// so the dense arm and the knob that selected it are gone. What survives is the part that was load
// bearing for a different reason -- a single place that derives every offset -- and that reason has
// outlived the dispatch.
//
// WHY THE INDIRECTION STAYS NOW THAT THERE IS NOTHING TO SELECT. The compact record is VARIABLE WIDTH:
// a term wider than six positions continues into further records, so the q'th query does not begin at
// q * <anything>. Every offset must come from a walk. Routing all of them through here is what makes a
// hardcoded stride impossible to write by accident, and that failure mode is not hypothetical on this
// branch -- it has shipped twice during this work, both times as a reader
// that indexed a buffer at a constant stride and passed silently until the layout moved under it.
// Inlining these back into their call sites would delete the guardrail, not the abstraction.

// How a query buffer is laid out. `fused` is a property of the BUFFER, not of the process: the scan's
// queries_r is always plain, and the fused query+value stream exists only in the send scratch and in
// what a ContractSink resolver receives. Passing it explicitly is what stopped the previous stride bug
// from being possible at all -- see ContractSink::kQuerierStride.
//
// It stays a struct rather than collapsing to a bare bool now that it holds one field: every signature
// below takes it positionally next to a size_t offset, and `next_off(buf, true, off)` reading as
// plausibly-correct-either-way is exactly the argument-order mistake a named field prevents.
struct QueryLayout {
    bool fused = false; // one value word follows each query
};

template <size_t NumModes>
struct QueryCodec {
    using CQ = CompactQuery<NumModes>;
    using PosT = typename CQ::PosT;

    // Words the QUERY at `off` occupies, NOT counting a trailing fused value word. Reads k out of the
    // record's own header, so a continuation is accounted for without the caller knowing it exists.
    [[nodiscard]] static auto query_words(const VecZ &buf, size_t off) -> size_t {
        return CQ::words_for(CQ::k_at(buf, off));
    }

    // Offset of the next query. `off` always names the START of a query; everything else is derived, so
    // no caller has to know whether a value word is present or how wide this record happened to be.
    [[nodiscard]] static auto next_off(const VecZ &buf, QueryLayout layout, size_t off) -> size_t {
        return off + query_words(buf, off) + (layout.fused ? 1U : 0U);
    }

    // Returns the WORDS written, which is NOT a constant the caller could have looked up: a wide term
    // continues into further records. Byte accounting that assumes a fixed width reports the inline
    // size whatever was actually written, which is how qbytes once reported 72 B a record for a
    // 16 B format.
    static auto push(VecZ &buf, const Monomial<NumModes> &mono, int phase) -> size_t {
        return CQ::push_mono(buf, mono, phase);
    }

    // Identical in both formats: the value is one bit_cast word after the query's words.
    static auto push_value(VecZ &buf, double v) -> void { buf.push_back(encode_value(v)); }

    // Inflates the record back into a dense Monomial. Still needed: the SELF-resolve path
    // (LayerBuildEngine::resolve_range_) keys deferred misses by Monomial, and read_positions is only
    // useful to a caller that can consume positions directly.
    static auto read_mono(const VecZ &buf, size_t off, Monomial<NumModes> &mono_out, int &phase_out) -> void {
        (void)CQ::read_mono(buf, off, mono_out, phase_out);
    }

    // The query's popcount, straight out of the record's header field.
    [[nodiscard]] static auto k_at(const VecZ &buf, size_t off) -> size_t { return CQ::k_at(buf, off); }

    // The query as its positions plus its phase, decoded into the CALLER's element type -- the store's
    // PosT is narrower than the wire's below 129 modes. This is the path that avoids materialising a
    // Monomial at all, and deleting that materialisation is where the receive side's 0.79x came from.
    template <typename OutT>
    static auto read_positions(const VecZ &buf, size_t off, OutT *out, int &phase_out) -> void {
        phase_out = CQ::phase_at(buf, off);
        (void)CQ::read_positions(buf, off, out);
    }

    [[nodiscard]] static auto phase_at(const VecZ &buf, size_t off) -> int { return CQ::phase_at(buf, off); }

    [[nodiscard]] static auto value_at(const VecZ &buf, QueryLayout layout, size_t off) -> double {
        assert(layout.fused && "there is no value word in a plain query buffer");
        (void)layout;
        return decode_value(buf[off + query_words(buf, off)]);
    }

    // The number of QUERIES. This is genuinely not a division: continuation records make
    // buf.size()/stride count RECORDS instead, which is what MPI wants for its element counts and not
    // what any caller here is asking for.
    [[nodiscard]] static auto count_queries(const VecZ &buf, QueryLayout layout) -> size_t {
        size_t off = 0;
        size_t n = 0;
        while (off < buf.size()) {
            off = next_off(buf, layout, off);
            ++n;
        }
        assert(off == buf.size() && "a compact query ran past the end of the buffer");
        return n;
    }

    // Interleave a plain query stream with its parallel v_src array into a fused one. Requires exactly
    // one value per query; the assert is the only thing standing between a mismatched pair of arrays and
    // a silently shifted coefficient stream, since both are just sizes.
    static auto build_fused(const VecZ &queries, const std::vector<double> &vals, VecZ &out) -> void {
        out.clear();
        out.reserve(queries.size() + vals.size());
        size_t off = 0;
        size_t i = 0;
        while (off < queries.size()) {
            const size_t n = query_words(queries, off);
            out.insert(out.end(),
                       queries.begin() + static_cast<std::ptrdiff_t>(off),
                       queries.begin() + static_cast<std::ptrdiff_t>(off + n));
            assert(i < vals.size() && "fused build needs exactly one value per query");
            out.push_back(encode_value(vals[i]));
            off += n;
            ++i;
        }
        assert(i == vals.size() && "fused build needs exactly one value per query");
    }

    // Copy the query at `src_off` (and its value word, if fused) to `dst_off` within the same buffer.
    // Used by the follower compaction, which drops queries a leader already matched. Returns the number
    // of words written, so the caller advances both cursors without recomputing the width.
    static auto move_query(VecZ &buf, QueryLayout layout, size_t src_off, size_t dst_off) -> size_t {
        const size_t n = query_words(buf, src_off) + (layout.fused ? 1U : 0U);
        if (src_off != dst_off) {
            assert(dst_off < src_off && "compaction only ever moves a query earlier");
            std::copy(buf.begin() + static_cast<std::ptrdiff_t>(src_off),
                      buf.begin() + static_cast<std::ptrdiff_t>(src_off + n),
                      buf.begin() + static_cast<std::ptrdiff_t>(dst_off));
        }
        return n;
    }
};

} // namespace monoprop::detail
