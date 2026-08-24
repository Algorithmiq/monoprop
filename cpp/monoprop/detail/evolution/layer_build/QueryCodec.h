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
#include "monoprop/detail/evolution/layer_build/SparseQuery.h"

namespace monoprop::detail {

// The one interface every site that walks a query buffer is written against: the record is variable
// width, so no caller may hold a stride.

// `fused` is a property of the BUFFER, not the process: queries_r is always plain, while the send
// scratch and what a ContractSink resolver receives are fused. A named field, not a bare bool, so
// `next_off(buf, true, off)` cannot read as plausibly-correct-either-way.
struct QueryLayout {
    bool fused = false; // one value word follows each query
};

// One alias, so the tests and the codec name the same record type.
template <size_t NumModes>
using QueryRecord = SparseQuery<NumModes>;

template <size_t NumModes>
struct QueryCodec {
    using CQ = QueryRecord<NumModes>;
    using PosT = typename CQ::PosT;

    // Words the QUERY at `off` occupies, NOT counting a trailing fused value word. Asked of the record
    // rather than derived from k, which does not determine the width once gw can vary.
    [[nodiscard]] static auto query_words(const VecZ &buf, size_t off) -> size_t { return CQ::words_at(buf, off); }

    // Complete mode pairs among ascending positions, exposed so no caller names a concrete record type.
    template <typename OutT>
    [[nodiscard]] static auto pair_count(const OutT *pos, size_t k) noexcept -> size_t {
        return CQ::pair_count(pos, k);
    }

    // Reserve hints, not correctness: sized from the measured mean of 5.33 positions per query.
    static constexpr size_t kReservePositionsPerQuery = 6;
    static constexpr size_t kReserveWordsPerQuery = 2;

    // Offset of the next query; `off` always names the START of one, and the rest is derived.
    [[nodiscard]] static auto next_off(const VecZ &buf, QueryLayout layout, size_t off) -> size_t {
        return off + query_words(buf, off) + (layout.fused ? 1U : 0U);
    }

    // Returns the WORDS written, which is not a constant: byte accounting must not assume a width.
    static auto push(VecZ &buf, const Monomial<NumModes> &mono, int phase) -> size_t {
        return CQ::push_mono(buf, mono, phase);
    }

    // From ascending positions, which is what the partner merge hands the emit site; the dense overload
    // above is for callers that hold only a bitset.
    template <typename PosU>
    static auto push_positions(VecZ &buf, const PosU *pos, size_t k, int phase) -> size_t {
        return CQ::push(buf, pos, k, phase);
    }

    // Identical in both formats: the value is one bit_cast word after the query's words.
    static auto push_value(VecZ &buf, double v) -> void { buf.push_back(encode_value(v)); }

    // Inflates the record back into a dense Monomial, for callers that cannot consume positions.
    static auto read_mono(const VecZ &buf, size_t off, Monomial<NumModes> &mono_out, int &phase_out) -> void {
        (void)CQ::read_mono(buf, off, mono_out, phase_out);
    }

    // The query's popcount, straight out of the record's header field.
    [[nodiscard]] static auto k_at(const VecZ &buf, size_t off) -> size_t { return CQ::k_at(buf, off); }

    // Positions plus phase, into the CALLER's element type: the store's PosT is narrower below 129 modes.
    template <typename OutT>
    static auto read_positions(const VecZ &buf, QueryLayout layout, size_t off, OutT *out, int &phase_out) -> size_t {
        phase_out = CQ::phase_at(buf, off);
        return CQ::read_positions(buf, off, out) + (layout.fused ? 1U : 0U);
    }

    [[nodiscard]] static auto phase_at(const VecZ &buf, size_t off) -> int { return CQ::phase_at(buf, off); }

    [[nodiscard]] static auto value_at(const VecZ &buf, [[maybe_unused]] QueryLayout layout, size_t off) -> double {
        assert(layout.fused && "there is no value word in a plain query buffer");
        return decode_value(buf[off + query_words(buf, off)]);
    }

    // The number of QUERIES. Genuinely a walk: records vary in width, so there is no stride to divide by.
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

    // Interleave a plain query stream with its parallel v_src array; a size mismatch shifts every coeff.
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

    // Copy the query at `src_off` (with its value word, if fused) to `dst_off`; returns words written.
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
