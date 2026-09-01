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

// The retired dense query record, frozen here as a deliberately independent oracle for
// sparse_query_tests.cpp's differential: test-only, and it does not follow the wire format.

#pragma once

#include <bit>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::test_ref {

using monoprop::VecZ;

// W monomial words + one ±1 phase word.
template <size_t NumModes>
inline constexpr size_t kQueryWords = mpi_detail::kWords<NumModes> + 1;

template <size_t NumModes>
inline constexpr size_t kQueryWordsFused = kQueryWords<NumModes> + 1;

template <size_t NumModes>
inline auto query_push(VecZ &buf, const Monomial<NumModes> &mono, int phase) -> void {
    mpi_detail::append_monomial_words<NumModes>(mono, buf);
    buf.push_back(static_cast<size_t>(static_cast<unsigned int>(phase)));
}

template <size_t NumModes, size_t QW = kQueryWords<NumModes>>
inline auto query_read(const VecZ &buf, size_t q, Monomial<NumModes> &mono_out, int &phase_out) -> void {
    const size_t base = q * QW;
    mono_out = mpi_detail::read_monomial_from_words<NumModes>(buf, base);
    phase_out = static_cast<int>(static_cast<unsigned int>(buf[base + mpi_detail::kWords<NumModes>]));
}

template <size_t NumModes>
inline auto query_value(const VecZ &buf, size_t q) -> double {
    return detail::decode_value(buf[q * kQueryWordsFused<NumModes> + mpi_detail::kWords<NumModes> + 1]);
}

// Requires v.size() == q.size()/kQueryWords: exactly one value per query record.
template <size_t NumModes>
inline auto build_fused_query_value(const VecZ &q, const std::vector<double> &v, VecZ &out) -> void {
    constexpr size_t W = kQueryWords<NumModes>;
    const size_t nq = q.empty() ? 0 : q.size() / W;
    out.clear();
    out.reserve(nq * kQueryWordsFused<NumModes>);
    for (size_t i = 0; i < nq; ++i) {
        out.insert(out.end(),
                   q.begin() + static_cast<std::ptrdiff_t>(i * W),
                   q.begin() + static_cast<std::ptrdiff_t>((i + 1) * W));
        out.push_back(detail::encode_value(v[i]));
    }
}

// Every field a wire header carries must be recoverable from the record's own positions, or it rots
// unnoticed. Checks the phase in range, k against the positions handed back, the positions strictly
// ascending and in bounds, and gw exactly the widest gap. Recomputed here rather than in the library,
// so it runs in the Release build the suite actually uses.
template <size_t NumModes, std::ranges::contiguous_range Pos>
[[nodiscard]] inline auto wire_header_is_consistent(std::span<const size_t> buf, size_t off, const Pos &pos) -> bool {
    const auto h = detail::QueryWire<NumModes>::header_at(buf, off);
    if (h.phase < -1 || h.phase > 1) {
        return false;
    }
    if (h.k != std::ranges::size(pos)) {
        return false;
    }
    for (size_t j = 0; j + 1 < h.k; ++j) {
        if (static_cast<size_t>(pos[j]) >= static_cast<size_t>(pos[j + 1])) {
            return false; // positions must arrive strictly ascending
        }
    }
    if (h.k != 0 && static_cast<size_t>(pos[h.k - 1]) >= 2 * NumModes) {
        return false;
    }
    // gw is the maximum gap width: too small truncates a gap silently, too large wastes bits.
    size_t g = 0;
    for (size_t j = 1; j < h.k; ++j) {
        const auto b = static_cast<size_t>(std::bit_width(static_cast<size_t>(pos[j] - pos[j - 1] - 1)));
        g = (b > g) ? b : g;
    }
    return g == h.gw;
}

} // namespace monoprop::test_ref
