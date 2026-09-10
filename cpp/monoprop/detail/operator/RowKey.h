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

// A term's 4-byte join key, folded from its positions or its dense form. The only place the operator
// store depends on routing, so OperatorIndex.h and TermTable.h do not include mpi/Routing.h themselves.

#include <cstddef>
#include <cstdint>

#include "monoprop/Bitset.h"
#include "monoprop/detail/mpi/Routing.h"

namespace monoprop::detail {

/*! @brief The join key of a 64-bit routing fingerprint: its top half, unmixed.
 *
 *  Taking the half rather than mixing it keeps the fingerprint's GF(2)-linearity -- join_tag(fp(S) ^
 *  fp(G)) == join_tag(fp(S)) ^ join_tag(fp(G)) -- which is what lets a receiver fold the same key off
 *  decoded positions that a sender folded off the partner it constructed. The top half also excludes
 *  the routing bits, which are the low ones, so the rows of one partition do not share a key prefix.
 */
[[nodiscard]] constexpr auto join_tag(uint64_t fp) noexcept -> uint32_t {
    return static_cast<uint32_t>(fp >> 32U);
}

//! The label table, bound through a local static so the per-term path does not re-enter routing's guard.
template <size_t NumBits>
[[nodiscard]] auto labels() noexcept -> const uint64_t * {
    static const uint64_t *const table = routing::linear_basis<NumBits>().data();
    return table;
}

//! The join key of a term given as ascending positions.
template <size_t NumBits, typename PosT>
[[nodiscard]] auto key_of_positions(const PosT *pos, size_t k) noexcept -> uint32_t {
    return join_tag(routing::fingerprint_positions(labels<NumBits>(), pos, k));
}

//! The join key of a dense term; the same map key_of_positions() applies to its positions.
template <size_t NumBits>
[[nodiscard]] auto key_of(const Bitset<NumBits> &bits) noexcept -> uint32_t {
    return join_tag(routing::linear_hash<NumBits>(bits));
}

} // namespace monoprop::detail
