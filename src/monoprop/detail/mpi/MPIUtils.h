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

#include <cstddef>
#include <cstring>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop::mpi_detail {

static_assert(sizeof(size_t) == sizeof(uint64_t), "MPI serialization assumes 64-bit size_t");

/// Number of size_t words per Monomial<NumModes>.
template <size_t NumModes>
inline constexpr size_t kWords = Monomial<NumModes>::num_words();

template <size_t NumModes>
inline auto append_monomial_words(const Monomial<NumModes> &mono, VecZ &buffer) -> void {
    const auto *src = mono.data();
    for (size_t i = 0; i < kWords<NumModes>; ++i)
        buffer.push_back(src[i]);
}

template <size_t NumModes>
inline auto read_monomial_from_words(const VecZ &buffer, size_t start) -> Monomial<NumModes> {
    Monomial<NumModes> mono;
    std::memcpy(mono.data(), &buffer[start], kWords<NumModes> * sizeof(uint64_t));
    return mono;
}

} // namespace monoprop::mpi_detail

namespace monoprop {

// Deterministic owner rank for a term: hash(mono) % n_ranks. Stateless and identical on every rank,
// so all ranks agree on which rank owns any given term without communication.
template <size_t NumModes>
auto find_rank(const Monomial<NumModes> &mono, const size_t n_ranks) -> size_t {
    if (n_ranks == 0) {
        return 0;
    }
    return monomial_hash<NumModes>(mono) % n_ranks;
}

} // namespace monoprop
