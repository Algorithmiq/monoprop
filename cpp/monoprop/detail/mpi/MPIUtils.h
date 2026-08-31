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

#include <cassert>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop::mpi_detail {

static_assert(sizeof(size_t) == sizeof(uint64_t), "MPI serialization assumes 64-bit size_t");

// The per-monomial wire width is the monomial's own word count, so a caller with no monomial in hand
// carries the count as a value.
inline auto append_monomial_words(const Bitset &mono, VecZ &buffer) -> void {
    const auto *src = mono.data();
    const size_t nw = mono.num_words();
    for (size_t i = 0; i < nw; ++i)
        buffer.push_back(src[i]);
}

// mono_out supplies the width: it is the destination, so it already knows how wide the record is, and
// reading into it avoids constructing a bitset per query on the resolve path.
inline auto read_monomial_from_words(const VecZ &buffer, size_t start, Bitset &mono_out) -> void {
    assert(mono_out.num_words() != 0 && "read_monomial_from_words needs a pre-sized destination");
    std::memcpy(mono_out.data(), &buffer[start], mono_out.num_words() * sizeof(uint64_t));
}

inline auto read_monomial_from_words(const VecZ &buffer, size_t start, size_t num_bits) -> Bitset {
    Bitset mono(num_bits);
    read_monomial_from_words(buffer, start, mono);
    return mono;
}

} // namespace monoprop::mpi_detail

namespace monoprop {

// Stateless and identical on every rank, so all ranks agree on a term's owner without communication.
// The ranks must also agree on the *width* they hash at: a monomial's storage width is part of its
// hash (SplitmixHash folds every word), so two ranks disagreeing about it would disagree about owners.
// Every rank derives it from the same propagator settings, so they do.
inline auto find_rank(const Bitset &mono, const size_t n_ranks) -> size_t {
    if (n_ranks == 0) {
        return 0;
    }
    return monomial_hash(mono) % n_ranks;
}

} // namespace monoprop
