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
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"

namespace monoprop::mpi_detail {

static_assert(sizeof(size_t) == sizeof(uint64_t), "MPI serialization assumes 64-bit size_t");

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

// Stateless and identical on every rank, so all ranks agree on a term's owner without communication.
// Both overloads go through routing::Router::dest and nothing else: this and Scan.h's query emission
// must return the same slot for the same monomial, and a divergence splits ownership silently.
template <size_t NumModes>
auto find_rank(const Monomial<NumModes> &mono, const routing::Router &router) -> size_t {
    return router.dest<NumModes>(mono);
}

// Flat-world overload, for callers that hold no geometry: the splitmix router (d = 0).
template <size_t NumModes>
auto find_rank(const Monomial<NumModes> &mono, const size_t n_ranks) -> size_t {
    if (n_ranks == 0) {
        return 0;
    }
    return routing::Router::splitmix(n_ranks).dest<NumModes>(mono);
}

// The router this communicator's geometry implies, honouring monoprop_ROUTING / _ROUTE_LINEAR_BITS.
inline auto router_for(const mpi::Comm &comm) -> routing::Router {
    const auto geom = mpi::geometry(comm);
    return routing::make_router(static_cast<size_t>(geom.ranks), static_cast<size_t>(geom.partitions));
}

} // namespace monoprop
