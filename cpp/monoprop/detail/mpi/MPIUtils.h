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
// Must agree with Scan.h's query emission, so it takes a Router and has no rank-count overload.
template <size_t NumModes>
auto find_rank(const Monomial<NumModes> &mono, const routing::Router &router) -> size_t {
    return router.dest<NumModes>(mono);
}

// The router for this communicator's geometry and monoprop_ROUTING, bound to this monomial width.
template <size_t NumModes>
inline auto router_for(const mpi::Comm &comm) -> routing::Router {
    const auto geom = mpi::geometry(comm);
    return routing::make_router<NumModes>(static_cast<size_t>(geom.ranks), static_cast<size_t>(geom.partitions));
}

// Agree the transport now, so a misconfigured rank throws at construction instead of hanging its peers.
inline auto check_routing_agreement(const mpi::Comm &comm) -> void {
    static_cast<void>(mpi::routes_pairwise(comm));
}

} // namespace monoprop
