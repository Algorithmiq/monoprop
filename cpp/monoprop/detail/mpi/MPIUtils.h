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
#include <format>
#include <stdexcept>
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

class RoutingDisagreement : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Every participant must resolve the SAME router, and the failure mode if they do not is a hang, not a
// wrong answer: linear routing makes each rank post receives from the peers its own bits imply, so a
// rank whose monoprop_ROUTING or _ROUTE_SEED did not reach it waits forever on a peer that never sends.
// One allreduce at construction turns that into an exception. Called once, never per gate.
inline auto check_routing_agreement(const mpi::Comm &comm) -> void {
    const auto router = router_for(comm);
    const size_t world = static_cast<size_t>(mpi::size(comm));
    if (world <= 1) {
        return;
    }
    const uint64_t mine =
        routing::mix64((static_cast<uint64_t>(router.linear_bits()) << 40) ^ routing::seed_from_env());
    const uint64_t total = mpi::allreduce_sum<uint64_t>(mine, comm);
    if (total != mine * static_cast<uint64_t>(world)) {
        throw RoutingDisagreement(
            std::format("routing configuration differs across the {} participants (this one: linear_bits={}, "
                        "seed={}). monoprop_ROUTING / monoprop_ROUTE_LINEAR_BITS / monoprop_ROUTE_SEED must "
                        "reach every rank identically -- under linear routing a disagreement deadlocks the "
                        "exchange rather than corrupting it.",
                        world,
                        router.linear_bits(),
                        routing::seed_from_env()));
    }
}

} // namespace monoprop
