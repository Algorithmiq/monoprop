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

#include <cstdint>

#if defined(monoprop_ENABLE_MPI)
#include <mpi.h>
#else
// Fallback MPI types for non-MPI builds (single process).
using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr MPI_Comm MPI_COMM_SELF = 0;
#endif

namespace monoprop::mpi {

class ShmComm;
class HybridComm;

// Runtime-tagged communicator handle: the same SPMD code drives real MPI (`Kind::Mpi`) or an in-process
// ShmComm (`Kind::Shm`). Trivially copyable, passed by value. The implicit MPI_Comm constructor is
// deliberate; there is no implicit conversion back (that would silently drop a Shm handle), so read
// `.mpi` explicitly where a raw communicator is required.
struct Comm {
    // Hybrid = R MPI ranks x S in-process partitions presented as one flat P=R*S SPMD world (size()==P).
    enum class Kind : std::uint8_t { Mpi, Shm, Hybrid };
    Kind kind = Kind::Mpi;
    MPI_Comm mpi = MPI_COMM_SELF; // valid iff kind == Mpi
    ShmComm *shm = nullptr;       // non-owning (PartitionGroup owns); valid iff kind == Shm
    HybridComm *hyb = nullptr;    // non-owning (PartitionGroup owns); valid iff kind == Hybrid
    int shm_rank = 0;             // this participant's local partition index; valid iff kind == Shm | Hybrid

    constexpr Comm() = default;
    constexpr Comm(MPI_Comm c) : mpi(c) {} // implicit on purpose (see above)

    static auto make_shm(ShmComm *group, int rank) -> Comm {
        Comm c;
        c.kind = Kind::Shm;
        c.shm = group;
        c.shm_rank = rank;
        return c;
    }

    static auto make_hybrid(HybridComm *group, int local_partition) -> Comm {
        Comm c;
        c.kind = Kind::Hybrid;
        c.hyb = group;
        c.shm_rank = local_partition;
        return c;
    }
};

} // namespace monoprop::mpi
