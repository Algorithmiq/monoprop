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
// Fallback MPI types for non-MPI builds (single process). The engine names these but the calls are
// guarded by rank-count and dispatched through the wrappers below, so they never execute.
using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr MPI_Comm MPI_COMM_SELF = 0;
#endif

namespace monoprop::mpi {

class ShmComm;    // defined in ShmComm.h — the in-process shared-memory SPMD transport.
class HybridComm; // defined in HybridComm.h — composes R MPI ranks x S shards into one flat world.

/**
 * @brief Runtime-tagged communicator handle threaded through the whole engine in place of raw
 * MPI_Comm. The SAME SPMD code drives either real MPI (`Kind::Mpi`, across nodes) or an in-process
 * ShmComm (`Kind::Shm`, across shards pinned to cores on one node). Trivially copyable and passed by
 * value, exactly like the MPI_Comm it replaces.
 *
 * The implicit MPI_Comm constructor is deliberate: it keeps every existing call site, test, and the
 * Python binding (which hands over an MPI_Comm) compiling and behaving unchanged — a plain
 * `MPI_COMM_WORLD` / `MPI_COMM_SELF` / mpi4py comm becomes a `Kind::Mpi` handle. There is no implicit
 * conversion back to MPI_Comm (that would silently drop a Shm handle); read `.mpi` explicitly where a
 * raw communicator is genuinely required (e.g. the public `comm()` accessor, mpi4py interop).
 */
struct Comm {
    // Hybrid = R MPI ranks x S in-process shards presented as one flat P=R*S SPMD world; the engine
    // sees size()==P and never distinguishes it from plain MPI or plain shards.
    enum class Kind : std::uint8_t { Mpi, Shm, Hybrid };
    Kind kind = Kind::Mpi;
    MPI_Comm mpi = MPI_COMM_SELF; // valid iff kind == Mpi
    ShmComm *shm = nullptr;       // non-owning (ShardGroup owns); valid iff kind == Shm
    HybridComm *hyb = nullptr;    // non-owning (ShardGroup owns); valid iff kind == Hybrid
    int shm_rank = 0;             // this participant's LOCAL shard index; valid iff kind == Shm | Hybrid

    constexpr Comm() = default;
    constexpr Comm(MPI_Comm c) : mpi(c) {} // implicit on purpose (see above)

    static auto make_shm(ShmComm *group, int rank) -> Comm {
        Comm c;
        c.kind = Kind::Shm;
        c.shm = group;
        c.shm_rank = rank;
        return c;
    }

    static auto make_hybrid(HybridComm *group, int local_shard) -> Comm {
        Comm c;
        c.kind = Kind::Hybrid;
        c.hyb = group;
        c.shm_rank = local_shard;
        return c;
    }
};

} // namespace monoprop::mpi
