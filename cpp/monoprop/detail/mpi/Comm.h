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
#include <cstdint>
#include <vector>

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

// Which destination RANKS a round can touch, when the caller knows. Under GF(2)-linear routing
// (routing::Router) the low `bits` of the destination rank are determined by the generator: they are
// this rank's own low bits XOR `shift`, so the peers are
//
//     peer(k) = ((me & (2^bits - 1)) ^ shift) | (k << bits),   k in [0, ranks >> bits)
//
// -- `ranks >> bits` of them instead of all `ranks`, and the relation is symmetric (XOR is an
// involution), so every rank derives the same pairing with no communication. That is what lets a verb
// replace a dense collective with point-to-point over the peers it can actually reach.
//
// bits == 0 is the dense default: peer(k) == k and count == ranks, so the same loops walk every rank
// and the verbs take their collective path. A caller that gets `shift` wrong does not corrupt data --
// the counts for a non-peer are zero -- it deadlocks, which is why the plan is derived in one place.
struct PeerPlan {
    int bits = 0;
    int shift = 0;

    [[nodiscard]] constexpr auto dense() const -> bool { return bits == 0; }
    [[nodiscard]] constexpr auto count(int ranks) const -> int { return bits == 0 ? ranks : (ranks >> bits); }
    [[nodiscard]] constexpr auto peer(int me, int k) const -> int {
        if (bits == 0) {
            return k;
        }
        const int mask = (1 << bits) - 1;
        return ((me & mask) ^ shift) | (k << bits);
    }
};

// Argument bundles for the variable all-to-all verbs, deliberately here rather than in HybridComm.h:
// ShmComm.h takes the resolve bundle and compiles in non-MPI builds, so neither bundle may name an
// MPI-only type. MPI_Datatype therefore stays a separate parameter on the HybridComm verbs that need
// one (MPI-only header, MPI-only argument) instead of becoming a #ifdef-guarded member, which would
// make this installed header's layout depend on monoprop_ENABLE_MPI.
//
// `send`, `send_counts` and `send_displs` are non-owning views into caller memory. The send buffer must
// stay alive and unmodified until the verb's second barrier: peer partitions read it in place rather
// than through a copy, so freeing or mutating it earlier corrupts what they scatter.
//
// Counts and displacements are in ELEMENTS while `send` / `recv` are raw bytes, so every offset is
// scaled by `elem`; the pointers carry no element type to scale by.
struct AlltoallvArgs {
    const std::byte *send = nullptr;  // read in place by peers until the second barrier
    const int *send_counts = nullptr; // [P] elements sent to each destination
    const int *send_displs = nullptr; // [P] element offsets into `send`
    std::byte *recv = nullptr;        // caller-owned, already sized for recv_counts
    const int *recv_counts = nullptr; // [P] input: the senders' transpose, same contract as MPI_Alltoallv
    const int *recv_displs = nullptr; // [P] element offsets into `recv`
    size_t elem = 0;                  // bytes per element; scales every count/displ above
};

// Typed view of the same six spans, for callers that hold `T` buffers: `bytes()` type-erases it for the
// in-process transports, which address payloads as raw bytes, while the MPI path keeps the typed
// pointers it passes alongside a datatype. Same lifetime and element-offset contract as AlltoallvArgs.
template <typename T>
struct FlatAlltoallvArgs {
    const T *send = nullptr;
    const int *send_counts = nullptr; // [P] elements sent to each destination
    const int *send_displs = nullptr; // [P] element offsets into `send`
    T *recv = nullptr;                // caller-owned, already sized for recv_counts
    const int *recv_counts = nullptr; // [P] input: the senders' transpose, same contract as MPI_Alltoallv
    const int *recv_displs = nullptr; // [P] element offsets into `recv`

    auto bytes() const -> AlltoallvArgs {
        return AlltoallvArgs{.send = reinterpret_cast<const std::byte *>(send),
                             .send_counts = send_counts,
                             .send_displs = send_displs,
                             .recv = reinterpret_cast<std::byte *>(recv),
                             .recv_counts = recv_counts,
                             .recv_displs = recv_displs,
                             .elem = sizeof(T)};
    }
};

// The fused resolve verb keeps its own shape because its recv side is an output, not a caller-supplied
// layout: `recv` is a reference the callee resizes, and the two recv arrays are written, not read. Being
// typed, it derives element bytes as sizeof(T) instead of carrying `elem`. Send side: same lifetime and
// element-offset contract as AlltoallvArgs.
template <typename T>
struct AlltoallvResolveArgs {
    const T *send = nullptr;
    const int *send_counts = nullptr; // [P] elements sent to each destination
    const int *send_displs = nullptr; // [P] element offsets into `send`
    std::vector<T> &recv;             // output, resized to the resolved total; a reference, so copying the
                                      // bundle still resizes the caller's vector
    int *recv_counts = nullptr;       // [P] output: resolved per-source counts
    int *recv_displs = nullptr;       // [P] output: resolved element offsets into `recv`
};

} // namespace monoprop::mpi
