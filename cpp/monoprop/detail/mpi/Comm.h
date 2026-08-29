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

#include <cassert>
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

// A window-relative index. Distinct from a flat slot on purpose: the two are the same number only when
// the window starts at 0, so a swap addresses the wrong peer while staying in bounds.
struct WindowIndex {
    size_t value = 0;

    constexpr WindowIndex() = default;
    explicit constexpr WindowIndex(size_t v) noexcept : value(v) {}
};

// The contiguous run of flat destination slots a round can reach. Slots are rank-major
// (slot = rank * S + partition), so one rank's S partitions are contiguous and the single peer sparse
// routing leaves is exactly one such run; dense is the count == P value of the same run, not a second
// shape. See PeerPlan::window.
struct SlotWindow {
    size_t base = 0;  // first reachable flat slot
    size_t count = 0; // slots in the run

    [[nodiscard]] constexpr auto stop() const -> size_t { return base + count; }
    [[nodiscard]] constexpr auto contains(size_t slot) const -> bool { return slot >= base && slot < stop(); }
    // The one flat-slot door: it asserts membership, so a slot from outside cannot become another's entry.
    [[nodiscard]] constexpr auto index(size_t slot) const -> WindowIndex {
        assert(contains(slot) && "flat slot outside the window it is being re-based into");
        return WindowIndex{slot - base};
    }
    [[nodiscard]] constexpr auto slot(WindowIndex i) const -> size_t {
        assert(i.value < count);
        return base + i.value;
    }
};

// A vector over a SlotWindow, addressed by flat slot through at_slot(); operator[] takes a WindowIndex,
// so a flat slot used as a raw index does not compile. Re-basing an array is only safe if every index
// site shifts together, and these two accessors are the only sites.
template <typename T>
class WindowVec {
public:
    using value_type = T;

    WindowVec() = default;
    explicit WindowVec(SlotWindow w) : win_(w), v_(w.count) {}

    auto reset(SlotWindow w) -> void {
        win_ = w;
        v_.assign(w.count, T{});
    }

    [[nodiscard]] auto window() const -> SlotWindow { return win_; }
    [[nodiscard]] auto size() const -> size_t { return v_.size(); }

    [[nodiscard]] auto operator[](WindowIndex i) -> T & { return v_[i.value]; }
    [[nodiscard]] auto operator[](WindowIndex i) const -> const T & { return v_[i.value]; }
    [[nodiscard]] auto at_slot(size_t slot) -> T & { return v_[win_.index(slot).value]; }
    [[nodiscard]] auto at_slot(size_t slot) const -> const T & { return v_[win_.index(slot).value]; }

    [[nodiscard]] auto begin() { return v_.begin(); }
    [[nodiscard]] auto end() { return v_.end(); }
    [[nodiscard]] auto begin() const { return v_.begin(); }
    [[nodiscard]] auto end() const { return v_.end(); }

private:
    SlotWindow win_{};
    std::vector<T> v_;
};

// Which destination RANKS a round can touch, when the caller knows. Two states, matching
// routing::Router: dense, or sparse over the single peer GF(2)-linear routing implies.
//
// Sparse means the destination rank of every block is determined by the generator: it is this rank's
// own index XOR `shift`, so
//
//     peer = me ^ shift,   count == 1
//
// -- one peer instead of all `ranks`, and the relation is symmetric (XOR is an involution), so every
// rank derives the same pairing with no communication. That is what lets a verb replace a dense
// collective with point-to-point. Linear routing takes ALL log2(ranks) rank bits, so there is no
// intermediate fanout to express here.
//
// Dense is the default: peer(k) == k and count == ranks, so the same loops walk every rank and the
// verbs take their collective path. Every single-rank run is dense (Router::is_linear is false at
// R == 1), so the collectives are not a fallback but the common case.
//
// Two distinct failure modes if `shift` is wrong, which is why the plan is derived in one place. Ranks
// that DISAGREE deadlock: the pairing stops being symmetric and someone waits on a send never posted.
// Ranks that all agree on the same wrong shift stay symmetric and never hang -- they silently DROP the
// blocks outside the peer set, because pack_count_matrix_ / size_staging_send_ / pack_send_ only ever
// touch peers. pack_count_matrix_ asserts the non-peer remainder is empty to catch that one.
struct PeerPlan {
    bool sparse = false;
    int shift = 0;

    [[nodiscard]] constexpr auto dense() const -> bool { return !sparse; }
    [[nodiscard]] constexpr auto count(int ranks) const -> int { return sparse ? 1 : ranks; }
    // `k` indexes the peer set, which is a singleton when sparse.
    [[nodiscard]] constexpr auto peer(int me, int k) const -> int { return sparse ? (me ^ shift) : k; }
    [[nodiscard]] constexpr auto contains(int me, int b) const -> bool { return !sparse || b == (me ^ shift); }
    // The flat slots reachable from `me_flat` over a `ranks` x `parts` world. One expression per field:
    // sparse names the peer rank's `parts` slots, dense is the same with peer rank 0 and count(ranks)
    // == ranks, i.e. the whole world.
    [[nodiscard]] constexpr auto window(size_t me_flat, size_t ranks, size_t parts) const -> SlotWindow {
        const size_t peer_rank = sparse ? ((me_flat / parts) ^ static_cast<size_t>(shift)) : 0;
        return SlotWindow{.base = peer_rank * parts,
                          .count = static_cast<size_t>(count(static_cast<int>(ranks))) * parts};
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
