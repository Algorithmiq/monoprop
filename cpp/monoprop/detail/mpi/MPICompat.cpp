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

#include "monoprop/detail/mpi/Exchange.h"

#include <format>
#include <print>
#include <stdexcept>

#include "monoprop/detail/Profile.h"

namespace monoprop::mpi {

namespace {

// The PROCESS's identity, not the rank of whichever communicator a call happens to use: a flat run
// makes collectives on MPI_COMM_SELF and on MPI_COMM_WORLD, so the last one's rank would leave the line
// unattributable in a shared log.
#ifdef monoprop_ENABLE_PROFILE
auto world_rank_() -> int {
#ifdef monoprop_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }
#endif
    return 0;
}

// TU-local: every caller sits after the Shm/Hybrid dispatch and immediately before a real MPI_* call,
// so the function-local statics below are first constructed with MPI already initialised.
auto flat_comm_slot() -> profile::CommSlot * {
    static profile::CommRegistry registry(1, {.mpi_rank = world_rank_(), .transport = "flat"});
    static profile::CommSlot *slot = registry.slot(0);
    return slot;
}
#endif

} // namespace

#ifdef monoprop_ENABLE_PROFILE
auto flat_verb_ns() -> uint64_t * {
    auto *slot = flat_comm_slot();
    if (slot == nullptr) {
        return nullptr;
    }
    ++slot->n_verbs;
    return &slot->mpi_ns;
}

auto flat_wait_ns() -> uint64_t * {
    auto *slot = flat_comm_slot();
    if (slot == nullptr) {
        return nullptr;
    }
    ++slot->n_barriers;
    return &slot->barrier_ns;
}
#endif

#ifdef monoprop_ENABLE_MPI
auto init(int *argc, char ***argv) -> void {
    auto initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        // serialized (not funneled): under the hybrid the one-at-a-time MPI calls come from each rank's
        // partition-0 master, not the main thread. mpi4py already requests >= serialized.
        auto required = MPI_THREAD_SERIALIZED;
        auto provided = 0;
        MPI_Init_thread(argc, argv, required, &provided);
        if (provided < required) {
            auto comm = MPI_COMM_WORLD;
            std::print("Sorry, the MPI library does not provide MPI_THREAD_SERIALIZED support, which is required "
                       "by the partition/MPI hybrid transport.\n");
            MPI_Abort(comm, 1);
        }
    }
}

auto finalize() -> void {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}
#endif // monoprop_ENABLE_MPI

auto rank(const Comm &comm) -> int {
    if (comm.kind == Comm::Kind::Shm) {
        return comm.shm_rank;
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        return comm.hyb->global_rank(comm.shm_rank);
    }
    int r = 0;
    if (MPI_Comm_rank(comm.mpi, &r) != MPI_SUCCESS) {
        throw CollectiveArgumentError("MPI_Comm_rank failed");
    }
    return r;
#else
    return 0;
#endif
}

auto size(const Comm &comm) -> int {
    if (comm.kind == Comm::Kind::Shm) {
        return comm.shm->size();
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        return comm.hyb->size();
    }
    int s = 0;
    if (MPI_Comm_size(comm.mpi, &s) != MPI_SUCCESS) {
        throw CollectiveArgumentError("MPI_Comm_size failed");
    }
    return s;
#else
    return 1;
#endif
}

auto allreduce_sum_inplace(VecD &values, Comm comm) -> void {
    if (comm.kind == Comm::Kind::Shm) {
        comm.shm->allreduce_sum_inplace(comm.shm_rank, values.data(), values.size());
        return;
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->allreduce_sum_inplace(comm.shm_rank, values.data(), values.size());
        return;
    }
    monoprop_PROF_SCOPE_AT(verb, flat_verb_ns());
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_SUM, comm.mpi);
#else
    (void)values; // single participant: identity
#endif
}

auto alltoall_counts(const int *send_counts, int *recv_counts, int n, Comm comm) -> void {
    if (comm.kind == Comm::Kind::Shm) {
        comm.shm->alltoall_counts(comm.shm_rank, send_counts, recv_counts);
        return;
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoall_counts(comm.shm_rank, send_counts, recv_counts);
        return;
    }
    (void)n;
    monoprop_PROF_SCOPE_AT(verb, flat_verb_ns());
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm.mpi);
#else
    for (int i = 0; i < n; ++i) {
        recv_counts[i] = send_counts[i];
    }
#endif
}

auto resolve_recv(std::span<const int> send_counts, const Comm &comm, RecvLayoutCache &cache) -> const RecvLayout & {
    const auto n = static_cast<int>(send_counts.size());
    const int comm_size = mpi::size(comm);
    // alltoall_counts moves comm_size ints each way regardless of `n`, so a send vector that is not
    // exactly one entry per rank reads and writes out of bounds — reachable because layouts outlive
    // propagator copies and pare rebuilds.
    if (n != comm_size) {
        throw CollectiveArgumentError(
            std::format("Exchange layout has {} send counts but the communicator has {} ranks — a graph built for one "
                        "communicator cannot be replayed on another of a different size.",
                        n,
                        comm_size));
    }
    if (cache.comm_size == comm_size && static_cast<int>(cache.layout.counts.size()) == n) {
        return cache.layout;
    }

    RecvLayout &out = cache.layout;
    out.counts.resize(static_cast<size_t>(n));
    alltoall_counts(send_counts.data(), out.counts.data(), n, comm);
    out.displs.resize(static_cast<size_t>(n));
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        out.displs[static_cast<size_t>(i)] = checked_mpi_count(total);
        total += out.counts[static_cast<size_t>(i)];
    }
    out.total = checked_mpi_count(total);
    cache.comm_size = comm_size;
    return out;
}

} // namespace monoprop::mpi
