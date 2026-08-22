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
#include <vector>

namespace monoprop::mpi {

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
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm.mpi);
#else
    for (int i = 0; i < n; ++i) {
        recv_counts[i] = send_counts[i];
    }
#endif
}

auto check_exchange_symmetry(std::span<const int> send_counts, const Comm &comm) -> void {
    const auto n = static_cast<int>(send_counts.size());
    const int comm_size = mpi::size(comm);
    // Above the gate deliberately: the width of the layout is a PRECONDITION of posting the
    // exchange at all, not part of the optional symmetry audit below, so it must hold on every
    // build and every path. MPI_Alltoallv reads comm_size counts and comm_size displacements
    // regardless of `n`, so begin_flat_exchange hands it a short array and the library reads off
    // the end — undefined behaviour where an exception belongs. Reachable because layouts outlive
    // propagator copies and pare rebuilds: a graph built for one communicator can be replayed on
    // another of a different size.
    if (n != comm_size) {
        throw CollectiveArgumentError(
            std::format("Exchange layout has {} send counts but the communicator has {} ranks — a graph built for one "
                        "communicator cannot be replayed on another of a different size.",
                        n,
                        comm_size));
    }

#ifndef monoprop_CHECK_EXCHANGE_SYMMETRY
    return; // audit compiled out; see the build option of the same name
#else
    // Compiled in or out, never selected at run time: what follows is a collective, and a
    // per-rank environment variable that one rank reads differently makes the ranks disagree
    // about whether the collective happens at all -- a job-wide hang, not a wrong number.
    // A build option cannot disagree between the ranks of one job.
    std::vector<int> recv_counts(static_cast<size_t>(n));
    alltoall_counts(send_counts.data(), recv_counts.data(), n, comm);
    for (int i = 0; i < n; ++i) {
        const int sent = send_counts[static_cast<size_t>(i)];
        const int received = recv_counts[static_cast<size_t>(i)];
        if (sent != received) {
            // Naming the slot and both counts, because the whole point of the check is that the
            // unguarded failure carries neither.
            throw CollectiveArgumentError(std::format(
                "Exchange count matrix is not symmetric at slot {}: this rank sends {} there but receives {} back. "
                "The exchange derives its recv layout from its send layout on the strength of that equality, so a "
                "routing change that breaks it must be caught here rather than as a hang in MPI_Alltoallv.",
                i,
                sent,
                received));
        }
    }
#endif // monoprop_CHECK_EXCHANGE_SYMMETRY
}

} // namespace monoprop::mpi
