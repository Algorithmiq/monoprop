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

#include <algorithm>
#include <format>
#include <print>
#include <stdexcept>

#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/Pairwise.h"
#endif

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

auto geometry(const Comm &comm) -> Geometry {
    if (comm.kind == Comm::Kind::Shm) {
        return {.ranks = 1, .partitions = comm.shm->size()};
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        return {.ranks = comm.hyb->ranks(), .partitions = comm.hyb->partitions()};
    }
    return {.ranks = size(comm), .partitions = 1};
#else
    return {.ranks = 1, .partitions = 1};
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

auto alltoall_counts(const int *send_counts, int *recv_counts, int n, Comm comm, PeerPlan plan) -> void {
    if (comm.kind == Comm::Kind::Shm) {
        comm.shm->alltoall_counts(comm.shm_rank, send_counts, recv_counts);
        return;
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoall_counts(comm.shm_rank, send_counts, recv_counts, plan);
        return;
    }
    if (!plan.dense()) {
        // S == 1 world: exchange one int with each reachable peer; the rest of the row is zero by
        // definition, so it must be cleared rather than left from a previous round.
        int me = 0;
        MPI_Comm_rank(comm.mpi, &me);
        std::fill(recv_counts, recv_counts + n, 0);
        const PeerLayout one{.block = 1};
        std::vector<MPI_Request> reqs;
        sparse_pairwise(plan,
                        me,
                        n,
                        comm.mpi,
                        kFlatCountTag,
                        MPI_INT,
                        sizeof(int),
                        reinterpret_cast<const std::byte *>(send_counts),
                        one,
                        reinterpret_cast<std::byte *>(recv_counts),
                        one,
                        reqs);
        return;
    }
    (void)n;
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm.mpi);
#else
    (void)plan; // single participant: nothing to narrow
    for (int i = 0; i < n; ++i) {
        recv_counts[i] = send_counts[i];
    }
#endif
}

auto check_exchange_layout_width(std::span<const int> send_counts, const Comm &comm) -> void {
    const auto n = static_cast<int>(send_counts.size());
    const int comm_size = mpi::size(comm);
    // MPI_Alltoallv reads comm_size counts and displacements whatever the span holds.
    if (n != comm_size) {
        throw CollectiveArgumentError(
            std::format("Exchange layout has {} send counts but the communicator has {} ranks — a graph built for one "
                        "communicator cannot be replayed on another of a different size.",
                        n,
                        comm_size));
    }
}

} // namespace monoprop::mpi
