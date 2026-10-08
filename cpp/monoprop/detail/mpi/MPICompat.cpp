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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <stdexcept>

#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/Pairwise.h"
#include "monoprop/detail/mpi/Routing.h"
#endif

namespace monoprop::mpi {

#ifdef monoprop_ENABLE_MPI
auto init(int *argc, char ***argv) -> void {
    auto initialized = 0;
    MPI_Initialized(&initialized);
    int provided = MPI_THREAD_SINGLE;
    if (!initialized) {
        // Funneled: only this thread, as the primary of the library's team, ever calls MPI.
        MPI_Init_thread(argc, argv, kRequiredThreadLevel, &provided);
    }
    else {
        // Host-initialized: never reinitialize, and never assume the host's request was granted.
        MPI_Query_thread(&provided);
    }
    if (!thread_level_satisfies(provided, kRequiredThreadLevel)) {
        // Fixed strings only: nothing here may allocate (or throw) before the abort.
        std::fputs("monoprop: the MPI library provides ", stderr);
        std::fputs(thread_level_name(provided), stderr);
        std::fputs(" but monoprop requires ", stderr);
        std::fputs(thread_level_name(kRequiredThreadLevel), stderr);
        std::fputs("; aborting.\n", stderr);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
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

auto require_thread_support() -> void {
#ifdef monoprop_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        return;
    }
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    if (!thread_level_satisfies(provided, kRequiredThreadLevel)) {
        throw MpiThreadLevelUnsupported(
            std::format("monoprop requires {} support from MPI, but the initialized MPI library provides {}. "
                        "Initialize MPI with MPI_Init_thread requesting {} or higher (mpi4py requests "
                        "MPI_THREAD_MULTIPLE unless mpi4py.rc.thread_level says otherwise).",
                        thread_level_name(kRequiredThreadLevel),
                        thread_level_name(provided),
                        thread_level_name(kRequiredThreadLevel)));
    }
#endif
}

auto require_initializing_thread() -> void {
#ifdef monoprop_ENABLE_MPI
    // MPI_Initialized, MPI_Finalized and MPI_Is_thread_main may be called from any thread; nothing else is.
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized == 0 || finalized != 0) {
        return;
    }
    int is_main = 0;
    MPI_Is_thread_main(&is_main);
    if (is_main == 0) {
        std::fputs("monoprop: an operation that uses MPI was called from a thread other than the thread that "
                   "initialized MPI. monoprop requires every such call to come from the thread that initialized "
                   "MPI, whatever thread support MPI provides. This process stops here without communicating; "
                   "the launcher ends its peers.\n",
                   stderr);
        std::fflush(stderr);
        std::abort();
    }
#endif
}

auto rank(const Comm &comm) -> int {
#ifdef monoprop_ENABLE_MPI
    int r = 0;
    if (MPI_Comm_rank(comm.mpi, &r) != MPI_SUCCESS) {
        throw CollectiveArgumentError("MPI_Comm_rank failed");
    }
    return r;
#else
    (void)comm;
    return 0;
#endif
}

auto size(const Comm &comm) -> int {
#ifdef monoprop_ENABLE_MPI
    int s = 0;
    if (MPI_Comm_size(comm.mpi, &s) != MPI_SUCCESS) {
        throw CollectiveArgumentError("MPI_Comm_size failed");
    }
    return s;
#else
    (void)comm;
    return 1;
#endif
}

#ifdef monoprop_ENABLE_MPI
namespace {
// Cached as a communicator attribute, so a recycled MPI_Comm handle cannot inherit a stale answer.
auto routing_keyval() -> int {
    static const int keyval = [] {
        int k = MPI_KEYVAL_INVALID;
        MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &k, nullptr);
        return k;
    }();
    return keyval;
}
} // namespace
#endif

auto routes_pairwise(const Comm &comm) -> bool {
#ifdef monoprop_ENABLE_MPI
    const int ranks = size(comm);
    if (ranks <= 1) {
        return false; // no peer to pair with, and no collective to agree through
    }
    void *cached = nullptr;
    int found = 0;
    MPI_Comm_get_attr(comm.mpi, routing_keyval(), &cached, &found);
    if (found != 0) {
        return reinterpret_cast<intptr_t>(cached) != 0;
    }
    const bool pairwise = agree_routes_pairwise(comm.mpi, ranks, routing::Config::from_env());
    MPI_Comm_set_attr(comm.mpi, routing_keyval(), reinterpret_cast<void *>(static_cast<intptr_t>(pairwise)));
    return pairwise;
#else
    (void)comm;
    return false; // one process: there is no inter-rank transport to choose
#endif
}

auto allreduce_sum_inplace(VecD &values, Comm comm) -> void {
#ifdef monoprop_ENABLE_MPI
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_SUM, comm.mpi);
#else
    (void)values; // single participant: identity
    (void)comm;
#endif
}

auto alltoall_counts(const int *send_counts, int *recv_counts, int n, Comm comm, PeerPlan plan) -> void {
    require_routable(plan, size(comm));
#ifdef monoprop_ENABLE_MPI
    if (!plan.dense()) {
        // One int with the peer. Non-peers are zero by definition, so clear them.
        int me = 0;
        MPI_Comm_rank(comm.mpi, &me);
        std::fill(recv_counts, recv_counts + n, 0);
        const PeerLayout one{.block = 1};
        // Eager: the caller reads recv_counts on return.
        std::vector<MPI_Request> reqs;
        const SparsePairwiseArgs pairwise{
            .plan = plan,
            .me = me,
            .num_ranks = n,
            .comm = comm.mpi,
            .tag = kFlatCountTag,
            .datatype = MPI_INT,
            .elem = sizeof(int),
            .send = reinterpret_cast<const std::byte *>(send_counts),
            .send_layout = one,
            .recv = reinterpret_cast<std::byte *>(recv_counts),
            .recv_layout = one,
        };
        const int posted = sparse_pairwise(pairwise, reqs);
        MPI_Waitall(posted, reqs.data(), MPI_STATUSES_IGNORE);
        return;
    }
    (void)n;
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm.mpi);
#else
    (void)plan; // single participant: nothing to narrow
    (void)comm;
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
