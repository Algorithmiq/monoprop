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

#include <memory>
#include <stdexcept>

#include "monoprop/detail/mpi/Comm.h"

namespace monoprop::mpi {

/// Owns a private duplicate of a caller's communicator and frees it on destruction.
///
/// monoprop talks point-to-point on fixed tags (see Pairwise.h), so running on the caller's own
/// communicator would let its messages and ours match each other. Immovable: the handle is freed exactly
/// once, by whoever holds the last `std::shared_ptr` (see duplicate_user_comm()).
class DuplicatedComm {
public:
    /// Collective over `parent`: every rank of the communicator must construct one, in the same order.
    explicit DuplicatedComm(MPI_Comm parent) : handle_(parent) {
#ifdef monoprop_ENABLE_MPI
        if (MPI_Comm_dup(parent, &handle_) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Comm_dup failed");
        }
#endif
    }

    DuplicatedComm(const DuplicatedComm &) = delete;
    auto operator=(const DuplicatedComm &) -> DuplicatedComm & = delete;

    // Python objects can outlive MPI_Finalize (mpi4py finalizes at interpreter exit), and freeing a
    // communicator afterwards is an error.
    ~DuplicatedComm() {
#ifdef monoprop_ENABLE_MPI
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized == 0) {
            MPI_Comm_free(&handle_);
        }
#endif
    }

    /// The duplicate; valid until the last owner is destroyed.
    [[nodiscard]] auto handle() const -> MPI_Comm { return handle_; }

private:
    MPI_Comm handle_;
};

/// Duplicates `comm` when it is a caller-supplied MPI communicator, collectively over it.
///
/// Returns null when there is nothing to isolate: an in-process (Shm / Hybrid) handle, which is already
/// owned by its PartitionGroup, MPI_COMM_NULL, and every build without MPI.
[[nodiscard]] inline auto duplicate_user_comm(const Comm &comm) -> std::shared_ptr<DuplicatedComm> {
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Mpi && comm.mpi != MPI_COMM_NULL) {
        return std::make_shared<DuplicatedComm>(comm.mpi);
    }
#else
    (void)comm;
#endif
    return nullptr;
}

} // namespace monoprop::mpi
