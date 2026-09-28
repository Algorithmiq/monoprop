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

#include <cstdio>
#include <exception>
#include <format>
#include <string>
#include <utility>

#include "monoprop/detail/mpi/Comm.h"

namespace monoprop::mpi {

/*!
 * \brief The message carried by \a error, for a diagnostic.
 *
 * Never throws; an exception that is not a `std::exception` reads as "unknown exception".
 */
inline auto describe_exception(const std::exception_ptr &error) noexcept -> std::string {
    try {
        std::rethrow_exception(error);
    }
    catch (const std::exception &e) {
        try {
            return e.what();
        }
        catch (...) {
            return {};
        }
    }
    catch (...) {
        return "unknown exception";
    }
}

/*!
 * \brief Handle a failure on the calling rank during a distributed operation, from the controlling thread.
 *
 * On an ordinary MPI communicator with more than one rank, the peers may already be committed to a
 * communication this rank will never join, and no exception can release them. So this reports the rank
 * and the underlying error on stderr and calls `MPI_Abort` on \a comm; if `MPI_Abort` returns, it calls
 * `std::terminate`. On one rank, without MPI, or on a legacy in-process communicator (whose partition
 * runtime poisons its own transport), it rethrows \a error unchanged, preserving the original type.
 *
 * Callers catch failures where they happen, inside the lifetime of any posted request handle
 * (mpi::Ticket, mpi::PendingAlltoallv), so that unwinding never drains requests peers will not complete.
 * Never call it from an OpenMP worker or from a thread other than MPI's initializing thread.
 *
 * \param comm  The communicator the operation runs on.
 * \param error The failure; must not be null.
 */
[[noreturn]] inline auto operation_failed(const Comm &comm, std::exception_ptr error) -> void {
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Mpi) {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        int size = 1;
        if (initialized != 0 && finalized == 0 && MPI_Comm_size(comm.mpi, &size) == MPI_SUCCESS && size > 1) {
            int rank = 0;
            MPI_Comm_rank(comm.mpi, &rank);
            const auto line =
                std::format("monoprop: rank {} of {}: a distributed operation failed on this rank: {}. Its peers may "
                            "already be waiting on communication this rank will not complete, so monoprop is "
                            "aborting the communicator rather than leaving them blocked.\n",
                            rank,
                            size,
                            describe_exception(error));
            std::fputs(line.c_str(), stderr);
            std::fflush(stderr);
            MPI_Abort(comm.mpi, 1);
            std::terminate();
        }
    }
#else
    (void)comm;
#endif
    std::rethrow_exception(std::move(error));
}

/*!
 * \brief Run \a fn and send any exception it throws to operation_failed().
 *
 * Wrap each phase that can throw while peers may be committed to communication, and place the wrapper
 * inside the lifetime of any request handle the phase overlaps.
 *
 * \return Whatever \a fn returns.
 */
template <typename Fn>
auto guard_distributed(const Comm &comm, Fn &&fn) -> decltype(std::forward<Fn>(fn)()) {
    try {
        return std::forward<Fn>(fn)();
    }
    catch (...) {
        operation_failed(comm, std::current_exception());
    }
}

} // namespace monoprop::mpi
