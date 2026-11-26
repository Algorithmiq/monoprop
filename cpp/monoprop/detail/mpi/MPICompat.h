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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <print>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Comm.h owns the MPI_Comm typedef (real or non-MPI fallback) and the ordinary mpi::Comm handle.
#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/Pairwise.h"
#endif

// These includes are here on purpose and should not be moved to the top
#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop::mpi {

/// Initialized MPI provides less thread support than monoprop requires (required_thread_level).
class MpiThreadLevelUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#ifdef monoprop_ENABLE_MPI
/// The MPI thread support monoprop requests and requires. FUNNELED: only the thread that initialized MPI, as the
/// primary of the library's OpenMP team, ever calls MPI. A higher level is accepted and changes nothing: workers still
/// never call MPI, and public calls must still come from the initializing thread.
inline constexpr int required_thread_level = MPI_THREAD_FUNNELED;

/// Whether \a provided thread support satisfies \a required; MPI orders the levels
/// SINGLE < FUNNELED < SERIALIZED < MULTIPLE. A higher level never relaxes the initializing-thread rule.
constexpr auto thread_level_satisfies(int provided, int required) noexcept -> bool {
    return provided >= required;
}

/// The MPI constant naming a thread support level, for diagnostics.
constexpr auto thread_level_name(int level) noexcept -> const char * {
    if (level == MPI_THREAD_SINGLE) {
        return "MPI_THREAD_SINGLE";
    }
    if (level == MPI_THREAD_FUNNELED) {
        return "MPI_THREAD_FUNNELED";
    }
    if (level == MPI_THREAD_SERIALIZED) {
        return "MPI_THREAD_SERIALIZED";
    }
    if (level == MPI_THREAD_MULTIPLE) {
        return "MPI_THREAD_MULTIPLE";
    }
    return "an unknown MPI thread level";
}

/// Initialize MPI requesting required_thread_level unless the host already did, then check the level actually
/// provided in either case: a level below it prints a diagnostic and aborts MPI_COMM_WORLD. Host-initialized MPI is
/// never reinitialized.
monoprop_EXPORT auto init(int *argc = nullptr, char ***argv = nullptr) -> void;
monoprop_EXPORT auto finalize() -> void;

namespace detail {
template <class>
inline constexpr bool unsupported_mpi_datatype_v = false;
} // namespace detail

template <typename T>
struct datatype {
    static auto get() -> MPI_Datatype {
        if constexpr (std::is_same_v<T, int>) {
            return MPI_INT;
        }
        else if constexpr (std::is_same_v<T, double>) {
            return MPI_DOUBLE;
        }
        else if constexpr (std::is_same_v<T, uint32_t>) {
            return MPI_UINT32_T;
        }
        else if constexpr (std::is_same_v<T, uint64_t>) {
            return MPI_UINT64_T;
        }
        else if constexpr (std::is_same_v<T, size_t>) {
            if constexpr (std::is_same_v<size_t, unsigned int>) {
                return MPI_UNSIGNED;
            }
            else if constexpr (std::is_same_v<size_t, unsigned long>) {
                return MPI_UNSIGNED_LONG;
            }
            else if constexpr (std::is_same_v<size_t, unsigned long long>) {
                return MPI_UNSIGNED_LONG_LONG;
            }
            else {
                static_assert(detail::unsupported_mpi_datatype_v<T>, "Unsupported size_t representation for MPI");
            }
        }
        else {
            static_assert(detail::unsupported_mpi_datatype_v<T>, "Unsupported MPI datatype");
        }
    }
};
#else
inline auto init(int * /*argc*/ = nullptr, char *** /*argv*/ = nullptr) -> void {}
inline auto finalize() -> void {}
#endif // monoprop_ENABLE_MPI

/// Check that initialized MPI provides at least required_thread_level, whoever initialized it.
/// \throws MpiThreadLevelUnsupported if it does not. A no-op without MPI or before MPI is initialized.
monoprop_EXPORT auto require_thread_support() -> void;

/// Fail fast unless the caller is the thread that initialized MPI.
///
/// A call from any other thread breaks the host contract, whatever thread support MPI provides: it prints a
/// diagnostic and calls std::abort without calling MPI_Abort or any other communication, so the launcher
/// must end the peers. A no-op without MPI or while MPI is not initialized. Every MPI-using propagator
/// operation checks it on entry.
monoprop_EXPORT auto require_initializing_thread() -> void;

monoprop_EXPORT auto rank(const Comm &comm) -> int;
monoprop_EXPORT auto size(const Comm &comm) -> int;

// Whether this communicator's exchanges go point-to-point: agreed collectively once and cached on it,
// because ranks that disagree hang. Throws RoutingDisagreement on every rank together on a mismatch.
monoprop_EXPORT auto routes_pairwise(const Comm &comm) -> bool;

template <typename T>
inline auto allreduce_sum(T local_val, Comm comm) -> T {
#ifdef monoprop_ENABLE_MPI
    T global_val{};
    MPI_Allreduce(&local_val, &global_val, 1, datatype<T>::get(), MPI_SUM, comm.mpi);
    return global_val;
#else
    (void)comm; // single participant: identity
    return local_val;
#endif
}

monoprop_EXPORT auto allreduce_sum_inplace(VecD &values, Comm comm) -> void;

} // namespace monoprop::mpi
