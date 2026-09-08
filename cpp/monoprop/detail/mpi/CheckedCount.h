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
#include <format>
#include <limits>
#include <stdexcept>

// Below MPICompat.h so the in-process transports (ShmComm, HybridComm), which MPICompat.h includes,
// narrow their counts through the same policy as the MPI collectives.

namespace monoprop::mpi {

// Covers both inputs inconsistent with the communicator (a count vector whose width is not the rank
// count) and payloads that outgrow MPI's int counts.
class CollectiveArgumentError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Per-rank counts are individually int-sized but their prefix sums need not be, so accumulate in
// `long long` and funnel every result through here: a signed int accumulator would be UB on overflow,
// and the wrapped value then sizes a buffer or becomes a negative displacement.
inline auto checked_mpi_count(long long value, const char *what = "Aggregate MPI count") -> int {
    if (value < 0 || value > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw CollectiveArgumentError(std::format("{} {} does not fit in the MPI int limit {} (message too large).",
                                                  what,
                                                  value,
                                                  std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

// A buffer size is not int-bounded to begin with: a per-peer payload is kWords<NumModes> + 1 elements
// per term, so a single wide-mode query round reaches INT_MAX well inside one partition's term space.
// Kept separate from the `long long` overload so a size_t above LLONG_MAX cannot sign-flip on the way
// into the check.
inline auto checked_mpi_count(size_t value, const char *what = "MPI count") -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw CollectiveArgumentError(std::format("{} {} does not fit in the MPI int limit {} (message too large).",
                                                  what,
                                                  value,
                                                  std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

} // namespace monoprop::mpi
