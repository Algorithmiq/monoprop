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

#include "binder.h"

namespace monoprop::bindings::detail {
auto get_mpi_comm(nb::object obj) -> MPI_Comm {
    if (obj.is_valid() && !obj.is_none()) {
        throw nb::type_error("monoprop._core is the serial backend and takes comm=None; communicators "
                             "are served by monoprop._core_mpi");
    }
    // Comm.h's fallback handle; the serial core never communicates
    return MPI_COMM_WORLD;
}

auto init_comm_bridge() -> void {}
} // namespace monoprop::bindings::detail
