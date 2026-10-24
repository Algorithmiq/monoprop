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

#include <mpi4py/mpi4py.h>

namespace monoprop::bindings::detail {
auto get_mpi_comm(nb::object obj) -> MPI_Comm {
    if (!obj || obj.is_none()) {
        throw nb::type_error("monoprop._core_mpi needs an mpi4py communicator; comm=None selects the "
                             "serial backend");
    }
    auto *comm_ptr = PyMPIComm_Get(obj.ptr());
    if (comm_ptr == nullptr) {
        throw nb::python_error();
    }
    return *comm_ptr;
}

auto init_comm_bridge() -> void {
    // mpi4py reports through the Python C API; nanobind turns that into the traceback
    if (import_mpi4py() < 0) {
        throw nb::python_error();
    }
}
} // namespace monoprop::bindings::detail
