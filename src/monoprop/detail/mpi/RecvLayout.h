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

#include <vector>

// Plain receive-layout types for the variable all-to-all facade (see Exchange.h). Kept MPI-free and
// dependency-light so graph-encoding types (LayerExchangeLayout) can embed the cache without pulling
// in <mpi.h> or the exchange machinery.

namespace monoprop::mpi {

// Resolved receive side of a variable all-to-all: per-rank recv counts + displacements and the total.
struct RecvLayout {
    std::vector<int> counts;
    std::vector<int> displs;
    int total = 0;
};

// Per-layer cache of a resolved RecvLayout, keyed by communicator size: a replayed graph's send pattern
// is fixed, so a hit (comm_size unchanged) skips the count round. Reset state is comm_size == -1.
struct RecvLayoutCache {
    RecvLayout layout;
    int comm_size = -1;
};

} // namespace monoprop::mpi
