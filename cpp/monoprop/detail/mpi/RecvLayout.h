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

#include <cstdint>
#include <vector>

// Kept MPI-free and dependency-light so graph-encoding types (LayerCore) can embed the cache
// without pulling in <mpi.h> or the exchange machinery (see Exchange.h).

namespace monoprop::mpi {

struct RecvLayout {
    std::vector<int> counts;
    std::vector<int> displs;
    int total = 0;
};

// The transpose of ONE send pattern. Reusing it for a different pattern returns wrong
// displacements silently, so the cache carries the identity of the pattern it was built from.
//
// What must be rank-uniform is the hit/miss DECISION, not the id's value. A miss runs
// alltoall_counts, which is a collective, so two ranks disagreeing about validity is a
// distributed HANG rather than a wrong answer. Binding the id to the layer that owns the cache
// gives that for free: every rank walks the same layers in the same order, so every rank misses
// on a layer's first resolve and hits afterwards, whatever the local id values happen to be.
//
// It must NOT be derived from the send counts themselves (a total, a checksum). Those are
// rank-local, so two patterns can collide on one rank and not on another -- and that difference
// is exactly the split decision that hangs.
struct RecvLayoutCache {
    RecvLayout layout;
    int comm_size = -1;
    uint64_t generation = 0; // 0 = never populated
};

} // namespace monoprop::mpi
