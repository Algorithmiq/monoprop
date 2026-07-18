#pragma once

#include <vector>

// Plain receive-layout types for the variable all-to-all facade (see Exchange.h). Kept MPI-free and
// dependency-light so graph-encoding types (LayerExchangeLayout) can embed the cache without pulling
// in <mpi.h> or the exchange machinery.

namespace monoprop::mpi {

/// Resolved receive side of a variable all-to-all: per-rank recv counts + displacements and the total.
struct RecvLayout {
    std::vector<int> counts;
    std::vector<int> displs;
    int total = 0;
};

/// Per-layer cache of a resolved RecvLayout, keyed by communicator size. The send-count pattern of a
/// replayed graph is fixed, so once resolved the recv counts/displs are identical on every subsequent
/// evaluation; a cache hit (comm_size unchanged) skips the MPI_Alltoall count round entirely. Reset
/// state is comm_size == -1.
struct RecvLayoutCache {
    RecvLayout layout;
    int comm_size = -1;
};

} // namespace monoprop::mpi
