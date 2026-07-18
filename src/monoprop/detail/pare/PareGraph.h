#pragma once

// pare_graph / get_pared_graph are declared in the public MPFunctions.h (reachable from the
// simulator impl header, which lives under include/). This internal header pulls in that public
// declaration plus the MPI compat layer the .cpp helpers need, so the .cpp definitions match.
#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"
