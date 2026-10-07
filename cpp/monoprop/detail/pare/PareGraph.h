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
#include <functional>

// pare_graph is declared in MPFunctions.h.
#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/MPGraphEncoding.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/monopropExport.h"

namespace monoprop::detail {

/*!
 * \brief Prune one owner's graph to the subgraph reaching `nonzero_inds`: the communicator-free core of pare_graph().
 *
 * A single backward sweep over the layers in the picture's order (last layer first in Heisenberg, first layer first
 * in Schrödinger). At each layer every partner-owner endpoint is kept before the cosine set is filtered: a
 * sin_recv target always, and a sin_send source only for a slot other than `flat_owner` (another shard of the same
 * process is a partner owner like a remote one). Cross-owner lists are never pruned. A layer whose filtered cosine
 * set is unchanged keeps recomputing; any other layer stores its filtered set, and an empty stored set replays
 * nothing. Cores stay shared with `graph`; the returned layer sequence is independently owned.
 *
 * \param graph             The owner's graph.
 * \param nonzero_inds      Rows to keep; indices at or past `local_index_count` are ignored.
 * \param local_index_count The owner's row count.
 * \param schrodinger       The picture, which fixes the sweep order.
 * \param flat_owner        This owner's flat routing slot, `rank * T + shard`; never derived from a communicator.
 * \param full_cos_of_layer Writes layer i's full cosine set into its second argument, replacing the contents. The
 *                          sweep passes the same buffer for every layer, so a writer that reuses its capacity
 *                          allocates once per owner instead of once per layer.
 * \return The pared graph. Each stored set is allocated at its exact size.
 */
monoprop_EXPORT auto pare_graph_owner(const MPGraph &graph,
                                      const VecZ &nonzero_inds,
                                      size_t local_index_count,
                                      bool schrodinger,
                                      size_t flat_owner,
                                      const std::function<void(size_t, CosMask &)> &full_cos_of_layer) -> MPGraph;

} // namespace monoprop::detail
