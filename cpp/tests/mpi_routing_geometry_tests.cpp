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

#include <boost/test/unit_test.hpp>

#include <optional>

#include "RankPreconditions.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"

#ifdef monoprop_ENABLE_MPI

#include <mpi.h>

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = 8;

} // namespace

// Linear routing XORs rank indices, so it has no routing for 3 ranks: construction must refuse on
// every member rank instead of hanging the first exchange.
BOOST_AUTO_TEST_CASE(linear_routing_refuses_a_three_rank_communicator,
                     *test_utils::min_ranks(3) * test_utils::linear_routing()) {
    const int world_rank = mpi::rank(MPI_COMM_WORLD);
    MPI_Comm three_ranks = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < 3 ? 0 : MPI_UNDEFINED, world_rank, &three_ranks);
    if (three_ranks == MPI_COMM_NULL) {
        return; // outside the three-rank communicator there is nothing to construct
    }
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    BOOST_CHECK_THROW(static_cast<void>(MonomialPropagator<kNumModes>(data.hamiltonian,
                                                                      2 * kNumModes,
                                                                      data.initial_state,
                                                                      std::nullopt,
                                                                      three_ranks,
                                                                      std::nullopt,
                                                                      std::nullopt,
                                                                      CutoffType::Length,
                                                                      std::nullopt)),
                      routing::UnroutableGeometry);
    MPI_Comm_free(&three_ranks);
}

#endif // monoprop_ENABLE_MPI
