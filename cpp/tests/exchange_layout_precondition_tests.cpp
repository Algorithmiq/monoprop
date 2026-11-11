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

// Preconditions on a layer's exchange layout, on an ordinary communicator (or the MPI-off stub).

#include <boost/test/unit_test.hpp>

#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"

using monoprop::mpi::CollectiveArgumentError;
using monoprop::mpi::Comm;

namespace {

auto require_width_checks(const Comm &comm) -> void {
    const int ranks = monoprop::mpi::size(comm);
    BOOST_REQUIRE_GE(ranks, 1);
    for (const int width : {0, ranks - 1, ranks + 1, ranks + 3}) {
        if (width < 0 || width == ranks) {
            continue;
        }
        BOOST_TEST_CONTEXT("width " << width << " on " << ranks << " rank(s)") {
            const std::vector<int> layout(static_cast<size_t>(width), 0);
            BOOST_CHECK_THROW(monoprop::mpi::check_exchange_layout_width(layout, comm), CollectiveArgumentError);
        }
    }
    const std::vector<int> exact(static_cast<size_t>(ranks), 0);
    BOOST_CHECK_NO_THROW(monoprop::mpi::check_exchange_layout_width(exact, comm));
}

} // namespace

BOOST_AUTO_TEST_CASE(exchange_layout_width_is_checked) {
    require_width_checks(Comm(MPI_COMM_SELF));
}

// The width check runs before any communication: only rank 0 calls it here, so a collective inside would leave rank 0
// waiting for peers that never join (a hang, which the MPI variants' timeout reports as a failure). In the MPI-off
// build and in the per-case serial runs the world has one rank.
BOOST_AUTO_TEST_CASE(exchange_layout_width_is_rejected_before_communication) {
    const Comm world(MPI_COMM_WORLD);
    if (monoprop::mpi::rank(world) == 0) {
        const std::vector<int> too_wide(static_cast<size_t>(monoprop::mpi::size(world)) + 2, 0);
        BOOST_CHECK_THROW(monoprop::mpi::check_exchange_layout_width(too_wide, world), CollectiveArgumentError);
        const std::vector<int> too_short(static_cast<size_t>(monoprop::mpi::size(world)) - 1, 0);
        BOOST_CHECK_THROW(monoprop::mpi::check_exchange_layout_width(too_short, world), CollectiveArgumentError);
    }
    require_width_checks(world);
}
