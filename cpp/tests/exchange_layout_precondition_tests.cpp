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

// Preconditions on a layer's exchange layout. Driven through ShmComm, so it needs no ranks.

#include <boost/test/unit_test.hpp>

#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/ShmComm.h"

using monoprop::mpi::CollectiveArgumentError;
using monoprop::mpi::Comm;
using monoprop::mpi::ShmComm;

BOOST_AUTO_TEST_CASE(exchange_layout_width_is_checked) {
    // Sized 4 but driven by one thread: the throw must precede any collective, or this hangs
    // instead of failing -- the width is checked against mpi::size(comm) alone.
    ShmComm world(4);
    const Comm comm = Comm::make_shm(&world, /*rank=*/0);
    BOOST_REQUIRE_EQUAL(monoprop::mpi::size(comm), 4);

    const std::vector<int> too_short(3, 0);
    BOOST_CHECK_THROW(monoprop::mpi::check_exchange_layout_width(too_short, comm), CollectiveArgumentError);

    const std::vector<int> too_long(5, 0);
    BOOST_CHECK_THROW(monoprop::mpi::check_exchange_layout_width(too_long, comm), CollectiveArgumentError);

    const std::vector<int> empty;
    BOOST_CHECK_THROW(monoprop::mpi::check_exchange_layout_width(empty, comm), CollectiveArgumentError);
}

// The complement: a layout of exactly the right width is accepted.
BOOST_AUTO_TEST_CASE(exchange_layout_of_the_right_width_is_accepted) {
    ShmComm world(1);
    const Comm comm = Comm::make_shm(&world, /*rank=*/0);

    const std::vector<int> exact(1, 0);
    BOOST_CHECK_NO_THROW(monoprop::mpi::check_exchange_layout_width(exact, comm));
}
