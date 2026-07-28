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

#define BOOST_TEST_MODULE "MonoProp Unit Tests"

#include <cstdlib>

#include <boost/test/unit_test.hpp>

#include "monoprop/detail/mpi/MPICompat.h"

static auto init() -> bool {
    return true;
}

auto main(int argc, char* argv[]) -> int {
    // The white-box accessors (mp_op()/indexing()/graph()/graph_data()) read this partition's state,
    // which is empty on a partition-backed facade, so partitioning is forced off for the suite. overwrite=0
    // keeps an explicit dev override working; partition_equivalence_tests passes partitions= and wins anyway.
    setenv("monoprop_PARTITIONS", "off", 0);
    monoprop::mpi::init(&argc, &argv);
    int result = boost::unit_test::unit_test_main(&init, argc, argv);
    monoprop::mpi::finalize();
    return result;
}
