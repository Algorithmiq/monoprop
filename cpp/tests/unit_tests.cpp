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

#include <csignal>
#include <cstdlib>

#include <boost/test/unit_test.hpp>

#include "monoprop/detail/mpi/MPICompat.h"

static auto init() -> bool {
    return true;
}

auto main(int argc, char* argv[]) -> int {
    // overwrite=0, so an explicit environment override still wins; why it is off: tests/cpp/README.md.
    setenv("monoprop_PARTITIONS", "off", 0);
    // Must precede mpi::init, and only the harness may do it -- changing a signal disposition is a
    // process-wide act, so the library cannot.
    //
    // In an MPI build every one of these per-case processes runs a *singleton* MPI_Init: no launcher, so
    // PMIx opens a socket to a daemon that is not there. Under enough concurrency a write to that dead
    // socket lands, and SIGPIPE's default action kills the process mid-init -- which surfaced as
    // load-dependent SIGPIPE exceptions in random cases under `ctest -j`, each passing when re-run alone.
    // Ignoring it makes the write return EPIPE for the MPI layer to handle. Python callers never saw this
    // because CPython already ignores SIGPIPE at startup.
    std::signal(SIGPIPE, SIG_IGN);
    monoprop::mpi::init(&argc, &argv);
    int result = boost::unit_test::unit_test_main(&init, argc, argv);
    monoprop::mpi::finalize();
    return result;
}
