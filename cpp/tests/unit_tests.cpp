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

#include <cstdlib>

#include <catch2/catch_session.hpp>

#include "monoprop/detail/mpi/MPICompat.h"

auto main(int argc, char* argv[]) -> int {
    // overwrite=0, so an explicit environment override still wins; why it is off: tests/cpp/README.md.
    setenv("monoprop_PARTITIONS", "off", 0);
    monoprop::mpi::init(&argc, &argv);

    Catch::Session session;
    const int command_line_result = session.applyCommandLine(argc, argv);
    if (command_line_result != 0) {
        monoprop::mpi::finalize();
        return command_line_result;
    }

    const int result = session.run();
    monoprop::mpi::finalize();
    return result;
}
