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

#include <boost/test/unit_test.hpp>

#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/logging/QuillWrapper.h"
#include "monoprop/logging/Utils.h"

static auto init() -> bool {
    monoprop::logging::setup_quill();
    monoprop_global_logger->set_log_level(monoprop::logging::log_level());
    return true;
}

auto main(int argc, char* argv[]) -> int {
    monoprop::mpi::init(&argc, &argv);
    int result = boost::unit_test::unit_test_main(&init, argc, argv);
    monoprop::mpi::finalize();
    return result;
}
