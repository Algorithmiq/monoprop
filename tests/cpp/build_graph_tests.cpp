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

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include "TestUtilities.h"

using namespace test_utils;
namespace utf = boost::unit_test;
namespace bdata = utf::data;

BOOST_DATA_TEST_CASE_F(ExampleDataFix,
                       build_graph_cases,
                       bdata::make(ds_pare_values) ^ bdata::make(ds_schrodinger_enabled),
                       pare,
                       sch_enabled) {
    const auto schrodinger_cutoff = make_schrodinger_cutoff(sch_enabled, cutoff);
    SimulatorConfig cfg{
        .schrodinger_cutoff = schrodinger_cutoff ? std::optional<unsigned int>(*schrodinger_cutoff) : std::nullopt,
        .cutoff_type = cutoff_type,
        .basis_change = basis_change,
    };
    test_evolve_build_graph<n_modes>(data, cfg, pare, data.actual_expval);
}

BOOST_DATA_TEST_CASE_F(ExampleDataFix,
                       build_graph_with_coeffs_cases,
                       bdata::make(ds_pare_values) ^ bdata::make(ds_schrodinger_enabled),
                       pare,
                       sch_enabled) {
    const auto schrodinger_cutoff = make_schrodinger_cutoff(sch_enabled, cutoff);
    SimulatorConfig cfg{
        .schrodinger_cutoff = schrodinger_cutoff ? std::optional<unsigned int>(*schrodinger_cutoff) : std::nullopt,
        .cutoff_type = cutoff_type,
        .basis_change = basis_change,
    };
    test_evolve_build_graph_with_coeffs<n_modes>(data, cfg, pare, data.actual_expval);
}
