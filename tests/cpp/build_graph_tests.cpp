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

// graph_size()'s cosine count must be the real one, recomputed from the operator's inverted index.
// Cosine-ONLY means cos-scaled but not a rotation endpoint, so it is legitimately zero when nothing is
// truncated (every anticommuting term's sine partner survives and is an endpoint). A tight cutoff drops
// those partners and the count must become positive.
BOOST_AUTO_TEST_CASE(graph_size_reports_real_cosine_only_count) {
    constexpr size_t N = 8;
    const auto data = test_utils::load_case_data<N>("random_exact.msgpack");

    const auto sized = [&](unsigned int cutoff) {
        auto sim = MonomialPropagator<N>(data.hamiltonian, cutoff, data.initial_state, std::nullopt, MPI_COMM_SELF);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        return sim.graph_size();
    };

    // Truncating cutoff: some cos-scaled terms lose their sine partner, so cosine-only is positive.
    const auto truncated = sized(8);
    BOOST_CHECK_GT(truncated.first, 0U);
    BOOST_CHECK_GT(truncated.second, 0U);

    // Exact cutoff: every cos index is also a rotation endpoint, so cosine-only is genuinely zero.
    const auto exact = sized(2 * N);
    BOOST_CHECK_EQUAL(exact.first, 0U);
    BOOST_CHECK_GT(exact.second, truncated.second);
}
