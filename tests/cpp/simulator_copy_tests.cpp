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

#include <type_traits>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// Copy-constructing a simulator must produce a fully independent DEEP copy: identical results, and
// mutating one instance must never affect the other. The operator store is non-copyable, so the
// copy rebuilds it via clone() -- find()/indexing() therefore have to work on the copy's own rows.
// The MPI communicator handle is shared (not dup'd). This is the mechanism behind Python
// __deepcopy__.

using namespace test_utils;
using namespace monoprop;

// Deep copy is exposed via the (implicit) copy CONSTRUCTOR only; the simulator stays movable, and
// copy assignment is intentionally left deleted (the unique_ptr-owned store needs no assignment).
static_assert(std::is_copy_constructible_v<MonomialPropagator<8>>, "simulator must be copyable");
static_assert(std::is_move_constructible_v<MonomialPropagator<8>>, "simulator must stay movable");
static_assert(!std::is_copy_assignable_v<MonomialPropagator<8>>, "copy assignment stays deleted");

BOOST_FIXTURE_TEST_CASE(copy_constructed_simulator_matches_energy, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    auto copy = sim; // copy AFTER evolution

    BOOST_TEST(copy.graph_layers() == sim.graph_layers());
    BOOST_TEST(copy.size() == sim.size());

    const double e_orig = sim.expectation_value_functional()(data.parameters);
    const double e_copy = copy.expectation_value_functional()(data.parameters);
    BOOST_CHECK_SMALL(e_orig - e_copy, 1e-13);
}

BOOST_FIXTURE_TEST_CASE(copy_is_independent_of_source, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);

    auto copy = sim; // copy the UN-evolved simulator
    BOOST_TEST(copy.graph_layers() == 0u);

    // Evolve only the source; the copy must be untouched.
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_TEST(sim.graph_layers() > 0u);
    BOOST_TEST(copy.graph_layers() == 0u);

    // Evolving the copy independently reproduces the same energy.
    copy.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double e_sim = sim.expectation_value_functional()(data.parameters);
    const double e_copy = copy.expectation_value_functional()(data.parameters);
    BOOST_CHECK_SMALL(e_sim - e_copy, 1e-13);
}

// The graph is copied as a PER-INSTANCE layer list (vector<Layer>); the immutable LayerCores are
// shared between copies via shared_ptr. This proves the layer list is genuinely independent: an
// in-place contraction truncates the contracting copy's OWN layer list (slice_graph(..., contract=
// true) does layers_.resize), and destroying that copy drops its references to the shared cores --
// yet the other copy's graph stays complete and still replays to the original energy (the shared
// cores remain alive by reference count). Were the graph shared, contracting/destroying one would
// corrupt the other.
BOOST_FIXTURE_TEST_CASE(copy_graph_survives_other_being_contracted_and_destroyed, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto original = build_simulator<n_modes>(data, cfg);
    original.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const size_t layers_before = original.graph_layers();
    BOOST_TEST(layers_before > 0u);
    const double e_before = original.expectation_value_functional()(data.parameters);

    {
        auto copy = original; // deep copy; shares the immutable LayerCores
        BOOST_TEST(copy.graph_layers() == layers_before);

        // Contract the COPY in place: this truncates the copy's own layer list.
        copy.contract_partially(data.parameters, /*inplace=*/true);
        BOOST_TEST(copy.graph_layers() < layers_before);      // the copy's graph shrank...
        BOOST_TEST(original.graph_layers() == layers_before); // ...the original's did not

        // `copy` is destroyed at the end of this scope, releasing its core references.
    }

    // The original graph is intact and still replays to the same energy.
    BOOST_TEST(original.graph_layers() == layers_before);
    const double e_after = original.expectation_value_functional()(data.parameters);
    BOOST_CHECK_SMALL(e_before - e_after, 1e-13);
}

BOOST_FIXTURE_TEST_CASE(copy_constructed_simulator_index_valid, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    auto copy = sim; // copy construction (the deepcopy mechanism)

    // Every stored term must round-trip through the copy's own index (find confirms hash hits
    // against the copy's own rows, not the source's).
    const auto &idx = copy.indexing();
    BOOST_TEST(idx.size() == sim.indexing().size());
    bool all_found = true;
    idx.for_each([&](const auto &maj, size_t i) {
        const auto f = idx.find(maj);
        if (!f || *f != i) {
            all_found = false;
        }
    });
    BOOST_TEST(all_found);
}
