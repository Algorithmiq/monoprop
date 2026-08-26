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

// Copy-constructing a simulator must produce a fully independent deep copy -- the mechanism behind
// Python __deepcopy__. The operator store is non-copyable, so the copy rebuilds it via clone() and
// find()/indexing() have to work on the copy's own rows. The MPI communicator handle is shared.

using namespace test_utils;
using namespace monoprop;

// Copy assignment is deliberately deleted: the unique_ptr-owned store needs no assignment.
static_assert(std::is_copy_constructible_v<MonomialPropagator<8>>, "simulator must be copyable");
static_assert(std::is_move_constructible_v<MonomialPropagator<8>>, "simulator must stay movable");
static_assert(!std::is_copy_assignable_v<MonomialPropagator<8>>, "copy assignment stays deleted");

// mp_op()/indexing() are single deducing-this members. indexing() reaches its result through a
// unique_ptr, whose operator* hands back a mutable referent regardless of the owner's const-ness, so
// without the forward_like a const propagator would silently expose a writable index.
// `detail` is qualified: the two using-directives above make an unqualified one ambiguous.
static_assert(
    std::is_same_v<decltype(std::declval<MonomialPropagator<8> &>().mp_op()), monoprop::detail::MPOperator<8> &>);
static_assert(std::is_same_v<decltype(std::declval<const MonomialPropagator<8> &>().mp_op()),
                             const monoprop::detail::MPOperator<8> &>);
static_assert(
    std::is_same_v<decltype(std::declval<MonomialPropagator<8> &>().indexing()), monoprop::detail::OperatorIndex<8> &>);
static_assert(std::is_same_v<decltype(std::declval<const MonomialPropagator<8> &>().indexing()),
                             const monoprop::detail::OperatorIndex<8> &>);

BOOST_FIXTURE_TEST_CASE(copy_constructed_simulator_matches_energy, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    auto copy = sim;

    BOOST_TEST(copy.graph_layers() == sim.graph_layers());
    BOOST_TEST(copy.size() == sim.size());

    const double e_orig = sim.expectation_value_functional()(data.parameters);
    const double e_copy = copy.expectation_value_functional()(data.parameters);
    BOOST_CHECK_SMALL(e_orig - e_copy, 1e-13);
}

BOOST_FIXTURE_TEST_CASE(copy_is_independent_of_source, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);

    auto copy = sim;
    BOOST_TEST(copy.graph_layers() == 0u);

    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_TEST(sim.graph_layers() > 0u);
    BOOST_TEST(copy.graph_layers() == 0u);

    copy.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double e_sim = sim.expectation_value_functional()(data.parameters);
    const double e_copy = copy.expectation_value_functional()(data.parameters);
    BOOST_CHECK_SMALL(e_sim - e_copy, 1e-13);
}

// The layer list is per-instance (vector<Layer>); the immutable LayerCores are shared via shared_ptr.
// Contracting one copy in place truncates only its own layer list, and destroying it only drops its
// core references.
BOOST_FIXTURE_TEST_CASE(copy_graph_survives_other_being_contracted_and_destroyed, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto original = build_simulator<n_modes>(data, cfg);
    original.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const size_t layers_before = original.graph_layers();
    BOOST_TEST(layers_before > 0u);
    const double e_before = original.expectation_value_functional()(data.parameters);

    {
        auto copy = original;
        BOOST_TEST(copy.graph_layers() == layers_before);

        copy.contract_partially(data.parameters, /*inplace=*/true);
        BOOST_TEST(copy.graph_layers() < layers_before);
        BOOST_TEST(original.graph_layers() == layers_before);
    }

    BOOST_TEST(original.graph_layers() == layers_before);
    const double e_after = original.expectation_value_functional()(data.parameters);
    BOOST_CHECK_SMALL(e_before - e_after, 1e-13);
}

BOOST_FIXTURE_TEST_CASE(copy_constructed_simulator_index_valid, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    auto copy = sim;

    const auto &idx = copy.indexing();
    BOOST_TEST(idx.size() == sim.indexing().size());
    bool all_found = true;
    idx.for_each([&](const auto &mono, size_t i) {
        const auto f = idx.find(mono);
        if (!f || *f != i) {
            all_found = false;
        }
    });
    BOOST_TEST(all_found);
}
