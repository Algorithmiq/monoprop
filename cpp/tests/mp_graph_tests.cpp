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

// White-box tests for MPGraph's views and MPGraphView, built by direct Layer construction
// (GraphBuildHarness). Each layer's distinct gate_index is the oracle for view ordering.

#include <boost/test/unit_test.hpp>

#include <stdexcept>
#include <vector>

#include "GraphBuildHarness.h"
#include "monoprop/MPGraph.h"

using namespace monoprop;
using test_utils::core_with_gate;
using test_utils::graph_with_gates;
using test_utils::layer_with_gate;

// graph_with_gates tags gate_index by arrival, so an AscendingSlot graph reads back in reverse. Every
// case below reads that way, which is the whole storage invariant in one line.
BOOST_AUTO_TEST_CASE(mp_graph_layer_index_reverses_an_ascending_slot_arrival) {
    auto descending = graph_with_gates(ArrivalOrder::DescendingSlot, 5);
    auto ascending = graph_with_gates(ArrivalOrder::AscendingSlot, 5);
    for (std::size_t i = 0; i < 5; ++i) {
        BOOST_CHECK_EQUAL(descending.get_layer_traversal(i).gate_index(), i);
        BOOST_CHECK_EQUAL(ascending.get_layer_traversal(i).gate_index(), 4U - i);
    }
    // Appending again keeps the mapping: the new layer is the last arrival either way.
    descending.append(std::make_shared<LayerCore>(), 0, 0.0, /*gate_index=*/5);
    ascending.append(std::make_shared<LayerCore>(), 0, 0.0, /*gate_index=*/5);
    BOOST_CHECK_EQUAL(descending.get_layer_traversal(5).gate_index(), 5U);
    BOOST_CHECK_EQUAL(ascending.get_layer_traversal(0).gate_index(), 5U);
}

// replay_view() is the evaluation order: layer order, so the arrival order shows through it.
BOOST_AUTO_TEST_CASE(mp_graph_replay_view_is_layer_order) {
    auto descending = graph_with_gates(ArrivalOrder::DescendingSlot, 4);
    auto ascending = graph_with_gates(ArrivalOrder::AscendingSlot, 4);
    const auto descending_view = descending.replay_view();
    const auto ascending_view = ascending.replay_view();
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(descending_view.get_layer_traversal(i).gate_index(), i);
        BOOST_CHECK_EQUAL(ascending_view.get_layer_traversal(i).gate_index(), 3U - i);
    }
}

// contraction_view() is the build order, so it yields the same sequence under either arrival order. That
// equality is the point: a contraction replays the circuit the way the build walked it.
BOOST_AUTO_TEST_CASE(mp_graph_contraction_view_is_build_order_under_either_arrival) {
    auto descending = graph_with_gates(ArrivalOrder::DescendingSlot, 4);
    auto ascending = graph_with_gates(ArrivalOrder::AscendingSlot, 4);
    const auto descending_view = descending.contraction_view();
    const auto ascending_view = ascending.contraction_view();
    BOOST_REQUIRE_EQUAL(descending_view.layers(), 4U);
    BOOST_REQUIRE_EQUAL(ascending_view.layers(), 4U);
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(descending_view.get_layer_traversal(i).gate_index(), i);
        BOOST_CHECK_EQUAL(ascending_view.get_layer_traversal(i).gate_index(), i);
    }
}

// replace_layer() addresses layers the way get_layer() does, which is what keeps pare_graph's sweep from
// writing its filtered layer onto the mirror-image one.
BOOST_AUTO_TEST_CASE(mp_graph_replace_layer_addresses_the_same_layer_as_get_layer) {
    for (const auto arrival : {ArrivalOrder::DescendingSlot, ArrivalOrder::AscendingSlot}) {
        auto graph = graph_with_gates(arrival, 4);
        graph.replace_layer(1, layer_with_gate(99));
        BOOST_CHECK_EQUAL(graph.get_layer_traversal(1).gate_index(), 99U);
        // The others are untouched, so nothing was written through the mirror index.
        BOOST_CHECK_NE(graph.get_layer_traversal(2).gate_index(), 99U);
        BOOST_CHECK_EQUAL(graph.layers(), 4U);
    }
}

BOOST_AUTO_TEST_CASE(mp_graph_clear_empties_and_leaves_the_graph_usable) {
    auto graph = graph_with_gates(ArrivalOrder::AscendingSlot, 5);
    graph.clear();
    BOOST_REQUIRE_EQUAL(graph.layers(), 0U);
    graph.append(std::make_shared<LayerCore>(), 0, 0.0, /*gate_index=*/7);
    BOOST_REQUIRE_EQUAL(graph.layers(), 1U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(0).gate_index(), 7U);
}

BOOST_AUTO_TEST_CASE(mp_graph_view_reverse_flag_flips_index_mapping) {
    std::vector<Layer> layers;
    for (std::size_t g = 10; g < 14; ++g) {
        layers.push_back(layer_with_gate(g)); // [10,11,12,13]
    }

    const MPGraphView fwd(layers, /*reverse=*/false);
    const MPGraphView rev(layers, /*reverse=*/true);
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(fwd.get_layer_traversal(i).gate_index(), 10U + i);
        BOOST_CHECK_EQUAL(rev.get_layer_traversal(i).gate_index(), 13U - i);
    }
    BOOST_CHECK_THROW(fwd.get_layer(4), std::out_of_range);
    BOOST_CHECK_THROW(rev.get_layer(4), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(mp_graph_get_layer_out_of_range_throws) {
    auto graph = graph_with_gates(ArrivalOrder::DescendingSlot, 3);
    BOOST_CHECK_NO_THROW((void)graph.get_layer(2));
    BOOST_CHECK_THROW((void)graph.get_layer(3), std::out_of_range);
    // const overload takes the same guard.
    const auto &cref = graph;
    BOOST_CHECK_THROW((void)cref.get_layer(3), std::out_of_range);
}
