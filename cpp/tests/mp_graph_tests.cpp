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

// replay_view() is the evaluation order: stored order, so the growth end shows through it.
BOOST_AUTO_TEST_CASE(mp_graph_replay_view_is_stored_order) {
    auto back = graph_with_gates(LayerGrowth::Back, 4);   // layers_ = [0,1,2,3]
    auto front = graph_with_gates(LayerGrowth::Front, 4); // layers_ = [3,2,1,0]
    const auto back_view = back.replay_view();
    const auto front_view = front.replay_view();
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(back_view.get_layer_traversal(i).gate_index(), i);
        BOOST_CHECK_EQUAL(front_view.get_layer_traversal(i).gate_index(), 3U - i);
    }
}

// contraction_view() is the build order, so it yields the same sequence under either growth end. That
// equality is the point: a contraction replays the circuit the way the build walked it.
BOOST_AUTO_TEST_CASE(mp_graph_contraction_view_is_build_order_under_either_growth) {
    auto back = graph_with_gates(LayerGrowth::Back, 4);
    auto front = graph_with_gates(LayerGrowth::Front, 4);
    const auto back_view = back.contraction_view();
    const auto front_view = front.contraction_view();
    BOOST_REQUIRE_EQUAL(back_view.layers(), 4U);
    BOOST_REQUIRE_EQUAL(front_view.layers(), 4U);
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(back_view.get_layer_traversal(i).gate_index(), i);
        BOOST_CHECK_EQUAL(front_view.get_layer_traversal(i).gate_index(), i);
    }
}

BOOST_AUTO_TEST_CASE(mp_graph_clear_empties_and_leaves_the_graph_usable) {
    auto graph = graph_with_gates(LayerGrowth::Front, 5);
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

    const MPGraphView fwd(layers, /*base=*/0, /*count=*/4, /*reverse=*/false);
    const MPGraphView rev(layers, /*base=*/0, /*count=*/4, /*reverse=*/true);
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(fwd.get_layer_traversal(i).gate_index(), 10U + i);
        BOOST_CHECK_EQUAL(rev.get_layer_traversal(i).gate_index(), 13U - i);
    }
    BOOST_CHECK_THROW(fwd.get_layer(4), std::out_of_range);
    BOOST_CHECK_THROW(rev.get_layer(4), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(mp_graph_get_layer_out_of_range_throws) {
    auto graph = graph_with_gates(LayerGrowth::Back, 3);
    BOOST_CHECK_NO_THROW((void)graph.get_layer(2));
    BOOST_CHECK_THROW((void)graph.get_layer(3), std::out_of_range);
    // const overload takes the same guard.
    const auto &cref = graph;
    BOOST_CHECK_THROW((void)cref.get_layer(3), std::out_of_range);
}
