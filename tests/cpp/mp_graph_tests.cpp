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

// White-box tests for MPGraph transforms and MPGraphView, built by direct Layer construction
// (GraphBuildHarness). Each layer's distinct gate_index is the oracle for slice / view ordering.

#include <boost/test/unit_test.hpp>

#include <stdexcept>
#include <vector>

#include "GraphBuildHarness.h"
#include "monoprop/MPGraph.h"

using namespace monoprop;
using test_utils::core_with_gate;
using test_utils::graph_with_gates;
using test_utils::layer_with_gate;

// ── slice_graph ────────────────────────────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(mp_graph_slice_graph_heisenberg_prefix_no_contract) {
    auto graph = graph_with_gates(/*schrodinger=*/false, 5); // layers_ = [0,1,2,3,4]
    auto sliced = graph.slice_graph(3, /*contract=*/false);

    BOOST_REQUIRE_EQUAL(sliced.layers(), 3U);
    BOOST_CHECK_EQUAL(sliced.get_layer_traversal(0).gate_index(), 0U);
    BOOST_CHECK_EQUAL(sliced.get_layer_traversal(1).gate_index(), 1U);
    BOOST_CHECK_EQUAL(sliced.get_layer_traversal(2).gate_index(), 2U);
    // Non-contracting slice leaves the source untouched.
    BOOST_CHECK_EQUAL(graph.layers(), 5U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(0).gate_index(), 0U);
}

BOOST_AUTO_TEST_CASE(mp_graph_slice_graph_schrodinger_contract_newest_first_copy_and_resize) {
    // Schrödinger stores newest-first: appending gates 0..4 gives layers_ = [4,3,2,1,0].
    auto graph = graph_with_gates(/*schrodinger=*/true, 5);
    // Slice the 2 earliest operations (gates 0,1) with contract -> they leave the source.
    auto sliced = graph.slice_graph(2, /*contract=*/true);

    // sliced = layers_[active_end-1-i] = layers_[4], layers_[3] = gates 0, 1 (oldest-first).
    BOOST_REQUIRE_EQUAL(sliced.layers(), 2U);
    BOOST_CHECK_EQUAL(sliced.get_layer_traversal(0).gate_index(), 0U);
    BOOST_CHECK_EQUAL(sliced.get_layer_traversal(1).gate_index(), 1U);

    // Contract resized layers_ to the newest 3 (gates 4,3,2, still newest-first).
    BOOST_REQUIRE_EQUAL(graph.layers(), 3U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(0).gate_index(), 4U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(1).gate_index(), 3U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(2).gate_index(), 2U);
}

BOOST_AUTO_TEST_CASE(mp_graph_slice_graph_key_clamped_to_size) {
    auto graph = graph_with_gates(/*schrodinger=*/false, 3);
    auto sliced = graph.slice_graph(100, /*contract=*/false); // key > layers() clamps to 3
    BOOST_CHECK_EQUAL(sliced.layers(), 3U);
}

// ── maybe_compact_layers arms, reached through Heisenberg slice_graph(contract=true) ─────────────

BOOST_AUTO_TEST_CASE(mp_graph_contract_clear_arm_when_prefix_covers_all) {
    auto graph = graph_with_gates(/*schrodinger=*/false, 5);
    (void)graph.slice_graph(5, /*contract=*/true); // front_offset == size -> clear
    BOOST_CHECK_EQUAL(graph.layers(), 0U);
    // Graph is still usable after a full clear.
    graph.append(std::make_shared<LayerCore>(), 0, 0.0, /*gate_index=*/42);
    BOOST_REQUIRE_EQUAL(graph.layers(), 1U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(0).gate_index(), 42U);
}

BOOST_AUTO_TEST_CASE(mp_graph_contract_noop_arm_keeps_dead_prefix_lazy) {
    auto graph = graph_with_gates(/*schrodinger=*/false, 100);
    (void)graph.slice_graph(3, /*contract=*/true); // front_offset 3 < 4096 -> no physical compaction
    BOOST_REQUIRE_EQUAL(graph.layers(), 97U);
    // Active window now starts at the 4th gate.
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(0).gate_index(), 3U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(96).gate_index(), 99U);
}

BOOST_AUTO_TEST_CASE(mp_graph_contract_erase_arm_above_threshold) {
    // The erase arm fires only when front_offset >= 4096 AND 2*front_offset >= size.
    auto graph = graph_with_gates(/*schrodinger=*/false, 8200);
    auto sliced = graph.slice_graph(4100, /*contract=*/true); // 4100 >= 4096 and 8200 >= 8200 -> erase
    BOOST_CHECK_EQUAL(sliced.layers(), 4100U);
    BOOST_CHECK_EQUAL(sliced.get_layer_traversal(0).gate_index(), 0U);

    BOOST_REQUIRE_EQUAL(graph.layers(), 4100U);
    // After the physical erase the dead prefix is gone; index 0 is the first surviving gate.
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(0).gate_index(), 4100U);
    BOOST_CHECK_EQUAL(graph.get_layer_traversal(4099).gate_index(), 8199U);
}

// ── slice_view (through MPGraph) ─────────────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(mp_graph_slice_view_heisenberg_forward_window) {
    auto graph = graph_with_gates(/*schrodinger=*/false, 5);
    auto view = graph.slice_view(3);
    BOOST_REQUIRE_EQUAL(view.layers(), 3U);
    BOOST_CHECK_EQUAL(view.get_layer_traversal(0).gate_index(), 0U);
    BOOST_CHECK_EQUAL(view.get_layer_traversal(1).gate_index(), 1U);
    BOOST_CHECK_EQUAL(view.get_layer_traversal(2).gate_index(), 2U);
}

BOOST_AUTO_TEST_CASE(mp_graph_slice_view_schrodinger_reversed_window) {
    // layers_ = [4,3,2,1,0]; slice_view(3) uses base=active_end-3=2, reverse=true.
    // get_layer_traversal(i) -> layers_[2 + (3-1-i)] -> gates 0,1,2 in replay order.
    auto graph = graph_with_gates(/*schrodinger=*/true, 5);
    auto view = graph.slice_view(3);
    BOOST_REQUIRE_EQUAL(view.layers(), 3U);
    BOOST_CHECK_EQUAL(view.get_layer_traversal(0).gate_index(), 0U);
    BOOST_CHECK_EQUAL(view.get_layer_traversal(1).gate_index(), 1U);
    BOOST_CHECK_EQUAL(view.get_layer_traversal(2).gate_index(), 2U);
}

// ── MPGraphView directly: the reverse mapping and the oob throw ──────────────────────────────────

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

// ── checked_layer_offset on the graph itself ─────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(mp_graph_get_layer_out_of_range_throws) {
    auto graph = graph_with_gates(/*schrodinger=*/false, 3);
    BOOST_CHECK_NO_THROW((void)graph.get_layer(2));
    BOOST_CHECK_THROW((void)graph.get_layer(3), std::out_of_range);
    // const overload takes the same guard.
    const auto &cref = graph;
    BOOST_CHECK_THROW((void)cref.get_layer(3), std::out_of_range);
}
