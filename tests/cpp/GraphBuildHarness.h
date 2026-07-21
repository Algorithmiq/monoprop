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

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "monoprop/MPGraph.h"

// Direct-construction helpers for white-box MPGraph/Layer tests. A layer's gate_index is used purely
// as a distinguishable tag so slice/view ordering can be asserted; the rest of the LayerCore is empty.
namespace test_utils {

inline auto core_with_gate(std::size_t gate_index) -> std::shared_ptr<monoprop::LayerCore> {
    auto core = std::make_shared<monoprop::LayerCore>();
    core->gate_index = gate_index;
    return core;
}

inline auto layer_with_gate(std::size_t gate_index) -> monoprop::Layer {
    return monoprop::Layer(core_with_gate(gate_index));
}

// An MPGraph of `n` layers with gate_index 0..n-1, built via append() so the picture's internal
// layer ordering (Heisenberg back-append, Schrödinger front-insert) is exactly as production builds it.
inline auto graph_with_gates(bool schrodinger, std::size_t n) -> monoprop::MPGraph {
    monoprop::MPGraph graph(schrodinger);
    for (std::size_t i = 0; i < n; ++i) {
        graph.append(std::make_shared<monoprop::LayerCore>(), /*param_index=*/0, /*gen_coeff=*/0.0, /*gate_index=*/i);
    }
    return graph;
}

} // namespace test_utils
