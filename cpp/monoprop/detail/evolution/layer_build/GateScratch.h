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

// Per-partition scratch the layer build reuses from gate to gate. Nothing here carries state between
// gates -- every member is re-initialised by the gate that uses it -- so it is owned by the propagator
// only to keep its capacity, and a copied propagator starts with an empty one.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/detail/evolution/layer_build/AntiTable.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

template <size_t NumModes>
struct GateScratch {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    AntiTable<NumModes> anti;  // this gate's anticommuting rows, keyed by fingerprint
    std::vector<PosT> partner; // one partner's positions; sized 2*NumModes, the partner's upper bound
    std::vector<uint16_t> gen; // the generator's ascending positions, the merge's second input

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return anti.memory_bytes() + (partner.capacity() * sizeof(PosT)) + (gen.capacity() * sizeof(uint16_t));
    }
};

} // namespace monoprop::detail
