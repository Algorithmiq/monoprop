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

// The independent reference for a layer's exchange layout, which derive_exchange_layout must match
// elementwise: counts[r] = scale * send_counts[r], displacements their prefix sum.
// Outside the library on purpose: an oracle refactored beside its subject asserts nothing.

#include <cstddef>
#include <format>
#include <string>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"

namespace exchange_layout_oracle {

inline auto build_layer_exchange_layout(const std::vector<size_t> &send_counts,
                                        int scale,
                                        const char *what = "Layer exchange") -> monoprop::LayerExchangeLayout {
    const std::string count_label = std::format("{} count", what);
    const std::string displacement_label = std::format("{} displacement", what);

    monoprop::LayerExchangeLayout layout;
    layout.counts.resize(send_counts.size());
    layout.displs.resize(send_counts.size());
    size_t total = 0;
    for (size_t r = 0; r < send_counts.size(); ++r) {
        const size_t count = static_cast<size_t>(scale) * send_counts[r];
        layout.counts[r] = monoprop::detail::checked_mpi_int(count, count_label.c_str());
        layout.displs[r] = monoprop::detail::checked_mpi_int(total, displacement_label.c_str());
        total += count;
    }
    layout.total_count = total;
    return layout;
}

} // namespace exchange_layout_oracle
