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
#include <vector>

#include "monoprop/MPFunctions.h"

namespace monoprop::masked_execution_plan_detail {

struct BuilderExchangeLayout final {
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    size_t total_send = 0;
    size_t total_recv = 0;
};

struct LayerPlanFilterResult final {
    CompressedCosineData masked_cos_data;
    CompressedPositionData local_cycle_positions;
    CompressedPositionData cross_rank_out_positions;
    CompressedPositionData cross_rank_in_positions;
    std::vector<detail::CrossRankMaskRange> cross_rank_ranges;
    bool preserves_cosine_data = true;
    bool preserves_local_cycles = true;
    bool preserves_cross_rank = true;
};

auto filter_layer_execution_plan(const Layer &layer,
                                 std::vector<char> &nodes_to_keep,
                                 bool has_remote_cross_rank,
                                 const BuilderExchangeLayout &source_keep_layout,
                                 const VecI &remote_src_keep,
                                 size_t my_rank,
                                 const BuilderExchangeLayout *selection_layout = nullptr,
                                 VecI *selected_incoming_flags = nullptr) -> LayerPlanFilterResult;

} // namespace monoprop::masked_execution_plan_detail
