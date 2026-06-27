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

#include "monoprop/MPFunctions.h"

namespace monoprop::masked_execution_plan_detail {

auto build_masked_execution_plan(const VecZ &nonzero_inds,
                                 size_t local_index_count,
                                 const MPGraph &graph,
                                 bool schrodinger,
                                 MPI_Comm comm) -> MPExecutionPlan;

} // namespace monoprop::masked_execution_plan_detail
