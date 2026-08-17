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

// Umbrella header for build_layer(); the implementation lives in the sibling layer_build/ headers.
// Pivot split: M and its partner M⊕G differ in every column of G including the pivot (G's lowest set
// column), so exactly one of the pair carries it — leader (pivot clear) vs follower (pivot set). Visiting
// leaders then the still-unmatched followers touches each pair once, no sort and no dedup.

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
