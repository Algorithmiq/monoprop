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
#include <utility>

#include "monoprop/Bitset.h"

namespace test_utils {

// Runs `body` once per compile-time width in [1, Bitset::kInlineWords], i.e. 32..256 storage modes.
//
// The regime, not the cap: monoprop_NARROW_KERNEL_MAX_WORDS decides which widths the scan *selects* a
// specialized kernel for, but the kernel and WordKernel are correct over the whole inline range, and
// raising the cap must not be what discovers otherwise. Anything above kInlineWords is excluded
// because the words spill to the heap there, which the word kernels forbid.
//
// One definition for every width sweep: the range is a contract two test files assert against, and a
// second copy of the fold could be narrowed in one of them without the other noticing.
template <size_t... Ws>
auto for_each_inline_width(std::index_sequence<Ws...>, auto &&body) -> void {
    // +1 because W == 0 is not a width any word kernel accepts.
    (body(std::integral_constant<size_t, Ws + 1>{}), ...);
}

auto for_each_inline_width(auto &&body) -> void {
    for_each_inline_width(std::make_index_sequence<monoprop::Bitset::kInlineWords>{}, body);
}

} // namespace test_utils
