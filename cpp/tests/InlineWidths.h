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

// Runs `body` once per compile-time width in [1, Bitset::kInlineWords] (32..256 inline modes).
//
// kNarrowKernelWords controls which widths pick a specialized scan kernel. The kernels themselves must
// work for the full inline range. We stop at kInlineWords because larger widths spill to heap storage,
// which word kernels do not allow.
//
// Keep this width sweep defined in one place. Multiple test files rely on the same range, and duplicating
// this fold could let one copy drift without notice.
template <size_t... Ws>
auto for_each_inline_width(std::index_sequence<Ws...>, auto &&body) -> void {
    // +1 because W == 0 is not a width any word kernel accepts.
    (body(std::integral_constant<size_t, Ws + 1>{}), ...);
}

auto for_each_inline_width(auto &&body) -> void {
    for_each_inline_width(std::make_index_sequence<monoprop::Bitset::kInlineWords>{},
                          std::forward<decltype(body)>(body));
}

} // namespace test_utils
