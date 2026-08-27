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

namespace monoprop::detail {

/// Bytes a container has taken from the allocator. Variadic so a roll-up over many members is one call.
template <typename... Vecs>
[[nodiscard]] inline auto capacity_bytes(const Vecs &...vecs) -> size_t {
    return (0uz + ... + (vecs.capacity() * sizeof(typename Vecs::value_type)));
}

/// Of capacity_bytes(): reserved and never written. Unfaulted, but not free -- a growth holds old+new at once.
template <typename... Vecs>
[[nodiscard]] inline auto capacity_slack_bytes(const Vecs &...vecs) -> size_t {
    return (0uz + ... + ((vecs.capacity() - vecs.size()) * sizeof(typename Vecs::value_type)));
}

} // namespace monoprop::detail
