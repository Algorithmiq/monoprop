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

#include <cstdint>
#include <type_traits>
#include <vector>

namespace monoprop::detail {

// One result per partition, indexed by partition rank, written concurrently from the partitions' own
// masters. `fan_out(emit)` runs `emit(rank, value)` once per rank, on whichever thread owns that rank.
//
// The staging vector exists for the bool case and only for it: std::vector<bool> is the bit-packed
// specialization, so concurrent writes to different logical elements can tear the same underlying word
// -- a data race even though the indices are disjoint. Everything else is written in place and moved out.
//
// Its own header, with no dependency beyond <vector>, because both fan-out paths need it and they must
// not see each other: MonomialPropagator reaches its partitions through a type-erased primitive
// precisely so the public header never sees PartitionGroup.
template <typename R, typename FanOut>
auto staged_collect(size_t n, FanOut &&fan_out) -> std::vector<R> {
    using Slot = std::conditional_t<std::is_same_v<R, bool>, std::uint8_t, R>;
    std::vector<Slot> staging(n);
    fan_out([&staging](int r, R value) { staging[static_cast<size_t>(r)] = static_cast<Slot>(value); });
    if constexpr (std::is_same_v<R, bool>) {
        return std::vector<R>(staging.begin(), staging.end());
    }
    else {
        return staging;
    }
}

} // namespace monoprop::detail
