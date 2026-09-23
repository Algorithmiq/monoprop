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

#include <algorithm>
#include <cstddef>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h"

// A [P] vector-of-vectors facade over the WindowVec-only exchange verbs, so the transport tests can
// build and check whole-world block arrays.
namespace test_utils {

template <typename T>
auto full_window(const std::vector<std::vector<T>> &blocks) -> monoprop::mpi::WindowVec<std::vector<T>> {
    monoprop::mpi::WindowVec<std::vector<T>> w(monoprop::mpi::SlotWindow{.base = 0, .count = blocks.size()});
    std::ranges::copy(blocks, w.begin());
    return w;
}

// Slots outside the round's window come back empty.
template <typename T>
auto flat_blocks(const monoprop::mpi::WindowVec<std::vector<T>> &w, size_t world) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> out(world);
    for (const auto wi : w.window().indices()) {
        out[w.window().slot(wi)] = w[wi];
    }
    return out;
}

template <typename T>
auto exchange_blocks(const std::vector<std::vector<T>> &send,
                     monoprop::mpi::Comm c,
                     bool skip_self = false,
                     const std::vector<int> *known_recv_counts = nullptr,
                     monoprop::mpi::PeerPlan plan = {}) -> std::vector<std::vector<T>> {
    monoprop::mpi::WindowVec<std::vector<T>> got;
    monoprop::mpi::begin_alltoallv(full_window(send), c, skip_self, known_recv_counts, plan).wait_into(got);
    return flat_blocks(got, static_cast<size_t>(monoprop::mpi::size(c)));
}

} // namespace test_utils
