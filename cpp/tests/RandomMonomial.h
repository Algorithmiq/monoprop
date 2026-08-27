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
#include <random>

#include "monoprop/Bitset.h"

namespace test_utils {

// A random monomial over `num_modes` modes occupying at most `max_slots` of them, each with a uniformly
// random non-empty code (one Majorana of the mode, the other, or the pair).
//
// One definition, deliberately: this is the input distribution of the whole randomized sparse/codes test
// surface, and what it biases toward -- paired slots in particular, which is what drives spills and
// product overflow -- decides what those tests actually cover. A per-file copy would let one of them be
// tuned and the rest silently left behind. Kept out of TestUtilities.h, which pulls in Boost.Test,
// MonomialPropagator and MPI; the files that want this want nothing else.
inline auto random_monomial(std::mt19937_64 &rng, size_t num_modes, size_t max_slots) -> monoprop::Bitset {
    monoprop::Bitset mono(2 * num_modes);
    const size_t occupied = rng() % (max_slots + 1);
    for (size_t k = 0; k < occupied; ++k) {
        const size_t mode = rng() % num_modes;
        const auto code = 1U + static_cast<unsigned int>(rng() % 3U);
        if ((code & 1U) != 0U) {
            mono.set(2 * mode);
        }
        if ((code & 2U) != 0U) {
            mono.set((2 * mode) + 1);
        }
    }
    return mono;
}

} // namespace test_utils
