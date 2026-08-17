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

// Deliberately dependency-free: exported headers (MPGraph.h, MPFunctions.h, MonomialPropagator.h)
// carry a Picture in their signatures. The behaviour behind each value lives in picture/Picture.h.

#include <cstdint>
#include <variant>

namespace monoprop {

// Which object the circuit propagates. Fixed at propagator construction; no path switches it later.
enum class Picture : uint8_t { Heisenberg, Schrodinger };

// The picture selector as a constructor argument. Only the Schrödinger arm carries a cutoff, so a
// Heisenberg run cannot be given one, and a Schrödinger run cannot omit it.
struct Heisenberg {}; // propagate the observable backwards; the reference state is held fixed

struct Schrodinger {
    unsigned int state_cutoff; // bounds the monomial expansion of the state, read like `cutoff`
};

using PictureSpec = std::variant<Heisenberg, Schrodinger>;

inline auto kind_of(const PictureSpec &spec) -> Picture {
    return std::holds_alternative<Schrodinger>(spec) ? Picture::Schrodinger : Picture::Heisenberg;
}

} // namespace monoprop
