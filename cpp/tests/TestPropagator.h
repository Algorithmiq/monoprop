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

#include <optional>
#include <vector>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace test_utils {
using namespace monoprop;

// Every propagator in this suite is built through here, so a change to the ctor's argument list is one
// edit rather than 26.
//
// num_modes is the system's width; the propagator rounds it up to a whole 32-mode block to get the width
// it stores monomials at. That width is not cosmetic -- indices are laid out MSb0, so it moves every bit
// position, which changes each monomial's hash, which changes owner routing, probe order, and therefore
// the order coefficients accumulate in. Expected values here are pinned to the rounded width.
inline auto make_propagator(size_t num_modes,
                            const OperatorDict& initial_operator,
                            unsigned int cutoff,
                            const VecZ& initial_state,
                            std::optional<unsigned int> schrodinger_cutoff = std::nullopt,
                            mpi::Comm comm = MPI_COMM_SELF,
                            std::optional<double> lower_atol = std::nullopt,
                            std::optional<double> upper_atol = std::nullopt,
                            CutoffType cutoff_type = CutoffType::Length,
                            std::optional<std::vector<VecZ>> basis_change = std::nullopt,
                            Basis basis = Basis::Majorana,
                            size_t partitions = 0) -> MonomialPropagator {
    return MonomialPropagator(initial_operator,
                              cutoff,
                              initial_state,
                              num_modes,
                              schrodinger_cutoff,
                              comm,
                              lower_atol,
                              upper_atol,
                              cutoff_type,
                              basis_change,
                              basis,
                              partitions);
}
} // namespace test_utils
