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

// Every propagator in this suite is built through here, and the reason is the storage width.
//
// The propagator used to be a class template storing monomials at exactly 2*NumModes bits; it now
// rounds a logical width up to a whole 32-mode block by default. That width is not
// cosmetic: indices are laid out MSb0, so widening moves every bit position, which changes each
// monomial's hash, which changes owner routing, probe order, and therefore the order coefficients
// accumulate in. The expected values in this suite were computed at the unrounded width, so tests pin
// storage_num_modes to their own NumModes instead of accepting the rounding. Production code (and the
// Python front-end) takes the default -- the rounding is what keeps the hash index's probe layout
// aligned across nearby system sizes.
//
// The argument order is the pre-refactor one, with logical_num_modes still in its old late position so
// the ctor-validation tests can override it on its own.
template <size_t NumModes>
inline auto make_propagator(const OperatorDict& initial_operator,
                            unsigned int cutoff,
                            const VecZ& initial_state,
                            std::optional<unsigned int> schrodinger_cutoff = std::nullopt,
                            mpi::Comm comm = MPI_COMM_SELF,
                            std::optional<double> lower_atol = std::nullopt,
                            std::optional<double> upper_atol = std::nullopt,
                            CutoffType cutoff_type = CutoffType::Length,
                            std::optional<std::vector<VecZ>> basis_change = std::nullopt,
                            size_t logical_num_modes = NumModes,
                            Basis basis = Basis::Majorana,
                            size_t partitions = 0) -> MonomialPropagator {
    return MonomialPropagator(initial_operator,
                              cutoff,
                              initial_state,
                              logical_num_modes,
                              schrodinger_cutoff,
                              comm,
                              lower_atol,
                              upper_atol,
                              cutoff_type,
                              basis_change,
                              basis,
                              partitions,
                              /*storage_num_modes=*/NumModes);
}
} // namespace test_utils
