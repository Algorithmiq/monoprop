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

// Project-wide type aliases only. Anything with behaviour belongs in the header that owns the concept —
// the row-store accessors in detail/operator/RowAccess.h, the allocator in DefaultInitAllocator.h.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "monoprop/DefaultInitAllocator.h"

namespace monoprop {

using VecCD = std::vector<std::complex<double>>;

using VecD = std::vector<double>;

using VecI = std::vector<int>;

using VecZ = std::vector<size_t>;

// One partition's term index. Fixed at 32 bits: 2^32 terms in a single partition would need
// hundreds of GiB of resident operator, so the ceiling is reached by under-partitioning, never by
// problem size; check_index_fits() and checked_term_index() enforce it.
using TermIndex = std::uint32_t;

// resize() leaves new trivial elements uninitialized; see the default_init_allocator caveat.
template <typename T>
using DefaultInitVector = std::vector<T, default_init_allocator<T>>;

// The keys are Majorana indices or the native symplectic slots of a Pauli string (not its JW
// image), per the runtime Basis.
using OperatorDict = std::map<VecZ, std::complex<double>>;

} // namespace monoprop
