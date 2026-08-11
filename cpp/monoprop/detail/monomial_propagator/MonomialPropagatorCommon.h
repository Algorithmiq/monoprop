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

#include <cassert>
#include <stdexcept>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/core/Monomial.h"

namespace monoprop::detail {
class UnknownCutoffTypeError : public std::runtime_error {
public:
    UnknownCutoffTypeError() : std::runtime_error("Unknown cutoff type") {}
};

// logical_num_modes and num_bits are both required: they used to arrive free from NumModes. num_bits is
// the storage width the functor's masks are built for, and it has to match the monomials the functor
// will be handed -- asserted in cutoff_sums.
//
// The stored type is load-bearing, not just an implementation choice. CutoffEvaluator recovers the
// concrete functor with std::function::target<T>(), which matches on the *exact* type -- so wrapping
// either functor in a lambda, or storing a structurally identical but distinct type, would disengage
// the scan's fast paths while still computing the right answer. Nothing downstream can detect that:
// the results are unchanged, so the tests pass and the bit-identity check stays green, and only a
// benchmark would notice. The asserts below pin the handshake here, where the type is chosen.
inline auto cutoff_function(CutoffType cutoff_type, unsigned int cutoff, size_t logical_num_modes, size_t num_bits)
    -> CutoffFn {
    switch (cutoff_type) {
        case CutoffType::Length: {
            CutoffFn fn = LengthCutoff{cutoff, logical_num_modes, num_bits};
            assert(fn.target<LengthCutoff>() != nullptr && "CutoffEvaluator's length fast path would not engage");
            return fn;
        }
        case CutoffType::Support: {
            CutoffFn fn = SupportCutoff{cutoff, logical_num_modes, num_bits};
            assert(fn.target<SupportCutoff>() != nullptr && "CutoffEvaluator's support fast path would not engage");
            return fn;
        }
        default:
            throw UnknownCutoffTypeError();
    }
}

// Lambdas by design, so CutoffEvaluator's target<>() probes find nothing and it calls through the
// std::function. That is the intended behaviour here, not the silent miss described above: the
// predicate is length/support applied to the *mapped* monomial, so the fast paths -- which read a raw
// popcount and a bare cutoff -- do not apply.
inline auto cutoff_function_basis_change(CutoffType cutoff_type,
                                         unsigned int cutoff,
                                         const MonomialList &basis,
                                         size_t logical_num_modes) -> CutoffFn {
    switch (cutoff_type) {
        case CutoffType::Length:
            return [cutoff, logical_num_modes, basis_copy = basis](const Bitset &mono) {
                const auto mapped_mono = change_basis(mono, basis_copy);
                return length_cutoff(mapped_mono, cutoff, logical_num_modes);
            };
        case CutoffType::Support:
            return [cutoff, logical_num_modes, basis_copy = basis](const Bitset &mono) {
                const auto mapped_mono = change_basis(mono, basis_copy);
                return support_cutoff(mapped_mono, cutoff, logical_num_modes);
            };
        default:
            throw UnknownCutoffTypeError();
    }
}
} // namespace monoprop::detail
