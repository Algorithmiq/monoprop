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

#include <cmath>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/Common.h"

namespace monoprop::detail {

// The drain paired with build_layer's fused emission: complete each rotation by adding its half to
// op_coeffs (the ContractImmediately forward path at all rank counts). The gate's cosine scale reaches
// the coefficients two ways:
//   • fused_scale (no length cap, default): the scan already scaled every anticommuting coeff, so no cos
//     pass runs here; slots born after that sweep (mints) fold cos in via their insert arm below.
//   • two-pass (length cap / cos==0 fallback): scale_cos_mask runs here, then every half is a plain add.
inline auto apply_fused_contract(FusedContract &fc, VecD &op_coeffs, const CosMask &cos, double param, bool fused_scale)
    -> void {
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);
    double *const c = op_coeffs.data();
    if (!fused_scale) {
        scale_cos_mask(c, cos, cos_val);
    }
    // Each op slot is touched by exactly one half (one partner per term, ⊕G-injective mints), which is
    // what makes the plain += safe in any order.
    for (const HalfRotationRec &h : fc.halves) {
        if (fused_scale && h.is_insert) {
            c[h.local_idx] = cos_val * c[h.local_idx] + sin_val * static_cast<double>(h.phase_signed) * h.v_partner;
        }
        else {
            c[h.local_idx] += sin_val * static_cast<double>(h.phase_signed) * h.v_partner;
        }
    }
}

} // namespace monoprop::detail
