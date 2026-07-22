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
#include "monoprop/detail/evolution/CosineRecompute.h"    // scale_cos_mask, CosMask
#include "monoprop/detail/evolution/layer_build/Common.h" // FusedContract, RotationRec
#include "monoprop/detail/profiling/RegionProfiler.h"

namespace monoprop::detail {

// The drain paired with build_layer's fused emission: complete each rotation by adding its sine term
// directly to op_coeffs (the ContractImmediately forward path at ALL rank counts). The gate's cosine
// scale reaches the coefficients two ways:
//   • fused_scale (k==0, default): the scan already scaled every anticommuting coeff in its own pass, so
//     no cos pass runs; slots born AFTER that sweep (fresh inserts) fold cos in via their apply arm below.
//   • two-pass (k>0 / cos==0 fallback): scale_cos_mask runs here over the build's cos set, then every arm
//     is a plain add — byte-for-byte the historical path.
// Same FP shape as evolve_step's D-apply. At R>1 each rank applies only the ADD to the slot it owns (half
// rotations in fc.cross_half), the partner coeff already carried over the wire. Not templated on NumModes.
inline auto apply_fused_contract(FusedContract &fc,
                                 VecD &op_coeffs,
                                 const CosMask &cos,
                                 double param,
                                 bool schrodinger,
                                 bool fused_scale) -> void {
    // (1) INSERT records: v_tgt is the freshly-inserted term's PRE-cos coeff, readable only now op_coeffs
    // is extended. Needed ONLY in Schrödinger — a Heisenberg fresh insert has coeff 0, so v_tgt stays 0.0
    // and the c[src] add is a no-op; skip the gather.
    if (schrodinger) {
        for (size_t k = 0; k < fc.inserts.size(); ++k) {
            fc.inserts[k].v_tgt = op_coeffs[fc.inserts[k].tgt];
        }
    }

    // (2) Two-pass mode only: cos scale over ALL anticommuting endpoints (inserts included). In fused_scale
    // mode the scan already did this in its own coefficient pass.
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);
    double *const c = op_coeffs.data();
    if (!fused_scale) {
        profiling::ScopedRegion prof_cs(profiling::Region::CosScale);
        scale_cos_mask(c, cos, cos_val);
    }

    // (3) One parallel apply over hits ++ inserts ++ cross_half. Each op slot is touched by exactly one add
    // (single-touch invariant: pivot split + ⊕G-injective targets + drop_matched_cross_rank_followers), so
    // the apply is order-free and thread-count invariant. Full rotations write BOTH local endpoints; half
    // rotations write only the slot THIS rank owns. In fused_scale mode a slot born after the sweep (insert
    // targets, resolver MISS halves) folds the gate's cos in here (c = cos·c + sin), preserving any nonzero
    // post-extension value.
    const size_t n_hit = fc.hits.size();
    const size_t n_full = n_hit + fc.inserts.size();
    const size_t n_cross = fc.cross_half.size();
    profiling::ScopedRegion prof_fa(profiling::Region::FusedApply);
    for (size_t k = 0; k < n_full + n_cross; ++k) {
        if (k < n_full) {
            const bool is_insert = k >= n_hit;
            const RotationRec &r = is_insert ? fc.inserts[k - n_hit] : fc.hits[k];
            c[r.src] += sin_val * static_cast<double>(-r.phase) * r.v_tgt;
            if (fused_scale && is_insert) {
                c[r.tgt] = cos_val * c[r.tgt] + sin_val * static_cast<double>(r.phase) * r.v_src;
            }
            else {
                c[r.tgt] += sin_val * static_cast<double>(r.phase) * r.v_src;
            }
        }
        else {
            // Cross-rank half rotations (R>1): add the wire-carried partner term to the one slot this
            // rank owns; resolver MISS halves (fresh inserts, unswept) fold the cos in first.
            const HalfRotationRec &h = fc.cross_half[k - n_full];
            if (fused_scale && h.is_insert) {
                c[h.local_idx] = cos_val * c[h.local_idx] + sin_val * static_cast<double>(h.phase_signed) * h.v_partner;
            }
            else {
                c[h.local_idx] += sin_val * static_cast<double>(h.phase_signed) * h.v_partner;
            }
        }
    }
}

} // namespace monoprop::detail
