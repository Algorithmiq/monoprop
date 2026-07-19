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

// ─── apply_fused_contract ─────────────────────────────────────────────────────
// The drain paired with build_layer's fused emission: complete each rotation by adding its sine term
// directly to op_coeffs — the ContractImmediately forward path at ALL rank counts, replacing the
// transient LayerCore + evolve_step. The gate's cosine scale reaches the coefficients one of two ways:
//   • fused_scale (k==0, the default): the scan already multiplied every anticommuting coefficient in
//     place during its own pass (see fused_find_and_collect), so no cos pass runs here and `cos` is
//     empty/ignored. Slots born AFTER that sweep — fresh inserts (self-rank misses and cross-rank
//     resolver misses) — get the cos folded in by their apply arm below (c = cos·c + sin term), which
//     is exactly the two-pass path's scale-then-add on those slots.
//   • two-pass (k>0 / cos==0 fallback): the eager kernel `scale_cos_mask` runs here over the build's
//     cos set (inserted endpoints included via append_inserted_endpoints_), then every arm is a plain
//     add — byte-for-byte the historical path.
// Same FP expression shape as evolve_step's D-apply (sin_val * static_cast<double>(phi) * value); the
// pre-cos v_src/v_tgt values are scan-captured / 1/cos-recovered (see Engine.h). `param` is the
// SCHRODINGER-SIGNED value the non-fused evolve_step receives; cos is even so cos(2·param) equals the
// sweep's cos(2·build_angle) bit-for-bit. At R>1 a rotation's two endpoints can live on different
// ranks: each rank applies only the ADD to the slot it owns (half rotations in fc.cross_half), with
// the partner coeff already carried over the wire during the build exchange — no MPI of its own. Not
// templated on NumModes — it operates purely on the resolved FusedContract + op_coeffs.
inline auto apply_fused_contract(FusedContract &fc,
                                 VecD &op_coeffs,
                                 const CosMask &cos,
                                 double param,
                                 bool schrodinger,
                                 bool fused_scale) -> void {
    // (1) INSERT records: v_tgt is the freshly-inserted term's PRE-cos coeff, available only now that
    // op_coeffs has been extended (insert slots are never swept by the scan, so this read is pre-cos in
    // both modes). Needed ONLY in the Schrödinger picture — Heisenberg fresh inserts have coeff 0
    // (extend zero-fills the appended tail), so v_tgt stays its initialized 0.0 and the c[src] += …·v_tgt
    // add is a no-op; skip the gather entirely. Parallel over the distinct insert targets.
    if (schrodinger) {
        for (size_t k = 0; k < fc.inserts.size(); ++k) {
            fc.inserts[k].v_tgt = op_coeffs[fc.inserts[k].tgt];
        }
    }

    // (2) Two-pass mode only: cos scale over ALL anticommuting endpoints (inserts included) — the
    // kernel identical to the non-fused cos_scale callback. In fused_scale mode the scan already did
    // this in its own coefficient pass.
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);
    double *const c = op_coeffs.data();
    if (!fused_scale) {
        profiling::ScopedRegion prof_cs(profiling::Region::CosScale);
        scale_cos_mask(c, cos, cos_val);
    }

    // (3) One parallel apply over hits ++ inserts ++ cross_half. Each op slot is touched by exactly one
    // add (single-touch invariant: pivot leader/follower split + ⊕G-injective targets +
    // drop_matched_cross_rank_followers), so the apply is order-free and thread-count
    // invariant. Full rotations (hits/inserts) write BOTH local endpoints; half rotations (cross_half —
    // resolver +φ and querier −φ) write only the single slot THIS rank owns, exactly like
    // Evolution.cpp's cross-rank D-apply. In fused_scale mode a slot born after the sweep (insert
    // full-rotation targets, resolver MISS halves) folds the gate's cos in here — c = cos·c + sin
    // term — reproducing scale-then-add while preserving any nonzero post-extension value (Schrödinger
    // HF score, or a pending initial-operator term drained into a fresh Heisenberg slot by the extend).
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
