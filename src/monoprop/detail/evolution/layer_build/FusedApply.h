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

#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CoeffFrame.h"         // mag_byte (magnitude byte maintenance)
#include "monoprop/detail/evolution/CosineRecompute.h"    // scale_cos_mask, CosMask
#include "monoprop/detail/evolution/layer_build/Common.h" // FusedContract, RotationRec, FrameRefs
#include "monoprop/detail/profiling/RegionProfiler.h"

namespace monoprop::detail {

// ─── apply_fused_contract ─────────────────────────────────────────────────────
// The drain paired with build_layer's fused emission: cos-scale, then add each rotation's sine term
// directly to op_coeffs — the ContractImmediately forward path at ALL rank counts, replacing the
// transient LayerCore + evolve_step. Byte-for-byte equal to that path: the same cos kernel, the same
// pre-cos v_src/v_tgt values, and the same FP expression shape (sin_val * static_cast<double>(phi) *
// value) as evolve_step's D-apply (Evolution.cpp). `param` is the SCHRODINGER-SIGNED value the non-fused
// evolve_step receives. At R>1 a rotation's two endpoints can live on different ranks: each rank applies
// only the ADD to the slot it owns (half rotations in fc.cross_half), with the partner coeff already
// carried over the wire during the build exchange — so this drain needs no MPI of its own. Not templated
// on NumModes — it operates purely on the resolved FusedContract + op_coeffs.
inline auto apply_fused_contract(FusedContract &fc,
                                 VecD &op_coeffs,
                                 const CosMask &cos,
                                 double param,
                                 bool schrodinger,
                                 const FrameRefs &frame = {}) -> void {
    // Lazy-cosine frame: when frame.stamp is set the eager cos pass is SKIPPED and each touched endpoint
    // is written directly to its post-firing true value (cos_val·v + sin term, where v are the
    // reconstructed pre-firing endpoint values) and re-stamped to frame.nfirings — so the firing's cos is
    // applied to endpoints exactly once here, and to the untouched (cold) anti terms lazily via the firing
    // log. Physics-identical to the eager path (same rotation set); differs only in FP rounding.
    const bool implicit = (frame.stamp != nullptr);
    uint32_t *const stamp = frame.stamp;
    // (1) INSERT records: v_tgt is the freshly-inserted term's PRE-cos coeff, available only now that
    // op_coeffs has been extended. Needed ONLY in the Schrödinger picture — Heisenberg fresh inserts have
    // coeff 0 (extend zero-fills the appended tail), so v_tgt stays its initialized 0.0 and the
    // c[src] += …·v_tgt add is a no-op; skip the gather entirely. Parallel over the distinct insert targets.
    if (schrodinger) {
        threading::parallel_for_ranges(fc.inserts.size(), [&](size_t begin, size_t end) {
            for (size_t k = begin; k < end; ++k) {
                fc.inserts[k].v_tgt = op_coeffs[fc.inserts[k].tgt];
            }
        });
    }

    // (2) Cos scale over ALL anticommuting endpoints (inserts included) — unchanged kernel, identical to
    // the non-fused cos_scale callback.
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);
    double *const c = op_coeffs.data();
    if (!implicit) {
        profiling::ScopedRegion prof_cs(profiling::Region::CosScale);
        scale_cos_mask(c, cos, cos_val);
    }

    // (3) One parallel apply over hits ++ inserts ++ cross_half. Each op slot is touched by exactly one
    // add (single-touch invariant: pivot leader/follower split + ⊕G-injective targets +
    // drop_matched_cross_rank_followers), so the apply is order-free and thread-count invariant. Full
    // rotations (hits/inserts) write BOTH local endpoints; half rotations (cross_half — resolver +φ and
    // querier −φ) write only the single slot THIS rank owns, exactly like Evolution.cpp's cross-rank D-apply.
    const size_t n_hit = fc.hits.size();
    const size_t n_full = n_hit + fc.inserts.size();
    const size_t n_cross = fc.cross_half.size();
    profiling::ScopedRegion prof_fa(profiling::Region::FusedApply);
    uint8_t *const magp = frame.mag;
    threading::parallel_for_ranges(n_full + n_cross, [&](size_t begin, size_t end) {
        for (size_t k = begin; k < end; ++k) {
            if (k < n_full) {
                const RotationRec &r = (k < n_hit) ? fc.hits[k] : fc.inserts[k - n_hit];
                if (implicit) {
                    // Assign the post-firing true value (cos·v + sin term) and re-stamp: equivalent to
                    // the eager "scale by cos, then add the sin term" on the same single-touch slot.
                    c[r.src] = cos_val * r.v_src + sin_val * static_cast<double>(-r.phase) * r.v_tgt;
                    c[r.tgt] = cos_val * r.v_tgt + sin_val * static_cast<double>(r.phase) * r.v_src;
                    stamp[r.src] = frame.nfirings;
                    stamp[r.tgt] = frame.nfirings;
                }
                else {
                    c[r.src] += sin_val * static_cast<double>(-r.phase) * r.v_tgt;
                    c[r.tgt] += sin_val * static_cast<double>(r.phase) * r.v_src;
                }
                // Magnitude byte: the two endpoints just grew — refresh their upper-bound bytes so the
                // byte stays valid (single-touch-per-slot ⇒ race-free). The cos scale above only shrinks,
                // so it needs no refresh (see CoeffFrame.h).
                if (magp != nullptr) {
                    magp[r.src] = mag_byte(c[r.src]);
                    magp[r.tgt] = mag_byte(c[r.tgt]);
                }
            }
            else {
                // Cross-rank half rotations (R>1). The lazy arm mirrors the full-rotation arm: write the
                // post-firing true value cos·v_local + sin·(partner) — v_local is the local slot's own
                // pre-firing coeff reconstructed at resolve time (see HalfRotationRec) — and re-stamp. The
                // eager arm adds the sin term to the already-cos-scaled slot exactly like Evolution.cpp.
                const HalfRotationRec &h = fc.cross_half[k - n_full];
                if (implicit) {
                    c[h.local_idx] = cos_val * h.v_local + sin_val * static_cast<double>(h.phase_signed) * h.v_partner;
                    stamp[h.local_idx] = frame.nfirings;
                }
                else {
                    c[h.local_idx] += sin_val * static_cast<double>(h.phase_signed) * h.v_partner;
                }
                if (magp != nullptr) {
                    magp[h.local_idx] = mag_byte(c[h.local_idx]);
                }
            }
        }
    });
}

} // namespace monoprop::detail
