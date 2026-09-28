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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/Workshare.h"

namespace monoprop::detail {

//! Records per apply_fused_contract logical range, for both the insert gather and the rotation pass.
inline constexpr size_t kFusedRecordRange = 1024;

/*!
 * \brief Whether every coefficient slot the contract writes has exactly one writing record.
 *
 * The slots are the sources and targets of `hits` and `inserts` and the local slots of `cross_half`.
 * apply_fused_contract's unsynchronized parallel pass relies on this; debug builds assert it on every
 * call. Sorts a transient copy of the slot IDs: O(records log records) time and O(records) memory.
 */
inline auto fused_add_owners_unique(const FusedContract &fc) -> bool {
    std::vector<size_t> slots;
    slots.reserve(2 * (fc.hits.size() + fc.inserts.size()) + fc.cross_half.size());
    for (const auto *records : {&fc.hits, &fc.inserts}) {
        for (const RotationRec &r : *records) {
            slots.push_back(r.src);
            slots.push_back(r.tgt);
        }
    }
    for (const HalfRotationRec &h : fc.cross_half) {
        slots.push_back(h.local_idx);
    }
    std::ranges::sort(slots);
    return std::ranges::adjacent_find(slots) == slots.end();
}

/*!
 * \brief Apply records `[lo, hi)` of hits ++ inserts ++ cross_half, each its complete rotation formula.
 *
 * Out of line on purpose: the serial and every threaded range call this one compiled body, so however the
 * compiler contracts `cos·c + sin·φ·v` into a fused multiply-add, the choice is the same at every budget
 * and serial and threaded results stay bitwise equal. The cost is one call per kFusedRecordRange records.
 */
[[gnu::noinline]] inline auto apply_fused_record_range(const FusedContract &fc,
                                                       double *c,
                                                       size_t lo,
                                                       size_t hi,
                                                       double cos_val,
                                                       double sin_val,
                                                       bool fused_scale) -> void {
    const size_t n_hit = fc.hits.size();
    const size_t n_full = n_hit + fc.inserts.size();
    for (size_t k = lo; k < hi; ++k) {
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

/*!
 * \brief The drain paired with build_layer's fused emission: complete each rotation by adding its sine term
 * directly to op_coeffs (the ContractImmediately forward path at all rank counts).
 *
 * The gate's cosine scale reaches the coefficients two ways:
 *   - fused_scale (no length cap, default): the scan already scaled every anticommuting coeff, so no cos pass
 *     runs here; slots born after that sweep (fresh inserts) fold cos in via their apply arm below.
 *   - two-pass (length cap / cos==0 fallback): scale_cos_mask runs here, then every arm is a plain add.
 * At R>1 each rank applies only the add to the slot it owns (half rotations in fc.cross_half).
 *
 * The caller has already sized and extended op_coeffs. The three phases below run in order, each joined
 * before the next, and each is split into fixed logical ranges that run on up to `options.threads` workers.
 *
 * \param observer Test seam; see NoRangeObserver.
 */
template <class Observer = NoRangeObserver>
auto apply_fused_contract(FusedContract &fc,
                          VecD &op_coeffs,
                          const CosMask &cos,
                          double param,
                          bool schrodinger,
                          bool fused_scale,
                          parallel::Options options = {},
                          const Observer &observer = {}) -> void {
    double *const c = op_coeffs.data();

    // (1) insert records: v_tgt is the freshly-inserted term's pre-cos coeff, readable only now op_coeffs
    // is extended. Needed only in Schrödinger — a Heisenberg fresh insert has coeff 0, so skip the gather.
    if (schrodinger) {
        RotationRec *const inserts = fc.inserts.data();
        const size_t n_inserts = fc.inserts.size();
        const size_t ranges = logical_ranges(n_inserts, kFusedRecordRange);
        observer.prepare(KernelRange::fused_gather, ranges);
        // Each range writes only its own records' snapshot field and reads coefficients nothing writes yet.
        parallel::for_blocks(ranges, options, [&](size_t range) {
            observer.visit(KernelRange::fused_gather, range);
            const size_t hi = std::min(n_inserts, (range + 1) * kFusedRecordRange);
            for (size_t k = range * kFusedRecordRange; k < hi; ++k) {
                inserts[k].v_tgt = c[inserts[k].tgt];
            }
        });
    }

    // (2) Two-pass mode only: cos scale over all anticommuting endpoints (inserts included).
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);
    if (!fused_scale) {
        scale_cos_mask(c, cos, cos_val, options, observer);
    }

    // (3) One pass over hits ++ inserts ++ cross_half. Each op slot is touched by exactly one add (pivot
    // split + ⊕G-injective targets + drop_matched_cross_rank_followers), which is what makes the plain +=
    // safe. In fused_scale mode a slot born after the sweep (insert targets, resolver miss halves) folds
    // the gate's cos in here (c = cos·c + sin).
    assert(fused_add_owners_unique(fc));
    const size_t n_records = fc.hits.size() + fc.inserts.size() + fc.cross_half.size();
    const size_t ranges = logical_ranges(n_records, kFusedRecordRange);
    observer.prepare(KernelRange::fused_apply, ranges);
    // Distinct records write distinct slots (the one-add-owner property above), and every value a record
    // reads is its own snapshot or its own slot, so the ranges need no synchronization. Never paper over a
    // violation with atomics: it would mean the scan emitted a slot twice.
    parallel::for_blocks(ranges, options, [&](size_t range) {
        observer.visit(KernelRange::fused_apply, range);
        const size_t lo = range * kFusedRecordRange;
        apply_fused_record_range(fc, c, lo, std::min(n_records, lo + kFusedRecordRange), cos_val, sin_val, fused_scale);
    });
}

} // namespace monoprop::detail
