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

// ─── CoeffFrame: the lazy (log-structured) cosine frame ─────────────────────────
// The ContractImmediately build applies one rotation exp(θ_f G_f) per gate firing f. Its dominant
// cost used to be the EAGER cosine pass: multiply EVERY anticommuting coefficient by cos(2θ_f) at
// every firing — a memory-bandwidth sweep over ~all terms, 90–97% of which are below the atol gate
// and never rotate. The coefficient of a term only shrinks (by |cos| ≤ 1) until it participates in a
// rotation, so instead of touching cold coefficients we FREEZE them and record each firing in an
// append-only log, reconstructing a term's true value on demand:
//
//     true_value(i) = stored[i] × ∏ cos(2θ_f)   over firings f ∈ (stamp[i], now]  with  a(row_i,G_f)=1
//
// where a(M,G) = parity(|M ∩ G|) ⊕ (|G| odd ? parity(|M|) : 0) is the anticommutation predicate —
// F₂-linear in M and a pure function of the (immutable) row and the generator. `stamp[i]` is the
// firing index at which term i was last made current (a rotation endpoint write, or a barrier).
//
// Three invariants carry the correctness:
//   • Rows are immutable ⇒ a(row_i, G_f) is well-defined for every past firing, so replaying the
//     window (stamp, now] is exact no matter how long term i slept.
//   • |cos| ≤ 1 ⇒ the frozen `stored[i]` is itself a valid UPPER BOUND on |true_value(i)|; the
//     1-byte `mag[i]` (log2 upper bound) is a cheaper proxy that lets the scan reject a cold term
//     without reading its 8-byte coefficient. Staleness is one-sided (bounds only overestimate).
//   • Every propagate ends with materialize_all() at the barrier, which writes back the true values,
//     resets stamps to 0, and empties the log — so frames are EMPTY at every public API boundary and
//     the reconstruction window never exceeds one Trotter step.
//
// Per-firing state is stored/reconstructed directly as cos PRODUCTS (not log-sum-exp): |cos| ≤ 1 so
// the product only shrinks — underflow to 0 is the physically-correct "annihilated" result (e.g. the
// θ=π/4 kicked-Ising cos ≈ 6e-17), needing no clamp — and the ascending-firing multiplication order
// matches the eager path's repeated multiply, so reconstruction of an untouched term is FP-faithful.
//
// The log is tiny: per firing one double (cos) + one odd-|G| bit + |G| support bits across the 2N
// posmask rows, emptied each barrier ⇒ ~one Trotter step resident, sub-MB. The per-term arrays
// (`stamp`: 4 B, optional `mag`: 1 B) run parallel to the picture's coefficient vector.

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <unistd.h> // sysconf — last-level cache size for the engage gate

#include "monoprop/MajoranaAlgebra.h" // MajoranaSet
#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// ─── magnitude byte helpers ─────────────────────────────────────────────────────
// Per-term 1-byte UPPER BOUND on log2|coefficient|. For c != 0, frexp(|c|) = m·2^e with 0.5 ≤ m < 1
// so |c| < 2^e — e is a valid upper-bound exponent; the stored byte is clamp(e + kMagBias, 1, 255)
// and byte 0 is the exact-zero / deep-underflow bucket. Decoded bound 2^(byte-kMagBias) ≥ |c|.
// Bias 200 fits chemistry coefficients (|c| ∈ ~[2^-745, 2^55]) and folds the deep tail into byte 0.
inline constexpr int kMagBias = 200;

// The magnitude byte is a prefilter that lets the scan reject a cold term (below the atol gate) without
// reconstructing it — which in the lazy frame means skipping a position gather + cosine-window product,
// far dearer than the eager path's single coefficient read. Measured LOAD-BEARING: with it OFF the
// ~90% cold anti terms all reconstruct and the lazy path is SLOWER than eager (Hubbard c6 16T: 67.3s
// off vs 48.8s on vs 56.3s eager). Its 1-byte/term is well spent; validity rests on the frozen coeff
// being a monotone upper bound (cosines only shrink), so a stale byte can only over-keep, never wrongly reject.
inline constexpr bool kFrameUseMagByte = true;

// Firing cap: the frame is flushed (materialized + log emptied) once the log reaches this many firings,
// bounding the reconstruction window. A single propagate() carrying thousands of gates would otherwise
// grow the log unboundedly and make per-gate reconstruction quadratic in the gate count. Per-Trotter-step
// callers (Hubbard, ~476 firings/call) never reach it — their per-step barrier flushes first — so their
// measured win is untouched; it is a safety bound for large-operator single-call circuits. Tunable here.
inline constexpr uint32_t kFrameFlushFirings = 512;

// ─── engaging the frame: does the operator fit in the CPU's last-level cache? ───
// The lazy win is a BANDWIDTH optimization: the eager cos pass re-streams every anticommuting term's
// row+coefficient from memory on EVERY firing, so skipping the ~90% cold terms only pays once that sweep
// spills out of cache to DRAM. That crossover is a property of the MACHINE, not a magic term count —
// engage the frame once the operator's working set exceeds the last-level cache. Below it the sweep is
// cache-resident and cheap and the frame's per-term overhead would only lose (a cache-resident, 223k-term
// kicked-Ising ran ~3.5× slower under a forced frame); above it the sweep is DRAM-bound and the frame
// wins, the margin growing with size. Reading the actual cache size makes this portable — a small-cache
// laptop engages sooner, a big-cache server later — where a fixed count would misjudge both. The crossover
// region is broad (physical operators sit far from it on the correct side, so the estimate need not be
// exact, and a misjudgement only costs a slower-but-correct eager build). Operator size is monotonic
// non-decreasing within a run ⇒ a clean one-way eager→lazy switch, decided from the identical pre-gate
// size in build_layer and the propagator.

// Last-level cache size in bytes, detected once. sysconf reports the aggregate L3 — the right basis here
// because the operator is sharded across all cores/CCXs — falling back to L2 then a conservative 32 MiB
// so the gate still separates small from large operators if the query is unavailable.
inline auto last_level_cache_bytes() -> size_t {
    static const size_t bytes = [] {
        for (const int name : {_SC_LEVEL3_CACHE_SIZE, _SC_LEVEL2_CACHE_SIZE}) {
            const long s = sysconf(name);
            if (s > 0) {
                return static_cast<size_t>(s);
            }
        }
        return size_t{32} << 20;
    }();
    return bytes;
}

// True iff the operator's row+coefficient working set exceeds the last-level cache — the point past which
// the eager cos sweep spills to DRAM and the lazy frame starts paying off. row_bytes is the picture's
// fixed majorana row width; + sizeof(double) is the coefficient the sweep also touches per term.
template <size_t NumModes>
inline auto engage_lazy_frame(size_t n_terms) -> bool {
    constexpr size_t row_bytes = ((2 * NumModes + 63) / 64) * 8;
    constexpr size_t bytes_per_term = row_bytes + sizeof(double);
    return n_terms > last_level_cache_bytes() / bytes_per_term;
}

[[gnu::always_inline]] inline auto mag_byte(double c) -> uint8_t {
    if (c == 0.0) {
        return 0;
    }
    int e = 0;
    (void)std::frexp(c, &e); // |c| < 2^e
    const int b = e + kMagBias;
    if (b <= 0) {
        return 0;
    }
    if (b > 255) {
        return 255;
    }
    return static_cast<uint8_t>(b);
}

// The per-firing reject threshold rt for the byte prefilter: reject term iff (double)byte <= rt.
// Returns -inf (never reject) when the atol gate is inactive or |sin| is zero. Because the byte is
// an upper bound this never rejects a term the exact is_below_sin would keep (survivors exact-confirm).
inline auto mag_reject_threshold(bool check_atol, double abs_sin_val, double atol_value) -> double {
    if (!check_atol || abs_sin_val <= 0.0 || atol_value <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(kMagBias) + std::log2(atol_value / abs_sin_val);
}

// ─── CoeffFrame<NumModes> ───────────────────────────────────────────────────────
// One frame per active picture (Heisenberg op / Schrödinger state). Owns the firing log plus the
// per-term stamp and (optional) magnitude byte arrays that ride parallel to the picture's coeffs.
template <size_t NumModes>
struct CoeffFrame {
    static constexpr size_t kColumns = 2 * NumModes;

    // ── firing log (emptied at each barrier) ──
    std::vector<double> cosv;                            // per firing f: cos(2θ_f), signed
    std::vector<uint64_t> oddmask;                       // bit f: |G_f| odd (parity(|M|) correction)
    std::array<std::vector<uint64_t>, kColumns> posmask; // posmask[p], bit f: p ∈ G_f
    size_t nfirings = 0;

    // ── per-term arrays (parallel to the picture coeffs) ──
    std::vector<uint32_t> stamp; // firing index at last materialization of each term
    std::vector<uint8_t> mag;    // upper-bound byte on log2|coeff| (empty ⇒ no byte prefilter)

    auto active() const -> bool { return nfirings != 0; }

    // Empty the firing log only (keeps the per-term arrays). Used by materialize_all after write-back.
    auto clear_log() -> void {
        cosv.clear();
        oddmask.clear();
        for (auto &row : posmask) {
            row.clear();
        }
        nfirings = 0;
    }

    // Full reset (frame invalidated): drop the log AND the per-term arrays. Used when the coefficients
    // are overwritten wholesale (update_initial_operator) — the new coeffs are exact, frame empty.
    auto clear() -> void {
        clear_log();
        stamp.clear();
        mag.clear();
    }

    // Bits of word w that fall in the half-open firing range [lo, hi).
    static auto range_mask(size_t w, size_t lo, size_t hi) -> uint64_t {
        const size_t wlo = w * 64;
        const size_t whi = wlo + 64;
        uint64_t m = ~uint64_t{0};
        if (lo > wlo) {
            m &= ~uint64_t{0} << (lo - wlo);
        }
        if (hi < whi) {
            m &= (hi - wlo >= 64) ? ~uint64_t{0} : ((uint64_t{1} << (hi - wlo)) - 1);
        }
        return m;
    }

    // Append firing f = nfirings, storing the fold columns `fold_gen` (the generator whose columns the
    // reconstruct XOR-folds — G for Majorana, J(G)=pair_swap(G) for Pauli) and the odd-|G| row-parity bit
    // `g_odd` (always false for Pauli). `two_theta` is the SAME argument the eager path passes to
    // std::cos/std::sin. See the 2-arg Majorana overload below.
    auto append_firing(double two_theta, const MajoranaSet<NumModes> &fold_gen, bool g_odd) -> void {
        const size_t f = nfirings;
        const size_t w = f >> 6;
        const uint64_t bit = uint64_t{1} << (f & 63U);
        if ((f & 63U) == 0) { // new word: extend every mask row by one zero word
            oddmask.push_back(0);
            for (auto &row : posmask) {
                row.push_back(0);
            }
        }
        cosv.push_back(std::cos(two_theta));
        if (g_odd) {
            oddmask[w] |= bit;
        }
        for (size_t b = fold_gen.find_first(); b < fold_gen.size(); b = fold_gen.find_next(b)) {
            posmask[b][w] |= bit;
        }
        ++nfirings;
    }

    // Majorana convenience overload: fold columns = G, odd-|G| bit derived from popcount.
    auto append_firing(double two_theta, const MajoranaSet<NumModes> &gen) -> void {
        append_firing(two_theta, gen, gen.count() % 2 != 0);
    }

    // Reconstruct a term's true coefficient from `stored` (its frozen value at firing `stamp_`) by
    // multiplying the cos of every firing in [stamp_, nfirings) it anticommutes with. `positions` is
    // the term's ascending support; `m_odd` = parity(|M|). Pure function of (stored, stamp_, positions,
    // log) ⇒ thread/rank invariant. Ascending-firing product order matches the eager repeated multiply.
    auto reconstruct(double stored, uint32_t stamp_, std::span<const uint32_t> positions, bool m_odd) const
        -> double {
        if (stored == 0.0 || stamp_ >= nfirings) {
            return stored;
        }
        const size_t w0 = static_cast<size_t>(stamp_) >> 6;
        const size_t w1 = (nfirings + 63) >> 6;
        double v = stored;
        for (size_t w = w0; w < w1; ++w) {
            uint64_t a = 0;
            for (const uint32_t p : positions) {
                a ^= posmask[p][w];
            }
            if (m_odd) {
                a ^= oddmask[w];
            }
            a &= range_mask(w, stamp_, nfirings);
            for (uint64_t bits = a; bits != 0; bits &= bits - 1) {
                v *= cosv[w * 64 + static_cast<size_t>(std::countr_zero(bits))];
            }
        }
        return v;
    }

    // true_value: gather term i's ascending positions from `store` and reconstruct. Read-only
    // (thread_local scratch) ⇒ safe under the parallel scan/resolve. Templated on the store so this
    // header need not include OperatorIndex (duck-typed: store.for_each_position(i, fn)).
    template <typename Store>
    [[gnu::always_inline]] auto true_value(const Store &store, double stored, uint32_t stamp_, size_t i) const
        -> double {
        if (stored == 0.0 || static_cast<size_t>(stamp_) >= nfirings) {
            return stored;
        }
        static thread_local std::vector<uint32_t> pos;
        pos.clear();
        size_t k = 0;
        store.for_each_position(i, [&](size_t p) {
            pos.push_back(static_cast<uint32_t>(p));
            ++k;
        });
        return reconstruct(stored, stamp_, std::span<const uint32_t>(pos.data(), pos.size()), (k & 1U) != 0);
    }

    // Ensure the per-term arrays cover `n` terms before a build pass. New entries only appear here at
    // frame start (nfirings == 0, just after a barrier/init), where stamp 0 is exact; the per-gate
    // extend path stamps fresh mid-frame terms itself (see extend_new_terms). mag grows lazily.
    // Grow the magnitude-byte prefilter to cover [old, n) from coeffs. No-op if mag is unused (empty)
    // or already large enough. Shared by ensure_capacity and extend_new_terms.
    auto grow_mag(const VecD &coeffs, size_t n) -> void {
        if (!mag.empty() && mag.size() < n) {
            const size_t old = mag.size();
            mag.resize(n);
            for (size_t i = old; i < n; ++i) {
                mag[i] = mag_byte(coeffs[i]);
            }
        }
    }

    auto ensure_capacity(size_t n, const VecD &coeffs) -> void {
        if (stamp.size() < n) {
            stamp.resize(n, 0);
        }
        grow_mag(coeffs, n);
    }

    // Grow the per-term arrays to cover freshly-inserted terms [old, coeffs.size()) born this firing:
    // they are exact NOW, so stamp them at `nf` (post-append epoch) and take their byte from coeffs.
    auto extend_new_terms(const VecD &coeffs, uint32_t nf) -> void {
        const size_t n = coeffs.size();
        if (stamp.size() < n) {
            stamp.resize(n, nf);
        }
        grow_mag(coeffs, n);
    }

    // Record a rotation endpoint write: coeffs[i] now holds the true post-firing value, current
    // through firing nf. Refresh the byte (the endpoint just grew) and re-stamp. Single-touch per
    // slot ⇒ race-free under the parallel apply.
    [[gnu::always_inline]] auto on_endpoint_write(size_t i, double v, uint32_t nf) -> void {
        stamp[i] = nf;
        if (!mag.empty()) {
            mag[i] = mag_byte(v);
        }
    }

    // Barrier sweep: write every term's true value back into `coeffs`, reset stamps, empty the log,
    // and (if the byte array is in use) rebuild it exactly from the materialized coeffs. Leaves the
    // frame empty ⇒ external readers see a plain eager coefficient vector.
    template <typename Store>
    auto materialize_all(const Store &store, VecD &coeffs, bool use_mag) -> void {
        const size_t n = coeffs.size();
        if (active()) {
            if (stamp.size() < n) {
                stamp.resize(n, 0);
            }
            const CoeffFrame &self = *this;
            threading::parallel_for_ranges(n, [&](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    coeffs[i] = self.true_value(store, coeffs[i], stamp[i], i);
                    stamp[i] = 0; // frame reset: log cleared below, window restarts empty
                }
            });
            clear_log();
        }
        if (use_mag) {
            rebuild_mag(coeffs);
        }
    }

    // (Re)build the byte array from the coefficient vector (parallel). Enables the byte prefilter.
    auto rebuild_mag(const VecD &coeffs) -> void {
        const size_t n = coeffs.size();
        mag.resize(n);
        if (n == 0) {
            return;
        }
        uint8_t *const b = mag.data();
        const double *const c = coeffs.data();
        threading::parallel_for_ranges(n, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                b[i] = mag_byte(c[i]);
            }
        });
    }

    auto memory_bytes() const -> size_t {
        size_t total = (cosv.capacity() + oddmask.capacity()) * 8;
        for (const auto &row : posmask) {
            total += row.capacity() * 8;
        }
        total += stamp.capacity() * sizeof(uint32_t) + mag.capacity();
        return total;
    }
};

} // namespace monoprop::detail
