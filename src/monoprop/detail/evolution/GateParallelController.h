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
#include <cmath>
#include <cstddef>

#include "monoprop/Threading.h"

namespace monoprop::detail {

// ─── GateParallelController ─────────────────────────────────────────────────────
// Chooses, per gate of the NATIVE PAULI engine, whether the gate's whole build+apply pipeline runs
// PARALLEL (the chunked policies in layer_build/Parallel.h and the Threading.h primitives) or fully
// SERIAL (every dispatch decision on the gate loop's thread sees gate_serial_override() and stays on
// the calling thread).
//
// WHY (thread-scaling campaign, 2026-07-07, ~/monoprop-perf-notes/PAULI_THREADS.md): on Pauli
// workloads whose per-gate hot set is L3-resident and mutated every gate (the fused cos sweep
// dirties the coefficient array; inserts grow the store), spreading the per-gate phases across L3
// (CCX) domains turns coefficient accesses into cross-domain dirty-line transfers: measured 15-34x
// task-body slowdowns and NEGATIVE whole-run scaling (kicked-Ising 3.9->5.2 s at 16T, growing
// random-Pauli 3.0->4.7 s), while truly large or static operators genuinely win (static 500k-term
// Pauli op 2.9x). No static per-gate feature separates the two regimes — operator size, anti count,
// emit rate, miss rate and per-chunk work were each measured and fail (the SAME anti count wins in
// one workload and loses in another) — and half-measures are WORSE than either extreme (a few fat
// chunks: 4.7->6.0 s). So the engine measures instead of predicting: an epsilon-greedy controller
// tracks an EWMA of ns-per-anticommuting-term for each mode, bootstraps by alternating, then locks
// onto the cheaper mode and stops probing. Probing re-opens on (a) a clear WORSENING of the locked
// mode's own cost — never on improvement, the natural growing-operator signature (+16% when probed
// on it) — or (b) a 2x drift in per-gate work (n_anti EWMA), after which the tradeoff may have
// flipped (measured on Hubbard-like growth through the parallel crossover).
//
// Scope: installed by the gate loop only for Basis::Pauli. Fermionic (Majorana) workloads keep the
// static always-parallel policy: they showed no negative scaling, and their Trotter steps mix gate
// sizes across 4 decades within one step (Hubbard), where a single whole-gate mode oscillates and
// per-size-bucket learning still paid a measured 9-25% exploration tax. Extending adaptivity there
// needs operator-scale-aware learning — future work.
//
// Mode choice only changes CHUNKING, never results: the build path's order-preserving merge is
// chunk-count invariant, so both modes are byte-identical.
class GateParallelController {
public:
    // Below this anticommuting count a gate is not worth learning: per-gate wall is timer noise and
    // the serial floors in the chunk policies make both modes near-identical anyway.
    static constexpr size_t kMinDecidableAnti = 2048;
    static constexpr int kBootstrapGates = 8;   // alternate modes to seed both EWMAs
    static constexpr int kProbePeriod = 8;      // pre-lock: probe the other mode every N decidable gates
    static constexpr int kConfirmProbes = 2;    // consecutive confirming probes before locking
    static constexpr double kAlpha = 0.3;       // EWMA weight of the newest sample
    static constexpr double kHysteresis = 0.9;  // switch only when the other mode is <90% of current
    static constexpr double kDecisiveGap = 2.0; // EWMA ratio that locks immediately
    // One-sided cost drift: only a clear WORSENING of the locked mode (vs the best cost seen since
    // locking) re-opens probing. This is also the growth safety net: a serial-locked operator that
    // outgrows the cache raises its own ns-per-anti, which fires this trigger. Re-probing on
    // IMPROVEMENT or on per-gate-size drift was tried and measured worse on Pauli workloads (their
    // per-gate anti counts are noisy across gates; a 2x n_anti-EWMA band caused constant re-probing,
    // kicked-Ising 3.9->4.8 s).
    static constexpr double kDriftHi = 1.5;

    // Decide the mode for the next gate. Call once per gate, before running it.
    auto begin_gate() -> bool /*serial*/ {
        if (threading::effective_parallelism() <= 1) {
            probe_ = false;
            return false; // single worker: modes are identical, measure nothing
        }
        if (decidable_seen_ < kBootstrapGates) {
            probe_ = false; // bootstrap alternation is not a "probe"; it seeds both EWMAs
            return (decidable_seen_ % 2) == 1;
        }
        if (!locked_ && (probe_due_ || (decidable_seen_ % kProbePeriod) == 0)) {
            probe_ = true;
            probe_due_ = false;
            return !serial_preferred_;
        }
        probe_ = false;
        return serial_preferred_;
    }

    // Record the finished gate. `serial` is the mode begin_gate() returned; `n_anti` is the scan's
    // anticommuting-term count (the per-gate work measure); `wall_ns` the gate's wall time.
    auto end_gate(bool serial, double wall_ns, size_t n_anti) -> void {
        if (threading::effective_parallelism() <= 1 || n_anti < kMinDecidableAnti) {
            return;
        }
        ++decidable_seen_;
        const double cost = wall_ns / static_cast<double>(n_anti);
        double &ewma = serial ? cost_serial_ : cost_parallel_;
        ewma = std::isnan(ewma) ? cost : (1.0 - kAlpha) * ewma + kAlpha * cost;

        if (locked_) {
            // One-sided cost drift (kDriftHi) re-opens probing; the next decidable gate probes
            // immediately.
            const double locked_cost = serial_preferred_ ? cost_serial_ : cost_parallel_;
            if (!std::isnan(locked_cost)) {
                best_since_lock_ = std::min(best_since_lock_, locked_cost);
                if (locked_cost > kDriftHi * best_since_lock_) {
                    locked_ = false;
                    confirm_count_ = 0;
                    probe_due_ = true;
                }
            }
            return;
        }
        if (std::isnan(cost_serial_) || std::isnan(cost_parallel_)) {
            return; // still seeding
        }
        const bool serial_wins = cost_serial_ < kHysteresis * cost_parallel_;
        const bool parallel_wins = cost_parallel_ < kHysteresis * cost_serial_;
        if (serial_wins != serial_preferred_ && (serial_wins || parallel_wins)) {
            serial_preferred_ = serial_wins;
            confirm_count_ = 0;
            return;
        }
        // A decisive gap locks immediately once both EWMAs are seeded; otherwise a probe that leaves
        // the preference unchanged confirms it, and enough confirmations lock.
        const double better = serial_preferred_ ? cost_serial_ : cost_parallel_;
        const double worse = serial_preferred_ ? cost_parallel_ : cost_serial_;
        if (worse > kDecisiveGap * better || (probe_ && ++confirm_count_ >= kConfirmProbes)) {
            locked_ = true;
            best_since_lock_ = better;
        }
    }

private:
    double cost_serial_ = std::nan("");
    double cost_parallel_ = std::nan("");
    double best_since_lock_ = 0.0;
    size_t decidable_seen_ = 0;
    int confirm_count_ = 0;
    bool serial_preferred_ = false; // status-quo default: parallel
    bool locked_ = false;
    bool probe_ = false;
    bool probe_due_ = false;
};

// RAII gate-mode scope: sets the thread-local serial override read by effective_parallelism() and the
// Threading.h / layer_build parallel dispatch points, for the duration of one gate's build+apply.
// The gate loop and every chunk-count decision run on this same thread (the parallel regions are
// dispatched FROM it), so a thread_local is the correct channel.
class GateModeScope {
public:
    explicit GateModeScope(bool serial) : prev_(threading::gate_serial_override()) {
        threading::gate_serial_override() = serial;
    }
    GateModeScope(const GateModeScope &) = delete;
    auto operator=(const GateModeScope &) -> GateModeScope & = delete;
    ~GateModeScope() { threading::gate_serial_override() = prev_; }

private:
    bool prev_;
};

// Per-gate scan-size report channel (build_layer writes the fused scan's n_anti; the gate loop reads
// it after the gate to feed the controller). Same-thread by the same argument as GateModeScope.
inline auto gate_scan_n_anti() -> size_t & {
    thread_local size_t n_anti = 0;
    return n_anti;
}

} // namespace monoprop::detail
