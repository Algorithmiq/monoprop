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

#include <atomic>
#include <stdexcept>
#include <thread>

#include "monoprop/detail/mpi/CpuRelax.h"

namespace monoprop::mpi {

/// Thrown by a collective when a peer shard unwound (set the poison flag) instead of arriving —
/// turns a would-be permanent barrier hang into a propagating exception on every participant.
class ShmCommPoisoned : public std::runtime_error {
public:
    ShmCommPoisoned() : std::runtime_error("ShmComm poisoned: a peer shard threw during a collective") {}
};

/// Sense-reversing generation barrier for a fixed number of in-process shard threads, with a poison
/// escape. Both in-process transports (ShmComm, HybridComm) drive their two-phase collectives on one
/// of these; the only per-transport difference is the participant count.
///
/// The completer (last arriver) resets the counter then bumps the generation, releasing spinners; a
/// poisoned peer that never arrives is covered because spinners also break on the poison flag and
/// throw. Each barrier word gets a private cache line: every arrival's fetch_add on `arrived_` takes
/// its line exclusive, and if `gen_` shared that line the spinners' reload would miss to L3 on every
/// peer arrival — O(S) coherence bounces per barrier (measured as the top hotspot at S=112).
class ShardBarrier {
public:
    explicit ShardBarrier(int participants) : participants_(participants) {}

    ShardBarrier(const ShardBarrier &) = delete;
    auto operator=(const ShardBarrier &) -> ShardBarrier & = delete;

    auto sync() -> void {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants_) {
            arrived_.store(0, std::memory_order_relaxed);
            gen_.store(g + 1, std::memory_order_release);
        }
        else {
            // Bounded on-core spin first: with one pinned shard per core the completer's release store
            // lands within the pause window, so the hot path never syscalls. Only genuinely long waits
            // (imbalance tails, oversubscription) fall back to yielding the timeslice.
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == g) {
                if (poisoned_.load(std::memory_order_acquire)) {
                    throw ShmCommPoisoned();
                }
                if (spins < detail::kSpinPauseIters) {
                    ++spins;
                    detail::cpu_relax();
                }
                else {
                    std::this_thread::yield();
                }
            }
        }
        if (poisoned_.load(std::memory_order_acquire)) {
            throw ShmCommPoisoned();
        }
    }

    /// Signal that this participant is unwinding (e.g. an engine exception): release peers spinning in
    /// a barrier so they throw ShmCommPoisoned rather than hang forever. Idempotent.
    auto poison() -> void { poisoned_.store(true, std::memory_order_release); }

    /// Clear the poison flag and the arrival counter. MUST be called only when every participant is
    /// quiescent (between collective rounds, e.g. by the shard dispatcher before a new job), so a round
    /// aborted by poison leaves no dirty state for the next round. The generation is left monotonic
    /// (each participant re-reads it at its next barrier).
    auto reset() -> void {
        poisoned_.store(false, std::memory_order_relaxed);
        arrived_.store(0, std::memory_order_relaxed);
    }

private:
    int participants_;
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<unsigned> gen_{0};
    alignas(64) std::atomic<bool> poisoned_{false};
};

} // namespace monoprop::mpi
