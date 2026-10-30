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

// Turns what would be a permanent hang into an exception on every participant.
class ShmCommPoisoned : public std::runtime_error {
public:
    ShmCommPoisoned() : std::runtime_error("ShmComm poisoned: a peer partition threw during a collective") {}
};

// Sense-reversing generation barrier for a fixed number of in-process partition threads. Each barrier
// word gets a private cache line: if `gen_` shared `arrived_`'s line, spinners' reloads would miss to
// L3 on every peer arrival — O(S) coherence bounces (measured top hotspot at S=112).
//
// The acquire/release/relaxed orderings below are load-bearing, not an oversight: the release store to
// `gen_` is what publishes the preceding relaxed reset of `arrived_`. Promoting them to seq_cst would
// put a full barrier in the spin loop of that same hotspot, so cpp:S8417 is suppressed for this file
// in sonar-project.properties. Do not "simplify" them to the default ordering.
class PartitionBarrier {
public:
    explicit PartitionBarrier(int participants) : participants_(participants) {}

    PartitionBarrier(const PartitionBarrier &) = delete;
    auto operator=(const PartitionBarrier &) -> PartitionBarrier & = delete;

    // Throws ShmCommPoisoned only for a round that cannot complete. Once every participant has arrived the
    // round has completed, and a poison set afterwards reports a later failure: its poisoner never arrives at
    // the next round, which surfaces it. Failing the completed round instead would fail a collective that
    // succeeded, and under the MPI hybrid partition 0 turns that into MPI_Abort.
    auto sync() -> void {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants_) {
            arrived_.store(0, std::memory_order_relaxed);
            gen_.store(g + 1, std::memory_order_release);
            return;
        }
        // Bounded on-core spin first (pinned partitions ⇒ the release store lands in the pause window,
        // no syscall); only long waits (imbalance, oversubscription) fall to yield.
        int spins = 0;
        while (gen_.load(std::memory_order_acquire) == g) {
            if (poisoned_.load(std::memory_order_acquire)) {
                // A poisoner that left this round saw gen_ advance first, so seeing its poison means seeing
                // the advance too: re-read it before failing a round that may have just completed.
                if (gen_.load(std::memory_order_acquire) != g) {
                    return;
                }
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

    // Signal that this participant is unwinding (e.g. an engine exception), releasing peers spinning in a
    // barrier. Idempotent.
    auto poison() -> void { poisoned_.store(true, std::memory_order_release); }

    // Must be called only when every participant is quiescent (between rounds), so a poison-aborted round
    // leaves no dirty state. `gen_` deliberately stays monotonic: each participant re-reads it at its next
    // barrier.
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
