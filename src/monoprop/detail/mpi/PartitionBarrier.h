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

// Thrown when a peer partition poisoned the barrier instead of arriving: a permanent hang becomes a
// propagating exception on every participant.
class ShmCommPoisoned : public std::runtime_error {
public:
    ShmCommPoisoned() : std::runtime_error("ShmComm poisoned: a peer partition threw during a collective") {}
};

// Sense-reversing generation barrier for a fixed number of in-process partition threads, with a poison
// escape. Each barrier word gets a private cache line: if `gen_` shared `arrived_`'s line, spinners'
// reloads would miss to L3 on every peer arrival — O(S) coherence bounces (measured top hotspot at S=112).
class PartitionBarrier {
public:
    explicit PartitionBarrier(int participants) : participants_(participants) {}

    PartitionBarrier(const PartitionBarrier &) = delete;
    auto operator=(const PartitionBarrier &) -> PartitionBarrier & = delete;

    auto sync() -> void {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants_) {
            arrived_.store(0, std::memory_order_relaxed);
            gen_.store(g + 1, std::memory_order_release);
        }
        else {
            // Bounded on-core spin first (pinned partitions ⇒ the release store lands in the pause window,
            // no syscall); only long waits (imbalance, oversubscription) fall to yield.
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

    // Signal that this participant is unwinding (e.g. an engine exception): release peers spinning in a
    // barrier so they throw ShmCommPoisoned rather than hang forever. Idempotent.
    auto poison() -> void { poisoned_.store(true, std::memory_order_release); }

    // Clear the poison flag and arrival counter. must be called only when every participant is quiescent
    // (between rounds), so a poison-aborted round leaves no dirty state. `gen_` deliberately stays
    // monotonic: each participant re-reads it at its next barrier.
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
