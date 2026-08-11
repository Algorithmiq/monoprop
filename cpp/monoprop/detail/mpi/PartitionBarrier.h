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
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

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
// Given a group id per participant (its L3 domain, from CpuTopology) the barrier becomes two-level: fan
// in within a domain, then across domains, then release within the domain. Both the arrival fetch_add
// and the release store then cost O(S/G) coherence transactions instead of O(S), and stay inside one L3
// slice. No group ids (pinning off, or /sys unreadable) or a single domain degrades to the flat barrier.
//
// The acquire/release/relaxed orderings below are load-bearing, not an oversight: the release store to
// `gen_` is what publishes the preceding relaxed reset of `arrived_`. Promoting them to seq_cst would
// put a full barrier in the spin loop of that same hotspot, so cpp:S8417 is suppressed for this file
// in sonar-project.properties. Do not "simplify" them to the default ordering.
class PartitionBarrier {
public:
    explicit PartitionBarrier(int participants, const std::vector<int> &group_of = {}) : participants_(participants) {
        if (static_cast<int>(group_of.size()) != participants || participants <= 0) {
            return; // flat
        }
        // Compact the domain ids to 0..G-1 in first-seen order, and count each group.
        std::vector<int> domains;
        group_of_.resize(static_cast<size_t>(participants));
        for (int p = 0; p < participants; ++p) {
            const auto it = std::find(domains.begin(), domains.end(), group_of[static_cast<size_t>(p)]);
            if (it == domains.end()) {
                group_of_[static_cast<size_t>(p)] = static_cast<int>(domains.size());
                domains.push_back(group_of[static_cast<size_t>(p)]);
            }
            else {
                group_of_[static_cast<size_t>(p)] = static_cast<int>(it - domains.begin());
            }
        }
        if (domains.size() < 2) {
            group_of_.clear();
            return; // flat
        }
        groups_ = static_cast<int>(domains.size());
        group_arrived_ = std::vector<Count>(domains.size());
        group_gen_ = std::vector<Gen>(domains.size());
        group_size_.assign(domains.size(), 0);
        for (int p = 0; p < participants; ++p) {
            ++group_size_[static_cast<size_t>(group_of_[static_cast<size_t>(p)])];
        }
    }

    PartitionBarrier(const PartitionBarrier &) = delete;
    auto operator=(const PartitionBarrier &) -> PartitionBarrier & = delete;

    // `participant` selects the caller's group; ignored on the flat path.
    auto sync(int participant) -> void {
        if (groups_ < 2) {
            const unsigned g = gen_.load(std::memory_order_acquire);
            if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants_) {
                arrived_.store(0, std::memory_order_relaxed);
                gen_.store(g + 1, std::memory_order_release);
            }
            else {
                spin_until_(gen_, g);
            }
        }
        else {
            const size_t gi = static_cast<size_t>(group_of_[static_cast<size_t>(participant)]);
            const unsigned seen = group_gen_[gi].v.load(std::memory_order_acquire);
            if (group_arrived_[gi].v.fetch_add(1, std::memory_order_acq_rel) + 1 == group_size_[gi]) {
                group_arrived_[gi].v.store(0, std::memory_order_relaxed);
                // The domain's last arriver represents it upstream, and releases the domain only after
                // the root barrier, so passing the domain word still implies everyone arrived.
                const unsigned root_seen = gen_.load(std::memory_order_acquire);
                if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == groups_) {
                    arrived_.store(0, std::memory_order_relaxed);
                    gen_.store(root_seen + 1, std::memory_order_release);
                }
                else {
                    spin_until_(gen_, root_seen);
                }
                group_gen_[gi].v.store(seen + 1, std::memory_order_release);
            }
            else {
                spin_until_(group_gen_[gi].v, seen);
            }
        }
        if (poisoned_.load(std::memory_order_acquire)) {
            throw ShmCommPoisoned();
        }
    }

    // L3 domains the participants were grouped into; < 2 means the flat barrier is in use. Reported by
    // CommProfile so a run states which barrier it ran instead of leaving it inferred from the timing.
    auto group_count() const -> int { return groups_; }

    // Signal that this participant is unwinding (e.g. an engine exception), releasing peers spinning in a
    // barrier. Idempotent.
    auto poison() -> void { poisoned_.store(true, std::memory_order_release); }

    // Must be called only when every participant is quiescent (between rounds), so a poison-aborted round
    // leaves no dirty state. The generations deliberately stay monotonic: each participant re-reads its
    // own at the next barrier.
    auto reset() -> void {
        poisoned_.store(false, std::memory_order_relaxed);
        arrived_.store(0, std::memory_order_relaxed);
        for (auto &c : group_arrived_) {
            c.v.store(0, std::memory_order_relaxed);
        }
    }

private:
    // Spin until `word` leaves `seen`. Bounded on-core spin first (pinned partitions ⇒ the release store
    // lands in the pause window, no syscall); only long waits (imbalance, oversubscription) fall to
    // yield. A poisoned peer releases us with an exception rather than a hang.
    auto spin_until_(const std::atomic<unsigned> &word, unsigned seen) const -> void {
        int spins = 0;
        while (word.load(std::memory_order_acquire) == seen) {
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

    // Per-domain words, each on its own cache line for the same reason the flat pair is split.
    struct alignas(64) Count {
        std::atomic<int> v{0};
    };
    struct alignas(64) Gen {
        std::atomic<unsigned> v{0};
    };

    int participants_;
    int groups_ = 0; // < 2 ⇒ flat: arrived_/gen_ count participants, not domains
    std::vector<int> group_of_;
    std::vector<int> group_size_;
    std::vector<Count> group_arrived_;
    std::vector<Gen> group_gen_;
    // Flat barrier, or (two-level) the root barrier the domains' last arrivers meet at.
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<unsigned> gen_{0};
    alignas(64) std::atomic<bool> poisoned_{false};
};

} // namespace monoprop::mpi
