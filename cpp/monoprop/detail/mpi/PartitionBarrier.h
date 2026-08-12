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
#include <chrono>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "monoprop/detail/EnvConfig.h"
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
// Given a group id per participant (its locality domain, from CpuTopology: the deepest shared cache, or
// the NUMA node) the barrier becomes two-level: fan in within a domain, then across domains, then release
// within the domain. Both the arrival fetch_add and the release store then cost O(S/G) coherence
// transactions instead of O(S), and stay inside one cache slice. No group ids (pinning off, or /sys
// unreadable), a single domain, all-singleton domains, or monoprop_BARRIER_GROUPING=0 degrade to the
// flat barrier. That last one exists because the domains come from the cpusets: without it the only way
// to get a flat barrier is to unpin, so "grouped vs flat" and "pinned vs unpinned" could never be
// separated, and the second level's value could not be measured on a real workload.
//
// The second level is not a free win. It replaces one fetch_add with two sequential hops, so it pays
// only when there is contention to relieve: at high partition counts, or when arrivals are skewed. On a
// collective short enough that the barrier is most of it, and with partitions arriving together, the
// extra hop is pure latency.
//
// The acquire/release/relaxed orderings below are load-bearing, not an oversight: the release store to
// `gen_` is what publishes the preceding relaxed reset of `arrived_`. Promoting them to seq_cst would
// put a full barrier in the spin loop of that same hotspot, so cpp:S8417 is suppressed for this file
// in sonar-project.properties. Do not "simplify" them to the default ordering.
class PartitionBarrier {
public:
    // `spin_budget` overrides how long a waiter stays on-core before yielding; unset takes
    // monoprop_SPIN_BUDGET_US, else kDefaultSpinBudgetUs. Injectable because config::get() caches on
    // first call, so a test cannot reach the env path in-process, and because sweeping the budget is
    // how its default is justified.
    explicit PartitionBarrier(int participants,
                              const std::vector<int> &group_of = {},
                              std::optional<std::chrono::microseconds> spin_budget = std::nullopt)
        : spin_budget_(spin_budget.value_or(
              std::chrono::microseconds{config::get().spin_budget_us.value_or(detail::kDefaultSpinBudgetUs)})),
          participants_(participants) {
        if (static_cast<int>(group_of.size()) != participants || participants <= 0 || !config::get().barrier_grouping) {
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
        // One domain has nothing to fan in across. So does a set of all-singleton domains, which is what
        // an unreadable topology produces: every participant would be its own representative and they
        // would all still meet at the root, i.e. the flat barrier plus a cache line and a release store
        // per participant. Reporting both as flat also stops group_count() from claiming a level that is
        // not actually running.
        if (domains.size() < 2 || domains.size() == static_cast<size_t>(participants)) {
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

    // Locality domains the participants were grouped into; < 2 means the flat barrier is in use. Reported by
    // CommProfile so a run states which barrier it ran instead of leaving it inferred from the timing.
    auto group_count() const -> int { return groups_; }

    // The on-core spin budget actually in force, so a test can observe the resolution rather than
    // re-computing it, and so a sweep can report what it swept.
    auto spin_budget() const -> std::chrono::microseconds { return spin_budget_; }

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
    // Wait until `word` leaves `seen`, backing off through the two phases CpuRelax.h describes: on-core
    // relax for a bounded time, then yield indefinitely. Both phases re-read `poisoned_` on every
    // iteration, so an unwinding peer releases us with an exception rather than a hang and poison() needs
    // no wakeup channel of its own. There is deliberately no third parking phase: sleeping past the yield
    // budget measured 2021 vs 762 us/sync when oversubscribed, because one sleeper's timer overshoot
    // delays every participant behind the barrier (see CpuRelax.h).
    auto spin_until_(const std::atomic<unsigned> &word, unsigned seen) const -> void {
        // The deadline is computed lazily, after the first load fails. The overwhelmingly common case is
        // that the release store has already landed, and a clock read per arrival is pure overhead on the
        // path this barrier exists to keep cheap.
        std::chrono::steady_clock::time_point spin_deadline{};
        bool have_deadline = false;
        bool spinning = true;
        int until_check = detail::kRelaxPerClockCheck;
        while (word.load(std::memory_order_acquire) == seen) {
            if (poisoned_.load(std::memory_order_acquire)) {
                throw ShmCommPoisoned();
            }
            if (!have_deadline) {
                spin_deadline = std::chrono::steady_clock::now() + spin_budget_;
                have_deadline = true;
            }
            if (spinning) {
                if (--until_check <= 0) {
                    until_check = detail::kRelaxPerClockCheck;
                    spinning = std::chrono::steady_clock::now() < spin_deadline;
                }
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

    // Resolved once per barrier, not once per spin: monoprop_SPIN_BUDGET_US is a sweep knob, and the spin
    // loop is the measured hotspot.
    std::chrono::microseconds spin_budget_;

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
