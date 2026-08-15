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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <print>
#include <vector>

// Opt-in accounting for where a partitioned collective's wall time goes (monoprop_COMM_PROFILE=1). Off by
// default and never allocated then, so the hot path pays one null check per instrumented region.
//
// The split that matters is table_p0 vs table_par: the offset/count tables a HybridComm collective
// rebuilds are O(R*S^2), and whether one partition fills them alone (table_p0, the other S-1 parked in a
// barrier) or every partition fills its own slice (table_par) is the difference between a serial and a
// parallel protocol. Barrier wait is attributed per partition to make that asymmetry visible. table_move
// separates payload memcpy, which a better protocol does not shrink, from bookkeeping, which it does.

namespace monoprop::mpi {

class CommProfile {
public:
    using Clock = std::chrono::steady_clock;

    // One cache-line-isolated accumulator per partition: every counter is written only by its own
    // partition thread, so profiling adds no coherence traffic of its own to what it measures.
    struct alignas(64) Slot {
        uint64_t barrier_ns = 0;
        uint64_t table_p0_ns = 0;   // fills executed under `if (local_partition == 0)`
        uint64_t table_par_ns = 0;  // fills every partition runs on its own slice
        uint64_t table_move_ns = 0; // payload memcpy (pack into / scatter out of staging), not bookkeeping
        uint64_t mpi_ns = 0;        // time inside MPI itself
        uint64_t n_barriers = 0;
        uint64_t n_verbs = 0;
    };

    explicit CommProfile(int n_partitions, int mpi_rank)
        : slots_(static_cast<size_t>(n_partitions)),
          mpi_rank_(mpi_rank) {}

    // Locality domains the transport's barrier grouped its partitions into (< 2 ⇒ the flat barrier ran).
    int barrier_groups = 0;

    // How many partitions actually got pinned, out of `partitions`. Reported because barrier_groups = 0 has
    // two legitimate causes that are otherwise indistinguishable from outside the process: nothing was
    // pinned, or every partition landed in one locality domain and so has nothing to fan in across.
    // -1 means the transport was never told (no PartitionGroup owns it, e.g. a bare-transport unit test).
    int pinned = -1;

    auto slot(int partition) -> Slot & { return slots_[static_cast<size_t>(partition)]; }

    // Aggregate over partitions, plus partition 0 broken out. Written to stderr at teardown, one
    // block per MPI rank; a driver greps `COMMPROF` and sums/compares across ranks.
    //
    // noexcept because the only caller is a transport destructor, which may run while an exception
    // unwinds: a diagnostic print that fails must be dropped, never escalated to a terminate.
    auto dump() const noexcept -> void {
        try {
            dump_();
        }
        catch (...) { // std::print can throw (formatting, or a write error on stderr)
        }
    }

private:
    auto dump_() const -> void {
        Slot total;
        for (const Slot &s : slots_) {
            total.barrier_ns += s.barrier_ns;
            total.table_p0_ns += s.table_p0_ns;
            total.table_par_ns += s.table_par_ns;
            total.table_move_ns += s.table_move_ns;
            total.mpi_ns += s.mpi_ns;
            total.n_barriers += s.n_barriers;
            total.n_verbs += s.n_verbs;
        }
        const Slot &p0 = slots_.front();
        // Peer barrier wait is the cost of the master's serial phases seen from the other side.
        const uint64_t peer_barrier_ns = total.barrier_ns - p0.barrier_ns;
        std::print(stderr,
                   "COMMPROF rank={} partitions={} barrier_groups={} pinned={} verbs={} barriers={} "
                   "table_p0_s={:.3f} table_par_s={:.3f} table_move_s={:.3f} mpi_s={:.3f} "
                   "barrier_p0_s={:.3f} barrier_peers_s={:.3f} barrier_per_sync_us={:.2f}\n",
                   mpi_rank_,
                   slots_.size(),
                   barrier_groups,
                   pinned,
                   p0.n_verbs,
                   p0.n_barriers,
                   to_s(p0.table_p0_ns),
                   to_s(total.table_par_ns) / static_cast<double>(slots_.size()),
                   to_s(total.table_move_ns) / static_cast<double>(slots_.size()),
                   to_s(p0.mpi_ns),
                   to_s(p0.barrier_ns),
                   to_s(peer_barrier_ns) / static_cast<double>(slots_.size() > 1 ? slots_.size() - 1 : 1),
                   total.n_barriers == 0
                       ? 0.0
                       : (static_cast<double>(total.barrier_ns) / static_cast<double>(total.n_barriers)) / 1000.0);
        std::fflush(stderr);
    }

    static auto to_s(uint64_t ns) -> double { return static_cast<double>(ns) / 1e9; }

    std::vector<Slot> slots_;
    int mpi_rank_;
};

// Adds the scope's duration to one counter. Held by value in the instrumented region; `target` must
// outlive it (it is a field of the comm's own CommProfile).
class ScopedNs {
public:
    explicit ScopedNs(uint64_t *target) : target_(target), start_(CommProfile::Clock::now()) {}
    ScopedNs(const ScopedNs &) = delete;
    auto operator=(const ScopedNs &) -> ScopedNs & = delete;
    ~ScopedNs() {
        if (target_ != nullptr) {
            *target_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(CommProfile::Clock::now() - start_).count());
        }
    }

private:
    uint64_t *target_;
    CommProfile::Clock::time_point start_;
};

} // namespace monoprop::mpi
