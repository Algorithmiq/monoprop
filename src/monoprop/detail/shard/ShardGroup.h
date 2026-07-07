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

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "monoprop/Threading.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/ShmComm.h"
#include "monoprop/detail/shard/CpuTopology.h"

// Intra-process shard runtime. Owns S single-threaded "master" threads, each pinned to a physical
// core and each running an independent MonomialPropagator that holds one hash-partition of the
// operator (built via a Kind::Shm comm). The masters execute the UNCHANGED SPMD engine — the same
// code an MPI rank runs — with the ShmComm standing in for the network. This is the in-process
// realisation of the Phase-0 MPI ceiling (one serial shard per core, spread across L3 domains).
//
// The facade MonomialPropagator fans a method call out to all masters via run_on_all(); because the
// engine's per-gate/per-eval collectives are barrier-synchronised inside ShmComm, every master must
// execute the call CONCURRENTLY, which is exactly what run_on_all guarantees.

namespace monoprop {

template <size_t NumModes>
class MonomialPropagator; // completed before any ShardGroup member body is instantiated (Impl.h)

namespace detail::shard {

template <size_t NumModes>
class ShardGroup {
public:
    // Constructs each shard's propagator with `factory(shard_comm)` ON its own master thread, so the
    // operator's heap allocations are first-touched on the owning core/CCX (the locality that makes
    // sharding win). `factory` must build a single-shard (shards=1) propagator wired to the given comm.
    using Factory = std::function<std::unique_ptr<MonomialPropagator<NumModes>>(mpi::Comm)>;

    ShardGroup(int n_shards, const Factory &factory)
        : n_(n_shards), comm_(n_shards), shards_(static_cast<size_t>(n_shards)),
          errs_(static_cast<size_t>(n_shards)), cpusets_(topo_shard_cpusets(n_shards)) {
        start_masters_();
        // First job: build each shard on its master (pinned, cache-warm on the owning core).
        run_on_all([&](int r) { shards_[static_cast<size_t>(r)] = factory(mpi::Comm::make_shm(&comm_, r)); });
    }

    // Clone: build each shard as a deep copy of `src`'s shard on the new master (fresh comm/threads),
    // then rebind the copy's comm to THIS group (the copy inherited a handle to src's ShmComm).
    ShardGroup(const ShardGroup &src)
        : n_(src.n_), comm_(src.n_), shards_(static_cast<size_t>(src.n_)),
          errs_(static_cast<size_t>(src.n_)), cpusets_(topo_shard_cpusets(src.n_)) {
        start_masters_();
        run_on_all([&](int r) {
            auto p = std::make_unique<MonomialPropagator<NumModes>>(*src.shards_[static_cast<size_t>(r)]);
            p->comm_ = mpi::Comm::make_shm(&comm_, r); // ShardGroup is a friend of MonomialPropagator
            shards_[static_cast<size_t>(r)] = std::move(p);
        });
    }
    auto operator=(const ShardGroup &) -> ShardGroup & = delete;

    ~ShardGroup() {
        {
            std::lock_guard lk(m_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto &t : masters_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    auto shard_count() const -> int { return n_; }
    auto shard(int s) -> MonomialPropagator<NumModes> & { return *shards_[static_cast<size_t>(s)]; }
    auto shard(int s) const -> const MonomialPropagator<NumModes> & { return *shards_[static_cast<size_t>(s)]; }

    /// Run `body(shard_rank)` on ALL masters concurrently; block until every master finishes; then
    /// rethrow the first exception any master raised (peers were released via ShmComm poison, so a
    /// throw on one master never hangs the others).
    auto run_on_all(const std::function<void(int)> &body) -> void {
        {
            std::lock_guard lk(m_);
            comm_.reset(); // clear any poison/arrival state left by a previously aborted round
            for (auto &e : errs_) {
                e = nullptr;
            }
            job_ = &body;
            done_count_ = 0;
            ++job_gen_;
        }
        cv_start_.notify_all();
        {
            std::unique_lock lk(m_);
            cv_done_.wait(lk, [&] { return done_count_ == n_; });
        }
        for (auto &e : errs_) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
    }

private:
    // Free-function wrapper so the header compiles on non-Linux (where shard_cpusets returns {}).
    static auto topo_shard_cpusets(int n) -> std::vector<cpu_set_t> {
        return monoprop::detail::shard::shard_cpusets(static_cast<size_t>(n));
    }

    auto start_masters_() -> void {
        masters_.reserve(static_cast<size_t>(n_));
        for (int r = 0; r < n_; ++r) {
            masters_.emplace_back([this, r] { master_loop_(r); });
        }
    }

    auto master_loop_(int rank) -> void {
        if (!cpusets_.empty()) {
            pin_this_thread(cpusets_[static_cast<size_t>(rank)]);
        }
        // Each shard runs the engine fully serially: one shard per core is the Phase-0 optimum, and it
        // keeps all of a shard's mutable data owned by a single core (no cross-CCX coherence traffic).
        threading::gate_serial_override() = true;
        unsigned seen = 0;
        for (;;) {
            const std::function<void(int)> *job = nullptr;
            {
                std::unique_lock lk(m_);
                cv_start_.wait(lk, [&] { return stop_ || job_gen_ != seen; });
                if (stop_) {
                    return;
                }
                seen = job_gen_;
                job = job_;
            }
            try {
                (*job)(rank);
            }
            catch (...) {
                errs_[static_cast<size_t>(rank)] = std::current_exception();
                comm_.poison(); // release peers waiting in a barrier so they don't hang
            }
            {
                std::lock_guard lk(m_);
                ++done_count_;
            }
            cv_done_.notify_one();
        }
    }

    int n_;
    mpi::ShmComm comm_;
    std::vector<std::unique_ptr<MonomialPropagator<NumModes>>> shards_;
    std::vector<std::exception_ptr> errs_;
    std::vector<cpu_set_t> cpusets_;
    std::vector<std::thread> masters_;

    // Job dispatch: the facade thread publishes one job and waits for all masters to complete it.
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int)> *job_ = nullptr;
    unsigned job_gen_ = 0;
    int done_count_ = 0;
    bool stop_ = false;
};

} // namespace detail::shard
} // namespace monoprop
