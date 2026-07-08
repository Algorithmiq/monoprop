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
#include "monoprop/detail/mpi/MPICompat.h" // mpi::size for the transport choice
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/HybridComm.h"
#endif
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

    // `parent` is the enclosing communicator (size R). R == 1 ⇒ the S shards trade over an in-process
    // ShmComm (single node). R > 1 ⇒ they trade over a HybridComm that composes the R ranks x S shards
    // into one flat P = R*S SPMD world (the MPI hybrid). Either way the shard propagators see a
    // P-partition comm and run the unchanged engine.
    ShardGroup(int n_shards, const Factory &factory, mpi::Comm parent)
        : n_(n_shards), parent_(parent), shards_(static_cast<size_t>(n_shards)),
          errs_(static_cast<size_t>(n_shards)), cpusets_(topo_shard_cpusets(n_shards)) {
        make_transport_();
        start_masters_();
        // First job: build each shard on its master (pinned, cache-warm on the owning core).
        run_on_all([&](int r) { shards_[static_cast<size_t>(r)] = factory(comm_for_(r)); });
    }

    // Clone: rebuild the transport for THIS group (fresh threads + fresh ShmComm/HybridComm over the
    // same parent), deep-copy each of `src`'s shards on the new master, then rebind the copy's comm to
    // this group's transport (the copy inherited a handle to src's).
    ShardGroup(const ShardGroup &src)
        : n_(src.n_), parent_(src.parent_), shards_(static_cast<size_t>(src.n_)),
          errs_(static_cast<size_t>(src.n_)), cpusets_(topo_shard_cpusets(src.n_)) {
        make_transport_();
        start_masters_();
        run_on_all([&](int r) {
            auto p = std::make_unique<MonomialPropagator<NumModes>>(*src.shards_[static_cast<size_t>(r)]);
            p->comm_ = comm_for_(r); // ShardGroup is a friend of MonomialPropagator
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
            transport_reset_(); // clear any poison/arrival state left by a previously aborted round
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

    // Build the shared transport: an in-process ShmComm for a single-rank parent, or a HybridComm that
    // folds the R parent ranks x S shards into one flat world when the parent spans multiple MPI ranks.
    auto make_transport_() -> void {
#ifdef monoprop_ENABLE_MPI
        if (parent_.kind == mpi::Comm::Kind::Mpi && mpi::size(parent_) > 1) {
            hyb_ = std::make_unique<mpi::HybridComm>(parent_.mpi, n_);
            return;
        }
#endif
        shm_ = std::make_unique<mpi::ShmComm>(n_);
    }
    auto comm_for_(int r) -> mpi::Comm {
#ifdef monoprop_ENABLE_MPI
        if (hyb_) {
            return mpi::Comm::make_hybrid(hyb_.get(), r);
        }
#endif
        return mpi::Comm::make_shm(shm_.get(), r);
    }
    auto transport_poison_() -> void {
#ifdef monoprop_ENABLE_MPI
        if (hyb_) {
            hyb_->poison();
            return;
        }
#endif
        shm_->poison();
    }
    auto transport_reset_() -> void {
#ifdef monoprop_ENABLE_MPI
        if (hyb_) {
            hyb_->reset();
            return;
        }
#endif
        shm_->reset();
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
                transport_poison_(); // release peers waiting in a barrier so they don't hang
            }
            {
                std::lock_guard lk(m_);
                ++done_count_;
            }
            cv_done_.notify_one();
        }
    }

    int n_;
    mpi::Comm parent_;                    // enclosing communicator (size R) — decides the transport
    std::unique_ptr<mpi::ShmComm> shm_;   // set iff R == 1
#ifdef monoprop_ENABLE_MPI
    std::unique_ptr<mpi::HybridComm> hyb_; // set iff R > 1
#endif
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
