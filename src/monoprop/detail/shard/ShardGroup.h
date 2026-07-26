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

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h" // mpi::size for the transport choice
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/HybridComm.h"
#endif
#include "monoprop/detail/shard/CpuTopology.h"

// Intra-process shard runtime: owns S single-threaded master threads, each pinned to a core and
// running an independent MonomialPropagator over one hash-partition via a Kind::Shm comm — the
// unchanged SPMD engine an MPI rank runs, with ShmComm standing in for the network. run_on_all must
// fan a call out to ALL masters concurrently, since the engine's collectives are barrier-synced inside ShmComm.

namespace monoprop {

template <size_t NumModes>
class MonomialPropagator; // completed before any ShardGroup member body is instantiated (Impl.h)

namespace detail::shard {

template <size_t NumModes>
class ShardGroup {
public:
    // Builds each shard's propagator via `factory(shard_comm)` ON its master thread, so heap allocations
    // are first-touched on the owning core/CCX (the locality win). `factory` must build a shards=1 propagator.
    using Factory = std::function<std::unique_ptr<MonomialPropagator<NumModes>>(mpi::Comm)>;

    // `parent` is the enclosing communicator (size R): R == 1 ⇒ shards trade over an in-process ShmComm;
    // R > 1 ⇒ a HybridComm folding R ranks x S shards into one flat P=R*S world. Shards run the unchanged
    // engine over a P-partition comm either way.
    ShardGroup(int n_shards, const Factory &factory, mpi::Comm parent)
        : n_(n_shards),
          parent_(parent),
          shards_(static_cast<size_t>(n_shards)),
          errs_(static_cast<size_t>(n_shards)) {
        make_transport_();
        discover_node_peers_();
        cpusets_ = topo_shard_cpusets(n_, node_rank_, node_size_);
        start_masters_();
        // First job: build each shard on its master (pinned, cache-warm on the owning core).
        // The masters are already running, so an exception here (any MonomialPropagator ctor
        // validation) must not escape before they are stopped: ~ShardGroup would never run, stop_ would
        // stay false, and destroying joinable threads during unwinding calls std::terminate.
        try {
            run_on_all([&](int r) { shards_[static_cast<size_t>(r)] = factory(comm_for_(r)); });
        }
        catch (...) {
            stop_and_join_();
            throw;
        }
    }

    // Clone: rebuild this group's transport (fresh threads + ShmComm/HybridComm over the same parent),
    // deep-copy each shard on the new master, then rebind the copy's comm (it inherited src's handle).
    ShardGroup(const ShardGroup &src)
        : n_(src.n_),
          parent_(src.parent_),
          node_rank_(src.node_rank_),
          node_size_(src.node_size_),
          shards_(static_cast<size_t>(src.n_)),
          errs_(static_cast<size_t>(src.n_)) {
        make_transport_();
        cpusets_ = topo_shard_cpusets(n_, node_rank_, node_size_);
        start_masters_();
        try { // see the primary ctor: a throw past live masters would std::terminate
            run_on_all([&](int r) {
                auto p = std::make_unique<MonomialPropagator<NumModes>>(*src.shards_[static_cast<size_t>(r)]);
                p->comm_ = comm_for_(r); // ShardGroup is a friend of MonomialPropagator
                shards_[static_cast<size_t>(r)] = std::move(p);
            });
        }
        catch (...) {
            stop_and_join_();
            throw;
        }
    }
    auto operator=(const ShardGroup &) -> ShardGroup & = delete;

    ~ShardGroup() { stop_and_join_(); }

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
    // Stop every master and join it. Shared by the destructor and the constructors' failure paths, which
    // must not let an exception escape past live threads. Poison first so a master parked in a barrier is
    // released rather than joined-on forever.
    auto stop_and_join_() noexcept -> void {
        transport_poison_();
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

    // Free-function wrapper so the header compiles on non-Linux (where shard_cpusets returns {}).
    static auto topo_shard_cpusets(int n, int group_index, int group_count)
        -> std::vector<monoprop::detail::shard::CpuSet> {
        return monoprop::detail::shard::shard_cpusets(static_cast<size_t>(n),
                                                      static_cast<size_t>(group_index),
                                                      static_cast<size_t>(group_count));
    }

    // Under an MPI parent, find how many ranks share this host and which we are, so each co-located rank
    // pins its shards to a disjoint core block (see shard_cpusets). Collective over `parent`; clones copy
    // the result instead of re-running it, keeping cloning rank-local.
    auto discover_node_peers_() -> void {
#ifdef monoprop_ENABLE_MPI
        if (parent_.kind == mpi::Comm::Kind::Mpi && mpi::size(parent_) > 1) {
            MPI_Comm node = MPI_COMM_NULL;
            MPI_Comm_split_type(parent_.mpi, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
            MPI_Comm_rank(node, &node_rank_);
            MPI_Comm_size(node, &node_size_);
            MPI_Comm_free(&node);
        }
#endif
    }

    // Build the shared transport: ShmComm for a single-rank parent, HybridComm when the parent spans R>1 ranks.
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
    mpi::Comm parent_;                  // enclosing communicator (size R) — decides the transport
    int node_rank_ = 0;                 // this rank's index among the ranks sharing the host
    int node_size_ = 1;                 // how many parent ranks share the host (1 unless MPI R>1)
    std::unique_ptr<mpi::ShmComm> shm_; // set iff R == 1
#ifdef monoprop_ENABLE_MPI
    std::unique_ptr<mpi::HybridComm> hyb_; // set iff R > 1
#endif
    std::vector<std::unique_ptr<MonomialPropagator<NumModes>>> shards_;
    std::vector<std::exception_ptr> errs_;
    std::vector<monoprop::detail::shard::CpuSet> cpusets_;
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
