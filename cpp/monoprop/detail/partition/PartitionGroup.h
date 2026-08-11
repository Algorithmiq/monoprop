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
#include <type_traits>
#include <vector>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h" // mpi::size for the transport choice
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/HybridComm.h"
#endif
#include "monoprop/detail/partition/CpuTopology.h"

// Intra-process partition runtime: S master threads, each pinned to a core and running an independent
// MonomialPropagator over one hash partition, with an in-process comm standing in for the network.
// run_on_all must fan out to all masters: the collectives are barrier-synced.

namespace monoprop {

template <size_t NumModes>
class MonomialPropagator; // completed before any PartitionGroup member body is instantiated (Impl.h)

namespace detail::partition {

template <size_t NumModes>
class PartitionGroup {
public:
    // Builds each partition's propagator via `factory(partition_comm)` ON its master thread, so heap allocations
    // are first-touched on the owning core. `factory` must build a partitions=1 propagator.
    using Factory = std::function<std::unique_ptr<MonomialPropagator<NumModes>>(mpi::Comm)>;

    // `parent` is the enclosing communicator (size R): R == 1 ⇒ an in-process ShmComm; R > 1 ⇒ a
    // HybridComm folding R ranks x S partitions into one flat P=R*S world.
    PartitionGroup(int n_partitions, const Factory &factory, mpi::Comm parent)
        : n_(n_partitions),
          parent_(parent),
          partitions_(static_cast<size_t>(n_partitions)),
          errs_(static_cast<size_t>(n_partitions)) {
        // Placement is decided before the transport, because the transport's barrier is grouped by the
        // L3 domain each partition will be pinned to.
        discover_node_peers_();
        cpusets_ = topo_partition_cpusets(n_, node_rank_, node_size_);
        make_transport_();
        start_masters_();
        // The masters are already running, so a ctor throw must not escape: ~PartitionGroup would never run,
        // and destroying joinable threads during unwinding calls std::terminate.
        try {
            run_on_all([&](int r) { partitions_[static_cast<size_t>(r)] = factory(comm_for_(r)); });
        }
        catch (...) {
            stop_and_join_();
            throw;
        }
    }

    // Fresh transport and threads over the same parent; each partition is deep-copied on its new master, then
    // its comm rebound (the copy inherited src's handle).
    PartitionGroup(const PartitionGroup &src)
        : n_(src.n_),
          parent_(src.parent_),
          node_rank_(src.node_rank_),
          node_size_(src.node_size_),
          partitions_(static_cast<size_t>(src.n_)),
          errs_(static_cast<size_t>(src.n_)) {
        cpusets_ = topo_partition_cpusets(n_, node_rank_, node_size_);
        make_transport_(); // after cpusets_, as in the primary ctor
        start_masters_();
        try { // see the primary ctor: a throw past live masters would std::terminate
            run_on_all([&](int r) {
                auto p = std::make_unique<MonomialPropagator<NumModes>>(*src.partitions_[static_cast<size_t>(r)]);
                p->comm_ = comm_for_(r); // PartitionGroup is a friend of MonomialPropagator
                partitions_[static_cast<size_t>(r)] = std::move(p);
            });
        }
        catch (...) {
            stop_and_join_();
            throw;
        }
    }
    auto operator=(const PartitionGroup &) -> PartitionGroup & = delete;

    ~PartitionGroup() { stop_and_join_(); }

    auto partition_count() const -> int { return n_; }
    auto partition(int s) -> MonomialPropagator<NumModes> & { return *partitions_[static_cast<size_t>(s)]; }
    auto partition(int s) const -> const MonomialPropagator<NumModes> & { return *partitions_[static_cast<size_t>(s)]; }

    // Run `body(partition_rank)` on all masters, block until every one finishes, then rethrow the first
    // exception raised (peers were released via poison, so a throw on one master never hangs the rest).
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
    // Poison first so a master parked in a barrier is released rather than joined-on forever.
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

    // Free-function wrapper so the header compiles on non-Linux (where partition_cpusets returns {}).
    static auto topo_partition_cpusets(int n, int group_index, int group_count)
        -> std::vector<monoprop::detail::partition::CpuSet> {
        return monoprop::detail::partition::partition_cpusets(static_cast<size_t>(n),
                                                              static_cast<size_t>(group_index),
                                                              static_cast<size_t>(group_count));
    }

    // Under an MPI parent, find how many ranks share this host and which we are, so each co-located rank
    // pins to a disjoint core block (see partition_cpusets). Collective over `parent`; clones copy the result.
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

    auto make_transport_() -> void {
        // Empty unless the partitions are pinned and /sys was readable ⇒ flat barrier (see
        // PartitionBarrier); cpusets_ must already be set.
        const std::vector<int> domains = monoprop::detail::partition::cpuset_l3_domains(cpusets_);
#ifdef monoprop_ENABLE_MPI
        if (parent_.kind == mpi::Comm::Kind::Mpi && mpi::size(parent_) > 1) {
            hyb_ = std::make_unique<mpi::HybridComm>(parent_.mpi, n_, domains);
            return;
        }
#endif
        shm_ = std::make_unique<mpi::ShmComm>(n_, domains);
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

    // Spawning is itself a failure point: one std::thread per partition is O(100) threads, and a mid-loop
    // std::system_error must not escape with live masters parked in masters_ (see the primary ctor).
    auto start_masters_() -> void {
        masters_.reserve(static_cast<size_t>(n_));
        try {
            for (int r = 0; r < n_; ++r) {
                masters_.emplace_back([this, r] { master_loop_(r); });
            }
        }
        catch (...) {
            stop_and_join_();
            throw;
        }
    }

    auto master_loop_(int rank) -> void {
        if (!cpusets_.empty()) {
            pin_this_thread(cpusets_[static_cast<size_t>(rank)]);
        }
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
                transport_poison_();
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
    std::vector<std::unique_ptr<MonomialPropagator<NumModes>>> partitions_;
    std::vector<std::exception_ptr> errs_;
    std::vector<monoprop::detail::partition::CpuSet> cpusets_;
    std::vector<std::thread> masters_;

    // Job dispatch: the facade thread publishes one job and waits for all masters to complete it.
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int)> *job_ = nullptr;
    unsigned job_gen_ = 0;
    int done_count_ = 0;
    bool stop_ = false;
};

// One result per partition, indexed by partition rank. The slots are written from the owning master, so
// `body` must not touch the vector itself.
template <size_t NumModes, typename Body, typename R = std::invoke_result_t<Body &, int>>
auto collect_on_all(PartitionGroup<NumModes> &group, Body body) -> std::vector<R> {
    std::vector<R> results(static_cast<size_t>(group.partition_count()));
    group.run_on_all([&](int r) { results[static_cast<size_t>(r)] = body(r); });
    return results;
}

// collect_on_all over the partition propagators themselves: `body(partition)` on each partition's master.
template <size_t NumModes, typename Body, typename R = std::invoke_result_t<Body &, MonomialPropagator<NumModes> &>>
auto map_partitions(PartitionGroup<NumModes> &group, Body body) -> std::vector<R> {
    return collect_on_all(group, [&](int r) -> R { return body(group.partition(r)); });
}

} // namespace detail::partition
} // namespace monoprop
