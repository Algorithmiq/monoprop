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

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <exception>
#include <format>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/detail/parallel/Options.h"

/*
 * Operation-scoped team and phase protocol for sharded execution.
 *
 * One OpenMP team of exactly T workers spans a whole operation. Worker t owns shard t for the duration of one
 * run_team() call; this is an OpenMP team index, not a persistent OS-thread identity across calls. The
 * orchestration body runs a common sequence of phases on every worker, each ending in a collective checkpoint.
 *
 * Failure protocol, for a TeamFailure shared by the team. A checkpoint is one barrier; its generation is the number
 * of checkpoints the calling worker has passed before it, identical on every worker because all of them run the same
 * phase sequence.
 *
 * - Error slots. Slot t is written only by worker t, through record() inside its own phase, before that
 *   worker arrives at the checkpoint's barrier. Nobody else writes it during the region.
 * - Stamping. record() also lowers the shared first-failed generation to the recording worker's current generation,
 *   before that worker arrives at the barrier. The value only ever decreases.
 * - Decision. After the barrier, each worker reads the first-failed generation and proceeds past checkpoint N if and
 *   only if it is greater than N. The barrier orders every stamp of phase N before every read for N.
 * - Generations. A fast worker that leaves checkpoint N and fails in phase N + 1 stamps N + 1, which is still greater
 *   than N, so a slower worker still reading for N sees the same decision. A failure in phase N + 1 is seen by
 *   checkpoint N + 1, which every worker reaches because it proceeds past N. Once a checkpoint fails, every later
 *   generation is greater than the stamp, so no later checkpoint can proceed.
 * - Join. Errors are delivered only after the implicit barrier ending the parallel region, when every worker
 *   has returned from the orchestration body. No worker can still be inside a phase, touching buffers or
 *   waiting at a barrier, when the caller inspects or acts on the error.
 *
 * run_team() returns the error instead of rethrowing it: a future caller may still own posted MPI requests
 * and the buffers they reference, and must decide on invalidation, abort or rethrow before any destructor
 * can wait on a peer. This header performs no MPI, owns no threads, queues or persistent state, and never
 * changes an OpenMP runtime setting.
 */

namespace monoprop::detail::sharded {

//! The default checkpoint observer: does nothing.
struct NoCheckpointObserver {
    //! Ignore the checkpoint generation.
    auto operator()(std::size_t /*generation*/) const noexcept -> void {}
};

/*!
 * \brief Operation-local failure state for one team: one error slot per worker and the first failed generation.
 *
 * Constructed by run_team() before its parallel region; not copyable or movable, so the team always shares
 * one instance. The ownership, publication and generation rules are described at the top of this header.
 */
class TeamFailure {
public:
    /*!
     * \brief Allocate one empty error slot per worker.
     * \param threads The team size T.
     * \throws std::bad_alloc if the slots cannot be allocated.
     */
    explicit TeamFailure(std::size_t threads) : slots_(threads) {}

    TeamFailure(const TeamFailure &) = delete;                     //!< The team shares one instance.
    auto operator=(const TeamFailure &) -> TeamFailure & = delete; //!< The team shares one instance.
    TeamFailure(TeamFailure &&) = delete;                          //!< The team shares one instance.
    auto operator=(TeamFailure &&) -> TeamFailure & = delete;      //!< The team shares one instance.
    ~TeamFailure() = default;                                      //!< Releases the captured errors.

    /*!
     * \brief Record a worker's exception in its own slot; a slot keeps its first nonempty error.
     *
     * Inside a team, only worker `shard` may call this, and only before it reaches the next checkpoint. Lowers the
     * first failed generation to that worker's current one, so the next checkpoint fails on every worker. An empty
     * `error` records nothing.
     *
     * \param shard The calling worker's index; must be less than the team size.
     * \param error The captured exception.
     */
    auto record(std::size_t shard, std::exception_ptr error) noexcept -> void {
        if (!error) {
            return;
        }
        auto &slot = slots_[shard];
        if (!slot.error) {
            slot.error = std::move(error);
        }
        auto first = first_failed_.load(std::memory_order_relaxed);
        while (slot.generation < first
               && !first_failed_.compare_exchange_weak(first, slot.generation, std::memory_order_release)) {
        }
    }

    /*!
     * \brief Collective checkpoint: every worker of the team must call it, in the same phase sequence.
     *
     * Must be called directly from the orchestration body (or code it calls), never inside a worksharing,
     * `single`, `masked` or `critical` construct, and never from a nested region. Outside any parallel
     * region it binds to a one-thread team, whose worker is 0.
     *
     * \param after_barrier Test-only observer, called as `after_barrier(generation)` on each worker between the
     *        barrier and its read of the decision, so tests can delay that read deterministically. Must not throw.
     * \return True, identically on every worker, if no error was recorded at or before this checkpoint's
     *         generation.
     */
    template <class Observer = NoCheckpointObserver>
    auto checkpoint(Observer &&after_barrier = {}) noexcept -> bool {
        // The barrier orders every record() of this phase before any worker's read below. A one-worker team has nothing
        // to order and runs without a region (run_team()), so it skips the barrier's runtime call.
        if (slots_.size() > 1) {
#pragma omp barrier
        }
        auto &slot = slots_[static_cast<std::size_t>(omp_get_thread_num())];
        const auto generation = slot.generation++;
        std::invoke(after_barrier, generation);
        return first_failed_.load(std::memory_order_acquire) > generation;
    }

    /*!
     * \brief The error of the lowest-numbered failing worker, or an empty pointer.
     *
     * Call only after the team has joined, or outside any parallel region.
     */
    [[nodiscard]] auto first_error() const noexcept -> std::exception_ptr {
        const auto found = std::ranges::find_if(slots_, [](const Slot &slot) { return static_cast<bool>(slot.error); });
        return found == slots_.end() ? std::exception_ptr{} : found->error;
    }

private:
    // One cache line per slot: each worker advances its own generation at every checkpoint.
    static constexpr std::size_t slot_alignment = 64;
    static constexpr std::size_t no_failure = std::numeric_limits<std::size_t>::max();

    //! One worker's failure state.
    struct alignas(slot_alignment) Slot {
        std::exception_ptr error;   //!< The worker's first recorded error.
        std::size_t generation = 0; //!< Checkpoints this worker has passed.
    };

    std::vector<Slot> slots_;                           //!< Slot t: written by worker t only; see the header comment.
    std::atomic<std::size_t> first_failed_{no_failure}; //!< Lowest generation with a recorded error.
};

/*!
 * \brief An orchestration body for run_team(): noexcept, callable as `body(shard, failure)`.
 *
 * Anything that may throw belongs inside phase(); an exception escaping the body terminates the program.
 */
template <class Fn>
concept TeamBody = std::is_nothrow_invocable_v<Fn &, std::size_t, TeamFailure &>;

/*!
 * \brief Run `body(shard, failure)` once on every worker of one OpenMP team of `options.threads` workers.
 *
 * Allocates the failure state, then opens one `omp parallel num_threads(options.threads)` region. Each worker
 * calls the body once with its OpenMP worker index, which stays fixed for that call, and the shared
 * TeamFailure. Worker 0 is the primary: the thread that called run_team(). The body is invoked concurrently
 * on the same object, so it must not mutate its own state without synchronization.
 *
 * A one-worker team opens no region: the caller runs the body directly as worker 0, at its own OpenMP level, and
 * checkpoints skip their barrier. The outcome is the same as a one-thread region's; only the runtime's per-region and
 * per-barrier cost, a fixed cost on every phase of a small operation, is saved.
 *
 * Preconditions, not checked: the caller is outside any OpenMP region, and the launch supplies exactly
 * `options.threads` workers (`OMP_DYNAMIC=FALSE`, no thread limit below the budget). There is no reduced-team
 * fallback; unlike parallel::for_blocks(), correctness requires the full team.
 *
 * \param options Requested team size; `options.threads` must be positive.
 * \param body    The noexcept orchestration body, a TeamBody.
 * \return After every worker has returned from the body and the team has joined: an empty pointer on
 *         success, otherwise the original exception of the lowest-numbered failing worker. Phase exceptions
 *         are never rethrown here.
 * \throws std::invalid_argument if `options.threads` is not positive, before the region.
 * \throws std::bad_alloc if the failure state cannot be allocated, before the region.
 */
template <class Fn>
    requires TeamBody<Fn>
auto run_team(parallel::Options options, Fn &&body) -> std::exception_ptr {
    if (options.threads < 1) {
        throw std::invalid_argument(
            std::format("sharded::run_team: the team size must be positive, got {}", options.threads));
    }
    auto failure = TeamFailure(static_cast<std::size_t>(options.threads));
    if (options.threads == 1) {
        std::invoke(body, std::size_t{0}, failure);
        return failure.first_error();
    }
#pragma omp parallel num_threads(options.threads)
    {
        std::invoke(body, static_cast<std::size_t>(omp_get_thread_num()), failure);
    }
    return failure.first_error();
}

/*!
 * \brief Run one worker's local part of a phase, then the collective checkpoint.
 *
 * Invokes `body()` and catches every exception, including values not derived from `std::exception`,
 * recording it in the calling worker's slot. The worker then reaches the checkpoint whether its body had
 * work, had none or failed.
 *
 * Every worker must call phase() for every phase of the common sequence and check the result: false ends the
 * sequence on every worker, and no later phase body may run. The body must not contain synchronization
 * (barriers, locks another worker needs, MPI waits) that an exception could bypass.
 *
 * \param failure The team's failure state, as passed to the orchestration body.
 * \param shard   The calling worker's index.
 * \param body    The local phase work, callable with no arguments.
 * \return The checkpoint decision, identical on every worker.
 */
template <class Fn>
    requires std::invocable<Fn &>
auto phase(TeamFailure &failure, std::size_t shard, Fn &&body) noexcept -> bool {
    try {
        std::invoke(body);
    }
    catch (...) {
        failure.record(shard, std::current_exception());
    }
    return failure.checkpoint();
}

/*!
 * \brief Admits at most a fixed number of a team's workers into a section at once.
 *
 * For memory, not correctness: a section whose transient footprint grows with the number of workers inside it (each
 * copying a large block, say) is entered by at most `limit` workers, while the others yield until a slot frees. The
 * order of entry is unspecified, and every worker that calls run() enters exactly once.
 *
 * Shared by the team for one section; not copyable or movable. The section must not synchronize with other workers
 * (no barrier, no wait on another worker's progress), or the workers queued outside it could never be admitted.
 */
class SectionLimit {
public:
    //! A limit of zero admits one worker at a time.
    explicit SectionLimit(std::size_t limit) noexcept
        : free_(static_cast<std::ptrdiff_t>(std::max<std::size_t>(limit, 1))) {}
    SectionLimit(const SectionLimit &) = delete;
    auto operator=(const SectionLimit &) -> SectionLimit & = delete;

    /*!
     * \brief Wait for a slot, run `section` and release the slot, also when `section` throws.
     * \return What `section` returns.
     */
    template <class Fn>
        requires std::invocable<Fn &>
    auto run(Fn &&section) -> decltype(auto) {
        for (auto free = free_.load(std::memory_order_relaxed);;) {
            if (free > 0) {
                if (free_.compare_exchange_weak(free, free - 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                    break;
                }
                continue;
            }
            std::this_thread::yield();
            free = free_.load(std::memory_order_relaxed);
        }
        struct Release {
            std::atomic<std::ptrdiff_t> &free;
            ~Release() { free.fetch_add(1, std::memory_order_release); }
        } release{free_};
        return std::invoke(section);
    }

private:
    std::atomic<std::ptrdiff_t> free_; // slots not held
};

} // namespace monoprop::detail::sharded
