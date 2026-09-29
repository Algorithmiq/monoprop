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
#include <concepts>
#include <cstddef>
#include <exception>
#include <format>
#include <functional>
#include <stdexcept>
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
 * Failure protocol, for a TeamFailure shared by the team:
 *
 * - Error slots. Slot t is written only by worker t, through record() inside its own phase, before that
 *   worker arrives at the checkpoint's leading barrier. Nobody else writes it during the region.
 * - Inspection. The slots are read only by the primary thread, between the leading and trailing barriers of
 *   a checkpoint, and by the caller after run_team() has joined the team. The leading barrier orders every
 *   owner's write before the primary's read.
 * - Publication. The primary writes the checkpoint's decision to one shared flag between the two barriers;
 *   the trailing barrier orders that write before every worker's read.
 * - Generations. The decision for checkpoint N is read by each worker after the trailing barrier of N and
 *   before it arrives at the leading barrier of N + 1. The primary cannot write the decision for N + 1 until
 *   every worker has passed that leading barrier, so a fast worker that fails in phase N + 1 cannot change
 *   the value a slower worker is still reading for N. A failure recorded in phase N + 1 is seen only by
 *   checkpoint N + 1, which every worker reaches because it proceeds past N.
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

/*!
 * \brief Operation-local failure state for one team: one error slot per worker and a checkpoint decision.
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
    explicit TeamFailure(std::size_t threads) : errors_(threads) {}

    TeamFailure(const TeamFailure &) = delete;                     //!< The team shares one instance.
    auto operator=(const TeamFailure &) -> TeamFailure & = delete; //!< The team shares one instance.
    TeamFailure(TeamFailure &&) = delete;                          //!< The team shares one instance.
    auto operator=(TeamFailure &&) -> TeamFailure & = delete;      //!< The team shares one instance.
    ~TeamFailure() = default;                                      //!< Releases the captured errors.

    /*!
     * \brief Record a worker's exception in its own slot; a slot keeps its first nonempty error.
     *
     * Inside a team, only worker `shard` may call this, and only before it reaches the next checkpoint.
     * An empty `error` records nothing.
     *
     * \param shard The calling worker's index; must be less than the team size.
     * \param error The captured exception.
     */
    auto record(std::size_t shard, std::exception_ptr error) noexcept -> void {
        auto &slot = errors_[shard];
        if (!slot) {
            slot = std::move(error);
        }
    }

    /*!
     * \brief Collective checkpoint: every worker of the team must call it, in the same phase sequence.
     *
     * Must be called directly from the orchestration body (or code it calls), never inside a worksharing,
     * `single`, `masked` or `critical` construct, and never from a nested region. Outside any parallel
     * region it binds to a one-thread team and simply inspects the slots.
     *
     * \return True, identically on every worker, if no slot holds an error at this checkpoint.
     */
    auto checkpoint() noexcept -> bool {
        // Leading barrier: every owner's record() for this phase happens before the primary reads the slots.
#pragma omp barrier
#pragma omp masked
        {
            proceed_ = std::ranges::none_of(errors_, [](const auto &error) { return static_cast<bool>(error); });
        }
        // Trailing barrier: the decision is published before any worker reads it. The next write waits for
        // the next leading barrier, which no worker reaches before it has read this value.
#pragma omp barrier
        return proceed_;
    }

    /*!
     * \brief The error of the lowest-numbered failing worker, or an empty pointer.
     *
     * Call only after the team has joined, or outside any parallel region.
     */
    [[nodiscard]] auto first_error() const noexcept -> std::exception_ptr {
        const auto found = std::ranges::find_if(errors_, [](const auto &error) { return static_cast<bool>(error); });
        return found == errors_.end() ? std::exception_ptr{} : *found;
    }

private:
    std::vector<std::exception_ptr> errors_; //!< Slot t: written by worker t only; see the header comment.
    bool proceed_ = true;                    //!< Current checkpoint decision; written by the primary only.
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

} // namespace monoprop::detail::sharded
