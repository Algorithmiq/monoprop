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

#include "monoprop/Threading.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "monoprop/detail/EnvConfig.h"

// The persistent thread pool behind threading::run_static. Deliberately small and boring:
//   • ONE primitive — run n_tasks tasks, claimed off a shared atomic counter (countdown completion).
//     No work stealing, no dynamic range splitting; callers pick their chunking (Threading.h
//     wrappers, Parallel.h chunk policies).
//   • The CALLER participates as a worker, so parallelism P needs only P-1 pool threads and a job
//     always makes progress even if every worker is busy elsewhere.
//   • A thread_local nesting guard makes nested parallel calls run inline and serial.
//   • Workers spin briefly between jobs (gate loops issue many small regions back-to-back) and then
//     park on a condition variable, so an idle pool consumes no CPU. Under the shard runtime — the
//     default engine — every dispatch site sees effective_parallelism() == 1 and no job is ever
//     submitted, so the pool threads are never even started.
//   • Each job is a shared_ptr-owned heap object with its OWN claim/done counters: a worker that
//     wakes late can only ever touch an exhausted job, never a recycled one. A task body that
//     throws terminates the process (workers have no exception channel); parallel loop bodies in
//     this codebase do not throw.
namespace monoprop::threading {
namespace {

auto hardware_parallelism() -> size_t {
#if defined(__linux__)
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        const int count = CPU_COUNT(&mask);
        if (count > 0) {
            return static_cast<size_t>(count);
        }
    }
#endif
    return std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
}

// Configured maximum parallelism (monoprop_NUM_THREADS via init_from_env, else the CPU budget);
// 0 = not yet resolved. Resolved lazily so an early ScopedParallelismCap cannot mis-size the pool.
std::atomic<size_t> g_configured{0};
// Live ScopedParallelismCap value; 0 = none. Process-wide, like the scoped global control it replaces.
std::atomic<size_t> g_scoped_cap{0};
std::once_flag g_init_once;

auto configured_parallelism() -> size_t {
    size_t configured = g_configured.load(std::memory_order_relaxed);
    if (configured == 0) {
        // Benign race: concurrent resolvers store the same default; an explicit monoprop_NUM_THREADS
        // (init_from_env runs before any parallel work) wins via the compare_exchange failure path.
        size_t expected = 0;
        g_configured.compare_exchange_strong(expected, hardware_parallelism(), std::memory_order_relaxed);
        configured = g_configured.load(std::memory_order_relaxed);
    }
    return configured;
}

inline auto cpu_pause() -> void {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

// True while the calling thread is executing a pool task (worker threads permanently, the submitting
// caller during its participation) — the nesting guard that keeps inner parallel calls inline.
thread_local bool t_in_pool_task = false;

// One parallel region: tasks [0, n) claimed off `next`, completion counted in `done`. Jointly owned
// (shared_ptr) by the submitter and any worker that woke for it.
struct Job final {
    void (*task)(void *, size_t) = nullptr;
    void *ctx = nullptr;
    size_t n = 0;
    size_t worker_limit = 0; // pool workers with id >= limit sit this job out (ScopedParallelismCap)
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};

    auto participate() -> void {
        for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < n;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
            // Count the claim resolved even if the task throws, so the countdown always completes.
            // On a WORKER an escaping exception still terminates (thread boundary); on the CALLER it
            // propagates out of run() after the join (see Pool::run).
            try {
                task(ctx, i);
            }
            catch (...) {
                done.fetch_add(1, std::memory_order_release);
                throw;
            }
            done.fetch_add(1, std::memory_order_release);
        }
    }
};

class Pool final {
public:
    static auto instance() -> Pool & {
        static Pool pool;
        return pool;
    }

    auto run(size_t n_tasks, void (*task)(void *, size_t), void *ctx) -> void {
        auto job = std::make_shared<Job>();
        job->task = task;
        job->ctx = ctx;
        job->n = n_tasks;
        job->worker_limit = current_max_parallelism() - 1;
        {
            const std::lock_guard lock(mutex_);
            start_workers_locked_();
            job_ = job;
            ++epoch_;
            epoch_hint_.store(epoch_, std::memory_order_release);
        }
        cv_.notify_all();
        t_in_pool_task = true;
        try {
            job->participate();
        }
        catch (...) {
            // Unwind-safely: restore the nesting guard and JOIN the job before rethrowing, so no
            // worker can still be touching caller-owned state (fn, outputs) during the unwind.
            t_in_pool_task = false;
            wait_until_done_(*job);
            throw;
        }
        t_in_pool_task = false;
        wait_until_done_(*job);
    }

    Pool(const Pool &) = delete;
    auto operator=(const Pool &) -> Pool & = delete;

private:
    // ~30–100 µs of polling before yielding (caller) or parking (workers): long enough to bridge the
    // serial gap between the many small per-gate regions, short enough that an idle pool is silent.
    static constexpr size_t kActiveSpins = size_t{1} << 14;

    Pool() = default;

    ~Pool() {
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &worker : workers_) {
            worker.join();
        }
    }

    // Atomic-countdown completion: returns once every task has run (the release increments in
    // participate() make all task side effects visible here). Stragglers that wake later can only
    // ever observe an exhausted job.
    static auto wait_until_done_(const Job &job) -> void {
        for (size_t spin = 0; job.done.load(std::memory_order_acquire) != job.n; ++spin) {
            if (spin < kActiveSpins) {
                cpu_pause();
            }
            else {
                std::this_thread::yield();
            }
        }
    }

    auto start_workers_locked_() -> void {
        if (!workers_.empty()) {
            return;
        }
        // Sized once from the configured parallelism (NOT any live scoped cap): a later, larger
        // region can use every configured core even if the first parallel region ran capped.
        const size_t n_workers = configured_parallelism() - 1; // the caller participates too
        workers_.reserve(n_workers);
        for (size_t id = 0; id < n_workers; ++id) {
            workers_.emplace_back([this, id] { worker_loop_(id); });
        }
    }

    auto worker_loop_(size_t id) -> void {
        t_in_pool_task = true; // everything a pool worker runs is pool work: nested calls go inline
        uint64_t seen = 0;
        std::unique_lock lock(mutex_);
        for (;;) {
            if (stop_) {
                return;
            }
            if (epoch_ != seen) {
                seen = epoch_;
                std::shared_ptr<Job> job = job_;
                lock.unlock();
                if (id < job->worker_limit) {
                    job->participate();
                }
                job.reset();
                // Poll briefly for the next region before parking (see kActiveSpins).
                for (size_t spin = 0; spin < kActiveSpins; ++spin) {
                    if (epoch_hint_.load(std::memory_order_acquire) != seen) {
                        break;
                    }
                    cpu_pause();
                }
                lock.lock();
                continue;
            }
            cv_.wait(lock);
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;    // guarded by mutex_ (only mutated once, at start)
    std::shared_ptr<Job> job_;            // guarded by mutex_
    uint64_t epoch_ = 0;                  // guarded by mutex_
    std::atomic<uint64_t> epoch_hint_{0}; // lock-free mirror of epoch_ for the workers' spin phase
    bool stop_ = false;                   // guarded by mutex_
};

// Set the configured maximum parallelism. File-local: the sole entry point is init_from_env reading
// monoprop_NUM_THREADS. Thread-safe; threads <= 0 ignored. Takes full effect only before the first
// parallel region (the pool sizes itself once, at first use).
auto set_num_threads(int threads) -> void {
    if (threads <= 0) {
        return;
    }
    g_configured.store(static_cast<size_t>(threads), std::memory_order_relaxed);
}

} // namespace

auto init_from_env() -> void {
    std::call_once(g_init_once, []() {
        const auto threads = config::get().num_threads;
        if (threads.has_value()) {
            set_num_threads(*threads);
        }
    });
}

auto current_max_parallelism() -> size_t {
    const size_t configured = configured_parallelism();
    const size_t cap = g_scoped_cap.load(std::memory_order_relaxed);
    return cap == 0 ? configured : std::min(configured, cap);
}

ScopedParallelismCap::ScopedParallelismCap(size_t cap)
    : previous_(g_scoped_cap.exchange(std::max<size_t>(1, cap), std::memory_order_relaxed)) {}

ScopedParallelismCap::~ScopedParallelismCap() {
    g_scoped_cap.store(previous_, std::memory_order_relaxed);
}

auto run_static_impl(size_t n_tasks, void (*task)(void *, size_t), void *ctx) -> void {
    if (n_tasks == 0) {
        return;
    }
    if (t_in_pool_task || n_tasks == 1 || effective_parallelism() <= 1) {
        for (size_t i = 0; i < n_tasks; ++i) {
            task(ctx, i);
        }
        return;
    }
    Pool::instance().run(n_tasks, task, ctx);
}

auto range_grain_size(size_t count, size_t min_grain) -> size_t {
    const size_t workers = effective_parallelism();
    const size_t scaled = count / std::max<size_t>(1, workers * 4);
    return std::max(min_grain, scaled);
}

} // namespace monoprop::threading
