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

// Per-partition attribution of the layer build, the replay and the collectives, compiled only under
// monoprop_ENABLE_PROFILE; within such a build monoprop_PROFILE selects the regions (EnvConfig.h names
// them, docs parallelism.mdx maps them to lines). `layer_ns` brackets all of build_layer, so layer_ns
// minus the phases is the unattributed remainder. Registry-owned slots outlive their thread. Output is
// stderr, whose fd-level capture needs `pytest -s`. CommRegistry is the one type a call site still
// names, because a transport holds one as a member; everything else comes through the macros below.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "monoprop/detail/EnvConfig.h"

#ifdef monoprop_ENABLE_PROFILE
#include <sys/resource.h>
#include <chrono>
#include <cstddef>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#endif

namespace monoprop::detail::profile {

#ifdef monoprop_ENABLE_PROFILE

inline auto to_s(uint64_t ns) -> double {
    return static_cast<double>(ns) / 1e9;
}

// One fwrite per line, so concurrent masters cannot interleave; the format literal is checked against
// the pack; noexcept because two callers are destructors that can run while unwinding.
template <typename... Args>
auto write_line(std::FILE *out, std::format_string<Args...> fmt, Args &&...args) noexcept -> void {
    try {
        const std::string line = std::format(fmt, std::forward<Args>(args)...);
        (void)std::fwrite(line.data(), 1, line.size(), out);
    }
    catch (...) { // a diagnostic that cannot be written must not take the process with it
    }
}

struct alignas(64) Slot {
    // `index_ns` covers the inverted-index rebuild/append, charged per gate but not a layer phase.
    // fold (container decode) and scan (the word loop) are split because only the fold changed between
    // index designs, so a container that moves scan_ns is measuring something else.
    uint64_t fold_ns = 0;
    uint64_t scan_ns = 0;
    uint64_t emit_ns = 0;
    uint64_t index_ns = 0;
    uint64_t resolve_ns = 0;
    uint64_t insert_ns = 0;
    uint64_t apply_ns = 0;
    // encode_ns is INSIDE build_layer, a child of layer_ns and zero on propagate; its children split the
    // graph store as the structures do (layout: exchange layouts, indexed by the FLAT world P = ranks x
    // partitions; pack: the P slot records). The drain loop between them is untimed.
    uint64_t encode_ns = 0;
    uint64_t encode_layout_ns = 0;
    uint64_t encode_pack_ns = 0;
    // OUTSIDE build_layer: a sibling of apply_ns under gate_ns, and a per-gate constant.
    uint64_t append_ns = 0;
    // total_ns brackets the gate loop incl. caches, gate_ns the callback, contract_ns the caches step.
    uint64_t total_ns = 0;
    uint64_t gate_ns = 0;
    uint64_t contract_ns = 0;
    uint64_t cache_op_ns = 0;     // get_operator(): binding init_op_map's terms to rows
    uint64_t cache_state_ns = 0;  // sparse/dense state: scoring the newly fully-paired rows
    uint64_t cache_shrink_ns = 0; // op_coeffs + state shrink_to_fit
    // Cross-rank legs, all inert at R == 1.
    uint64_t sendbuf_ns = 0; // packing queries (+values) into the contiguous send stream
    // The two alltoallv legs: on the GRAPH path the response payload is read only inside an assert.
    uint64_t exchange_ns = 0;      // inside MPI: the QUERY leg (this rank's queries out, others' in)
    uint64_t exchange_resp_ns = 0; // inside MPI: the RESPONSE leg (resolved answers out, in)
    uint64_t incoming_ns = 0;      // resolving other ranks' queries against this store
    uint64_t layer_ns = 0;

    uint64_t n_gates = 0;
    uint64_t n_anti = 0; // terms the fold marked anticommuting
    uint64_t n_foll = 0; // of those, followers
    uint64_t n_emit = 0; // reached emit_term_products (i.e. passed the dynamic gate)
    // `n_atol` is refused before any row is touched, `n_reject` after a partner was built and thrown away.
    uint64_t n_atol = 0;
    uint64_t n_reject = 0;
    // How many 64-term words held even ONE survivor: far below n_words, the operator could be sparse.
    uint64_t n_words = 0;
    uint64_t n_live_words = 0;
    // The dedup table's growth rehash: not a layer phase, and charged to `exchange` on every peer.
    uint64_t rehash_ns = 0;
    uint64_t n_rehash = 0;      // growth events
    uint64_t n_rehash_rows = 0; // entries re-placed, summed over those events
    uint64_t n_push = 0;        // query records pushed
    uint64_t n_hit = 0;         // self-resolve hits
    uint64_t n_miss = 0;        // self-resolve misses (deferred inserts)
    uint64_t query_words = 0;

    /* `replay` shares this Slot through its OWN accessor, so a replay run arms no layer timer above: ev
     * runs build_layer inside itself. These NEST and print flat -- ev > evolve > {cos_lazy, cos_mask},
     * grad > {evolve, deriv}, deriv > {cos_lazy, cos_mask}, `pare` disjoint -- so a sum double-counts. */
    uint64_t ev_ns = 0;
    uint64_t grad_ns = 0;
    uint64_t evolve_ns = 0;   // the forward operator evolution, entered by both ev and ev_and_grad
    uint64_t deriv_ns = 0;    // ev_and_grad's reverse per-parameter derivative loop
    uint64_t pare_ns = 0;     // pare_graph, which is neither
    uint64_t cos_lazy_ns = 0; // recomputed from the index (scale_/accumulate_cos_lazy)
    uint64_t cos_mask_ns = 0; // read from a stored per-layer bitmap (scale_/accumulate_cos_mask)
    uint64_t n_ev = 0;
    uint64_t n_grad = 0;
    uint64_t n_pare = 0;
    uint64_t n_cos_lazy = 0; // layer-cosine applications, the denominator for cos_lazy_ns
    uint64_t n_cos_mask = 0;

    // Member by member, not an array walk: pointer arithmetic across distinct members is not allowed.
    [[nodiscard]] auto any_nonzero() const -> bool {
        return (fold_ns | scan_ns | emit_ns | index_ns | resolve_ns | insert_ns | apply_ns | total_ns | gate_ns
                | contract_ns | cache_op_ns | cache_state_ns | cache_shrink_ns | sendbuf_ns | exchange_ns
                | exchange_resp_ns | incoming_ns | layer_ns | encode_ns | encode_layout_ns | encode_pack_ns | append_ns
                | n_gates | n_anti | n_foll | n_emit | n_atol | n_reject | n_words | n_live_words | rehash_ns | n_rehash
                | n_rehash_rows | n_push | n_hit | n_miss | query_words)
               != 0;
    }

    // Separate from any_nonzero(): with `layer,replay` on, one slot carries both field sets.
    [[nodiscard]] auto replay_nonzero() const -> bool {
        return (ev_ns | grad_ns | evolve_ns | deriv_ns | pare_ns | cos_lazy_ns | cos_mask_ns | n_ev | n_grad | n_pare
                | n_cos_lazy | n_cos_mask)
               != 0;
    }
};

/* One LAYERPROF line for `s`, or none; the count is what a test asserts on. OUTERMOST FIRST, because
 * the timers nest and the line is flat, so a sum double-counts:
 *   total > gate > layer > {fold, scan, emit, index, resolve, insert, sendbuf,
 *                           exchange > {exchangeq, exchangeresp}, incoming,
 *                           encode > {encodelayout, encodepack}}
 *   total > gate > {apply, append}   (outside build_layer)
 *   total > contract > {cacheop, cachestate, index, cacheshrink}
 * `index` accumulates from BOTH Scan.h and the caches step, so it has two parents. Out-of-tree parsers
 * key on `fold_s=` and friends, so those names never change. */
inline auto emit_layer_line(std::FILE *out, bool want_layer, size_t index, const Slot &s) -> int {
    if (!want_layer || !s.any_nonzero()) {
        return 0;
    }
    // `row` is print order over the rows that printed, not the partition index, and is not stable run to run.
    write_line(out,
               "LAYERPROF row={} gates={} total_s={:.4f} gate_s={:.4f} layer_s={:.4f} "
               "fold_s={:.4f} scan_s={:.4f} emit_s={:.4f} index_s={:.4f} "
               "resolve_s={:.4f} insert_s={:.4f} sendbuf_s={:.4f} "
               "exchange_s={:.4f} exchangeq_s={:.4f} exchangeresp_s={:.4f} incoming_s={:.4f} "
               "encode_s={:.4f} encodelayout_s={:.4f} encodepack_s={:.4f} "
               "apply_s={:.4f} append_s={:.4f} "
               "contract_s={:.4f} cacheop_s={:.4f} cachestate_s={:.4f} cacheshrink_s={:.4f} "
               "anti={} foll={} emit={} "
               "atol={} reject={} push={} hit={} miss={} qbytes={} "
               "words={} livewords={} rehash_s={:.4f} rehash={} rehashrows={}\n",
               index,
               s.n_gates,
               to_s(s.total_ns),
               to_s(s.gate_ns),
               to_s(s.layer_ns),
               to_s(s.fold_ns),
               to_s(s.scan_ns),
               to_s(s.emit_ns),
               to_s(s.index_ns),
               to_s(s.resolve_ns),
               to_s(s.insert_ns),
               to_s(s.sendbuf_ns),
               to_s(s.exchange_ns + s.exchange_resp_ns),
               to_s(s.exchange_ns),
               to_s(s.exchange_resp_ns),
               to_s(s.incoming_ns),
               to_s(s.encode_ns),
               to_s(s.encode_layout_ns),
               to_s(s.encode_pack_ns),
               to_s(s.apply_ns),
               to_s(s.append_ns),
               to_s(s.contract_ns),
               to_s(s.cache_op_ns),
               to_s(s.cache_state_ns),
               to_s(s.cache_shrink_ns),
               s.n_anti,
               s.n_foll,
               s.n_emit,
               s.n_atol,
               s.n_reject,
               s.n_push,
               s.n_hit,
               s.n_miss,
               s.query_words * 8,
               s.n_words,
               s.n_live_words,
               to_s(s.rehash_ns),
               s.n_rehash,
               s.n_rehash_rows);
    return 1;
}

// One REPLAYPROF line for `s`, or none. Outermost-first; the tree is in Slot.
inline auto emit_replay_line(std::FILE *out, bool want_replay, size_t index, const Slot &s) -> int {
    if (!want_replay || !s.replay_nonzero()) {
        return 0;
    }
    write_line(out,
               "REPLAYPROF row={} ev={} grad={} pare={} "
               "ev_s={:.4f} grad_s={:.4f} evolve_s={:.4f} deriv_s={:.4f} pare_s={:.4f} "
               "coslazy_s={:.4f} coslazy={} cosmask_s={:.4f} cosmask={}\n",
               index,
               s.n_ev,
               s.n_grad,
               s.n_pare,
               to_s(s.ev_ns),
               to_s(s.grad_ns),
               to_s(s.evolve_ns),
               to_s(s.deriv_ns),
               to_s(s.pare_ns),
               to_s(s.cos_lazy_ns),
               s.n_cos_lazy,
               to_s(s.cos_mask_ns),
               s.n_cos_mask);
    return 1;
}

// One LAYERRUSAGE line for this PROCESS, or none; getrusage runs only when the region is on.
inline auto emit_rusage_line(std::FILE *out, bool want_mem) -> int {
    struct rusage ru{};
    if (!want_mem || getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0;
    }
    // ru_maxrss is KILOBYTES on Linux and BYTES on Darwin; normalised to KiB here so one parser reads both.
#if defined(__APPLE__)
    const auto maxrss_kb = static_cast<uint64_t>(ru.ru_maxrss) / 1024U;
#else
    const auto maxrss_kb = static_cast<uint64_t>(ru.ru_maxrss);
#endif
    write_line(out,
               "LAYERRUSAGE minflt={} majflt={} maxrss_kb={} nvcsw={} nivcsw={}\n",
               static_cast<uint64_t>(ru.ru_minflt),
               static_cast<uint64_t>(ru.ru_majflt),
               maxrss_kb,
               static_cast<uint64_t>(ru.ru_nvcsw),
               static_cast<uint64_t>(ru.ru_nivcsw));
    return 1;
}

class Registry {
public:
    static auto add() -> Slot * {
        auto &r = instance();
        const std::lock_guard<std::mutex> lock(r.mu_);
        r.slots_.push_back(std::make_unique<Slot>());
        return r.slots_.back().get();
    }

    // Force the singleton to EXIST without allocating a slot: `mem`'s line is written by its destructor.
    static auto arm() -> void { (void)instance(); }

    // Order-sound: slot() reads config::get() before add(), so Settings dies last; flags cached here.
    ~Registry() {
        if (want_layer_ || want_mem_ || want_replay_) {
            dump();
        }
    }
    Registry(const Registry &) = delete;
    auto operator=(const Registry &) -> Registry & = delete;
    Registry(Registry &&) = delete;
    auto operator=(Registry &&) -> Registry & = delete;

    static auto dump() -> void {
        auto &r = instance();
        const std::lock_guard<std::mutex> lock(r.mu_);
        size_t p = 0;
        for (const auto &s : r.slots_) {
            p += static_cast<size_t>(emit_layer_line(stderr, r.want_layer_, p, *s));
        }

        // A second pass, not a branch in the first: the two line kinds cannot share a loop index.
        size_t rp = 0;
        for (const auto &s : r.slots_) {
            rp += static_cast<size_t>(emit_replay_line(stderr, r.want_replay_, rp, *s));
        }

        // One line per PROCESS, gated on `mem`: capacity handed back is re-faulted when next needed.
        (void)emit_rusage_line(stderr, r.want_mem_);
        std::fflush(stderr);
    }

private:
    Registry()
        : want_layer_(config::get().profile.layer),
          want_mem_(config::get().profile.mem),
          want_replay_(config::get().profile.replay) {}
    static auto instance() -> Registry & {
        static Registry r;
        return r;
    }

    std::mutex mu_;
    std::vector<std::unique_ptr<Slot>> slots_;
    bool want_layer_ = false;
    bool want_mem_ = false;
    bool want_replay_ = false;
};

// So `mem`'s teardown line prints even from a run that constructs no propagator.
inline auto arm_from_config() -> void {
    if (config::get().profile.mem) {
        Registry::arm();
    }
}

namespace slot_detail {

// The calling thread's ONE Slot; both accessors below hand back this same object.
inline auto owned() -> Slot * {
    static thread_local Slot *s = [] {
        const auto &p = config::get().profile;
        return (p.layer || p.replay) ? Registry::add() : nullptr;
    }();
    return s;
}

} // namespace slot_detail

// nullptr unless `layer` is on; the SAME object replay_slot() returns when both regions are armed.
inline auto slot() -> Slot * {
    static thread_local Slot *s = config::get().profile.layer ? slot_detail::owned() : nullptr;
    return s;
}

// The SAME object slot() returns, null unless `replay` is on: the regions must not arm each other.
inline auto replay_slot() -> Slot * {
    static thread_local Slot *s = config::get().profile.replay ? slot_detail::owned() : nullptr;
    return s;
}

// The one timer type, null-tolerant because the region selector stays a runtime choice here.
class ScopedNs {
public:
    [[gnu::always_inline]] explicit ScopedNs(uint64_t *target) noexcept
        : target_(target),
          start_(target != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ScopedNs(const ScopedNs &) = delete;
    auto operator=(const ScopedNs &) -> ScopedNs & = delete;
    ScopedNs(ScopedNs &&) = delete;
    auto operator=(ScopedNs &&) -> ScopedNs & = delete;
    [[gnu::always_inline]] ~ScopedNs() {
        if (target_ != nullptr) {
            const auto end = std::chrono::steady_clock::now();
            *target_ +=
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
        }
    }

private:
    uint64_t *target_;
    std::chrono::steady_clock::time_point start_;
};

#else // monoprop_ENABLE_PROFILE

// The only symbol a non-profiling build still needs: bindings.cpp calls it unconditionally at import.
inline auto arm_from_config() -> void {}

#endif // monoprop_ENABLE_PROFILE

// Where a partitioned collective's wall time goes. Five fields and no more: a field the transport
// cannot account for exactly is a field someone divides by.

// One cache line per partition, written only by its master: no coherence traffic of its own.
struct alignas(64) CommSlot {
    uint64_t barrier_ns = 0; // parked inside PartitionBarrier::sync -- see CommRegistry for the split
    uint64_t mpi_ns = 0;     // inside an MPI_* call itself; only partition 0 ever reaches one
    uint64_t n_barriers = 0; // syncs entered
    uint64_t n_verbs = 0;    // collectives entered, counted on every partition, not just the master
    uint64_t pinned = 0;     // 1 iff this partition's master got a CPU affinity restriction
};

// Passed in rather than read inside CommRegistry, so a test can drive an ENABLED one into its own stream.
struct CommOptions {
    int mpi_rank = 0; // printed as rank=; the transport supplies it
    // Which of the three transports produced the line: `partitions=1` is equally true of a
    // one-partition ShmComm and of the flat path, and their fields do not mean the same thing.
    const char *transport = "?";
    bool enabled = config::get().profile.comm; // the only thing that allocates a slot or reads a clock
    std::FILE *out = stderr;                   // stderr in production, like LAYERPROF (needs `pytest -s`)
};

#ifndef monoprop_ENABLE_PROFILE

// Hollow: no slots, so the transports' `prof != nullptr` guards fold away. A type, because they hold one.
class CommRegistry {
public:
    CommRegistry(int /*n_partitions*/, CommOptions /*opts*/) {}
    [[nodiscard]] auto slot(int /*partition*/) -> CommSlot * { return nullptr; }
    [[nodiscard]] auto emit() noexcept -> int { return 0; }
};

#else

// One slot per partition, one COMMPROF line at teardown -- emitted whenever the region is on, because a
// transport that ran no collectives is a finding. One line per TRANSPORT, and copying a propagator
// clones its transport, so never average by rank.
class CommRegistry {
public:
    CommRegistry(int n_partitions, CommOptions opts)
        : slots_(opts.enabled ? static_cast<size_t>(n_partitions) : 0),
          mpi_rank_(opts.mpi_rank),
          transport_(opts.transport),
          out_(opts.out) {}
    ~CommRegistry() { emit(); }
    CommRegistry(const CommRegistry &) = delete;
    auto operator=(const CommRegistry &) -> CommRegistry & = delete;
    CommRegistry(CommRegistry &&) = delete;
    auto operator=(CommRegistry &&) -> CommRegistry & = delete;

    // nullptr when the region is off; the returned slot is stable (the vector is never resized).
    [[nodiscard]] auto slot(int partition) -> CommSlot * {
        return slots_.empty() ? nullptr : &slots_[static_cast<size_t>(partition)];
    }

    // Returns how many lines it wrote: the only thing separating "both arms measured the same" from
    // "the instrument never fired". One-shot, so the contract holds even if a caller dumps early.
    auto emit() noexcept -> int {
        if (emitted_ || slots_.empty()) {
            return 0;
        }
        emitted_ = true;
        CommSlot total;
        for (const CommSlot &s : slots_) {
            total.barrier_ns += s.barrier_ns;
            total.mpi_ns += s.mpi_ns;
            total.n_barriers += s.n_barriers;
            total.pinned += s.pinned;
        }
        const CommSlot &p0 = slots_.front();
        // Reported per peer, not summed: summed it grows with S and reads as a cost scaling caused.
        const uint64_t peer_barrier_ns = total.barrier_ns - p0.barrier_ns;
        // mpi_s sums over partitions and equals p0's, because only p0 calls MPI: a measurement, not a bet.
        const auto peers = static_cast<double>(slots_.size() > 1 ? slots_.size() - 1 : 1);
        write_line(out_,
                   "COMMPROF rank={} transport={} partitions={} pinned={} verbs={} barriers={} "
                   "mpi_s={:.4f} barrier_p0_s={:.4f} barrier_peers_s={:.4f} barrier_per_sync_us={:.2f}\n",
                   mpi_rank_,
                   transport_,
                   slots_.size(),
                   total.pinned,
                   p0.n_verbs,
                   p0.n_barriers,
                   to_s(total.mpi_ns),
                   to_s(p0.barrier_ns),
                   to_s(peer_barrier_ns) / peers,
                   total.n_barriers == 0
                       ? 0.0
                       : (static_cast<double>(total.barrier_ns) / static_cast<double>(total.n_barriers)) / 1000.0);
        std::fflush(out_);
        return 1;
    }

private:
    // Empty iff the region is off: it is the single allocation the whole instrument makes.
    std::vector<CommSlot> slots_;
    int mpi_rank_;
    const char *transport_; // a string literal owned by the caller; never freed, never copied
    std::FILE *out_;
    bool emitted_ = false;
};

#endif // monoprop_ENABLE_PROFILE

} // namespace monoprop::detail::profile

/* The seam: the only way a call site reaches the instrument, so a non-profiling build names no profile
 * type and the namespace above disappears. monoprop_PROF_SCOPE(prof, fold) takes the region NAME and
 * derives `fold_ns`, so a different backend is a change here and not in every hot header; CommSlot's
 * barrier and mpi are spelled the same way. monoprop_PROF(...) absorbs the accumulators and the fold
 * that adds them in, so nothing needs [[maybe_unused]]. Read the slot ONCE per gate, never inside a
 * scope macro: a per-block thread_local read would land inside the timer it is arming. */
#ifdef monoprop_ENABLE_PROFILE
#define monoprop_PROF(...)       __VA_ARGS__
#define monoprop_PROF_SLOT(name) ::monoprop::detail::profile::Slot *const name = ::monoprop::detail::profile::slot()
#define monoprop_PROF_REPLAY_SLOT(name) \
    ::monoprop::detail::profile::Slot *const name = ::monoprop::detail::profile::replay_slot()
#define monoprop_PROF_SCOPE(slot, region)                                                          \
    const ::monoprop::detail::profile::ScopedNs monoprop_prof_##region((slot) == nullptr ? nullptr \
                                                                                         : &(slot)->region##_ns)
// For the handful of scopes whose target comes from a helper rather than a slot member.
#define monoprop_PROF_SCOPE_AT(tag, target) const ::monoprop::detail::profile::ScopedNs monoprop_prof_##tag(target)
#else
#define monoprop_PROF(...)
#define monoprop_PROF_SLOT(name)
#define monoprop_PROF_REPLAY_SLOT(name)
#define monoprop_PROF_SCOPE(slot, region)
#define monoprop_PROF_SCOPE_AT(tag, target)
#endif
