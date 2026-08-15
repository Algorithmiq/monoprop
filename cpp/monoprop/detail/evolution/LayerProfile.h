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

// Per-partition attribution of the layer build, opt-in via monoprop_LAYER_PROFILE.
//
// Two instruments with deliberately different costs:
//   * phase timers and per-gate counters -- O(1) clock reads per gate, and per-term counters that
//     live in locals and are folded into the slot once per gate, so the emit loop keeps its registers;
//   * a store population sweep (k/d histograms, paired and overflow fractions) -- O(n), so it runs
//     only on power-of-two gate indices, which also yields the population's growth curve for free.
//
// Slots are heap-allocated per thread and owned by the registry, never freed: a partition master can
// outlive or predecease the dump, and a dangling slot would be a use-after-free in a diagnostic.
// Output goes to stderr, like COMMPROF -- pytest's capture is fd-level, so reading it needs `-s`.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::detail::layer_profile {

inline constexpr size_t kHistSlots = 34; // k, d clamped into [0, 33]; slot 33 is the overflow bucket

struct alignas(64) Slot {
    // Phase wall time. `index_ns` covers inverted-index rebuild/append, which is not a layer phase but
    // is charged per gate and has never been attributed.
    uint64_t fold_ns = 0;
    uint64_t emit_ns = 0;
    uint64_t index_ns = 0;
    uint64_t resolve_ns = 0;
    uint64_t insert_ns = 0;
    uint64_t contract_ns = 0;
    // Cross-rank legs. `layer_ns` brackets the whole of build_layer, so layer_ns minus the phases is
    // the unattributed remainder -- without it a phase split says nothing about the wall time.
    uint64_t sendbuf_ns = 0;  // packing queries (+values) into the contiguous send stream
    uint64_t exchange_ns = 0; // inside MPI: both alltoallv legs
    uint64_t incoming_ns = 0; // resolving other ranks' queries against this store
    uint64_t layer_ns = 0;

    uint64_t n_gates = 0;
    uint64_t n_anti = 0;   // terms the fold marked anticommuting
    uint64_t n_foll = 0;   // of those, followers
    uint64_t n_emit = 0;   // reached emit_term_products (i.e. passed the dynamic gate)
    uint64_t n_cutoff = 0; // fell through passes_with_popcount's early-out into cutoff_sums
    uint64_t n_reject = 0; // dropped by the structural cutoff
    uint64_t n_push = 0;   // query records pushed
    uint64_t n_hit = 0;    // self-resolve hits
    uint64_t n_miss = 0;   // self-resolve misses (deferred inserts)
    uint64_t query_words = 0;
    // query_words is byte accounting, summed from what push() actually WROTE rather than from
    // record_count * an assumed stride. The compact record is variable-width, so there is no stride to
    // assume -- and an assumed one is how this counter previously reported the old fixed record's width
    // on a buffer that no longer held it, which reads exactly like a change that did not happen. It is
    // charged at emit, so it still counts records that drop_matched_cross_rank_followers removes and the
    // self slot that never travels: read it as an upper bound on wire bytes, not as a byte count.

    // Population sample, overwritten each time the sweep runs (last sample wins; the per-sample line
    // is emitted at sweep time, so the growth curve is in the log even though only the last is here).
    uint64_t pop_rows = 0;
    uint64_t pop_paired = 0;
    uint64_t pop_overflow = 0;
    uint64_t k_hist[kHistSlots] = {};
    uint64_t d_hist[kHistSlots] = {};
};

class Registry {
public:
    static auto add() -> Slot * {
        auto &r = instance();
        const std::lock_guard<std::mutex> lock(r.mu_);
        r.slots_.push_back(std::make_unique<Slot>());
        return r.slots_.back().get();
    }

    // Dumping from the destructor rather than a separate atexit object keeps the ordering sound:
    // slot() evaluates config::get() before Registry::add(), so Settings is constructed first and
    // therefore destroyed last, and nothing this reads can already be gone.
    ~Registry() {
        if (config::get().layer_profile) {
            dump();
        }
    }
    Registry(const Registry &) = delete;
    Registry &operator=(const Registry &) = delete;
    Registry(Registry &&) = delete;
    Registry &operator=(Registry &&) = delete;

    static auto dump() -> void {
        auto &r = instance();
        const std::lock_guard<std::mutex> lock(r.mu_);
        size_t p = 0;
        for (const auto &s : r.slots_) {
            if (s->n_gates == 0) {
                continue;
            }
            std::fprintf(stderr,
                         "LAYERPROF part=%zu gates=%llu fold_s=%.4f emit_s=%.4f index_s=%.4f "
                         "resolve_s=%.4f insert_s=%.4f contract_s=%.4f sendbuf_s=%.4f exchange_s=%.4f "
                         "incoming_s=%.4f layer_s=%.4f anti=%llu foll=%llu emit=%llu "
                         "cutoff=%llu reject=%llu push=%llu hit=%llu miss=%llu qbytes=%llu\n",
                         p,
                         ull(s->n_gates),
                         to_s(s->fold_ns),
                         to_s(s->emit_ns),
                         to_s(s->index_ns),
                         to_s(s->resolve_ns),
                         to_s(s->insert_ns),
                         to_s(s->contract_ns),
                         to_s(s->sendbuf_ns),
                         to_s(s->exchange_ns),
                         to_s(s->incoming_ns),
                         to_s(s->layer_ns),
                         ull(s->n_anti),
                         ull(s->n_foll),
                         ull(s->n_emit),
                         ull(s->n_cutoff),
                         ull(s->n_reject),
                         ull(s->n_push),
                         ull(s->n_hit),
                         ull(s->n_miss),
                         ull(s->query_words * 8));
            ++p;
        }
        std::fflush(stderr);
    }

private:
    Registry() = default;
    static auto instance() -> Registry & {
        static Registry r;
        return r;
    }
    static auto to_s(uint64_t ns) -> double { return static_cast<double>(ns) / 1e9; }
    static auto ull(uint64_t v) -> unsigned long long { return static_cast<unsigned long long>(v); }

    std::mutex mu_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

// nullptr when profiling is off, so every call site is one predictable branch and no clock is read.
inline auto slot() -> Slot * {
    static thread_local Slot *s = config::get().layer_profile ? Registry::add() : nullptr;
    return s;
}

// RAII accumulator; inert when constructed with a null target, so instrumented regions need no #ifdef.
class ScopedNs {
public:
    explicit ScopedNs(uint64_t *target) noexcept
        : target_(target),
          start_(target != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ScopedNs(const ScopedNs &) = delete;
    ScopedNs &operator=(const ScopedNs &) = delete;
    ScopedNs(ScopedNs &&) = delete;
    ScopedNs &operator=(ScopedNs &&) = delete;
    ~ScopedNs() {
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

// O(n) over the store: k and d histograms, the fully-paired fraction and the overflow fraction --
// the numbers that decide whether the paired-row and digest work is worth landing. Runs only on
// power-of-two gate indices (see the file comment), so the log carries the growth curve.
template <size_t NumModes, typename Store>
auto sample_population(Slot *s, const Store &store, size_t gate_index) -> void {
    if (s == nullptr) {
        return;
    }
    const uint64_t g = s->n_gates;
    if (g != 0 && (g & (g - 1)) != 0) {
        return;
    }
    for (auto &v : s->k_hist) {
        v = 0;
    }
    for (auto &v : s->d_hist) {
        v = 0;
    }
    s->pop_rows = 0;
    s->pop_paired = 0;
    s->pop_overflow = 0;

    const size_t n = store.size();
    for (size_t i = 0; i < n; ++i) {
        // Mode m owns physical bits (2m, 2m+1); d counts modes holding both, so on an ascending
        // position list a pair is an even entry immediately followed by its successor.
        size_t k = 0;
        size_t d = 0;
        size_t prev = static_cast<size_t>(-2);
        store.for_each_position(i, [&](size_t pos) {
            ++k;
            if (pos == prev + 1 && (prev % 2) == 0) {
                ++d;
            }
            prev = pos;
        });
        ++s->pop_rows;
        s->k_hist[k < kHistSlots ? k : kHistSlots - 1] += 1;
        s->d_hist[d < kHistSlots ? d : kHistSlots - 1] += 1;
        if (k == 2 * d) {
            ++s->pop_paired;
        }
    }
    s->pop_overflow = store.overflow_size();

    // One buffered write: partition masters sample concurrently, and a line assembled by several
    // fprintf calls interleaves across threads into something no parser can recover.
    char line[4096];
    int off = std::snprintf(line,
                            sizeof(line),
                            "LAYERPOP gate=%zu rows=%llu paired=%llu overflow=%llu width=%zu k_hist=",
                            gate_index,
                            static_cast<unsigned long long>(s->pop_rows),
                            static_cast<unsigned long long>(s->pop_paired),
                            static_cast<unsigned long long>(s->pop_overflow),
                            store.inline_width());
    const auto append_hist = [&](const uint64_t (&h)[kHistSlots]) {
        for (size_t j = 0; j < kHistSlots && off > 0 && static_cast<size_t>(off) < sizeof(line); ++j) {
            off += std::snprintf(line + off,
                                 sizeof(line) - static_cast<size_t>(off),
                                 "%llu%s",
                                 static_cast<unsigned long long>(h[j]),
                                 j + 1 < kHistSlots ? "," : "");
        }
    };
    append_hist(s->k_hist);
    if (off > 0 && static_cast<size_t>(off) < sizeof(line)) {
        off += std::snprintf(line + off, sizeof(line) - static_cast<size_t>(off), " d_hist=");
    }
    append_hist(s->d_hist);
    std::fprintf(stderr, "%s\n", line);
}

} // namespace monoprop::detail::layer_profile
