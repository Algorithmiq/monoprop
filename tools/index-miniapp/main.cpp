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

/*
 * Task 0 index mini-app: one MPI-off process, one OpenMP team, three ownership/index designs over the
 * current packed rows (see README.md). Exact key/ID checks establish index-result agreement only, never
 * retained-operator equivalence or propagation performance.
 */

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <format>
#include <map>
#include <memory>
#include <numeric>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/version.hpp>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"

#ifdef monoprop_ENABLE_MPI
#error "index-miniapp is MPI-off only"
#endif
static_assert(BOOST_VERSION >= 108500, "the project's Boost floor is 1.85");

namespace {

using monoprop::Monomial;
using monoprop::TermIndex;
using monoprop::detail::OperatorIndex;
using Clock = std::chrono::steady_clock;

constexpr size_t kNotFound = OperatorIndex<128>::kNotFound;

// Release-active check: correctness gates must not vanish with NDEBUG.
auto require(bool ok, std::string_view what) -> void {
    if (!ok) {
        throw std::runtime_error(std::string(what));
    }
}

auto lap(Clock::time_point &t) -> double {
    const auto now = Clock::now();
    const double s = std::chrono::duration<double>(now - t).count();
    t = now;
    return s;
}

auto splitmix(uint64_t &state) -> uint64_t {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Self-test-only fault injection: the worker with this thread number throws inside its region.
std::atomic<int> g_fault_tid{-1};
auto maybe_inject(int tid) -> void {
    if (tid == g_fault_tid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("injected worker fault");
    }
}

// Balanced contiguous span [lo, hi) of n items for worker tid; short spans and empty tails are allowed.
auto span_of(size_t n, int team, int tid) -> std::pair<size_t, size_t> {
    const auto t = static_cast<size_t>(team);
    return {n * static_cast<size_t>(tid) / t, n * static_cast<size_t>(tid + 1) / t};
}

auto rethrow_first(const std::vector<std::exception_ptr> &errors) -> void {
    for (const auto &e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

// Per-owner sub-phases of the sharded arm overlap in time: their sums are nonadditive diagnostics.
enum Phase : uint8_t { kRoute, kPrefix, kScatter, kProbe, kAssign, kFill, kPublish, kGather, kOwnerElapsed, kOwnerMax };
constexpr size_t kNumPhases = 10;
constexpr std::array<std::string_view, kNumPhases> kPhaseNames{"route",
                                                               "prefix",
                                                               "scatter",
                                                               "probe",
                                                               "assign",
                                                               "fill",
                                                               "publish",
                                                               "gather",
                                                               "owner_elapsed",
                                                               "owner_max"};

struct BatchObs {
    int team = 0;           //!< Smallest actual team among this batch's regions.
    int probe_active = 0;   //!< Workers that probed at least one query.
    int publish_active = 0; //!< Workers that published at least one row; 0 when nothing was published.
    size_t bulk_calls = 0;  //!< Bulk cvisit calls issued by the probe (shared-concurrent bulk lookup only).
    std::array<double, kNumPhases> phase{};
};

// Flat query batch: query q is pos[off[q], off[q]+k[q]), strictly ascending positions.
template <size_t N>
struct Batch {
    using PosT = typename OperatorIndex<N>::PosT;
    std::vector<PosT> pos;
    std::vector<size_t> off;
    std::vector<uint32_t> k;

    [[nodiscard]] auto size() const -> size_t { return off.size(); }
    [[nodiscard]] auto key(size_t q) const -> std::span<const PosT> { return {pos.data() + off[q], k[q]}; }
    auto clear() -> void {
        pos.clear();
        off.clear();
        k.clear();
    }
    auto push(std::span<const PosT> p) -> void {
        off.push_back(pos.size());
        k.push_back(static_cast<uint32_t>(p.size()));
        pos.insert(pos.end(), p.begin(), p.end());
    }
};

// Results in original query order; owner is 0 for the shared variants.
struct Results {
    std::vector<size_t> id;
    std::vector<uint32_t> owner;
    std::vector<uint8_t> hit;
    auto resize(size_t n) -> void {
        id.assign(n, kNotFound);
        owner.assign(n, 0);
        hit.assign(n, 0);
    }
};

template <size_t N>
auto monomial_of(std::span<const typename OperatorIndex<N>::PosT> pos) -> Monomial<N> {
    Monomial<N> m;
    for (const auto p : pos) {
        m.set(p);
    }
    return m;
}

// Exact identity through public row access; a spilled row has no position array and compares densely.
template <size_t N>
auto row_matches(const OperatorIndex<N> &s, size_t row, std::span<const typename OperatorIndex<N>::PosT> q) -> bool {
    const auto rp = s.row_positions(row);
    return rp.inlined() ? std::ranges::equal(rp.pos, q) : s.row(row) == monomial_of<N>(q);
}

template <size_t N>
auto rows_equal(const OperatorIndex<N> &s, size_t a, size_t b) -> bool {
    const auto ra = s.row_positions(a);
    const auto rb = s.row_positions(b);
    return (ra.inlined() && rb.inlined()) ? std::ranges::equal(ra.pos, rb.pos) : s.row(a) == s.row(b);
}

// bulk_insert_hashed never deduplicates, so a publication list holding one key twice must be refused before
// any row grows. Self-tests enable this; measured inputs are distinct by construction and validated afterwards.
template <size_t N>
auto require_distinct(const Batch<N> &b, std::vector<std::pair<uint32_t, size_t>> hq) -> void {
    std::ranges::sort(hq, [&](const auto &x, const auto &y) {
        return x.first != y.first ? x.first < y.first
                                  : std::ranges::lexicographical_compare(b.key(x.second), b.key(y.second));
    });
    for (size_t i = 1; i < hq.size(); ++i) {
        require(hq[i].first != hq[i - 1].first || !std::ranges::equal(b.key(hq[i].second), b.key(hq[i - 1].second)),
                "invalid publication fixture: duplicate missing key");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Variant 1: T disjoint current stores. Owner = monomial_hash % shards (R=1 routing); each owner is the single
// writer of its store. Bucketing is a simplified shared-memory model, not the legacy transport.
template <size_t N>
class PerThread {
public:
    using Store = OperatorIndex<N>;
    using PosT = typename Store::PosT;
    static constexpr bool kIdsAreOrdinals = false;
    static constexpr std::array<bool, kNumPhases> kPhases{true, true, true, true, true, true, true, true, true, true};

    PerThread(int shards, int threads, size_t width) : threads_(threads) {
        for (int s = 0; s < shards; ++s) {
            stores_.push_back(std::make_unique<Store>(width));
        }
    }
    bool check_distinct = false; //!< Refuse duplicate misses before publication (self-test fixtures).

    [[nodiscard]] auto owners() const -> size_t { return stores_.size(); }
    [[nodiscard]] auto owner_size(size_t o) const -> size_t { return stores_[o]->size(); }
    [[nodiscard]] auto owner_of(std::span<const PosT> key) const -> uint32_t {
        return static_cast<uint32_t>(monoprop::monomial_hash<N>(monomial_of<N>(key)) % owners());
    }
    [[nodiscard]] auto matches(size_t o, size_t id, std::span<const PosT> key) const -> bool {
        return id < stores_[o]->size() && row_matches(*stores_[o], id, key);
    }
    [[nodiscard]] auto row_bytes() const -> size_t {
        return sum_([](const Store &s) { return s.memory_bytes(); });
    }
    [[nodiscard]] auto index_bytes() const -> size_t {
        return sum_([](const Store &s) { return s.index_estimated_memory_bytes(); });
    }
    [[nodiscard]] static auto harness_bytes() -> size_t { return 0; }

    // route/count -> ordered prefixes -> stable scatter into owner buckets -> per-owner probe, ordered ID
    // assignment, row fill and publication into bucket slots -> each source chunk pulls its own results. Owners
    // never write query-order arrays, so no two owners share a result cache line. One team; phases are separated
    // by barriers, and a failed phase only skips later work so every worker still reaches every barrier and join.
    auto run(const Batch<N> &b, bool grow, Results &r, BatchObs &obs, bool profile) -> void {
        const size_t n = b.size();
        const size_t shards = owners();
        const auto threads = static_cast<size_t>(threads_);
        r.resize(n);
        owner_q_.resize(n);
        perm_.resize(n);
        boff_.resize(n);
        bk_.resize(n);
        lout_.resize(n);
        lhash_.resize(n);
        miss_.resize(n);
        slot_.resize(n);
        bhit_.resize(n);
        count_.assign(threads * shards, 0);
        cursor_.assign(threads * shards, 0);
        start_.assign(shards + 1, 0);
        probed_.assign(threads, 0);
        published_.assign(threads, 0);
        owner_t_.assign(shards, {});
        std::vector<std::exception_ptr> errors(threads);
        std::atomic<bool> failed{false};
        auto guarded = [&](int tid, auto &&fn) {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            try {
                fn();
            }
            catch (...) {
                errors[static_cast<size_t>(tid)] = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        };
        int team_seen = 0;
        auto t = Clock::now();
#pragma omp parallel num_threads(threads_)
        {
            const int tid = omp_get_thread_num();
            const int team = omp_get_num_threads();
            const auto [lo, hi] = span_of(n, team, tid);
            if (tid == 0) {
                team_seen = team;
            }
            guarded(tid, [&] {
                size_t *cnt = &count_[static_cast<size_t>(tid) * shards];
                for (size_t q = lo; q < hi; ++q) {
                    owner_q_[q] = static_cast<uint32_t>(monoprop::monomial_hash<N>(monomial_of<N>(b.key(q))) % shards);
                    ++cnt[owner_q_[q]];
                }
            });
#pragma omp barrier
            if (profile && tid == 0) {
                obs.phase[kRoute] += lap(t);
            }
            guarded(tid, [&] {
                for (auto o = static_cast<size_t>(tid); o < shards; o += static_cast<size_t>(team)) {
                    size_t total = 0;
                    for (size_t c = 0; c < static_cast<size_t>(team); ++c) {
                        cursor_[(c * shards) + o] = total;
                        total += count_[(c * shards) + o];
                    }
                    start_[o + 1] = total;
                }
            });
#pragma omp barrier
#pragma omp single
            guarded(tid, [&] { std::partial_sum(start_.begin(), start_.end(), start_.begin()); });
            if (profile && tid == 0) {
                obs.phase[kPrefix] += lap(t);
            }
            guarded(tid, [&] {
                size_t *cur = &cursor_[static_cast<size_t>(tid) * shards];
                for (size_t q = lo; q < hi; ++q) {
                    const size_t p = start_[owner_q_[q]] + cur[owner_q_[q]]++;
                    perm_[p] = q;
                    slot_[q] = p;
                    boff_[p] = b.off[q];
                    bk_[p] = b.k[q];
                }
            });
#pragma omp barrier
            if (profile && tid == 0) {
                obs.phase[kScatter] += lap(t);
            }
            guarded(tid, [&] {
                maybe_inject(tid);
                for (auto o = static_cast<size_t>(tid); o < shards; o += static_cast<size_t>(team)) {
                    owner_work_(o, tid, b, grow, profile);
                }
            });
#pragma omp barrier
            if (profile && tid == 0) {
                obs.phase[kOwnerElapsed] += lap(t);
            }
            guarded(tid, [&] {
                for (size_t q = lo; q < hi; ++q) {
                    r.id[q] = lout_[slot_[q]];
                    r.hit[q] = bhit_[slot_[q]];
                    r.owner[q] = owner_q_[q];
                }
            });
        }
        rethrow_first(errors);
        obs.team = team_seen;
        obs.probe_active = static_cast<int>(std::ranges::count(probed_, 1));
        obs.publish_active = static_cast<int>(std::ranges::count(published_, 1));
        if (profile) {
            obs.phase[kGather] += lap(t);
            double owner_max = 0;
            for (const auto &ot : owner_t_) {
                double sum = 0;
                for (size_t p = kProbe; p <= kPublish; ++p) {
                    obs.phase[p] += ot[p];
                    sum += ot[p];
                }
                owner_max = std::max(owner_max, sum);
            }
            obs.phase[kOwnerMax] += owner_max;
        }
    }

private:
    auto owner_work_(size_t o, int tid, const Batch<N> &b, bool grow, bool profile) -> void {
        const size_t lo = start_[o];
        const size_t count = start_[o + 1] - lo;
        if (count == 0) {
            return;
        }
        probed_[static_cast<size_t>(tid)] = 1;
        Store &st = *stores_[o];
        auto t = Clock::now();
        auto tick = [&](Phase p) {
            if (profile) {
                owner_t_[o][p] += lap(t);
            }
        };
        st.find_batch_positions(b.pos,
                                std::span<const size_t>(boff_).subspan(lo, count),
                                std::span<const uint32_t>(bk_).subspan(lo, count),
                                std::span(lout_).subspan(lo, count),
                                std::span(lhash_).subspan(lo, count));
        tick(kProbe);
        size_t m = 0;
        for (size_t j = lo; j < lo + count; ++j) {
            bhit_[j] = lout_[j] != kNotFound ? 1 : 0;
            if (grow && lout_[j] == kNotFound) {
                miss_[lo + m++] = j;
            }
        }
        if (m > 0) {
            if (check_distinct) {
                std::vector<std::pair<uint32_t, size_t>> hq;
                for (size_t i = 0; i < m; ++i) {
                    hq.emplace_back(lhash_[miss_[lo + i]], perm_[miss_[lo + i]]);
                }
                require_distinct(b, std::move(hq));
            }
            tick(kAssign);
            const size_t base = st.grow_rows_geometric(m);
            for (size_t i = 0; i < m; ++i) {
                st.set_positions(base + i, b.key(perm_[miss_[lo + i]]));
            }
            tick(kFill);
            st.bulk_insert_hashed(m, base, [&](size_t i) { return lhash_[miss_[lo + i]]; });
            for (size_t i = 0; i < m; ++i) {
                lout_[miss_[lo + i]] = base + i;
            }
            published_[static_cast<size_t>(tid)] = 1;
            tick(kPublish);
        }
    }

    template <class Fn>
    [[nodiscard]] auto sum_(Fn fn) const -> size_t {
        size_t total = 0;
        for (const auto &s : stores_) {
            total += fn(*s);
        }
        return total;
    }

    int threads_;
    std::vector<std::unique_ptr<Store>> stores_;
    // Batch scratch: O(batch) records plus O(threads*shards) counters, reused across batches.
    std::vector<uint32_t> owner_q_, bk_, lhash_;
    std::vector<size_t> perm_, slot_, boff_, lout_, miss_, count_, cursor_, start_;
    std::vector<uint8_t> bhit_, probed_, published_;
    std::vector<std::array<double, kNumPhases>> owner_t_;
};

// ---------------------------------------------------------------------------------------------------------------
// Variant 2: one current store. Parallel frozen probes into disjoint output spans; the caller assigns IDs in
// query order, grows/initializes rows and publishes with bulk_insert_hashed.
template <size_t N>
class SharedSerial {
public:
    using Store = OperatorIndex<N>;
    using PosT = typename Store::PosT;
    static constexpr bool kIdsAreOrdinals = true;
    static constexpr std::array<bool, kNumPhases>
        kPhases{false, false, false, true, true, true, true, false, false, false};

    SharedSerial(int threads, size_t width) : threads_(threads), store_(std::make_unique<Store>(width)) {}
    bool check_distinct = false; //!< Refuse duplicate misses before publication (self-test fixtures).

    [[nodiscard]] static auto owners() -> size_t { return 1; }
    [[nodiscard]] auto owner_size(size_t /*o*/) const -> size_t { return store_->size(); }
    [[nodiscard]] static auto owner_of(std::span<const PosT> /*key*/) -> uint32_t { return 0; }
    [[nodiscard]] auto matches(size_t /*o*/, size_t id, std::span<const PosT> key) const -> bool {
        return id < store_->size() && row_matches(*store_, id, key);
    }
    [[nodiscard]] auto row_bytes() const -> size_t { return store_->memory_bytes(); }
    [[nodiscard]] auto index_bytes() const -> size_t { return store_->index_estimated_memory_bytes(); }
    [[nodiscard]] static auto harness_bytes() -> size_t { return 0; }

    auto run(const Batch<N> &b, bool grow, Results &r, BatchObs &obs, bool profile) -> void {
        const size_t n = b.size();
        r.resize(n);
        hash_.resize(n);
        probed_.assign(static_cast<size_t>(threads_), 0);
        std::vector<std::exception_ptr> errors(static_cast<size_t>(threads_));
        int team_seen = 0;
        auto t = Clock::now();
#pragma omp parallel num_threads(threads_)
        {
            const int tid = omp_get_thread_num();
            const int team = omp_get_num_threads();
            if (tid == 0) {
                team_seen = team;
            }
            try {
                maybe_inject(tid);
                const auto [lo, hi] = span_of(n, team, tid);
                if (lo < hi) {
                    probed_[static_cast<size_t>(tid)] = 1;
                    store_->find_batch_positions(b.pos,
                                                 std::span<const size_t>(b.off).subspan(lo, hi - lo),
                                                 std::span<const uint32_t>(b.k).subspan(lo, hi - lo),
                                                 std::span(r.id).subspan(lo, hi - lo),
                                                 std::span(hash_).subspan(lo, hi - lo));
                }
            }
            catch (...) {
                errors[static_cast<size_t>(tid)] = std::current_exception();
            }
        }
        rethrow_first(errors);
        obs.team = team_seen;
        obs.probe_active = static_cast<int>(std::ranges::count(probed_, 1));
        if (profile) {
            obs.phase[kProbe] += lap(t);
        }
        miss_.clear();
        for (size_t q = 0; q < n; ++q) {
            r.hit[q] = r.id[q] != kNotFound ? 1 : 0;
            if (grow && r.id[q] == kNotFound) {
                miss_.push_back(q);
            }
        }
        if (miss_.empty()) {
            return;
        }
        if (check_distinct) {
            std::vector<std::pair<uint32_t, size_t>> hq;
            for (const auto q : miss_) {
                hq.emplace_back(hash_[q], q);
            }
            require_distinct(b, std::move(hq));
        }
        if (profile) {
            obs.phase[kAssign] += lap(t);
        }
        const size_t base = store_->grow_rows_geometric(miss_.size());
        for (size_t i = 0; i < miss_.size(); ++i) {
            store_->set_positions(base + i, b.key(miss_[i]));
        }
        if (profile) {
            obs.phase[kFill] += lap(t);
        }
        store_->bulk_insert_hashed(miss_.size(), base, [&](size_t i) { return hash_[miss_[i]]; });
        for (size_t i = 0; i < miss_.size(); ++i) {
            r.id[miss_[i]] = base + i;
        }
        obs.publish_active = 1;
        if (profile) {
            obs.phase[kPublish] += lap(t);
        }
    }

private:
    int threads_;
    std::unique_ptr<Store> store_;
    std::vector<uint32_t> hash_;
    std::vector<size_t> miss_;
    std::vector<uint8_t> probed_;
};

// ---------------------------------------------------------------------------------------------------------------
// Variant 3: one packed row backing plus boost::concurrent_flat_map keyed by compact row handles. The backing
// OperatorIndex keeps its constant empty 16-slot table (harness overhead); it is never populated or reserved.
template <size_t N>
class SharedConcurrent {
public:
    using Store = OperatorIndex<N>;
    using PosT = typename Store::PosT;
    static constexpr bool kIdsAreOrdinals = true;
    static constexpr std::array<bool, kNumPhases>
        kPhases{false, false, false, true, true, true, true, false, false, false};

    SharedConcurrent(int threads, size_t width)
        : threads_(threads),
          rows_(std::make_unique<Store>(width)),
          map_(0, KeyHash{}, KeyEqual{rows_.get()}) {}
    SharedConcurrent(const SharedConcurrent &) = delete;
    auto operator=(const SharedConcurrent &) -> SharedConcurrent & = delete;
    bool check_distinct = false; //!< Refuse duplicate misses before publication (self-test fixtures).
    bool bulk_lookup = false;    //!< Probe through Boost's bulk cvisit in 16-query chunks.

    [[nodiscard]] static auto owners() -> size_t { return 1; }
    [[nodiscard]] auto owner_size(size_t /*o*/) const -> size_t { return rows_->size(); }
    [[nodiscard]] static auto owner_of(std::span<const PosT> /*key*/) -> uint32_t { return 0; }
    [[nodiscard]] auto matches(size_t /*o*/, size_t id, std::span<const PosT> key) const -> bool {
        return id < rows_->size() && row_matches(*rows_, id, key);
    }
    [[nodiscard]] auto index_entries() const -> size_t { return map_.size(); }
    [[nodiscard]] auto row_bytes() const -> size_t { return rows_->memory_bytes(); }
    // Estimate: FOA slot array plus 16 metadata bytes per 15-slot group and its concurrent group-access word.
    [[nodiscard]] auto index_bytes() const -> size_t {
        const size_t buckets = map_.bucket_count();
        return sizeof(Map) + (buckets * sizeof(typename Map::value_type)) + ((buckets / 15 + 1) * (16 + 8));
    }
    [[nodiscard]] auto harness_bytes() const -> size_t { return rows_->index_estimated_memory_bytes(); }
#ifdef BOOST_UNORDERED_ENABLE_STATS
    //! Boost's container statistics; diagnostic builds only (the concurrent counters serialize on one lock).
    [[nodiscard]] auto boost_stats() const { return map_.get_stats(); }
    //! Clears Boost's container statistics.
    auto reset_boost_stats() -> void { map_.reset_stats(); }
#endif

    // Parallel hash + per-query visit (frozen map) -> caller-ordered IDs and row fill -> join -> parallel
    // publication of preassigned IDs. Rows are immutable while the map runs; joins precede any row growth.
    auto run(const Batch<N> &b, bool grow, Results &r, BatchObs &obs, bool profile) -> void {
        const size_t n = b.size();
        r.resize(n);
        hash_.resize(n);
        probed_.assign(static_cast<size_t>(threads_), 0);
        published_.assign(static_cast<size_t>(threads_), 0);
        std::vector<std::exception_ptr> errors(static_cast<size_t>(threads_));
        std::vector<size_t> bulk_calls(static_cast<size_t>(threads_), 0);
        int team_seen = threads_;
        auto t = Clock::now();
#pragma omp parallel num_threads(threads_)
        {
            const int tid = omp_get_thread_num();
            const int team = omp_get_num_threads();
            if (tid == 0) {
                team_seen = team;
            }
            try {
                maybe_inject(tid);
                const auto [lo, hi] = span_of(n, team, tid);
                if (bulk_lookup) {
                    for (size_t c = lo; c < hi; c += kBulkChunk) {
                        probe_chunk_(b, r, c, std::min(hi, c + kBulkChunk));
                        ++bulk_calls[static_cast<size_t>(tid)];
                    }
                }
                else {
                    for (size_t q = lo; q < hi; ++q) {
                        hash_[q] = Store::fold_hash_positions(b.key(q));
                        size_t found = kNotFound;
                        // One bounded visitor per query keeps results aligned; it never re-enters the map.
                        map_.cvisit(QueryKey{b.key(q), hash_[q]}, [&found](const auto &kv) { found = kv.second; });
                        r.id[q] = found;
                    }
                }
                probed_[static_cast<size_t>(tid)] = lo < hi ? 1 : 0;
            }
            catch (...) {
                errors[static_cast<size_t>(tid)] = std::current_exception();
            }
        }
        rethrow_first(errors);
        obs.team = team_seen;
        obs.probe_active = static_cast<int>(std::ranges::count(probed_, 1));
        obs.bulk_calls = std::accumulate(bulk_calls.begin(), bulk_calls.end(), size_t{0});
        if (profile) {
            obs.phase[kProbe] += lap(t);
        }
        miss_.clear();
        for (size_t q = 0; q < n; ++q) {
            r.hit[q] = r.id[q] != kNotFound ? 1 : 0;
            if (grow && r.id[q] == kNotFound) {
                miss_.push_back(q);
            }
        }
        if (miss_.empty()) {
            return;
        }
        if (check_distinct) {
            std::vector<std::pair<uint32_t, size_t>> hq;
            for (const auto q : miss_) {
                hq.emplace_back(hash_[q], q);
            }
            require_distinct(b, std::move(hq));
        }
        if (profile) {
            obs.phase[kAssign] += lap(t);
        }
        // grow_rows_geometric refuses an append past the fixed32 ceiling, so every ID below fits TermIndex.
        const size_t base = rows_->grow_rows_geometric(miss_.size());
        for (size_t i = 0; i < miss_.size(); ++i) {
            rows_->set_positions(base + i, b.key(miss_[i]));
        }
        if (profile) {
            obs.phase[kFill] += lap(t);
        }
        const size_t m = miss_.size();
#pragma omp parallel num_threads(threads_)
        {
            const int tid = omp_get_thread_num();
            const int team = omp_get_num_threads();
            if (tid == 0) {
                team_seen = std::min(team_seen, team);
            }
            try {
                const auto [lo, hi] = span_of(m, team, tid);
                for (size_t i = lo; i < hi; ++i) {
                    const auto id = static_cast<TermIndex>(base + i);
                    require(map_.emplace(RowKey{id, hash_[miss_[i]]}, id),
                            "invalid publication: duplicate missing key");
                    r.id[miss_[i]] = id;
                }
                published_[static_cast<size_t>(tid)] = lo < hi ? 1 : 0;
            }
            catch (...) {
                errors[static_cast<size_t>(tid)] = std::current_exception();
            }
        }
        rethrow_first(errors);
        obs.team = team_seen;
        obs.publish_active = static_cast<int>(std::ranges::count(published_, 1));
        if (profile) {
            obs.phase[kPublish] += lap(t);
        }
    }

private:
    // Boost processes bulk visits in groups of 16 (its internal bulk_visit_size); one chunk per call keeps the
    // result matching below a scan of at most 16 queries.
    static constexpr size_t kBulkChunk = 16;

    // Bulk probe of queries [c, e). The visitor receives only the matched element, never the query, and Boost
    // documents no callback order, so it records (hash, row) pairs and matching to queries happens after the call,
    // outside Boost's group locks. Boost invokes it only for an element exactly equal to some chunk query, and
    // equal keys have equal hashes: a lone query with that hash is the match, while several (duplicates or a
    // 32-bit collision) are separated by exact comparison. Misses produce no call and keep kNotFound.
    auto probe_chunk_(const Batch<N> &b, Results &r, size_t c, size_t e) -> void {
        std::array<QueryKey, kBulkChunk> keys{};
        for (size_t q = c; q < e; ++q) {
            hash_[q] = Store::fold_hash_positions(b.key(q));
            keys[q - c] = QueryKey{b.key(q), hash_[q]};
        }
        std::array<RowKey, kBulkChunk> found{};
        size_t nfound = 0;
        map_.cvisit(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(e - c), [&](const auto &kv) {
            if (nfound < kBulkChunk) {
                found[nfound] = RowKey{kv.second, kv.first.hash};
            }
            ++nfound;
        });
        require(nfound <= e - c, "bulk visit reported more matches than queries");
        for (size_t i = 0; i < nfound; ++i) {
            size_t candidates = 0;
            size_t only = kNotFound;
            for (size_t q = c; q < e; ++q) {
                if (hash_[q] == found[i].hash) {
                    ++candidates;
                    only = q;
                }
            }
            if (candidates == 1) {
                r.id[only] = found[i].row;
                continue;
            }
            for (size_t q = c; q < e; ++q) {
                if (hash_[q] == found[i].hash && row_matches(*rows_, found[i].row, b.key(q))) {
                    r.id[q] = found[i].row;
                }
            }
        }
    }

    // Compact handle into the packed rows with its cached 32-bit fold hash; never a copied monomial.
    struct RowKey {
        TermIndex row;
        uint32_t hash;
    };
    // Heterogeneous probe key: a view of one query's positions, valid only for the probe call.
    struct QueryKey {
        std::span<const PosT> pos;
        uint32_t hash;
    };
    struct KeyHash {
        using is_transparent = void;
        auto operator()(const RowKey &k) const noexcept -> size_t { return k.hash; }
        auto operator()(const QueryKey &k) const noexcept -> size_t { return k.hash; }
    };
    struct KeyEqual {
        using is_transparent = void;
        const Store *rows;
        auto operator()(const RowKey &a, const RowKey &b) const -> bool {
            return a.row == b.row || (a.hash == b.hash && rows_equal(*rows, a.row, b.row));
        }
        auto operator()(const QueryKey &q, const RowKey &k) const -> bool {
            return q.hash == k.hash && row_matches(*rows, k.row, q.pos);
        }
        auto operator()(const RowKey &k, const QueryKey &q) const -> bool { return (*this)(q, k); }
    };
    using Map = boost::concurrent_flat_map<RowKey, TermIndex, KeyHash, KeyEqual>;

    int threads_;
    std::unique_ptr<Store> rows_;
    Map map_;
    std::vector<uint32_t> hash_;
    std::vector<size_t> miss_;
    std::vector<uint8_t> probed_, published_;
};

// ---------------------------------------------------------------------------------------------------------------
// Injective synthetic keys: 2N positions split into `length` disjoint bands of `radix` positions; digit i of the
// ordinal (radix `radix`) selects one position of band i after a seed-derived cyclic offset. No dedup table.
template <size_t N>
class KeyGen {
public:
    using PosT = typename OperatorIndex<N>::PosT;

    KeyGen(size_t length, uint64_t seed) : length_(length), radix_(length == 0 ? 0 : 2 * N / length) {
        require(length >= 1 && radix_ >= 2, "key length must be in [1, modes]");
        uint64_t state = seed ^ 0x6A09E667F3BCC909ULL;
        for (size_t i = 0; i < length_; ++i) {
            offset_.push_back(splitmix(state) % radix_);
        }
        capacity_ = 1;
        for (size_t i = 0; i < length_ && capacity_ != UINT64_MAX; ++i) {
            capacity_ = capacity_ > UINT64_MAX / radix_ ? UINT64_MAX : capacity_ * radix_;
        }
    }
    [[nodiscard]] auto length() const -> size_t { return length_; }
    [[nodiscard]] auto capacity() const -> uint64_t { return capacity_; }
    auto key(uint64_t ordinal, PosT *out) const -> void {
        require(ordinal < capacity_, "ordinal does not fit the key encoding");
        for (size_t i = 0; i < length_; ++i) {
            out[i] = static_cast<PosT>((i * radix_) + (((ordinal % radix_) + offset_[i]) % radix_));
            ordinal /= radix_;
        }
    }

private:
    size_t length_;
    uint64_t radix_;
    uint64_t capacity_ = 0;
    std::vector<uint64_t> offset_;
};

// One mixed batch: `hits` distinct existing ordinals (a coprime stride over [0, population)) and `misses` fresh
// ordinals fresh_base, fresh_base+1, ... assigned in query order, so shared IDs equal ordinals after growth.
template <size_t N>
auto make_batch(const KeyGen<N> &gen,
                uint64_t seed,
                uint64_t batch_index,
                uint64_t population,
                uint64_t fresh_base,
                size_t hits,
                size_t misses,
                Batch<N> &b,
                std::vector<uint64_t> &ordinal,
                std::vector<uint8_t> &declared_hit) -> void {
    require(hits <= population, "more distinct hits requested than existing rows");
    const size_t n = hits + misses;
    declared_hit.assign(n, 0);
    std::fill_n(declared_hit.begin(), hits, uint8_t{1});
    uint64_t state = (seed * 0xD1B54A32D192ED03ULL) ^ (batch_index + 1);
    for (size_t i = n; i > 1; --i) {
        std::swap(declared_hit[i - 1], declared_hit[splitmix(state) % i]);
    }
    const uint64_t start = population == 0 ? 0 : splitmix(state) % population;
    uint64_t stride = std::max<uint64_t>(1, static_cast<uint64_t>(static_cast<double>(population) * 0.6180339887));
    while (population > 1 && std::gcd(stride, population) != 1) {
        ++stride;
    }
    b.clear();
    ordinal.resize(n);
    std::vector<typename KeyGen<N>::PosT> buf(gen.length());
    uint64_t j = 0;
    uint64_t k = 0;
    for (size_t q = 0; q < n; ++q) {
        ordinal[q] = declared_hit[q] != 0
                         ? static_cast<uint64_t>((start + (static_cast<unsigned __int128>(j++) * stride)) % population)
                         : fresh_base + k++;
        gen.key(ordinal[q], buf.data());
        b.push(buf);
    }
}

// Sequential ordinals [first, first+count) in order, for initial streaming fills and final validation.
template <size_t N>
auto make_range(const KeyGen<N> &gen, uint64_t first, size_t count, Batch<N> &b) -> void {
    b.clear();
    std::vector<typename KeyGen<N>::PosT> buf(gen.length());
    for (size_t i = 0; i < count; ++i) {
        gen.key(first + i, buf.data());
        b.push(buf);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Measurement driver: one variant/case per fresh process, one CSV row on success.
struct Config {
    std::string variant;
    std::string lookup = "per-query"; //!< shared-concurrent probe kernel: per-query or bulk.
    bool grow = false;
    uint64_t rows = 0;
    uint64_t batch = 4096;
    uint64_t batches = 16;
    uint64_t miss_percent = 0;
    uint64_t key_length = 6;
    uint64_t modes = 128;
    uint64_t seed = 0;
    uint64_t inline_width = 11;
    bool profile = false;
};

struct Totals {
    int team_min = 1 << 30, team_max = 0, probe_min = 1 << 30, probe_max = 0, publish_min = 1 << 30, publish_max = 0;
    std::array<double, kNumPhases> phase{};
    auto add(const BatchObs &o) -> void {
        team_min = std::min(team_min, o.team);
        team_max = std::max(team_max, o.team);
        probe_min = std::min(probe_min, o.probe_active);
        probe_max = std::max(probe_max, o.probe_active);
        if (o.publish_active > 0) {
            publish_min = std::min(publish_min, o.publish_active);
            publish_max = std::max(publish_max, o.publish_active);
        }
        for (size_t p = 0; p < kNumPhases; ++p) {
            phase[p] += o.phase[p];
        }
    }
};

template <size_t N, class V>
auto check_batch(const V &v,
                 const Batch<N> &b,
                 const std::vector<uint64_t> &ordinal,
                 const std::vector<uint8_t> &declared_hit,
                 const Results &r,
                 bool grow,
                 std::vector<size_t> &next_new) -> void {
    for (size_t q = 0; q < b.size(); ++q) {
        const auto key = b.key(q);
        const uint32_t o = v.owner_of(key);
        require(r.hit[q] == declared_hit[q], "hit/miss disagrees with the declared workload");
        require(r.owner[q] == o, "result owner disagrees with the routing owner");
        if (!declared_hit[q] && !grow) {
            require(r.id[q] == kNotFound, "lookup miss returned a row");
            continue;
        }
        require(v.matches(o, r.id[q], key), "returned row does not hold the exact query key");
        if constexpr (V::kIdsAreOrdinals) {
            require(r.id[q] == ordinal[q], "shared ID is not the canonical ordinal");
        }
        if (!declared_hit[q]) {
            require(r.id[q] == next_new[o]++, "published ID breaks per-owner query order");
        }
    }
}

template <size_t N, class V>
auto measure(V &v, const Config &c, int threads) -> int {
    const KeyGen<N> gen(c.key_length, c.seed);
    const uint64_t misses = c.batch * c.miss_percent / 100;
    const uint64_t hits = c.batch - misses;
    const uint64_t final_rows = c.grow ? c.rows + (c.batches * misses) : c.rows;
    Batch<N> b;
    Results r;
    BatchObs scratch;
    std::vector<uint64_t> ordinal;
    std::vector<uint8_t> declared;
    std::vector<size_t> next_new(v.owners(), 0);
    constexpr uint64_t kFillChunk = 65536;
    const auto t_setup = Clock::now();
    for (uint64_t first = 0; first < c.rows; first += kFillChunk) {
        const size_t count = std::min(kFillChunk, c.rows - first);
        make_range(gen, first, count, b);
        v.run(b, true, r, scratch, false);
        ordinal.resize(count);
        std::iota(ordinal.begin(), ordinal.end(), first);
        declared.assign(count, 0);
        check_batch(v, b, ordinal, declared, r, true, next_new);
    }
    const double setup_s = std::chrono::duration<double>(Clock::now() - t_setup).count();

#ifdef BOOST_UNORDERED_ENABLE_STATS
    // Statistics cover the measured batches only: reset after setup, snapshot before the final validation.
    if constexpr (requires { v.reset_boost_stats(); }) {
        v.reset_boost_stats();
    }
#endif
    Totals tot;
    double batch_s = 0;
    double validate_s = 0;
    uint64_t checksum = 0;
    uint64_t population = c.rows;
    uint64_t fresh = c.rows;
    for (uint64_t bi = 0; bi < c.batches; ++bi) {
        make_batch(gen, c.seed, bi, population, fresh, hits, misses, b, ordinal, declared);
        BatchObs obs;
        const auto t0 = Clock::now();
        v.run(b, c.grow, r, obs, c.profile);
        batch_s += std::chrono::duration<double>(Clock::now() - t0).count();
        tot.add(obs);
        const auto tv = Clock::now();
        check_batch(v, b, ordinal, declared, r, c.grow, next_new);
        for (const auto id : r.id) {
            checksum = (checksum * 0x100000001B3ULL) ^ id;
        }
        validate_s += std::chrono::duration<double>(Clock::now() - tv).count();
        fresh += misses;
        population += c.grow ? misses : 0;
    }

    std::string stats_header;
    std::string stats_row;
#ifdef BOOST_UNORDERED_ENABLE_STATS
    if constexpr (requires { v.boost_stats(); }) {
        const auto s = v.boost_stats();
        stats_header = ",bs_insert_count,bs_insert_probe_mean,bs_hit_count,bs_hit_probe_mean,bs_hit_cmp_mean,"
                       "bs_miss_count,bs_miss_probe_mean,bs_miss_cmp_mean";
        stats_row = std::format(",{},{:.4f},{},{:.4f},{:.4f},{},{:.4f},{:.4f}",
                                s.insertion.count,
                                s.insertion.probe_length.average,
                                s.successful_lookup.count,
                                s.successful_lookup.probe_length.average,
                                s.successful_lookup.num_comparisons.average,
                                s.unsuccessful_lookup.count,
                                s.unsuccessful_lookup.probe_length.average,
                                s.unsuccessful_lookup.num_comparisons.average);
    }
#endif

    // Retained-set check without a shadow map: exact count, then a streaming lookup of every expected key.
    // Ordinal order is insertion order within every owner, so expected owner-local IDs are one counter each.
    const auto tv = Clock::now();
    size_t total = 0;
    size_t min_rows = SIZE_MAX;
    size_t max_rows = 0;
    for (size_t o = 0; o < v.owners(); ++o) {
        total += v.owner_size(o);
        min_rows = std::min(min_rows, v.owner_size(o));
        max_rows = std::max(max_rows, v.owner_size(o));
    }
    require(total == final_rows, "final row count differs from the workload");
    if constexpr (requires { v.index_entries(); }) {
        require(v.index_entries() == final_rows, "concurrent index size differs from the row count");
    }
    std::vector<size_t> expect(v.owners(), 0);
    for (uint64_t first = 0; first < final_rows; first += kFillChunk) {
        const size_t count = std::min(kFillChunk, final_rows - first);
        make_range(gen, first, count, b);
        v.run(b, false, r, scratch, false);
        for (size_t q = 0; q < count; ++q) {
            const uint32_t o = v.owner_of(b.key(q));
            require(r.hit[q] == 1 && r.owner[q] == o && r.id[q] == expect[o]++, "retained key has a wrong owner/ID");
            if constexpr (V::kIdsAreOrdinals) {
                require(r.id[q] == first + q, "retained shared ID is not canonical");
            }
            require(v.matches(o, r.id[q], b.key(q)), "retained row does not hold its exact key");
        }
    }
    validate_s += std::chrono::duration<double>(Clock::now() - tv).count();

    const uint64_t queries = c.batch * c.batches;
    std::string header =
        "variant,mode,modes,key_length,inline_width,rows,batch,batches,miss_percent,seed,requested_threads,team_min,"
        "team_max,probe_active_min,probe_active_max,publish_active_min,publish_active_max,initial_rows,final_rows,"
        "hits,misses,setup_s,batch_s,queries_per_s";
    std::string row = std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6f},{:.6f},{:.1f}",
                                  c.variant,
                                  c.grow ? "grow" : "lookup",
                                  c.modes,
                                  c.key_length,
                                  c.inline_width,
                                  c.rows,
                                  c.batch,
                                  c.batches,
                                  c.miss_percent,
                                  c.seed,
                                  threads,
                                  tot.team_min,
                                  tot.team_max,
                                  tot.probe_min,
                                  tot.probe_max,
                                  tot.publish_max == 0 ? 0 : tot.publish_min,
                                  tot.publish_max,
                                  c.rows,
                                  final_rows,
                                  hits * c.batches,
                                  misses * c.batches,
                                  setup_s,
                                  batch_s,
                                  static_cast<double>(queries) / batch_s);
    for (size_t p = 0; p < kNumPhases; ++p) {
        header += std::format(",{}_s", kPhaseNames[p]);
        // Phases a variant or mode does not have stay empty rather than reporting invented zeros.
        const bool applies = V::kPhases[p] && (c.grow || (p != kAssign && p != kFill && p != kPublish));
        row += c.profile && applies ? std::format(",{:.6f}", tot.phase[p]) : std::string(",");
    }
    header += ",min_shard_rows,max_shard_rows,row_bytes_est,index_bytes_est,harness_bytes_est,validate_s,validation,"
              "checksum,profile,lookup";
    row += std::format(",{},{},{},{},{},{:.6f},ok,{:016x},{},{}",
                       min_rows,
                       max_rows,
                       v.row_bytes(),
                       v.index_bytes(),
                       v.harness_bytes(),
                       validate_s,
                       checksum,
                       c.profile ? 1 : 0,
                       c.lookup);
    std::println("{}{}\n{}{}", header, stats_header, row, stats_row);
    return 0;
}

template <size_t N>
auto measure_variant(const Config &c) -> int {
    const int threads = omp_get_max_threads(); // captured once; every region requests exactly this team
    if (c.variant == "per-thread") {
        PerThread<N> v(threads, threads, c.inline_width);
        return measure<N>(v, c, threads);
    }
    if (c.variant == "shared-serial") {
        SharedSerial<N> v(threads, c.inline_width);
        return measure<N>(v, c, threads);
    }
    SharedConcurrent<N> v(threads, c.inline_width);
    v.bulk_lookup = c.lookup == "bulk";
    return measure<N>(v, c, threads);
}

// ---------------------------------------------------------------------------------------------------------------
// Self-tests. The serial key oracle below exists only here, never in measured processes.
template <size_t N>
struct Oracle {
    using PosT = typename OperatorIndex<N>::PosT;
    std::vector<std::map<std::vector<PosT>, size_t>> ids;
    explicit Oracle(size_t owners) : ids(owners) {}
    [[nodiscard]] auto owner_of(std::span<const PosT> key) const -> uint32_t {
        return ids.size() == 1 ? 0
                               : static_cast<uint32_t>(monoprop::monomial_hash<N>(monomial_of<N>(key)) % ids.size());
    }
    // Probes the frozen pre-batch state, then assigns misses in query order per owner.
    auto apply(const Batch<N> &b, bool grow) -> Results {
        Results e;
        e.resize(b.size());
        for (size_t q = 0; q < b.size(); ++q) {
            const auto o = owner_of(b.key(q));
            const auto it = ids[o].find({b.key(q).begin(), b.key(q).end()});
            e.owner[q] = o;
            e.hit[q] = it != ids[o].end();
            e.id[q] = e.hit[q] != 0 ? it->second : kNotFound;
        }
        for (size_t q = 0; grow && q < b.size(); ++q) {
            if (e.hit[q] == 0) {
                auto &m = ids[e.owner[q]];
                e.id[q] = m.size();
                m.emplace(std::vector<PosT>(b.key(q).begin(), b.key(q).end()), e.id[q]);
            }
        }
        return e;
    }
};

template <size_t N>
auto batch_of(std::initializer_list<std::vector<typename OperatorIndex<N>::PosT>> keys) -> Batch<N> {
    Batch<N> b;
    for (const auto &k : keys) {
        b.push(k);
    }
    return b;
}

// Runs one batch through the variant and the oracle; results and every returned row's exact key must agree.
template <size_t N, class V>
auto step(V &v, Oracle<N> &oracle, const Batch<N> &b, bool grow, std::string_view what) -> Results {
    Results r;
    BatchObs obs;
    v.run(b, grow, r, obs, true);
    const Results e = oracle.apply(b, grow);
    for (size_t q = 0; q < b.size(); ++q) {
        require(r.hit[q] == e.hit[q] && r.id[q] == e.id[q] && r.owner[q] == e.owner[q],
                std::format("{}: query {} got hit={} owner={} id={}, expected hit={} owner={} id={}",
                            what,
                            q,
                            r.hit[q],
                            r.owner[q],
                            r.id[q],
                            e.hit[q],
                            e.owner[q],
                            e.id[q]));
        require(r.id[q] == kNotFound || v.matches(r.owner[q], r.id[q], b.key(q)),
                std::format("{}: wrong row key", what));
    }
    return r;
}

// Two distinct keys with the same cached 32-bit fold hash, found by a birthday search over synthetic keys.
template <size_t N>
auto hash_collision() -> std::pair<std::vector<uint16_t>, std::vector<uint16_t>> {
    const KeyGen<N> gen(6, 7);
    std::unordered_map<uint32_t, uint64_t> seen;
    std::vector<typename KeyGen<N>::PosT> a(6);
    std::vector<typename KeyGen<N>::PosT> c(6);
    for (uint64_t ord = 0;; ++ord) {
        gen.key(ord, a.data());
        const auto [it, fresh] = seen.emplace(OperatorIndex<N>::fold_hash_positions(a), ord);
        if (!fresh) {
            gen.key(it->second, c.data());
            return {{c.begin(), c.end()}, {a.begin(), a.end()}};
        }
    }
}

template <size_t N, class V>
auto fixtures(V &v, size_t owners, std::string_view name) -> void {
    using PosT = typename OperatorIndex<N>::PosT;
    Oracle<N> oracle(owners);
    const std::vector<PosT> a{0, 1}, bk{2, 3}, c{4, 5}, d{6, 7};
    const std::vector<PosT> far{0, static_cast<PosT>(2 * N - 1)};
    v.check_distinct = true;
    step(v, oracle, Batch<N>{}, true, std::format("{} empty batch/empty store", name));
    step(v, oracle, batch_of<N>({a, far}), false, std::format("{} lookup in empty store", name));
    step(v, oracle, batch_of<N>({a}), true, std::format("{} initial A", name));
    const Results r1 = step(v, oracle, batch_of<N>({bk, a, c}), true, std::format("{} B,A,C", name));
    const Results r2 = step(v, oracle, batch_of<N>({c, bk, a}), false, std::format("{} C,B,A", name));
    if (owners == 1) {
        require(r1.hit == std::vector<uint8_t>{0, 1, 0} && r1.id == std::vector<size_t>{1, 0, 2}, "B,A,C shared IDs");
        require(r2.id == std::vector<size_t>{2, 1, 0}, "C,B,A shared IDs");
    }
    step(v, oracle, batch_of<N>({a, a}), false, std::format("{} repeated A,A", name));
    step(v, oracle, batch_of<N>({a, a}), true, std::format("{} repeated A,A (grow)", name));
    const size_t before = [&] {
        size_t s = 0;
        for (size_t o = 0; o < v.owners(); ++o) {
            s += v.owner_size(o);
        }
        return s;
    }();
    bool rejected = false;
    try {
        Results r;
        BatchObs obs;
        v.run(batch_of<N>({d, d}), true, r, obs, false);
    }
    catch (const std::runtime_error &e) {
        rejected = std::string_view(e.what()).find("duplicate") != std::string_view::npos;
    }
    size_t after = 0;
    for (size_t o = 0; o < v.owners(); ++o) {
        after += v.owner_size(o);
    }
    require(rejected && after == before, std::format("{}: duplicate-miss fixture not rejected cleanly", name));
    step(v, oracle, Batch<N>{}, false, std::format("{} empty batch/non-empty store", name));

    // Tails around the 16-query probe pipeline and 64-query blocks, then all-hit and all-miss repeats.
    const KeyGen<N> gen(6, 11);
    uint64_t next = 1000;
    for (const size_t n : {1UL, 15UL, 16UL, 17UL, 63UL, 64UL, 65UL, 4096UL}) {
        Batch<N> fresh;
        make_range(gen, next, n, fresh);
        Batch<N> absent;
        make_range(gen, next + n, n, absent);
        next += 2 * n;
        step(v, oracle, fresh, true, std::format("{} tail {} all-miss grow", name, n));
        step(v, oracle, fresh, false, std::format("{} tail {} all-hit", name, n));
        step(v, oracle, absent, false, std::format("{} tail {} all-miss lookup", name, n));
    }
    // Mixed batches from the measured generator.
    std::vector<uint64_t> ord;
    std::vector<uint8_t> declared;
    for (uint64_t bi = 0; bi < 4; ++bi) {
        Batch<N> mixed;
        make_batch(gen, 3, bi, 1000, next, 700, 300, mixed, ord, declared);
        next += 300;
        step(v, oracle, mixed, bi % 2 == 0, std::format("{} mixed batch {}", name, bi));
    }

    // Equal cached hashes: exact comparison must separate the pair before and after table growth.
    const auto [x16, y16] = hash_collision<N>();
    const std::vector<PosT> x(x16.begin(), x16.end()), y(y16.begin(), y16.end());
    require(OperatorIndex<N>::fold_hash_positions(x) == OperatorIndex<N>::fold_hash_positions(y) && x != y,
            "collision search");
    step(v, oracle, batch_of<N>({x}), true, std::format("{} collision X", name));
    step(v, oracle, batch_of<N>({y}), false, std::format("{} collision Y absent", name));
    step(v, oracle, batch_of<N>({y, x}), true, std::format("{} collision Y publish", name));
    Batch<N> filler;
    make_range(gen, next, 3000, filler);
    next += 3000;
    step(v, oracle, filler, true, std::format("{} collision growth", name));
    step(v, oracle, batch_of<N>({x, y}), false, std::format("{} collision after growth", name));

    // Spilled length-16 rows beside inline rows, across repeated backing reallocations.
    const KeyGen<N> spill_gen(16, 5);
    std::vector<Batch<N>> earlier;
    for (uint64_t g = 0; g < 12; ++g) {
        Batch<N> spill;
        make_range(spill_gen, g * 250, 250, spill);
        step(v, oracle, spill, true, std::format("{} spill growth {}", name, g));
        earlier.push_back(std::move(spill));
        for (const auto &e : earlier) {
            step(v, oracle, e, false, std::format("{} spill re-lookup {}", name, g));
        }
    }
}

// Exercises the three variants at one requested team size.
template <size_t N>
auto self_test_team(int team) -> void {
    {
        PerThread<N> v(team, team, 11);
        fixtures<N>(v, v.owners(), std::format("per-thread N={} T={} S={}", N, team, team));
    }
    {
        PerThread<N> v(3, team, 11); // fixed shard count: IDs must not depend on the team
        fixtures<N>(v, 3, std::format("per-thread N={} T={} S=3", N, team));
    }
    {
        SharedSerial<N> v(team, 11);
        fixtures<N>(v, 1, std::format("shared-serial N={} T={}", N, team));
    }
    {
        SharedConcurrent<N> v(team, 11);
        fixtures<N>(v, 1, std::format("shared-concurrent N={} T={}", N, team));
    }
    {
        SharedConcurrent<N> v(team, 11);
        v.bulk_lookup = true;
        fixtures<N>(v, 1, std::format("shared-concurrent/bulk N={} T={}", N, team));
    }
}

// Actual participation in probe/publication bodies and joined worker exceptions, at a team of four.
template <class V>
auto self_test_team_behaviour(std::string_view name, V &&make) -> void {
    Batch<128> b;
    make_range(KeyGen<128>(6, 1), 0, 4096, b);
    Results r;
    BatchObs obs;
    auto v = make();
    v->run(b, true, r, obs, true);
    if constexpr (requires { v->bulk_lookup; }) {
        // 4096 queries in 16-query chunks: the bulk path issues exactly one call per chunk at any team size.
        require(!v->bulk_lookup || obs.bulk_calls == 4096 / 16, std::format("{}: bulk path not taken", name));
    }
    if (obs.team < 4) {
        std::println("SKIP participation {}: runtime limited the team to {} (not T-core evidence)", name, obs.team);
    }
    else {
        require(obs.probe_active == 4, std::format("{}: only {} probe workers active", name, obs.probe_active));
        std::println("participation {}: team={} probe_active={} publish_active={}",
                     name,
                     obs.team,
                     obs.probe_active,
                     obs.publish_active);
    }
    if (obs.team < 2) {
        std::println("SKIP worker-exception {}: team of one", name);
        return;
    }
    auto w = make();
    g_fault_tid.store(1);
    bool caught = false;
    try {
        w->run(b, true, r, obs, false);
    }
    catch (const std::runtime_error &e) {
        caught = std::string_view(e.what()) == "injected worker fault";
    }
    g_fault_tid.store(-1);
    require(caught, std::format("{}: worker exception was not joined and rethrown", name));
    std::println("worker-exception {}: joined and rethrown on the caller", name);
}

#ifdef BOOST_UNORDERED_ENABLE_STATS
// Diagnostic builds only: Boost's own counters must see exactly the index traffic the adapter issues.
auto self_test_boost_stats() -> void {
    for (const bool bulk : {false, true}) {
        SharedConcurrent<128> v(1, 11);
        v.bulk_lookup = bulk;
        Batch<128> b;
        make_range(KeyGen<128>(6, 2), 0, 100, b);
        Results r;
        BatchObs obs;
        v.reset_boost_stats();
        v.run(b, true, r, obs, false);  // 100 misses, then 100 inserts
        v.run(b, false, r, obs, false); // 100 hits
        const auto s = v.boost_stats();
        // Boost counts its own traffic too: each emplace performs an absence lookup (an unsuccessful lookup, retried
        // after a rehash), and rehash transfers count as insertions. Only hits map one-to-one to adapter probes.
        require(s.successful_lookup.count == 100 && s.unsuccessful_lookup.count >= 200 && s.insertion.count >= 100,
                std::format("boost stats (bulk={}): miss={} insert={} hit={}",
                            bulk,
                            s.unsuccessful_lookup.count,
                            s.insertion.count,
                            s.successful_lookup.count));
    }
    std::println("boost container stats: hit counts match the adapter's probes");
}
#endif

auto self_test() -> int {
#ifdef BOOST_UNORDERED_ENABLE_STATS
    self_test_boost_stats();
#endif
    for (const int team : {1, 2, 3, 4}) {
        self_test_team<128>(team);
        self_test_team<1024>(team);
        std::println("fixtures ok at requested team {}", team);
    }
    self_test_team_behaviour("per-thread", [] { return std::make_unique<PerThread<128>>(4, 4, 11); });
    self_test_team_behaviour("shared-serial", [] { return std::make_unique<SharedSerial<128>>(4, 11); });
    self_test_team_behaviour("shared-concurrent", [] { return std::make_unique<SharedConcurrent<128>>(4, 11); });
    self_test_team_behaviour("shared-concurrent/bulk", [] {
        auto v = std::make_unique<SharedConcurrent<128>>(4, 11);
        v->bulk_lookup = true;
        return v;
    });
    std::println("self-test passed (Boost {}.{}, _OPENMP {})",
                 BOOST_VERSION / 100000,
                 BOOST_VERSION / 100 % 1000,
                 _OPENMP);
    return 0;
}

auto parse_u64(std::string_view flag, std::string_view text) -> uint64_t {
    uint64_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    require(ec == std::errc{} && end == text.data() + text.size() && !text.empty(),
            std::format("{}: expected a non-negative decimal integer, got '{}'", flag, text));
    return value;
}

auto checked_mul(uint64_t a, uint64_t b) -> uint64_t {
    uint64_t out = 0;
    require(!__builtin_mul_overflow(a, b, &out), "workload size overflows 64-bit arithmetic");
    return out;
}

auto parse(int argc, char **argv) -> Config {
    Config c;
    std::string mode;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--profile") {
            c.profile = true;
            continue;
        }
        require(i + 1 < argc, std::format("{}: missing value", flag));
        const std::string_view value = argv[++i];
        if (flag == "--variant") {
            c.variant = value;
        }
        else if (flag == "--lookup") {
            c.lookup = value;
        }
        else if (flag == "--mode") {
            mode = value;
        }
        else if (flag == "--rows") {
            c.rows = parse_u64(flag, value);
        }
        else if (flag == "--batch") {
            c.batch = parse_u64(flag, value);
        }
        else if (flag == "--batches") {
            c.batches = parse_u64(flag, value);
        }
        else if (flag == "--miss-percent") {
            c.miss_percent = parse_u64(flag, value);
        }
        else if (flag == "--key-length") {
            c.key_length = parse_u64(flag, value);
        }
        else if (flag == "--modes") {
            c.modes = parse_u64(flag, value);
        }
        else if (flag == "--seed") {
            c.seed = parse_u64(flag, value);
        }
        else if (flag == "--inline-width") {
            c.inline_width = parse_u64(flag, value);
        }
        else {
            require(false, std::format("unknown option {}", flag));
        }
    }
    require(c.variant == "per-thread" || c.variant == "shared-serial" || c.variant == "shared-concurrent",
            "--variant must be per-thread, shared-serial or shared-concurrent");
    require(mode == "lookup" || mode == "grow", "--mode must be lookup or grow");
    require(c.lookup == "per-query" || (c.lookup == "bulk" && c.variant == "shared-concurrent"),
            "--lookup must be per-query, or bulk with --variant shared-concurrent");
    c.grow = mode == "grow";
    require(c.modes == 128 || c.modes == 1024, "--modes must be 128 or 1024");
    require(c.miss_percent <= 100, "--miss-percent must be in [0, 100]");
    require(c.batch >= 1 && c.batches >= 1, "--batch and --batches must be positive");
    require(c.inline_width >= 1 && c.inline_width <= OperatorIndex<128>::kMaxInlinePositions,
            "--inline-width must be in [1, 32]");
    require(c.key_length >= 1 && c.key_length <= c.modes, "--key-length must be in [1, modes]");
    const uint64_t misses = checked_mul(c.batch, c.miss_percent) / 100;
    require(c.batch - misses <= c.rows, "more distinct hits per batch than initial rows");
    const uint64_t added = checked_mul(c.batches, misses);
    require(added <= UINT64_MAX - c.rows, "workload size overflows 64-bit arithmetic");
    // Lookup misses use fresh never-inserted ordinals too, so both modes need rows+added encodable keys.
    require(c.rows + added < OperatorIndex<128>::kIndexCeiling, "study sizes must stay below the fixed32 ceiling");
    const uint64_t capacity =
        c.modes == 128 ? KeyGen<128>(c.key_length, 0).capacity() : KeyGen<1024>(c.key_length, 0).capacity();
    require(c.rows + added <= capacity, "key length cannot encode this many distinct keys");
    return c;
}

} // namespace

auto main(int argc, char **argv) -> int {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
            return self_test();
        }
        const Config c = parse(argc, argv);
        return c.modes == 128 ? measure_variant<128>(c) : measure_variant<1024>(c);
    }
    catch (const std::exception &e) {
        std::println(stderr, "index-miniapp: FAILED: {}", e.what());
        return 1;
    }
}
