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

// Test-only support for the threaded cosine and fused-apply kernels: a range observer that records which
// OpenMP worker ran each logical range, a synthetic inverted index large enough to span several fold
// blocks, and an independent add-owner count over fused rotation records.
//
// Workers never assert. RangeLog slots are sized by the kernel's prepare() on the calling thread, before
// any worker starts, and each slot is written only by the sole owner of its range; tests inspect them
// after the kernel has joined.

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/InvertedIndex.h"

namespace kernel_test {

using monoprop::detail::KernelRange;

inline constexpr size_t kKinds = 9;

// What one kernel invocation reported for one kind of range.
struct RangeSlots {
    size_t prepares = 0;     // prepare() calls, on the calling thread
    std::vector<int> worker; // omp_get_thread_num() of each range's owner, -1 if never visited
    std::vector<int> team;   // omp_get_num_threads() seen by that owner
    std::vector<int> level;  // omp_get_level() seen by that owner (0: not inside any region)
    std::vector<int> visits; // visits per range; each element has one writer
};

struct RangeLog {
    std::array<RangeSlots, kKinds> slots{};
    auto operator[](KernelRange kind) -> RangeSlots & { return slots[static_cast<size_t>(kind)]; }
};

// Records the worker that owns each logical range. Optionally throws from one range, on its worker, to
// exercise the helper's capture-and-join path from inside a real kernel body.
struct RecordingObserver {
    RangeLog *log;
    KernelRange throw_kind = KernelRange::cos_lazy;
    size_t throw_range = SIZE_MAX;

    auto prepare(KernelRange kind, size_t ranges) const -> void {
        auto &s = (*log)[kind];
        ++s.prepares;
        s.worker.assign(ranges, -1);
        s.team.assign(ranges, 0);
        s.level.assign(ranges, 0);
        s.visits.assign(ranges, 0);
    }
    auto visit(KernelRange kind, size_t range) const -> void {
        auto &s = (*log)[kind];
        s.worker[range] = omp_get_thread_num();
        s.team[range] = omp_get_num_threads();
        s.level[range] = omp_get_level();
        ++s.visits[range];
        if (kind == throw_kind && range == throw_range) {
            throw std::runtime_error("injected kernel worker failure");
        }
    }
};

// Keeps every prepare() call, for kernels that run once per window or once per exchange pass (the self probe,
// the incoming decode/probe/scatter): prepare appends a fresh record on the calling thread, and each range's
// owner writes only its own slots of the newest record. Records are inspected after the kernel has joined.
struct PhaseLog {
    struct Call {
        KernelRange kind;
        std::vector<int> worker; // -1 if never visited
        std::vector<int> team;
        std::vector<int> level;
        std::vector<int> visits;
    };
    std::vector<Call> calls;

    //! The calls of one kind, in order.
    [[nodiscard]] auto of(KernelRange kind) const -> std::vector<const Call *> {
        std::vector<const Call *> out;
        for (const auto &c : calls) {
            if (c.kind == kind) {
                out.push_back(&c);
            }
        }
        return out;
    }
};

struct AccumulatingObserver {
    PhaseLog *log;
    KernelRange throw_kind = KernelRange::cos_lazy;
    size_t throw_call = SIZE_MAX; // which call of throw_kind, counted from zero
    size_t throw_range = SIZE_MAX;

    auto prepare(KernelRange kind, size_t ranges) const -> void {
        log->calls.push_back({kind,
                              std::vector<int>(ranges, -1),
                              std::vector<int>(ranges, 0),
                              std::vector<int>(ranges, 0),
                              std::vector<int>(ranges, 0)});
    }
    auto visit(KernelRange kind, size_t range) const -> void {
        auto &c = log->calls.back();
        c.worker[range] = omp_get_thread_num();
        c.team[range] = omp_get_num_threads();
        c.level[range] = omp_get_level();
        ++c.visits[range];
        if (kind == throw_kind && range == throw_range && log->of(kind).size() == throw_call + 1) {
            throw std::runtime_error("injected resolve worker failure");
        }
    }
};

// A PhaseLog call as RangeSlots, so the participation/serial checks below apply to it.
inline auto slots_of(const PhaseLog::Call &c) -> RangeSlots {
    return RangeSlots{.prepares = 1, .worker = c.worker, .team = c.team, .level = c.level, .visits = c.visits};
}

// Evidence that one kernel's own ranges ran on several workers of a region the kernel opened.
struct Participation {
    size_t ranges = 0;
    size_t distinct_workers = 0;
    int team = 0;
    bool every_range_once = false;
    bool inside_region = false; // every range saw omp_get_level() >= 1
};

inline auto participation(const RangeSlots &s) -> Participation {
    Participation p;
    p.ranges = s.worker.size();
    p.every_range_once = std::ranges::all_of(s.visits, [](int v) { return v == 1; });
    p.inside_region = !s.level.empty() && std::ranges::all_of(s.level, [](int l) { return l >= 1; });
    const std::set<int> distinct(s.worker.begin(), s.worker.end());
    p.distinct_workers = distinct.size();
    p.team = s.team.empty() ? 0 : *std::ranges::max_element(s.team);
    return p;
}

// Whether a two-worker team is a fair expectation: a dynamic runtime or a thread limit of one may shrink
// the region, which is then reported, never counted as participation.
inline auto team_may_shrink() -> bool {
    return omp_get_dynamic() != 0 || omp_get_thread_limit() < 2;
}

// Checks on the caller that the kernel's ranges ran inside its own region, each exactly once, on as many
// distinct workers as the region had, and on at least two of them unless the runtime may shrink teams.
inline auto check_participation(const RangeSlots &s, size_t expected_ranges, const char *label) -> void {
    const auto p = participation(s);
    BOOST_TEST_CONTEXT(label << ": ranges=" << p.ranges << " distinct_workers=" << p.distinct_workers
                             << " team=" << p.team) {
        BOOST_TEST(s.prepares == 1u);
        BOOST_TEST(p.ranges == expected_ranges);
        BOOST_TEST(p.every_range_once);
        BOOST_TEST(p.inside_region);
        // A static schedule over at least as many ranges as workers gives every worker a range.
        BOOST_TEST(p.distinct_workers == static_cast<size_t>(p.team));
        if (team_may_shrink()) {
            BOOST_WARN_MESSAGE(p.team >= 2,
                               label << ": multi-worker participation NOT demonstrated (team " << p.team
                                     << " under a dynamic or limited runtime)");
        }
        else {
            BOOST_TEST(p.team >= 2);
        }
    }
    BOOST_TEST_MESSAGE(label << ": " << p.distinct_workers << " distinct workers over " << p.ranges << " ranges");
}

// Checks that a kernel stayed on the calling thread: every range visited once, outside any region.
inline auto check_serial(const RangeSlots &s, size_t expected_ranges, const char *label) -> void {
    const auto p = participation(s);
    BOOST_TEST_CONTEXT(label) {
        BOOST_TEST(s.prepares == 1u);
        BOOST_TEST(p.ranges == expected_ranges);
        BOOST_TEST(p.every_range_once);
        BOOST_TEST(std::ranges::all_of(s.level, [](int l) { return l == 0; }));
    }
}

// A test-side probe, not library behavior: can this runtime give a two-thread region right now?
inline auto runtime_offers_two_workers(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    if (omp_get_thread_limit() < 2) {
        boost::test_tools::assertion_result result(false);
        result.message() << "OMP_THREAD_LIMIT is below two";
        return result;
    }
    int team = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    boost::test_tools::assertion_result result(team >= 2);
    result.message() << "the OpenMP runtime provided " << team << " worker(s) for a two-thread request";
    return result;
}

// Sets monoprop_NUM_THREADS for one scope, so an object constructed inside captures that budget, then
// restores the previous value. Only the one-store prototype (explicit partitions=1 on an ordinary
// communicator) captures it; tests confirm selection through PropagatorTestAccess::options.
class ScopedBudget {
public:
    explicit ScopedBudget(int threads) {
        if (const char *old = std::getenv(kVariable)) {
            old_ = std::string(old);
        }
        ::setenv(kVariable, std::to_string(threads).c_str(), 1);
    }
    ScopedBudget(const ScopedBudget &) = delete;
    auto operator=(const ScopedBudget &) -> ScopedBudget & = delete;
    ~ScopedBudget() {
        if (old_) {
            ::setenv(kVariable, old_->c_str(), 1);
        }
        else {
            ::unsetenv(kVariable);
        }
    }

private:
    static constexpr const char *kVariable = "monoprop_NUM_THREADS";
    std::optional<std::string> old_;
};

// Deterministic, library-independent pseudo-random words (splitmix64).
struct SplitMix {
    uint64_t state;
    auto next() -> uint64_t {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    auto below(uint64_t n) -> uint64_t { return next() % n; }
};

inline constexpr size_t kIndexModes = 8; // 16 columns

// A synthetic InvertedIndex<8> over `rows` rows with every column tier a fold can meet: dense random
// columns, sparse columns (including rows straddling fold-block edges), a dense-but-sparse column and
// empty columns. Consistent with an operator whose row r touches column c iff bit r of column c is set,
// so its lazily built row parity is well defined. Bits beyond `rows` stay clear, as in the real index.
inline auto make_synthetic_index(size_t rows, uint64_t seed) -> monoprop::detail::InvertedIndex<kIndexModes> {
    using Index = monoprop::detail::InvertedIndex<kIndexModes>;
    Index index;
    index.row_count = rows;
    const size_t words = index.words();
    const uint64_t last_mask = rows % 64 == 0 ? ~uint64_t{0} : (uint64_t{1} << (rows % 64)) - 1;
    SplitMix rng{seed};
    const auto dense = [&](size_t c, int sparsity_shift) {
        auto &col = index.cols[c];
        col.is_dense = true;
        col.words.assign(words, 0);
        for (auto &w : col.words) {
            w = rng.next();
            for (int s = 0; s < sparsity_shift; ++s) {
                w &= rng.next();
            }
        }
        if (words != 0) {
            col.words.back() &= last_mask;
        }
    };
    const auto sparse = [&](size_t c, uint64_t one_in) {
        auto &col = index.cols[c];
        for (size_t r = 0; r < rows; ++r) {
            if (rng.below(one_in) == 0) {
                col.set_rows.push_back(static_cast<monoprop::TermIndex>(r));
            }
        }
    };
    for (size_t c = 0; c < 6; ++c) {
        dense(c, 0); // density 1/2
    }
    sparse(6, 200);
    sparse(7, 997);
    // Rows straddling every fold-block edge, so a block boundary splits a sparse column's row list.
    {
        auto &col = index.cols[8];
        const size_t block_rows = monoprop::detail::kColumnBlockWords * 64;
        for (size_t edge = block_rows; edge < rows; edge += block_rows) {
            for (size_t r = edge - 3; r < std::min(rows, edge + 3); ++r) {
                col.set_rows.push_back(static_cast<monoprop::TermIndex>(r));
            }
        }
        if (rows != 0) {
            col.set_rows.push_back(static_cast<monoprop::TermIndex>(rows - 1)); // the global tail row
        }
        std::ranges::sort(col.set_rows);
        col.set_rows.erase(std::ranges::unique(col.set_rows).begin(), col.set_rows.end());
    }
    dense(9, 4); // dense representation, density 1/32
    // Columns 10 and 11 stay empty.
    sparse(12, 64);
    dense(13, 1); // density 1/4
    sparse(14, 5000);
    dense(15, 0);
    return index;
}

// Append rows [index.rows(), new_rows) the way operator growth does: new bits for dense columns (including
// the old tail word's unused bits), ascending new rows for sparse ones. Existing rows keep their columns.
// Does not touch row_parity_; the caller decides whether to drop or keep it.
inline auto grow_synthetic_index(monoprop::detail::InvertedIndex<kIndexModes> &index, size_t new_rows, uint64_t seed)
    -> void {
    const size_t old_rows = index.row_count;
    index.row_count = new_rows;
    const size_t words = index.words();
    SplitMix rng{seed};
    for (auto &col : index.cols) {
        if (col.is_dense) {
            col.words.resize(words, 0);
            for (size_t r = old_rows; r < new_rows; ++r) {
                if (rng.below(3) == 0) {
                    col.words[r >> 6] |= uint64_t{1} << (r & 63U);
                }
            }
        }
        else if (!col.set_rows.empty()) {
            for (size_t r = old_rows; r < new_rows; ++r) {
                if (rng.below(150) == 0) {
                    col.set_rows.push_back(static_cast<monoprop::TermIndex>(r));
                }
            }
        }
    }
}

// A Majorana/Pauli generator over the synthetic index's 16 columns.
inline auto generator(std::initializer_list<size_t> columns) -> monoprop::Monomial<kIndexModes> {
    monoprop::Monomial<kIndexModes> gen{};
    for (const auto c : columns) {
        gen.set(c);
    }
    return gen;
}

// Independent add-owner count: every slot a fused contract writes (sources and targets of full
// rotations, local slots of half rotations) mapped to how many records write it.
inline auto add_owner_counts(const monoprop::detail::FusedContract &fc) -> std::map<size_t, size_t> {
    std::map<size_t, size_t> owners;
    for (const auto &r : fc.hits) {
        ++owners[r.src];
        ++owners[r.tgt];
    }
    for (const auto &r : fc.inserts) {
        ++owners[r.src];
        ++owners[r.tgt];
    }
    for (const auto &h : fc.cross_half) {
        ++owners[h.local_idx];
    }
    return owners;
}

} // namespace kernel_test
