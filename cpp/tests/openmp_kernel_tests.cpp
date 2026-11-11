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

// The threaded cosine and fused-apply kernels: exact agreement with their one-thread execution, and
// kernel-specific worker participation. Participation is read from a test-only range observer that each
// kernel calls at the start of each of its own logical ranges (KernelTestSupport.h); a team created
// anywhere else does not count. Lazy-fold equivalence against the materialised-fold oracle lives in
// combined_recompute_equivalence.cpp; fused records from the real build path in fused_cos_sweep_tests.cpp.

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "KernelTestSupport.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/parallel/Options.h"

namespace {

using namespace monoprop;
using kernel_test::RangeLog;
using kernel_test::RecordingObserver;
using monoprop::detail::KernelRange;
using monoprop::detail::parallel::Options;

constexpr auto kBudgets = {1, 2, 3, 4};

// Distinct, non-degenerate values, so a missed, extra or doubled scaling shows up bitwise.
auto distinct_values(size_t n) -> VecD {
    VecD v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0 + static_cast<double>(i) * 1e-3;
    }
    return v;
}

auto bitwise_equal(const VecD &a, const VecD &b) -> bool {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// Equal up to the rounding of one multiply-add whose terms are bounded by |original| + 1 (|cos|, |sin| and every
// snapshot value are at most one here): the difference a separately compiled fused multiply-add can make.
auto equal_up_to_contraction(const VecD &a, const VecD &b, const VecD &original) -> bool {
    if (a.size() != b.size() || a.size() != original.size()) {
        return false;
    }
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 2 * eps * (std::abs(original[i]) + 1.0)) {
            return false;
        }
    }
    return true;
}

// The plan's mask: 4097 words, alternating bits, and only bit zero in the final word.
auto plan_mask() -> CosMask {
    CosMask mask;
    for (size_t w = 0; w < 4097; ++w) {
        const auto bits = w == 4096 ? uint64_t{1} : uint64_t{0x5555555555555555};
        mask.blocks.emplace_back(w * 64, bits);
        mask.total_count += static_cast<size_t>(std::popcount(bits));
    }
    return mask;
}

// Every row below `rows` selected, one block per word, the last word partial when rows % 64 != 0.
auto full_mask(size_t rows) -> CosMask {
    CosMask mask;
    for (size_t base = 0; base < rows; base += 64) {
        const size_t n = std::min<size_t>(64, rows - base);
        const uint64_t bits = n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
        mask.blocks.emplace_back(base, bits);
        mask.total_count += n;
    }
    return mask;
}

// Selected indices of a mask, independent of the kernel under test.
auto selected(const CosMask &mask, size_t n) -> std::vector<char> {
    std::vector<char> sel(n, 0);
    for (const auto &[base, bits] : mask.blocks) {
        for (size_t t = 0; t < 64; ++t) {
            if ((bits >> t) & 1U) {
                sel[base + t] = 1;
            }
        }
    }
    return sel;
}

// Every budget against the one-thread kernel, and the one-thread kernel against a hand oracle that
// multiplies exactly the selected coefficients and leaves the rest untouched.
auto check_mask_budgets(const CosMask &mask, size_t n, const char *label) -> void {
    constexpr double cos_val = 0.625;
    const VecD original = distinct_values(n);
    VecD oracle = original;
    const auto sel = selected(mask, n);
    for (size_t i = 0; i < n; ++i) {
        if (sel[i] != 0) {
            oracle[i] = original[i] * cos_val;
        }
    }
    VecD serial = original;
    monoprop::detail::scale_cos_mask(serial.data(), mask, cos_val, Options{.threads = 1});
    BOOST_TEST_CONTEXT(label) {
        BOOST_TEST(bitwise_equal(serial, oracle));
        for (const int threads : kBudgets) {
            VecD threaded = original;
            monoprop::detail::scale_cos_mask(threaded.data(), mask, cos_val, Options{.threads = threads});
            BOOST_TEST_INFO("threads " << threads);
            BOOST_TEST(bitwise_equal(threaded, serial));
        }
        // More workers than ranges.
        VecD many = original;
        monoprop::detail::scale_cos_mask(many.data(), mask, cos_val, Options{.threads = 64});
        BOOST_TEST(bitwise_equal(many, serial));
    }
}

// --- hand-built fused records -------------------------------------------------------------------------------

// A contract whose destinations are a shuffled permutation of distinct slots: hits and inserts own two slots
// each, halves one. The two-pass cosine mask covers every destination plus a padding region no record
// touches (anticommuting terms whose rotation produced no record), so it spans several mask ranges; a few
// slots are in neither.
struct HandContract {
    monoprop::detail::FusedContract fc;
    CosMask cos;
    size_t slots = 0;
};

auto hand_contract(size_t n_hits, size_t n_inserts, size_t n_halves, uint64_t seed) -> HandContract {
    HandContract h;
    const size_t record_slots = 2 * (n_hits + n_inserts) + n_halves + 97;
    constexpr size_t pad_words = 2100;
    h.slots = record_slots + pad_words * 64;
    std::vector<size_t> perm(record_slots);
    for (size_t i = 0; i < record_slots; ++i) {
        perm[i] = i;
    }
    kernel_test::SplitMix rng{seed};
    for (size_t i = record_slots - 1; i > 0; --i) {
        std::swap(perm[i], perm[rng.below(i + 1)]);
    }
    size_t next = 0;
    const auto value = [&rng] { return static_cast<double>(rng.below(1U << 20)) / (1U << 19) - 1.0; };
    const auto phase = [&rng] { return rng.below(2) == 0 ? int32_t{1} : int32_t{-1}; };
    for (size_t k = 0; k < n_hits; ++k) {
        h.fc.hits.push_back(
            {.src = perm[next], .tgt = perm[next + 1], .v_src = value(), .v_tgt = value(), .phase = phase()});
        next += 2;
    }
    for (size_t k = 0; k < n_inserts; ++k) {
        // v_tgt stays zero: Schrödinger gathers it in the apply, Heisenberg inserts are born at zero.
        h.fc.inserts.push_back({.src = perm[next], .tgt = perm[next + 1], .v_src = value(), .phase = phase()});
        next += 2;
    }
    for (size_t k = 0; k < n_halves; ++k) {
        h.fc.cross_half.push_back(
            {.local_idx = perm[next], .v_partner = value(), .phase_signed = phase(), .is_insert = rng.below(3) == 0});
        ++next;
    }
    std::vector<size_t> selected_slots(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(next));
    for (size_t s = record_slots; s < h.slots; s += 2) {
        selected_slots.push_back(s);
    }
    std::ranges::sort(selected_slots);
    CosineWordBuilder builder;
    for (const auto s : selected_slots) {
        builder.push_index(s);
    }
    h.cos = builder.finish();
    return h;
}

// Read at run time, so the oracle and the kernel both take std::cos/std::sin from the math library; a
// compile-time constant could be folded with a differently rounded result.
auto rotation_angle() -> double {
    static volatile double angle = 0.37;
    return angle;
}

// Independent oracle: the pre-threading serial apply_fused_contract, verbatim in its phase order, record order
// and arithmetic, over a copy of the contract.
auto reference_apply(const HandContract &h, const VecD &coeffs, bool schrodinger, bool fused_scale) -> VecD {
    auto fc = h.fc;
    VecD out = coeffs;
    const double param = rotation_angle();
    if (schrodinger) {
        for (auto &r : fc.inserts) {
            r.v_tgt = out[r.tgt];
        }
    }
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);
    double *const c = out.data();
    if (!fused_scale) {
        for (const auto &[base, bits] : h.cos.blocks) {
            for (size_t t = 0; t < 64; ++t) {
                if ((bits >> t) & 1U) {
                    c[base + t] *= cos_val;
                }
            }
        }
    }
    const size_t n_hit = fc.hits.size();
    for (size_t k = 0; k < n_hit + fc.inserts.size(); ++k) {
        const bool is_insert = k >= n_hit;
        const auto &r = is_insert ? fc.inserts[k - n_hit] : fc.hits[k];
        c[r.src] += sin_val * static_cast<double>(-r.phase) * r.v_tgt;
        if (fused_scale && is_insert) {
            c[r.tgt] = cos_val * c[r.tgt] + sin_val * static_cast<double>(r.phase) * r.v_src;
        }
        else {
            c[r.tgt] += sin_val * static_cast<double>(r.phase) * r.v_src;
        }
    }
    for (const auto &half : fc.cross_half) {
        if (fused_scale && half.is_insert) {
            c[half.local_idx] =
                cos_val * c[half.local_idx] + sin_val * static_cast<double>(half.phase_signed) * half.v_partner;
        }
        else {
            c[half.local_idx] += sin_val * static_cast<double>(half.phase_signed) * half.v_partner;
        }
    }
    return out;
}

auto apply_copy(const HandContract &h,
                const VecD &coeffs,
                bool schrodinger,
                bool fused_scale,
                int threads,
                const RecordingObserver *observer = nullptr) -> VecD {
    auto fc = h.fc;
    VecD out = coeffs;
    const double param = rotation_angle();
    if (observer != nullptr) {
        monoprop::detail::apply_fused_contract(fc, out, h.cos, param, schrodinger, fused_scale, {threads}, *observer);
    }
    else {
        monoprop::detail::apply_fused_contract(fc, out, h.cos, param, schrodinger, fused_scale, {threads});
    }
    return out;
}

} // namespace

// --- stored-mask cosine ------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_cos_mask_exact) {
    const CosMask mask = plan_mask();
    VecD s(4097 * 64, 1.25), p = s;
    monoprop::detail::scale_cos_mask(s.data(), mask, 0.625, {.threads = 1});
    monoprop::detail::scale_cos_mask(p.data(), mask, 0.625, {.threads = 3});
    BOOST_CHECK_EQUAL_COLLECTIONS(s.begin(), s.end(), p.begin(), p.end());
    check_mask_budgets(mask, 4097 * 64, "plan mask, 4097 words");
}

BOOST_AUTO_TEST_CASE(openmp_cos_mask_edge_cases) {
    check_mask_budgets(CosMask{}, 128, "empty mask");
    for (const size_t tail : {63, 64, 65}) {
        const size_t rows = 2048 * 64 + tail;
        BOOST_TEST_CONTEXT("tail " << tail) {
            check_mask_budgets(full_mask(rows), rows, "full mask with a partial tail");
        }
    }
    // Exactly one range (serial whatever the budget), and one block over it (two ranges).
    check_mask_budgets(full_mask(1024 * 64), 1024 * 64, "1024 blocks");
    check_mask_budgets(full_mask(1024 * 64 + 1), 1024 * 64 + 1, "1025 blocks");
    // Sparse blocks: bases need not be consecutive, only ascending and distinct.
    CosMask sparse;
    for (size_t w = 0; w < 5000; w += 3) {
        sparse.blocks.emplace_back(w * 64, uint64_t{0x8000000000000001});
        sparse.total_count += 2;
    }
    check_mask_budgets(sparse, 5000 * 64, "sparse blocks");
}

BOOST_AUTO_TEST_CASE(openmp_cos_mask_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    const CosMask mask = plan_mask();
    VecD c(4097 * 64, 1.0);
    RangeLog log;
    monoprop::detail::scale_cos_mask(c.data(), mask, 0.5, Options{.threads = 4}, RecordingObserver{&log});
    kernel_test::check_participation(log[KernelRange::cos_mask], 5, "scale_cos_mask budget 4");
}

BOOST_AUTO_TEST_CASE(openmp_cos_mask_small_or_serial_work_stays_on_the_caller) {
    VecD c(4097 * 64, 1.0);
    RangeLog one_thread;
    monoprop::detail::scale_cos_mask(c.data(), plan_mask(), 0.5, Options{.threads = 1}, RecordingObserver{&one_thread});
    kernel_test::check_serial(one_thread[KernelRange::cos_mask], 5, "budget 1");
    RangeLog tiny;
    monoprop::detail::scale_cos_mask(c.data(),
                                     full_mask(1024 * 64),
                                     0.5,
                                     Options{.threads = 4},
                                     RecordingObserver{&tiny});
    kernel_test::check_serial(tiny[KernelRange::cos_mask], 1, "one range, budget 4");
    RangeLog empty;
    monoprop::detail::scale_cos_mask(c.data(), CosMask{}, 0.5, Options{.threads = 4}, RecordingObserver{&empty});
    kernel_test::check_serial(empty[KernelRange::cos_mask], 0, "empty mask, budget 4");
}

// --- lazy cosine ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_cos_lazy_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    // Three full fold blocks plus a partial fourth.
    const size_t rows = 3 * monoprop::detail::kColumnBlockWords * 64 + 1000;
    const auto index = kernel_test::make_synthetic_index(rows, 11);
    const auto gen = kernel_test::generator({0, 3, 7}); // odd: exercises the caller-prepared row parity
    const auto recipe = monoprop::detail::make_lazy_fold<kernel_test::kIndexModes>(index, gen, rows, Basis::Majorana);
    VecD c = distinct_values(rows);
    RangeLog log;
    monoprop::detail::scale_cos_lazy<kernel_test::kIndexModes>(index,
                                                               recipe,
                                                               c.data(),
                                                               0.5,
                                                               Options{.threads = 4},
                                                               RecordingObserver{&log});
    kernel_test::check_participation(log[KernelRange::cos_lazy], 4, "scale_cos_lazy budget 4");
}

BOOST_AUTO_TEST_CASE(openmp_cos_lazy_small_or_serial_work_stays_on_the_caller) {
    const size_t rows = 3 * monoprop::detail::kColumnBlockWords * 64 + 1000;
    const auto index = kernel_test::make_synthetic_index(rows, 12);
    const auto gen = kernel_test::generator({1, 4});
    const auto recipe = monoprop::detail::make_lazy_fold<kernel_test::kIndexModes>(index, gen, rows, Basis::Majorana);
    VecD c = distinct_values(rows);
    RangeLog one_thread;
    monoprop::detail::scale_cos_lazy<kernel_test::kIndexModes>(index,
                                                               recipe,
                                                               c.data(),
                                                               0.5,
                                                               Options{.threads = 1},
                                                               RecordingObserver{&one_thread});
    kernel_test::check_serial(one_thread[KernelRange::cos_lazy], 4, "budget 1");
    // A partial scaled_count inside the first block: one range, serial at any budget.
    const auto short_recipe =
        monoprop::detail::make_lazy_fold<kernel_test::kIndexModes>(index, gen, 1000, Basis::Majorana);
    RangeLog tiny;
    monoprop::detail::scale_cos_lazy<kernel_test::kIndexModes>(index,
                                                               short_recipe,
                                                               c.data(),
                                                               0.5,
                                                               Options{.threads = 4},
                                                               RecordingObserver{&tiny});
    kernel_test::check_serial(tiny[KernelRange::cos_lazy], 1, "one range, budget 4");
}

// --- fused apply ------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_fused_apply_exact_on_disjoint_records) {
    // Several record ranges in every stream, with partial final ranges.
    const auto h = hand_contract(3 * 1024 + 17, 2 * 1024 + 5, 1500, 7);
    const auto owners = kernel_test::add_owner_counts(h.fc);
    BOOST_TEST(std::ranges::all_of(owners, [](const auto &kv) { return kv.second == 1; }));
    const VecD coeffs = distinct_values(h.slots);
    for (const bool schrodinger : {false, true}) {
        for (const bool fused_scale : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " fused_scale=" << fused_scale) {
                const VecD serial = apply_copy(h, coeffs, schrodinger, fused_scale, 1);
                BOOST_TEST(!bitwise_equal(serial, coeffs));
                // The oracle is compiled separately, so where a slot is `cos·c + sin·φ·v` (fused-scale insert
                // arms) the compiler may contract a different product into a fused multiply-add. Every other
                // arm is a single multiply-add and must match bitwise.
                const VecD reference = reference_apply(h, coeffs, schrodinger, fused_scale);
                if (fused_scale) {
                    BOOST_TEST(equal_up_to_contraction(serial, reference, coeffs));
                }
                else {
                    BOOST_TEST(bitwise_equal(serial, reference));
                }
                for (const int threads : {2, 3, 4, 64}) {
                    BOOST_TEST_INFO("threads " << threads);
                    BOOST_TEST(bitwise_equal(apply_copy(h, coeffs, schrodinger, fused_scale, threads), serial));
                }
                // Slots no record or mask touches keep their values.
                const auto sel = selected(h.cos, h.slots);
                for (size_t i = 0; i < h.slots; ++i) {
                    if (sel[i] == 0 && !owners.contains(i)) {
                        BOOST_TEST(serial[i] == coeffs[i]);
                    }
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_fused_apply_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    const auto h = hand_contract(3 * 1024 + 17, 2 * 1024 + 5, 1500, 8);
    const VecD coeffs = distinct_values(h.slots);
    RangeLog log;
    const RecordingObserver observer{&log};
    (void)apply_copy(h, coeffs, /*schrodinger=*/true, /*fused_scale=*/false, 4, &observer);
    const size_t records = h.fc.hits.size() + h.fc.inserts.size() + h.fc.cross_half.size();
    kernel_test::check_participation(log[KernelRange::fused_gather], 3, "insert gather budget 4");
    kernel_test::check_participation(log[KernelRange::cos_mask],
                                     monoprop::detail::logical_ranges(h.cos.blocks.size(), 1024),
                                     "two-pass cosine phase budget 4");
    kernel_test::check_participation(log[KernelRange::fused_apply],
                                     monoprop::detail::logical_ranges(records, 1024),
                                     "rotation records budget 4");
}

BOOST_AUTO_TEST_CASE(openmp_fused_apply_small_or_serial_work_stays_on_the_caller) {
    const auto h = hand_contract(3 * 1024 + 17, 2 * 1024 + 5, 1500, 9);
    const VecD coeffs = distinct_values(h.slots);
    RangeLog one_thread;
    const RecordingObserver serial_observer{&one_thread};
    (void)apply_copy(h, coeffs, true, false, 1, &serial_observer);
    const size_t records = h.fc.hits.size() + h.fc.inserts.size() + h.fc.cross_half.size();
    kernel_test::check_serial(one_thread[KernelRange::fused_gather], 3, "gather budget 1");
    kernel_test::check_serial(one_thread[KernelRange::fused_apply],
                              monoprop::detail::logical_ranges(records, 1024),
                              "records budget 1");

    // Fused-scale Heisenberg: no gather and no cosine phase at all.
    RangeLog heisenberg;
    const RecordingObserver heisenberg_observer{&heisenberg};
    (void)apply_copy(h, coeffs, false, true, 4, &heisenberg_observer);
    BOOST_TEST(heisenberg[KernelRange::fused_gather].prepares == 0u);
    BOOST_TEST(heisenberg[KernelRange::cos_mask].prepares == 0u);

    const auto tiny = hand_contract(300, 200, 100, 10);
    const VecD tiny_coeffs = distinct_values(tiny.slots);
    RangeLog small;
    const RecordingObserver small_observer{&small};
    (void)apply_copy(tiny, tiny_coeffs, true, false, 4, &small_observer);
    kernel_test::check_serial(small[KernelRange::fused_gather], 1, "tiny gather budget 4");
    kernel_test::check_serial(small[KernelRange::fused_apply], 1, "tiny records budget 4");

    HandContract empty;
    empty.slots = 16;
    RangeLog none;
    const RecordingObserver none_observer{&none};
    const VecD untouched = apply_copy(empty, distinct_values(16), true, false, 4, &none_observer);
    BOOST_TEST(bitwise_equal(untouched, distinct_values(16)));
    kernel_test::check_serial(none[KernelRange::fused_apply], 0, "empty contract budget 4");
}

// --- worker failures inside the kernels' own ranges --------------------------------------------------------

// A worker failure inside a kernel range is captured, every worker joins, and the original exception reaches
// the caller; no range runs twice. Which other ranges ran is unspecified.
BOOST_AUTO_TEST_CASE(openmp_kernel_worker_failures_join_before_rethrow) {
    const auto check = [](const RangeLog &log, KernelRange kind, const char *label) {
        const auto &s = log.slots[static_cast<size_t>(kind)];
        BOOST_TEST_CONTEXT(label) {
            BOOST_TEST(s.visits.at(2) == 1);
            BOOST_TEST(std::ranges::all_of(s.visits, [](int v) { return v <= 1; }));
        }
    };
    for (const int threads : {1, 4}) {
        BOOST_TEST_CONTEXT("threads " << threads) {
            VecD c(4097 * 64, 1.0);
            RangeLog mask_log;
            BOOST_CHECK_THROW(monoprop::detail::scale_cos_mask(c.data(),
                                                               plan_mask(),
                                                               0.5,
                                                               Options{.threads = threads},
                                                               RecordingObserver{&mask_log, KernelRange::cos_mask, 2}),
                              std::runtime_error);
            check(mask_log, KernelRange::cos_mask, "scale_cos_mask");

            const size_t rows = 3 * monoprop::detail::kColumnBlockWords * 64 + 1000;
            const auto index = kernel_test::make_synthetic_index(rows, 13);
            const auto recipe =
                monoprop::detail::make_lazy_fold<kernel_test::kIndexModes>(index,
                                                                           kernel_test::generator({0, 3, 7}),
                                                                           rows,
                                                                           Basis::Majorana);
            VecD lazy = distinct_values(rows);
            RangeLog lazy_log;
            BOOST_CHECK_THROW(monoprop::detail::scale_cos_lazy<kernel_test::kIndexModes>(
                                  index,
                                  recipe,
                                  lazy.data(),
                                  0.5,
                                  Options{.threads = threads},
                                  RecordingObserver{&lazy_log, KernelRange::cos_lazy, 2}),
                              std::runtime_error);
            check(lazy_log, KernelRange::cos_lazy, "scale_cos_lazy");

            const auto h = hand_contract(3 * 1024 + 17, 2 * 1024 + 5, 1500, 14);
            for (const auto kind : {KernelRange::fused_gather, KernelRange::cos_mask, KernelRange::fused_apply}) {
                RangeLog fused_log;
                const RecordingObserver observer{&fused_log, kind, 2};
                BOOST_CHECK_THROW((void)apply_copy(h, distinct_values(h.slots), true, false, threads, &observer),
                                  std::runtime_error);
                check(fused_log, kind, "apply_fused_contract");
            }
        }
    }
}

// The debug-build owner check agrees with the independent count, and rejects a slot written twice: a hit's
// target reused as a half's local slot, and a source shared by two inserts.
BOOST_AUTO_TEST_CASE(openmp_fused_add_owner_check_detects_a_shared_slot) {
    auto h = hand_contract(3 * 1024 + 17, 2 * 1024 + 5, 1500, 15);
    BOOST_TEST(monoprop::detail::fused_add_owners_unique(h.fc));
    BOOST_TEST(monoprop::detail::fused_add_owners_unique(monoprop::detail::FusedContract{}));
    auto shared_half = h.fc;
    shared_half.cross_half.back().local_idx = shared_half.hits.front().tgt;
    BOOST_TEST(!monoprop::detail::fused_add_owners_unique(shared_half));
    auto shared_insert = h.fc;
    shared_insert.inserts.back().src = shared_insert.inserts.front().src;
    BOOST_TEST(!monoprop::detail::fused_add_owners_unique(shared_insert));
}
