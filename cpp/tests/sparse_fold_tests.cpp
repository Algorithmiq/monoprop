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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/operator/InvertedIndex.h"

// The fold matches a full-width oracle bit for bit, in pass 1 and in the graph-replay kernels.
// Column tiers are set by hand to cover mixes the density rule would not produce, and column densities
// send blocks down both the posting-driven and the word-sweeping kernel.

using namespace monoprop;
using namespace monoprop::detail;

namespace {

constexpr size_t N = 32; // 64 columns
using Sc = InvertedIndex<N>;
constexpr size_t kBlockRows = kColumnBlockWords * 64;

// Column roles shared by every fixture.
enum : size_t {
    kSparseA = 0,      // random 1/300 plus postings on word and block boundaries
    kSparseB = 1,      // random 1/300, sharing some boundary rows with kSparseA (partial XOR cancel)
    kEmpty = 2,        // sparse, no postings
    kConcentrated = 3, // sparse, every 50th row of block 1 only: more postings than block words
    kDenseA = 4,       // dense tier, ~1/4
    kDenseB = 5,       // dense tier, ~1/3
    kCopyOfA = 6,      // sparse, identical to kSparseA: folding both cancels every word
    kSparseC = 7,      // sparse but ~1/100, above kSparseFoldPostingsPerWord: sweeps words
};

struct Fixture {
    Sc sc;
    std::vector<std::vector<uint64_t>> member; // per column, full-height bitmap of the authored rows
};

auto set_column(Fixture &f, size_t c, std::vector<size_t> rows, bool dense) -> void {
    std::ranges::sort(rows);
    const auto [first, last] = std::ranges::unique(rows);
    rows.erase(first, last);
    std::erase_if(rows, [&f](size_t r) { return r >= f.sc.row_count; });
    auto &bits = f.member[c];
    bits.assign(f.sc.words(), 0);
    for (const size_t r : rows) {
        bits[r >> 6] |= uint64_t{1} << (r & 63U);
    }
    auto &col = f.sc.cols[c];
    col.is_dense = dense;
    col.words.clear();
    col.set_rows.clear();
    if (dense) {
        col.words = bits;
    }
    else {
        for (const size_t r : rows) {
            col.set_rows.push_back(static_cast<TermIndex>(r));
        }
    }
}

auto random_rows(size_t row_count, size_t one_in, std::mt19937_64 &rng) -> std::vector<size_t> {
    std::vector<size_t> rows;
    for (size_t r = 0; r < row_count; ++r) {
        if (rng() % one_in == 0) {
            rows.push_back(r);
        }
    }
    return rows;
}

auto make_fixture(size_t row_count, uint64_t seed) -> Fixture {
    Fixture f;
    f.sc.row_count = row_count;
    f.member.resize(Sc::kNumColumns);
    for (size_t c = 0; c < Sc::kNumColumns; ++c) {
        set_column(f, c, {}, false);
    }
    std::mt19937_64 rng(seed);
    const std::vector<size_t> boundaries{0,
                                         63,
                                         64,
                                         kBlockRows - 1,
                                         kBlockRows,
                                         2 * kBlockRows - 64,
                                         2 * kBlockRows,
                                         3 * kBlockRows,
                                         row_count - 1};
    auto a = random_rows(row_count, 300, rng);
    a.insert(a.end(), boundaries.begin(), boundaries.end());
    auto b = random_rows(row_count, 300, rng);
    for (const size_t r : {size_t{63}, kBlockRows, 3 * kBlockRows, row_count - 1}) {
        b.push_back(r);
    }
    std::vector<size_t> concentrated;
    for (size_t r = kBlockRows; r < 2 * kBlockRows; r += 50) {
        concentrated.push_back(r);
    }
    set_column(f, kSparseA, a, false);
    set_column(f, kSparseB, b, false);
    set_column(f, kConcentrated, concentrated, false);
    set_column(f, kDenseA, random_rows(row_count, 4, rng), true);
    set_column(f, kDenseB, random_rows(row_count, 3, rng), true);
    set_column(f, kCopyOfA, a, false);
    set_column(f, kSparseC, random_rows(row_count, 100, rng), false);
    return f;
}

// Fold sets: all sparse, mixed, all dense, and empty.
const std::vector<std::vector<size_t>> kFolds{
    {kSparseA},
    {kSparseA, kSparseB},
    {kSparseA, kSparseB, kEmpty},
    {kEmpty},
    {kSparseA, kCopyOfA},
    {kSparseA, kConcentrated},
    {kSparseA, kSparseC},
    {kSparseA, kDenseA},
    {kDenseA, kDenseB},
    {kDenseA, kSparseA, kSparseB},
    {},
};
const std::vector<size_t> kPivots{kSparseA, kSparseB, kConcentrated, kDenseA, kEmpty};

auto oracle_word(const Fixture &f, std::span<const size_t> fold, size_t wi) -> uint64_t {
    uint64_t bits = 0;
    for (const size_t c : fold) {
        bits ^= f.member[c][wi];
    }
    return bits;
}

auto last_mask_for(size_t count) -> uint64_t {
    return (count % 64 == 0) ? ~uint64_t{0} : ((uint64_t{1} << (count % 64)) - 1);
}

auto zero_scratch_is_clean(const ZeroBlockScratch &zs) -> bool {
    return std::ranges::all_of(zs.words, [](uint64_t w) { return w == 0; })
           && std::ranges::all_of(zs.summary, [](uint64_t w) { return w == 0; });
}

auto same_nz(const std::vector<EvenParityNzWord> &a, const std::vector<EvenParityNzWord> &b) -> bool {
    return std::ranges::equal(a, b, [](const EvenParityNzWord &x, const EvenParityNzWord &y) {
        return x.base == y.base && x.overlap == y.overlap && x.foll == y.foll;
    });
}

// Pass 1 over [wlo, whi) on every fold, pivot and parity setting, against the oracle.
auto check_pass1(const Fixture &f, size_t wlo, size_t whi, size_t last_word, uint64_t last_mask) -> void {
    std::mt19937_64 rng(7);
    std::vector<uint64_t> parity(f.sc.words());
    for (auto &w : parity) {
        w = rng();
    }
    for (size_t fi = 0; fi < kFolds.size(); ++fi) {
        const std::span<const size_t> fold(kFolds[fi]);
        for (const size_t pivot : kPivots) {
            for (const bool g_odd : {false, true}) {
                std::vector<EvenParityNzWord> expected;
                size_t expected_anti = 0;
                size_t expected_foll = 0;
                for (size_t wi = wlo; wi < whi; ++wi) {
                    uint64_t bits = oracle_word(f, fold, wi) ^ (g_odd ? parity[wi] : 0);
                    if (wi == last_word) {
                        bits &= last_mask;
                    }
                    if (bits != 0) {
                        const uint64_t foll = bits & f.member[pivot][wi];
                        expected.push_back({wi * 64, bits, foll});
                        expected_anti += static_cast<size_t>(std::popcount(bits));
                        expected_foll += static_cast<size_t>(std::popcount(foll));
                    }
                }
                BOOST_TEST_CONTEXT("fold " << fi << " pivot " << pivot << " g_odd " << g_odd) {
                    std::vector<EvenParityNzWord> nz{{1, 1, 1}}; // pre-dirtied: pass 1 must clear it
                    size_t n_anti = 99;
                    size_t n_foll = 99;
                    even_parity_scan_pass1<
                        N>(f.sc, fold, pivot, wlo, whi, last_word, last_mask, g_odd, parity.data(), nz, n_anti, n_foll);
                    BOOST_TEST(same_nz(nz, expected));
                    BOOST_TEST(n_anti == expected_anti);
                    BOOST_TEST(n_foll == expected_foll);
                    BOOST_TEST(zero_scratch_is_clean(fold_zero_scratch()));
                    BOOST_TEST(zero_scratch_is_clean(pivot_zero_scratch()));
                }
            }
        }
    }
}

} // namespace

// Four blocks and a partial last word.
BOOST_AUTO_TEST_CASE(sparse_fold_pass1_matches_oracle_partial_last_word) {
    const size_t rows = 3 * kBlockRows + 777;
    const auto f = make_fixture(rows, 1);
    const size_t words = f.sc.words();
    check_pass1(f, 0, words, words - 1, last_mask_for(rows));
}

// Whole words and whole blocks.
BOOST_AUTO_TEST_CASE(sparse_fold_pass1_matches_oracle_word_aligned) {
    const size_t rows = 4 * kBlockRows;
    const auto f = make_fixture(rows, 2);
    const size_t words = f.sc.words();
    check_pass1(f, 0, words, words - 1, last_mask_for(rows));
}

// One partial word in one partial block.
BOOST_AUTO_TEST_CASE(sparse_fold_pass1_matches_oracle_tiny) {
    const auto f = make_fixture(100, 3);
    check_pass1(f, 0, f.sc.words(), 0, last_mask_for(100));
}

// A mid-block sub-range: posting cursors must start at wlo.
BOOST_AUTO_TEST_CASE(sparse_fold_pass1_matches_oracle_sub_range) {
    const auto f = make_fixture(3 * kBlockRows + 777, 4);
    check_pass1(f, 1500, 2100, 2099, last_mask_for(37));
}

// A throwing visitor leaves the scratch zeroed. kSparseA alone is well under kSparseFoldPostingsPerWord,
// so the posting-driven kernel runs.
BOOST_AUTO_TEST_CASE(sparse_fold_throwing_visitor_keeps_scratch_zero) {
    const auto f = make_fixture(2 * kBlockRows, 5);
    const std::vector<size_t> fold{kSparseA};
    const size_t words = f.sc.words();
    BOOST_CHECK_THROW(for_each_fold_word<N>(
                          f.sc,
                          fold,
                          0,
                          words,
                          words - 1,
                          ~uint64_t{0},
                          nullptr,
                          [](size_t, uint64_t) { throw std::runtime_error("visitor"); },
                          [](size_t, size_t) {}),
                      std::runtime_error);
    BOOST_TEST(zero_scratch_is_clean(fold_zero_scratch()));
}

// Graph-replay kernels truncated at scaled_count match the oracle.
BOOST_AUTO_TEST_CASE(sparse_fold_replay_kernels_match_oracle) {
    const size_t rows = 3 * kBlockRows + 777;
    const auto f = make_fixture(rows, 6);
    // Full, mid-block, mid-word, on block and word boundaries, under one word, and empty.
    const std::vector<size_t> scaled_counts{rows,
                                            kBlockRows + 1024 * 32 + 37,
                                            2 * kBlockRows,
                                            2 * kBlockRows + 64 * 5,
                                            37,
                                            0};
    std::vector<double> state0(rows);
    std::vector<double> ham0(rows);
    for (size_t i = 0; i < rows; ++i) {
        state0[i] = 1.0 + static_cast<double>(i) * 1e-3;
        ham0[i] = 0.5 - static_cast<double>(i % 977) * 1e-4;
    }
    const double cos_val = 0.6234;
    const double sec_val = 1.0 / cos_val;
    // Odd-|G| layers XOR the index's own row parity, which this fixture's columns define.
    const uint64_t *const parity = f.sc.row_parity_words();

    for (size_t fi = 0; fi < kFolds.size(); ++fi) {
        for (const size_t scaled_count : scaled_counts) {
            for (const bool g_odd : {false, true}) {
                LazyFold<N> r;
                r.columns = kFolds[fi];
                r.fold.g_odd = g_odd;
                r.fold.mask_words = std::min(f.sc.words(), (scaled_count + 63) / 64);
                r.fold.last_word = r.fold.mask_words == 0 ? 0 : r.fold.mask_words - 1;
                r.fold.last_mask = last_mask_for(scaled_count);

                std::vector<TermIndex> expected_idx;
                for (size_t wi = 0; wi < r.fold.mask_words; ++wi) {
                    uint64_t bits = oracle_word(f, r.columns, wi) ^ (g_odd ? parity[wi] : 0);
                    if (wi == r.fold.last_word) {
                        bits &= r.fold.last_mask;
                    }
                    for_each_cos_index(wi * 64, bits, [&](size_t i) {
                        expected_idx.push_back(static_cast<TermIndex>(i));
                    });
                }
                std::vector<double> expected_scaled = state0;
                std::vector<double> expected_state = state0;
                std::vector<double> expected_ham = ham0;
                double expected_loc = 0.0;
                for (const TermIndex i : expected_idx) {
                    expected_scaled[i] *= cos_val;
                    expected_loc += expected_state[i] * expected_ham[i];
                    expected_ham[i] *= sec_val;
                    expected_state[i] *= cos_val;
                }

                BOOST_TEST_CONTEXT("fold " << fi << " scaled_count " << scaled_count << " g_odd " << g_odd) {
                    std::vector<TermIndex> idx;
                    cos_indices_lazy<N>(f.sc, r, idx);
                    BOOST_TEST(idx == expected_idx, boost::test_tools::per_element());

                    std::vector<double> scaled = state0;
                    scale_cos_lazy<N>(f.sc, r, scaled.data(), cos_val);
                    BOOST_TEST(std::ranges::equal(scaled, expected_scaled, [](double x, double y) {
                        return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y);
                    }));

                    std::vector<double> state = state0;
                    std::vector<double> ham = ham0;
                    const double loc = accumulate_cos_lazy<N>(f.sc, r, state.data(), ham.data(), cos_val, sec_val);
                    // The oracle may fuse differently.
                    BOOST_TEST(std::abs(loc - expected_loc) <= 1e-12 * std::max(1.0, std::abs(expected_loc)));
                    BOOST_TEST(std::ranges::equal(state, expected_state, [](double x, double y) {
                        return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y);
                    }));
                    BOOST_TEST(std::ranges::equal(ham, expected_ham, [](double x, double y) {
                        return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y);
                    }));
                    BOOST_TEST(zero_scratch_is_clean(fold_zero_scratch()));
                }
            }
        }
    }
}
