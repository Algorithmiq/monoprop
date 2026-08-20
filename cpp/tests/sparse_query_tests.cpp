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

// Differential against the frozen dense oracle in dense_query_reference.h; cases chosen, not sampled.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/QueryCodec.h"
#include "monoprop/detail/evolution/layer_build/SparseQuery.h"

#include "dense_query_reference.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

template <size_t N>
auto differential(const std::vector<uint16_t> &pos, int phase) -> size_t {
    using SQ = SparseQuery<N>;
    const size_t k = pos.size();

    Monomial<N> want;
    for (const auto p : pos) {
        want.set(p);
    }
    BOOST_REQUIRE_EQUAL(want.count(), k); // the caller must not hand us duplicates
    VecZ dbuf;
    test_ref::query_push<N>(dbuf, want, phase);
    Monomial<N> dmono;
    int dphase = 99;
    test_ref::query_read<N>(dbuf, 0, dmono, dphase);
    BOOST_REQUIRE((dmono == want));
    BOOST_REQUIRE_EQUAL(dphase, phase);

    VecZ sbuf;
    const size_t sw = SQ::push(sbuf, pos.data(), k, phase);
    BOOST_REQUIRE_EQUAL(sbuf.size(), sw);

    BOOST_TEST(SQ::words_at(sbuf, 0) == sw);
    BOOST_TEST(SQ::k_at(sbuf, 0) == k);
    BOOST_TEST(SQ::phase_at(sbuf, 0) == phase);

    std::vector<uint16_t> sout(k == 0 ? 1 : k);
    const size_t snext = SQ::read_positions(sbuf, 0, sout.data());
    BOOST_TEST(snext == sw);
    sout.resize(k);
    BOOST_TEST(sout == pos, boost::test_tools::per_element());

    Monomial<N> sm;
    int sp = 99;
    (void)SQ::read_mono(sbuf, 0, sm, sp);
    BOOST_TEST(sm.count() == k);
    BOOST_TEST((sm == dmono));
    BOOST_TEST(sp == dphase);

    VecZ mbuf;
    const size_t mw = SQ::push_mono(mbuf, want, phase);
    BOOST_TEST(mw == sw);
    BOOST_TEST(mbuf == sbuf, boost::test_tools::per_element());

    return sw;
}

auto strided(size_t k, size_t start, size_t step, size_t universe) -> std::vector<uint16_t> {
    std::vector<uint16_t> v;
    for (size_t j = 0; j < k; ++j) {
        const size_t p = start + j * step;
        if (p >= universe) {
            break;
        }
        v.push_back(static_cast<uint16_t>(p));
    }
    return v;
}

// Uniform draws are what can reach bitmap mode -- gap coding wins every regular pattern -- but see
// sparse_record_actually_exercises_its_bitmap_mode: at narrow widths they cannot reach it either.
auto scattered(size_t k, size_t universe, std::mt19937_64 &rng) -> std::vector<uint16_t> {
    std::vector<uint16_t> pool(universe);
    for (size_t j = 0; j < universe; ++j) {
        pool[j] = static_cast<uint16_t>(j);
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    pool.resize(std::min(k, universe));
    std::sort(pool.begin(), pool.end());
    return pool;
}

// The reference d, written against the definition (mode m owns bits 2m and 2m+1), not the codec.
auto reference_pair_count(const std::vector<uint16_t> &pos) -> size_t {
    size_t d = 0;
    for (size_t j = 0; j + 1 < pos.size(); ++j) {
        if ((pos[j] % 2 == 0) && (pos[j + 1] == pos[j] + 1)) {
            ++d;
        }
    }
    return d;
}

} // namespace

// Flat names: boostAddTests.cmake strips the indentation encoding suite nesting, so a suite-wrapped
// case errors at setup having asserted nothing.

BOOST_AUTO_TEST_CASE(sparse_record_agrees_with_the_dense_oracle_across_widths) {
    for (const int phase : {-1, 0, 1}) {
        for (const size_t k : {size_t{0}, size_t{1}, size_t{2}, size_t{5}, size_t{6}, size_t{7}, size_t{15}}) {
            differential<32>(strided(k, 0, 2, 64), phase);
            differential<128>(strided(k, 3, 7, 256), phase);
            differential<250>(strided(k, 11, 23, 500), phase);
            differential<512>(strided(k, 1, 41, 1024), phase);
            differential<1024>(strided(k, 5, 97, 2048), phase);
        }
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_handles_widths_that_are_not_whole_words) {
    // Widths with no whole word: 12 modes (LiH) is kBits=24, so the bitmap payload is a partial word.
    std::mt19937_64 rng(0xB17U);
    for (const int phase : {-1, 0, 1}) {
        for (const size_t k : {size_t{0}, size_t{1}, size_t{5}, size_t{9}, size_t{16}, size_t{24}}) {
            differential<12>(strided(k, 0, 1, 24), phase); // kBits=24: no whole word at all
            differential<12>(strided(k, 0, 2, 24), phase);
            differential<12>(scattered(k, 24, rng), phase);
        }
        for (const size_t k : {size_t{1}, size_t{7}, size_t{20}, size_t{40}, size_t{70}}) {
            differential<50>(strided(k, 0, 1, 100), phase);  // kBits=100: one whole word plus 36 bits
            differential<33>(strided(k, 0, 1, 66), phase);   // kBits=66: one whole word plus 2 bits
            differential<250>(strided(k, 0, 1, 500), phase); // kBits=500: seven words plus 52 bits
            differential<50>(scattered(k, 100, rng), phase);
            differential<33>(scattered(k, 66, rng), phase);
            differential<250>(scattered(k, 500, rng), phase);
        }
    }
    for (const size_t bits : {size_t{24}, size_t{100}, size_t{66}}) {
        std::vector<uint16_t> all(bits);
        for (size_t j = 0; j < bits; ++j) {
            all[j] = static_cast<uint16_t>(j);
        }
        if (bits == 24) {
            differential<12>(all, 1);
        }
        else if (bits == 100) {
            differential<50>(all, 1);
        }
        else {
            differential<33>(all, 1);
        }
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_actually_exercises_its_bitmap_mode) {
    // Two generators because uniform draws CANNOT reach bitmap at kBits=24: bitmap is one word there, so
    // gap must need two (19 + (k-1)*gw > 64), which only a dense run plus one far outlier forces.
    const auto count_bitmap = [](auto tag, size_t universe) {
        using SQ = SparseQuery<decltype(tag)::value>;
        size_t used = 0;
        size_t bad = 0;
        const auto tally = [&](const std::vector<uint16_t> &pos) {
            if (pos.size() < 2 || pos.size() > SQ::kMaxPositions) {
                return;
            }
            VecZ buf;
            (void)SQ::push(buf, pos.data(), pos.size(), 1);
            if (SQ::header_at(buf, 0).mode != SQ::kModeBitmap) {
                return;
            }
            ++used;
            std::vector<uint16_t> back(pos.size());
            SQ::read_positions(buf, 0, back.data());
            bad += static_cast<size_t>(back != pos || SQ::k_at(buf, 0) != pos.size());
        };
        std::mt19937_64 rng(0xB1747U ^ universe);
        for (size_t trial = 0; trial < 600; ++trial) {
            tally(scattered(1 + (rng() % universe), universe, rng));
        }
        for (size_t run = 1; run < universe; ++run) {
            for (size_t outlier = run; outlier < universe; ++outlier) {
                std::vector<uint16_t> pos;
                pos.reserve(run + 1);
                for (size_t j = 0; j < run; ++j) {
                    pos.push_back(static_cast<uint16_t>(j));
                }
                pos.push_back(static_cast<uint16_t>(outlier));
                tally(pos);
            }
        }
        return std::pair<size_t, size_t>{used, bad};
    };
    const auto narrow = count_bitmap(std::integral_constant<size_t, 12>{}, 24);
    const auto partial = count_bitmap(std::integral_constant<size_t, 250>{}, 500);
    const auto bucket = count_bitmap(std::integral_constant<size_t, 128>{}, 256);
    BOOST_TEST(narrow.first > 0U);
    BOOST_TEST(partial.first > 0U);
    BOOST_TEST(bucket.first > 0U);
    BOOST_TEST(narrow.second == 0U);
    BOOST_TEST(partial.second == 0U);
    BOOST_TEST(bucket.second == 0U);
}

BOOST_AUTO_TEST_CASE(sparse_record_survives_the_six_bit_k_escape) {
    for (const size_t k : {size_t{62}, size_t{63}, size_t{64}, size_t{200}}) {
        const auto pos = strided(k, 0, 3, 2048);
        BOOST_REQUIRE_EQUAL(pos.size(), k);
        differential<1024>(pos, 1);
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_bounds_the_fully_paired_term) {
    // Every bit set: 514 words in FIXED at NumModes=1024, one word in BITMAP. Hence the argmin.
    std::vector<uint16_t> all(2048);
    for (size_t j = 0; j < all.size(); ++j) {
        all[j] = static_cast<uint16_t>(j);
    }
    const size_t sw = differential<1024>(all, 1);
    BOOST_TEST(sw == 1U);
}

BOOST_AUTO_TEST_CASE(sparse_record_never_exceeds_the_dense_words_it_replaces) {
    const auto check = [](auto tag, size_t universe, size_t dense_words) {
        using SQ = SparseQuery<decltype(tag)::value>;
        std::mt19937_64 rng(0xC0FFEE ^ universe);
        for (size_t k = 0; k <= universe; k += std::max<size_t>(1, universe / 37)) {
            const auto pos = scattered(k, universe, rng);
            VecZ buf;
            const size_t w = SQ::push(buf, pos.data(), pos.size(), 1);
            BOOST_TEST(w <= dense_words + 1,
                       "k=" << k << " at U=" << universe << " took " << w << " words vs dense " << dense_words);
        }
    };
    check(std::integral_constant<size_t, 32>{}, 64, 2);
    check(std::integral_constant<size_t, 128>{}, 256, 5);
    check(std::integral_constant<size_t, 250>{}, 500, 9);
}

BOOST_AUTO_TEST_CASE(sparse_record_walks_a_multi_query_buffer_exactly) {
    // Mixed width and mixed mode, which is the case a hardcoded stride gets wrong.
    using SQ = SparseQuery<128>;
    using QC = QueryCodec<128>;
    const QueryLayout layout{/*fused=*/false};
    VecZ buf;
    std::vector<size_t> offs;
    size_t off = 0;
    const std::vector<std::vector<uint16_t>> terms = {
        strided(3, 0, 1, 256),   // consecutive -> gap width 0
        strided(6, 10, 40, 256), // wide gaps -> gap width near the raw position width
        {},                      // empty
        strided(40, 0, 6, 256),  // wide enough that bitmap becomes competitive
        strided(1, 255, 1, 256), // single position at the very top
        strided(20, 7, 2, 256),  // uniform stride 2
    };
    for (const auto &t : terms) {
        offs.push_back(off);
        off += SQ::push(buf, t.data(), t.size(), 1);
    }
    BOOST_TEST(QC::count_queries(buf, layout) == terms.size());

    off = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(off == offs[i]);
        BOOST_TEST(SQ::k_at(buf, off) == terms[i].size());
        std::vector<uint16_t> out(terms[i].size() + 1);
        (void)SQ::read_positions(buf, off, out.data());
        out.resize(terms[i].size());
        BOOST_TEST(out == terms[i], boost::test_tools::per_element());
        off = QC::next_off(buf, layout, off);
    }
    BOOST_TEST(off == buf.size());
}

BOOST_AUTO_TEST_CASE(sparse_record_picks_the_smallest_of_its_three_modes) {
    // Load-bearing: gap ALONE is 8.56 B/term against fixed lanes' 8.00 on uniform draws at 250 modes.
    using SQ = SparseQuery<128>;
    std::mt19937_64 rng(12345);
    for (size_t trial = 0; trial < 400; ++trial) {
        const size_t k = rng() % 60;
        const auto pos = scattered(k, 256, rng);
        VecZ buf;
        const size_t w = SQ::push(buf, pos.data(), pos.size(), 1);
        const size_t gwid = SQ::gap_width(pos.data(), pos.size());
        const size_t best = std::min({SQ::words_of(SQ::fixed_bits(pos.size())),
                                      SQ::words_of(SQ::gap_bits(pos.size(), gwid)),
                                      SQ::words_of(SQ::bitmap_bits(pos.size()))});
        BOOST_TEST(w == best, "k=" << k << " wrote " << w << " words, best was " << best);
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_position_width_is_the_compile_time_bucket) {
    BOOST_TEST(SparseQuery<32>::kPosBits == 6U);    // U=64
    BOOST_TEST(SparseQuery<128>::kPosBits == 8U);   // U=256, both lattice models
    BOOST_TEST(SparseQuery<250>::kPosBits == 9U);   // U=500
    BOOST_TEST(SparseQuery<512>::kPosBits == 10U);  // U=1024
    BOOST_TEST(SparseQuery<1024>::kPosBits == 11U); // U=2048
}

BOOST_AUTO_TEST_CASE(sparse_record_carries_the_extreme_bit_positions) {
    // MSb0 ordering puts logical index 0 at the TOP, so bit 2N-1 is the common case, not a rare one.
    differential<32>({0}, 1);
    differential<32>({63}, 1);
    differential<32>({0, 63}, -1);
    differential<128>({0, 255}, 1);
    differential<250>({0, 499}, 1);
    differential<1024>({0, 2047}, -1);
    differential<12>({0, 23}, 1); // the non-word-multiple width, at both extremes
}

BOOST_AUTO_TEST_CASE(sparse_record_encoding_is_deterministic) {
    std::mt19937_64 rng(0xDE7ULL);
    for (size_t trial = 0; trial < 200; ++trial) {
        const auto pos = scattered(rng() % 40, 256, rng);
        VecZ a;
        VecZ b;
        const size_t wa = SparseQuery<128>::push(a, pos.data(), pos.size(), 1);
        const size_t wb = SparseQuery<128>::push(b, pos.data(), pos.size(), 1);
        BOOST_TEST(wa == wb);
        BOOST_TEST(a == b, boost::test_tools::per_element());
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_pair_count_recomputes_d_from_positions) {
    // d is recomputed from positions, so it must agree with the definition: mode m owns bits 2m, 2m+1.
    std::mt19937_64 rng(0xD1D1ULL);
    for (size_t trial = 0; trial < 100; ++trial) {
        const auto pos = scattered(rng() % 30, 256, rng);
        BOOST_TEST(SparseQuery<128>::pair_count(pos.data(), pos.size()) == reference_pair_count(pos));
    }
    const std::vector<uint16_t> straddle{1, 2, 5, 6};
    BOOST_TEST(SparseQuery<128>::pair_count(straddle.data(), straddle.size()) == 0U);
    const std::vector<uint16_t> real{2, 3, 6, 7};
    BOOST_TEST(SparseQuery<128>::pair_count(real.data(), real.size()) == 2U);
    std::vector<uint16_t> paired;
    for (uint16_t m = 0; m < 16; ++m) {
        paired.push_back(static_cast<uint16_t>(2 * m));
        paired.push_back(static_cast<uint16_t>(2 * m + 1));
    }
    BOOST_TEST(SparseQuery<128>::pair_count(paired.data(), paired.size()) == paired.size() / 2);
}

BOOST_AUTO_TEST_CASE(sparse_record_fused_stream_interleaves_values_and_stays_walkable) {
    using QC = QueryCodec<128>;
    std::mt19937_64 rng(0xF5EDULL);
    std::vector<std::vector<uint16_t>> terms;
    std::vector<double> vals;
    VecZ plain;
    for (size_t i = 0; i < 24; ++i) {
        terms.push_back(scattered(rng() % 45, 256, rng));
        vals.push_back(static_cast<double>(i) * 0.5 - 3.25);
        (void)QC::push(
            plain,
            [&] {
                Monomial<128> m;
                for (const auto p : terms.back()) {
                    m.set(p);
                }
                return m;
            }(),
            1);
    }
    VecZ fused;
    QC::build_fused(plain, vals, fused);

    const QueryLayout layout{.fused = true};
    BOOST_TEST(QC::count_queries(fused, layout) == terms.size());
    size_t off = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(QC::k_at(fused, off) == terms[i].size());
        BOOST_TEST(QC::value_at(fused, layout, off) == vals[i]);
        std::vector<uint16_t> out(terms[i].size() + 1);
        int phase = 0;
        const size_t next = QC::read_positions(fused, layout, off, out.data(), phase);
        out.resize(terms[i].size());
        BOOST_TEST(out == terms[i], boost::test_tools::per_element());
        off = QC::next_off(fused, layout, off);
        BOOST_TEST(next == off);
    }
    BOOST_TEST(off == fused.size());
}

// A separate case because `-0.0 == 0.0` is TRUE, so bit-exactness needs its own assertion.
BOOST_AUTO_TEST_CASE(sparse_record_fused_value_channel_is_bit_exact_and_reusable) {
    using QC = QueryCodec<128>;
    const QueryLayout layout{.fused = true};

    auto push_terms = [](VecZ &buf, const std::vector<std::vector<uint16_t>> &terms) {
        for (const auto &t : terms) {
            Monomial<128> m;
            for (const auto p : t) {
                m.set(p);
            }
            (void)QC::push(buf, m, 1);
        }
    };

    // 1. BIT-EXACTNESS via memcmp, so -0.0 stays distinguished from 0.0. Widths differ per term.
    const std::vector<double> values = {
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.141592653589793,
        -2.718281828459045e-300,            // near-denormal magnitude
        std::numeric_limits<double>::min(), // smallest normal
        std::numeric_limits<double>::denorm_min(),
    };
    const std::vector<std::vector<uint16_t>> terms = {
        {3},
        {0, 255},
        {1, 2, 3, 4, 5, 6, 7},
        {9, 40},
        {2, 3},
        {5, 60, 61, 200},
        {17},
        {0, 1, 2},
    };
    BOOST_REQUIRE_EQUAL(terms.size(), values.size());

    VecZ plain;
    push_terms(plain, terms);
    VecZ fused;
    QC::build_fused(plain, values, fused);

    size_t off = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double v_out = QC::value_at(fused, layout, off);
        BOOST_CHECK(std::memcmp(&v_out, &values[i], sizeof(double)) == 0);
        off = QC::next_off(fused, layout, off);
    }
    BOOST_TEST(off == fused.size());

    // 2. BUFFER REUSE: size must be exact, or a shorter gate reads the previous gate's trailing words.
    VecZ plain_big;
    std::vector<double> vbig;
    std::vector<std::vector<uint16_t>> big;
    for (size_t r = 0; r < 32; ++r) {
        big.push_back({static_cast<uint16_t>(r), static_cast<uint16_t>(r + 60)});
        vbig.push_back(static_cast<double>(r) * 1.5 - 7.0);
    }
    push_terms(plain_big, big);
    VecZ out;
    QC::build_fused(plain_big, vbig, out);
    const size_t cap_after_big = out.capacity();

    VecZ plain_small;
    const std::vector<double> vsmall = {42.0, -42.0, 0.25};
    const std::vector<std::vector<uint16_t>> small = {{1}, {2, 3}, {4, 5, 6}};
    push_terms(plain_small, small);
    QC::build_fused(plain_small, vsmall, out);
    BOOST_TEST(QC::count_queries(out, layout) == vsmall.size());
    BOOST_CHECK_GE(out.capacity(), cap_after_big);
    off = 0;
    for (size_t i = 0; i < vsmall.size(); ++i) {
        const double v_out = QC::value_at(out, layout, off);
        BOOST_CHECK(std::memcmp(&v_out, &vsmall[i], sizeof(double)) == 0);
        off = QC::next_off(out, layout, off);
    }
    BOOST_TEST(off == out.size());

    // 3. EMPTY INPUT: the self slot is cleared before the exchange, into a buffer holding stale words.
    VecZ empty;
    VecZ dirty{1, 2, 3};
    QC::build_fused(empty, {}, dirty);
    BOOST_TEST(dirty.empty());
}
