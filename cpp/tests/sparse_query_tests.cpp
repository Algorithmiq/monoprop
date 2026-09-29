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
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"

#include "dense_query_reference.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

template <size_t N>
auto differential(const std::vector<uint16_t> &pos, int phase) -> size_t {
    using QW = QueryWire<N>;
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
    const size_t sw = QW::push(sbuf, pos, phase);
    BOOST_REQUIRE_EQUAL(sbuf.size(), sw);

    BOOST_TEST(QW::words_at(sbuf, 0) == sw);
    BOOST_TEST(QW::k_at(sbuf, 0) == k);
    BOOST_TEST(QW::phase_at(sbuf, 0) == phase);

    std::vector<uint16_t> sout(k == 0 ? 1 : k);
    const auto sread = QW::read_positions(sbuf, 0, sout);
    BOOST_TEST(sread.next == sw);
    BOOST_TEST(sread.phase == phase);
    sout.resize(k);
    BOOST_TEST(sout == pos, boost::test_tools::per_element());
    // The header must be recoverable from the positions it describes; see dense_query_reference.h.
    BOOST_TEST(test_ref::wire_header_is_consistent<N>(sbuf, 0, sout));

    // Round-trips the decoded positions through a Monomial, cross-checked against the dense oracle.
    Monomial<N> sm;
    for (size_t j = 0; j < k; ++j) {
        sm.set(sout[j]);
    }
    BOOST_TEST(sm.count() == k);
    BOOST_TEST((sm == dmono));

    // Encoding from positions taken off `want` itself must match encoding from `pos` directly: production
    // always builds queries from an ascending position vector, never from a bitset.
    VecZ mbuf;
    std::vector<uint16_t> from_mono;
    for (size_t b = want.find_first(); b < want.size(); b = want.find_next(b)) {
        from_mono.push_back(static_cast<uint16_t>(b));
    }
    const size_t mw = QW::push(mbuf, from_mono, phase);
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

// Uniform draws are the widest gap widths, which is the case a regular pattern never reaches; see
// sparse_record_reaches_the_widest_gap_width for why a narrow universe needs a run plus one outlier.
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

// The reference d, written against the definition (mode m owns bits 2m and 2m+1), not the wire record.
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
    // Widths with no whole word: 12 modes (LiH) is kBits=24, so the record's payload is a partial word.
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

BOOST_AUTO_TEST_CASE(sparse_record_reaches_the_widest_gap_width) {
    // gw == kPosBits is the record's worst case, and uniform draws cannot reach it at kBits=24: it needs
    // one gap of at least half the universe, which only a dense run plus one far outlier forces.
    const auto count_widest = [](auto tag, size_t universe) {
        using QW = QueryWire<decltype(tag)::value>;
        size_t used = 0;
        size_t bad = 0;
        const auto tally = [&](const std::vector<uint16_t> &pos) {
            if (pos.size() < 2 || pos.size() > QW::kMaxPositions) {
                return;
            }
            VecZ buf;
            const size_t w = QW::push(buf, pos, 1);
            const size_t gw = QW::gap_width(pos);
            if (gw != QW::kPosBits) {
                return;
            }
            ++used;
            std::vector<uint16_t> back(pos.size());
            (void)QW::read_positions(buf, 0, back);
            bad += static_cast<size_t>(back != pos || QW::k_at(buf, 0) != pos.size()
                                       || w != QW::words_of(QW::gap_bits(pos.size(), gw)));
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
    const auto narrow = count_widest(std::integral_constant<size_t, 12>{}, 24);
    const auto partial = count_widest(std::integral_constant<size_t, 250>{}, 500);
    const auto bucket = count_widest(std::integral_constant<size_t, 128>{}, 256);
    BOOST_TEST(narrow.first > 0U);
    BOOST_TEST(partial.first > 0U);
    BOOST_TEST(bucket.first > 0U);
    BOOST_TEST(narrow.second == 0U);
    BOOST_TEST(partial.second == 0U);
    BOOST_TEST(bucket.second == 0U);
}

BOOST_AUTO_TEST_CASE(sparse_record_survives_the_five_bit_k_escape) {
    // The escape is at k = 31; both boundaries are exercised here so a field-width change is caught.
    for (const size_t k : {size_t{30}, size_t{31}, size_t{32}, size_t{62}, size_t{63}, size_t{64}, size_t{200}}) {
        const auto pos = strided(k, 0, 3, 2048);
        BOOST_REQUIRE_EQUAL(pos.size(), k);
        differential<1024>(pos, 1);
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_bounds_the_fully_paired_term) {
    // Every bit set means every gap is 0, so gw is 0 and the header plus one raw position fit one word.
    std::vector<uint16_t> all(2048);
    for (size_t j = 0; j < all.size(); ++j) {
        all[j] = static_cast<uint16_t>(j);
    }
    const size_t sw = differential<1024>(all, 1);
    BOOST_TEST(sw == 1U);
}

BOOST_AUTO_TEST_CASE(sparse_record_never_exceeds_its_own_raw_lanes) {
    // Exhaustive rather than sampled: gw = bit_width(max gap) <= kPosBits, so kPosBits + (k-1)*gw <=
    // k*kPosBits at every k.
    const auto check = [](auto tag) {
        using QW = QueryWire<decltype(tag)::value>;
        size_t cells = 0;
        size_t bad = 0;
        for (size_t k = 0; k <= QW::kMaxPositions; ++k) {
            for (size_t gw = 0; gw <= QW::kPosBits; ++gw) {
                const size_t lanes = QW::words_of(QW::header_bits_for(k) + (k * QW::kPosBits));
                bad += static_cast<size_t>(QW::words_of(QW::gap_bits(k, gw)) > lanes);
                ++cells;
            }
        }
        return std::pair<size_t, size_t>{cells, bad};
    };
    const auto narrow = check(std::integral_constant<size_t, 12>{});
    const auto bucket = check(std::integral_constant<size_t, 128>{});
    const auto wide = check(std::integral_constant<size_t, 250>{});
    BOOST_TEST(narrow.second == 0U);
    BOOST_TEST(bucket.second == 0U);
    BOOST_TEST(wide.second == 0U);
    // A guarded loop that asserted nothing would pass the three above; these are the cell counts.
    BOOST_TEST(narrow.first == 25U * 6U);
    BOOST_TEST(bucket.first == 257U * 9U);
    BOOST_TEST(wide.first == 501U * 10U);
}

BOOST_AUTO_TEST_CASE(sparse_record_walks_a_multi_query_buffer_exactly) {
    // Mixed width, which is the case a hardcoded stride gets wrong.
    using QW = QueryWire<128>;
    const QueryForm form = QueryForm::Plain;
    VecZ buf;
    std::vector<size_t> offs;
    size_t off = 0;
    const std::vector<std::vector<uint16_t>> terms = {
        strided(3, 0, 1, 256),   // consecutive -> gap width 0
        strided(6, 10, 40, 256), // wide gaps -> gap width near the raw position width
        {},                      // empty
        strided(40, 0, 6, 256),  // wide, evenly spread positions
        strided(1, 255, 1, 256), // single position at the very top
        strided(20, 7, 2, 256),  // uniform stride 2
    };
    for (const auto &t : terms) {
        offs.push_back(off);
        off += QW::push(buf, t, 1);
    }
    BOOST_TEST(QW::count_queries(buf, form) == terms.size());

    off = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(off == offs[i]);
        BOOST_TEST(QW::k_at(buf, off) == terms[i].size());
        std::vector<uint16_t> out(terms[i].size() + 1);
        (void)QW::read_positions(buf, off, out);
        out.resize(terms[i].size());
        BOOST_TEST(out == terms[i], boost::test_tools::per_element());
        off = QW::next_off(buf, form, off);
    }
    BOOST_TEST(off == buf.size());
}

BOOST_AUTO_TEST_CASE(sparse_record_is_exactly_the_gap_code_it_costed) {
    // The encoder's width and the costing function must agree on every draw, uniform ones included.
    using QW = QueryWire<128>;
    std::mt19937_64 rng(12345);
    for (size_t trial = 0; trial < 400; ++trial) {
        const size_t k = rng() % 60;
        const auto pos = scattered(k, 256, rng);
        VecZ buf;
        const size_t w = QW::push(buf, pos, 1);
        const size_t gwid = QW::gap_width(pos);
        BOOST_CHECK_MESSAGE(
            w == QW::words_of(QW::gap_bits(pos.size(), gwid)),
            "k=" << k << " wrote " << w << " words, costed " << QW::words_of(QW::gap_bits(pos.size(), gwid)));
        BOOST_TEST(w <= QW::words_of(QW::header_bits_for(pos.size()) + (pos.size() * QW::kPosBits)));
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_position_width_is_the_compile_time_bucket) {
    BOOST_TEST(QueryWire<32>::kPosBits == 6U);    // U=64
    BOOST_TEST(QueryWire<128>::kPosBits == 8U);   // U=256, both lattice models
    BOOST_TEST(QueryWire<250>::kPosBits == 9U);   // U=500
    BOOST_TEST(QueryWire<512>::kPosBits == 10U);  // U=1024
    BOOST_TEST(QueryWire<1024>::kPosBits == 11U); // U=2048
}

BOOST_AUTO_TEST_CASE(sparse_record_carries_the_extreme_bit_positions) {
    // MSb0 ordering puts logical index 0 at the top, so bit 2N-1 is the common case, not a rare one.
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
        const size_t wa = QueryWire<128>::push(a, pos, 1);
        const size_t wb = QueryWire<128>::push(b, pos, 1);
        BOOST_TEST(wa == wb);
        BOOST_TEST(a == b, boost::test_tools::per_element());
    }
}

BOOST_AUTO_TEST_CASE(sparse_record_pair_count_recomputes_d_from_positions) {
    // d is recomputed from positions, so it must agree with the definition: mode m owns bits 2m, 2m+1.
    std::mt19937_64 rng(0xD1D1ULL);
    for (size_t trial = 0; trial < 100; ++trial) {
        const auto pos = scattered(rng() % 30, 256, rng);
        BOOST_TEST(QueryWire<128>::pair_count(pos) == reference_pair_count(pos));
    }
    const std::vector<uint16_t> straddle{1, 2, 5, 6};
    BOOST_TEST(QueryWire<128>::pair_count(straddle) == 0U);
    const std::vector<uint16_t> real{2, 3, 6, 7};
    BOOST_TEST(QueryWire<128>::pair_count(real) == 2U);
    std::vector<uint16_t> paired;
    for (uint16_t m = 0; m < 16; ++m) {
        paired.push_back(static_cast<uint16_t>(2 * m));
        paired.push_back(static_cast<uint16_t>(2 * m + 1));
    }
    BOOST_TEST(QueryWire<128>::pair_count(paired) == paired.size() / 2);
}

namespace {
// Extracts an ascending position vector from a Monomial, mirroring what a real query source (the scan,
// the partner merge) already holds; production never encodes straight from a bitset.
template <size_t N>
auto positions_of(const Monomial<N> &m) -> std::vector<uint16_t> {
    std::vector<uint16_t> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<uint16_t>(b));
    }
    return pos;
}
} // namespace

BOOST_AUTO_TEST_CASE(sparse_record_fused_stream_interleaves_values_and_stays_walkable) {
    using QW = QueryWire<128>;
    std::mt19937_64 rng(0xF5EDULL);
    std::vector<std::vector<uint16_t>> terms;
    std::vector<double> vals;
    VecZ plain;
    for (size_t i = 0; i < 24; ++i) {
        terms.push_back(scattered(rng() % 45, 256, rng));
        vals.push_back(static_cast<double>(i) * 0.5 - 3.25);
        Monomial<128> m;
        for (const auto p : terms.back()) {
            m.set(p);
        }
        const auto pos = positions_of<128>(m);
        (void)QW::push(plain, pos, 1);
    }
    VecZ fused;
    QW::build_fused(plain, vals, fused);

    const QueryForm form = QueryForm::Fused;
    BOOST_TEST(QW::count_queries(fused, form) == terms.size());
    size_t off = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(QW::k_at(fused, off) == terms[i].size());
        BOOST_TEST(QW::value_at(fused, off) == vals[i]);
        std::vector<uint16_t> out(terms[i].size() + 1);
        const auto rq = QW::read_query(fused, form, off, out);
        BOOST_TEST(rq.phase == 1);
        out.resize(terms[i].size());
        BOOST_TEST(out == terms[i], boost::test_tools::per_element());
        off = QW::next_off(fused, form, off);
        BOOST_TEST(rq.next == off);
    }
    BOOST_TEST(off == fused.size());
}

// A separate case because `-0.0 == 0.0` evaluates true, so bit-exactness needs its own assertion.
BOOST_AUTO_TEST_CASE(sparse_record_fused_value_channel_is_bit_exact_and_reusable) {
    using QW = QueryWire<128>;
    const QueryForm form = QueryForm::Fused;

    auto push_terms = [](VecZ &buf, const std::vector<std::vector<uint16_t>> &terms) {
        for (const auto &t : terms) {
            Monomial<128> m;
            for (const auto p : t) {
                m.set(p);
            }
            const auto pos = positions_of<128>(m);
            (void)QW::push(buf, pos, 1);
        }
    };

    // Bit-exactness via memcmp, so -0.0 stays distinguished from 0.0; widths differ per term.
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
    QW::build_fused(plain, values, fused);

    size_t off = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double v_out = QW::value_at(fused, off);
        BOOST_CHECK(std::memcmp(&v_out, &values[i], sizeof(double)) == 0);
        off = QW::next_off(fused, form, off);
    }
    BOOST_TEST(off == fused.size());

    // Buffer reuse: sizes must be exact, or a shorter gate reads the previous gate's trailing words.
    VecZ plain_big;
    std::vector<double> vbig;
    std::vector<std::vector<uint16_t>> big;
    for (size_t r = 0; r < 32; ++r) {
        big.push_back({static_cast<uint16_t>(r), static_cast<uint16_t>(r + 60)});
        vbig.push_back(static_cast<double>(r) * 1.5 - 7.0);
    }
    push_terms(plain_big, big);
    VecZ out;
    QW::build_fused(plain_big, vbig, out);
    const size_t cap_after_big = out.capacity();

    VecZ plain_small;
    const std::vector<double> vsmall = {42.0, -42.0, 0.25};
    const std::vector<std::vector<uint16_t>> small = {{1}, {2, 3}, {4, 5, 6}};
    push_terms(plain_small, small);
    QW::build_fused(plain_small, vsmall, out);
    BOOST_TEST(QW::count_queries(out, form) == vsmall.size());
    BOOST_CHECK_GE(out.capacity(), cap_after_big);
    off = 0;
    for (size_t i = 0; i < vsmall.size(); ++i) {
        const double v_out = QW::value_at(out, off);
        BOOST_CHECK(std::memcmp(&v_out, &vsmall[i], sizeof(double)) == 0);
        off = QW::next_off(out, form, off);
    }
    BOOST_TEST(off == out.size());

    // Empty input: the self slot is cleared before the exchange, into a buffer holding stale words.
    VecZ empty;
    VecZ dirty{1, 2, 3};
    QW::build_fused(empty, {}, dirty);
    BOOST_TEST(dirty.empty());
}

// Positions with exactly k entries whose widest gap is exactly `gw`: one gap of 2^(gw-1) -- the smallest
// value of that bit width -- then a contiguous run. Empty if the shape does not fit the universe.
namespace {
auto gap_shaped(size_t k, size_t gw, size_t universe) -> std::vector<uint16_t> {
    std::vector<uint16_t> v;
    if (k == 0) {
        return v;
    }
    if (gw == 0 || k == 1) {
        if (gw != 0 || k > universe) {
            return v;
        }
        for (size_t j = 0; j < k; ++j) {
            v.push_back(static_cast<uint16_t>(j));
        }
        return v;
    }
    const size_t second = (size_t{1} << (gw - 1)) + 1U; // gap value 2^(gw-1), i.e. bit_width == gw
    if (second + (k - 2U) >= universe) {
        return v;
    }
    v.push_back(0);
    for (size_t j = 0; j + 1 < k; ++j) {
        v.push_back(static_cast<uint16_t>(second + j));
    }
    return v;
}
} // namespace

BOOST_AUTO_TEST_CASE(sparse_record_round_trips_every_reachable_k_and_gap_width) {
    // The whole (k, gw) surface, constructed rather than drawn: a random draw reaches neither
    // gw = kPosBits nor the escape boundary. `cells` guards against a shape silently failing to fit.
    using QW = QueryWire<128>;
    size_t cells = 0;
    for (size_t k = 0; k <= 40U; ++k) {
        for (size_t gw = 0; gw <= QW::kPosBits; ++gw) {
            if (k < 2U && gw > 0U) {
                continue; // one gap width is reachable below k = 2, so the other rows are the same cell
            }
            const auto pos = gap_shaped(k, gw, QW::kBits);
            if (pos.size() != k) {
                continue;
            }
            BOOST_REQUIRE_EQUAL(QW::gap_width(pos), k < 2 ? 0U : gw);
            const size_t w = differential<128>(pos, (k % 3U) == 0U ? 0 : ((k % 3U) == 1U ? 1 : -1));
            BOOST_TEST(w == QW::words_of(QW::gap_bits(k, k < 2 ? 0U : gw)));
            ++cells;
        }
    }
    BOOST_TEST(cells == 353U);

    // The width boundary: k = kBits is a fully paired term, one word because every gap is 0.
    for (const size_t k : {size_t{254}, size_t{255}, size_t{256}}) {
        const auto pos = gap_shaped(k, 0, QW::kBits);
        BOOST_REQUIRE_EQUAL(pos.size(), k);
        BOOST_TEST(differential<128>(pos, 1) == 1U);
    }
}

// --- ordered merge of per-range scan results ------------------------------------------------------------------

namespace {

constexpr size_t kMergeModes = 32;
using MergeResult = FusedScanResult<kMergeModes>;

auto reset_streams(MergeResult &r, mpi::SlotWindow w, bool capture) -> void {
    r.leader_queries.reset(w);
    r.leader_src.reset(w);
    r.follower_queries.reset(w);
    r.follower_src.reset(w);
    if (capture) {
        r.leader_val.reset(w);
        r.follower_val.reset(w);
    }
}

auto same_stage(const SelfQueryStage<kMergeModes> &a, const SelfQueryStage<kMergeModes> &b) -> bool {
    if (a.size() != b.size() || a.positions() != b.positions()) {
        return false;
    }
    for (size_t q = 0; q < a.size(); ++q) {
        if (a.pos_off[q] != b.pos_off[q] || a.k_of[q] != b.k_of[q] || a.phase_of[q] != b.phase_of[q]) {
            return false;
        }
    }
    return std::equal(a.pos_flat.begin(),
                      a.pos_flat.begin() + static_cast<std::ptrdiff_t>(a.positions()),
                      b.pos_flat.begin());
}

auto bits_equal(const std::vector<double> &a, const std::vector<double> &b) -> bool {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

} // namespace

// Hand-built per-range pieces with variable-length wire records (escaped k included), a window at a
// non-zero base, self-owned queries on both stages, an empty middle piece and value capture on and off.
// The merge must reproduce the single stream a serial scan would have produced: each destination's
// records, sources and values in range order, each self stage rebased in range order, cosine sets in
// range order, and every piece's buffers released afterwards.
BOOST_AUTO_TEST_CASE(openmp_scan_merge_keeps_source_order_windows_and_self_stages) {
    using QW = QueryWire<kMergeModes>;
    const mpi::SlotWindow window{.base = 2, .count = 3};
    const size_t me = 3;
    for (const bool capture : {false, true}) {
        std::mt19937_64 rng(0x3E26EULL);
        std::vector<MergeResult> pieces(4);
        MergeResult expect;
        reset_streams(expect, window, capture);
        size_t src = 0;
        for (size_t p = 0; p < pieces.size(); ++p) {
            auto &piece = pieces[p];
            reset_streams(piece, window, capture);
            CosineWordBuilder cos;
            if (p == 1) {
                piece.cos_blocks.push_back(cos.finish()); // an empty range still reports its (empty) set
                continue;
            }
            for (size_t q = 0; q < 40 + (p * 7); ++q) {
                const size_t dest = window.slot(mpi::WindowIndex{rng() % window.count});
                const bool follower = (rng() & 1U) != 0;
                const std::array<size_t, 6> widths{0, 1, 5, 31, 40, 64};
                const size_t k = widths[rng() % widths.size()];
                std::vector<uint16_t> pos(64);
                std::iota(pos.begin(), pos.end(), uint16_t{0});
                std::shuffle(pos.begin(), pos.end(), rng);
                pos.resize(k);
                std::ranges::sort(pos);
                const int phase = static_cast<int>(rng() % 3) - 1;
                const double v = std::ldexp(static_cast<double>(rng() % 1000) - 500.0, -7);
                for (auto *r : {&piece, &expect}) {
                    if (dest == me) {
                        (follower ? r->follower_self : r->leader_self).push(pos, phase);
                    }
                    else {
                        QW::push((follower ? r->follower_queries : r->leader_queries).at_slot(dest), pos, phase);
                    }
                    (follower ? r->follower_src : r->leader_src).at_slot(dest).push_back(src);
                    if (capture) {
                        (follower ? r->follower_val : r->leader_val).at_slot(dest).push_back(v);
                    }
                }
                cos.push_index((p * 1024 * 64) + (q * 3));
                ++src;
            }
            piece.cos_blocks.push_back(cos.finish());
        }
        std::vector<CosMask> expect_cos;
        for (const auto &piece : pieces) {
            expect_cos.push_back(piece.cos_blocks.front());
        }

        const auto merged = merge_scan_ranges<kMergeModes>(pieces, window, me, capture);

        BOOST_TEST_CONTEXT("capture=" << capture) {
            BOOST_TEST((merged.leader_queries.window() == window));
            BOOST_TEST((merged.follower_src.window() == window));
            BOOST_TEST(merged.leader_val.size() == (capture ? window.count : 0U));
            BOOST_TEST(merged.follower_val.size() == (capture ? window.count : 0U));
            for (const auto wi : window.indices()) {
                BOOST_TEST((merged.leader_queries[wi] == expect.leader_queries[wi]));
                BOOST_TEST((merged.follower_queries[wi] == expect.follower_queries[wi]));
                BOOST_TEST((merged.leader_src[wi] == expect.leader_src[wi]));
                BOOST_TEST((merged.follower_src[wi] == expect.follower_src[wi]));
                if (capture) {
                    BOOST_TEST(bits_equal(merged.leader_val[wi], expect.leader_val[wi]));
                    BOOST_TEST(bits_equal(merged.follower_val[wi], expect.follower_val[wi]));
                }
            }
            // Self wire stays empty; the stages carry the self slot's queries in order.
            BOOST_TEST(merged.leader_queries.at_slot(me).empty());
            BOOST_TEST(merged.follower_queries.at_slot(me).empty());
            BOOST_TEST(expect.leader_self.size() > 0U);
            BOOST_TEST(expect.follower_self.size() > 0U);
            BOOST_TEST(same_stage(merged.leader_self, expect.leader_self));
            BOOST_TEST(same_stage(merged.follower_self, expect.follower_self));
            BOOST_TEST(merged.leader_self.size() == merged.leader_src.at_slot(me).size());
            BOOST_TEST(merged.follower_self.size() == merged.follower_src.at_slot(me).size());
            BOOST_TEST_REQUIRE(merged.cos_blocks.size() == expect_cos.size());
            for (size_t p = 0; p < expect_cos.size(); ++p) {
                BOOST_TEST((merged.cos_blocks[p].blocks == expect_cos[p].blocks));
                BOOST_TEST(merged.cos_blocks[p].total_count == expect_cos[p].total_count);
            }
            // Drained pieces hold no allocation: clear() alone would have kept their capacity.
            size_t retained = 0;
            for (const auto &piece : pieces) {
                for (const auto wi : window.indices()) {
                    retained += piece.leader_queries[wi].capacity() + piece.follower_queries[wi].capacity();
                    retained += piece.leader_src[wi].capacity() + piece.follower_src[wi].capacity();
                    if (capture) {
                        retained += piece.leader_val[wi].capacity() + piece.follower_val[wi].capacity();
                    }
                }
                for (const auto *stage : {&piece.leader_self, &piece.follower_self}) {
                    retained += stage->pos_flat.capacity() + stage->pos_off.capacity() + stage->k_of.capacity()
                                + stage->phase_of.capacity();
                }
                retained += piece.cos_blocks.capacity();
            }
            BOOST_TEST(retained == 0U);
        }
    }
}

// The merge moves a buffer instead of copying it when one piece holds a whole stream, or when the first
// nonempty piece already has room for the total; otherwise it reserves the exact total once.
BOOST_AUTO_TEST_CASE(openmp_scan_merge_moves_the_first_useful_buffer) {
    const mpi::SlotWindow window{.base = 0, .count = 2};
    const size_t me = 0;
    std::vector<MergeResult> pieces(3);
    for (auto &piece : pieces) {
        reset_streams(piece, window, false);
    }
    // Slot 0 sources: only piece 1 has any, so the merged stream is that very buffer.
    auto &only = pieces[1].leader_src.at_slot(0);
    only.reserve(100);
    only.assign({4, 5, 6});
    const auto *only_data = only.data();
    // Slot 1 sources: piece 0 has room for all of them, so its buffer is kept and piece 2 appended.
    auto &roomy = pieces[0].follower_src.at_slot(1);
    roomy.reserve(64);
    roomy.assign({1, 2});
    const auto *roomy_data = roomy.data();
    pieces[2].follower_src.at_slot(1).assign({9});
    // Slot 1 leader sources: no piece has room, so the result is one exact reservation.
    pieces[0].leader_src.at_slot(1).assign({1, 2});
    pieces[0].leader_src.at_slot(1).shrink_to_fit();
    pieces[2].leader_src.at_slot(1).assign({8, 9, 10});
    // Self stage, aligned with slot 0's sources: only piece 1 has queries, so its stage is moved whole.
    const std::vector<uint16_t> pos{3, 9};
    for (int q = 0; q < 3; ++q) {
        pieces[1].leader_self.push(pos, 1);
    }
    pieces[0].leader_self.reserve(8, 4); // capacity without content is not a query
    const auto *stage_data = pieces[1].leader_self.pos_flat.data();

    auto merged = merge_scan_ranges<kMergeModes>(pieces, window, me, false);
    BOOST_TEST(merged.leader_src.at_slot(0).data() == only_data);
    BOOST_TEST(merged.follower_src.at_slot(1).data() == roomy_data);
    BOOST_TEST((merged.follower_src.at_slot(1) == std::vector<size_t>{1, 2, 9}));
    BOOST_TEST((merged.leader_src.at_slot(1) == std::vector<size_t>{1, 2, 8, 9, 10}));
    BOOST_TEST(merged.leader_src.at_slot(1).capacity() == 5U);
    BOOST_TEST(merged.leader_self.size() == 3U);
    BOOST_TEST(merged.leader_self.pos_flat.data() == stage_data);
}

// --- checked record boundaries and decoded-position validation (resolver input is untrusted) ---------------

namespace {

// A single hand-packed header word: phase field, k field (and the wide-k field when escaped), gw field.
template <size_t N>
auto header_word(uint64_t phase_field, uint64_t k_field, std::optional<uint64_t> long_k, uint64_t gw) -> size_t {
    using QW = QueryWire<N>;
    uint64_t w = phase_field | (k_field << QW::kPhaseBits);
    size_t at = QW::kPhaseBits + QW::kKBits;
    if (long_k) {
        w |= *long_k << at;
        at += QW::kLongKBits;
    }
    w |= gw << at;
    return static_cast<size_t>(w);
}

template <size_t N>
auto expect_malformed(const VecZ &buf, QueryForm form, size_t off, const char *label) -> void {
    BOOST_TEST_CONTEXT(label) {
        BOOST_CHECK_THROW((void)QueryWire<N>::checked_extent(buf, form, off), MalformedQueryStream);
    }
}

} // namespace

// Every record push() can produce passes the check, which reports exactly the words the decoder consumes, plus
// the fused value word. Covers the whole (k, gw) surface at 128 modes, the escape boundary and full width.
BOOST_AUTO_TEST_CASE(sparse_record_checked_extent_accepts_every_canonical_record) {
    const auto check = []<size_t N>(std::integral_constant<size_t, N>, const std::vector<uint16_t> &pos, int phase) {
        using QW = QueryWire<N>;
        for (const auto form : {QueryForm::Plain, QueryForm::Fused}) {
            VecZ buf;
            const size_t words = QW::push(buf, pos, phase);
            if (form == QueryForm::Fused) {
                QW::push_value(buf, -0.25);
            }
            const auto ext = QW::checked_extent(buf, form, 0);
            BOOST_TEST(ext.k == pos.size());
            BOOST_TEST(ext.words == buf.size());
            BOOST_TEST(ext.words == words + (form == QueryForm::Fused ? 1U : 0U));
            std::vector<uint16_t> out(pos.size());
            const auto d = QW::read_query(buf, form, 0, out);
            BOOST_TEST(d.next == ext.words);
            BOOST_TEST(out == pos, boost::test_tools::per_element());
        }
    };
    size_t cells = 0;
    for (size_t k = 0; k <= 40U; ++k) {
        for (size_t gw = 0; gw <= QueryWire<128>::kPosBits; ++gw) {
            const auto pos = gap_shaped(k, gw, QueryWire<128>::kBits);
            if (pos.size() != k || (k < 2U && gw > 0U)) {
                continue;
            }
            check(std::integral_constant<size_t, 128>{}, pos, static_cast<int>(k % 3U) - 1);
            ++cells;
        }
    }
    BOOST_TEST(cells == 353U);
    for (const size_t k : {size_t{30}, size_t{31}, size_t{32}, size_t{200}, size_t{2048}}) {
        check(std::integral_constant<size_t, 1024>{}, strided(k, 0, k == 2048 ? 1 : 3, 2048), 1);
    }
    check(std::integral_constant<size_t, 250>{}, std::vector<uint16_t>{0, 257, 499}, -1);
}

// The plan's fixture: one Plain word at 128 modes with phase +1, an escaped k = 6, gw = 8 and no payload. The
// decoder would consume 20 header bits + 8 + 5*8 = 68 bits (two words), but the canonical k = 6 header is 11
// bits, so the unchecked size calculation claims 59 bits, one word, and the decode would read past the buffer.
BOOST_AUTO_TEST_CASE(sparse_record_checked_extent_rejects_the_noncanonical_escape) {
    using QW = QueryWire<128>;
    const VecZ buf{size_t{0x8037e}};
    const auto h = QW::header_at(buf, 0);
    BOOST_TEST(h.phase == 1);
    BOOST_TEST(h.k == 6U);
    BOOST_TEST(h.gw == 8U);
    BOOST_TEST(h.bits == 20U);
    BOOST_TEST(QW::header_bits_for(h.k) == 11U);
    BOOST_TEST(QW::words_at(buf, 0) == 1U); // the undercount the check exists to catch
    expect_malformed<128>(buf, QueryForm::Plain, 0, "0x8037e alone");
    // Padding that makes the decoder's read fit does not make the header canonical.
    expect_malformed<128>(VecZ{size_t{0x8037e}, 0, 0}, QueryForm::Plain, 0, "0x8037e padded");
    expect_malformed<128>(VecZ{size_t{0x8037e}, 0}, QueryForm::Fused, 0, "0x8037e fused");
}

BOOST_AUTO_TEST_CASE(sparse_record_checked_extent_rejects_malformed_headers_and_boundaries) {
    using QW8 = QueryWire<8>; // kBits 16, kPosBits 4, kLongKBits 5
    expect_malformed<8>(VecZ{}, QueryForm::Plain, 0, "offset at the end");
    expect_malformed<8>(VecZ{header_word<8>(3, 0, std::nullopt, 0)}, QueryForm::Plain, 0, "phase field 3");
    expect_malformed<8>(VecZ{header_word<8>(1, 31, 20, 0)}, QueryForm::Plain, 0, "escaped k above kMaxPositions");
    expect_malformed<8>(VecZ{header_word<8>(1, 31, 3, 0)}, QueryForm::Plain, 0, "escaped k below kKEscape");
    expect_malformed<8>(VecZ{header_word<8>(1, 2, std::nullopt, QW8::kPosBits + 1)}, QueryForm::Plain, 0, "gw");
    // A fused record without its value word, and a two-word record cut to one.
    VecZ fused;
    QW8::push(fused, std::vector<uint16_t>{1, 4}, 1);
    expect_malformed<8>(fused, QueryForm::Fused, 0, "fused value word missing");
    using QW = QueryWire<1024>;
    VecZ wide;
    const size_t words = QW::push(wide, strided(40, 0, 50, 2048), -1);
    BOOST_REQUIRE(words >= 2U);
    wide.pop_back();
    expect_malformed<1024>(wide, QueryForm::Plain, 0, "truncated multiword record");
    // A valid record followed by a lone partial word: the second record's header claims more than remains.
    VecZ tail;
    QW::push(tail, std::vector<uint16_t>{3}, 1);
    const auto first = QW::checked_extent(tail, QueryForm::Plain, 0);
    tail.push_back(wide[0]);
    expect_malformed<1024>(tail, QueryForm::Plain, first.words, "trailing partial record");
}

// read_positions checks the cumulative size_t position before narrowing it to the output type: each must stay
// below 2*NumModes, which also bounds the next addition and keeps the positions strictly ascending.
BOOST_AUTO_TEST_CASE(sparse_record_decode_rejects_positions_outside_the_width) {
    // 250 modes: a 9-bit first position reaches 511 although only 0..499 exist.
    {
        using QW = QueryWire<250>;
        VecZ buf;
        QW::push(buf, std::vector<uint16_t>{505}, 1);
        std::vector<uint16_t> out(1);
        BOOST_CHECK_THROW((void)QW::read_positions(buf, 0, out), MalformedQueryStream);
        VecZ ok;
        QW::push(ok, std::vector<uint16_t>{499}, 1);
        BOOST_CHECK_NO_THROW((void)QW::read_positions(ok, 0, out));
    }
    // 32 modes: a gap walks past position 63.
    {
        using QW = QueryWire<32>;
        VecZ buf;
        QW::push(buf, std::vector<uint16_t>{10, 70}, -1);
        std::vector<uint8_t> out(QW::kBits); // room for any k the header could claim
        BOOST_CHECK_THROW((void)QW::read_positions(buf, 0, out), MalformedQueryStream);
    }
    // 128 modes into the store's uint8: 257 would narrow to 1, and 300 to 44, both "valid" after narrowing.
    {
        using QW = QueryWire<128>;
        for (const auto &pos : {std::vector<uint16_t>{1, 257}, std::vector<uint16_t>{100, 200, 300}}) {
            VecZ buf;
            QW::push(buf, pos, 1);
            BOOST_REQUIRE_NO_THROW((void)QW::checked_extent(buf, QueryForm::Plain, 0));
            std::vector<uint8_t> out(QW::kBits);
            BOOST_CHECK_THROW((void)QW::read_query(buf, QueryForm::Plain, 0, out), MalformedQueryStream);
        }
    }
}
