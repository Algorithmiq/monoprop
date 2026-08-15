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

// The compact query record, tested against the dense one it replaced.
//
// That dense format is no longer production code -- it lives in dense_query_reference.h as a frozen,
// test-only second implementation. Keeping it is what makes the differential at the bottom of this
// file mean anything: with one format left, a round-trip can only ask whether the codec agrees with
// ITSELF, which catches a decoder that mirrors its encoder's mistake not at all. Reference numbers
// like `kQueryWords<kN> == 9` below therefore describe the format that WAS, and are here to keep the
// comparison honest about what it is comparing.
//
// The load-bearing case here is the CONTINUATION ESCAPE, and it is the one a realistic workload cannot
// reach: at the 29M-term bench configuration k is 4 or 6 for all but 94 rows in 20,860,967, so a fuzz
// over the measured population emits zero continuation records and would prove nothing about them.
// Every k below is therefore chosen, not sampled, and the boundaries (5, 6, 7 around kInlinePositions;
// 14, 15 around the first continuation filling) are named individually.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <random>
#include <vector>

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/CompactQuery.h"

#include "dense_query_reference.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

constexpr size_t kN = 250; // the production width: 2N = 500 bits
using CQ = CompactQuery<kN>;
using PosT = CQ::PosT;

// Ascending positions -> the dense monomial the rest of the tree still speaks.
auto mono_from(const std::vector<PosT> &pos) -> Monomial<kN> {
    Monomial<kN> m;
    for (const PosT p : pos) {
        m.set(static_cast<size_t>(p));
    }
    return m;
}

auto positions_of(const Monomial<kN> &m) -> std::vector<PosT> {
    std::vector<PosT> out;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        out.push_back(static_cast<PosT>(b));
    }
    return out;
}

// d = modes carrying both Majoranas. Mode m owns physical bits (2m, 2m+1).
auto pair_count(const std::vector<PosT> &pos) -> size_t {
    size_t d = 0;
    for (size_t j = 0; j + 1 < pos.size(); ++j) {
        if ((pos[j] % 2 == 0) && (pos[j + 1] == pos[j] + 1)) {
            ++d;
        }
    }
    return d;
}

// k distinct ascending positions. paired=true builds a fully paired term (k = 2d), which is the
// population that can exceed the inline width, since a paired term is kept regardless of the cutoff.
auto make_positions(size_t k, bool paired, std::mt19937_64 &rng) -> std::vector<PosT> {
    std::vector<PosT> pos;
    if (paired) {
        BOOST_REQUIRE(k % 2 == 0);
        std::vector<size_t> modes;
        std::uniform_int_distribution<size_t> mode(0, kN - 1);
        while (modes.size() < k / 2) {
            const size_t m = mode(rng);
            if (std::find(modes.begin(), modes.end(), m) == modes.end()) {
                modes.push_back(m);
            }
        }
        std::sort(modes.begin(), modes.end());
        for (const size_t m : modes) {
            pos.push_back(static_cast<PosT>(2 * m));
            pos.push_back(static_cast<PosT>((2 * m) + 1));
        }
    }
    else {
        std::uniform_int_distribution<size_t> bit(0, (2 * kN) - 1);
        std::vector<size_t> bits;
        while (bits.size() < k) {
            const size_t b = bit(rng);
            if (std::find(bits.begin(), bits.end(), b) == bits.end()) {
                bits.push_back(b);
            }
        }
        std::sort(bits.begin(), bits.end());
        for (const size_t b : bits) {
            pos.push_back(static_cast<PosT>(b));
        }
    }
    return pos;
}

} // namespace

// A term of k <= 6 must occupy exactly the 2-word stride the whole design is built on. If this moves,
// the 4.5x byte claim and every count-by-division site move with it.
BOOST_AUTO_TEST_CASE(compact_query_record_is_two_words_for_the_measured_population) {
    BOOST_TEST(CQ::kWordsPerRecord == 2U);
    BOOST_TEST(test_ref::kQueryWords<kN> == 9U); // the record it replaced: 8 monomial words + 1 phase
    // 72 B -> 16 B on the population that is 99.9995% of the measured 20,860,967 rows.
    BOOST_TEST(CQ::records_for(4) == 1U);
    BOOST_TEST(CQ::records_for(6) == 1U);

    std::mt19937_64 rng(0xC0FFEEULL);
    for (const size_t k : {size_t{4}, size_t{6}}) {
        VecZ buf;
        const auto pos = make_positions(k, /*paired=*/false, rng);
        CQ::push(buf, pos.data(), k, 1);
        BOOST_TEST(buf.size() == 2U);
    }
}

// records_for is the arithmetic every count-by-division site depends on; pin it against the decoder's
// actual advance rather than against itself.
BOOST_AUTO_TEST_CASE(compact_query_records_for_matches_the_decoder_advance) {
    std::mt19937_64 rng(0x5EEDULL);
    for (size_t k = 0; k <= 40; ++k) {
        const bool paired = (k % 2 == 0);
        const auto pos = make_positions(k, paired, rng);
        VecZ buf;
        CQ::push(buf, pos.data(), k, -1);
        BOOST_REQUIRE(buf.size() == CQ::words_for(k));

        Monomial<kN> m;
        int ph = 0;
        const size_t next = CQ::read_mono(buf, 0, m, ph);
        BOOST_TEST(next == CQ::words_for(k)); // offsets are WORDS, so this is also the encoded size
    }
    // The boundaries the escape turns on, named so a regression says which one broke.
    BOOST_TEST(CQ::records_for(5) == 1U);
    BOOST_TEST(CQ::records_for(6) == 1U);
    BOOST_TEST(CQ::records_for(7) == 2U);  // first continuation
    BOOST_TEST(CQ::records_for(14) == 2U); // 6 + 8 exactly fills it
    BOOST_TEST(CQ::records_for(15) == 3U); // one over
    BOOST_TEST(CQ::records_for(22) == 3U);
    // words_for is what every offset site actually uses; pin it against records_for rather than letting
    // the two drift apart silently.
    for (size_t k = 0; k <= 40; ++k) {
        BOOST_TEST(CQ::words_for(k) == CQ::records_for(k) * CQ::kWordsPerRecord);
    }
}

// push() reports the words it wrote, and LayerProfile's qbytes is that sum. It used to be
// prof_push * kQueryWords -- a product, which reports 72 B/record whatever the codec actually wrote,
// so the one instrument that can say whether the compact format is live was blind to it. Anchoring
// the return against buf.size() rather than against words_for() means a codec that miscounts and a
// codec that misreports both fail here.
BOOST_AUTO_TEST_CASE(compact_query_push_returns_the_words_it_wrote) {
    std::mt19937_64 rng(0xB17E5ULL);
    for (size_t k = 0; k <= 40; ++k) {
        VecZ buf;
        const auto pos = make_positions(k, /*paired=*/(k % 2 == 0), rng);
        const size_t before = buf.size();
        const size_t written = CQ::push(buf, pos.data(), k, 1);
        BOOST_TEST(written == buf.size() - before);
        BOOST_TEST(written == CQ::words_for(k));

        // And again into a NON-empty buffer, since the accounting sums across a whole gate: a return
        // computed from buf.size() instead of from k would pass the first check and fail this one.
        const size_t before2 = buf.size();
        const size_t written2 = CQ::push(buf, pos.data(), k, -1);
        BOOST_TEST(written2 == buf.size() - before2);
        BOOST_TEST(written2 == written);
    }
    // push_mono goes through the dense monomial and must agree with the position-list form. Both a
    // narrow and a WIDE monomial: at k <= 6 the honest answer and the constant 2 coincide, so a narrow
    // case alone cannot tell a real count from kWordsPerRecord -- measured, that mutation survived.
    for (const size_t k : {size_t{5}, size_t{7}, size_t{20}}) {
        Monomial<kN> mono;
        size_t b = 0;
        while (mono.count() < k) {
            mono.set(b);
            b += 3; // keeps positions inside 2*kN for every k here and mixes paired with unpaired
        }
        BOOST_REQUIRE(mono.count() == k);
        VecZ buf;
        const size_t written = CQ::push_mono(buf, mono, 1);
        BOOST_TEST(written == buf.size());
        BOOST_TEST(written == CQ::words_for(k));
    }
    BOOST_TEST(CQ::words_for(20) > CQ::kWordsPerRecord); // the case that makes the check above bite
}

// The escape itself: a term wider than the inline slots must survive the wire unchanged.
BOOST_AUTO_TEST_CASE(compact_query_continuation_roundtrips_wide_terms) {
    std::mt19937_64 rng(0xABCDEFULL);
    size_t continued = 0;
    for (const size_t k :
         {size_t{7}, size_t{8}, size_t{13}, size_t{14}, size_t{15}, size_t{16}, size_t{24}, size_t{40}}) {
        const bool paired = (k % 2 == 0);
        const auto pos = make_positions(k, paired, rng);
        VecZ buf;
        CQ::push(buf, pos.data(), k, 1);
        BOOST_REQUIRE(buf.size() > CQ::kWordsPerRecord); // it really did continue
        ++continued;

        std::vector<PosT> out(k);
        const size_t next = CQ::read_positions(buf, 0, out.data());
        BOOST_TEST(next == buf.size());
        const bool positions_match = (out == pos);
        BOOST_TEST(positions_match);

        Monomial<kN> m;
        int ph = 0;
        CQ::read_mono(buf, 0, m, ph);
        const bool mono_matches = (m == mono_from(pos));
        BOOST_TEST(mono_matches);
        BOOST_TEST(m.count() == k);
    }
    // Guard against this case silently testing nothing if the inline width ever grows.
    BOOST_TEST(continued == 8U);
}

// k and the phase share one word with two positions; a shift wrong by 8 bits corrupts a neighbour.
BOOST_AUTO_TEST_CASE(compact_query_header_fields_do_not_alias) {
    std::mt19937_64 rng(0x1234ULL);
    for (const size_t k : {size_t{0}, size_t{1}, size_t{2}, size_t{4}, size_t{6}, size_t{8}, size_t{20}}) {
        const bool paired = (k % 2 == 0);
        const auto pos = make_positions(k, paired, rng);
        for (const int phase : {1, -1}) {
            VecZ buf;
            CQ::push(buf, pos.data(), k, phase);
            BOOST_TEST(CQ::k_at(buf, 0) == k);
            BOOST_TEST(CQ::phase_at(buf, 0) == phase);
        }
    }
    const auto paired8 = make_positions(8, /*paired=*/true, rng);
    VecZ buf;
    CQ::push(buf, paired8.data(), 8, -1);
    BOOST_TEST(CQ::k_at(buf, 0) == 8U);
    BOOST_TEST(CQ::phase_at(buf, 0) == -1); // sign extension through the uint8 slot
}

// k needs SIXTEEN bits, and this is the case that says why.
//
// The first cut of this format gave k one byte, reasoning that 255 positions is far above anything
// reaching this path. That is true for terms bounded by the cutoff -- and false for the ones that are
// not: a fully paired term is kept unconditionally, so at MAX_NUM_MODES = 1024 a paired term can carry
// k = 2048. With a one-byte k, 2048 truncates to 0 and the record decodes as the EMPTY monomial: a
// different, structurally valid term that inserts as a duplicate row and corrupts a coefficient with
// nothing anywhere reporting an error.
BOOST_AUTO_TEST_CASE(compact_query_k_survives_a_fully_paired_term_at_the_widest_width) {
    using Widest = CompactQuery<1024>; // MAX_NUM_MODES: 2N = 2048
    static_assert(Widest::kMaxPositions >= 2048, "k must hold a fully paired term at the widest width");

    std::vector<Widest::PosT> pos;
    for (size_t m = 0; m < 1024; ++m) { // every mode carrying both Majoranas
        pos.push_back(static_cast<Widest::PosT>(2 * m));
        pos.push_back(static_cast<Widest::PosT>((2 * m) + 1));
    }
    BOOST_REQUIRE(pos.size() == 2048U);

    VecZ buf;
    Widest::push(buf, pos.data(), pos.size(), 1);
    BOOST_TEST(Widest::k_at(buf, 0) == 2048U); // a one-byte k reads 0 here
    BOOST_TEST(Widest::words_for(2048) == buf.size());

    std::vector<Widest::PosT> out(pos.size());
    const size_t next = Widest::read_positions(buf, 0, out.data());
    BOOST_TEST(next == buf.size());
    const bool same = (out == pos);
    BOOST_TEST(same);

    Monomial<1024> m;
    int ph = 0;
    Widest::read_mono(buf, 0, m, ph);
    BOOST_TEST(m.count() == 2048U);
    BOOST_TEST(m.test(0));
    BOOST_TEST(m.test(2047));
    BOOST_TEST(ph == 1);
    // Fully paired means d == k/2, and d is recomputed rather than carried.
    BOOST_TEST(Widest::pair_count(out.data(), out.size()) == 1024U);
}

// d is no longer on the wire, so the thing that must be right is the recomputation.
BOOST_AUTO_TEST_CASE(compact_query_pair_count_recomputes_d_from_positions) {
    std::mt19937_64 rng(0xD1D1ULL);
    for (const size_t k : {size_t{0}, size_t{1}, size_t{2}, size_t{4}, size_t{6}, size_t{8}, size_t{16}}) {
        for (const bool paired : {false, true}) {
            if (paired && (k % 2 != 0)) {
                continue;
            }
            const auto pos = make_positions(k, paired, rng);
            // pair_count in this file is the independent reference: it is written against the
            // definition (mode m owns bits 2m, 2m+1), not against the codec's implementation.
            BOOST_TEST(CQ::pair_count(pos.data(), pos.size()) == pair_count(pos));
            if (paired) {
                BOOST_TEST(CQ::pair_count(pos.data(), pos.size()) == k / 2); // k == 2d
            }
        }
    }
    // An odd position followed by the next even one is NOT a pair -- (1,2) spans two modes.
    const std::vector<PosT> straddle{1, 2, 5, 6};
    BOOST_TEST(CQ::pair_count(straddle.data(), straddle.size()) == 0U);
    const std::vector<PosT> real{2, 3, 6, 7};
    BOOST_TEST(CQ::pair_count(real.data(), real.size()) == 2U);
}

// A stream, because the wire carries one per query and Resolve.h mints miss indices in record order --
// so order is not cosmetic here, it is the floating-point accumulation order.
BOOST_AUTO_TEST_CASE(compact_query_stream_preserves_order_and_content) {
    std::mt19937_64 rng(0xFEEDULL);
    std::vector<std::vector<PosT>> terms;
    std::vector<int> phases;
    VecZ buf;
    for (size_t t = 0; t < 400; ++t) {
        // Deliberately mixes inline and continued terms so a continuation sits between two short ones.
        const size_t k = (t % 37 == 0) ? 16U : ((t % 3 == 0) ? 4U : 6U);
        const auto pos = make_positions(k, /*paired=*/(k == 16), rng);
        const int phase = ((t % 2) == 0) ? 1 : -1;
        CQ::push(buf, pos.data(), k, phase);
        terms.push_back(pos);
        phases.push_back(phase);
    }
    BOOST_TEST(CQ::count_queries(buf) == terms.size());

    size_t rec = 0; // a WORD offset, not a record index
    size_t decoded = 0;
    for (size_t t = 0; t < terms.size(); ++t) {
        Monomial<kN> m;
        int ph = 0;
        rec = CQ::read_mono(buf, rec, m, ph);
        const bool same = (m == mono_from(terms[t]));
        BOOST_TEST(same);
        BOOST_TEST(ph == phases[t]);
        ++decoded;
    }
    BOOST_TEST(decoded == terms.size());
    BOOST_TEST(rec == buf.size());
}

// count_queries is what replaces buf.size() / stride once continuations exist. Getting it wrong by one
// misaligns every miss index on the receiving rank, so pin it against a stream whose answer is known.
BOOST_AUTO_TEST_CASE(compact_query_count_queries_is_not_a_division) {
    std::mt19937_64 rng(0x99ULL);
    VecZ buf;
    const auto a = make_positions(4, false, rng);
    const auto b = make_positions(16, true, rng); // 3 records
    const auto c = make_positions(6, false, rng);
    CQ::push(buf, a.data(), 4, 1);
    CQ::push(buf, b.data(), 16, 1);
    CQ::push(buf, c.data(), 6, 1);

    const size_t n_records = buf.size() / CQ::kWordsPerRecord;
    BOOST_TEST(n_records == 5U);              // 1 + 3 + 1
    BOOST_TEST(CQ::count_queries(buf) == 3U); // and NOT 5 -- the whole point
    BOOST_TEST(CQ::count_queries(VecZ{}) == 0U);
}

// Unused position slots are written as zero, not left indeterminate, so a wire dump is comparable
// between runs and two encodings of the same term are bit-identical.
BOOST_AUTO_TEST_CASE(compact_query_encoding_is_deterministic) {
    std::mt19937_64 rng(0x77ULL);
    for (const size_t k : {size_t{1}, size_t{4}, size_t{6}, size_t{9}}) {
        const auto pos = make_positions(k, /*paired=*/false, rng);
        VecZ x;
        VecZ y;
        CQ::push(x, pos.data(), k, 1);
        // A dirty buffer must not leak into the record.
        VecZ z{0xDEADBEEFULL, 0xDEADBEEFULL};
        CQ::push(y, pos.data(), k, 1);
        const bool identical = (x == y);
        BOOST_TEST(identical);
        z.clear();
        CQ::push(z, pos.data(), k, 1);
        const bool clean = (z == x);
        BOOST_TEST(clean);
    }
}

// The differential that matters: for every term the dense codec can carry, the compact one must deliver
// the identical (monomial, phase) to the identical consumer. This is the oracle a replacement must pass.
BOOST_AUTO_TEST_CASE(compact_query_agrees_with_the_dense_codec) {
    std::mt19937_64 rng(0x2026ULL);
    size_t checked = 0;
    VecZ dense;
    VecZ compact;
    std::vector<Monomial<kN>> expect;
    std::vector<int> expect_ph;
    for (size_t t = 0; t < 500; ++t) {
        const size_t k = 1 + (t % 12); // spans both sides of the inline width
        const auto pos = make_positions(k, /*paired=*/false, rng);
        const auto m = mono_from(pos);
        const int phase = ((t % 2) == 0) ? 1 : -1;

        test_ref::query_push<kN>(dense, m, phase);
        CQ::push(compact, pos.data(), k, phase);
        expect.push_back(m);
        expect_ph.push_back(phase);
    }
    // The dense codec's own round-trip is the reference, read exactly as Resolve.h reads it.
    size_t rec = 0;
    for (size_t q = 0; q < expect.size(); ++q) {
        Monomial<kN> dm;
        int dph = 0;
        test_ref::query_read<kN>(dense, q, dm, dph);
        Monomial<kN> cm;
        int cph = 0;
        rec = CQ::read_mono(compact, rec, cm, cph);
        const bool agree = (dm == cm);
        BOOST_TEST(agree);
        BOOST_TEST(dph == cph);
        const bool right = (dm == expect[q]);
        BOOST_TEST(right);
        BOOST_TEST(dph == expect_ph[q]);
        ++checked;
    }
    BOOST_TEST(checked == 500U);
    // And the reason for the exercise: the compact stream is materially smaller.
    BOOST_TEST(compact.size() < dense.size() / 3);
}

// The 16-bit lane packing, tested directly at the full range rather than through a monomial.
//
// This case exists because a mutation SURVIVED the rest of the file: narrowing pos_in's mask from
// 0xFFFF to 0x0FFF changed nothing, since the widest monomial anything else here builds is 2N = 1024
// and MAX_NUM_MODES = 1024 caps production at 2N = 2048 -- all comfortably under 4096. So every
// position-carrying test was silently exercising only the low 12 bits of a 16-bit field. The codec's
// own static_assert permits 2N up to 65536, so the lanes must be pinned over their declared range and
// not over the range today's widths happen to use.
BOOST_AUTO_TEST_CASE(compact_query_lanes_carry_the_full_sixteen_bits) {
    for (size_t j = 0; j < 4; ++j) {
        for (const PosT p : {PosT{0}, PosT{1}, PosT{0x0FFF}, PosT{0x1000}, PosT{0xABCD}, PosT{0xFFFF}}) {
            const uint64_t w = CQ::pack_pos(0, j, p);
            BOOST_TEST(CQ::pos_in(w, j) == p);
            // and it must not bleed into a neighbouring lane
            for (size_t o = 0; o < 4; ++o) {
                if (o != j) {
                    BOOST_TEST(CQ::pos_in(w, o) == PosT{0});
                }
            }
        }
    }
    // All four lanes loaded at once, so a shift that is right in isolation but overlaps still fails.
    uint64_t w = 0;
    const PosT vals[4] = {0xFFFF, 0x1234, 0x8000, 0x0001};
    for (size_t j = 0; j < 4; ++j) {
        w = CQ::pack_pos(w, j, vals[j]);
    }
    for (size_t j = 0; j < 4; ++j) {
        BOOST_TEST(CQ::pos_in(w, j) == vals[j]);
    }

    // And end to end at a width whose positions exceed 12 bits, which no other case reaches.
    using Wide = CompactQuery<4096>; // 2N = 8192
    BOOST_TEST(Wide::kBits == 8192U);
    // 8190/8191 is a complete mode pair, so d is 1 here, not 0 -- which check_header caught when this
    // case was first written with d=0. Kept deliberately: it exercises d on a wide record too.
    const std::vector<Wide::PosT> pos{0, 4095, 4096, 8190, 8191};
    VecZ buf;
    Wide::push(buf, pos.data(), pos.size(), 1);
    std::vector<Wide::PosT> out(pos.size());
    Wide::read_positions(buf, 0, out.data());
    const bool same = (out == pos);
    BOOST_TEST(same);
    Monomial<4096> m;
    int ph = 0;
    Wide::read_mono(buf, 0, m, ph);
    BOOST_TEST(m.test(8191));
    BOOST_TEST(m.test(4096));
    BOOST_TEST(m.count() == pos.size());
}

// check_header is the only thing making the continued flag verifiable: nothing in the wire change
// itself reads it, so without a check it is a write-only field that would ship a wrong value undetected
// until a later change started trusting it. Test it by corrupting each field in turn -- a validator that
// never returns false is not a validator.
BOOST_AUTO_TEST_CASE(compact_query_check_header_rejects_each_corrupted_field) {
    std::mt19937_64 rng(0xBADULL);
    const auto pos = make_positions(6, /*paired=*/false, rng);
    VecZ good;
    CQ::push(good, pos.data(), 6, 1);
    std::vector<PosT> out(6);
    CQ::read_positions(good, 0, out.data());
    BOOST_TEST(CQ::check_header(good, 0, out.data()));

    // A corrupted high byte of k. This is the byte d used to occupy, which is exactly why it is worth
    // testing: k is now 16 bits, so a stray bit here turns k = 6 into k = 262 rather than perturbing a
    // field nothing reads. The flag then disagrees with k, and that is what catches it.
    {
        VecZ bad = good;
        bad[1] ^= (uint64_t{1} << 40U);
        BOOST_TEST(CQ::k_at(bad, 0) == 262U);
        BOOST_TEST(!CQ::check_header(bad, 0, out.data()));
    }
    // continued flag set on a term that does not continue.
    {
        VecZ bad = good;
        bad[1] |= (static_cast<uint64_t>(CQ::kFlagContinued) << 56U);
        BOOST_TEST(!CQ::check_header(bad, 0, out.data()));
    }
    // an unknown flag bit: a format this reader does not understand must not be read as if it did.
    {
        VecZ bad = good;
        bad[1] |= (uint64_t{0x80} << 56U);
        BOOST_TEST(!CQ::check_header(bad, 0, out.data()));
    }
    // positions not ascending.
    {
        auto swapped = out;
        std::swap(swapped[0], swapped[1]);
        BOOST_TEST(!CQ::check_header(good, 0, swapped.data()));
    }
    // and a wide term whose flag was cleared.
    {
        const auto wide = make_positions(16, /*paired=*/true, rng);
        VecZ w;
        CQ::push(w, wide.data(), 16, 1);
        std::vector<PosT> wout(16);
        CQ::read_positions(w, 0, wout.data());
        BOOST_TEST(CQ::check_header(w, 0, wout.data()));
        w[1] &= ~(static_cast<uint64_t>(CQ::kFlagContinued) << 56U);
        BOOST_TEST(!CQ::check_header(w, 0, wout.data()));
    }
}

// Positions round-trip exactly, including the extremes of the physical bit range. indices_to_bitset
// sets bit 2N-1-j, so the top of the range is the one a width mistake would clip.
BOOST_AUTO_TEST_CASE(compact_query_carries_the_extreme_bit_positions) {
    const std::vector<PosT> pos{0,
                                1,
                                2,
                                static_cast<PosT>((2 * kN) - 3),
                                static_cast<PosT>((2 * kN) - 2),
                                static_cast<PosT>((2 * kN) - 1)};
    VecZ buf;
    CQ::push(buf, pos.data(), pos.size(), 1);
    std::vector<PosT> out(pos.size());
    CQ::read_positions(buf, 0, out.data());
    const bool same = (out == pos);
    BOOST_TEST(same);

    Monomial<kN> m;
    int ph = 0;
    CQ::read_mono(buf, 0, m, ph);
    BOOST_TEST(m.test(0));
    BOOST_TEST(m.test((2 * kN) - 1));
    BOOST_TEST(m.count() == pos.size());
    const bool via_positions = (positions_of(m) == pos);
    BOOST_TEST(via_positions);
}

// A width where 2N is not a multiple of 64, and a one-word width, since kWords changes but the compact
// record's stride must not.
BOOST_AUTO_TEST_CASE(compact_query_stride_is_independent_of_the_monomial_width) {
    BOOST_TEST(CompactQuery<32>::kWordsPerRecord == 2U);  // 2N = 64, kWords = 1
    BOOST_TEST(CompactQuery<250>::kWordsPerRecord == 2U); // 2N = 500, kWords = 8, not a multiple of 64
    BOOST_TEST(CompactQuery<512>::kWordsPerRecord == 2U); // 2N = 1024, kWords = 16
    // At one word the dense record is 2 words, so compact is a wash there and must not be a regression.
    BOOST_TEST(test_ref::kQueryWords<32> == 2U);

    std::mt19937_64 rng(0x31337ULL);
    (void)rng;
    Monomial<32> m;
    m.set(0);
    m.set(63);
    VecZ buf;
    const std::vector<CompactQuery<32>::PosT> pos{0, 63};
    CompactQuery<32>::push(buf, pos.data(), 2, -1);
    Monomial<32> back;
    int ph = 0;
    CompactQuery<32>::read_mono(buf, 0, back, ph);
    const bool same = (back == m);
    BOOST_TEST(same);
    BOOST_TEST(ph == -1);
}

// skip() is what keeps a cursor in step when a query is passed over -- resolve_range_ skips followers a
// leader already matched, and a cursor that does not advance there silently reads the wrong term for
// every query after the first skip. Pin it against the full decode.
BOOST_AUTO_TEST_CASE(compact_query_skip_advances_exactly_like_a_decode) {
    std::mt19937_64 rng(0x5C1D0ULL);
    VecZ buf;
    std::vector<size_t> ks;
    for (size_t t = 0; t < 80; ++t) {
        const size_t k = (t % 11 == 0) ? 20U : ((t % 3 == 0) ? 4U : 6U);
        const auto pos = make_positions(k, /*paired=*/(k == 20), rng);
        CQ::push(buf, pos.data(), k, 1);
        ks.push_back(k);
    }
    size_t skip_off = 0;
    size_t read_off = 0;
    for (size_t t = 0; t < ks.size(); ++t) {
        BOOST_REQUIRE(skip_off == read_off);
        skip_off = CQ::skip(buf, skip_off);
        Monomial<kN> m;
        int ph = 0;
        read_off = CQ::read_mono(buf, read_off, m, ph);
    }
    BOOST_TEST(skip_off == buf.size());
    BOOST_TEST(read_off == buf.size());
}

// The fused stream: query records followed by one value word. This is the reason offsets are counted in
// words -- a record index cannot name where the next query begins once a single word sits between them.
BOOST_AUTO_TEST_CASE(compact_query_fused_stream_interleaves_values_and_stays_walkable) {
    std::mt19937_64 rng(0xF05EDULL);
    VecZ buf;
    std::vector<std::vector<PosT>> terms;
    std::vector<double> vals;
    for (size_t t = 0; t < 120; ++t) {
        const size_t k = (t % 13 == 0) ? 16U : ((t % 2 == 0) ? 4U : 6U);
        const auto pos = make_positions(k, /*paired=*/(k == 16), rng);
        CQ::push(buf, pos.data(), k, (t % 2 == 0) ? 1 : -1);
        // A value that is not representable as anything but its exact bits, so a conversion would show.
        const double v = -1.0 / (static_cast<double>(t) + 3.0);
        CQ::push_value(buf, v);
        terms.push_back(pos);
        vals.push_back(v);
    }
    size_t off = 0;
    for (size_t t = 0; t < terms.size(); ++t) {
        Monomial<kN> m;
        int ph = 0;
        const size_t value_off = CQ::read_mono(buf, off, m, ph);
        const bool same = (m == mono_from(terms[t]));
        BOOST_TEST(same);
        BOOST_TEST(ph == ((t % 2 == 0) ? 1 : -1));
        // bit-identical, not merely close: the coefficient rides the wire as its own bits.
        const bool exact = (CQ::value_at(buf, value_off) == vals[t]);
        BOOST_TEST(exact);
        off = value_off + 1;
    }
    BOOST_TEST(off == buf.size());
    // Skipping must step over the value too, or a fused cursor desynchronises after the first query.
    size_t soff = 0;
    for (size_t t = 0; t < terms.size(); ++t) {
        soff = CQ::skip(buf, soff) + 1;
    }
    BOOST_TEST(soff == buf.size());
}
