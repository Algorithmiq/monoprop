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

#include <cstring>
#include <limits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"

// Query+value fusion codec: the fused R>1 exchange rides the source coefficient (v_src) on each query
// record as a trailing bit-cast word so one alltoallv carries both streams. Round-tripping must be
// byte-for-byte, including the FP corner cases a lossy value channel would mangle.

namespace {

using namespace monoprop;
using monoprop::detail::build_fused_query_value;
using monoprop::detail::kQueryHeaderWords;
using monoprop::detail::query_buffer;
using monoprop::detail::query_push;
using monoprop::detail::query_read;
using monoprop::detail::query_record_count;
using monoprop::detail::query_value;
using monoprop::detail::query_words;
using monoprop::detail::query_words_fused;

constexpr size_t kModes = 8;         // 2*kModes = 16 majorana bits, one 64-bit word
constexpr size_t kBits = 2 * kModes; // a monomial's width is data, so every one here is built at it
// The wire record width is in words now, not derived from a NumModes template argument.
constexpr size_t kWordsPerMono = (kBits + 63) / 64;

// query_value takes the word count; wrap it so the assertions below stay readable.
auto query_value_at(const VecZ &buf, size_t q) -> double {
    return query_value(buf, q, kWordsPerMono);
}

// A deterministic, distinct majorana bit pattern per record index.
auto make_mono(size_t r) -> Bitset {
    Bitset m(kBits);
    for (size_t b = 0; b < kBits; ++b) {
        if (((r * 2654435761u + b * 40503u) & 3u) == 0u) {
            m.set(b);
        }
    }
    return m;
}

BOOST_AUTO_TEST_CASE(fused_record_roundtrip_exact) {
    const std::vector<int> phases = {1, -1, 1, -1, 1, 1, -1};
    const std::vector<double> values = {
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.141592653589793,
        -2.718281828459045e-300,            // near-denormal magnitude
        std::numeric_limits<double>::min(), // smallest normal
    };
    const size_t nq = values.size();

    VecZ plain = query_buffer();
    // Sized up front and assigned into, so the fill value has to carry the width.
    MonomialList monos(nq, Bitset(kBits));
    for (size_t r = 0; r < nq; ++r) {
        monos[r] = make_mono(r);
        query_push(plain, monos[r], phases[r]);
    }
    BOOST_REQUIRE_EQUAL(plain.size(), kQueryHeaderWords + nq * query_words(kWordsPerMono));
    BOOST_REQUIRE_EQUAL(query_record_count(plain), nq);

    VecZ fused;
    build_fused_query_value(plain, values, fused, kWordsPerMono);
    BOOST_REQUIRE_EQUAL(fused.size(), kQueryHeaderWords + nq * query_words_fused(kWordsPerMono));
    // The count survives the re-layout: it is what the resolver reads records off, at either stride.
    BOOST_REQUIRE_EQUAL(query_record_count(fused), nq);

    for (size_t q = 0; q < nq; ++q) {
        Bitset m_out(kBits); // pre-sized: the reader copies mono_out.num_words() words into it
        int ph_out = 0;
        query_read(fused, q, query_words_fused(kWordsPerMono), m_out, ph_out);
        BOOST_CHECK(m_out == monos[q]);
        BOOST_CHECK_EQUAL(ph_out, phases[q]);
        // Compare the raw payload, so -0.0 and denormals stay distinguished from 0.0.
        const double v_out = query_value_at(fused, q);
        BOOST_CHECK(std::memcmp(&v_out, &values[q], sizeof(double)) == 0);
    }
}

// Reusing `out` across calls must leak no stale words: capacity is a high-water mark, size is exact.
// This is the reuse pattern LayerBuildEngine::combined_qv_ relies on gate to gate.
BOOST_AUTO_TEST_CASE(fused_buffer_reuse_shrinks_logical_size) {
    VecZ plain_big = query_buffer();
    std::vector<double> vbig;
    for (size_t r = 0; r < 32; ++r) {
        query_push(plain_big, make_mono(r), (r % 2 == 0) ? 1 : -1);
        vbig.push_back(static_cast<double>(r) * 1.5 - 7.0);
    }
    VecZ out;
    build_fused_query_value(plain_big, vbig, out, kWordsPerMono);
    const size_t cap_after_big = out.capacity();

    VecZ plain_small = query_buffer();
    std::vector<double> vsmall = {42.0, -42.0, 0.25};
    for (size_t r = 0; r < vsmall.size(); ++r) {
        query_push(plain_small, make_mono(100 + r), 1);
    }
    build_fused_query_value(plain_small, vsmall, out, kWordsPerMono);
    BOOST_CHECK_EQUAL(out.size(), kQueryHeaderWords + vsmall.size() * query_words_fused(kWordsPerMono));
    BOOST_CHECK_GE(out.capacity(), cap_after_big);
    for (size_t q = 0; q < vsmall.size(); ++q) {
        const double v_out = query_value_at(out, q);
        BOOST_CHECK(std::memcmp(&v_out, &vsmall[q], sizeof(double)) == 0);
    }
}

// Empty input arises for the self slot, which resolve_self_queries clears before the exchange -- not even a
// header, which is why an under-length buffer has to read as no stream rather than as a malformed one.
BOOST_AUTO_TEST_CASE(fused_empty_input) {
    VecZ empty;
    std::vector<double> no_values;
    VecZ out{1, 2, 3}; // pre-dirtied; build must clear it
    build_fused_query_value(empty, no_values, out, kWordsPerMono);
    BOOST_CHECK(out.empty());

    // A stream that was created but never pushed to is a header alone, and re-lays out as one.
    VecZ header_only = query_buffer();
    build_fused_query_value(header_only, no_values, out, kWordsPerMono);
    BOOST_CHECK_EQUAL(out.size(), kQueryHeaderWords);
    BOOST_CHECK_EQUAL(query_record_count(out), 0U);
}

} // namespace
