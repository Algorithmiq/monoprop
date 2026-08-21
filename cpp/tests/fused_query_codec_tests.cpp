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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
using monoprop::detail::kQueryWords;
using monoprop::detail::kQueryWordsFused;
using monoprop::detail::query_push;
using monoprop::detail::query_read;
using monoprop::detail::query_value;

constexpr size_t kModes = 8; // 2*kModes = 16 majorana bits, one 64-bit word

// A deterministic, distinct majorana bit pattern per record index.
auto make_mono(size_t r) -> Monomial<kModes> {
    Monomial<kModes> m;
    for (size_t b = 0; b < 2 * kModes; ++b) {
        if (((r * 2654435761u + b * 40503u) & 3u) == 0u) {
            m.set(b);
        }
    }
    return m;
}

TEST_CASE("fused_record_roundtrip_exact") {
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

    VecZ plain;
    std::vector<Monomial<kModes>> monos(nq);
    for (size_t r = 0; r < nq; ++r) {
        monos[r] = make_mono(r);
        query_push<kModes>(plain, monos[r], phases[r]);
    }
    REQUIRE((plain.size()) == (nq * kQueryWords<kModes>));

    VecZ fused;
    build_fused_query_value<kModes>(plain, values, fused);
    REQUIRE((fused.size()) == (nq * kQueryWordsFused<kModes>));

    for (size_t q = 0; q < nq; ++q) {
        Monomial<kModes> m_out;
        int ph_out = 0;
        query_read<kModes, kQueryWordsFused<kModes>>(fused, q, m_out, ph_out);
        CHECK(m_out == monos[q]);
        CHECK((ph_out) == (phases[q]));
        // Compare the raw payload, so -0.0 and denormals stay distinguished from 0.0.
        const double v_out = query_value<kModes>(fused, q);
        CHECK(std::memcmp(&v_out, &values[q], sizeof(double)) == 0);
    }
}

// Reusing `out` across calls must leak no stale words: capacity is a high-water mark, size is exact.
// This is the reuse pattern LayerBuildEngine::combined_qv_ relies on gate to gate.
TEST_CASE("fused_buffer_reuse_shrinks_logical_size") {
    VecZ plain_big;
    std::vector<double> vbig;
    for (size_t r = 0; r < 32; ++r) {
        query_push<kModes>(plain_big, make_mono(r), (r % 2 == 0) ? 1 : -1);
        vbig.push_back(static_cast<double>(r) * 1.5 - 7.0);
    }
    VecZ out;
    build_fused_query_value<kModes>(plain_big, vbig, out);
    const size_t cap_after_big = out.capacity();

    VecZ plain_small;
    std::vector<double> vsmall = {42.0, -42.0, 0.25};
    for (size_t r = 0; r < vsmall.size(); ++r) {
        query_push<kModes>(plain_small, make_mono(100 + r), 1);
    }
    build_fused_query_value<kModes>(plain_small, vsmall, out);
    CHECK((out.size()) == (vsmall.size() * kQueryWordsFused<kModes>));
    CHECK((out.capacity()) >= (cap_after_big));
    for (size_t q = 0; q < vsmall.size(); ++q) {
        const double v_out = query_value<kModes>(out, q);
        CHECK(std::memcmp(&v_out, &vsmall[q], sizeof(double)) == 0);
    }
}

// Empty input arises for the self slot, which resolve_self_queries clears before the exchange.
TEST_CASE("fused_empty_input") {
    VecZ empty;
    std::vector<double> no_values;
    VecZ out{1, 2, 3}; // pre-dirtied; build must clear it
    build_fused_query_value<kModes>(empty, no_values, out);
    CHECK(out.empty());
}

} // namespace
