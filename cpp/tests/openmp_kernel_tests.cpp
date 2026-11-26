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

// The threaded cosine kernel against its one-thread execution and a hand oracle: serial and threaded runs must agree
// bitwise.

#include <boost/test/unit_test.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "KernelTestSupport.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/parallel/Options.h"

namespace {

using namespace monoprop;
using kernel_test::bitwise_equal;
using kernel_test::distinct_values;
using monoprop::detail::parallel::Options;

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

} // namespace

// Every budget against the one-thread kernel, and the one-thread kernel against a hand oracle that multiplies exactly
// the selected coefficients and leaves the rest untouched.
BOOST_AUTO_TEST_CASE(openmp_cos_mask_exact) {
    constexpr double cos_val = 0.625;
    const CosMask mask = plan_mask();
    const size_t n = 4097 * 64;
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
    BOOST_TEST(bitwise_equal(serial, oracle));
    // 64 is more workers than ranges.
    for (const int threads : {2, 3, 4, 64}) {
        VecD threaded = original;
        monoprop::detail::scale_cos_mask(threaded.data(), mask, cos_val, Options{.threads = threads});
        BOOST_TEST_INFO("threads " << threads);
        BOOST_TEST(bitwise_equal(threaded, serial));
    }
}
