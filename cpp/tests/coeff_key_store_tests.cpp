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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/CoeffKeyStore.h"

using namespace monoprop;
using detail::CoeffKeyStore;
using detail::CoeffSpan;
using detail::MutCoeffSpan;

// The whole point of the container: one cell carries both fields, and 12 B is exactly what the two
// separate arrays cost, so nothing about B/term changes when the key moves next to the coefficient.
BOOST_AUTO_TEST_CASE(coeff_key_store_record_layout_and_stride) {
    BOOST_TEST(CoeffKeyStore::kKeyOnlyStride == 4U);
    BOOST_TEST(CoeffKeyStore::kPackedStride == 12U);
    BOOST_TEST(CoeffKeyStore::kPackedStride == CoeffKeyStore::kCoeffBytes + CoeffKeyStore::kKeyBytes);

    CoeffKeyStore s;
    BOOST_TEST(!s.has_coeffs());
    BOOST_TEST(s.stride() == CoeffKeyStore::kKeyOnlyStride);
    s.resize(4);
    BOOST_TEST(s.size() == 4U);
    s.enable_coeffs();
    BOOST_TEST(s.has_coeffs());
    BOOST_TEST(s.stride() == CoeffKeyStore::kPackedStride);

    // The cells really are adjacent: cell i's key sits 8 bytes past its coefficient, and consecutive
    // coefficients are 12 bytes apart.
    for (size_t i = 0; i < s.size(); ++i) {
        s.set_coeff(i, 1.0 + static_cast<double>(i));
        s.set_key(i, static_cast<uint32_t>(0xA0000000U + i));
    }
    const auto span = s.coeff_span();
    const auto keys = s.key_reader();
    BOOST_TEST(span.stride == CoeffKeyStore::kPackedStride);
    BOOST_TEST(keys.stride == CoeffKeyStore::kPackedStride);
    BOOST_TEST(keys.base == span.base + CoeffKeyStore::kCoeffBytes);
    for (size_t i = 0; i < s.size(); ++i) {
        BOOST_TEST(span[i] == 1.0 + static_cast<double>(i));
        BOOST_TEST(keys(i) == 0xA0000000U + i);
        BOOST_TEST(s.coeff(i) == span[i]);
        BOOST_TEST(s.key(i) == keys(i));
    }
}

// A keys-only store is exactly the 4 B/term array it replaces; promotion keeps every key and starts the
// coefficients at 0.0, which is what the lazy op_coeffs.resize(size(), 0.0) used to produce.
BOOST_AUTO_TEST_CASE(coeff_key_store_enable_coeffs_preserves_keys_and_zeroes_coeffs) {
    CoeffKeyStore s;
    s.resize(5);
    BOOST_TEST(s.stride() == CoeffKeyStore::kKeyOnlyStride);
    for (size_t i = 0; i < s.size(); ++i) {
        s.set_key(i, static_cast<uint32_t>((i * 2654435761U) ^ 0x5A5AU));
    }
    std::vector<uint32_t> before;
    for (size_t i = 0; i < s.size(); ++i) {
        before.push_back(s.key(i));
    }
    BOOST_TEST(s.key_bytes() >= s.size() * sizeof(uint32_t));
    BOOST_TEST(s.coeff_bytes() == 0U);

    s.enable_coeffs();
    for (size_t i = 0; i < s.size(); ++i) {
        BOOST_TEST(s.key(i) == before[i]);
        BOOST_TEST(s.coeff(i) == 0.0);
    }
    BOOST_TEST(s.coeff_bytes() >= s.size() * sizeof(double));
    // Promotion is idempotent and one-way.
    s.set_coeff(2, 7.5);
    s.enable_coeffs();
    BOOST_TEST(s.coeff(2) == 7.5);
    BOOST_TEST(s.key(2) == before[2]);
}

// Growth keeps the cells already written, gives a fresh cell a zero coefficient (the contract apply's
// insert arm reads a minted slot before writing it) and leaves its key to the writer.
BOOST_AUTO_TEST_CASE(coeff_key_store_growth_keeps_cells_and_zeroes_new_coeffs) {
    CoeffKeyStore s;
    s.enable_coeffs();
    s.resize(3);
    for (size_t i = 0; i < 3; ++i) {
        s.set_coeff(i, 10.0 + static_cast<double>(i));
        s.set_key(i, static_cast<uint32_t>(i + 1));
    }
    s.reserve(64);
    s.resize(9);
    BOOST_TEST(s.size() == 9U);
    BOOST_TEST(s.capacity() >= 64U);
    for (size_t i = 0; i < 3; ++i) {
        BOOST_TEST(s.coeff(i) == 10.0 + static_cast<double>(i));
        BOOST_TEST(s.key(i) == i + 1);
    }
    for (size_t i = 3; i < 9; ++i) {
        BOOST_TEST(s.coeff(i) == 0.0);
    }
    // The seam the slack accounting hangs on: unused capacity, in bytes of the live stride.
    BOOST_TEST(s.slack_bytes() == (s.capacity() - s.size()) * s.stride());
    // Shrinking below the written prefix keeps it.
    s.resize(2);
    BOOST_TEST(s.size() == 2U);
    BOOST_TEST(s.coeff(1) == 11.0);
    BOOST_TEST(s.key(1) == 2U);
}

// A copy is a value copy of both columns -- OperatorIndex::clone() leans on it.
BOOST_AUTO_TEST_CASE(coeff_key_store_copy_round_trips_both_columns) {
    CoeffKeyStore s;
    s.resize(4);
    s.enable_coeffs();
    const VecD vals = {-1.25, 0.0, 3.5, 1e-17};
    s.assign_coeffs(vals);
    for (size_t i = 0; i < s.size(); ++i) {
        s.set_key(i, static_cast<uint32_t>(0xFFFF0000U | i));
    }

    const CoeffKeyStore copy = s;
    BOOST_TEST(copy.size() == s.size());
    BOOST_TEST(copy.has_coeffs());
    BOOST_TEST(copy.stride() == s.stride());
    for (size_t i = 0; i < copy.size(); ++i) {
        BOOST_TEST(copy.coeff(i) == vals[i]);
        BOOST_TEST(copy.key(i) == (0xFFFF0000U | i));
    }
    BOOST_TEST(copy.coeffs_to_vector() == vals, boost::test_tools::per_element());

    // A shorter assignment zeroes the tail rather than leaving a stale coefficient behind.
    CoeffKeyStore t = s;
    t.assign_coeffs(VecD{9.0});
    BOOST_TEST(t.coeff(0) == 9.0);
    for (size_t i = 1; i < t.size(); ++i) {
        BOOST_TEST(t.coeff(i) == 0.0);
    }
    for (size_t i = 0; i < t.size(); ++i) {
        BOOST_TEST(t.key(i) == (0xFFFF0000U | i));
    }
}

// The two spans are how the scan and the apply address either layout; a plain vector must read back
// through them exactly as the packed cells do.
BOOST_AUTO_TEST_CASE(coeff_key_store_spans_address_both_layouts) {
    VecD plain = {1.0, 2.0, 3.0};
    const CoeffSpan flat(plain);
    BOOST_TEST(flat.stride == sizeof(double));
    BOOST_TEST(flat.size() == 3U);
    BOOST_TEST(flat[2] == 3.0);
    BOOST_TEST(flat.at_or_zero(3) == 0.0);

    MutCoeffSpan mflat(plain);
    mflat.set(1, -4.0);
    BOOST_TEST(plain[1] == -4.0);
    BOOST_TEST(mflat.as_const()[1] == -4.0);

    CoeffKeyStore s;
    s.resize(3);
    s.assign_coeffs(plain);
    const auto packed = s.mut_coeff_span();
    BOOST_TEST(packed.stride == CoeffKeyStore::kPackedStride);
    for (size_t i = 0; i < 3; ++i) {
        BOOST_TEST(packed[i] == plain[i]);
    }
    // Writing through the packed span must not disturb the key in the same cell.
    s.set_key(1, 0xDEADBEEFU);
    packed.set(1, 99.0);
    BOOST_TEST(s.coeff(1) == 99.0);
    BOOST_TEST(s.key(1) == 0xDEADBEEFU);

    // An empty span is the "no coefficients" signal the scan tests for.
    const CoeffSpan none;
    BOOST_TEST(none.empty());
    BOOST_TEST(none.base == nullptr);
    BOOST_TEST(CoeffKeyStore{}.coeff_span().base == nullptr);
}
