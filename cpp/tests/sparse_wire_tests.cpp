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

// The support-form query record: a row must survive the wire exactly, and the record must keep the dense
// record's shape so the stride arithmetic, alltoallv counts and phase/value readers work unchanged.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

struct OwnedRow {
    std::vector<RowMode> lanes;
    RowCodes codes = 0;

    [[nodiscard]] auto view() const -> SparseRow { return SparseRow{lanes.data(), codes}; }
};

auto random_row(std::mt19937_64 &rng, size_t num_modes, size_t capacity) -> OwnedRow {
    OwnedRow row{std::vector<RowMode>(capacity, 0), 0};
    // Ascending distinct modes, which is what a real row is; a random unsorted list would not be one.
    std::vector<size_t> modes;
    for (size_t m = 0; m < num_modes && modes.size() < capacity; ++m) {
        if ((rng() % 4) == 0) {
            modes.push_back(m);
        }
    }
    for (size_t j = 0; j < modes.size(); ++j) {
        row.lanes[j] = static_cast<RowMode>(modes[j]);
        row.codes |= static_cast<RowCodes>(1U + (rng() % 3U)) << (2 * j);
    }
    return row;
}

} // namespace

// A record is lane words, then codes, then phase -- so the phase sits at offset `payload`, exactly where
// the dense reader expects it for a payload of that many words. That is the property that lets the record
// machinery stay untouched, so it is pinned rather than left implicit.
BOOST_AUTO_TEST_CASE(sparse_wire_record_keeps_the_dense_record_shape) {
    BOOST_TEST(sparse_lane_words(1U) == 1U);
    BOOST_TEST(sparse_lane_words(4U) == 1U);
    BOOST_TEST(sparse_lane_words(5U) == 2U);
    BOOST_TEST(sparse_lane_words(12U) == 3U);
    BOOST_TEST(sparse_lane_words(32U) == 8U);
    BOOST_TEST(sparse_payload_words(12U) == 4U);
    BOOST_TEST(query_words(sparse_payload_words(12U)) == 5U);
    BOOST_TEST(query_words_fused(sparse_payload_words(12U)) == 6U);

    // A 12-slot row rides 5 words where a 1024-mode monomial needs 33, and the two are equal at 128 modes
    // -- which is roughly where the crossover puts the sparse backend in a wheel build anyway.
    BOOST_TEST(query_words(sparse_payload_words(12U)) < query_words(1024U / 32U));
    BOOST_TEST(query_words(sparse_payload_words(12U)) == query_words(128U / 32U));

    constexpr size_t kCapacity = 12;
    VecZ buf;
    OwnedRow row{std::vector<RowMode>(kCapacity, 0), 0};
    row.lanes[0] = 7;
    row.codes = 0b11ULL;
    sparse_query_push(buf, row.view(), kCapacity, -1);
    BOOST_REQUIRE(buf.size() == query_words(sparse_payload_words(kCapacity)));
    // query_phase reads the phase off the payload width alone, with no idea what the payload holds.
    BOOST_TEST(query_phase(buf, 0, sparse_payload_words(kCapacity)) == -1);
}

BOOST_AUTO_TEST_CASE(sparse_wire_round_trips_rows_exactly) {
    std::mt19937_64 rng(20260812U);
    size_t full_rows = 0;
    size_t empty_rows = 0;
    for (const size_t capacity : {1U, 4U, 5U, 12U, 32U}) {
        for (const size_t num_modes : {32U, 64U, 1024U, 32000U}) {
            const size_t stride = query_words(sparse_payload_words(capacity));
            VecZ buf;
            std::vector<OwnedRow> rows;
            for (size_t t = 0; t < 40; ++t) {
                // Every fifth record is the empty row: at these mode counts the random generator would
                // essentially never produce one, and an empty row is the record whose lanes are all
                // padding.
                rows.push_back((t % 5) == 0 ? OwnedRow{std::vector<RowMode>(capacity, 0), 0}
                                            : random_row(rng, num_modes, capacity));
                // Phases are +-1 on the real path; both must survive the unsigned round-trip.
                sparse_query_push(buf, rows.back().view(), capacity, (t % 2) == 0 ? 1 : -1);
            }
            BOOST_REQUIRE(buf.size() == rows.size() * stride);

            for (size_t q = 0; q < rows.size(); ++q) {
                std::vector<RowMode> lanes(capacity, 0xEEEE); // poisoned, so an unwritten lane shows up
                RowCodes codes = 0;
                int phase = 0;
                sparse_query_read(buf, q, stride, capacity, lanes.data(), codes, phase);
                BOOST_TEST(codes == rows[q].codes);
                BOOST_TEST(phase == ((q % 2) == 0 ? 1 : -1));
                const size_t n = row_slot_count(codes);
                BOOST_REQUIRE(n == rows[q].view().num_slots());
                for (size_t j = 0; j < n; ++j) {
                    BOOST_TEST(lanes[j] == rows[q].lanes[j]);
                }
                full_rows += n == capacity ? 1 : 0;
                empty_rows += n == 0 ? 1 : 0;
            }
        }
    }
    // Both edges of the capacity have to have occurred, or the packing was never pushed to its bounds.
    BOOST_TEST(full_rows > 0U);
    BOOST_TEST(empty_rows > 0U);
}

// Mode indices are packed four to a word, so a lane must not bleed into its neighbours. The largest mode
// a store admits is kMaxModes - 1, which is also the widest lane value.
BOOST_AUTO_TEST_CASE(sparse_wire_packs_lanes_without_bleeding) {
    constexpr size_t kCapacity = 8; // two lane words, so the boundary between them is exercised
    const size_t stride = query_words(sparse_payload_words(kCapacity));
    const auto top = static_cast<RowMode>(SparseRowStore::kMaxModes - 1);

    OwnedRow row{std::vector<RowMode>(kCapacity, 0), 0};
    // Ascending, and straddling the 4-lane word boundary with extreme values on both sides of it.
    const std::vector<RowMode> modes{0, 1, top - 2, top - 1, top, 0, 0, 0};
    for (size_t j = 0; j < 5; ++j) {
        row.lanes[j] = modes[j];
        row.codes |= RowCodes{0b11} << (2 * j);
    }

    VecZ buf;
    sparse_query_push(buf, row.view(), kCapacity, 1);
    std::vector<RowMode> lanes(kCapacity, 0);
    RowCodes codes = 0;
    int phase = 0;
    sparse_query_read(buf, 0, stride, kCapacity, lanes.data(), codes, phase);
    BOOST_TEST(codes == row.codes);
    for (size_t j = 0; j < 5; ++j) {
        BOOST_TEST(lanes[j] == modes[j]);
    }
}

// Records are read by position out of one flat buffer, so a short row must still occupy a full stride and
// must not be able to see the previous record's lanes.
BOOST_AUTO_TEST_CASE(sparse_wire_short_rows_keep_the_full_stride) {
    constexpr size_t kCapacity = 12;
    const size_t stride = query_words(sparse_payload_words(kCapacity));
    VecZ buf;

    OwnedRow full{std::vector<RowMode>(kCapacity, 0), 0};
    for (size_t j = 0; j < kCapacity; ++j) {
        full.lanes[j] = static_cast<RowMode>(100 + j);
        full.codes |= RowCodes{0b11} << (2 * j);
    }
    OwnedRow empty{std::vector<RowMode>(kCapacity, 0), 0};

    sparse_query_push(buf, full.view(), kCapacity, 1);
    sparse_query_push(buf, empty.view(), kCapacity, -1);
    BOOST_REQUIRE(buf.size() == 2 * stride);

    std::vector<RowMode> lanes(kCapacity, 0xEEEE);
    RowCodes codes = 0xDEAD;
    int phase = 0;
    sparse_query_read(buf, 1, stride, kCapacity, lanes.data(), codes, phase);
    BOOST_TEST(codes == 0U);
    BOOST_TEST(phase == -1);
    BOOST_TEST(row_slot_count(codes) == 0U);
    // Nothing was written into lanes, so the poison is still there -- the empty row cannot have inherited
    // the previous record's modes.
    BOOST_TEST(lanes[0] == 0xEEEE);
}

// --- the query key batches -------------------------------------------------------------------------
//
// Both resolve paths fill one of these from wire records and hand it to find_batch contiguously. They are
// grow-only and reused across layers, which is the shape the measurements asked for, and that reuse is
// where the two ways to get it wrong live: a stale element read before being overwritten, and -- for the
// sparse batch, whose keys are views -- a view left pointing into storage that growth moved.

static_assert(std::is_same_v<QueryKeysFor<OperatorIndex>::type, DenseQueryKeys>);
static_assert(std::is_same_v<QueryKeysFor<SparseRowStore>::type, SparseQueryKeys>);

BOOST_AUTO_TEST_CASE(query_keys_dense_batch_round_trips_records) {
    constexpr size_t kNumBits = 128;
    const size_t stride = query_words(kNumBits / 64);
    VecZ buf;
    std::vector<Bitset> monos;
    for (size_t t = 0; t < 40; ++t) {
        Bitset mono(kNumBits);
        mono.set(t);
        mono.set(kNumBits - 1 - t);
        monos.push_back(mono);
        query_push(buf, mono, (t % 2) == 0 ? 1 : -1);
    }

    DenseQueryKeys keys;
    keys.configure(kNumBits);
    keys.ensure(monos.size());
    for (size_t q = 0; q < monos.size(); ++q) {
        BOOST_TEST(keys.read_record(buf, q, stride, q) == ((q % 2) == 0 ? 1 : -1));
    }
    for (size_t q = 0; q < monos.size(); ++q) {
        BOOST_TEST((keys[q] == monos[q]));
        BOOST_TEST((keys.data()[q] == monos[q]));
    }
}

BOOST_AUTO_TEST_CASE(query_keys_sparse_batch_survives_growth) {
    constexpr size_t kCapacity = 12;
    const size_t stride = query_words(sparse_payload_words(kCapacity));
    std::mt19937_64 rng(20260812U);
    VecZ buf;
    std::vector<OwnedRow> rows;
    for (size_t t = 0; t < 64; ++t) {
        rows.push_back(random_row(rng, 1024, kCapacity));
        sparse_query_push(buf, rows.back().view(), kCapacity, (t % 2) == 0 ? 1 : -1);
    }

    SparseQueryKeys keys;
    keys.configure(kCapacity);

    // Fill a small prefix, then grow: the lane array reallocates, so every view has to be rebuilt. If only
    // the new tail were, the prefix read back below would be reading freed storage.
    keys.ensure(8);
    for (size_t q = 0; q < 8; ++q) {
        BOOST_TEST(keys.read_record(buf, q, stride, q) == ((q % 2) == 0 ? 1 : -1));
    }
    keys.ensure(rows.size());
    // Checked as a pointer invariant rather than by reading the prefix and hoping it looks wrong: a view
    // left over from before the growth points into freed storage, which is undefined behaviour and might
    // read back plausibly. Every view must address this batch's current lane array at its own stride.
    for (size_t q = 0; q < rows.size(); ++q) {
        BOOST_TEST(keys.data()[q].modes == keys.data()[0].modes + (q * kCapacity));
    }
    for (size_t q = 8; q < rows.size(); ++q) {
        BOOST_TEST(keys.read_record(buf, q, stride, q) == ((q % 2) == 0 ? 1 : -1));
    }

    for (size_t q = 0; q < 8; ++q) {
        // Re-read the prefix records so the prefix slots are written again after the growth; what is being
        // checked is that the view still addresses this batch's own lanes.
        BOOST_TEST(keys.read_record(buf, q, stride, q) == ((q % 2) == 0 ? 1 : -1));
    }
    for (size_t q = 0; q < rows.size(); ++q) {
        const auto &key = keys.data()[q];
        BOOST_TEST(key.codes == rows[q].codes);
        const size_t n = key.num_slots();
        BOOST_REQUIRE(n == rows[q].view().num_slots());
        for (size_t j = 0; j < n; ++j) {
            BOOST_TEST(key.mode(j) == rows[q].view().mode(j));
        }
    }
}

// configure() with a different extent must drop the storage: a thread servicing two propagators of
// different widths would otherwise write a wide record into a narrow element.
BOOST_AUTO_TEST_CASE(query_keys_reconfigure_resizes_the_elements) {
    DenseQueryKeys dense;
    dense.configure(64);
    dense.ensure(4);
    BOOST_TEST(dense[0].size() == 64U);
    dense.configure(256);
    dense.ensure(4);
    BOOST_TEST(dense[0].size() == 256U);

    SparseQueryKeys sparse;
    sparse.configure(4);
    sparse.ensure(4);
    const auto *narrow_base = sparse.data()[0].modes;
    const auto *narrow_next = sparse.data()[1].modes;
    BOOST_TEST(narrow_next - narrow_base == 4);
    sparse.configure(12);
    sparse.ensure(4);
    BOOST_TEST(sparse.data()[1].modes - sparse.data()[0].modes == 12);
}
