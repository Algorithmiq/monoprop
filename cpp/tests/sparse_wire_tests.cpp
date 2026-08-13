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
//
// Plus the two things that shape carries with it -- the header word every buffer opens with, and the tail a
// query too wide for any fixed-stride sparse record escapes to.

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

// The four batch/codec cases below key their assertions off a per-record shape, so a default-constructed
// placeholder stands in for the escaped records' absent row.
static_assert(std::is_default_constructible_v<OwnedRow>);

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
    VecZ buf = query_buffer();
    OwnedRow row{std::vector<RowMode>(kCapacity, 0), 0};
    row.lanes[0] = 7;
    row.codes = 0b11ULL;
    sparse_query_push(buf, row.view(), kCapacity, -1);
    BOOST_REQUIRE(buf.size() == kQueryHeaderWords + query_words(sparse_payload_words(kCapacity)));
    BOOST_TEST(query_record_count(buf) == 1U);
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
            VecZ buf = query_buffer();
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
            BOOST_REQUIRE(buf.size() == kQueryHeaderWords + (rows.size() * stride));
            BOOST_REQUIRE(query_record_count(buf) == rows.size());

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

    VecZ buf = query_buffer();
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
    VecZ buf = query_buffer();

    OwnedRow full{std::vector<RowMode>(kCapacity, 0), 0};
    for (size_t j = 0; j < kCapacity; ++j) {
        full.lanes[j] = static_cast<RowMode>(100 + j);
        full.codes |= RowCodes{0b11} << (2 * j);
    }
    OwnedRow empty{std::vector<RowMode>(kCapacity, 0), 0};

    sparse_query_push(buf, full.view(), kCapacity, 1);
    sparse_query_push(buf, empty.view(), kCapacity, -1);
    BOOST_REQUIRE(buf.size() == kQueryHeaderWords + (2 * stride));

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

// Both backends key their records densely for now: the query record is still the monomial's words even
// when the rows are sparse (see SparseTermProducts::dense_row), and a SparseRowStore finds a Bitset key
// as readily as a row key. SparseQueryKeys is what the sparse specialization becomes when the record
// swaps, which is why it is still tested below.
static_assert(std::is_same_v<QueryKeysFor<OperatorIndex>::type, DenseQueryKeys>);
static_assert(std::is_same_v<QueryKeysFor<SparseRowStore>::type, DenseQueryKeys>);

BOOST_AUTO_TEST_CASE(query_keys_dense_batch_round_trips_records) {
    constexpr size_t kNumBits = 128;
    const size_t stride = query_words(kNumBits / 64);
    VecZ buf = query_buffer();
    std::vector<Bitset> monos;
    for (size_t t = 0; t < 40; ++t) {
        Bitset mono(kNumBits);
        mono.set(t);
        mono.set(kNumBits - 1 - t);
        monos.push_back(mono);
        query_push(buf, mono, (t % 2) == 0 ? 1 : -1);
    }

    BOOST_TEST(query_record_count(buf) == monos.size());
    DenseQueryKeys keys;
    keys.configure(kNumBits, /*capacity=*/0);
    keys.ensure(monos.size());
    keys.begin_batch();
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
    VecZ buf = query_buffer();
    std::vector<OwnedRow> rows;
    for (size_t t = 0; t < 64; ++t) {
        rows.push_back(random_row(rng, 1024, kCapacity));
        sparse_query_push(buf, rows.back().view(), kCapacity, (t % 2) == 0 ? 1 : -1);
    }

    SparseQueryKeys keys;
    keys.configure(/*num_bits=*/2048, kCapacity);

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
        BOOST_TEST(keys.data()[q].row.modes == keys.data()[0].row.modes + (q * kCapacity));
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
        BOOST_REQUIRE(!key.is_spilled());
        BOOST_TEST(key.row.codes == rows[q].codes);
        const size_t n = key.row.num_slots();
        BOOST_REQUIRE(n == rows[q].view().num_slots());
        for (size_t j = 0; j < n; ++j) {
            BOOST_TEST(key.row.mode(j) == rows[q].view().mode(j));
        }
    }
}

// configure() with a different extent must drop the storage: a thread servicing two propagators of
// different widths would otherwise write a wide record into a narrow element.
BOOST_AUTO_TEST_CASE(query_keys_reconfigure_resizes_the_elements) {
    DenseQueryKeys dense;
    dense.configure(64, 0);
    dense.ensure(4);
    BOOST_TEST(dense[0].size() == 64U);
    dense.configure(256, 0);
    dense.ensure(4);
    BOOST_TEST(dense[0].size() == 256U);

    SparseQueryKeys sparse;
    sparse.configure(64, 4);
    sparse.ensure(4);
    const auto *narrow_base = sparse.data()[0].row.modes;
    const auto *narrow_next = sparse.data()[1].row.modes;
    BOOST_TEST(narrow_next - narrow_base == 4);
    sparse.configure(64, 12);
    sparse.ensure(4);
    BOOST_TEST(sparse.data()[1].row.modes - sparse.data()[0].row.modes == 12);
}

// --- the escape tail ---------------------------------------------------------------------------------
//
// A query is M ⊕ G and a fully paired product escapes the cutoff, so a query's support is unbounded and no
// fixed-stride sparse record can hold every one. The escaped record keeps its place and its stride and
// names a tail entry instead, which is what leaves the engine's offsets, counts and compaction as plain
// arithmetic.

namespace {

// The scan's shape: records into one buffer, escape monomials into another, concatenated once pushing is
// done -- a record cannot be appended after the tail has started.
struct WireBuffers {
    VecZ records = query_buffer();
    VecZ escapes;

    auto finish() -> VecZ {
        VecZ out = records;
        out.insert(out.end(), escapes.begin(), escapes.end());
        return out;
    }
};

auto wide_monomial(size_t num_bits, size_t seed) -> Bitset {
    Bitset mono(num_bits);
    // Fully paired and far wider than any record capacity below: the shape that escapes the cutoff and so
    // cannot be sized away.
    for (size_t mode = seed % 3; mode < (num_bits / 2); mode += 3) {
        mono.set(2 * mode);
        mono.set((2 * mode) + 1);
    }
    return mono;
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_wire_escaped_records_carry_their_monomial_in_the_tail) {
    constexpr size_t kCapacity = 8;
    constexpr size_t kNumBits = 512;
    const size_t stride = query_words(sparse_payload_words(kCapacity));
    std::mt19937_64 rng(20260813U);

    WireBuffers wire;
    std::vector<OwnedRow> rows;
    std::vector<Bitset> escaped;
    std::vector<bool> is_escape;
    for (size_t t = 0; t < 24; ++t) {
        // Every third record escapes, so escaped and plain records interleave -- the ordering the tail
        // indices have to survive.
        if (t % 3 == 2) {
            escaped.push_back(wide_monomial(kNumBits, t));
            sparse_query_push_escape(wire.records, wire.escapes, escaped.back(), kCapacity, (t % 2) == 0 ? 1 : -1);
            rows.emplace_back();
            is_escape.push_back(true);
            continue;
        }
        rows.push_back(random_row(rng, kNumBits / 2, kCapacity));
        sparse_query_push(wire.records, rows.back().view(), kCapacity, (t % 2) == 0 ? 1 : -1);
        is_escape.push_back(false);
    }
    const VecZ buf = wire.finish();

    // The header counts records, escaped ones included; the tail sits right after them.
    BOOST_REQUIRE(query_record_count(buf) == is_escape.size());
    BOOST_REQUIRE(query_tail_offset(buf, stride) == kQueryHeaderWords + (is_escape.size() * stride));
    BOOST_REQUIRE(buf.size() == query_tail_offset(buf, stride) + (escaped.size() * (kNumBits / 64)));

    SparseQueryKeys keys;
    keys.configure(kNumBits, kCapacity);
    keys.ensure(is_escape.size());
    keys.begin_batch();
    for (size_t q = 0; q < is_escape.size(); ++q) {
        BOOST_TEST(keys.read_record(buf, q, stride, q) == ((q % 2) == 0 ? 1 : -1));
    }

    size_t next_escape = 0;
    for (size_t q = 0; q < is_escape.size(); ++q) {
        const auto &key = keys[q];
        BOOST_REQUIRE(key.is_spilled() == is_escape[q]);
        if (is_escape[q]) {
            // The monomial has to arrive bit-for-bit: it is the only form this query exists in.
            BOOST_TEST((*key.spilled == escaped[next_escape]));
            ++next_escape;
            continue;
        }
        BOOST_TEST(key.row.codes == rows[q].codes);
        for (size_t j = 0; j < key.row.num_slots(); ++j) {
            BOOST_TEST(key.row.mode(j) == rows[q].view().mode(j));
        }
    }
    BOOST_TEST(next_escape == escaped.size());
}

// The fused sink widens every record by a value word, which moves the tail. The escape indices survive
// because they name a position *within* the tail rather than an offset into the buffer -- so the same
// records read back at the fused stride.
BOOST_AUTO_TEST_CASE(sparse_wire_escape_indices_survive_the_fused_relayout) {
    constexpr size_t kCapacity = 6;
    constexpr size_t kNumBits = 256;
    const size_t payload = sparse_payload_words(kCapacity);
    std::mt19937_64 rng(20260814U);

    WireBuffers wire;
    std::vector<Bitset> escaped;
    std::vector<double> values;
    std::vector<bool> is_escape;
    for (size_t t = 0; t < 12; ++t) {
        if (t % 2 == 0) {
            escaped.push_back(wide_monomial(kNumBits, t));
            sparse_query_push_escape(wire.records, wire.escapes, escaped.back(), kCapacity, 1);
            is_escape.push_back(true);
        }
        else {
            const OwnedRow row = random_row(rng, kNumBits / 2, kCapacity);
            sparse_query_push(wire.records, row.view(), kCapacity, -1);
            is_escape.push_back(false);
        }
        values.push_back(static_cast<double>(t) * 0.5 - 3.0);
    }
    const VecZ plain = wire.finish();

    VecZ fused;
    build_fused_query_value(plain, values, fused, payload);
    BOOST_REQUIRE(query_record_count(fused) == is_escape.size());
    BOOST_REQUIRE(fused.size()
                  == kQueryHeaderWords + (is_escape.size() * query_words_fused(payload))
                         + (escaped.size() * (kNumBits / 64)));

    SparseQueryKeys keys;
    keys.configure(kNumBits, kCapacity);
    keys.ensure(is_escape.size());
    keys.begin_batch();
    size_t next_escape = 0;
    for (size_t q = 0; q < is_escape.size(); ++q) {
        BOOST_TEST(keys.read_record(fused, q, query_words_fused(payload), q) == (is_escape[q] ? 1 : -1));
        BOOST_TEST(query_value(fused, q, payload) == values[q]);
        BOOST_REQUIRE(keys[q].is_spilled() == is_escape[q]);
        if (is_escape[q]) {
            BOOST_TEST((*keys[q].spilled == escaped[next_escape]));
            ++next_escape;
        }
    }
    BOOST_TEST(next_escape == escaped.size());
}

// Deferred self-misses are inserted after both resolve passes, by which time the batch has been refilled
// many times over -- so a key that must survive that has to be retained, and retaining it has to copy.
BOOST_AUTO_TEST_CASE(query_keys_retained_survive_a_refill) {
    constexpr size_t kCapacity = 6;
    constexpr size_t kNumBits = 256;
    const size_t stride = query_words(sparse_payload_words(kCapacity));
    std::mt19937_64 rng(20260815U);

    WireBuffers wire;
    std::vector<OwnedRow> rows;
    std::vector<Bitset> escaped;
    std::vector<bool> is_escape;
    for (size_t t = 0; t < 8; ++t) {
        if (t % 4 == 3) {
            escaped.push_back(wide_monomial(kNumBits, t));
            sparse_query_push_escape(wire.records, wire.escapes, escaped.back(), kCapacity, 1);
            rows.emplace_back();
            is_escape.push_back(true);
            continue;
        }
        rows.push_back(random_row(rng, kNumBits / 2, kCapacity));
        sparse_query_push(wire.records, rows.back().view(), kCapacity, 1);
        is_escape.push_back(false);
    }
    const VecZ buf = wire.finish();

    SparseQueryKeys keys;
    keys.configure(kNumBits, kCapacity);
    keys.ensure(2);

    // Read the records two at a time into the same two slots, retaining every key -- the resolve path's
    // batching, with kResolveBatch of 2.
    std::vector<size_t> handles;
    for (size_t q = 0; q < is_escape.size(); q += 2) {
        keys.begin_batch();
        for (size_t j = 0; j < 2; ++j) {
            (void)keys.read_record(buf, q + j, stride, j);
        }
        for (size_t j = 0; j < 2; ++j) {
            handles.push_back(keys.retain(j));
        }
    }

    size_t next_escape = 0;
    for (size_t q = 0; q < handles.size(); ++q) {
        const auto key = keys.retained(handles[q]);
        BOOST_REQUIRE(key.is_spilled() == is_escape[q]);
        if (is_escape[q]) {
            BOOST_TEST((*key.spilled == escaped[next_escape]));
            ++next_escape;
            continue;
        }
        BOOST_TEST(key.row.codes == rows[q].codes);
        for (size_t j = 0; j < key.row.num_slots(); ++j) {
            BOOST_TEST(key.row.mode(j) == rows[q].view().mode(j));
        }
    }
    BOOST_TEST(next_escape == escaped.size());

    // The dense batch owes the same guarantee, and its keys are whole monomials.
    VecZ dense_buf = query_buffer();
    std::vector<Bitset> monos;
    for (size_t t = 0; t < 8; ++t) {
        Bitset mono(kNumBits);
        mono.set(t);
        mono.set(kNumBits - 1 - t);
        monos.push_back(mono);
        query_push(dense_buf, mono, 1);
    }
    DenseQueryKeys dense;
    dense.configure(kNumBits, 0);
    dense.ensure(2);
    std::vector<size_t> dense_handles;
    for (size_t q = 0; q < monos.size(); q += 2) {
        dense.begin_batch();
        for (size_t j = 0; j < 2; ++j) {
            (void)dense.read_record(dense_buf, q + j, query_words(kNumBits / 64), j);
        }
        for (size_t j = 0; j < 2; ++j) {
            dense_handles.push_back(dense.retain(j));
        }
    }
    for (size_t q = 0; q < dense_handles.size(); ++q) {
        BOOST_TEST((dense.retained(dense_handles[q]) == monos[q]));
    }
}
