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
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/operator/InvertedIndex.h"

// Internals of the even-parity scan inverted index: the tiered column store (delta+LEB128 coded
// postings vs dense bit-vectors, with the tier chosen by an encoded-BYTE comparison rather than a
// density proxy), the posting codec itself, the lazily-built per-row parity(|M|) bitmap, and the
// fill. The inverted index reads its rows through the backend-agnostic for_each_row_position
// accessor, which is defined for a plain std::vector<Monomial<N>> — so these tests build one
// directly, no operator store required.

using namespace monoprop;
using namespace monoprop::detail;

namespace {
constexpr size_t N = 32; // 2N = 64 majorana columns
using Sc = InvertedIndex<N>;
using MSet = Monomial<N>;
MSet bs(const VecZ &r) {
    return indices_to_bitset<N>(r);
}
// indices_to_bitset maps mode index m to bit position 2N-1-m, and the inverted index indexes its columns by
// raw bit position — so mode m populates column col_of(m).
constexpr size_t col_of(size_t mode) {
    return 2 * N - 1 - mode;
}

// What the codec emits for an ascending row list, computed independently of the encoder: one absolute
// header per block of kPostingsPerBlock, then a LEB128 gap for every other posting. The tier rule
// compares exactly this against dense_bytes(), so the tests can predict tiers from first principles.
template <size_t W>
auto predicted_encoded_bytes(const std::vector<TermIndex> &rows) -> size_t {
    size_t bytes = 0;
    size_t prev = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i % InvertedIndex<W>::kPostingsPerBlock == 0) {
            bytes += InvertedIndex<W>::kBlockHeaderBytes;
        }
        else {
            bytes += leb128_size(static_cast<size_t>(rows[i]) - prev);
        }
        prev = rows[i];
    }
    return bytes;
}

// Full-width reference fold: XOR every set row of every column into one bitmap, no blocking, no
// tiers. combine_columns_block must reproduce this for ANY block decomposition (XOR associativity).
template <size_t W>
auto reference_fold(const InvertedIndex<W> &sc, const std::vector<size_t> &cols) -> std::vector<uint64_t> {
    std::vector<uint64_t> ref(sc.words(), 0);
    for (size_t c : cols) {
        for (TermIndex r : sc.column_rows(c)) {
            ref[r >> 6] ^= uint64_t{1} << (r & 63U);
        }
    }
    return ref;
}

// The same fold via the production kernel, in blocks of `block_words`.
template <size_t W>
auto blocked_fold(const InvertedIndex<W> &sc,
                  const std::vector<size_t> &cols,
                  size_t block_words) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(sc.words(), 0);
    std::vector<uint64_t> blk(block_words, 0);
    for (size_t bb = 0; bb < sc.words(); bb += block_words) {
        const size_t be = std::min(bb + block_words, sc.words());
        combine_columns_block<W>(sc, std::span<const size_t>(cols.data(), cols.size()), blk.data(), bb, be);
        for (size_t w = bb; w < be; ++w) {
            out[w] = blk[w - bb];
        }
    }
    return out;
}

// A deterministic multi-density operator: column c is set every `stride(c)` rows, so densities span
// three decades. With one-byte gaps a column codes to ~num_rows/stride bytes against a num_rows/8
// bitmap, so strides below ~8 land DENSE and the rest stay coded — both tiers, and multi-block
// posting streams, in one fixture.
template <size_t W>
auto strided_operator(size_t num_rows) -> std::vector<Monomial<W>> {
    std::vector<Monomial<W>> op(num_rows);
    for (size_t c = 0; c < Monomial<W>::size(); ++c) {
        const size_t stride = 1 + (c * 37) % 700;
        for (size_t r = c % stride; r < num_rows; r += stride) {
            op[r].set(c);
        }
    }
    return op;
}
} // namespace

// row_parity_words() builds the packed popcount(|M|)&1 bitmap over the current rows. Verify each
// bit against the known parity of the row it was built from.
BOOST_AUTO_TEST_CASE(inverted_index_row_parity_matches_popcount) {
    const std::vector<MSet> op{
        bs({0, 1}),       // |M| = 2 -> even
        bs({0, 1, 2}),    // |M| = 3 -> odd
        bs({5}),          // |M| = 1 -> odd
        bs({4, 5, 6, 7}), // |M| = 4 -> even
    };
    Sc sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == op.size());

    const uint64_t *parity = sc.row_parity_words();
    bool all_match = true;
    for (size_t i = 0; i < op.size(); ++i) {
        const bool bit = ((parity[i >> 6] >> (i & 63U)) & 1U) != 0;
        if (bit != static_cast<bool>(op[i].count() & 1U)) {
            all_match = false;
        }
    }
    BOOST_TEST(all_match);
    // row_parity_words is idempotent (a lazy cache): a second call must not change the bitmap.
    BOOST_TEST(((sc.row_parity_words()[0] >> 1U) & 1U) == 1U); // row 1 is odd
}

// THE tier rule: a column stays SPARSE only while its delta+LEB128 postings encode strictly smaller
// than its full-height bitmap would be, times kDenseBias — an actual byte comparison, not a density
// threshold. The two straddling columns below are chosen so the verdict is the same whichever width
// TermIndex has. The concrete tier expectations hold at the default bias of 1.0; a build that
// deliberately biases toward bitmaps is MEANT to move columns, so those are guarded and the general
// rule check below is bias-aware.
BOOST_AUTO_TEST_CASE(inverted_index_tier_is_the_encoded_byte_comparison) {
    constexpr size_t kR = 1024; // words() = 16 -> dense_bytes() = 128
    std::vector<MSet> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        VecZ pos;
        if (i < 100) {
            pos.push_back(0); // mode 0: 100 consecutive rows -> 1 block + 99 one-byte gaps
        }
        if (i < 200) {
            pos.push_back(1); // mode 1: 200 consecutive rows -> 2 blocks + 198 one-byte gaps
        }
        pos.push_back(2); // mode 2: every row -> a bitmap is far cheaper
        op.push_back(indices_to_bitset<N>(pos));
    }
    Sc sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == kR);
    BOOST_TEST(sc.dense_bytes() == 128u);

    // Mode 1 codes to 2*kBlockHeaderBytes + 198 bytes (214 at u32) — over the bitmap, so DENSE at any
    // bias >= 1. Mode 2 is set in every row, where a bitmap wins by a mile.
    BOOST_TEST(sc.column_is_dense(col_of(1)));
    BOOST_TEST(predicted_encoded_bytes<N>(sc.column_rows(col_of(1))) >= sc.dense_bytes());
    BOOST_TEST(sc.column_is_dense(col_of(2)));

    if constexpr (Sc::kDenseBias == 1.0) {
        // Mode 0 codes to kBlockHeaderBytes + 99 bytes (107 at u32), under the 128-byte bitmap.
        BOOST_TEST(!sc.column_is_dense(col_of(0)));
        BOOST_TEST(sc.sparse_column_count(col_of(0)) == 100u);
        BOOST_TEST(sc.sparse_encoded_bytes(col_of(0)) == Sc::kBlockHeaderBytes + 99u);
        BOOST_TEST(sc.sparse_encoded_bytes(col_of(0)) < sc.dense_bytes());
        // The rule genuinely CHANGED: the retired threshold promoted at density >= 1/64, i.e. at >= 16
        // of these 1024 rows, so it would have made mode 0 dense. Coding it is cheaper, so now it is not.
        BOOST_TEST(sc.sparse_column_count(col_of(0)) * 64u >= sc.rows());
    }

    // Every tier decision must agree with the byte comparison, in both directions, at ANY bias.
    bool tiers_follow_the_rule = true;
    for (size_t c = 0; c < Sc::kNumColumns; ++c) {
        const auto rows = sc.column_rows(c);
        if (rows.empty()) {
            continue;
        }
        const bool should_be_dense = static_cast<double>(predicted_encoded_bytes<N>(rows)) * Sc::kDenseBias
                                     >= static_cast<double>(sc.dense_bytes());
        if (sc.column_is_dense(c) != should_be_dense) {
            tiers_follow_the_rule = false;
        }
    }
    BOOST_TEST(tiers_follow_the_rule);
}

// A single posting is recorded losslessly, and an untouched column costs nothing. Note the row count
// has to be large enough that the bitmap costs more than one block header, or NOTHING can be sparse:
// below 8*kBlockHeaderBytes rows the tier rule correctly makes even a one-posting column dense.
BOOST_AUTO_TEST_CASE(inverted_index_codes_a_lone_posting_losslessly) {
    constexpr size_t kR = 8192; // dense_bytes() = 1024, far above one header at any sane bias
    std::vector<MSet> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        op.push_back(i == 7 ? bs({1}) : MSet{});
    }
    Sc sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == kR);
    BOOST_TEST(!sc.column_is_dense(col_of(1)));
    BOOST_TEST(sc.sparse_column_count(col_of(1)) == 1u);
    BOOST_TEST(sc.column_rows(col_of(1)) == std::vector<TermIndex>{7u}, boost::test_tools::per_element());
    // One posting = one block header and no gap bytes at all.
    BOOST_TEST(sc.sparse_encoded_bytes(col_of(1)) == Sc::kBlockHeaderBytes);
    BOOST_TEST(sc.sparse_column_count(col_of(5)) == 0u);
    BOOST_TEST(sc.sparse_encoded_bytes(col_of(5)) == 0u);
    BOOST_TEST(sc.column_rows(col_of(5)).empty());

    // A one-posting stream still answers the range query, and only inside the range.
    std::vector<TermIndex> got;
    sc.for_each_sparse_row_in_range(col_of(1), 0, 64, [&got](TermIndex r) { got.push_back(r); });
    BOOST_TEST(got == std::vector<TermIndex>{7u}, boost::test_tools::per_element());
    got.clear();
    sc.for_each_sparse_row_in_range(col_of(1), 8, kR, [&got](TermIndex r) { got.push_back(r); });
    BOOST_TEST(got.empty());
    sc.for_each_sparse_row_in_range(col_of(5), 0, kR, [&got](TermIndex r) { got.push_back(r); });
    BOOST_TEST(got.empty()); // and an empty column yields nothing rather than reading the stream
}

// The codec's three functions must agree exactly: leb128_size predicts what leb128_encode emits (the
// tier rule and rebuild's exact reserve both depend on that), and leb128_decode inverts it.
BOOST_AUTO_TEST_CASE(inverted_index_leb128_roundtrips_and_size_is_exact) {
    const std::vector<size_t> values{1, 2, 127, 128, 129, 300, 16'383, 16'384, 1u << 20, (1u << 28) + 7};
    std::vector<uint8_t> stream;
    size_t predicted = 0;
    for (size_t v : values) {
        const size_t before = stream.size();
        leb128_encode(stream, v);
        BOOST_TEST(stream.size() - before == leb128_size(v));
        predicted += leb128_size(v);
    }
    BOOST_TEST(stream.size() == predicted);

    const uint8_t *cursor = stream.data();
    bool all_roundtrip = true;
    for (size_t v : values) {
        if (leb128_decode(cursor) != v) {
            all_roundtrip = false;
        }
    }
    BOOST_TEST(all_roundtrip);
    BOOST_TEST(cursor == stream.data() + stream.size()); // consumed exactly, no over/under-read
}

// rebuild encodes columns in row order, so postings come out STRICTLY ASCENDING — the invariant the
// fold's range query rests on. Build a many-mode operator kept below the tier crossover so columns
// stay sparse, then walk each one through its public iterator and assert the ordering.
BOOST_AUTO_TEST_CASE(inverted_index_fill_yields_ascending_sparse_rows) {
    constexpr size_t M = 64; // 2M = 128 columns
    using ScW = InvertedIndex<M>;
    constexpr size_t kR = 16'385; // large operator, many sparse columns
    std::vector<Monomial<M>> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        // one mode per row spread over all 128 columns: each column set ~128 times, so its ~136 coded
        // bytes stay far under the 2056-byte bitmap and every column stays sparse.
        op.push_back(indices_to_bitset<M>({i % 128}));
    }
    ScW sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == kR);

    bool all_sorted = true;
    bool saw_nonempty_sparse = false;
    for (size_t c = 0; c < 128; ++c) {
        if (sc.column_is_dense(c)) {
            continue;
        }
        if (sc.sparse_column_count(c) != 0) {
            saw_nonempty_sparse = true;
        }
        // Walk the tier through its public iteration entry point: strictly ascending is the contract
        // combine_columns_block's range query depends on, so assert it on what the fold actually sees.
        TermIndex prev = 0;
        bool first = true;
        sc.for_each_sparse_row(c, [&](TermIndex r) {
            if (!first && r <= prev) {
                all_sorted = false;
            }
            prev = r;
            first = false;
        });
    }
    BOOST_TEST(saw_nonempty_sparse); // the fill actually populated sparse columns
    BOOST_TEST(all_sorted);          // and they are ascending, at any thread count
}

// THE codec's headline invariant. An index grown by repeated append_rows must hold exactly the same
// rows as one built by a single rebuild over the same final operator — the incremental encoder and the
// two-pass one are different code paths, and re-tiering runs only on the incremental one.
//
// Their TIERS can still differ slightly, because the two-way re-tier is amortized: it fires when the
// row count doubles, so columns that drift out of tier during the last partial interval stay that way
// until the next pass. Dense under rebuild always implies dense under append (eager promotion is exact
// and runs on every fill); the converse holds only at a re-tier boundary, which
// inverted_index_demotes_columns_the_growing_bitmap_outgrows pins separately. Either way the rows and
// the fold are identical, which the checks below prove.
BOOST_AUTO_TEST_CASE(inverted_index_incremental_append_matches_rebuild) {
    constexpr size_t M = 64; // 2M = 128 columns
    using ScW = InvertedIndex<M>;
    constexpr size_t kR = 6'000;
    const auto op = strided_operator<M>(kR);

    ScW batch;
    batch.rebuild(op);

    ScW incr;
    for (size_t base = 0; base < kR;) {
        const size_t n = std::min<size_t>(1 + (base % 97), kR - base); // ragged chunks, incl. n == 1
        incr.append_rows(op, base, n);
        base += n;
    }

    BOOST_TEST(batch.rows() == kR);
    BOOST_TEST(incr.rows() == kR);

    bool rows_match = true;
    bool promotion_is_monotone = true;
    size_t dense_batch = 0;
    size_t dense_incr = 0;
    for (size_t c = 0; c < Monomial<M>::size(); ++c) {
        if (batch.column_rows(c) != incr.column_rows(c)) {
            rows_match = false;
        }
        if (batch.column_is_dense(c) && !incr.column_is_dense(c)) {
            promotion_is_monotone = false;
        }
        dense_batch += static_cast<size_t>(batch.column_is_dense(c));
        dense_incr += static_cast<size_t>(incr.column_is_dense(c));
    }
    BOOST_TEST(rows_match);            // lossless under BOTH build paths
    BOOST_TEST(promotion_is_monotone); // rebuild-dense => append-dense
    BOOST_TEST(dense_incr >= dense_batch);
    // Re-tiering must pull the grown index CLOSE to the rebuilt one, not merely bound it from above:
    // with one-way promotion this fixture stranded columns dense by a wide margin.
    BOOST_TEST(dense_incr <= dense_batch + 4u);
    BOOST_TEST(dense_batch > 0u); // the fixture really does straddle the crossover
    BOOST_TEST(dense_batch < Monomial<M>::size());

    // ...and the two fold to the same bits despite any tier divergence.
    const std::vector<size_t> gen{3, 17, 40, 91};
    BOOST_TEST(blocked_fold(batch, gen, kColumnBlockWords) == blocked_fold(incr, gen, kColumnBlockWords),
               boost::test_tools::per_element());
    // The lazy parity bitmap is derived by iterating both tiers, so it must agree too.
    const uint64_t *pb = batch.row_parity_words();
    const uint64_t *pi = incr.row_parity_words();
    bool parity_matches = true;
    for (size_t w = 0; w < batch.words(); ++w) {
        if (pb[w] != pi[w]) {
            parity_matches = false;
        }
    }
    BOOST_TEST(parity_matches);
}

// The tier decision must be REVISABLE, not just one-way. A bitmap costs O(row_count) whether or not
// the column gains postings, so a column touched only while the operator was tiny is cheapest as a
// bitmap then and hopelessly over-committed later. Leaving that unrevised stranded 4.29 MiB across 210
// columns on hubbard, whose operator grows 32 -> 1,169,024 rows.
//
// Growth here ends exactly on a doubling, i.e. on a re-tier boundary, so the grown index must match a
// rebuild at the final size EXACTLY -- same tiers, not merely the same rows.
BOOST_AUTO_TEST_CASE(inverted_index_demotes_columns_the_growing_bitmap_outgrows) {
    constexpr size_t M = 64;
    using ScW = InvertedIndex<M>;
    constexpr size_t kR = 8192;  // power of two: the last append lands on a re-tier boundary
    constexpr size_t kEarly = 4; // column 0 is touched in these rows and never again

    std::vector<Monomial<M>> op(kR);
    for (size_t r = 0; r < kEarly; ++r) {
        op[r].set(0);
    }
    for (size_t c = 1; c < Monomial<M>::size(); ++c) {
        const size_t stride = 1 + (c * 37) % 700;
        for (size_t r = c % stride; r < kR; r += stride) {
            op[r].set(c);
        }
    }

    ScW grown;
    grown.append_rows(op, 0, kEarly);
    // At 4 rows the bitmap is 8 B while the postings cost a whole 8 B block header plus 3 gap bytes,
    // so the byte comparison correctly picks the bitmap. This is the promotion that used to be final.
    BOOST_TEST(grown.dense_bytes() == 8u);
    BOOST_TEST(grown.encoded_bytes_if_coded(0) == ScW::kBlockHeaderBytes + 3u);
    BOOST_TEST(grown.column_is_dense(0));

    for (size_t have = kEarly; have < kR;) {
        const size_t next = std::min(kR, have * 2);
        grown.append_rows(op, have, next - have);
        have = next;
    }
    BOOST_TEST(grown.rows() == kR);

    // By 8192 rows the bitmap costs 1024 B against ~11 B of postings, so it must have been demoted --
    // and the re-encode must be lossless.
    BOOST_TEST(grown.dense_bytes() == 1024u);
    BOOST_TEST(!grown.column_is_dense(0));
    BOOST_TEST(grown.sparse_column_count(0) == kEarly);
    BOOST_TEST(grown.sparse_encoded_bytes(0) == ScW::kBlockHeaderBytes + 3u);
    BOOST_TEST(grown.column_rows(0) == std::vector<TermIndex>({0u, 1u, 2u, 3u}), boost::test_tools::per_element());

    // Landing on a re-tier boundary makes the grown index tier-for-tier identical to a rebuild, since
    // both apply the same byte comparison at the same row count.
    ScW fresh;
    fresh.rebuild(op);
    bool tiers_identical = true;
    bool rows_identical = true;
    for (size_t c = 0; c < Monomial<M>::size(); ++c) {
        if (grown.column_is_dense(c) != fresh.column_is_dense(c)) {
            tiers_identical = false;
        }
        if (grown.column_rows(c) != fresh.column_rows(c)) {
            rows_identical = false;
        }
    }
    BOOST_TEST(tiers_identical);
    BOOST_TEST(rows_identical);
    // ...so nothing is left mis-tiered, which is the criterion the stranded-column report used. Only
    // meaningful at the byte-optimal bias: above 1.0 the rule deliberately keeps columns dense that the
    // oracle would code, so delta_wins counts memory knowingly traded for fold speed, not a defect.
    if constexpr (ScW::kDenseBias == 1.0) {
        BOOST_TEST(grown.delta_coded_bytes()[2] == 0u);
        BOOST_TEST(fresh.delta_coded_bytes()[2] == 0u);
    }
    // Demotion is a re-encode, so prove it did not perturb a single folded bit.
    const std::vector<size_t> gen{0, 7, 33, 96};
    BOOST_TEST(blocked_fold(grown, gen, kColumnBlockWords) == blocked_fold(fresh, gen, kColumnBlockWords),
               boost::test_tools::per_element());
}

// Dense bitmaps must carry BOUNDED growth slack -- bounded by one CHUNK per column, not by a fraction
// of the bitmap. memory_bytes()/tier_memory_bytes() charge capacity(), correctly, because that is
// resident memory, and a flat bitmap extended once per layer by plain vector::resize grows capacity
// geometrically, ending up with up to ~2x the words it needs. Chunking answers that without the
// alternative failure mode (see inverted_index_dense_growth_never_copies_earlier_chunks): only the tail
// chunk is ever short, so only it is ever reallocated.
//
// The row count is deliberately not a power of two: that is the alignment where a doubling allocator
// strands the most, and where a probe at 2^20 rows would have shown a misleading 0.4%.
BOOST_AUTO_TEST_CASE(inverted_index_dense_bitmaps_carry_no_growth_slack) {
    constexpr size_t M = 64;
    using ScW = InvertedIndex<M>;
    constexpr size_t kR = 9'000; // 141 words per bitmap
    const auto op = strided_operator<M>(kR);

    ScW grown;
    for (size_t have = 0; have < kR;) {
        const size_t next = std::min(kR, have + 1 + have / 3); // ~1.33x per layer, Trotter-like
        grown.append_rows(op, have, next - have);
        have = next;
    }
    BOOST_TEST(grown.rows() == kR);
    BOOST_TEST(grown.dense_bytes() == 141u * sizeof(uint64_t));

    size_t dense_columns = 0;
    for (size_t c = 0; c < Monomial<M>::size(); ++c) {
        dense_columns += static_cast<size_t>(grown.column_is_dense(c));
    }
    BOOST_TEST(dense_columns > 0u); // the fixture really does hold bitmaps
    // THE assertion: the whole dense tier is within one tail chunk's geometric overshoot of the exact
    // bitmaps. 141 words is a fraction of a chunk, so a chunk rounded up to kDenseChunkWords would blow
    // this by 7x -- chunks are sized to their share for exactly that reason.
    const size_t exact = dense_columns * grown.dense_bytes();
    BOOST_TEST(grown.tier_memory_bytes()[0] >= exact);
    BOOST_TEST(grown.tier_memory_bytes()[0] <= 2 * exact);

    // The lazy parity bitmap grows by the same mechanism and is charged the same way, so it gets the
    // same treatment. Build it partway through so the later appends must actually extend it.
    ScW two;
    two.append_rows(op, 0, 1'000);
    BOOST_TEST(two.row_parity_words() != nullptr);
    two.append_rows(op, 1'000, kR - 1'000);
    BOOST_TEST(two.rows() == kR);
    // Everything memory_bytes() counts that the two tiers do not is the parity bitmap.
    const auto tiers = two.tier_memory_bytes();
    BOOST_TEST(two.memory_bytes() - tiers[0] - tiers[1] == two.words() * sizeof(uint64_t));
    // ...and extending it in place must not have corrupted it.
    const uint64_t *parity = two.row_parity_words();
    bool parity_matches = true;
    for (size_t i = 0; i < kR; ++i) {
        const bool bit = ((parity[i >> 6] >> (i & 63U)) & 1U) != 0;
        if (bit != static_cast<bool>(op[i].count() & 1U)) {
            parity_matches = false;
        }
    }
    BOOST_TEST(parity_matches);
}

// Extending a dense bitmap must not move the words already written. That is the ONE property the
// chunking exists for: fill_rows extends every dense column once per gate, i.e. O(10^4) times per run,
// so an extension that reallocates costs O(dense columns x rows) PER GATE rather than amortizing over
// the run. A flat bitmap with capacity pinned to size does exactly that, and measured 2.7-3.2x on the
// two bench models once the tier rule was biased far enough to produce dense columns at all.
//
// The assertion is a captured chunk pointer that must not change across ~1,300 appends past that
// chunk's completion -- allocator- and timing-independent, unlike a byte count or a wall-clock probe.
// The pre-existing growth tests append ~30 times over a bitmap of 141 words, i.e. a fraction of ONE
// chunk, and so cannot see this at all.
BOOST_AUTO_TEST_CASE(inverted_index_dense_growth_never_copies_earlier_chunks) {
    constexpr size_t M = 64;
    using ScW = InvertedIndex<M>;
    constexpr size_t kStep = 100;
    constexpr size_t kAppends = 2'000;      // 200,000 rows = 3,125 words = 4 chunks
    constexpr size_t kR = kStep * kAppends; // ...so chunk 0 completes ~1/3 of the way in

    std::vector<Monomial<M>> op(kR);
    for (size_t r = 0; r < kR; ++r) {
        op[r].set(0); // fully set: dense at every row count, so the tier never flips under the test
        if (r % 1'000 == 0) {
            op[r].set(7); // ...and one column far below the crossover, so the mixed path is exercised
        }
    }

    ScW sc;
    const uint64_t *chunk0 = nullptr;
    size_t moves = 0;
    size_t observations = 0;
    for (size_t a = 0; a < kAppends; ++a) {
        sc.append_rows(op, a * kStep, kStep);
        BOOST_TEST_REQUIRE(sc.column_is_dense(0));
        if (sc.words() <= ScW::kDenseChunkWords) {
            continue; // chunk 0 is still the tail, and the tail is allowed to be reallocated
        }
        const uint64_t *const now = sc.dense_chunk(0, 0);
        ++observations;
        if (chunk0 == nullptr) {
            chunk0 = now;
        }
        else if (now != chunk0) {
            ++moves;
        }
    }
    BOOST_TEST(observations > 1'000u); // the test really did keep appending after chunk 0 completed
    BOOST_TEST(moves == 0u);           // THE assertion; the flat exact-reserve bitmap moves on every one

    // Growth must also not have corrupted a bit, at block widths that STRADDLE chunk boundaries. Every
    // production fold block is chunk-aligned, so an absolute word index into a chunk-relative pointer
    // would pass the whole suite and silently mis-fold only here -- and only on a multi-chunk index,
    // which no other fixture builds.
    const std::vector<size_t> gen{0, 7, 33};
    const auto ref = reference_fold(sc, gen);
    bool all_match = true;
    for (size_t block_words : {size_t{1},
                               size_t{3},
                               size_t{17},
                               ScW::kDenseChunkWords - 1,
                               ScW::kDenseChunkWords,
                               ScW::kDenseChunkWords + 1,
                               size_t{3'000}}) {
        if (blocked_fold(sc, gen, block_words) != ref) {
            all_match = false;
        }
    }
    BOOST_TEST(all_match);
    BOOST_TEST(std::ranges::any_of(ref, [](uint64_t w) { return w != 0; })); // not trivially zero

    // ...and the slack the chunking buys does not scale with the row count: it is one tail chunk either
    // way, so a 10x taller bitmap must not carry ~10x the slack. This is the property the flat geometric
    // bitmap lacked (a fixed ~33% of the whole thing) and the reason a fraction is the wrong bound.
    ScW tenth;
    tenth.append_rows(op, 0, kR / 10);
    const auto slack = [](const ScW &s) {
        size_t dense_columns = 0;
        for (size_t c = 0; c < Monomial<M>::size(); ++c) {
            dense_columns += static_cast<size_t>(s.column_is_dense(c));
        }
        return s.tier_memory_bytes()[0] - dense_columns * s.dense_bytes();
    };
    BOOST_TEST(slack(sc) <= 4 * slack(tenth) + 1'024u); // 10x the rows, nowhere near 10x the slack
}

// combine_columns_block must reproduce the full-width reference fold for ANY block decomposition:
// XOR is associative, so splitting the row range cannot change the result. Block widths that do not
// divide the word count, and a width of 1, are the cases most likely to expose an off-by-one in the
// posting range query's seek.
BOOST_AUTO_TEST_CASE(inverted_index_blocked_fold_matches_full_width_reference) {
    constexpr size_t M = 64;
    using ScW = InvertedIndex<M>;
    ScW sc;
    sc.rebuild(strided_operator<M>(4'000));

    const std::vector<std::vector<size_t>> gens{
        {0},                   // a single column, sparse or dense
        {5, 6},                // a pair
        {1, 2, 3, 4, 5, 6, 7}, // an odd number, mixed tiers
        {127, 0, 64, 33, 12},  // unsorted column order
        {11, 11},              // the same column twice: must cancel to zero
    };
    bool all_match = true;
    for (const auto &gen : gens) {
        const auto ref = reference_fold(sc, gen);
        for (size_t block_words : {size_t{1}, size_t{3}, size_t{17}, size_t{64}, kColumnBlockWords}) {
            if (blocked_fold(sc, gen, block_words) != ref) {
                all_match = false;
            }
        }
    }
    BOOST_TEST(all_match);
    // Sanity: the fixture is not trivially all-zero, and a column XORed with itself really cancels.
    BOOST_TEST(std::ranges::any_of(reference_fold(sc, {0, 5, 33}), [](uint64_t w) { return w != 0; }));
    BOOST_TEST(std::ranges::none_of(reference_fold(sc, {11, 11}), [](uint64_t w) { return w != 0; }));
}

// The range query must return exactly the postings in [lo, hi). It seeks by skip block and then decodes
// forward, so the cases that matter are block-aligned vs interior bounds, ranges before the first and
// after the last posting, and empty ranges. Multi-block columns are what make the seek non-trivial.
BOOST_AUTO_TEST_CASE(inverted_index_range_query_matches_filtered_full_walk) {
    constexpr size_t M = 64;
    using ScW = InvertedIndex<M>;
    ScW sc;
    sc.rebuild(strided_operator<M>(20'000)); // dense_bytes() = 2504, so wide columns stay coded

    bool saw_multi_block = false;
    bool all_match = true;
    for (size_t c = 0; c < Monomial<M>::size(); ++c) {
        if (sc.column_is_dense(c)) {
            continue;
        }
        const auto rows = sc.column_rows(c);
        if (rows.size() > ScW::kPostingsPerBlock) {
            saw_multi_block = true;
        }
        const size_t last = rows.empty() ? 0 : static_cast<size_t>(rows.back());
        for (size_t lo : {size_t{0}, size_t{1}, size_t{63}, size_t{64}, size_t{4033}, last, last + 1}) {
            for (size_t width : {size_t{0}, size_t{1}, size_t{64}, size_t{1000}, size_t{25'000}}) {
                const size_t hi = lo + width;
                std::vector<TermIndex> expected;
                for (TermIndex r : rows) {
                    if (static_cast<size_t>(r) >= lo && static_cast<size_t>(r) < hi) {
                        expected.push_back(r);
                    }
                }
                std::vector<TermIndex> got;
                sc.for_each_sparse_row_in_range(c, lo, hi, [&got](TermIndex r) { got.push_back(r); });
                if (got != expected) {
                    all_match = false;
                }
            }
        }
    }
    BOOST_TEST(saw_multi_block); // the seek was actually exercised across blocks
    BOOST_TEST(all_match);
}

// delta_coded_bytes() was the pre-implementation sizing study that chose the byte comparison over a
// density threshold; now that the comparison IS the tier rule it verifies it. On a rebuilt index the
// oracle -- the cheaper of {bitmap, coded postings} per column -- must already be what we store, and
// no column can be one the coded form would win, because such a column would have stayed sparse.
BOOST_AUTO_TEST_CASE(inverted_index_realizes_the_min_bitmap_delta_oracle) {
    constexpr size_t M = 64;
    using ScW = InvertedIndex<M>;
    ScW sc;
    sc.rebuild(strided_operator<M>(20'000));

    size_t realized = 0;
    size_t dense_columns = 0;
    for (size_t c = 0; c < Monomial<M>::size(); ++c) {
        realized += sc.column_is_dense(c) ? sc.dense_bytes() : sc.sparse_encoded_bytes(c);
        dense_columns += static_cast<size_t>(sc.column_is_dense(c));
    }
    const auto diag = sc.delta_coded_bytes();
    BOOST_TEST(diag[0] > 0u);
    // Blanket-coding every column must cost MORE than the per-column choice -- the whole reason the
    // rule is a comparison and not "code everything". Only meaningful if some column went dense; with
    // no dense column the two figures coincide, which is itself the correct answer.
    BOOST_TEST(dense_columns > 0u);
    BOOST_TEST(diag[0] > diag[1]);
    // The oracle is byte-optimal, so it is what we store only at the byte-optimal bias. Above 1.0 the
    // gap to the oracle IS the memory deliberately spent on fold speed, and delta_wins counts the
    // columns it was spent on -- expected, not a defect.
    if constexpr (ScW::kDenseBias == 1.0) {
        BOOST_TEST(diag[1] == realized); // oracle bytes == what we actually store
        BOOST_TEST(diag[2] == 0u);       // no column left where coding it would still win
    }
}
