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

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "monoprop/detail/operator/ChunkedArray.h"

// The pooled chunk allocator and the chunked array built on it: that an arena is carved, reused and
// given back to the OS when it empties, that chunks land on the alignment the pool advertises, and
// that indexing, growth and cloning behave across chunk boundaries.

using monoprop::detail::ChunkedArray;
using monoprop::detail::ChunkPool;

namespace {

// Small arenas keep the tests cheap and make "one more chunk needs another arena" reachable in a few
// allocations. 4096 is the page size the pool rounds to anyway, so nothing is rounded away here.
constexpr size_t kChunkBytes = 4096;
constexpr size_t kChunksPerArena = 4;
constexpr size_t kArenaBytes = kChunkBytes * kChunksPerArena;

// Whether `addr` falls inside any mapping of this process, read straight from /proc/self/maps. This is
// what distinguishes "the pool stopped counting the arena" from "the pool actually unmapped it".
auto address_is_mapped(const void *addr) -> bool {
    const auto target = reinterpret_cast<uintptr_t>(addr);
    std::FILE *maps = std::fopen("/proc/self/maps", "re");
    if (maps == nullptr) {
        return false;
    }
    char line[512];
    bool found = false;
    while (!found && std::fgets(line, sizeof(line), maps) != nullptr) {
        unsigned long long lo = 0;
        unsigned long long hi = 0;
        if (std::sscanf(line, "%llx-%llx", &lo, &hi) == 2) {
            found = target >= lo && target < hi;
        }
    }
    std::fclose(maps);
    return found;
}

} // namespace

// The pool rounds the requested chunk size up to whole pages and then tiles an arena with an exact
// number of them, so no arena byte is unreachable and no chunk straddles two arenas.
BOOST_AUTO_TEST_CASE(chunk_pool_tiles_its_arena_with_page_rounded_chunks) {
    ChunkPool pool(kChunkBytes - 1, kArenaBytes);
    BOOST_CHECK_EQUAL(pool.chunk_bytes(), kChunkBytes);
    BOOST_CHECK_EQUAL(pool.chunks_per_arena(), kChunksPerArena);
    BOOST_CHECK_EQUAL(pool.arena_bytes(), pool.chunk_bytes() * pool.chunks_per_arena());
    BOOST_CHECK_EQUAL(pool.arena_count(), 0U); // nothing is mapped before the first allocation
    BOOST_CHECK_EQUAL(pool.mapped_bytes(), 0U);

    // A chunk of at least a huge page is rounded to a multiple of one, so it can be huge-page-aligned.
    ChunkPool huge(ChunkPool::kHugePageBytes + 1, 8 * ChunkPool::kHugePageBytes);
    BOOST_CHECK_EQUAL(huge.chunk_bytes(), 2 * ChunkPool::kHugePageBytes);
    BOOST_CHECK_EQUAL(huge.chunk_alignment(), ChunkPool::kHugePageBytes);
}

// Every chunk sits on the alignment the pool advertises: arenas start on a 2 MiB boundary and the
// chunks tile them, so transparent huge pages can back the big ones.
BOOST_AUTO_TEST_CASE(chunk_pool_hands_out_chunks_on_its_advertised_alignment) {
    ChunkPool pool(kChunkBytes, kArenaBytes);
    std::vector<void *> chunks;
    for (size_t i = 0; i < 2 * kChunksPerArena; ++i) {
        chunks.push_back(pool.allocate());
        BOOST_REQUIRE(chunks.back() != nullptr);
        BOOST_CHECK_EQUAL(reinterpret_cast<uintptr_t>(chunks.back()) % pool.chunk_alignment(), 0U);
    }
    BOOST_CHECK_EQUAL(pool.arena_count(), 2U);
    // The arena itself is huge-page-aligned even when the chunks are only page-aligned.
    BOOST_CHECK_EQUAL(reinterpret_cast<uintptr_t>(chunks.front()) % ChunkPool::kHugePageBytes, 0U);
    for (void *chunk : chunks) {
        pool.deallocate(chunk);
    }

    ChunkPool huge(ChunkPool::kHugePageBytes, 4 * ChunkPool::kHugePageBytes);
    void *big = huge.allocate();
    BOOST_CHECK_EQUAL(reinterpret_cast<uintptr_t>(big) % ChunkPool::kHugePageBytes, 0U);
    huge.deallocate(big);
}

// A returned chunk is handed out again before a new arena is mapped: the pool's whole point is that a
// store which shrinks and regrows -- as the index does on every rebuild -- costs no new mappings.
BOOST_AUTO_TEST_CASE(chunk_pool_reuses_a_freed_chunk_before_mapping_another_arena) {
    ChunkPool pool(kChunkBytes, kArenaBytes);
    std::vector<void *> chunks;
    for (size_t i = 0; i < kChunksPerArena; ++i) {
        chunks.push_back(pool.allocate());
    }
    BOOST_REQUIRE_EQUAL(pool.arena_count(), 1U);
    BOOST_CHECK_EQUAL(pool.live_chunks(), kChunksPerArena);

    void *const returned = chunks[1];
    pool.deallocate(returned);
    BOOST_CHECK_EQUAL(pool.live_chunks(), kChunksPerArena - 1);
    BOOST_CHECK_EQUAL(pool.arena_count(), 1U); // the arena still holds three live chunks

    void *const again = pool.allocate();
    BOOST_CHECK_EQUAL(again, returned);
    BOOST_CHECK_EQUAL(pool.arena_count(), 1U);
    chunks[1] = again;

    // One chunk past the arena's capacity, and only then does a second arena appear.
    chunks.push_back(pool.allocate());
    BOOST_CHECK_EQUAL(pool.arena_count(), 2U);
    for (void *chunk : chunks) {
        pool.deallocate(chunk);
    }
    BOOST_CHECK_EQUAL(pool.arena_count(), 0U);
}

// The arena goes back to the OS the moment its last chunk returns -- not to a free list the process
// keeps. /proc/self/maps is the witness: the ledger's bytes are only believable if the mapping is gone.
BOOST_AUTO_TEST_CASE(chunk_pool_unmaps_an_arena_once_its_last_chunk_returns) {
    ChunkPool pool(kChunkBytes, kArenaBytes);
    std::vector<void *> first;
    for (size_t i = 0; i < kChunksPerArena; ++i) {
        first.push_back(pool.allocate());
    }
    std::vector<void *> second;
    for (size_t i = 0; i < kChunksPerArena; ++i) {
        second.push_back(pool.allocate());
    }
    BOOST_REQUIRE_EQUAL(pool.arena_count(), 2U);
    BOOST_CHECK_EQUAL(pool.mapped_bytes(), 2 * pool.arena_bytes());
    const void *const witness = second.front();
    BOOST_CHECK(address_is_mapped(witness));

    // Drain the second arena only: the first must keep its mapping.
    for (void *chunk : second) {
        pool.deallocate(chunk);
    }
    BOOST_CHECK_EQUAL(pool.arena_count(), 1U);
    BOOST_CHECK_EQUAL(pool.mapped_bytes(), pool.arena_bytes());
    BOOST_CHECK(!address_is_mapped(witness));
    BOOST_CHECK(address_is_mapped(first.front()));

    for (void *chunk : first) {
        pool.deallocate(chunk);
    }
    BOOST_CHECK_EQUAL(pool.arena_count(), 0U);
    BOOST_CHECK_EQUAL(pool.mapped_bytes(), 0U);
    BOOST_CHECK_EQUAL(pool.live_chunks(), 0U);
    BOOST_CHECK(!address_is_mapped(first.front()));
}

// Indexing is the whole contract: element i must round-trip whichever chunk it fell in, and the three
// accessors (operator[], chunk_base, contiguous_at) must name the same byte.
BOOST_AUTO_TEST_CASE(chunked_array_indexes_across_chunk_boundaries) {
    constexpr size_t kElemsPerChunk = 8;
    constexpr size_t kElems = 3 * kElemsPerChunk + 3; // deliberately not a whole number of chunks
    ChunkPool pool(kChunkBytes, kArenaBytes);
    ChunkedArray<uint64_t> array(pool, kElemsPerChunk);
    BOOST_CHECK(array.attached());
    BOOST_CHECK_EQUAL(array.size(), 0U);
    BOOST_CHECK_EQUAL(array.bytes(), 0U);

    array.grow(kElems);
    BOOST_CHECK_EQUAL(array.size(), kElems);
    BOOST_CHECK_EQUAL(array.chunk_count(), 4U);
    BOOST_CHECK_EQUAL(array.capacity(), 4 * kElemsPerChunk);
    BOOST_CHECK_EQUAL(array.slack_bytes(), (4 * kElemsPerChunk - kElems) * sizeof(uint64_t));
    BOOST_CHECK_GE(array.bytes(), 4 * pool.chunk_bytes());

    for (size_t i = 0; i < kElems; ++i) {
        array[i] = 0x1000U + i;
    }
    for (size_t i = 0; i < kElems; ++i) {
        BOOST_CHECK_EQUAL(array[i], 0x1000U + i);
        BOOST_CHECK_EQUAL(array.chunk_base(i)[i % kElemsPerChunk], 0x1000U + i);
        BOOST_CHECK_EQUAL(*array.contiguous_at(i), 0x1000U + i);
        BOOST_CHECK_EQUAL(array.elems_left_in_chunk(i), kElemsPerChunk - (i % kElemsPerChunk));
    }
    // A run that starts on a chunk boundary is readable as a plain array for the whole chunk.
    const uint64_t *run = array.contiguous_at(kElemsPerChunk);
    for (size_t j = 0; j < kElemsPerChunk; ++j) {
        BOOST_CHECK_EQUAL(run[j], 0x1000U + kElemsPerChunk + j);
    }

    // Growth appends chunks and never disturbs what is already there.
    array.grow(kElems + kElemsPerChunk);
    BOOST_CHECK_EQUAL(array.chunk_count(), 5U);
    for (size_t i = 0; i < kElems; ++i) {
        BOOST_CHECK_EQUAL(array[i], 0x1000U + i);
    }
    // Append-only: a request at or below the current size changes nothing.
    array.grow(1);
    BOOST_CHECK_EQUAL(array.size(), kElems + kElemsPerChunk);
}

// Chunks come back from the pool dirty, so grow_zeroed() must clear exactly the range it adds --
// nothing before it, and (so it never faults in slack) nothing past it.
BOOST_AUTO_TEST_CASE(chunked_array_grow_zeroed_clears_recycled_chunks) {
    constexpr size_t kElemsPerChunk = 8;
    ChunkPool pool(kChunkBytes, kArenaBytes);
    {
        ChunkedArray<uint64_t> dirty(pool, kElemsPerChunk);
        dirty.grow(2 * kElemsPerChunk);
        for (size_t i = 0; i < 2 * kElemsPerChunk; ++i) {
            dirty[i] = 0xdeadbeefU;
        }
    } // chunks return to the pool still carrying 0xdeadbeef

    ChunkedArray<uint64_t> array(pool, kElemsPerChunk);
    array.grow_zeroed(kElemsPerChunk + 1);
    BOOST_REQUIRE_EQUAL(pool.arena_count(), 1U); // it did reuse the dirty chunks
    for (size_t i = 0; i < kElemsPerChunk + 1; ++i) {
        BOOST_CHECK_EQUAL(array[i], 0U);
    }
    for (size_t i = 0; i < kElemsPerChunk + 1; ++i) {
        array[i] = i + 1;
    }
    // A second growth zeroes only the new elements, leaving the live prefix alone.
    array.grow_zeroed(3 * kElemsPerChunk);
    for (size_t i = 0; i < kElemsPerChunk + 1; ++i) {
        BOOST_CHECK_EQUAL(array[i], i + 1);
    }
    for (size_t i = kElemsPerChunk + 1; i < 3 * kElemsPerChunk; ++i) {
        BOOST_CHECK_EQUAL(array[i], 0U);
    }
}

// A clone owns its own chunks: the operator is copied whenever a propagator is, and a shared chunk
// would make the copy alias the original's index.
BOOST_AUTO_TEST_CASE(chunked_array_clone_is_an_independent_deep_copy) {
    constexpr size_t kElemsPerChunk = 8;
    constexpr size_t kElems = 2 * kElemsPerChunk + 5;
    ChunkPool pool(kChunkBytes, kArenaBytes);
    ChunkedArray<uint64_t> array(pool, kElemsPerChunk);
    array.grow(kElems);
    for (size_t i = 0; i < kElems; ++i) {
        array[i] = i * i;
    }

    ChunkedArray<uint64_t> same_pool = array.clone();
    ChunkPool other_pool(kChunkBytes, kArenaBytes);
    ChunkedArray<uint64_t> other = array.clone_into(other_pool);
    BOOST_CHECK_EQUAL(same_pool.size(), kElems);
    BOOST_CHECK_EQUAL(other.size(), kElems);
    for (size_t i = 0; i < kElems; ++i) {
        BOOST_CHECK_EQUAL(same_pool[i], i * i);
        BOOST_CHECK_EQUAL(other[i], i * i);
    }
    BOOST_CHECK(same_pool.chunk_base(0) != array.chunk_base(0));
    BOOST_CHECK_EQUAL(other_pool.live_chunks(), other.chunk_count());

    array[0] = 99;
    array[kElemsPerChunk] = 99;
    BOOST_CHECK_EQUAL(same_pool[0], 0U);
    BOOST_CHECK_EQUAL(same_pool[kElemsPerChunk], kElemsPerChunk * kElemsPerChunk);
    BOOST_CHECK_EQUAL(other[kElemsPerChunk], kElemsPerChunk * kElemsPerChunk);
}

// reset() is what a rebuild does to every column: the chunks must reach the pool (and, here, the OS)
// rather than being retained, and the array must stay attached and usable afterwards.
BOOST_AUTO_TEST_CASE(chunked_array_reset_and_move_return_every_chunk) {
    constexpr size_t kElemsPerChunk = 8;
    ChunkPool pool(kChunkBytes, kArenaBytes);
    ChunkedArray<uint64_t> array(pool, kElemsPerChunk);
    array.grow(3 * kElemsPerChunk);
    BOOST_REQUIRE_EQUAL(pool.live_chunks(), 3U);

    array.reset();
    BOOST_CHECK_EQUAL(array.size(), 0U);
    BOOST_CHECK_EQUAL(array.bytes(), 0U);
    BOOST_CHECK_EQUAL(pool.live_chunks(), 0U);
    BOOST_CHECK_EQUAL(pool.arena_count(), 0U);
    BOOST_CHECK(array.attached()); // still bound: a rebuild refills the same array

    array.grow_zeroed(kElemsPerChunk);
    BOOST_CHECK_EQUAL(pool.live_chunks(), 1U);
    ChunkedArray<uint64_t> moved = std::move(array);
    BOOST_CHECK_EQUAL(moved.size(), kElemsPerChunk);
    BOOST_CHECK_EQUAL(pool.live_chunks(), 1U); // moving transfers the chunks, it does not copy them
    ChunkedArray<uint64_t> target(pool, kElemsPerChunk);
    target.grow(2 * kElemsPerChunk);
    BOOST_REQUIRE_EQUAL(pool.live_chunks(), 3U);
    target = std::move(moved); // the overwritten chunks must be released, not leaked
    BOOST_CHECK_EQUAL(target.size(), kElemsPerChunk);
    BOOST_CHECK_EQUAL(pool.live_chunks(), 1U);
}
