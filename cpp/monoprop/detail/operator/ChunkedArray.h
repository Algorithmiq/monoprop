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

#pragma once

// Chunked backing for the big append-only term stores.
//
// A std::vector that grows to a billion terms pays twice: the geometric overshoot it never returns
// before quiescence, and a reallocation that holds the old and the new buffer at once. A store split
// into fixed-size chunks pays neither -- growth appends a chunk, nothing is copied, and the only slack
// is the tail of the last chunk. The cost is one indirection per element access, which the hot loops
// hoist by resolving a chunk base once per block of consecutive indices (chunk_base/contiguous_at).
//
// The chunks come from a per-owner ChunkPool rather than one mapping each: at 91 dense index columns
// times 128 partitions, a mapping per chunk would run into vm.max_map_count (65530) long before the
// memory ran out.

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <numeric>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace monoprop::detail {

/*! @brief Arena allocator handing out equally sized chunks to the chunked stores of one owner.
 *
 *  One size class per pool: every chunk it hands out is `chunk_bytes()` long, so a freed chunk fits any
 *  later request and the free list needs no search. Chunks are carved from arenas of `arena_bytes()`,
 *  mapped on demand and unmapped as soon as the last chunk in them comes back -- which is what makes a
 *  drained store give its memory back to the OS rather than to the allocator's own retention.
 *
 *  Arenas are 2 MiB-aligned so transparent huge pages can back them, and a chunk of 2 MiB or more is
 *  itself huge-page-aligned (see chunk_alignment()).
 *
 *  Not thread-safe, and deliberately so: one pool belongs to one partition's stores, which a single
 *  partition master drives.
 */
class ChunkPool {
public:
    //! Default arena size. Large enough that a 1 GiB column costs tens of mappings, not thousands.
    static constexpr size_t kArenaBytes = size_t{32} << 20;
    //! Transparent-huge-page granularity on x86-64; the alignment every arena is placed on.
    static constexpr size_t kHugePageBytes = size_t{2} << 20;

    /*! @brief Builds a pool whose chunks are at least @a chunk_bytes long.
     *
     *  The request is rounded up to a whole number of pages, or to a multiple of kHugePageBytes once it
     *  reaches that size, so the chunks tile the arena exactly. @a arena_bytes is a hint: the arena is
     *  sized to a whole number of chunks and is never smaller than one.
     */
    explicit ChunkPool(size_t chunk_bytes, size_t arena_bytes = kArenaBytes)
        : chunk_bytes_(round_up_(std::max(chunk_bytes, size_t{1}),
                                 chunk_bytes >= kHugePageBytes ? kHugePageBytes : page_bytes_())),
          chunks_per_arena_(std::max(size_t{1}, arena_bytes / chunk_bytes_)),
          arena_bytes_(chunks_per_arena_ * chunk_bytes_) {}

    ~ChunkPool() {
        // The owning stores release their chunks in their own destructors, and they are declared so as
        // to be destroyed first; anything still out here would be a leak either way, so unmap regardless.
        for (auto &arena : arenas_) {
            release_mapping_(*arena);
        }
    }

    ChunkPool(const ChunkPool &) = delete;
    auto operator=(const ChunkPool &) -> ChunkPool & = delete;
    // Non-movable on purpose: every ChunkedArray holds a bare pointer to its pool, so the pool object
    // must not change address. Owners that are themselves movable hold it through a unique_ptr.
    ChunkPool(ChunkPool &&) = delete;
    auto operator=(ChunkPool &&) -> ChunkPool & = delete;

    //! Length of every chunk this pool hands out, page-rounded up from the requested size.
    [[nodiscard]] auto chunk_bytes() const noexcept -> size_t { return chunk_bytes_; }
    //! Length of one arena: a whole number of chunks.
    [[nodiscard]] auto arena_bytes() const noexcept -> size_t { return arena_bytes_; }
    //! Chunks carved from one arena.
    [[nodiscard]] auto chunks_per_arena() const noexcept -> size_t { return chunks_per_arena_; }
    //! Alignment every chunk satisfies: kHugePageBytes for chunks that are a multiple of it, else the
    //! largest power of two dividing both the chunk size and the arena's own 2 MiB alignment.
    [[nodiscard]] auto chunk_alignment() const noexcept -> size_t { return std::gcd(chunk_bytes_, kHugePageBytes); }
    //! Arenas currently mapped.
    [[nodiscard]] auto arena_count() const noexcept -> size_t { return arenas_.size(); }
    //! Bytes currently mapped by this pool, free chunks included.
    [[nodiscard]] auto mapped_bytes() const noexcept -> size_t { return arenas_.size() * arena_bytes_; }
    //! Chunks handed out and not yet returned.
    [[nodiscard]] auto live_chunks() const noexcept -> size_t { return live_chunks_; }

    /*! @brief Hands out one chunk, mapping a new arena when no partly used one has room.
     *  @throws std::bad_alloc if the mapping fails.
     *  The chunk's bytes are indeterminate: a fresh mapping reads as zero, a recycled chunk does not.
     */
    [[nodiscard]] auto allocate() -> void * {
        if (partial_.empty()) {
            map_arena_();
        }
        Arena &arena = *partial_.back();
        std::byte *chunk = nullptr;
        if (!arena.free_chunks.empty()) {
            chunk = arena.free_chunks.back();
            arena.free_chunks.pop_back();
        }
        else {
            chunk = arena.base + (arena.next_unused * chunk_bytes_);
            ++arena.next_unused;
        }
        ++arena.live;
        ++live_chunks_;
        if (arena.free_chunks.empty() && arena.next_unused == chunks_per_arena_) {
            arena.in_partial = false;
            partial_.pop_back();
        }
        return chunk;
    }

    //! Returns a chunk from allocate(); unmaps its arena once every chunk in it is back.
    auto deallocate(void *chunk) noexcept -> void {
        if (chunk == nullptr) {
            return;
        }
        Arena *arena = arena_of_(static_cast<std::byte *>(chunk));
        assert(arena != nullptr && "chunk does not belong to this pool");
        --arena->live;
        --live_chunks_;
        if (arena->live == 0) {
            drop_arena_(*arena);
            return;
        }
        arena->free_chunks.push_back(static_cast<std::byte *>(chunk));
        if (!arena->in_partial) {
            arena->in_partial = true;
            partial_.push_back(arena);
        }
    }

private:
    //! One mapping, carved into chunks_per_arena_ chunks handed out in address order then recycled.
    struct Arena {
        std::byte *base = nullptr;              //!< 2 MiB-aligned start of the usable region
        std::byte *raw = nullptr;               //!< what the platform allocator returned (fallback path only)
        size_t next_unused = 0;                 //!< chunks never yet handed out start here
        size_t live = 0;                        //!< chunks handed out and not returned
        std::vector<std::byte *> free_chunks{}; //!< returned chunks, ready to hand out again
        bool in_partial = false;                //!< mirrors membership of ChunkPool::partial_
    };

    static auto round_up_(size_t value, size_t alignment) noexcept -> size_t {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    //! The platform page size, read once. Powers the rounding above, so it must be a power of two.
    static auto page_bytes_() noexcept -> size_t {
#if defined(__linux__)
        static const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        return page;
#else
        return size_t{4096};
#endif
    }

    auto map_arena_() -> void {
        auto arena = std::make_unique<Arena>();
#if defined(__linux__)
        // Over-map by one huge page and trim, so the arena starts on a 2 MiB boundary and is left as
        // exactly one VMA: THP can back it, and its footprint is one line of /proc/self/maps.
        const size_t requested = arena_bytes_ + kHugePageBytes;
        void *raw = ::mmap(nullptr, requested, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            throw std::bad_alloc();
        }
        auto *start = static_cast<std::byte *>(raw);
        auto *base = reinterpret_cast<std::byte *>(round_up_(reinterpret_cast<uintptr_t>(start), kHugePageBytes));
        const size_t head = static_cast<size_t>(base - start);
        if (head != 0) {
            ::munmap(start, head);
        }
        const size_t tail = requested - head - arena_bytes_;
        if (tail != 0) {
            ::munmap(base + arena_bytes_, tail);
        }
#if defined(MADV_HUGEPAGE)
        ::madvise(base, arena_bytes_, MADV_HUGEPAGE); // advisory: a refusal costs nothing but page size
#endif
        arena->base = base;
#else
        // Off Linux there is no unmap to make: posix_memalign gives the same alignment, and returning
        // the block to the platform allocator is the closest equivalent of dropping the mapping.
        void *raw = nullptr;
        if (::posix_memalign(&raw, kHugePageBytes, arena_bytes_) != 0) {
            throw std::bad_alloc();
        }
        arena->base = static_cast<std::byte *>(raw);
        arena->raw = arena->base;
#endif
        Arena *inserted = arena.get();
        // arenas_ stays sorted by base so arena_of_ is a binary search rather than a scan.
        const auto at = std::ranges::upper_bound(arenas_, inserted->base, {}, [](const auto &a) { return a->base; });
        arenas_.insert(at, std::move(arena));
        inserted->in_partial = true;
        partial_.push_back(inserted);
    }

    //! Gives @a arena's memory back to the OS. Leaves the bookkeeping object for the caller to erase.
    auto release_mapping_(Arena &arena) noexcept -> void {
#if defined(__linux__)
        ::munmap(arena.base, arena_bytes_);
#else
        std::free(arena.raw);
#endif
    }

    //! Drops @a arena's mapping and forgets it. The caller has already accounted for its chunks.
    auto drop_arena_(Arena &arena) noexcept -> void {
        if (arena.in_partial) {
            std::erase(partial_, &arena);
        }
        release_mapping_(arena);
        const auto at = std::ranges::lower_bound(arenas_, arena.base, {}, [](const auto &a) { return a->base; });
        assert(at != arenas_.end() && at->get() == &arena);
        arenas_.erase(at);
    }

    //! The arena containing @a chunk, or nullptr if no arena of this pool does.
    [[nodiscard]] auto arena_of_(std::byte *chunk) const noexcept -> Arena * {
        const auto at = std::ranges::upper_bound(arenas_, chunk, {}, [](const auto &a) { return a->base; });
        if (at == arenas_.begin()) {
            return nullptr;
        }
        Arena *arena = std::prev(at)->get();
        return chunk < arena->base + arena_bytes_ ? arena : nullptr;
    }

    size_t chunk_bytes_;
    size_t chunks_per_arena_;
    size_t arena_bytes_;
    size_t live_chunks_ = 0;
    std::vector<std::unique_ptr<Arena>> arenas_{}; //!< every mapped arena, ascending by base
    std::vector<Arena *> partial_{};               //!< arenas with a chunk available, each listed once
};

/*! @brief An append-only array of trivially copyable @a T held as a list of equally sized chunks.
 *
 *  Growth appends whole chunks from the pool the array is attached to: nothing is ever copied and the
 *  only slack is the tail of the last chunk. Elements are default-initialized (never zeroed) like
 *  DefaultInitVector, so a caller that needs zeros asks for grow_zeroed(); elements past size() are
 *  indeterminate and must not be read.
 *
 *  The chunk length is a power of two, so index -> (chunk, offset) is a shift and a mask. A run of
 *  consecutive indices that is known not to cross a chunk boundary can be read through a bare pointer
 *  from contiguous_at(), which is how the fold keeps its memcpy/XOR inner loop.
 *
 *  Move-only: the array owns its chunks and the pool must outlive it.
 */
template <typename T>
    requires std::is_trivially_copyable_v<T>
class ChunkedArray {
public:
    //! An unattached, empty array. attach() binds it to a pool before it can grow.
    ChunkedArray() noexcept = default;
    //! Binds directly to @a pool with @a elems_per_chunk elements per chunk (a power of two).
    ChunkedArray(ChunkPool &pool, size_t elems_per_chunk) { attach(pool, elems_per_chunk); }

    ChunkedArray(const ChunkedArray &) = delete;
    auto operator=(const ChunkedArray &) -> ChunkedArray & = delete;

    ChunkedArray(ChunkedArray &&other) noexcept { swap_(other); }
    auto operator=(ChunkedArray &&other) noexcept -> ChunkedArray & {
        if (this != &other) {
            reset();
            swap_(other);
        }
        return *this;
    }
    ~ChunkedArray() { reset(); }

    /*! @brief Binds an empty array to @a pool.
     *  @pre The array holds no chunks, and @a elems_per_chunk is a power of two that fits a chunk.
     */
    auto attach(ChunkPool &pool, size_t elems_per_chunk) -> void {
        assert(chunks_.empty() && "attach() rebinds only an empty array");
        assert(std::has_single_bit(elems_per_chunk));
        assert(elems_per_chunk * sizeof(T) <= pool.chunk_bytes());
        pool_ = &pool;
        elems_per_chunk_ = elems_per_chunk;
        shift_ = static_cast<size_t>(std::countr_zero(elems_per_chunk));
        mask_ = elems_per_chunk - 1;
    }

    //! Whether a pool has been bound, i.e. whether the array may grow.
    [[nodiscard]] auto attached() const noexcept -> bool { return pool_ != nullptr; }
    //! Elements per chunk, as bound by attach().
    [[nodiscard]] auto elems_per_chunk() const noexcept -> size_t { return elems_per_chunk_; }
    //! Live elements: indices [0, size()) are readable, the rest are not.
    [[nodiscard]] auto size() const noexcept -> size_t { return size_; }
    //! Whether the array holds no live elements.
    [[nodiscard]] auto empty() const noexcept -> bool { return size_ == 0; }
    //! Elements the chunks already hold room for: size() rounded up to a whole chunk.
    [[nodiscard]] auto capacity() const noexcept -> size_t { return chunks_.size() * elems_per_chunk_; }
    //! Chunks currently held.
    [[nodiscard]] auto chunk_count() const noexcept -> size_t { return chunks_.size(); }
    /*! @brief Bytes this array holds: the page-rounded chunks plus the chunk index that addresses them.
     *  The index is one pointer per chunk, so it is a part in ten thousand of the chunks themselves.
     */
    [[nodiscard]] auto bytes() const noexcept -> size_t {
        return (pool_ == nullptr ? 0 : chunks_.size() * pool_->chunk_bytes()) + (chunks_.capacity() * sizeof(T *));
    }
    //! The bytes of the last chunk that lie past size(): the array's whole growth slack.
    [[nodiscard]] auto slack_bytes() const noexcept -> size_t { return (capacity() - size_) * sizeof(T); }

    //! Element @a i. @pre i < size().
    [[nodiscard]] auto operator[](size_t i) noexcept -> T & { return chunks_[i >> shift_][i & mask_]; }
    //! Element @a i. @pre i < size().
    [[nodiscard]] auto operator[](size_t i) const noexcept -> const T & { return chunks_[i >> shift_][i & mask_]; }

    //! First element of the chunk holding @a i, so chunk_base(i)[i % elems_per_chunk()] is element @a i.
    [[nodiscard]] auto chunk_base(size_t i) noexcept -> T * { return chunks_[i >> shift_]; }
    //! @copydoc chunk_base
    [[nodiscard]] auto chunk_base(size_t i) const noexcept -> const T * { return chunks_[i >> shift_]; }

    //! Element @a i as a bare pointer, valid for the elems_left_in_chunk(i) elements from there on.
    [[nodiscard]] auto contiguous_at(size_t i) noexcept -> T * { return chunks_[i >> shift_] + (i & mask_); }
    //! @copydoc contiguous_at
    [[nodiscard]] auto contiguous_at(size_t i) const noexcept -> const T * {
        return chunks_[i >> shift_] + (i & mask_);
    }
    //! Elements from @a i to the end of its chunk: the length contiguous_at(i) is good for.
    [[nodiscard]] auto elems_left_in_chunk(size_t i) const noexcept -> size_t { return elems_per_chunk_ - (i & mask_); }

    /*! @brief Grows to @a n elements, appending whole chunks. New elements are indeterminate.
     *  A request at or below the current size is ignored: the array is append-only.
     */
    auto grow(size_t n) -> void {
        if (n <= size_) {
            return;
        }
        reserve_(n);
        size_ = n;
    }

    //! Grows to @a n elements as grow() does, with the new elements [size(), n) set to zero.
    auto grow_zeroed(size_t n) -> void {
        if (n <= size_) {
            return;
        }
        reserve_(n);
        // Only up to n: the tail of the last chunk is slack, not storage, and zeroing it would fault in
        // pages the caller may never touch.
        for (size_t i = size_; i < n;) {
            const size_t run = std::min(elems_left_in_chunk(i), n - i);
            std::memset(contiguous_at(i), 0, run * sizeof(T));
            i += run;
        }
        size_ = n;
    }

    //! Releases every chunk back to the pool and empties the array, keeping it attached.
    auto reset() noexcept -> void {
        for (T *chunk : chunks_) {
            pool_->deallocate(chunk);
        }
        chunks_.clear();
        chunks_.shrink_to_fit();
        size_ = 0;
    }

    //! A deep copy taking its chunks from the same pool.
    [[nodiscard]] auto clone() const -> ChunkedArray { return pool_ == nullptr ? ChunkedArray{} : clone_into(*pool_); }

    /*! @brief A deep copy taking its chunks from @a pool, for an owner that carries its own pool.
     *  @pre @a pool's chunks are at least as long as this array's.
     */
    [[nodiscard]] auto clone_into(ChunkPool &pool) const -> ChunkedArray {
        ChunkedArray copy;
        if (pool_ == nullptr) {
            return copy;
        }
        copy.attach(pool, elems_per_chunk_);
        copy.grow(size_);
        // Chunk by chunk, and only the live prefix: the slack past size_ is indeterminate and copying it
        // would report as a read of uninitialized memory.
        for (size_t i = 0; i < size_; i += elems_per_chunk_) {
            std::memcpy(copy.chunks_[i >> shift_],
                        chunks_[i >> shift_],
                        std::min(elems_per_chunk_, size_ - i) * sizeof(T));
        }
        return copy;
    }

private:
    auto reserve_(size_t n) -> void {
        assert(pool_ != nullptr && "a ChunkedArray must be attached to a pool before it grows");
        while (capacity() < n) {
            chunks_.push_back(static_cast<T *>(pool_->allocate()));
        }
    }

    auto swap_(ChunkedArray &other) noexcept -> void {
        std::swap(pool_, other.pool_);
        chunks_.swap(other.chunks_);
        std::swap(size_, other.size_);
        std::swap(elems_per_chunk_, other.elems_per_chunk_);
        std::swap(shift_, other.shift_);
        std::swap(mask_, other.mask_);
    }

    ChunkPool *pool_ = nullptr;
    std::vector<T *> chunks_{}; //!< chunk c holds elements [c * elems_per_chunk_, (c+1) * elems_per_chunk_)
    size_t size_ = 0;
    size_t elems_per_chunk_ = 0;
    size_t shift_ = 0;
    size_t mask_ = 0;
};

} // namespace monoprop::detail
