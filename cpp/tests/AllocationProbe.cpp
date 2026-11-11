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

#include "AllocationProbe.h"

#if defined(__APPLE__)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

/*
 * Replacement of every replaceable global allocation and deallocation function, so allocation and
 * deallocation stay paired on malloc/free under ASan/TSan, whose own operator new/delete this replaces.
 * The thread-local counters are constant-initialized PODs, so reading them inside operator new needs no
 * dynamic TLS initialization. No new_handler is consulted: a test build has none installed.
 */

namespace {

auto usable_size(void *block) noexcept -> std::size_t {
#if defined(__APPLE__)
    return malloc_size(block);
#else
    return malloc_usable_size(block);
#endif
}

thread_local std::size_t t_bytes = 0;
thread_local std::size_t t_fail_after = 0; // 0: disarmed; n: the n-th next allocation fails

std::atomic<bool> g_tracking{false};
std::atomic<long long> g_live{0};

auto should_fail() noexcept -> bool {
    return t_fail_after != 0 && --t_fail_after == 0;
}

auto track_allocation(void *block) noexcept -> void {
    if (block != nullptr && g_tracking.load(std::memory_order_relaxed)) {
        g_live.fetch_add(static_cast<long long>(usable_size(block)), std::memory_order_relaxed);
    }
}

auto release(void *block) noexcept -> void {
    if (block != nullptr && g_tracking.load(std::memory_order_relaxed)) {
        g_live.fetch_sub(static_cast<long long>(usable_size(block)), std::memory_order_relaxed);
    }
    std::free(block);
}

auto allocate(std::size_t size, std::size_t alignment) noexcept -> void * {
    if (should_fail()) {
        return nullptr;
    }
    t_bytes += size;
    size = size == 0 ? 1 : size;
    void *block = nullptr;
    if (alignment <= alignof(std::max_align_t)) {
        block = std::malloc(size);
    }
    else {
        block = std::aligned_alloc(alignment, (size + alignment - 1) / alignment * alignment);
    }
    track_allocation(block);
    return block;
}

auto allocate_or_throw(std::size_t size, std::size_t alignment) -> void * {
    void *block = allocate(size, alignment);
    if (block == nullptr) {
        throw std::bad_alloc();
    }
    return block;
}

} // namespace

namespace test_utils::allocation {

auto thread_bytes() noexcept -> std::size_t {
    return t_bytes;
}

auto arm_failure(std::size_t nth) noexcept -> void {
    t_fail_after = nth;
}

auto disarm_failure() noexcept -> bool {
    const bool pending = t_fail_after != 0;
    t_fail_after = 0;
    return pending;
}

LiveBytes::LiveBytes() noexcept {
    g_live.store(0);
    g_tracking.store(true);
}

LiveBytes::~LiveBytes() {
    g_tracking.store(false);
}

auto LiveBytes::net() const noexcept -> long long {
    return g_live.load();
}

} // namespace test_utils::allocation

#if MONOPROP_TEST_ALLOCATION_PROBE
// NOLINTBEGIN(misc-new-delete-overloads)
auto operator new(std::size_t size) -> void * {
    return allocate_or_throw(size, alignof(std::max_align_t));
}
auto operator new[](std::size_t size) -> void * {
    return allocate_or_throw(size, alignof(std::max_align_t));
}
auto operator new(std::size_t size, std::align_val_t alignment) -> void * {
    return allocate_or_throw(size, static_cast<std::size_t>(alignment));
}
auto operator new[](std::size_t size, std::align_val_t alignment) -> void * {
    return allocate_or_throw(size, static_cast<std::size_t>(alignment));
}
auto operator new(std::size_t size, const std::nothrow_t &) noexcept -> void * {
    return allocate(size, alignof(std::max_align_t));
}
auto operator new[](std::size_t size, const std::nothrow_t &) noexcept -> void * {
    return allocate(size, alignof(std::max_align_t));
}
auto operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept -> void * {
    return allocate(size, static_cast<std::size_t>(alignment));
}
auto operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept -> void * {
    return allocate(size, static_cast<std::size_t>(alignment));
}

auto operator delete(void *block) noexcept -> void {
    release(block);
}
auto operator delete[](void *block) noexcept -> void {
    release(block);
}
auto operator delete(void *block, std::size_t) noexcept -> void {
    release(block);
}
auto operator delete[](void *block, std::size_t) noexcept -> void {
    release(block);
}
auto operator delete(void *block, std::align_val_t) noexcept -> void {
    release(block);
}
auto operator delete[](void *block, std::align_val_t) noexcept -> void {
    release(block);
}
auto operator delete(void *block, std::size_t, std::align_val_t) noexcept -> void {
    release(block);
}
auto operator delete[](void *block, std::size_t, std::align_val_t) noexcept -> void {
    release(block);
}
auto operator delete(void *block, const std::nothrow_t &) noexcept -> void {
    release(block);
}
auto operator delete[](void *block, const std::nothrow_t &) noexcept -> void {
    release(block);
}
auto operator delete(void *block, std::align_val_t, const std::nothrow_t &) noexcept -> void {
    release(block);
}
auto operator delete[](void *block, std::align_val_t, const std::nothrow_t &) noexcept -> void {
    release(block);
}
// NOLINTEND(misc-new-delete-overloads)
#endif
