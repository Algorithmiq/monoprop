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

/*
 * Allocation observation and controlled allocation failure for the unit-test executable.
 *
 * AllocationProbe.cpp replaces the global operator new/delete family for the whole of
 * monoprop_unit_tests.x (and only it; the library and installed consumers keep the default). While
 * nothing is armed the replacement is a plain malloc/free pass-through that also counts, per thread,
 * the bytes requested through operator new. A test can make the n-th following operator new on the
 * calling thread throw std::bad_alloc (the nothrow forms return nullptr), which exercises the real
 * allocation sites of a constructor instead of a callback that merely throws.
 *
 * Live-byte tracking is process-wide and only meaningful while no unrelated thread allocates.
 */

#include <cstddef>

/*
 * Sanitizer runtimes linked statically (Clang's ASan/TSan, GCC's TSan) define the operator new/delete family
 * themselves, so the replacement is compiled out there: available() is false, thread_bytes() stays 0 and nothing
 * can be armed. Cases that need the probe skip through a precondition.
 */
#if defined(__SANITIZE_THREAD__)
#define MONOPROP_TEST_ALLOCATION_PROBE 0
#elif defined(__clang__) && defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define MONOPROP_TEST_ALLOCATION_PROBE 0
#endif
#endif
#ifndef MONOPROP_TEST_ALLOCATION_PROBE
#define MONOPROP_TEST_ALLOCATION_PROBE 1
#endif

namespace test_utils::allocation {

//! Whether this build replaces operator new/delete, so the functions below observe and inject anything.
constexpr auto available() noexcept -> bool {
    return MONOPROP_TEST_ALLOCATION_PROBE != 0;
}

//! Bytes requested through operator new on the calling thread since it started.
auto thread_bytes() noexcept -> std::size_t;

//! Make the `nth` (>= 1) next operator new on the calling thread fail; 0 disarms.
auto arm_failure(std::size_t nth) noexcept -> void;

//! Disarm the calling thread; returns true if an armed failure had not fired yet.
auto disarm_failure() noexcept -> bool;

/*!
 * \brief Process-wide live-byte accounting over a scope: blocks allocated minus blocks freed, in usable
 *        bytes, through operator new/delete while the scope is active. Not nestable.
 */
class LiveBytes {
public:
    LiveBytes() noexcept;                                      //!< Starts tracking from zero.
    ~LiveBytes();                                              //!< Stops tracking.
    LiveBytes(const LiveBytes &) = delete;                     //!< One active scope.
    auto operator=(const LiveBytes &) -> LiveBytes & = delete; //!< One active scope.

    //! Net usable bytes allocated and not yet freed since construction.
    [[nodiscard]] auto net() const noexcept -> long long;
};

} // namespace test_utils::allocation
