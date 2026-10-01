# Shared-core foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or
> executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land on `main` the groundwork the GPU port builds on: the `Backend` template parameter on
`MonomialPropagator`, the `monoprop_HOST_DEVICE` macro, the shared bit helpers, a shared `Bitset`, and a
CI check that compiles the shared headers with nvcc.

**Architecture:** A new `cpp/monoprop/shared/` directory holds code compiled for both host and device.
Functions there are marked `monoprop_HOST_DEVICE` (empty outside nvcc) and call only `monoprop::shared`
helpers, which pick `std::` on the host, CUDA intrinsics on the device, and a portable branch during
constant evaluation. `Bitset` moves into the shared core; its host-only extras stay in
`monoprop/Bitset.h`. Nothing about the CPU results changes: a golden baseline proves it byte for byte.

**Out of scope:** the rest of spec Section 7's shared core (per-term algebra, cutoff checks, fold-word
masking, partner merge, term hashing, routing arithmetic, `QueryWire`, replay formulas, data views) and
the overflow-row layout (D12). Those move into the shared core with the Stage 1 and Stage 2 plans, as the
device code that needs them is written.

**Tech Stack:** C++23 (GCC ≥ 14, Clang ≥ 18), Boost.Test, CMake via scikit-build-core, `just`, nvcc 13.3
(CI container only), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-01-gpu-port-design.md` (on branch `feat-gpu-port`; read it
from this plan's branch with `git show feat-gpu-port:docs/superpowers/specs/2026-10-01-gpu-port-design.md`),
Sections 5 and 7, decisions D3, D6 and D7, and Appendix B.

## Global Constraints

- Branch: `feat/shared-core-foundation`, created from `main`. One PR into `main` (D3).
- C++23; the CPU build must keep compiling with GCC 14 and Clang 18 (CI lanes `g++-14`, `clang++-18`).
- Shared code: `monoprop_HOST_DEVICE` expands to `__host__ __device__` under nvcc (`__CUDACC__`) and to
  nothing otherwise.
- Shared code uses only type-level standard-library facilities (fixed-width integers, `<type_traits>`)
  and `monoprop::shared` helpers. Host/device selection happens inside the helpers only: `__CUDA_ARCH__`
  chooses CUDA intrinsics over `std::`, and `if consteval` covers compile-time evaluation.
- nvcc flags for shared code: `-std=c++23`, no `--expt-relaxed-constexpr`,
  `-Werror cross-execution-space-call`, `-arch=sm_80`.
- Type changes stay layout-compatible (size, alignment, trivially copyable, MPI/`memcpy` safety).
- The CPU engine never includes device headers and contains no `#ifdef monoprop_ENABLE_CUDA`.
- No CPU regression beyond noise; results byte-identical against the golden baseline (`just diff-baseline`).
- Repository rules (`AGENTS.md`): trailing return types; snake_case for variables and functions; only
  private members end in `_`; Qt-style Doxygen (`///`) on header declarations; `//` comments only for
  invariants and non-obvious choices; the Apache license header on every new C++ file; `clang-format`.
- Commits: `<type>(<scope>): <gitmoji> <description>`, with the trailer `Assisted-by: <harness>:<model>`
  for agent commits (the executor's own harness and model, for example `Assisted-by: Pi:claude-opus-5-5`)
  and no `Co-authored-by`. Use plain `git` (not GitButler).
- Run `prek run --all-files` and the relevant tests before every push. `prek` comes with the default
  `dev` dependency group; if it is not on `PATH`, run it as `uv run --no-sync prek ...`.

The license header every new C++ (`.h`, `.cpp`, `.cu`) file starts with:

```cpp
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
```

Build and test commands used throughout:

- Build (editable install plus the C++ tree): `just build`
- Rebuild only the C++ test binary after editing C++: `cmake --build build/editable/Release --target monoprop_unit_tests.x -j`
  (this builds the existing tree; never configure it directly)
- Run one Boost case: `build/editable/Release/bin/monoprop_unit_tests.x --run_test=<case_name>`
- Full C++ suite: `just test-cpp`; Python suite: `just test-py`

## File structure

| File | Responsibility |
|---|---|
| `cpp/include/monoprop/MonomialPropagatorFwd.h` (new) | Backend tags and the one declaration of `MonomialPropagator<NumModes, Backend = backend::Cpu>` |
| `cpp/include/monoprop/MonomialPropagator.h` | The CPU class becomes the `backend::Cpu` partial specialization |
| `cpp/include/monoprop/CMakeLists.txt` | Installs the forward-declaration header |
| `cpp/monoprop/detail/partition/PartitionGroup.h` | Uses the forward-declaration header instead of its own declaration |
| `cpp/monoprop/shared/HostDevice.h` (new) | The `monoprop_HOST_DEVICE` macro |
| `cpp/monoprop/shared/Bits.h` (new) | `popcount`, `parity`, `countr_zero`, `bit_width` for host, device and constant evaluation |
| `cpp/monoprop/shared/Bitset.h` (new) | `Bitset` and `SplitmixHash<Bitset>`, device-safe |
| `cpp/monoprop/shared/CMakeLists.txt` (new) | Installs the shared headers |
| `cpp/monoprop/CMakeLists.txt` | Adds the `shared` subdirectory |
| `cpp/monoprop/Bitset.h` | Host-only extras (`operator<<`, `std::hash`) on top of the shared `Bitset` |
| `cpp/tests/monomial_propagator_backend_tests.cpp` (new) | Backend default and incompleteness checks |
| `cpp/tests/shared/BitsVectors.h` (new) | Fixed test words, reused later by the device half of the equivalence tests |
| `cpp/tests/shared_bits_tests.cpp` (new) | Host half of the bit-helper equivalence tests |
| `cpp/tests/bitset_tests.cpp` | Adds constant-evaluation and layout characterization tests |
| `cpp/tests/shared/nvcc_compile_check.cu` (new) | Every shared function called from device code |
| `cpp/tests/shared/nvcc_negative_check.cu` (new) | Device code calling a host-only function; must be rejected |
| `tools/check-shared-nvcc.sh` (new) | The check itself: compile the shared headers as device code, require the negative check to fail |
| `justfile` | `check-shared-nvcc` (local nvcc) and `check-shared-nvcc-docker` (CUDA container) recipes |
| `.github/workflows/shared-nvcc.yml` (new) | Runs `just check-shared-nvcc-docker` on a standard runner |
| `docs/content/docs/testing.mdx` | Documents the shared headers and the recipe |
| `AGENTS.md` | One rule for code in `cpp/monoprop/shared/` |

---

### Task 1: `Backend` template parameter on `MonomialPropagator`

**Files:**
- Create: `cpp/include/monoprop/MonomialPropagatorFwd.h`
- Modify: `cpp/include/monoprop/MonomialPropagator.h` (includes block and the class head at `template <size_t NumModes>\nclass MonomialPropagator {`)
- Modify: `cpp/include/monoprop/CMakeLists.txt`
- Modify: `cpp/monoprop/detail/partition/PartitionGroup.h` (includes block; the forward declaration after `namespace monoprop {`)
- Test: `cpp/tests/monomial_propagator_backend_tests.cpp`

**Interfaces:**
- Produces: `namespace monoprop::backend { struct Cpu; struct Cuda; }` and
  `template <size_t NumModes, typename Backend = backend::Cpu> class MonomialPropagator;` in
  `monoprop/MonomialPropagatorFwd.h`. `MonomialPropagator<N>` names
  `MonomialPropagator<N, backend::Cpu>`; `MonomialPropagator<N, backend::Cuda>` stays incomplete.

- [ ] **Step 1: Create the branch and capture the golden baseline**

```bash
git switch main && git pull
git switch -c feat/shared-core-foundation
just build
just capture-baseline golden
```

Expected: `.baseline-capture/golden/` exists. Every later `just diff-baseline` compares against it.

- [ ] **Step 2: Write the failing test**

Create `cpp/tests/monomial_propagator_backend_tests.cpp` (license header first):

```cpp
// The Backend parameter: the default must keep naming today's CPU class, and a backend without a
// visible specialization must stay incomplete instead of silently becoming the CPU class.

#include <boost/test/unit_test.hpp>

#include <type_traits>

#include "monoprop/MonomialPropagator.h"

namespace {

template <typename T, typename = void>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

using monoprop::MonomialPropagator;
namespace backend = monoprop::backend;

} // namespace

static_assert(std::is_same_v<MonomialPropagator<64>, MonomialPropagator<64, backend::Cpu>>);
static_assert(is_complete<MonomialPropagator<64>>::value);
// The CUDA specialization exists only in CUDA translation units on the GPU branch.
static_assert(!is_complete<MonomialPropagator<64, backend::Cuda>>::value);

BOOST_AUTO_TEST_CASE(monomial_propagator_default_backend_is_cpu) {
    BOOST_TEST((std::is_same_v<MonomialPropagator<64>, MonomialPropagator<64, backend::Cpu>>));
}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cmake --build build/editable/Release --target monoprop_unit_tests.x -j`
Expected: compile error in `monomial_propagator_backend_tests.cpp`: `monoprop::backend` has not been
declared, or `MonomialPropagator` takes one template argument.

- [ ] **Step 4: Add the forward-declaration header**

Create `cpp/include/monoprop/MonomialPropagatorFwd.h` (license header first):

```cpp
#pragma once

#include <cstddef>

namespace monoprop {

/// Tags selecting the engine behind a MonomialPropagator.
namespace backend {
/// The CPU engine: today's MonomialPropagator, the default.
struct Cpu;
/// The CUDA engine. Its specialization is visible only in CUDA translation units of the GPU port.
struct Cuda;
} // namespace backend

/// Majorana/Pauli propagator; `Backend` selects the engine. The primary template is never defined, so
/// naming a backend whose specialization is not visible fails to compile rather than silently using
/// another engine. This is the only declaration that may carry the default argument.
template <size_t NumModes, typename Backend = backend::Cpu>
class MonomialPropagator;

} // namespace monoprop
```

- [ ] **Step 5: Make the CPU class the `backend::Cpu` partial specialization**

In `cpp/include/monoprop/MonomialPropagator.h`, add the include next to the other `monoprop/` includes:

```cpp
#include "monoprop/MonomialPropagatorFwd.h"
```

Replace the class head

```cpp
template <size_t NumModes>
class MonomialPropagator {
public:
```

with

```cpp
/// The CPU engine. Out-of-line members keep the `MonomialPropagator<NumModes>` spelling, which names
/// this specialization through the default `Backend`.
template <size_t NumModes>
class MonomialPropagator<NumModes, backend::Cpu> {
public:
```

Leave every other line of the class and of `MonomialPropagator.inl` unchanged: out-of-line members spelled
`MonomialPropagator<NumModes>::` name this specialization through the default argument.

- [ ] **Step 6: Use the header in `PartitionGroup.h`**

In `cpp/monoprop/detail/partition/PartitionGroup.h`, add to the includes:

```cpp
#include "monoprop/MonomialPropagatorFwd.h" // completed before any PartitionGroup member body is instantiated
```

and delete the forward declaration inside `namespace monoprop {`:

```cpp
template <size_t NumModes>
class MonomialPropagator; // completed before any PartitionGroup member body is instantiated (Impl.h)
```

- [ ] **Step 7: Install the new header**

In `cpp/include/monoprop/CMakeLists.txt`, add to the `api_headers` file list, after the
`MonomialPropagator.h` line:

```cmake
        "${PROJECT_SOURCE_DIR}/cpp/include/${PROJECT_NAME}/MonomialPropagatorFwd.h"
```

- [ ] **Step 8: Run the test and the suite**

Run: `just build && build/editable/Release/bin/monoprop_unit_tests.x --run_test=monomial_propagator_default_backend_is_cpu`
Expected: `*** No errors detected`.

Run: `just test-cpp && just test-py && just test-find-package`
Expected: all pass.

Run: `just diff-baseline`
Expected: no output from `diff -rq` (byte-identical).

- [ ] **Step 9: Commit**

```bash
git add cpp/include/monoprop/MonomialPropagatorFwd.h cpp/include/monoprop/MonomialPropagator.h \
        cpp/include/monoprop/CMakeLists.txt cpp/monoprop/detail/partition/PartitionGroup.h \
        cpp/tests/monomial_propagator_backend_tests.cpp
git commit -m "refactor(core): ♻️ add a Backend template parameter to MonomialPropagator" \
           -m "Assisted-by: <harness>:<model>"
```

---

### Task 2: `monoprop_HOST_DEVICE` and the shared bit helpers

**Files:**
- Create: `cpp/monoprop/shared/HostDevice.h`
- Create: `cpp/monoprop/shared/Bits.h`
- Create: `cpp/monoprop/shared/CMakeLists.txt`
- Modify: `cpp/monoprop/CMakeLists.txt` (the `add_subdirectory` block)
- Create: `cpp/tests/shared/BitsVectors.h`
- Test: `cpp/tests/shared_bits_tests.cpp`

**Interfaces:**
- Produces, in `monoprop/shared/Bits.h`, namespace `monoprop::shared`, all
  `monoprop_HOST_DEVICE constexpr ... noexcept`:
  - `auto popcount(uint64_t x) -> int`
  - `auto parity(uint64_t x) -> bool` (true when `popcount(x)` is odd)
  - `auto countr_zero(uint64_t x) -> int` (64 for `x == 0`)
  - `auto bit_width(uint64_t x) -> int` (0 for `x == 0`)
  - portable references in `monoprop::shared::bits_detail`: `popcount_portable`,
    `countr_zero_portable`, `bit_width_portable` (same signatures).
- Produces `monoprop::test::bits_test_vectors` (`std::array<uint64_t, 16>`) in `cpp/tests/shared/BitsVectors.h`.

- [ ] **Step 1: Write the failing test**

Create `cpp/tests/shared/BitsVectors.h` (license header first):

```cpp
#pragma once

#include <array>
#include <cstdint>

namespace monoprop::test {

/// Words the shared bit helpers are checked on: here on the host, and in a kernel by the device half of
/// the equivalence tests on the GPU branch. Edges (empty, full, single bits at the word ends and the
/// 32-bit seam) plus fixed mixed patterns.
inline constexpr std::array<uint64_t, 16> bits_test_vectors{
    0x0000000000000000ULL, 0x0000000000000001ULL, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
    0x0000000080000000ULL, 0x0000000100000000ULL, 0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL,
    0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL, 0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL, 0x0000000000001000ULL,
};

} // namespace monoprop::test
```

Create `cpp/tests/shared_bits_tests.cpp` (license header first):

```cpp
// Host half of the shared bit helpers' equivalence tests: each helper against the standard library, on
// the fixed vectors the device half replays and on random words. The static_asserts cover the
// constant-evaluation branch.

#include <boost/test/unit_test.hpp>

#include <bit>
#include <cstdint>
#include <random>

#include "monoprop/shared/Bits.h"
#include "shared/BitsVectors.h"

namespace shared = monoprop::shared;

static_assert(shared::popcount(0) == 0);
static_assert(shared::popcount(~uint64_t{0}) == 64);
static_assert(shared::popcount(0x8000000000000001ULL) == 2);
static_assert(shared::parity(0b1011ULL));
static_assert(!shared::parity(0b1001ULL));
static_assert(shared::countr_zero(0) == 64);
static_assert(shared::countr_zero(1) == 0);
static_assert(shared::countr_zero(0x8000000000000000ULL) == 63);
static_assert(shared::bit_width(0) == 0);
static_assert(shared::bit_width(1) == 1);
static_assert(shared::bit_width(255) == 8);
static_assert(shared::bit_width(~uint64_t{0}) == 64);

namespace {

auto check_against_std(uint64_t x) -> void {
    const int expected_popcount = std::popcount(x);
    const int expected_countr_zero = std::countr_zero(x);
    const int expected_bit_width = static_cast<int>(std::bit_width(x));
    BOOST_CHECK_MESSAGE(shared::popcount(x) == expected_popcount, "popcount of " << x);
    BOOST_CHECK_MESSAGE(shared::parity(x) == ((expected_popcount & 1) != 0), "parity of " << x);
    BOOST_CHECK_MESSAGE(shared::countr_zero(x) == expected_countr_zero, "countr_zero of " << x);
    BOOST_CHECK_MESSAGE(shared::bit_width(x) == expected_bit_width, "bit_width of " << x);
    BOOST_CHECK_MESSAGE(shared::bits_detail::popcount_portable(x) == expected_popcount, "portable popcount of " << x);
    BOOST_CHECK_MESSAGE(shared::bits_detail::countr_zero_portable(x) == expected_countr_zero,
                        "portable countr_zero of " << x);
    BOOST_CHECK_MESSAGE(shared::bits_detail::bit_width_portable(x) == expected_bit_width,
                        "portable bit_width of " << x);
}

} // namespace

BOOST_AUTO_TEST_CASE(shared_bits_fixed_vectors) {
    for (const uint64_t x : monoprop::test::bits_test_vectors) {
        check_against_std(x);
    }
}

BOOST_AUTO_TEST_CASE(shared_bits_random_words) {
    std::mt19937_64 rng(20261001);
    for (int i = 0; i < 4096; ++i) {
        const uint64_t x = rng();
        check_against_std(x);
        check_against_std(x & rng()); // sparser words
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build build/editable/Release --target monoprop_unit_tests.x -j`
Expected: compile error: `monoprop/shared/Bits.h: No such file or directory`.

- [ ] **Step 3: Add the macro header**

Create `cpp/monoprop/shared/HostDevice.h` (license header first):

```cpp
#pragma once

/// Marks a function compiled for both host and device. Expands to `__host__ __device__` under nvcc and
/// to nothing elsewhere, so the CPU build sees plain C++.
#if defined(__CUDACC__)
#define monoprop_HOST_DEVICE __host__ __device__
#else
#define monoprop_HOST_DEVICE
#endif
```

- [ ] **Step 4: Add the bit helpers**

Create `cpp/monoprop/shared/Bits.h` (license header first):

```cpp
#pragma once

#include <bit>
#include <cstdint>

#include "monoprop/shared/HostDevice.h"

/*
 * Bit primitives for code shared by host and device. Standard-library bit functions must not be called
 * from device code: nvcc treats them as host-only, and reached through --expt-relaxed-constexpr they
 * miscompiled silently (spec, Appendix A.3). Each helper picks std:: on the host, a CUDA intrinsic on
 * the device, and a portable branch during constant evaluation, where neither may be called.
 */

namespace monoprop::shared {

namespace bits_detail {

/// Portable popcount (SWAR), for constant evaluation and as the tests' reference.
monoprop_HOST_DEVICE constexpr auto popcount_portable(uint64_t x) noexcept -> int {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<int>((x * 0x0101010101010101ULL) >> 56);
}

/// Portable count of trailing zeros; 64 for zero.
monoprop_HOST_DEVICE constexpr auto countr_zero_portable(uint64_t x) noexcept -> int {
    if (x == 0) {
        return 64;
    }
    // (x & -x) - 1 sets exactly the bits below the lowest set bit.
    return popcount_portable((x & (~x + 1)) - 1);
}

/// Portable number of bits needed to represent x; 0 for zero.
monoprop_HOST_DEVICE constexpr auto bit_width_portable(uint64_t x) noexcept -> int {
    int width = 0;
    while (x != 0) {
        x >>= 1;
        ++width;
    }
    return width;
}

} // namespace bits_detail

/// Number of set bits.
monoprop_HOST_DEVICE constexpr auto popcount(uint64_t x) noexcept -> int {
    if consteval {
        return bits_detail::popcount_portable(x);
    }
    else {
#if defined(__CUDA_ARCH__)
        return __popcll(x);
#else
        return std::popcount(x);
#endif
    }
}

/// True when the number of set bits is odd.
monoprop_HOST_DEVICE constexpr auto parity(uint64_t x) noexcept -> bool {
    return (popcount(x) & 1) != 0;
}

/// Number of trailing zero bits; 64 for zero.
monoprop_HOST_DEVICE constexpr auto countr_zero(uint64_t x) noexcept -> int {
    if consteval {
        return bits_detail::countr_zero_portable(x);
    }
    else {
#if defined(__CUDA_ARCH__)
        return x == 0 ? 64 : __ffsll(static_cast<long long>(x)) - 1;
#else
        return std::countr_zero(x);
#endif
    }
}

/// Number of bits needed to represent x; 0 for zero.
monoprop_HOST_DEVICE constexpr auto bit_width(uint64_t x) noexcept -> int {
    if consteval {
        return bits_detail::bit_width_portable(x);
    }
    else {
#if defined(__CUDA_ARCH__)
        return 64 - __clzll(static_cast<long long>(x));
#else
        return static_cast<int>(std::bit_width(x));
#endif
    }
}

} // namespace monoprop::shared
```

- [ ] **Step 5: Install the shared headers**

Create `cpp/monoprop/shared/CMakeLists.txt`:

```cmake
target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Bits.h"
        "HostDevice.h"
)
```

In `cpp/monoprop/CMakeLists.txt`, extend the `add_subdirectory` block:

```cmake
add_subdirectory(algebra)
add_subdirectory(core)
add_subdirectory(detail)
add_subdirectory(shared)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `just build && build/editable/Release/bin/monoprop_unit_tests.x --run_test=shared_bits_fixed_vectors,shared_bits_random_words`
Expected: `*** No errors detected`.

- [ ] **Step 7: Commit**

```bash
git add cpp/monoprop/shared/HostDevice.h cpp/monoprop/shared/Bits.h cpp/monoprop/shared/CMakeLists.txt \
        cpp/monoprop/CMakeLists.txt cpp/tests/shared/BitsVectors.h cpp/tests/shared_bits_tests.cpp
git commit -m "refactor(shared): ♻️ add monoprop_HOST_DEVICE and the shared bit helpers" \
           -m "Assisted-by: <harness>:<model>"
```

---

### Task 3: Shared `Bitset`

**Files:**
- Create: `cpp/monoprop/shared/Bitset.h`
- Modify (rewrite): `cpp/monoprop/Bitset.h`
- Modify: `cpp/monoprop/shared/CMakeLists.txt`
- Test: `cpp/tests/bitset_tests.cpp` (append), `cpp/tests/shared_bits_tests.cpp` (append)

**Interfaces:**
- Consumes: `monoprop::shared::popcount`, `monoprop::shared::countr_zero` (Task 2).
- Produces: `monoprop::Bitset<NumBits>` and `SplitmixHash<monoprop::Bitset<NumBits>>` (global namespace)
  in `monoprop/shared/Bitset.h`, every member `monoprop_HOST_DEVICE`; the public interface is unchanged
  (`count`, `test`, `any`, `none`, `size`, `count_and`, `parity_and`, `set`, `&= |= ^=`, `~`, `& | ^`,
  `>>= >>`, `==`, `num_words`, `data`, `word`, `find_first`, `find_next`). `monoprop/Bitset.h` adds
  `operator<<` and `std::hash`.

- [ ] **Step 1: Write the characterization tests**

Append to `cpp/tests/bitset_tests.cpp` (it already has `using monoprop::Bitset;` and includes
`<cstdint>`; add `#include <type_traits>` to its includes):

```cpp
// Moving Bitset into the shared core must keep its layout: MPI and memcpy move it as raw words.
static_assert(sizeof(Bitset<130>) == 3 * sizeof(uint64_t));
static_assert(alignof(Bitset<130>) == alignof(uint64_t));
static_assert(std::is_trivially_copyable_v<Bitset<130>>);

// Bitset is constexpr end to end, which exercises the shared bit helpers' constant-evaluation branch.
static_assert([] {
    Bitset<130> b;
    b.set(0).set(64).set(129);
    return b.count() == 3 && b.find_first() == 0 && b.find_next(0) == 64 && b.find_next(64) == 129
           && b.find_next(129) == 130 && (b >> 64).count() == 2;
}());

BOOST_AUTO_TEST_CASE(bitset_stream_output_msb_first) {
    Bitset<10> b;
    b.set(0).set(9);
    std::ostringstream out;
    out << b;
    BOOST_TEST(out.str() == "1000000001");
}
```

Add `#include <sstream>` to the includes of `cpp/tests/bitset_tests.cpp`.

Append to `cpp/tests/shared_bits_tests.cpp` (add `#include "monoprop/shared/Bitset.h"` to its includes;
this test only sees the shared header, which proves the header stands alone):

```cpp
BOOST_AUTO_TEST_CASE(shared_bitset_without_host_extras) {
    monoprop::Bitset<192> b;
    b.set(1).set(64).set(191);
    BOOST_TEST(b.count() == 3U);
    BOOST_TEST(b.find_first() == 1U);
    BOOST_TEST(b.find_next(64) == 191U);
    BOOST_TEST(SplitmixHash<monoprop::Bitset<192>>{}(b) == SplitmixHash<monoprop::Bitset<192>>{}(b));
}
```

- [ ] **Step 2: Run them against the current `Bitset`**

Run: `cmake --build build/editable/Release --target monoprop_unit_tests.x -j`
Expected: compile error in `shared_bits_tests.cpp` only: `monoprop/shared/Bitset.h: No such file or
directory`. The `bitset_tests.cpp` additions compile against today's header.

- [ ] **Step 3: Create the shared `Bitset`**

Create `cpp/monoprop/shared/Bitset.h` (license header first):

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

#include "monoprop/shared/Bits.h"
#include "monoprop/shared/HostDevice.h"

namespace monoprop {

/// Fixed-width bitset over contiguous `uint64_t` words, shared by host and device code: zero-copy MPI,
/// word-wise hashing, portable bit scanning, `memcpy`-safe. Stream output and `std::hash` are host-only
/// and live in `monoprop/Bitset.h`.
template <size_t NumBits>
class Bitset {
    static_assert(NumBits > 0, "Bitset requires at least 1 bit");

    using word_type = uint64_t;
    static constexpr auto word_width = sizeof(word_type) * 8;

    static constexpr auto kNumWords = (NumBits + word_width - 1) / word_width;
    static constexpr auto kTopBits = NumBits % word_width;
    static constexpr auto kTopMask = kTopBits ? ((word_type{1} << kTopBits) - 1) : ~word_type{0};

    // A plain array, not std::array: nvcc treats std::array's members as host-only. Same layout.
    word_type words_[kNumWords]{};

    monoprop_HOST_DEVICE constexpr auto sanitize_top() noexcept -> void {
        if constexpr (kTopBits != 0) {
            words_[kNumWords - 1] &= kTopMask;
        }
    }

public:
    constexpr Bitset() noexcept = default;

    monoprop_HOST_DEVICE constexpr explicit(false) Bitset(uint64_t val) noexcept : words_{val} { sanitize_top(); }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto count() const noexcept -> size_t {
        size_t c = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            c += static_cast<size_t>(shared::popcount(words_[i]));
        return c;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto test(size_t pos) const noexcept -> bool {
        return (words_[pos / word_width] >> (pos % word_width)) & 1;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto any() const noexcept -> bool {
        for (size_t i = 0; i < kNumWords; ++i)
            if (words_[i])
                return true;
        return false;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto none() const noexcept -> bool { return !any(); }

    [[nodiscard]] monoprop_HOST_DEVICE static constexpr auto size() noexcept -> size_t { return NumBits; }

    // popcount(*this & other) without materializing the temporary.
    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto count_and(const Bitset &o) const noexcept -> size_t {
        size_t c = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            c += static_cast<size_t>(shared::popcount(words_[i] & o.words_[i]));
        return c;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto parity_and(const Bitset &o) const noexcept -> bool {
        word_type parity_word = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            parity_word ^= words_[i] & o.words_[i];
        return shared::parity(parity_word);
    }

    monoprop_HOST_DEVICE constexpr auto set(size_t pos) noexcept -> Bitset & {
        words_[pos / word_width] |= uint64_t(1) << (pos % word_width);
        return *this;
    }

    monoprop_HOST_DEVICE constexpr auto operator&=(const Bitset &rhs) noexcept -> Bitset & {
        for (auto i = 0uz; i < kNumWords; ++i)
            words_[i] &= rhs.words_[i];
        return *this;
    }

    monoprop_HOST_DEVICE constexpr auto operator|=(const Bitset &rhs) noexcept -> Bitset & {
        for (auto i = 0uz; i < kNumWords; ++i)
            words_[i] |= rhs.words_[i];
        return *this;
    }

    monoprop_HOST_DEVICE constexpr auto operator^=(const Bitset &rhs) noexcept -> Bitset & {
        for (auto i = 0uz; i < kNumWords; ++i)
            words_[i] ^= rhs.words_[i];
        return *this;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto operator~() const noexcept -> Bitset {
        Bitset r = *this;
        for (auto i = 0uz; i < kNumWords; ++i)
            r.words_[i] = ~r.words_[i];
        r.sanitize_top();
        return r;
    }

    [[nodiscard]] friend monoprop_HOST_DEVICE constexpr auto operator&(const Bitset &lhs, const Bitset &rhs) noexcept
        -> Bitset {
        Bitset r = lhs;
        r &= rhs;
        return r;
    }

    [[nodiscard]] friend monoprop_HOST_DEVICE constexpr auto operator|(const Bitset &lhs, const Bitset &rhs) noexcept
        -> Bitset {
        Bitset r = lhs;
        r |= rhs;
        return r;
    }

    [[nodiscard]] friend monoprop_HOST_DEVICE constexpr auto operator^(const Bitset &lhs, const Bitset &rhs) noexcept
        -> Bitset {
        Bitset r = lhs;
        r ^= rhs;
        return r;
    }

    monoprop_HOST_DEVICE constexpr auto operator>>=(size_t pos) noexcept -> Bitset & {
        if (pos >= NumBits) {
            for (auto &w : words_)
                w = 0;
            return *this;
        }
        if constexpr (kNumWords == 1) {
            words_[0] >>= pos;
        }
        else {
            const size_t word_shift = pos / word_width;
            const size_t limit = kNumWords - word_shift;
            if (const size_t bit_shift = pos % word_width; bit_shift == 0) {
                for (size_t i = 0; i < limit; ++i)
                    words_[i] = words_[i + word_shift];
            }
            else {
                const size_t inv_shift = word_width - bit_shift;
                for (size_t i = 0; i + 1 < limit; ++i) {
                    words_[i] = (words_[i + word_shift] >> bit_shift) | (words_[i + word_shift + 1] << inv_shift);
                }
                words_[limit - 1] = words_[kNumWords - 1] >> bit_shift;
            }
            for (size_t i = limit; i < kNumWords; ++i)
                words_[i] = 0;
        }
        return *this;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto operator>>(size_t pos) const noexcept -> Bitset {
        Bitset r = *this;
        r >>= pos;
        return r;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto operator==(const Bitset &o) const noexcept -> bool {
        for (size_t i = 0; i < kNumWords; ++i)
            if (words_[i] != o.words_[i])
                return false;
        return true;
    }

    [[nodiscard]] monoprop_HOST_DEVICE static constexpr auto num_words() noexcept -> size_t { return kNumWords; }
    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto data() const noexcept -> const uint64_t * { return words_; }
    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto data() noexcept -> uint64_t * { return words_; }
    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto word(size_t i) const noexcept -> uint64_t { return words_[i]; }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto find_first() const noexcept -> size_t { // NumBits if none
        for (size_t i = 0; i < kNumWords; ++i) {
            if (words_[i])
                return (i * word_width) + static_cast<size_t>(shared::countr_zero(words_[i]));
        }
        return NumBits;
    }

    [[nodiscard]] monoprop_HOST_DEVICE constexpr auto find_next(size_t pos) const noexcept -> size_t { // NumBits if none
        if (++pos >= NumBits)
            return NumBits;
        if constexpr (kNumWords == 1) {
            if (const uint64_t w = words_[0] >> pos; w)
                return pos + static_cast<size_t>(shared::countr_zero(w));
            return NumBits;
        }
        else {
            size_t wi = pos / word_width;
            if (const uint64_t w = words_[wi] >> (pos % word_width); w)
                return pos + static_cast<size_t>(shared::countr_zero(w));
            for (++wi; wi < kNumWords; ++wi) {
                if (words_[wi])
                    return (wi * word_width) + static_cast<size_t>(shared::countr_zero(words_[wi]));
            }
            return NumBits;
        }
    }
};

} // namespace monoprop

/// Splitmix64-based hash, specialized per hashed type.
template <typename T>
struct SplitmixHash;

/// Splitmix64 over a Bitset's words.
template <size_t NumBits>
struct SplitmixHash<monoprop::Bitset<NumBits>> {
    monoprop_HOST_DEVICE static constexpr auto mix(uint64_t x) noexcept -> uint64_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    monoprop_HOST_DEVICE auto operator()(const monoprop::Bitset<NumBits> &bs) const noexcept -> size_t {
        constexpr size_t W = monoprop::Bitset<NumBits>::num_words();
        if constexpr (W == 1) {
            return static_cast<size_t>(mix(bs.word(0)));
        }
        else {
            uint64_t h = 0;
            for (size_t i = 0; i < W; ++i) {
                h ^= mix(bs.word(i) + static_cast<uint64_t>(i));
            }
            return static_cast<size_t>(h);
        }
    }
};
```

- [ ] **Step 4: Reduce `monoprop/Bitset.h` to the host-only extras**

Replace the whole content of `cpp/monoprop/Bitset.h` after its license header with:

```cpp
#pragma once

// Bitset itself is shared with device code (monoprop/shared/Bitset.h); this header adds the host-only
// extras. The standard headers below were included by the pre-split Bitset.h, and includers still rely
// on them transitively.
#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <ostream>

#include "monoprop/shared/Bitset.h"

namespace monoprop {

/// Stream output MSB→LSB (std::bitset convention).
template <size_t NumBits>
auto operator<<(std::ostream &os, const Bitset<NumBits> &bs) -> std::ostream & {
    for (size_t i = NumBits; i-- > 0;)
        os << (bs.test(i) ? '1' : '0');
    return os;
}

} // namespace monoprop

namespace std {
template <size_t NumBits>
struct hash<monoprop::Bitset<NumBits>> {
    auto operator()(const monoprop::Bitset<NumBits> &bs) const noexcept -> size_t {
        return SplitmixHash<monoprop::Bitset<NumBits>>{}(bs);
    }
};
} // namespace std
```

Add `"Bitset.h"` to the file list in `cpp/monoprop/shared/CMakeLists.txt`:

```cmake
      FILES
        "Bits.h"
        "Bitset.h"
        "HostDevice.h"
```

- [ ] **Step 5: Run the tests and the baseline**

Run: `just build && build/editable/Release/bin/monoprop_unit_tests.x --run_test=bitset_stream_output_msb_first,shared_bitset_without_host_extras`
Expected: `*** No errors detected`.

Run: `just test-cpp && just test-py && just test-find-package`
Expected: all pass.

Run: `just diff-baseline`
Expected: no output from `diff -rq`.

- [ ] **Step 6: Commit**

```bash
git add cpp/monoprop/shared/Bitset.h cpp/monoprop/Bitset.h cpp/monoprop/shared/CMakeLists.txt \
        cpp/tests/bitset_tests.cpp cpp/tests/shared_bits_tests.cpp
git commit -m "refactor(shared): ♻️ move Bitset into the shared host/device core" \
           -m "Assisted-by: <harness>:<model>"
```

---

### Task 4: nvcc compile check of the shared headers

**Files:**
- Create: `cpp/tests/shared/nvcc_compile_check.cu`
- Create: `cpp/tests/shared/nvcc_negative_check.cu`
- Create: `tools/check-shared-nvcc.sh`
- Modify: `justfile` (two recipes after `test-find-package`)
- Create: `.github/workflows/shared-nvcc.yml`
- Modify: `docs/content/docs/testing.mdx` (new subsection after `### Golden baselines`)
- Modify: `AGENTS.md` (`## Rules` list)

**Interfaces:**
- Consumes: every function in `monoprop/shared/Bits.h` and `monoprop/shared/Bitset.h` (Tasks 2–3).
- Produces: `tools/check-shared-nvcc.sh [nvcc]`, wrapped by `just check-shared-nvcc [NVCC]` (a local
  CUDA toolkit, e.g. on Deucalion) and `just check-shared-nvcc-docker [IMAGE]` (no local CUDA; used by
  CI). Both exit 0 when the shared headers compile as device code under the shared-core flags and nvcc
  rejects the negative check.

The `.cu` files sit in `cpp/tests/shared/`, which the C++ test target's non-recursive `*.cpp` glob never
picks up, so the CPU build ignores them.

Why Docker rather than a `container:` job: `just` evaluates the justfile's backtick assignments eagerly,
including `version := uvx setuptools-scm`, which needs `uv` and the git history. A bare CUDA image has
neither, and installing them with `apt-get` in a workflow step is forbidden by
`tools/check-workflow-commands.py`. So the job runs on a standard runner, where the justfile loads as
everywhere else, and only nvcc runs inside the CUDA image.

- [ ] **Step 1: Write the compile check**

Create `cpp/tests/shared/nvcc_compile_check.cu` (license header first):

```cpp
// Compile-only check run by `just check-shared-nvcc`: every shared host/device function is called from
// device code, so nvcc must generate it for the GPU under the shared-core flags. Not part of the CPU
// build.

#include <cstddef>
#include <cstdint>

#include "monoprop/shared/Bits.h"
#include "monoprop/shared/Bitset.h"

namespace {

template <size_t NumBits>
__device__ auto exercise_bitset(uint64_t seed) -> uint64_t {
    using B = monoprop::Bitset<NumBits>;
    B a(seed);
    B b;
    b.set(NumBits - 1).set(0);
    a |= b;
    a ^= B(seed >> 7);
    a &= ~b | a;
    const B c = (a >> 3) | (a & b) | (a ^ b);
    uint64_t acc = c.count() + a.count_and(b) + (a.parity_and(b) ? 1U : 0U) + (c.any() ? 1U : 0U)
                   + (c.none() ? 1U : 0U) + (a == b ? 1U : 0U) + (c.test(1) ? 1U : 0U) + c.word(0) + c.data()[0]
                   + B::size() + B::num_words();
    for (size_t i = c.find_first(); i < NumBits; i = c.find_next(i)) {
        acc += i;
    }
    return acc + SplitmixHash<B>{}(c) + SplitmixHash<B>::mix(seed);
}

__global__ void shared_core_kernel(const uint64_t *in, uint64_t *out) {
    namespace shared = monoprop::shared;
    const uint64_t x = in[threadIdx.x];
    const int bits = shared::popcount(x) + shared::countr_zero(x) + shared::bit_width(x) + (shared::parity(x) ? 1 : 0);
    out[threadIdx.x] =
        static_cast<uint64_t>(bits) + exercise_bitset<64>(x) + exercise_bitset<100>(x) + exercise_bitset<2048>(x);
}

} // namespace
```

Create `cpp/tests/shared/nvcc_negative_check.cu` (license header first):

```cpp
// Must NOT compile under `just check-shared-nvcc`: device code calling a host-only standard-library
// function is the silent miscompile the shared-core flags exist to reject (spec, Appendix A.3).

#include <bit>
#include <cstdint>

__global__ void host_function_from_device(const uint64_t *in, int *out) {
    out[threadIdx.x] = std::popcount(in[threadIdx.x]);
}
```

- [ ] **Step 2: Add the check script and the recipes**

Create `tools/check-shared-nvcc.sh` and make it executable (`chmod +x tools/check-shared-nvcc.sh`):

```bash
#!/usr/bin/env bash
# Compile the shared host/device headers as device code under the shared-core flags (spec Section 7,
# decision D7), then require nvcc to reject device code calling a host-only function. Run from the
# repository root. Needs the CUDA toolkit (13.3 or newer), not a GPU.
#
#   tools/check-shared-nvcc.sh [nvcc]
set -euo pipefail

nvcc="${1:-nvcc}"
flags=(-std=c++23 -arch=sm_80 -Werror cross-execution-space-call -I cpp)

"$nvcc" --version | tail -n 2
"$nvcc" "${flags[@]}" -c cpp/tests/shared/nvcc_compile_check.cu -o /dev/null

if output="$("$nvcc" "${flags[@]}" -c cpp/tests/shared/nvcc_negative_check.cu -o /dev/null 2>&1)"; then
    echo "nvcc accepted device code calling a host-only function: the shared-core safety net is off" >&2
    exit 1
fi
if ! grep -q "is not allowed" <<<"$output"; then
    echo "nvcc rejected the negative check for an unexpected reason:" >&2
    echo "$output" >&2
    exit 1
fi
echo "shared headers compile as device code; nvcc rejects host-only calls from device code"
```

Append to `justfile`, after the `test-find-package` recipe:

```just
# Compile the shared host/device headers as device code with a local nvcc (CUDA 13.3 or newer) and check
# that nvcc rejects device code calling a host-only function. Needs the CUDA toolkit, not a GPU.

check-shared-nvcc NVCC='nvcc':
    tools/check-shared-nvcc.sh {{ quote(NVCC) }}

# The same check inside a CUDA container, for machines without the CUDA toolkit (and for CI).

check-shared-nvcc-docker IMAGE='nvidia/cuda:13.3.1-devel-ubuntu24.04':
    docker run --rm -v "{{ project_source_dir }}:/src" -w /src {{ quote(IMAGE) }} tools/check-shared-nvcc.sh nvcc
```

- [ ] **Step 3: Add the workflow**

Create `.github/workflows/shared-nvcc.yml`:

```yaml
name: Shared core under nvcc

on:
  pull_request:
    paths:
      - "cpp/monoprop/shared/**"
      - "cpp/tests/shared/**"
      - "justfile"
      - ".github/workflows/shared-nvcc.yml"
  push:
    branches:
      - main
    paths:
      - "cpp/monoprop/shared/**"
      - "cpp/tests/shared/**"
      - "justfile"
      - ".github/workflows/shared-nvcc.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  shared-nvcc:
    name: Compile the shared headers with nvcc
    runs-on: ubuntu-26.04

    steps:
      - uses: actions/checkout@v7.0.1
        with:
          # the justfile's version assignment runs setuptools-scm, which needs the history
          fetch-depth: 0

      - name: Install just
        uses: extractions/setup-just@v4.0.0

      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1

      # CUDA 13.3, the GPU port's toolkit (spec Section 12); nvcc accepts C++23 from 13.3 on.
      - name: Compile the shared headers as device code
        run: just check-shared-nvcc-docker
```

- [ ] **Step 4: Document it**

In `docs/content/docs/testing.mdx`, insert before `## Adding tests`:

````mdx
### Shared host/device headers

Headers under `cpp/monoprop/shared/` are compiled by nvcc as well as by the host compiler, because the
GPU port runs the same code on the device. Their functions are marked `monoprop_HOST_DEVICE` and call
only the `monoprop::shared` helpers, never standard-library functions: nvcc treats those as host-only,
and calling them from device code can miscompile silently.

```bash
just check-shared-nvcc            # with a local CUDA toolkit (13.3 or newer); no GPU needed
just check-shared-nvcc-docker     # without one: runs nvcc in an nvidia/cuda container
```

Both compile the shared headers as device code and confirm that nvcc rejects device code calling a
host-only function. CI runs the Docker variant whenever the shared headers change.
````

In `AGENTS.md`, add to the `## Rules` list:

```markdown
- Headers in `cpp/monoprop/shared/` are also compiled by nvcc: mark their functions `monoprop_HOST_DEVICE`,
  call only `monoprop::shared` helpers (never standard-library functions), and keep
  `just check-shared-nvcc` passing.
```

- [ ] **Step 5: Run the checks that work without CUDA**

Run: `prek run --files justfile tools/check-shared-nvcc.sh .github/workflows/shared-nvcc.yml cpp/tests/shared/nvcc_compile_check.cu cpp/tests/shared/nvcc_negative_check.cu docs/content/docs/testing.mdx AGENTS.md`
Expected: all hooks pass. `check workflows call recipes` passes because the workflow's only command is
`just check-shared-nvcc-docker`.

Then run the check itself, with whichever of these the machine supports:

- a local nvcc 13.3 or newer (`command -v nvcc`): `just check-shared-nvcc`
- Docker (`docker info` succeeds): `just check-shared-nvcc-docker`

Expected last line: `shared headers compile as device code; nvcc rejects host-only calls from device code`.
If neither is available, the workflow run on the PR (Task 5, Step 5) is the first execution.

- [ ] **Step 6: Commit**

```bash
git add cpp/tests/shared/nvcc_compile_check.cu cpp/tests/shared/nvcc_negative_check.cu \
        tools/check-shared-nvcc.sh justfile .github/workflows/shared-nvcc.yml \
        docs/content/docs/testing.mdx AGENTS.md
git commit -m "chore(ci): 👷 compile the shared host/device headers with nvcc" \
           -m "Assisted-by: <harness>:<model>"
```

---

### Task 5: Verify, benchmark and open the PR

**Files:**
- Modify (only if Step 5 fails on `if consteval`): `cpp/monoprop/shared/Bits.h`

**Interfaces:**
- Consumes: everything above.
- Produces: a PR into `main` titled `refactor(shared): ♻️ add the shared host/device core foundation`.

- [ ] **Step 1: Full local verification**

```bash
prek run --all-files
just build
just test-cpp
just test-py
just test-find-package
just diff-baseline
```

Expected: every command succeeds; `diff-baseline` prints nothing.

- [ ] **Step 2: Benchmark before and after on the same machine**

Start from a clean tree (`git status` shows nothing to commit), so both runs build exactly what is
committed.

```bash
git switch main && just build
monoprop_PARTITIONS=1 monoprop_NUM_THREADS=1 just bench shared-core-before \
    --num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000 --bench-rounds=3 \
    -k "test_random_propagate and heisenberg or test_random_energy and heisenberg"
git switch feat/shared-core-foundation && just build
monoprop_PARTITIONS=1 monoprop_NUM_THREADS=1 just bench shared-core-after \
    --num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000 --bench-rounds=3 \
    -k "test_random_propagate and heisenberg or test_random_energy and heisenberg"
```

Compare the medians in `benches/results/time-shared-core-before.json` and
`benches/results/time-shared-core-after.json` (`benchmarks[].stats.median` per test name).
Expected: each `after` median within 3% of its `before` median. If not, rerun both with `--bench-rounds=5`;
if the gap stays above 3%, stop and report it instead of opening the PR.

- [ ] **Step 3: Push**

```bash
git push -u origin feat/shared-core-foundation
```

- [ ] **Step 4: Open the PR**

Write the body to `/tmp/pr-shared-core.md`:

```markdown
:robot: _AI text below_ :robot:

## Summary

Groundwork for the GPU port (design: `docs/superpowers/specs/2026-10-01-gpu-port-design.md` on
`feat-gpu-port`, Section 7). Adds a `Backend` template parameter to `MonomialPropagator`, the
`monoprop_HOST_DEVICE` macro, the shared bit helpers and a shared `Bitset`, plus a CI check that compiles
the shared headers with nvcc. CPU results are byte-identical to `main` (`just diff-baseline`), and the
random Heisenberg propagate/energy benchmarks are within noise.

## Changes

- `MonomialPropagator<NumModes, Backend = backend::Cpu>`: the CPU class is the `backend::Cpu` partial
  specialization; one forward-declaration header carries the default.
- `cpp/monoprop/shared/`: `HostDevice.h`, `Bits.h` (`popcount`, `parity`, `countr_zero`, `bit_width`),
  `Bitset.h`. `monoprop/Bitset.h` keeps the host-only `operator<<` and `std::hash`.
- Host half of the bit-helper equivalence tests; constant-evaluation and layout tests for `Bitset`.
- `just check-shared-nvcc` (local nvcc), `just check-shared-nvcc-docker` and the `Shared core under nvcc`
  workflow (CUDA 13.3 image); docs and an `AGENTS.md` rule for code in `cpp/monoprop/shared/`.

## Checklist

- [x] Tests added or updated to cover the changes
- [x] Documentation updated (docstrings, `docs/`, `CONTRIBUTING.md`) if needed
- [ ] `CHANGELOG` / release notes updated if applicable

## AI/LLM disclosure

- [ ] I did not use LLM tooling, or used it only privately for ideation
- [x] I used the following tool to help write this PR description: <harness>:<model>
- [x] I used the following tool to generate or modify code: <harness>:<model>
```

```bash
gh pr create --base main --head feat/shared-core-foundation \
    --title "refactor(shared): ♻️ add the shared host/device core foundation" \
    --body-file /tmp/pr-shared-core.md
```

- [ ] **Step 5: Check the nvcc workflow run on the PR**

Run: `gh pr checks --watch`
Expected: `Compile the shared headers with nvcc` passes alongside the existing checks.

If it fails with an error pointing at an `if consteval` line in `cpp/monoprop/shared/Bits.h`, replace
each of the three `if consteval {` with `if (__builtin_is_constant_evaluated()) {` (the `else` branches
stay), then rerun Task 5 Step 1, commit with
`fix(shared): 🐛 avoid if consteval in the shared bit helpers for nvcc`, push, and check again.

If it fails because the image tag does not exist, change the default `IMAGE` of the
`check-shared-nvcc-docker` recipe in `justfile` to `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` (a tag
confirmed to exist), commit with `chore(ci): 👷 use an existing CUDA 13.3 image`, push, and check again.
