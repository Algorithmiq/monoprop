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

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <format>

#include "monoprop/Bitset.h"
// Basis-agnostic monomial vocabulary (Monomial, Basis, CutoffFn, ...) lives in core/Monomial.h; the
// storage-backend plumbing (TermIndex, OperatorIndex/InvertedIndex/MPOperator, row accessors) stays here.
#include "monoprop/core/Monomial.h"

// Forward-declare (not include) OperatorIndex: it includes this header, so a full include would cycle.
namespace monoprop::detail {
template <size_t NumModes>
class OperatorIndex;
}

namespace monoprop {

// Backend-agnostic row access: dense/packed backends present one surface. materialize_row() returns a
// const ref (dense, zero-copy) or a fresh value (packed — bind with `const auto&` to extend lifetime).
template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const std::vector<Monomial<NumModes>> &op, size_t i)
    -> const Monomial<NumModes> & {
    return op[i];
}
template <size_t NumModes>
inline auto assign_row(std::vector<Monomial<NumModes>> &op, size_t i, const Monomial<NumModes> &maj) -> void {
    op[i] = maj;
}
template <size_t NumModes>
[[nodiscard]] inline auto row_popcount(const std::vector<Monomial<NumModes>> &op, size_t i) -> size_t {
    return op[i].count();
}

// Iterate row i's set-bit positions (ascending) without materializing a dense bitset when the backend
// can avoid it. Used by the even-parity inverted index, the heaviest per-row op reader.
template <size_t NumModes, typename Fn>
inline auto for_each_row_position(const std::vector<Monomial<NumModes>> &op, size_t i, Fn &&fn) -> void {
    const auto &m = op[i];
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        fn(b);
    }
}
// OperatorIndex overloads for these accessors are defined after the OperatorIndex.h include below.

using VecCD = std::vector<std::complex<double>>;

using VecD = std::vector<double>;

using VecI = std::vector<int>;

using VecZ = std::vector<size_t>;

// Compile-time build knobs (cmake -D<name>=...):
//   monoprop_ENABLE_MPI         real MPI transport vs single-rank stubs         (default OFF)
//   monoprop_WIDE_TERM_INDEX    TermIndex = u64 vs u32 → >2^32 local terms/rank (default OFF)
//   monoprop_MAX_NUM_MODES      NumModes codegen/instantiation ceiling          (default 250)
//   monoprop_ENABLE_ARCH_FLAGS  -march=native / -xHost (non-Debug)              (default ON)
// Runtime (env-var) knobs live in detail/EnvConfig.h.
//
// TermIndex: operator row index. Default u32; monoprop_WIDE_TERM_INDEX widens to u64 for > 2^32 local terms/rank.
#if defined(monoprop_WIDE_TERM_INDEX)
using TermIndex = std::uint64_t;
#else
using TermIndex = std::uint32_t;
#endif

// Allocator that DEFAULT-initializes (no zero-fill) on the no-arg construct. Use ONLY for buffers whose
// every grown element is overwritten before read — otherwise it exposes indeterminate values.
template <typename T, typename A = std::allocator<T>>
struct default_init_allocator : A {
    using a_traits = std::allocator_traits<A>;
    template <typename U>
    struct rebind {
        using other = default_init_allocator<U, typename a_traits::template rebind_alloc<U>>;
    };
    using A::A;
    default_init_allocator() = default;
    template <typename U>
    default_init_allocator(const default_init_allocator<U, typename a_traits::template rebind_alloc<U>> &o) noexcept
        : A(static_cast<const typename a_traits::template rebind_alloc<U> &>(o)) {}

    template <typename U>
    void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void *>(ptr)) U; // default-init: no zero-fill for trivial U
    }
    template <typename U, typename... Args>
    void construct(U *ptr, Args &&...args) {
        a_traits::construct(static_cast<A &>(*this), ptr, std::forward<Args>(args)...);
    }
};

// Vector that skips resize() zero-init for trivial elements. See default_init_allocator caveat.
template <typename T>
using DefaultInitVector = std::vector<T, default_init_allocator<T>>;

using FermiOperatorMap = std::map<VecZ, std::complex<double>>;

using CyclesType = std::vector<std::vector<std::pair<size_t, size_t>>>;

} // namespace monoprop

// Included here (not at the top) because OperatorIndex needs TermIndex/MonomialHash defined above.
#include "monoprop/detail/operator/OperatorIndex.h"

// OperatorIndex row-accessor overloads. MUST be declared BEFORE InvertedIndex.h/MPOperator.h: those
// templates reach these via ordinary lookup, not ADL (ADL searches only monoprop::detail).
namespace monoprop {
template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const detail::OperatorIndex<NumModes> &op, size_t i) -> Monomial<NumModes> {
    return op.row(i);
}
template <size_t NumModes>
inline auto assign_row(detail::OperatorIndex<NumModes> &op, size_t i, const Monomial<NumModes> &maj) -> void {
    op.set(i, maj);
}
template <size_t NumModes>
[[nodiscard]] inline auto row_popcount(const detail::OperatorIndex<NumModes> &op, size_t i) -> size_t {
    return op.popcount(i);
}
template <size_t NumModes, typename Fn>
inline auto for_each_row_position(const detail::OperatorIndex<NumModes> &op, size_t i, Fn &&fn) -> void {
    op.for_each_position(i, std::forward<Fn>(fn));
}
} // namespace monoprop

#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"
