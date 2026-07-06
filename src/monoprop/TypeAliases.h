#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <cassert>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <optional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <format>
#include <boost/unordered/unordered_flat_map.hpp>

#include "monoprop/Bitset.h"
namespace monoprop {
/*!
 * @brief Bitset container for a multi-qubit operator in Majorana basis.
 * @tparam NumModes Number of Fermionic modes.
 */
template <size_t NumModes>
using MajoranaSet = Bitset<2 * NumModes>;
} // namespace monoprop

// Forward-declare (not include) OperatorIndex: it includes THIS header for MPHash/MajoranaSet,
// so a full include here would be a cycle. The packed row-accessor overloads below take it by
// reference (incomplete type is fine for the declaration); the complete type is in scope wherever
// they are instantiated, since every such TU includes operator/OperatorIndex.h.
namespace monoprop::detail {
template <size_t NumModes>
class OperatorIndex;
}

namespace monoprop {

/*!
 * @brief Generic Majorana term-list type: a dense std::vector<MajoranaSet>.
 * @tparam NumModes Number of Fermionic modes.
 *
 * Used for plain term lists (gradient ham/state pairs, basis-change vectors, commutator
 * pipeline operands). The evolved operator's row storage (`MPOperator::op`) is NOT this
 * alias — it is the entropy-packed detail::OperatorIndex (position-list rows plus
 * hash index, always-on for every NumModes; see that header). Functions that must accept
 * both go through the backend-agnostic row accessors below and template their rows parameter.
 */
template <size_t NumModes>
using MajoranaVector = std::vector<MajoranaSet<NumModes>>;

// --- Backend-agnostic row access -------------------------------------------------------------
// All operator-row consumers go through these so the dense and packed backends present one
// surface. For the dense backend materialize_row() returns a const reference (zero-copy); for the
// packed backend it returns a freshly reconstructed value (bind with `const auto&` to extend
// its lifetime). assign_row() overwrites an already-sized slot (parallel miss-fill paths).
template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const std::vector<MajoranaSet<NumModes>> &op, size_t i)
    -> const MajoranaSet<NumModes> & {
    return op[i];
}
template <size_t NumModes>
inline auto assign_row(std::vector<MajoranaSet<NumModes>> &op, size_t i, const MajoranaSet<NumModes> &maj) -> void {
    op[i] = maj;
}
template <size_t NumModes>
[[nodiscard]] inline auto row_popcount(const std::vector<MajoranaSet<NumModes>> &op, size_t i) -> size_t {
    return op[i].count();
}

// Iterate the set-bit positions of row i (ascending) without materializing a dense bitset where
// the backend can avoid it. The dense backend scans words; the packed backend reads its stored
// position list directly. Used by the even-parity inverted index, the heaviest per-row op reader.
template <size_t NumModes, typename Fn>
inline auto for_each_row_position(const std::vector<MajoranaSet<NumModes>> &op, size_t i, Fn &&fn) -> void {
    const auto &m = op[i];
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        fn(b);
    }
}
// OperatorIndex overloads for materialize_row / assign_row / row_popcount /
// for_each_row_position are defined after the OperatorIndex.h include at the bottom of
// this file (OperatorIndex needs TermIndex + MPHash which are defined later in this file).

/*!
 * @brief Transparent hash for MajoranaSet (is_transparent enables heterogeneous map lookup).
 */
template <size_t NumModes>
struct MPHash final {
    using is_transparent = void;

    auto operator()(const MajoranaSet<NumModes> &arr) const noexcept -> size_t {
        return SplitmixHash<MajoranaSet<NumModes>>{}(arr);
    }
};

template <size_t NumModes>
struct MPEqual final {
    using is_transparent = void;

    auto operator()(const MajoranaSet<NumModes> &lhs, const MajoranaSet<NumModes> &rhs) const noexcept -> bool {
        return lhs == rhs;
    }
};

template <size_t NumModes>
using MajoranaOperator =
    boost::unordered_flat_map<MajoranaSet<NumModes>, double, MPHash<NumModes>, MPEqual<NumModes>>;

template <size_t NumModes>
inline auto majorana_hash(const MajoranaSet<NumModes> &maj) noexcept -> size_t {
    if constexpr (MajoranaSet<NumModes>::num_words() == 1) {
        return static_cast<size_t>(SplitmixHash<MajoranaSet<NumModes>>::mix(maj.word(0)));
    }
    else {
        return MPHash<NumModes>{}(maj);
    }
}

using VecCD = std::vector<std::complex<double>>;

using VecD = std::vector<double>;

using VecI = std::vector<int>;

using VecZ = std::vector<size_t>;

// ── Compile-time build knobs (set at configure time: cmake -D<name>=...) ──────────────────────
//   monoprop_ENABLE_MPI         real MPI transport vs single-rank stubs         (default OFF)
//   monoprop_WIDE_TERM_INDEX    TermIndex = u64 vs u32 → >2^32 local terms/rank (default OFF)
//   monoprop_MAX_NUM_MODES      NumModes codegen/instantiation ceiling          (default 250)
//   monoprop_ENABLE_ARCH_FLAGS  -march=native / -xHost (non-Debug)              (default ON)
// Runtime (env-var) knobs live in detail/EnvConfig.h. Storage-width regimes (Bitset word count,
// OperatorIndex::PosT) are template/constexpr decisions derived from NumModes, not build switches.
//
// TermIndex: operator row index. Default uint32_t (memory-minimal); monoprop_WIDE_TERM_INDEX widens
// it to uint64_t to support > 2^32 local terms on a single rank, at the cost of doubling the per-term
// index arrays.
#if defined(monoprop_WIDE_TERM_INDEX)
using TermIndex = std::uint64_t;
#else
using TermIndex = std::uint32_t;
#endif

// Allocator that DEFAULT-initializes (placement-new `U`) on the no-arg construct instead of
// value-initializing (`U()`). For trivially-default-constructible T this leaves resize()-grown
// elements UNINITIALIZED (no serial zero-fill). Use ONLY for buffers whose every grown element is
// overwritten before it is read (e.g. parallel-scatter gather destinations) — otherwise it exposes
// indeterminate values. Lets a parallel fill avoid the serial memset that resize() would otherwise do.
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

template <size_t NumModes>
using CutoffFn = std::function<bool(const MajoranaSet<NumModes> &)>;

/**
 * @brief Structural truncation criterion applied to Majorana monomials after each gate.
 *
 * Both criteria share one rule: a *fully paired* monomial -- one whose support
 * consists entirely of complete pairs m_{2j-1} m_{2j} on a mode -- is always kept,
 * regardless of the cutoff. Fully paired monomials are exactly the terms that can
 * contribute to an expectation value against a computational-basis state or Slater
 * determinant, so discarding them would throw away signal. The criteria differ only
 * in how they measure the remaining, partially paired monomials.
 */
enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

} // namespace monoprop

// Data classes extracted into focused detail headers. Included here so existing consumers
// of TypeAliases.h continue to see all types without modification.
// OperatorIndex needs TermIndex and MPHash (defined above), so it is included here rather
// than at the top where only MajoranaSet is yet in scope.
#include "monoprop/detail/operator/OperatorIndex.h"

// OperatorIndex backend-agnostic row-accessor overloads (requires OperatorIndex defined).
// MUST be declared BEFORE InvertedIndex.h / MPOperator.h: those headers' templates call these accessors,
// and since OperatorIndex lives in monoprop::detail, ADL from those templates searches only
// monoprop::detail — it would NOT reach these monoprop-namespace overloads. Declaring them here
// puts them in ordinary-lookup scope at the point those headers are parsed.
namespace monoprop {
template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const detail::OperatorIndex<NumModes> &op, size_t i)
    -> MajoranaSet<NumModes> {
    return op.row(i);
}
template <size_t NumModes>
inline auto assign_row(detail::OperatorIndex<NumModes> &op, size_t i, const MajoranaSet<NumModes> &maj) -> void {
    // All callers of this overload (Engine.h insert_deferred_self_misses, Resolve.h miss scatter) write
    // freshly grown, disjoint, never-before-written rows, so use the fresh path that skips the overflow
    // pre-read/erase (unnecessary + UB-adjacent on default-init rows inside the parallel scatter).
    op.set_fresh(i, maj);
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
