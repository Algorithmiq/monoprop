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

// One row-reader/writer vocabulary over all three operator backends: the dense MonomialList, the packed
// detail::OperatorIndex and the fixed-width-lane detail::SparseRowStore. Templates parameterized on the
// row store (`Rows`) call these unqualified, so every such template must include this header — ADL
// cannot reach monoprop:: from an argument in monoprop::detail, and a later declaration is not found for
// an already-parsed template definition.

#include <cstddef>
#include <utility>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

namespace monoprop {

// materialize_row() returns a const ref (dense backend, zero-copy) or a fresh value (packed/sparse
// backend), so callers must bind with `const auto&` to extend the temporary's lifetime.
[[nodiscard]] inline auto materialize_row(const std::vector<MonomialLike auto> &op, size_t i) -> decltype(auto) {
    return op[i];
}
inline auto assign_row(std::vector<MonomialLike auto> &op, size_t i, const auto &mono) -> void {
    op[i] = mono;
}
[[nodiscard]] inline auto row_popcount(const std::vector<MonomialLike auto> &op, size_t i) -> size_t {
    return op[i].count();
}

// Visits row i's set-bit positions ascending, without materializing a dense bitset when the backend can
// avoid it. Hot: the even-parity inverted index is the heaviest per-row op reader.
template <typename Fn>
inline auto for_each_row_position(const std::vector<MonomialLike auto> &op, size_t i, Fn &&fn) -> void {
    const auto &m = op[i];
    const size_t n = m.size();
    for (size_t b = m.find_first(); b < n; b = m.find_next(b)) {
        fn(b);
    }
}

// Structural stand-in for "a row store shaped like detail::OperatorIndex/detail::SparseRowStore": exposes
// value_type and a row(i) accessor returning it, so the overloads below take a store without naming its
// row type -- the same idea as the MonomialLike overloads above, one level up (op itself is not
// MonomialLike; its rows are). A std::vector has no row(), so this never collides with the overloads above.
template <typename T>
concept RowStoreLike = requires(const T &t, size_t i) {
    typename T::value_type;
    { t.row(i) } -> std::same_as<typename T::value_type>;
};

template <RowStoreLike Op>
[[nodiscard]] inline auto materialize_row(const Op &op, size_t i) -> typename Op::value_type {
    return op.row(i);
}
template <RowStoreLike Op>
inline auto assign_row(Op &op, size_t i, const typename Op::value_type &mono) -> void {
    op.set(i, mono);
}
// A row written from a key that is already in the store's own form -- what the insert of an absent term
// does once the query record it came from was read in that form. Only the support-form store has a form
// of its own, so this is the one accessor with a backend-specific overload rather than a generic one:
// there is no such thing as an OperatorIndex-shaped key that is not simply a monomial.
inline auto assign_row(detail::SparseRowStore &op, size_t i, const detail::SparseRow &row) -> void {
    op.set(i, row);
}
inline auto assign_row(detail::SparseRowStore &op, size_t i, const detail::SparseRowKey &key) -> void {
    op.set(i, key);
}

template <RowStoreLike Op>
[[nodiscard]] inline auto row_popcount(const Op &op, size_t i) -> size_t {
    return op.popcount(i);
}
template <RowStoreLike Op, typename Fn>
inline auto for_each_row_position(const Op &op, size_t i, Fn &&fn) -> void {
    op.for_each_position(i, std::forward<Fn>(fn));
}

} // namespace monoprop
