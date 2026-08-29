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
template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const std::vector<Monomial<NumModes>> &op, size_t i)
    -> const Monomial<NumModes> & {
    return op[i];
}
template <size_t NumModes>
inline auto assign_row(std::vector<Monomial<NumModes>> &op, size_t i, const Monomial<NumModes> &mono) -> void {
    op[i] = mono;
}
template <size_t NumModes>
[[nodiscard]] inline auto row_popcount(const std::vector<Monomial<NumModes>> &op, size_t i) -> size_t {
    return op[i].count();
}

// Visits row i's set-bit positions ascending, without materializing a dense bitset when the backend can
// avoid it. Hot: the even-parity inverted index is the heaviest per-row op reader.
template <size_t NumModes, typename Fn>
inline auto for_each_row_position(const std::vector<Monomial<NumModes>> &op, size_t i, Fn &&fn) -> void {
    const auto &m = op[i];
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        fn(b);
    }
}

// The packed and support-form backends answer the same four accessors, one overload set each. They are
// written out per backend rather than behind a concept because every call site names its width
// explicitly (`materialize_row<NumModes>(op, i)`), which a concept-constrained `Op` cannot deduce.
template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const detail::OperatorIndex<NumModes> &op, size_t i) -> Monomial<NumModes> {
    return op.row(i);
}
template <size_t NumModes>
inline auto assign_row(detail::OperatorIndex<NumModes> &op, size_t i, const Monomial<NumModes> &mono) -> void {
    op.set(i, mono);
}
template <size_t NumModes>
[[nodiscard]] inline auto row_popcount(const detail::OperatorIndex<NumModes> &op, size_t i) -> size_t {
    return op.popcount(i);
}
template <size_t NumModes, typename Fn>
inline auto for_each_row_position(const detail::OperatorIndex<NumModes> &op, size_t i, Fn &&fn) -> void {
    op.for_each_position(i, std::forward<Fn>(fn));
}

template <size_t NumModes>
[[nodiscard]] inline auto materialize_row(const detail::SparseRowStore<NumModes> &op, size_t i) -> Monomial<NumModes> {
    return op.row(i);
}
template <size_t NumModes>
inline auto assign_row(detail::SparseRowStore<NumModes> &op, size_t i, const Monomial<NumModes> &mono) -> void {
    op.set(i, mono);
}
// A row written from a key that is already in the store's own form -- what the insert of an absent term
// does once the query record it came from was read in that form. Only the support-form store has a form
// of its own, so these two have no counterpart on the other backends: there is no such thing as an
// OperatorIndex-shaped key that is not simply a monomial.
template <size_t NumModes>
inline auto assign_row(detail::SparseRowStore<NumModes> &op, size_t i, const detail::SparseRow &row) -> void {
    op.set(i, row);
}
template <size_t NumModes>
inline auto assign_row(detail::SparseRowStore<NumModes> &op, size_t i, const detail::SparseRowKey<2 * NumModes> &key)
    -> void {
    op.set(i, key);
}
template <size_t NumModes>
[[nodiscard]] inline auto row_popcount(const detail::SparseRowStore<NumModes> &op, size_t i) -> size_t {
    return op.popcount(i);
}
template <size_t NumModes, typename Fn>
inline auto for_each_row_position(const detail::SparseRowStore<NumModes> &op, size_t i, Fn &&fn) -> void {
    op.for_each_position(i, std::forward<Fn>(fn));
}

} // namespace monoprop
