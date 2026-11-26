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

// Test-only helpers shared by the operator-store and kernel suites.

#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace kernel_test {

// An operator whose rows are `terms`, in order.
template <size_t N>
auto make_op(const std::vector<monoprop::Monomial<N>> &terms, monoprop::Basis basis = monoprop::Basis::Majorana)
    -> monoprop::detail::MPOperator<N> {
    monoprop::detail::MPOperator<N> op;
    op.basis = basis;
    if (terms.empty()) {
        return op;
    }
    monoprop::detail::insert_absent_terms<N>(
        op,
        terms.size(),
        [&](size_t k) -> const monoprop::Monomial<N> & { return terms[k]; },
        [&](size_t k, size_t base) { monoprop::assign_row<N>(*op.store, base + k, terms[k]); });
    return op;
}

// A column's set rows, whichever tier holds them, so two indices that picked different tiers compare equal.
template <size_t N>
auto column_rows(const monoprop::detail::InvertedIndex<N> &index, size_t c) -> std::vector<size_t> {
    if (!index.column_is_dense(c)) {
        const auto sparse = index.sparse_column_rows(c);
        return std::vector<size_t>(sparse.begin(), sparse.end());
    }
    std::vector<size_t> rows;
    const uint64_t *w = index.dense_column_data(c);
    for (size_t r = 0; r < index.rows(); ++r) {
        if (((w[r >> 6] >> (r & 63U)) & 1U) != 0U) {
            rows.push_back(r);
        }
    }
    return rows;
}

// Distinct, non-degenerate values, so a missed, extra or doubled scaling shows up bitwise.
inline auto distinct_values(size_t n) -> monoprop::VecD {
    monoprop::VecD v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0 + static_cast<double>(i) * 1e-3;
    }
    return v;
}

inline auto bitwise_equal(std::span<const double> a, std::span<const double> b) -> bool {
    // memcmp must not see an empty vector's null data pointer, even for zero bytes.
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

} // namespace kernel_test
