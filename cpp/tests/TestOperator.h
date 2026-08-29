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

#include <cstddef>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace test_utils {

// An MPOperator holding `terms` whose rows are also *findable*: append_term writes a row and nothing
// else, so find()/find_batch see nothing until the hash index is populated, which only the
// insert_absent_terms path does. That sequence is the one correct incantation for "an operator a resolve
// can look terms up in", so it lives here rather than being copied into each test that needs one.
inline auto indexed_operator(size_t num_bits,
                             const monoprop::MonomialList &terms,
                             monoprop::Basis basis = monoprop::Basis::Majorana) -> monoprop::detail::MPOperator {
    monoprop::detail::MPOperator op(num_bits);
    op.basis = basis;
    op.with_store([&](auto &rows) {
        monoprop::detail::insert_absent_terms(
            op,
            rows,
            terms.size(),
            [&](size_t k) -> const monoprop::Bitset & { return terms[k]; },
            [&](size_t k, size_t base) { assign_row(rows, base + k, terms[k]); });
    });
    return op;
}

} // namespace test_utils
