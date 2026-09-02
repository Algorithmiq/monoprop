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

// A transient monomial -> row map over a range of an OperatorIndex's rows. The operator store keeps no
// persistent key index (propagation resolves partners inside each gate's anticommuting set, see
// layer_build/AntiTable.h), so the few call sites that look a term up by value outside a gate -- draining
// pending initial-operator coefficients, re-weighting the initial operator -- build one of these over the
// rows they need and drop it.

#include <cassert>
#include <cstddef>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

template <size_t NumModes>
using TermLookup =
    boost::unordered_flat_map<Monomial<NumModes>, TermIndex, MonomialHash<NumModes>, MonomialEqual<NumModes>>;

// The map over rows [first, last) of `store`. Rows are distinct by the store's invariant, so a repeated key
// is a caller error; the later row wins, which an assert reports in debug builds.
template <size_t NumModes>
[[nodiscard]] auto build_term_lookup(const OperatorIndex<NumModes> &store, size_t first, size_t last)
    -> TermLookup<NumModes> {
    TermLookup<NumModes> map;
    if (last > first) {
        map.reserve(last - first);
    }
    for (size_t i = first; i < last; ++i) {
        [[maybe_unused]] const auto [it, inserted] = map.emplace(store.row(i), static_cast<TermIndex>(i));
        assert(inserted && "OperatorIndex holds a duplicate row");
    }
    return map;
}

} // namespace monoprop::detail
