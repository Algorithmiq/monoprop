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

#include <optional>
#include <stdexcept>

#include "monoprop/Evolution.h"
#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

inline auto remove_incoming_cycle_targets_compressed(const VecZ &cos_inds, const SplitCycleResult &split)
    -> CompressedCosineData {
    VecZ incoming;
    size_t total_incoming = 0;
    for (const auto &cross_rank : split.cross_rank) {
        total_incoming += cross_rank.in_indices.size();
    }
    incoming.reserve(total_incoming);

    for (const auto &cross_rank : split.cross_rank) {
        incoming.insert(incoming.end(), cross_rank.in_indices.begin(), cross_rank.in_indices.end());
    }

    return build_filtered_compressed_cosine_data(cos_inds, incoming);
}

template <size_t NumModes>
auto cutoff_function(CutoffType cutoff_type, unsigned int cutoff, size_t logical_num_modes = NumModes)
    -> CutoffFn<NumModes> {
    switch (cutoff_type) {
        case CutoffType::Length:
            return detail::LengthCutoff<NumModes>{cutoff, logical_num_modes};
        case CutoffType::Support:
            return detail::SupportCutoff<NumModes>{cutoff, logical_num_modes};
        default:
            throw std::runtime_error("Unknown cutoff type");
    }
}

template <size_t NumModes>
auto cutoff_function_basis_change(CutoffType cutoff_type,
                                  unsigned int cutoff,
                                  const MajoranaVector<NumModes> &basis,
                                  size_t logical_num_modes = NumModes) -> CutoffFn<NumModes> {
    switch (cutoff_type) {
        case CutoffType::Length:
            return [cutoff, logical_num_modes, basis_copy = basis](const MajoranaSet<NumModes> &maj) {
                const auto mapped_maj = change_basis<NumModes>(maj, basis_copy);
                return length_cutoff<NumModes>(mapped_maj, cutoff, logical_num_modes);
            };
        case CutoffType::Support:
            return [cutoff, logical_num_modes, basis_copy = basis](const MajoranaSet<NumModes> &maj) {
                const auto mapped_maj = change_basis<NumModes>(maj, basis_copy);
                return support_cutoff<NumModes>(mapped_maj, cutoff, logical_num_modes);
            };
        default:
            throw std::runtime_error("Unknown cutoff type");
    }
}

} // namespace monoprop::detail
