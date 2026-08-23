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

#include <functional>
#include <memory>
#include <optional>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/MPOperator.h"

// The fat binary's seam: the one call the ISA tiers are chosen behind.
//
// build_layer's instantiation tree is where the vectorization lands -- of the four AVX-512 signal
// counters in a v4+vpopcntdq build, MonomialPropagator.cpp's object holds 1274 of 1592 zmm operands and
// 102 of 111 vpopcnt instructions -- so tiering that one tree buys almost everything tiering the whole
// library buys, at one copy of it instead of one copy of the library. Everything else in the wheel is
// compiled once, at the baseline. Measured: it buys *all* of it, tier for tier, the nine baseline
// translation units costing nothing detectable.
//
// One condition, and it is not negotiable: the tiers' copies of this TU must land in separate links, or
// their template instantiations -- weak COMDATs whose mangled names carry no tier -- deduplicate down
// to one and the dispatch below chooses between four names for the same code. That is what
// monoprop_FAT_BINARY_MODE=tier-dso arranges and what narrow-seam fails to; see FatBinary.cmake.
//
// Why a struct and not the argument list: build_layer takes seventeen parameters, five of them
// defaulted. A tier entry point has to spell all seventeen, once per tier plus once in the dispatcher's
// declaration, and the compiler cannot catch two of them being swapped. Bundling them names each one at
// the only place it is written.

namespace monoprop::detail {

/// One layer build's arguments. Field order and meaning follow build_layer()'s parameters.
struct LayerBuildRequest final {
    MPOperator &local_op;       ///< evolved in place; also the source of the storage width
    const Bitset &gen;          ///< the gate generator, and the width every monomial built here takes
    const CutoffFn &cutoff_fn;  ///< structural cutoff; a std::function, so it carries no width
    std::optional<double> atol; ///< lower coefficient cutoff
    std::optional<std::reference_wrapper<const VecD>> local_coeffs; ///< present when a cutoff reads coefficients
    std::optional<double> upper_atol;                               ///< upper coefficient cutoff
    std::optional<double> param;                                    ///< rotation angle, absent when building a graph
    std::optional<size_t> only_rotate_len_k;                        ///< length cap on rotated terms
    MatchedEpochSet &matched_scratch;                               ///< follower marks, reused across gates
    mpi::Comm comm;
    size_t logical_num_modes;
    CosMask *out_cos = nullptr;              ///< non-null to receive the layer's cosine mask
    FusedContract *fused_contract = nullptr; ///< non-null selects the fused contract-immediately sink
    bool schrodinger = false;
    VecD *fused_scale_coeffs = nullptr; ///< must alias local_coeffs when non-null
    bool *fused_scale_out = nullptr;    ///< receives build_layer's fused-scale decision; the apply must follow it
    Basis basis = Basis::Majorana;
};

/// Build one layer, on the widest ISA tier the running CPU supports.
///
/// One indirect call per gate against a scan that is O(operator) per gate, so the seam itself does not
/// show up in a measurement. A build that ships a single tier resolves this to a direct call.
auto build_layer_dispatch(const LayerBuildRequest &request) -> std::shared_ptr<LayerCore>;

} // namespace monoprop::detail
