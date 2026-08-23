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

// The tiered translation unit: this file, and only this file, is compiled once per ISA tier. Its
// enclosing namespace is a per-tier identifier so the copies coexist in one shared object, and
// TierDispatch.cpp picks one of them at run time.
//
// Everything the tier is for is reached from here: build_layer binds the algebra, the row backend and
// the kernel width, and the whole scan/fold tree instantiates into this TU at this TU's -march. Nothing
// is inlined across the seam and nothing needs to be -- the call is once per gate.

#include "monoprop/detail/evolution/TieredLayerBuild.h"

#include <string_view>

#include "monoprop/Variants.h"
#include "monoprop/detail/evolution/LayerBuilder.h"

// Undefined in a single-tier build, where there is exactly one copy and the name is arbitrary.
#ifndef monoprop_TIER_SLUG
#define monoprop_TIER_SLUG only
#endif

namespace monoprop::detail::tiers::monoprop_TIER_SLUG {

auto build_layer_entry(const LayerBuildRequest &request) -> std::shared_ptr<LayerCore> {
    return build_layer(request.local_op,
                       request.gen,
                       request.cutoff_fn,
                       request.atol,
                       request.local_coeffs,
                       request.upper_atol,
                       request.param,
                       request.only_rotate_len_k,
                       request.matched_scratch,
                       request.comm,
                       request.logical_num_modes,
                       request.out_cos,
                       request.fused_contract,
                       request.schrodinger,
                       request.fused_scale_coeffs,
                       request.fused_scale_out,
                       request.basis);
}

// Both read this TU's own Variants.h, which the tier's include path puts ahead of the build's default
// one. That is what makes them a check and not a restatement of the table: the dispatcher asserts that
// the tier it selected reports the id the table selected it under, so a tier target whose include path
// drifted from its namespace slug fails loudly instead of silently reporting the wrong provenance.
auto tier_id() -> std::string_view {
    return variant();
}

auto tier_machine_flags() -> std::string_view {
    return variant_flags();
}

} // namespace monoprop::detail::tiers::monoprop_TIER_SLUG
