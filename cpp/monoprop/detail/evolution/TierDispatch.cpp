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

// Which copy of the tiered translation unit runs. Compiled once, always at the baseline ISA: this file
// executes on every machine the wheel installs on, including the ones that cannot run any tier above
// the floor, so it may not itself contain a widened instruction.
//
// Contrast with the whole-library fat binary, where the same decision is made in Python before any
// engine code is loaded at all (src/monoprop/_bootstrap.py). Here the decision is inside the library,
// which is what lets a single _core.so carry every tier -- and why there is no separate ISA probe
// module whose tier list could fall out of step.

#include "monoprop/Tiers.h"

#include <cstdlib>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "monoprop/detail/evolution/TieredLayerBuild.h"

#ifdef monoprop_FAT_NARROW_SEAM
#include "monoprop/FatVariants.h"
#endif

#ifndef monoprop_FAT_NARROW_SEAM

// ---------------------------------------------------------------------------------------------------
// Single-tier build: the seam is a direct call and there is no tier to report. The accessors stay
// defined so that nothing downstream has to know which kind of build it is in.
// ---------------------------------------------------------------------------------------------------

namespace monoprop {
namespace detail {
namespace tiers::only {
auto build_layer_entry(const LayerBuildRequest &request) -> std::shared_ptr<LayerCore>;
} // namespace tiers::only

auto build_layer_dispatch(const LayerBuildRequest &request) -> std::shared_ptr<LayerCore> {
    return tiers::only::build_layer_entry(request);
}
} // namespace detail

auto active_tier() -> std::string_view {
    return {};
}

auto active_tier_flags() -> std::string_view {
    return {};
}

auto shipped_tiers() -> std::vector<std::string_view> {
    return {};
}

auto supported_tiers() -> std::vector<std::string_view> {
    return {};
}
} // namespace monoprop

#else

namespace monoprop {
namespace detail {

// One declaration block per shipped tier, from the same generated table the predicates come from, so a
// tier cannot be built without being dispatchable or dispatched without being built.
#define monoprop_DECLARE_TIER(tid, tslug, tpred)                                                             \
    namespace tiers::tslug {                                                                                 \
    auto build_layer_entry(const LayerBuildRequest &) -> std::shared_ptr<LayerCore>;                         \
    auto tier_id() -> std::string_view;                                                                      \
    auto tier_machine_flags() -> std::string_view;                                                            \
    }
monoprop_FAT_VARIANT_TIERS(monoprop_DECLARE_TIER)
#undef monoprop_DECLARE_TIER

} // namespace detail

namespace {

struct TierRow final {
    std::string_view id;
    bool (*supported)();
    std::shared_ptr<LayerCore> (*build)(const detail::LayerBuildRequest &);
    std::string_view (*compiled_id)();
    std::string_view (*flags)();
};

// Best ISA first, which is the selection order: the table's order is load-bearing, not cosmetic.
#define monoprop_TIER_ROW(tid, tslug, tpred)                                                                 \
    TierRow{.id = tid,                                                                                       \
            .supported = +[]() -> bool { return static_cast<bool>(tpred); },                                 \
            .build = &detail::tiers::tslug::build_layer_entry,                                                \
            .compiled_id = &detail::tiers::tslug::tier_id,                                                    \
            .flags = &detail::tiers::tslug::tier_machine_flags},
const TierRow kTiers[] = {monoprop_FAT_VARIANT_TIERS(monoprop_TIER_ROW)};
#undef monoprop_TIER_ROW

// Shared with the whole-library mode's Python loader, so a pin written against one build shape keeps
// working against the other.
constexpr const char *kPinEnvVar = "monoprop_VARIANT";

// __builtin_cpu_supports reads feature bits a companion builtin has to load first, so: once, before any
// predicate runs.
auto probe_cpu() -> void {
    __builtin_cpu_init();
}

auto join_tiers(bool supported_only) -> std::string {
    std::string out;
    for (const auto &row : kTiers) {
        if (supported_only && !row.supported()) {
            continue;
        }
        if (!out.empty()) {
            out += ", ";
        }
        out += row.id;
    }
    return out;
}

auto select_tier() -> const TierRow & {
    probe_cpu();

    const char *pinned = std::getenv(kPinEnvVar);
    if (pinned != nullptr && pinned[0] != '\0') {
        for (const auto &row : kTiers) {
            if (row.id != pinned) {
                continue;
            }
            // A pin the CPU cannot run is refused rather than honoured. Honouring it means SIGILL
            // somewhere inside the scan, with nothing left to point at the pin that caused it.
            if (!row.supported()) {
                throw std::invalid_argument(std::string(kPinEnvVar) + "='" + pinned
                                            + "' names an ISA tier this CPU cannot execute. Supported here: "
                                            + join_tiers(/*supported_only=*/true));
            }
            return row;
        }
        throw std::invalid_argument(std::string(kPinEnvVar) + "='" + pinned
                                    + "' is not an ISA tier this build ships. Shipped tiers: "
                                    + join_tiers(/*supported_only=*/false));
    }

    for (const auto &row : kTiers) {
        if (row.supported()) {
            return row;
        }
    }
    // Unreachable by construction: the baseline tier's predicate is the literal true. Kept because that
    // is a property of a generated table, and the table is generated from a list somebody can edit.
    throw std::runtime_error("No shipped ISA tier is executable on this CPU. Shipped tiers: "
                             + join_tiers(/*supported_only=*/false));
}

auto active_row() -> const TierRow & {
    static const TierRow &row = []() -> const TierRow & {
        const TierRow &selected = select_tier();
        // The tier's own Variants.h against the table that selected it. These are two independent paths
        // out of one tier list -- an include directory and a namespace slug -- and a build that wired
        // them to different tiers would otherwise run one tier while reporting another.
        if (selected.compiled_id() != selected.id) {
            throw std::runtime_error(std::string("ISA tier '") + std::string(selected.id) + "' was compiled as '"
                                     + std::string(selected.compiled_id())
                                     + "': its per-tier include path does not match its namespace slug.");
        }
        return selected;
    }();
    return row;
}

} // namespace

namespace detail {
auto build_layer_dispatch(const LayerBuildRequest &request) -> std::shared_ptr<LayerCore> {
    return active_row().build(request);
}
} // namespace detail

auto active_tier() -> std::string_view {
    return active_row().id;
}

auto active_tier_flags() -> std::string_view {
    return active_row().flags();
}

auto shipped_tiers() -> std::vector<std::string_view> {
    std::vector<std::string_view> out;
    out.reserve(std::size(kTiers));
    for (const auto &row : kTiers) {
        out.emplace_back(row.id);
    }
    return out;
}

auto supported_tiers() -> std::vector<std::string_view> {
    probe_cpu();
    std::vector<std::string_view> out;
    for (const auto &row : kTiers) {
        if (row.supported()) {
            out.emplace_back(row.id);
        }
    }
    return out;
}

} // namespace monoprop

#endif
