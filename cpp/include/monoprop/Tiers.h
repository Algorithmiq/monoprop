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

#include <string_view>
#include <vector>

// The ISA tier a *running* process resolved to, as opposed to Variants.h's variant(), which is the tier
// the calling translation unit was compiled for. The two differ in a narrow-seam fat binary and only
// there: every TU but the tiered one is compiled at the baseline, so variant() answers "baseline"
// throughout while the kernel doing the work is a wider tier. Anything reporting provenance to a user
// -- monoprop.__variant__, a benchmark artifact's machine flags -- wants these.

namespace monoprop {
/// Id of the ISA tier the propagation kernel dispatched to; empty when the build ships a single tier.
auto active_tier() -> std::string_view;

/// Machine-dependent flags the active tier was compiled with; empty when the build ships a single tier.
auto active_tier_flags() -> std::string_view;

/// Every tier this build ships, best ISA first; empty when the build ships a single tier.
auto shipped_tiers() -> std::vector<std::string_view>;

/// The subset of shipped_tiers() the running CPU can execute, best first.
auto supported_tiers() -> std::vector<std::string_view>;
} // namespace monoprop
