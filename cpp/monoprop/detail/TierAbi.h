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

// The two-symbol-direction contract between the shared engine and a per-ISA-tier shared object.
//
// Only a tiered build puts a boundary here at all: the tiered translation unit is
// linked into libmonoprop-tier-<slug>.so, everything else into the module that loads it. The engine is
// compiled -fvisibility=hidden throughout (CXX_VISIBILITY_PRESET in cpp/monoprop/CMakeLists.txt), so a
// symbol that has to cross needs saying so, and the point of naming the two directions separately is
// that the *set* on each side is small enough to check -- tools/check-tier-symbols.py asserts it.
//
// The macros expand to the visibility attribute in every build shape, not only in tier-dso mode. In the
// others the boundary is not a link boundary and the attribute costs a symbol that stays in .dynsym;
// making it conditional would mean the shapes no longer compile the same source, which is how one of
// them stops being tested.

#pragma once

/// A symbol the *tier* provides and the shared engine calls: the entry points in
/// cpp/monoprop/detail/evolution/TieredLayerBuild.cpp, one set per tier namespace.
#define monoprop_TIER_ENTRY [[gnu::visibility("default")]]

/// A symbol the shared *engine* provides and a tier calls. Deliberately a closed set -- the tiered TU's
/// only out-of-line dependencies on the rest of the engine, everything else it needs being either a
/// header template it instantiates itself or libstdc++. Adding one is adding to a published ABI: it must
/// be stateless, because in tier-dso mode the tier binds to the loading module's definition while the
/// rest of the engine may hold its own, and two copies of a *cache* would diverge.
#define monoprop_TIER_ABI [[gnu::visibility("default")]]
