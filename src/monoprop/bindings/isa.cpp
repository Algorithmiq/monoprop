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

// The fat binary's CPU probe: the smallest possible extension module, built for the baseline ISA and
// linked against no engine object library, whose only job is to answer "which of the shipped variants
// can this machine execute". It exists because that question has to be answered *before* any tiered
// code is loaded, and answering it in Python would mean parsing /proc/cpuinfo -- Linux-only, and
// wrong about whether the OS has actually enabled the AVX-512 register state.
//
// __builtin_cpu_supports gets that right: it checks CPUID and XGETBV, so an AVX-512-capable CPU under
// a kernel or hypervisor that has not enabled ZMM state reports the feature as absent, which is the
// answer that keeps us from taking SIGILL.

#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "monoprop/FatVariants.h"

namespace nb = nanobind;

namespace {
/// One shipped variant: its id, whether this CPU can execute it, and whether it should be given it.
struct VariantRow final {
    std::string id;
    bool runnable;  ///< the CPU has the instructions: what a monoprop_VARIANT pin is allowed to ask for
    bool preferred; ///< the CPU should be handed this one unasked: what the automatic selection reads
};

// Every shipped variant, best first. Both answers come off the same generated table so they cannot
// drift apart, and they differ for exactly one variant -- see prefers_narrow_vectors() there.
auto variant_rows() -> std::vector<VariantRow> {
    auto out = std::vector<VariantRow>{};
#if defined(__x86_64__) || defined(_M_X64)
    // Required before any other __builtin_cpu_* call in a translation unit that may run before
    // libgcc's own constructor has.
    __builtin_cpu_init();
#define monoprop_ADD_VARIANT(id, runnable, preferred) \
    out.emplace_back(id, static_cast<bool>(runnable), static_cast<bool>(preferred));
    monoprop_FAT_VARIANT_TABLE(monoprop_ADD_VARIANT)
#undef monoprop_ADD_VARIANT
#endif
        return out;
}

auto filtered(bool VariantRow::*field) -> std::vector<std::string> {
    auto out = std::vector<std::string>{};
    for (const auto &row : variant_rows()) {
        if (row.*field) {
            out.push_back(row.id);
        }
    }
    return out;
}

auto supported_variants() -> std::vector<std::string> {
    return filtered(&VariantRow::preferred);
}

auto runnable_variants() -> std::vector<std::string> {
    return filtered(&VariantRow::runnable);
}

auto known_variants() -> std::vector<std::string> {
    auto out = std::vector<std::string>{};
    for (const auto &row : variant_rows()) {
        out.push_back(row.id);
    }
    return out;
}
} // namespace

NB_MODULE(_isa, mod) {
    mod.doc() = "CPU feature probe for the fat binary's import-time variant selection.";

    mod.def("supported_variants",
            &supported_variants,
            "ISA variants this CPU should be given, best first -- what the automatic selection reads.");
    mod.def("runnable_variants",
            &runnable_variants,
            "ISA variants this CPU can execute, best first -- a superset of supported_variants().");
    mod.def("known_variants", &known_variants, "ISA variants this build ships, best first, regardless of CPU support.");
}
