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
#include <cstdlib>
#include <optional>

// Single home for runtime environment configuration. Kept dependency-free by design, because it is
// pulled into hot-path headers.
//
//   monoprop_NUM_THREADS    positive int (1..1e6), else ignored                → num_threads
//   monoprop_PARTITION_PINNING  bool, default ON; 0/false disables per-core pinning → partition_pinning
//   monoprop_PARTITIONS         int N | "auto" | "off"; parsed where it is used (resolve_partition_count_)
//   monoprop_LAYER_PROFILE      bool, default OFF; per-partition layer-build attribution to stderr → layer_profile
//   monoprop_DIGEST_CUTOFF      bool, default ON; decide it from an inline dense digest instead of
//                               cutoff_sums, keeping the dense path otherwise intact → digest_cutoff

namespace monoprop::config {

namespace detail {

inline auto parse_flag(const char *value, bool default_value) -> bool {
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    const char c = value[0];
    return !(c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N');
}

inline auto parse_positive_int(const char *text) -> std::optional<int> {
    if (text == nullptr) {
        return std::nullopt;
    }
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return std::nullopt;
    }
    if (value <= 0 || value > 1'000'000) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

} // namespace detail

struct Settings {
    std::optional<int> num_threads;
    bool partition_pinning = true;
    bool layer_profile = false;
    // On computes the structural cutoff's d inline from the dense words (paired_mode_count) and drops
    // the popcount<=cutoff early-out, instead of calling out to cutoff_sums. The dense path is
    // otherwise untouched -- it changes no representation and moves no data.
    //
    // Default ON. Measured: emit_s 0.9200x, 11/12 paired reps over two nodes, sign-test p=0.0063; and
    // pooled with two further shapes (128 and 512 modes) 0.9214x, 19/20, p=4.0e-5. Counters
    // (emit/reject/push/qbytes) are bit-identical between arms in all 20 pairs, memory is flat, and
    // nothing measured is worse anywhere.
    //
    // Size it honestly: emit_s is ~18% of layer_s and layer_s ~57% of build_graph wall, so this is
    // ~0.8% of build_graph -- NOT the ~3% the microbenchmark projected. The kernel really is 0.415x in
    // isolation but only 0.92x in the real scan, which has more surrounding ILP to hide it. It is also
    // below build_graph's paired noise floor here (4.2x spread), so it can never be confirmed by an
    // end-to-end A/B on this system, and at N>1 exchange grows and this shrinks further.
    //
    // What justifies enabling it is not the timing but the instruction count: retired instructions
    // around build_graph are 0.9128x on/off, 4/4 reps, -23.4 per emitted term, with branch-misses
    // unchanged. Two arms running the same instruction stream cannot differ.
    bool digest_cutoff = true;
};

// Parse the environment once; the Settings are cached and shared across TUs.
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        s.partition_pinning = detail::parse_flag(std::getenv("monoprop_PARTITION_PINNING"), true);
        s.layer_profile = detail::parse_flag(std::getenv("monoprop_LAYER_PROFILE"), false);
        s.digest_cutoff = detail::parse_flag(std::getenv("monoprop_DIGEST_CUTOFF"), true);
        return s;
    }();
    return settings;
}

} // namespace monoprop::config
