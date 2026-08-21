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
#include <cstdio>
#include <string>

/* ── COMMPLACE: what the launcher handed this rank ─────────────────────────────
 *
 * `--cpu-bind=none` and `--cpu-bind=cores` give the same thread count and the same partition count,
 * and differ only in the mask the kernel enforces. A rank seeing 16 of a host's 128 CPUs is equally
 * "Slurm gave me my own 16" and "eight of us share these 16"; only the peers' masks separate them,
 * and the benchmark harness cannot substitute -- it samples /proc for the process it runs in, which
 * says nothing about a peer rank on the same host.
 *
 * So the affinity-mask exchange PartitionGroup already runs to pick a placement also reports.
 * Nothing here branches on any field below, and nothing below is computed when the knob is off.
 *
 * One line per rank, at propagator construction. A clone does not re-emit: the mask is a property
 * of the process, and cloning does not change it.
 */

namespace monoprop::detail::partition {

struct PlacementReport {
    int mpi_rank = 0;  //!< rank in the propagator's parent communicator
    int node_rank = 0; //!< this rank's index among the ranks sharing its host
    int node_size = 1; //!< how many ranks share its host

    /*! How the co-located ranks' affinity masks relate. Four states, and the distinction between
     *  the last two is the whole point of exchanging them:
     *    "private" — pairwise disjoint: the launcher gave every rank its own share.
     *    "shared"  — at least two ranks can be scheduled onto the same CPU.
     *    "alone"   — this rank is the only one on its host; nothing to be disjoint from.
     *    "unknown" — some rank's mask did not fit the exchanged window, so no verdict is sound.
     *  A string literal, never freed and never copied. */
    const char *masks = "unknown";

    size_t cpus = 0;           //!< CPUs in THIS rank's mask
    size_t node_cpus = 0;      //!< CPUs in the union of the masks over this host
    std::string node_cpu_list; //!< that union as ascending ranges, e.g. "0-15,64-79"
};

/*! @brief One COMMPLACE line on @p out when @p want, none otherwise; returns the lines written.
 *
 * A free function taking the flag rather than reading it, because monoprop_COMMPLACE is parsed once
 * per process and cached: a test binary not launched with it set could not otherwise reach the
 * emitting path, and the assertion would be skipped in the configuration everybody runs.
 *
 * The flag gates the printing and nothing else. The caller's mask exchange must NOT be gated on it:
 * the environment is per-rank, so a predicate over it is not rank-uniform, and one rank skipping a
 * collective its peers entered is a hang rather than a missing diagnostic.
 */
inline auto emit_place_line(std::FILE *out, bool want, const PlacementReport &r) -> int {
    if (!want) {
        return 0;
    }
    std::fprintf(out,
                 "COMMPLACE rank=%d node_rank=%d node_size=%d masks=%s cpus=%zu node_cpus=%zu cpu_list=%s\n",
                 r.mpi_rank,
                 r.node_rank,
                 r.node_size,
                 r.masks,
                 r.cpus,
                 r.node_cpus,
                 r.node_cpu_list.empty() ? "none" : r.node_cpu_list.c_str());
    std::fflush(out);
    return 1;
}

} // namespace monoprop::detail::partition
