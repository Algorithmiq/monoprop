# Stage 0 on Deucalion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or
> executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Stage 0 of the GPU port: confirm the CPU path builds and passes its tests with the
GCC 15.2 toolchain, measure how much the ladder circuits would exchange between GPUs, and run the
throwaway replay spike that decides gate G0.

**Architecture:** Code is written locally and pushed; every run happens on Deucalion and is done by the
developer, who has the account. Reusable Stage 0 tooling (`tools/deucalion/`) lives on `feat-gpu-port`.
The replay spike lives on the throwaway branch `spike/gpu-replay`, which adds one binding that exports a
single-partition graph as flat arrays and a Python script that replays it with NumPy (correctness), CuPy
(GPU timing, as a CUDA graph) and the CPU engine (baseline). Results go into the spec's Appendix A.

**Tech Stack:** Deucalion modules (GCC 15.2, OpenMPI 5.0.10, CUDA 13.3, NCCL 2.30.4, Boost), Slurm,
`uv`, `just`, nanobind, NumPy, CuPy (`cupy-cuda13x`), Nsight Compute (`ncu`).

**Spec:** `docs/superpowers/specs/2026-10-01-gpu-port-design.md` (branch `feat-gpu-port`): Section 10
(Stage 0 and gate G0), Section 12 (toolchain), Section 13 (performance model), Section 15 (open items),
Appendix A (Deucalion data).

## Global Constraints

- Modules: `OpenMPI/5.0.10-GCC-15.2.0` and `NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0-CMake`, which load
  GCC 15.2.0 and CUDA 13.3.0 (spec Section 12).
- Driver R580 (CUDA 13.0): nothing may rely on PTX compiled by the driver at load time.
- Slurm: GPU partitions `dev-a100-40`/`dev-a100-80` (1 node, 4 h) and `normal-a100-40`/`normal-a100-80`
  (up to 4 nodes, 48 h); 32 CPUs per GPU; `--exclusive` for every timing run (nodes are shared by
  default).
- The Slurm account is never committed: every command reads it from `DEUCALION_ACCOUNT`, which the
  developer exports in their shell profile.
- Gate G0 (spec Section 10): forward replay on one GPU at least 2× faster than the whole 128-core node on
  the same problem. If it fails: stop and reassess with the data.
- The spike branch `spike/gpu-replay` is never merged. Fixes to the CPU path go to `main` as separate PRs
  (D3).
- Python follows the repository's ruff configuration (annotations, Google-style docstrings, no `print`:
  write to `sys.stdout`); every new Python and C++ file starts with the Apache license header.
- Commits: `<type>(<scope>): <gitmoji> <description>` plus `Assisted-by: <harness>:<model>` for agent
  commits, with the executor's own harness and model (for example `Assisted-by: Pi:claude-opus-5-5`).
  Use plain `git`.
- Format Python with the repository's pinned ruff before linting: `uvx ruff@0.15.20 format <file>`; the
  `ruff format` hook only checks. `prek` comes with the default `dev` dependency group; if it is not on
  `PATH`, run it as `uv run --no-sync prek ...`.

The Python license header:

```python
# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

Conventions on Deucalion used below: the repository is cloned at `~/monoprop`, results go to
`~/stage0/` (`mkdir -p ~/stage0`), and `~/monoprop-env.sh` is a copy of `tools/deucalion/env.sh` taken
from `feat-gpu-port` (Task 1), so it stays available when other branches are checked out.

## File structure

| File | Branch | Responsibility |
|---|---|---|
| `tools/deucalion/env.sh` (new) | `feat-gpu-port` | Loads the toolchain and a matching Boost |
| `tools/deucalion/rotation_fraction.py` (new) | `feat-gpu-port` | Rotations per layer and the implied exchange volume per GPU |
| `src/monoprop/bindings/binder.h` | `spike/gpu-replay` | `_spike_replay_arrays()`: a single-partition graph as flat arrays |
| `tools/spike/gpu_replay.py` (new) | `spike/gpu-replay` | `export`, `numpy`, `cpu` and `gpu` subcommands |
| `docs/superpowers/specs/2026-10-01-gpu-port-design.md` | `feat-gpu-port` | Stage 0 results (Appendix A.7), G0 decision, open items |

---

### Task 1: Deucalion environment script

**Files:**
- Create: `tools/deucalion/env.sh` (branch `feat-gpu-port`)

**Interfaces:**
- Produces: `source ~/monoprop-env.sh` loads the toolchain, the newest Boost ≥ 1.85 built with
  GCC 15.2.0, and exports `CC=gcc CXX=g++ CUDAHOSTCXX=g++`; it returns non-zero (without exiting the
  shell) when a module is missing.

- [ ] **Step 1: Write the script**

On `feat-gpu-port` (`git switch feat-gpu-port && git pull`), create `tools/deucalion/env.sh`:

```bash
# Source, don't execute:  source tools/deucalion/env.sh
#
# Deucalion toolchain for the GPU port (spec Section 12): GCC 15.2, OpenMPI 5.0.10, CUDA 13.3 and
# NCCL 2.30.4 with its CMake config, plus the newest Boost >= 1.85 built with GCC 15.2.0.

module purge
module load OpenMPI/5.0.10-GCC-15.2.0 NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0-CMake || return 1

# Lmod's terse listing goes to stderr and may tag the default version with "(default)".
boost_module="$(module -t avail Boost 2>&1 \
    | sed -e 's/(default)$//' \
    | grep -E '^Boost/1\.(8[5-9]|9[0-9])(\.[0-9]+)*-GCC(core)?-15\.2\.0$' \
    | sort -V | tail -n 1)"
if [[ -z "$boost_module" ]]; then
    echo "env.sh: no Boost >= 1.85 module built with GCC 15.2.0; ask the Deucalion admins for one" >&2
    unset boost_module
    return 1
fi
module load "$boost_module" || return 1
unset boost_module

export CC=gcc CXX=g++ CUDAHOSTCXX=g++
```

- [ ] **Step 2: Check the syntax locally**

Run: `bash -n tools/deucalion/env.sh`
Expected: no output, exit status 0.

- [ ] **Step 3: Commit and push**

```bash
git add tools/deucalion/env.sh
git commit -m "chore(deucalion): 🔧 add the Deucalion toolchain environment" -m "Assisted-by: <harness>:<model>"
git push
```

- [ ] **Step 4: Install it on Deucalion (developer)**

```bash
cd ~/monoprop && git fetch && git show origin/feat-gpu-port:tools/deucalion/env.sh > ~/monoprop-env.sh
source ~/monoprop-env.sh && module list
command -v just || uv tool install rust-just
just --version
mkdir -p ~/stage0
```

Expected: `module list` shows the 20 toolchain modules from spec Appendix A.2 plus a `Boost/…-GCC-15.2.0`
module; `just --version` prints a version. If `env.sh` reports the missing Boost, stop and ask the
admins for a Boost ≥ 1.85 built with GCC 15.2.0, as with the NCCL CMake config.

---

### Task 2: CPU path and shared-core check with the GCC 15.2 toolchain

**Files:** none (runs only; results go into Task 7).

**Interfaces:**
- Consumes: `~/monoprop-env.sh` (Task 1); for Step 3, the shared-core foundation plan
  (`docs/superpowers/plans/2026-10-01-shared-core-foundation.md`) merged into `main`, or its branch
  `feat/shared-core-foundation`.
- Produces: `~/stage0/cpu-gcc15.log`, `~/stage0/cpu-gcc15-mpi.log`, `~/stage0/shared-nvcc.log`.

- [ ] **Step 1: Serial build and tests (developer, on Deucalion)**

```bash
cd ~/monoprop && git switch main && git pull
srun -p dev-a100-40 -A "$DEUCALION_ACCOUNT" -N1 --gpus=1 --cpus-per-task=32 -t 02:00:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop &&
    just build --group workspace-test && just test-py && just test-cpp' 2>&1 | tee ~/stage0/cpu-gcc15.log
```

Expected: the pytest summary line reports only `passed` (and `skipped`), and CTest ends with
`100% tests passed`.

- [ ] **Step 2: MPI build and tests (developer, on Deucalion)**

```bash
srun -p dev-a100-40 -A "$DEUCALION_ACCOUNT" -N1 --gpus=1 --cpus-per-task=32 -t 02:00:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop &&
    monoprop_ENABLE_MPI=ON just build --group workspace-test --reinstall-package monoprop --no-cache &&
    just test-cpp-mpi && just test-py-mpi 2' 2>&1 | tee ~/stage0/cpu-gcc15-mpi.log
```

Expected: CTest ends with `100% tests passed`; the two-rank pytest run reports only `passed`/`skipped`.

If Step 1 or Step 2 fails: keep the log, stop this task, and report the failing test names and the first
error. Fixes are a separate PR into `main`; rerun the step once it has merged.

- [ ] **Step 3: Shared headers under Deucalion's nvcc (developer, login node)**

```bash
cd ~/monoprop && git switch main && git pull     # or: git switch feat/shared-core-foundation
source ~/monoprop-env.sh && just check-shared-nvcc 2>&1 | tee ~/stage0/shared-nvcc.log
```

Expected last line: `shared headers compile as device code; nvcc rejects host-only calls from device code`.
This settles spec Section 15's safety-net item with Deucalion's own nvcc and GCC 15.2.

---

### Task 3: Rotations per layer of the ladder circuits

**Files:**
- Create: `tools/deucalion/rotation_fraction.py` (branch `feat-gpu-port`)

**Interfaces:**
- Consumes: `monoprop_bench_tools.models.make_random_problem` and `build_random_propagator`;
  `MajoranaPropagator.size()`, `.graph_layers`, `.graph_size()` (returns `(cosine_only, rotations)`).
- Produces: one JSON record per run on stdout with `terms`, `layers`, `rotations`,
  `rotations_per_layer`, `rotations_per_layer_per_term`, `build_s`, and per GPU count the exchanged MB per
  GPU per remote layer and the routing regime.

- [ ] **Step 1: Write the script**

Create `tools/deucalion/rotation_fraction.py` (Python license header first):

```python
# ruff: noqa: INP001

"""Rotations per graph layer of the ladder's random rows, measured on CPU graphs.

Stage 0 of the GPU port (spec Sections 13 and 15). In a layer whose partner terms live on another GPU,
every rotation crosses GPUs, so the rotation count per layer sets the exchange volume.

Run with ``uv run --no-sync python tools/deucalion/rotation_fraction.py [options]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from monoprop_bench_tools.models import build_random_propagator, make_random_problem

# Below about this many bytes per GPU per layer, pairwise routing is as fast as all-to-all
# (spec Section 13).
PAIRWISE_THRESHOLD_BYTES = 1_000_000


def _parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The ladder-row options, the picture, and the GPU counts to report the exchange for.
    """
    parser = argparse.ArgumentParser(description="Rotations per layer of a random ladder row.")
    parser.add_argument("--picture", choices=("heisenberg", "schrodinger"), default="heisenberg")
    parser.add_argument("--obs-terms", type=int, default=295_000)
    parser.add_argument("--num-generators", type=int, default=1000)
    parser.add_argument("--num-modes", type=int, default=142)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 8, 16])
    return parser.parse_args()


def main() -> None:
    """Build the row's graph and report its rotations and the exchange they imply."""
    args = _parse_args()
    problem = make_random_problem(
        gen_length=4,
        obs_terms=args.obs_terms,
        num_generators=args.num_generators,
        num_modes=args.num_modes,
        cutoff=args.cutoff,
        seed=args.seed,
    )
    propagator, circuit = build_random_propagator(
        problem, schrodinger=args.picture == "schrodinger"
    )
    start = time.perf_counter()
    propagator.build_graph(circuit)
    build_s = time.perf_counter() - start

    terms = propagator.size()
    layers = propagator.graph_layers
    cosine_only, rotations = propagator.graph_size()
    rotations_per_layer = rotations / layers

    # A remote rotation has one endpoint on each of two GPUs, and each GPU sends 8 bytes per endpoint it
    # owns for the energy (16 for the gradient); over g GPUs each owns 2 * rotations / g endpoints.
    exchange = {}
    for gpus in args.gpus:
        energy_bytes = 8 * 2 * rotations_per_layer / gpus
        exchange[str(gpus)] = {
            "energy_mb_per_gpu_per_remote_layer": energy_bytes / 1e6,
            "gradient_mb_per_gpu_per_remote_layer": 2 * energy_bytes / 1e6,
            "regime": "all-to-all"
            if energy_bytes > PAIRWISE_THRESHOLD_BYTES
            else "pairwise",
        }

    record = {
        "picture": args.picture,
        "obs_terms": args.obs_terms,
        "num_generators": args.num_generators,
        "num_modes": args.num_modes,
        "cutoff": args.cutoff,
        "partitions": os.environ.get("monoprop_PARTITIONS", "auto"),
        "terms": terms,
        "layers": layers,
        "rotations": rotations,
        "cosine_only": cosine_only,
        "rotations_per_layer": rotations_per_layer,
        "rotations_per_layer_per_term": rotations_per_layer / terms,
        "build_s": build_s,
        "exchange": exchange,
    }
    sys.stdout.write(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it locally on a small row**

```bash
just build --group workspace-test
uv run --no-sync python tools/deucalion/rotation_fraction.py --obs-terms 20000 --num-generators 200
```

Expected: a JSON record with `terms > 0`, `layers == 200`, `rotations > 0`, and
`0 < rotations_per_layer_per_term < 1`.

- [ ] **Step 3: Format, lint, commit and push**

```bash
uvx ruff@0.15.20 format tools/deucalion/rotation_fraction.py
prek run --files tools/deucalion/rotation_fraction.py
git add tools/deucalion/rotation_fraction.py
git commit -m "feat(deucalion): ✨ measure rotations per layer of the ladder rows" -m "Assisted-by: <harness>:<model>"
git push
```

Expected: all hooks pass before the commit.

- [ ] **Step 4: Run the ladder rows on Deucalion (developer)**

```bash
cd ~/monoprop && git switch feat-gpu-port && git pull
srun -p dev-a100-80 -A "$DEUCALION_ACCOUNT" -N1 --exclusive --ntasks=1 --gpus=4 --cpus-per-task=128 -t 04:00:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop && just build --group workspace-test &&
    uv run --no-sync python tools/deucalion/rotation_fraction.py \
        > ~/stage0/rotations-heisenberg-l1.json &&
    uv run --no-sync python tools/deucalion/rotation_fraction.py --obs-terms 14750000 \
        > ~/stage0/rotations-heisenberg-l2a.json &&
    uv run --no-sync python tools/deucalion/rotation_fraction.py --picture schrodinger \
        --num-generators 5800 --obs-terms 200000 > ~/stage0/rotations-schrodinger-l2a.json'
```

Expected: three JSON files. Their `terms` should match the LADDER row sizes (about 19.9M, 0.949B and
0.597B terms).

---

### Task 4: Replay spike code

**Files:**
- Modify: `src/monoprop/bindings/binder.h` (branch `spike/gpu-replay`; end of `bind_monomial_propagator`)
- Create: `tools/spike/gpu_replay.py` (branch `spike/gpu-replay`)

**Interfaces:**
- Produces: `MonomialPropagatorNNN._spike_replay_arrays() -> dict` with `num_terms` (int) and raw
  little-endian bytes for `cos` (uint32), `cos_off` (uint64, layers + 1 entries), `b` (uint32), `d`
  (uint32), `phase` (int8), `end_off` (uint64, layers + 1 entries), `param_index` (uint64), `gen_coeff`
  (float64). Layer `i`'s cosine set is `cos[cos_off[i]:cos_off[i+1]]`; its rotation endpoints are
  `b`/`d`/`phase` over `[end_off[i], end_off[i+1])`, where `b` holds the `sin_send` indices, `d` the
  `sin_recv` indices and `phase` the signed phases. Only valid for a single-partition propagator.
- Produces: `tools/spike/gpu_replay.py {export,numpy,cpu,gpu}`; `export` writes an `.npz` with the arrays
  above plus `params`, `initial` and `reference`; every subcommand prints one JSON record.

The replay step mirrors `evolve_step` in `cpp/monoprop/Evolution.cpp` for the self slot: snapshot
`op[b]`, scale the layer's cosine set by `cos(2θ)`, then `op[d] += sin(2θ) · phase · snapshot`, with
`θ = params[param_index] · gen_coeff` per layer in graph order.

The spec's Stage 0 row says "`graph_data()` bound in Python". This plan binds a dedicated
`_spike_replay_arrays()` instead: `graph_data()` returns nested Python lists, one Python integer per
index, which is far too slow and too large for graphs of tens of millions of terms. The flat byte arrays
carry the same information (cosine sets, `sin_send`/`sin_recv` indices and phases per layer) plus each
layer's parameter index and generator coefficient.

- [ ] **Step 1: Create the spike branch**

```bash
git switch main && git pull && git switch -c spike/gpu-replay
```

- [ ] **Step 2: Add the binding**

In `src/monoprop/bindings/binder.h`, at the end of `bind_monomial_propagator`, after the
`graph_memory_breakdown` definition and before the function's closing brace, add:

```cpp
    // SPIKE (spike/gpu-replay, never merged): a single-partition graph flattened for the Stage 0 GPU
    // replay spike. Arrays are raw little-endian bytes for numpy.frombuffer.
    cls.def(
        "_spike_replay_arrays",
        [](MonomialPropagator<NumModes> &self) {
            namespace md = ::monoprop::detail;
            const MPGraph &graph = self.graph(); // throws on a multi-partition propagator
            const auto &inverted = self.mp_op().inverted_index();
            std::vector<uint32_t> cos_idx, b_idx, d_idx;
            std::vector<int8_t> phase;
            std::vector<uint64_t> cos_off{0}, end_off{0}, param_index;
            std::vector<double> gen_coeff;
            const auto push_cos = [&cos_idx](size_t i) { cos_idx.push_back(static_cast<uint32_t>(i)); };
            for (size_t layer = 0; layer < graph.layers(); ++layer) {
                const auto t = graph.get_layer_traversal(layer);
                if (const CosMask *stored = t.stored_cos(); stored != nullptr) {
                    for (const auto &[base, bits] : stored->blocks) {
                        md::for_each_cos_index(base, bits, push_cos);
                    }
                }
                else if (!t.generator_words().empty()) {
                    const auto gen = md::generator_from_words<NumModes>(t.generator_words());
                    const auto fold = md::make_fold_cache<NumModes>(inverted, gen, t.scaled_count(), self.basis());
                    for (const size_t i : md::fold_to_indices<NumModes>(fold)) {
                        push_cos(i);
                    }
                }
                cos_off.push_back(cos_idx.size());
                const auto slot = t.cross_rank_self_slot();
                for (size_t k = 0; k < slot.sin_send_count; ++k) {
                    b_idx.push_back(static_cast<uint32_t>(md::slot_sin_send_index(slot, k)));
                    d_idx.push_back(static_cast<uint32_t>(md::slot_sin_recv_index(slot, k)));
                    phase.push_back(static_cast<int8_t>(md::slot_sin_recv_phase(slot, k)));
                }
                end_off.push_back(b_idx.size());
                param_index.push_back(t.param_index());
                gen_coeff.push_back(t.gen_coeff());
            }
            const auto as_bytes = [](const auto &v) {
                return nb::bytes(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(v[0]));
            };
            nb::dict out;
            out["num_terms"] = self.size();
            out["cos"] = as_bytes(cos_idx);
            out["cos_off"] = as_bytes(cos_off);
            out["b"] = as_bytes(b_idx);
            out["d"] = as_bytes(d_idx);
            out["phase"] = as_bytes(phase);
            out["end_off"] = as_bytes(end_off);
            out["param_index"] = as_bytes(param_index);
            out["gen_coeff"] = as_bytes(gen_coeff);
            return out;
        },
        "SPIKE: flat replay arrays of a single-partition graph");
```

`md` must be spelled `::monoprop::detail`: inside `monoprop::bindings::detail`, a bare `detail::` names
the bindings namespace.

- [ ] **Step 3: Write the spike script**

Create `tools/spike/gpu_replay.py` (Python license header first):

```python
# ruff: noqa: INP001, PLC0415

"""SPIKE (spike/gpu-replay, never merged): Stage 0 forward-replay timing, one GPU against the CPU.

Spec: docs/superpowers/specs/2026-10-01-gpu-port-design.md, Section 10 (gate G0). The problem is the
ladder's random Heisenberg row; the replay is Section 8's forward step with materialised cosine lists in
place of the inverted-index fold.

Subcommands:
    export  single-partition build; writes the replay arrays and a CPU reference to an .npz
            (run with monoprop_PARTITIONS=1)
    numpy   replays an .npz with NumPy and checks it against the reference (no GPU)
    cpu     times the CPU energy functional with the partition count taken from the environment
    gpu     replays an .npz on one GPU with CuPy, as a CUDA graph unless --no-graph, and times it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

from monoprop_bench_tools.models import build_random_propagator, make_random_problem

ARRAY_DTYPES: dict[str, Any] = {
    "cos": np.uint32,
    "cos_off": np.uint64,
    "b": np.uint32,
    "d": np.uint32,
    "phase": np.int8,
    "end_off": np.uint64,
    "param_index": np.uint64,
    "gen_coeff": np.float64,
}

KERNELS = r"""
extern "C" __global__ void gather(const double *op, const unsigned int *idx, double *out, long long n) {
    const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k < n) out[k] = op[idx[k]];
}
extern "C" __global__ void scale(double *op, const unsigned int *idx, double c, long long n) {
    const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k < n) op[idx[k]] *= c;
}
extern "C" __global__ void scatter_add(double *op, const unsigned int *idx, const signed char *phase,
                                       const double *snap, double s, long long n) {
    const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k < n) op[idx[k]] += s * (double)phase[k] * snap[k];
}
"""

THREADS = 256
NUMPY_TOLERANCE = 1e-12
GPU_TOLERANCE = 1e-10


def _emit(record: dict[str, Any]) -> None:
    """Write one JSON record to stdout.

    Args:
        record: The values to report.
    """
    sys.stdout.write(json.dumps(record, indent=2) + "\n")


def _build(args: argparse.Namespace) -> tuple[Any, np.ndarray, float]:
    """Build the random Heisenberg row's graph.

    Args:
        args: The parsed command line.

    Returns:
        The propagator with its graph built, the parameter values, and the build time in seconds.
    """
    problem = make_random_problem(
        gen_length=4,
        obs_terms=args.obs_terms,
        num_generators=args.num_generators,
        num_modes=args.num_modes,
        cutoff=args.cutoff,
        seed=args.seed,
    )
    propagator, circuit = build_random_propagator(problem)
    start = time.perf_counter()
    propagator.build_graph(circuit)
    parameters = np.asarray(problem.parameters, dtype=np.float64)
    return propagator, parameters, time.perf_counter() - start


def _load(path: str) -> dict[str, np.ndarray]:
    """Load an exported .npz fully into memory.

    Args:
        path: The file written by ``export``.

    Returns:
        Every array in the file, by name.
    """
    with np.load(path) as npz:
        return dict(npz)


def _angles(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer cos(2θ) and sin(2θ), θ = params[param_index] · gen_coeff, in graph order.

    Args:
        data: The exported arrays.

    Returns:
        The cosine and sine factors of every layer.
    """
    theta = data["params"][data["param_index"].astype(np.int64)] * data["gen_coeff"]
    return np.cos(2.0 * theta), np.sin(2.0 * theta)


def _replay_numpy(data: dict[str, np.ndarray], layers: int) -> np.ndarray:
    """Replay the first ``layers`` layers on the CPU with NumPy.

    Args:
        data: The exported arrays.
        layers: How many layers to replay, from the first.

    Returns:
        The evolved coefficients.
    """
    cos_idx, cos_off = data["cos"], data["cos_off"].astype(np.int64)
    b, d, end_off = data["b"], data["d"], data["end_off"].astype(np.int64)
    phase = data["phase"].astype(np.float64)
    c2, s2 = _angles(data)
    op = data["initial"].copy()
    for i in range(layers):
        lo, hi = end_off[i], end_off[i + 1]
        snap = op[b[lo:hi]]
        op[cos_idx[cos_off[i] : cos_off[i + 1]]] *= c2[i]
        op[d[lo:hi]] += s2[i] * phase[lo:hi] * snap
    return op


def _max_rel_err(values: np.ndarray, reference: np.ndarray) -> float:
    """Largest absolute difference, relative to the reference's largest magnitude (at least 1).

    Args:
        values: Coefficients to check.
        reference: Coefficients to check against.

    Returns:
        The relative error.
    """
    if reference.size == 0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(reference))))
    return float(np.max(np.abs(values - reference))) / scale


def cmd_export(args: argparse.Namespace) -> None:
    """Write the replay arrays, the initial coefficients and the CPU reference to an .npz.

    Args:
        args: The parsed command line.
    """
    if os.environ.get("monoprop_PARTITIONS") != "1":
        sys.exit("export needs a single-partition graph: run it with monoprop_PARTITIONS=1")
    propagator, parameters, build_s = _build(args)
    # Zero angles make every layer the identity, so this is the operator before evolution.
    initial = np.asarray(propagator.contract_partially(np.zeros_like(parameters), inplace=False))
    reference = np.asarray(propagator.contract_partially(parameters, inplace=False))
    raw = propagator._simulator._spike_replay_arrays()
    arrays = {name: np.frombuffer(raw[name], dtype=dtype) for name, dtype in ARRAY_DTYPES.items()}
    np.savez(args.npz, params=parameters, initial=initial, reference=reference, **arrays)
    _emit({
        "terms": int(raw["num_terms"]),
        "layers": int(arrays["param_index"].size),
        "cos_entries": int(arrays["cos"].size),
        "endpoints": int(arrays["b"].size),
        "build_s": build_s,
        "npz": args.npz,
    })


def cmd_numpy(args: argparse.Namespace) -> None:
    """Replay an export with NumPy and require agreement with the CPU reference.

    Args:
        args: The parsed command line.
    """
    data = _load(args.npz)
    layers = int(data["param_index"].size)
    error = _max_rel_err(_replay_numpy(data, layers), data["reference"])
    _emit({"layers": layers, "max_rel_err": error})
    if error > NUMPY_TOLERANCE:
        sys.exit(f"NumPy replay disagrees with the CPU reference: max relative error {error:.3e}")


def cmd_cpu(args: argparse.Namespace) -> None:
    """Time the CPU energy functional (forward replay plus a sparse dot product).

    Args:
        args: The parsed command line.
    """
    propagator, parameters, build_s = _build(args)
    energy = propagator.expectation_value_functional()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        energy(parameters)
        times.append(time.perf_counter() - start)
    _emit({
        "terms": propagator.size(),
        "layers": propagator.graph_layers,
        "partitions": os.environ.get("monoprop_PARTITIONS", "auto"),
        "build_s": build_s,
        "energy_s": times,
        "energy_min_s": min(times),
    })


def cmd_gpu(args: argparse.Namespace) -> None:
    """Replay an export on one GPU with CuPy, check it, and time it.

    Args:
        args: The parsed command line.
    """
    import cupy as cp

    data = _load(args.npz)
    total_layers = int(data["param_index"].size)
    layers = total_layers if args.layers is None else min(args.layers, total_layers)
    cos_off = data["cos_off"].astype(np.int64)
    end_off = data["end_off"].astype(np.int64)
    c2, s2 = _angles(data)
    max_endpoints = int(np.max(np.diff(end_off))) if total_layers else 0

    needed = 2 * data["initial"].nbytes + 8 * max_endpoints
    needed += sum(data[name].nbytes for name in ("cos", "b", "d", "phase"))
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    if needed > 0.9 * free_bytes:
        sys.exit(f"needs {needed / 1e9:.1f} GB on the GPU, {free_bytes / 1e9:.1f} GB free")

    module = cp.RawModule(code=KERNELS)
    gather = module.get_function("gather")
    scale = module.get_function("scale")
    scatter_add = module.get_function("scatter_add")

    initial = cp.asarray(data["initial"])
    op = cp.empty_like(initial)
    cos_idx = cp.asarray(data["cos"])
    b = cp.asarray(data["b"])
    d = cp.asarray(data["d"])
    phase = cp.asarray(data["phase"])
    snap = cp.empty(max(1, max_endpoints), dtype=cp.float64)

    def blocks(n: int) -> tuple[int]:
        return ((n + THREADS - 1) // THREADS,)

    def replay() -> None:
        for i in range(layers):
            lo, hi = int(end_off[i]), int(end_off[i + 1])
            clo, chi = int(cos_off[i]), int(cos_off[i + 1])
            if hi > lo:
                gather(blocks(hi - lo), (THREADS,), (op, b[lo:hi], snap, np.int64(hi - lo)))
            if chi > clo:
                scale(
                    blocks(chi - clo),
                    (THREADS,),
                    (op, cos_idx[clo:chi], np.float64(c2[i]), np.int64(chi - clo)),
                )
            if hi > lo:
                scatter_add(
                    blocks(hi - lo),
                    (THREADS,),
                    (op, d[lo:hi], phase[lo:hi], snap, np.float64(s2[i]), np.int64(hi - lo)),
                )

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        # Warm-up outside the capture, so loading the kernels is not part of any timed run.
        cp.copyto(op, initial)
        replay()
        stream.synchronize()
        graph = None
        if args.graph:
            stream.begin_capture()
            replay()
            graph = stream.end_capture()

        times = []
        for _ in range(args.repeats):
            cp.copyto(op, initial)
            start = stream.record()
            if graph is None:
                replay()
            else:
                graph.launch(stream)
            end = stream.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end) / 1e3)
        result = cp.asnumpy(op)

    reference = data["reference"] if layers == total_layers else _replay_numpy(data, layers)
    error = _max_rel_err(result, reference)
    # Lower bound on traffic: per cosine entry 4 B index + 16 B read/write; per endpoint 20 B for the
    # gather and 29 B for the scatter-add. Scattered accesses move whole 32 B sectors, so real traffic
    # is higher.
    bytes_moved = 20 * int(cos_off[layers]) + 49 * int(end_off[layers])
    best = min(times)
    _emit({
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "terms": int(data["initial"].size),
        "layers": layers,
        "graph": args.graph,
        "gpu_s": times,
        "gpu_min_s": best,
        "bytes_moved_lower_bound": bytes_moved,
        "effective_gbps": bytes_moved / best / 1e9,
        "max_rel_err": error,
    })
    if error > GPU_TOLERANCE:
        sys.exit(f"GPU replay disagrees with the reference: max relative error {error:.3e}")


def _parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The subcommand and its options.
    """
    parser = argparse.ArgumentParser(description="Stage 0 replay spike.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_problem_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--obs-terms", type=int, default=295_000)
        command.add_argument("--num-generators", type=int, default=1000)
        command.add_argument("--num-modes", type=int, default=142)
        command.add_argument("--cutoff", type=int, default=6)
        command.add_argument("--seed", type=int, default=0)

    export = sub.add_parser("export")
    add_problem_options(export)
    export.add_argument("--npz", required=True)

    numpy_cmd = sub.add_parser("numpy")
    numpy_cmd.add_argument("--npz", required=True)

    cpu = sub.add_parser("cpu")
    add_problem_options(cpu)
    cpu.add_argument("--repeats", type=int, default=3)

    gpu = sub.add_parser("gpu")
    gpu.add_argument("--npz", required=True)
    gpu.add_argument("--layers", type=int, default=None)
    gpu.add_argument("--repeats", type=int, default=5)
    gpu.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """Dispatch to the requested subcommand."""
    args = _parse_args()
    {"export": cmd_export, "numpy": cmd_numpy, "cpu": cmd_cpu, "gpu": cmd_gpu}[args.command](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Check the export and the replay locally (no GPU)**

```bash
uvx ruff@0.15.20 format tools/spike/gpu_replay.py
just build --group workspace-test
monoprop_PARTITIONS=1 uv run --no-sync python tools/spike/gpu_replay.py export \
    --obs-terms 2000 --num-generators 100 --num-modes 32 --npz /tmp/spike-small.npz
uv run --no-sync python tools/spike/gpu_replay.py numpy --npz /tmp/spike-small.npz
```

Expected: `export` prints `layers: 100` and non-zero `cos_entries` and `endpoints`; `numpy` prints
`max_rel_err` below `1e-12` and exits 0. A larger error means the arrays or the angles do not match
`evolve_step`: fix this step before going to Deucalion.

- [ ] **Step 5: Commit and push the spike branch**

```bash
git add src/monoprop/bindings/binder.h tools/spike/gpu_replay.py
git commit -m "test(spike): 🧪 add the Stage 0 GPU replay spike" -m "Assisted-by: <harness>:<model>"
git push -u origin spike/gpu-replay
```

---

### Task 5: Replay spike on Deucalion and gate G0

**Files:** none (runs only; results go into Task 7).

**Interfaces:**
- Consumes: branch `spike/gpu-replay` (Task 4), `~/monoprop-env.sh` (Task 1).
- Produces: in `~/stage0/`: `spike-<size>.npz`, `export-<size>.json`, `gpu-<size>.json`,
  `cpu-<size>.json` for each size, plus `spike-ncu.txt`.

Sizes are named by `--obs-terms`: `l1` = 295000 (about 19.9M terms), then `1m` = 1000000 and
`3m` = 3000000. Stop scaling at the first size the `gpu` subcommand refuses for lack of memory.

- [ ] **Step 1: Build the spike and install CuPy (developer)**

```bash
cd ~/monoprop && git fetch && git switch spike/gpu-replay
source ~/monoprop-env.sh && just build --group workspace-test
uv pip install "cupy-cuda13x>=14"
srun -p dev-a100-80 -A "$DEUCALION_ACCOUNT" -N1 --gpus=1 --cpus-per-task=32 -t 00:10:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop &&
    uv run --no-sync python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion(), cupy.cuda.Device(0).compute_capability)"'
```

Expected: a CUDA 13 runtime version (`130xx`) and `80`. If `cupy-cuda13x` fails to install or import,
use `uv pip install "cupy-cuda12x[ctk]>=14"` instead; the R580 driver runs CUDA 12 natively. `just build`
removes CuPy again, so reinstall it after any rebuild.

- [ ] **Step 2: Export, check and replay the L1 size (developer)**

```bash
srun -p dev-a100-80 -A "$DEUCALION_ACCOUNT" -N1 --gpus=1 --cpus-per-task=32 -t 02:00:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop &&
    monoprop_PARTITIONS=1 uv run --no-sync python tools/spike/gpu_replay.py export \
        --obs-terms 295000 --npz ~/stage0/spike-l1.npz > ~/stage0/export-l1.json &&
    uv run --no-sync python tools/spike/gpu_replay.py numpy --npz ~/stage0/spike-l1.npz &&
    uv run --no-sync python tools/spike/gpu_replay.py gpu --npz ~/stage0/spike-l1.npz \
        > ~/stage0/gpu-l1.json'
```

Expected: `numpy` reports `max_rel_err` below `1e-12`; `gpu-l1.json` reports `max_rel_err` below `1e-10`
and five `gpu_s` timings.

- [ ] **Step 3: Time the CPU at the same size (developer, whole node)**

```bash
srun -p dev-a100-80 -A "$DEUCALION_ACCOUNT" -N1 --exclusive --ntasks=1 --gpus=4 --cpus-per-task=128 -t 02:00:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop &&
    monoprop_PARTITIONS=128 monoprop_NUM_THREADS=128 uv run --no-sync python tools/spike/gpu_replay.py cpu \
        --obs-terms 295000 > ~/stage0/cpu-l1.json'
```

Expected: `cpu-l1.json` with the same `terms` and `layers` as `export-l1.json`, and three `energy_s` timings.

- [ ] **Step 4: Scale up (developer)**

Repeat Steps 2 and 3 with `--obs-terms 1000000` (files `*-1m.*`), then with `--obs-terms 3000000`
(files `*-3m.*`), using `-t 04:00:00`. Stop at the first size where `gpu` exits with
`needs … GB on the GPU`.

- [ ] **Step 5: Nsight Compute on the first layers (developer)**

```bash
srun -p dev-a100-80 -A "$DEUCALION_ACCOUNT" -N1 --exclusive --gpus=1 --cpus-per-task=32 -t 01:00:00 bash -lc '
    source ~/monoprop-env.sh && cd ~/monoprop &&
    ncu --target-processes all --launch-count 30 --section SpeedOfLight --section MemoryWorkloadAnalysis \
        -o ~/stage0/spike-ncu uv run --no-sync python tools/spike/gpu_replay.py gpu \
        --npz ~/stage0/spike-l1.npz --layers 10 --repeats 1 --no-graph &&
    ncu --import ~/stage0/spike-ncu.ncu-rep --page details > ~/stage0/spike-ncu.txt'
```

Expected: `spike-ncu.txt` lists `gather`, `scale` and `scatter_add` launches with memory throughput, L2
hit rate and sectors per request. If `ncu` reports `ERR_NVGPUCTRPERM`, the counters are restricted to
administrators: record that and skip this step.

- [ ] **Step 6: Decide G0**

For each size, compute `speedup = cpu.energy_min_s / gpu.gpu_min_s` from `cpu-<size>.json` and
`gpu-<size>.json`. G0 passes when the speedup at the largest size that fits is at least 2. The GPU timing
leaves out the sparse dot product the CPU energy includes, which touches about 0.07% of the rows; note
this next to the result.

---

### Task 6: Optional interconnect measurements

**Files:** none (runs only, in the developer's `nccl-tests` directory holding `xchg.cpp` from spec
Appendix A.6).

**Interfaces:**
- Produces: `~/stage0/xchg-win0-4g.txt`, `~/stage0/xchg-win0-write.txt`, `~/stage0/xchg-xor-repeat.txt`.

- [ ] **Step 1: All-to-all at larger sizes and in write mode (developer)**

In `xchg.cpp`, change `const size_t max_bytes = size_t{1} << 30;` to `const size_t max_bytes = size_t{1} << 32;`
and rebuild with the command from spec Appendix A.6. Then:

```bash
srun -p dev-a100-40 -A "$DEUCALION_ACCOUNT" -N1 --exclusive --ntasks-per-node=4 --gpus=4 --cpus-per-task=32 \
    ./xchg win 0 > ~/stage0/xchg-win0-4g.txt
srun -p dev-a100-40 -A "$DEUCALION_ACCOUNT" -N1 --exclusive --ntasks-per-node=4 --gpus=4 --cpus-per-task=32 \
    env NCCL_P2P_READ_ENABLE=0 ./xchg win 0 > ~/stage0/xchg-win0-write.txt
```

Expected: tables like spec Appendix A.6, extending to 4 GiB per GPU.

- [ ] **Step 2: Repeat the pairwise runs (developer)**

```bash
srun -p dev-a100-40 -A "$DEUCALION_ACCOUNT" -N1 --exclusive --ntasks-per-node=4 --gpus=4 --cpus-per-task=32 \
    bash -c 'for run in 1 2 3; do ./xchg xor 1; ./xchg xor 2; ./xchg xor 3; done' > ~/stage0/xchg-xor-repeat.txt
```

Expected: three tables per pairing. Compare the largest-size bandwidth of `xor 2` with `xor 1` and `xor 3`.

---

### Task 7: Record the results in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-01-gpu-port-design.md` (branch `feat-gpu-port`): Section 10
  (after the stage table), Section 15, and a new Appendix A.7 at the end of Appendix A.

**Interfaces:**
- Consumes: every file in `~/stage0/` from Tasks 2, 3, 5 and 6, copied from Deucalion or pasted in.

- [ ] **Step 1: Add Appendix A.7**

Insert before `## Appendix B` a section with this structure. Each cell takes the value from the named
file and field; leave out the tables of tasks that were not run.

```markdown
### A.7 Stage 0 results

**CPU path with GCC 15.2** (`cpu-gcc15.log`, `cpu-gcc15-mpi.log`): Python and C++ suites, serial and
MPI: pass, or the failing tests and the PR that fixed them.

**Shared headers under nvcc** (`shared-nvcc.log`): the last line of `just check-shared-nvcc`.

**Rotations per layer** (`rotations-*.json`):

| Row | Terms | Layers | Rotations per layer | Per term | Energy MB per GPU per remote layer (4 / 8 / 16 GPUs) | Regime at 4 GPUs |
|---|---|---|---|---|---|---|

**Replay spike** (`export-*.json`, `gpu-*.json`, `cpu-*.json`):

| Size | Terms | Layers | GPU replay min (s) | CPU energy min, 128 cores (s) | Speedup | Effective GB/s (lower bound) | Max rel. error |
|---|---|---|---|---|---|---|---|

**Nsight Compute** (`spike-ncu.txt`): per kernel (`gather`, `scale`, `scatter_add`): memory throughput
as % of peak, L2 hit rate, sectors per request.

**Interconnect** (`xchg-*.txt`): `win 0` bandwidth at 4 GiB per GPU, in read and in write mode; `xor 2`
against `xor 1` and `xor 3` over three runs.
```

- [ ] **Step 2: Record the G0 decision**

In Section 10, after the stage table, add one paragraph: `**G0 (Stage 0, <date>):** passed/failed` with
the speedup at the largest size that fits one GPU, the sizes measured, and a pointer to Appendix A.7.
If G0 failed, add what the data suggests (from the Nsight metrics and the effective bandwidth) and state
that the port is on hold until reassessed.

- [ ] **Step 3: Update the open items**

In Section 15, remove the items Stage 0 settled (rotation fraction, CPU build with GCC 15.2, safety-net
check, and each optional measurement that was run), and renumber. Where a result changes the
performance model (for example the all-to-all efficiency in write mode), update the table in Section 13
to match Appendix A.7.

- [ ] **Step 4: Commit and push**

```bash
prek run --files docs/superpowers/specs/2026-10-01-gpu-port-design.md
git add docs/superpowers/specs/2026-10-01-gpu-port-design.md
git commit -m "docs(gpu): 📝 record the Stage 0 results and the G0 decision" -m "Assisted-by: <harness>:<model>"
git push
```

Expected: hooks pass (`don't commit to branch` passes on `feat-gpu-port`).
