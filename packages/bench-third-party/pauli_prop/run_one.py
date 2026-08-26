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

"""Run ONE backend at ONE lattice size and append a totals record to a JSONL file.

This is the unit of work `run_scaling.py` spawns. One backend per process is what makes
a sweep survivable: a backend that exceeds its time budget, dies on an unavailable GPU,
or gets OOM-killed takes down only its own point, and its RSS reading is not polluted
by the other backends' allocations.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
from pathlib import Path

import backends as backend_mod
from model import SETTINGS_PATH, Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=[*backend_mod.CPU_BACKENDS, *backend_mod.GPU_BACKENDS],
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument(
        "--step-max",
        type=int,
        default=None,
        help="Override the number of Trotter steps from settings.json.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=None,
        help="Term budget for the Qiskit backend (its truncation is term-count based, "
        "not weight based). Defaults to monoprop's final term count when the driver "
        "knows it.",
    )
    parser.add_argument("--settings", type=Path, default=None)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings.load(
        args.settings or SETTINGS_PATH,
        nx=args.nx,
        ny=args.ny,
        step_max=args.step_max,
    )
    label = backend_mod.LABELS[args.backend]
    print(f"[{label}] {settings.describe()}", flush=True)

    # Taken before the first measurement window opens, because opening one resets
    # ru_maxrss too (both are the kernel's mm->hiwater_rss). This covers the import and
    # setup phase; the run's own peaks come from the per-step windows.
    setup_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    if args.backend == "monoprop":
        result = backend_mod.run_monoprop(settings)
    elif args.backend == "ppvm":
        result = backend_mod.run_ppvm(settings)
    elif args.backend == "qiskit":
        if args.max_terms is None:
            raise SystemExit(
                "--max-terms is required for the Qiskit backend: pauli-prop truncates "
                "on a mandatory positive term budget."
            )
        result = backend_mod.run_qiskit(settings, args.max_terms)
    elif args.backend == "cupauliprop":
        result = backend_mod.run_cupauliprop(settings)
    else:  # unreachable: argparse constrains the choices
        raise SystemExit(f"unknown backend {args.backend}")

    # Peak over the whole run: the largest per-step window, or the setup phase if the
    # workload never exceeded it. ru_maxrss cannot be used here (see setup_peak_mb).
    peak_rss_mb = max(setup_peak_mb, *result.memory)

    record = {
        "backend": args.backend,
        "label": label,
        "nx": settings.nx,
        "ny": settings.ny,
        "num_qubits": settings.num_qubits,
        "num_steps": len(result.runtime),
        "threads": backend_mod.num_threads(args.backend),
        "host": platform.node(),
        "status": "ok",
        "total_runtime_s": sum(result.runtime),
        # The per-step series' first entry includes one-off warm-up (allocation, JIT,
        # first-touch), which the fixed-size benchmark drops from its curves.
        "total_runtime_excl_first_s": sum(result.runtime[1:]),
        "final_step_s": result.runtime[-1],
        "final_memory_MB": result.memory[-1],
        "memory_metric": result.memory_metric,
        "operator_memory_MB": result.operator_memory_mb,
        "operator_memory_metric": result.operator_memory_metric,
        "peak_rss_MB": peak_rss_mb,
        "final_num_terms": int(result.num_terms[-1]),
        "final_expval": float(result.expvals[-1]),
        "max_terms_budget": args.max_terms,
        # The truncation, as fields rather than only prose in `settings`, so a plot can label
        # itself without parsing. A weight cutoff of num_qubits cannot truncate anything, so
        # record that case as "none" rather than as a number that looks like a real bound.
        "lower_atol": settings.lower_atol,
        "weight_cutoff": (
            None
            if settings.max_pauli_weight >= settings.num_qubits
            else settings.max_pauli_weight
        ),
        "settings": settings.describe(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"[{label}] {settings.num_qubits} qubits: {record['total_runtime_s']:.3f} s "
        f"total, {record['final_memory_MB']:.1f} MB, "
        f"{record['final_num_terms']:,} terms, expval {record['final_expval']:.10f}",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
