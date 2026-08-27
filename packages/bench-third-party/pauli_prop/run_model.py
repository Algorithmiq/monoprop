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

"""Run every Python backend at one lattice size and write the per-step curves.

This is the fixed-size benchmark: it produces the `results.json` that `run_model.jl`
appends PauliPropagation.jl to and `plot_results.py` turns into per-step runtime and
memory curves. All backends run in one process here, so the RSS-growth proxies used for
ppvm and Qiskit are only approximate — `run_scaling.py` isolates each backend instead.

For scaling with lattice size, use `run_scaling.py`.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import backends as backend_mod
from model import SETTINGS_PATH, Settings

RESULTS_PATH = Path(__file__).parent / "results.json"


def _provenance(settings: Settings, backends: list[str]) -> dict:
    """Record where and how this run was taken, beside the numbers it produced.

    The scaling sweep stamps every record with its host and thread count; without the same
    here, the hardware a published per-step figure was measured on survives only as prose
    in the docs, where nothing can check it against the data.
    """
    return {
        "host": platform.node(),
        "settings": settings.describe(),
        "memory_metric": backend_mod.HOST_MEMORY_METRIC,
        "threads": {
            backend_mod.LABELS[backend]: backend_mod.num_threads(backend)
            for backend in backends
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    parser.add_argument("--output", "-o", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["monoprop", "ppvm", "qiskit", "cupauliprop"],
        choices=[*backend_mod.CPU_BACKENDS, *backend_mod.GPU_BACKENDS],
    )
    args = parser.parse_args()

    settings = Settings.load(
        args.settings, nx=args.nx, ny=args.ny, step_max=args.step_max
    )
    print(settings.describe(), flush=True)

    results: dict[str, backend_mod.BackendResult] = {}
    for backend in args.backends:
        print(f"running {backend_mod.LABELS[backend]} ...", flush=True)
        if backend == "monoprop":
            results[backend] = backend_mod.run_monoprop(settings)
        elif backend == "ppvm":
            results[backend] = backend_mod.run_ppvm(settings)
        elif backend == "qiskit":
            # pauli-prop has no weight-based cutoff, only a mandatory positive term
            # budget: track monoprop's own per-step term count so the two stay
            # comparable. Without a monoprop run there is nothing to track.
            if "monoprop" not in results:
                raise SystemExit(
                    "the qiskit backend needs a monoprop run in the same invocation "
                    "for its per-step term budget"
                )
            results[backend] = backend_mod.run_qiskit(
                settings, results["monoprop"].num_terms
            )
        elif backend == "cupauliprop":
            results[backend] = backend_mod.run_cupauliprop(settings)

    # Step 0's runtime is dropped: it carries one-off warm-up, and plot_results.py
    # expects the runtime series to be one shorter than the step range.
    payload = {
        "step_range": list(settings.step_range),
        "num_terms": {r.label: r.num_terms for r in results.values()},
        "runtime": {r.label: r.runtime[1:] for r in results.values()},
        "memory": {r.label: r.memory for r in results.values()},
        "native_memory": {
            r.label: r.operator_memory for r in results.values() if r.operator_memory
        },
        "expvals": {r.label: r.expvals for r in results.values()},
    }
    provenance = _provenance(settings, list(results))
    # Preserve any backend already in the file (e.g. PauliPropagation.jl from an earlier
    # run_model.jl run) rather than dropping it on rewrite.
    if args.output.exists():
        with args.output.open() as f:
            existing = json.load(f)
        for section, values in payload.items():
            if section == "step_range":
                continue
            merged = dict(existing.get(section, {}))
            merged.update(values)
            payload[section] = merged
        # Keep the other engines' provenance (e.g. run_model.jl's) alongside this run's.
        merged_prov = dict(existing.get("provenance", {}))
        merged_prov.update(provenance)
        merged_prov["threads"] = {
            **existing.get("provenance", {}).get("threads", {}),
            **provenance["threads"],
        }
        provenance = merged_prov

    payload["provenance"] = provenance
    with args.output.open("w") as f:
        json.dump(payload, f, indent=4)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
