#!/usr/bin/env python3
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

"""One calibration point: how many terms a (model, cutoff, size, lower_atol) actually reaches.

Reads off `terms` and a wall time so a run's cells can be chosen before spending
node-hours on them, and so `--time` can be set from a measurement rather than a guess.

Propagate-only by default: the term count is the same either way, and build_graph needs its
own propagator, which doubles the resident operator to learn nothing new about size.

`CALIB_BUILD_GRAPH=1` adds a second process-lifetime measurement that is NOT redundant, and
skipping it cost this project six cells. `propagate` releases each layer as it contracts it;
`steps x build_graph` RETAINS all of them, because that is what a gradient functional needs
to walk backwards. On a model with many Trotter steps the two differ by orders of magnitude:
hubbard at 23.9M terms propagates in 1.7 GiB and builds its 29-layer graph in more than 229
GiB, so a cell sized from `propagate` alone is sized from the wrong number and dies. Size
the `graph` cell from `graph_vmhwm_mib`, never from `vmhwm_mib`.

One process per point, so a point that OOMs or overruns does not take the rest of the ladder
with it and each point's peak RSS is its own.

    terms_calib.py <model> [field=value ...]
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import fields
from pathlib import Path

# Import the builders from the checkout under test: _builders.py is versioned alongside the
# C++, so importing another tree's builders would silently calibrate a different problem.
BENCHES = os.environ.get("CALIB_BENCHES") or str(
    Path(__file__).resolve().parents[2] / "src/mp-invidx/benches"
)
sys.path.insert(0, BENCHES)

try:
    # Post-refactor layout: monoprop_bench_tools is a proper (pip-installed) distribution and
    # `_builders` is now its `models` submodule.
    from monoprop_bench_tools.models import MODELS  # noqa: E402
except ImportError:
    try:
        # Pre-refactor layout: a bare `_builders.py` reached via BENCHES above.
        from _builders import MODELS  # noqa: E402
    except ImportError as exc:
        raise SystemExit(
            "terms_calib: could not import MODELS.\n"
            "  tried: monoprop_bench_tools.models (post-refactor package; not installed in "
            "this interpreter's site-packages)\n"
            "  tried: _builders on sys.path via {benches!r} (pre-refactor layout; "
            "$CALIB_BENCHES, or the mp-invidx checkout if unset)\n"
            "  Neither import succeeded ({exc}). Install monoprop_bench_tools in this venv, "
            "or set CALIB_BENCHES to a checkout containing benches/_builders.py.".format(
                benches=BENCHES, exc=exc
            )
        ) from exc


def coerce(config_cls: type, name: str, raw: str) -> object:
    """Cast `raw` to the type of `config_cls`'s field `name`."""
    for f in fields(config_cls):
        if f.name == name:
            return type(f.default)(raw)
    raise SystemExit(f"{config_cls.__name__} has no field {name!r}")


def vmhwm_kb() -> int:
    """Peak resident set size in KiB, or 0 where /proc is unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def main() -> int:
    """Build the model at one size, propagate it, and print its term count."""
    model = sys.argv[1]
    config_cls, build_fn, steps_fn = MODELS[model]
    overrides = {}
    for arg in sys.argv[2:]:
        key, _, raw = arg.partition("=")
        overrides[key] = coerce(config_cls, key, raw)
    config = config_cls(**overrides)
    steps = steps_fn(config)

    # Graph mode REPLACES propagate rather than following it, because VmHWM is a high-water
    # mark that never falls: run after a propagate arm, the graph's own peak is hidden behind
    # whatever the propagate arm already touched, and the excess over it is not the number a
    # cell has to fit. One process, one arm, one absolute peak.
    graph_mode = os.environ.get("CALIB_BUILD_GRAPH") == "1"

    t0 = time.perf_counter()
    propagator, circuit = build_fn(config, comm=None)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(steps):
        if graph_mode:
            propagator.build_graph(circuit)
        else:
            propagator.propagate(circuit)
    propagate_s = time.perf_counter() - t0

    terms = propagator.size() if callable(propagator.size) else propagator.size

    # Capacity, not residency -- quoted next to VmHWM on purpose, never instead of it.
    b_per_term = {}
    breakdown = getattr(propagator._simulator, "operator_memory_breakdown", None)
    if breakdown is not None and terms:
        b = dict(breakdown())
        b_per_term["invidx"] = round(b["inverted_index_bytes"] / terms, 2)
        b_per_term["total"] = round(b["total_bytes"] / terms, 2)

    spec = " ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "default"
    arm = "build_graph" if graph_mode else "propagate"
    peak_mib = vmhwm_kb() / 1024
    print(
        f"CALIB model={model} {spec} steps={steps} terms={terms} arm={arm} "
        f"build_s={build_s:.1f} {arm}_s={propagate_s:.1f} "
        f"vmhwm_mib={peak_mib:.0f} "
        f"b_per_term_resident={peak_mib * 1024 * 1024 / terms if terms else 0:.0f} "
        f"b_per_term={b_per_term}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
