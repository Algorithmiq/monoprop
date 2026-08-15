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

Reads off `terms` and a wall time so the campaign's cells can be chosen before spending
node-hours on them, and so `--time` can be set from a measurement rather than a guess.

Deliberately propagate-only. build_graph needs its own propagator (this one has consumed its
circuit), and holding two at these term counts doubles the resident operator for nothing --
the term count is the same either way, which is the number this exists to report.

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

from _builders import MODELS  # noqa: E402


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

    t0 = time.perf_counter()
    propagator, circuit = build_fn(config, comm=None)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(steps):
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
    print(
        f"CALIB model={model} {spec} steps={steps} terms={terms} "
        f"build_s={build_s:.1f} propagate_s={propagate_s:.1f} "
        f"vmhwm_mib={vmhwm_kb() / 1024:.0f} "
        f"b_per_term={b_per_term}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
