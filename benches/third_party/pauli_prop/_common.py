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

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

BENCH_DIR = Path(__file__).parent
SETTINGS_FILE = BENCH_DIR / "settings.json"
RESULTS_FILE = BENCH_DIR / "results.json"

# The repository's own benchmark suite owns the memory instrumentation; this directory is a
# separate uv project, so reach it by path rather than by dependency.
sys.path.insert(0, str(BENCH_DIR.parents[1]))
from _memory import HighWaterMark  # noqa: E402, F401


@dataclass(frozen=True)
class Settings:
    """Simulation parameters shared by every engine, loaded from settings.json."""

    nx: int
    ny: int
    nq: int
    hx: float
    hz: float
    j: float
    dt: float
    theta_x: float
    theta_z: float
    theta_zz: float
    step_range: range
    lower_atol: float
    max_pauli_weight: int
    obs_qubits: tuple[int, int]
    grid_edges: list[tuple[int, int]]


def _grid_edges(nx: int, ny: int) -> list[tuple[int, int]]:
    """Nearest-neighbor edges of an nx-by-ny grid, row-major qubit indexing."""
    edges = []
    for row in range(ny):
        for col in range(nx):
            idx = row * nx + col
            if col + 1 < nx:
                edges.append((idx, idx + 1))
            if row + 1 < ny:
                edges.append((idx, idx + nx))
    return edges


def load_settings() -> Settings:
    with open(SETTINGS_FILE) as file:
        raw = json.load(file)
    nx, ny = raw["nx"], raw["ny"]
    nq = nx * ny
    return Settings(
        nx=nx,
        ny=ny,
        nq=nq,
        hx=raw["hx"],
        hz=raw["hz"],
        j=raw["j"],
        dt=raw["dt"],
        theta_x=raw["dt"] * raw["hx"],
        theta_z=raw["dt"] * raw["hz"],
        theta_zz=raw["dt"] * raw["j"],
        step_range=range(raw["step_min"], raw["step_max"] + 1, raw["step_size"]),
        lower_atol=raw["lower_atol"],
        max_pauli_weight=nq if raw["cutoff"] is None else raw["cutoff"],
        obs_qubits=tuple(raw["obs_qubits"]),
        grid_edges=_grid_edges(nx, ny),
    )


_RESULTS_KEYS = (
    "step_range",
    "num_terms",
    "runtime",
    "memory",
    "native_memory",
    "expvals",
)


def init_results(settings: Settings) -> None:
    """(Re)create results.json's skeleton from settings.json.

    Called unconditionally by run_monoprop.py, since it's meant to start every full run from
    scratch. Every other engine script instead calls ensure_results_file(), which only creates
    the skeleton if results.json is missing/invalid, so it doesn't clobber earlier engines'
    results when run after them.
    """
    RESULTS_FILE.write_text(
        json.dumps(
            dict.fromkeys(_RESULTS_KEYS[1:], {})
            | {"step_range": list(settings.step_range)},
            indent=4,
        )
    )


def ensure_results_file(settings: Settings) -> None:
    """Make sure results.json exists and has the expected shape before this engine starts.

    Lets every engine script be run standalone, in any order, without results.json already
    existing: without this, a missing/invalid/stale results.json would only surface as a
    confusing JSONDecodeError from update_results() at the very end — after the engine already
    ran its full (possibly expensive) simulation. Unlike init_results(), this leaves an
    already-valid results.json (e.g. with other engines' results already in it) untouched.
    """
    try:
        with open(RESULTS_FILE) as file:
            data = json.load(file)
        if not all(key in data for key in _RESULTS_KEYS):
            raise ValueError("results.json is missing expected keys")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        init_results(settings)


def update_results(
    label: str,
    *,
    runtime: list[float],
    memory: list[float],
    expvals: list[float],
    num_terms: list[int],
    native_memory: list[float] | None = None,
) -> None:
    """Merge one engine's results into the shared results.json (read-modify-write).

    ``memory`` is the OS/driver-level peak (host RSS or GPU device memory, depending on the
    engine) and is what gets plotted. ``native_memory``, where available, is that engine's own
    internal accounting (e.g. monoprop's operator-memory API, cuPauliProp's cupy pool) kept for
    reference/cross-checking — it is not necessarily the true peak, since it can miss memory the
    engine allocates outside of what it tracks itself.
    """
    with open(RESULTS_FILE) as file:
        data = json.load(file)
    data["runtime"][label] = runtime
    data["memory"][label] = memory
    data["expvals"][label] = expvals
    data["num_terms"][label] = num_terms
    if native_memory is not None:
        data.setdefault("native_memory", {})[label] = native_memory
    with open(RESULTS_FILE, "w") as file:
        json.dump(data, file, indent=4)
