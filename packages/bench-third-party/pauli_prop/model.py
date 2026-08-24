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

"""The benchmark model: a Trotterized 2D tilted-field Ising model (TFIM).

Every backend in `backends.py` propagates *this* circuit and observable, so the
definition lives here once. Nothing in this module imports a propagation library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(__file__).parent / "settings.json"


@dataclass(frozen=True)
class Settings:
    """One benchmark point: lattice size, couplings, and truncation."""

    nx: int
    ny: int
    hx: float
    hz: float
    j: float
    dt: float
    step_min: int
    step_max: int
    step_size: int
    lower_atol: float
    # None means "no weight cutoff", which each backend spells as num_qubits.
    cutoff: int | None
    # None means "the central horizontally-adjacent pair", which is what the
    # committed 6x6 setting ([20, 21]) is — so the lattice can be resized without
    # the observable silently landing on a non-adjacent or out-of-range pair.
    obs_qubits: tuple[int, int] | None = None

    @classmethod
    def load(cls, path: Path | str = SETTINGS_PATH, **overrides: Any) -> Settings:
        with open(path) as f:
            raw = json.load(f)
        obs = raw.get("obs_qubits")
        settings = cls(
            nx=raw["nx"],
            ny=raw["ny"],
            hx=raw["hx"],
            hz=raw["hz"],
            j=raw["j"],
            dt=raw["dt"],
            step_min=raw["step_min"],
            step_max=raw["step_max"],
            step_size=raw["step_size"],
            lower_atol=raw["lower_atol"],
            cutoff=raw.get("cutoff"),
            obs_qubits=tuple(obs) if obs else None,
        )
        # A resized lattice keeps the file's observable only if it still fits and is
        # still a lattice bond; otherwise it falls back to the central pair.
        resized = {k: v for k, v in overrides.items() if v is not None}
        if resized:
            settings = replace(settings, **resized)
            if ("nx" in resized or "ny" in resized) and "obs_qubits" not in resized:
                settings = replace(settings, obs_qubits=None)
        return settings

    @property
    def num_qubits(self) -> int:
        return self.nx * self.ny

    @property
    def max_pauli_weight(self) -> int:
        return self.num_qubits if self.cutoff is None else self.cutoff

    @property
    def step_range(self) -> range:
        return range(self.step_min, self.step_max + 1, self.step_size)

    @property
    def theta_zz(self) -> float:
        return self.dt * self.j

    @property
    def theta_z(self) -> float:
        return self.dt * self.hz

    @property
    def theta_x(self) -> float:
        return self.dt * self.hx

    @property
    def observable_qubits(self) -> tuple[int, int]:
        if self.obs_qubits is not None:
            return self.obs_qubits
        return central_bond(self.nx, self.ny)

    def describe(self) -> str:
        return (
            f"{self.nx}x{self.ny} ({self.num_qubits} qubits), dt={self.dt}, "
            f"atol={self.lower_atol}, weight cutoff={self.max_pauli_weight}, "
            f"obs=ZZ{list(self.observable_qubits)}, "
            f"steps {self.step_min}..{self.step_max} by {self.step_size}"
        )


def grid_edges(nx: int, ny: int) -> list[tuple[int, int]]:
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


def central_bond(nx: int, ny: int) -> tuple[int, int]:
    """The horizontally-adjacent qubit pair nearest the middle of the grid.

    For 6x6 this is (20, 21) — the pair the committed settings.json pins — so
    scaling sweeps stay comparable with the fixed-size benchmark at its base size.
    """
    row = ny // 2
    col = max(nx // 2 - 1, 0)
    idx = row * nx + col
    return idx, idx + 1


def step_circuit(settings: Settings):
    """One Trotter step as a Qiskit circuit: all RZZ bonds, then RZ, then RX."""
    from qiskit.circuit import QuantumCircuit

    nq = settings.num_qubits
    circ = QuantumCircuit(nq)
    for i, k in grid_edges(settings.nx, settings.ny):
        circ.rzz(settings.theta_zz, i, k)
    for i in range(nq):
        circ.rz(settings.theta_z, i)
    for i in range(nq):
        circ.rx(settings.theta_x, i)
    return circ


def observable(settings: Settings):
    """The ZZ observable on the benchmark's central bond, as a SparsePauliOp."""
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_sparse_list(
        [("ZZ", list(settings.observable_qubits), 1.0)], num_qubits=settings.num_qubits
    )


def pauli_rotations(settings: Settings) -> list[tuple[float, list[str], list[int]]]:
    """The same Trotter step as (angle, paulis, qubits) triples.

    Library-agnostic on purpose: backends that build their own gate objects
    (cuPauliProp) map over this instead of re-deriving the circuit.
    """
    gates: list[tuple[float, list[str], list[int]]] = [
        (settings.theta_zz, ["Z", "Z"], [i, k])
        for i, k in grid_edges(settings.nx, settings.ny)
    ]
    gates += [(settings.theta_z, ["Z"], [i]) for i in range(settings.num_qubits)]
    gates += [(settings.theta_x, ["X"], [i]) for i in range(settings.num_qubits)]
    return gates
