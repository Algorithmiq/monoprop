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

"""Single-layer scaling benchmark for monoprop (Majorana + Pauli).

Applies exactly ONE circuit layer to a local observable and records the number
of operator terms, the operator memory, the wall-clock time to apply the layer,
and the resulting expectation value. Sweeps qubit count for a fixed support
cutoff so that per-term cost (O(cutoff) here vs O(N) in the dense Julia
references) can be compared. Single-threaded: set ``monoprop_NUM_THREADS=1``.

Models mirror the reference Julia packages:
  * ``--basis majorana`` -> 1D Hubbard chain (one first-order Trotter step),
    matching ``MajoranaPropagation.jl`` (support cutoff == ``max_unpaired``).
  * ``--basis pauli``    -> kicked-Ising chain (one X-layer + one ZZ-layer),
    matching ``PauliPropagation.jl`` (support cutoff == ``truncateweight``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from monoprop import Circuit, ExpGate, MajoranaPropagator, Pauli, PauliPropagator
from monoprop.fermi import FermiOperator
from monoprop.pauli import PauliOperator

# --------------------------------------------------------------------------- #
# Majorana / 1D Hubbard (one Trotter step == one layer)
# --------------------------------------------------------------------------- #


def _mode(site: int, spin: str) -> int:
    """Interleaved spin-orbital index (even = up, odd = down)."""
    return 2 * site if spin == "up" else 2 * site + 1


def _bricklayer_topology(num_sites: int) -> list[tuple[int, int]]:
    """0-indexed nearest-neighbour bonds, even bonds then odd bonds."""
    topo = [(i, i + 1) for i in range(0, num_sites - 1, 2)]
    topo += [(i, i + 1) for i in range(1, num_sites - 1, 2)]
    return topo


def _hubbard_terms(
    num_sites: int, hopping: float, interaction: float
) -> list[FermiOperator]:
    """First-order Trotter terms of the 1D Hubbard model (mu = 0)."""
    terms: list[FermiOperator] = []
    topo = _bricklayer_topology(num_sites)
    for spin in ("up", "down"):
        for left_site, right_site in topo:
            left, right = _mode(left_site, spin), _mode(right_site, spin)
            terms.append(
                FermiOperator(
                    terms=[((left, "+"), (right, "-")), ((right, "+"), (left, "-"))],
                    coefficients=[-hopping, -hopping],
                )
            )
    for site in range(num_sites):
        up, down = _mode(site, "up"), _mode(site, "down")
        terms.append(
            FermiOperator(
                terms=[((up, "+"), (up, "-"), (down, "+"), (down, "-"))],
                coefficients=[interaction],
            )
        )
    return terms


def _neel_occupied(num_sites: int, start_spin: str = "up") -> list[int]:
    other = "down" if start_spin == "up" else "up"
    return [
        _mode(s, start_spin if s % 2 == 0 else other) for s in range(num_sites)
    ]


def build_majorana(num_qubits: int, cutoff: int, lower_atol: float, observable: str):
    """Return (propagator_factory, circuit) for a one-layer Hubbard problem.

    ``num_qubits`` must be even; ``num_sites = num_qubits // 2``. ``observable``
    is ``"extensive"`` (total spin-up number, sum over all sites -> O(N) terms
    after one layer, the scaling-relevant choice) or ``"local"`` (single
    mid-site number operator).
    """
    if num_qubits % 2:
        raise ValueError("majorana num_qubits must be even (2 per site)")
    num_sites = num_qubits // 2
    hopping, interaction, dt = 1.0, 1.5, 0.07
    gates = [ExpGate(t) for t in _hubbard_terms(num_sites, hopping, interaction)]
    circuit = Circuit(
        gates=gates,
        parameters=[dt] * len(gates),
        initial_state=_neel_occupied(num_sites, "up"),
    )
    if observable == "extensive":
        # Staggered magnetisation sum_s (-1)^s (n_{s,up} - n_{s,down}). Unlike the
        # total number sum_s n_{s,up} (a Hubbard symmetry that does not evolve),
        # this is non-conserved so it spreads under the circuit -> O(N) terms.
        terms, coeffs = [], []
        for s in range(num_sites):
            sign = 1.0 if s % 2 == 0 else -1.0
            terms.append(((_mode(s, "up"), "+"), (_mode(s, "up"), "-")))
            coeffs.append(sign)
            terms.append(((_mode(s, "down"), "+"), (_mode(s, "down"), "-")))
            coeffs.append(-sign)
        observable_op = FermiOperator(
            terms=terms, coefficients=coeffs, num_modes=num_qubits
        )
    else:
        obs_site = num_sites // 2 - 1
        observable_op = FermiOperator(
            terms=[((_mode(obs_site, "up"), "+"), (_mode(obs_site, "up"), "-"))],
            coefficients=[1.0],
            num_modes=num_qubits,
        )

    def factory():
        return MajoranaPropagator(
            observable_op,
            circuit.initial_state,
            cutoff=cutoff,
            cutoff_type="support",  # matches Julia max_unpaired
            lower_atol=lower_atol,
        )

    return factory, circuit


# --------------------------------------------------------------------------- #
# Pauli / kicked-Ising chain (one X-layer + one ZZ-layer == one layer)
# --------------------------------------------------------------------------- #


def build_pauli(num_qubits: int, cutoff: int, lower_atol: float, observable: str):
    """Return (propagator_factory, circuit) for a one-layer kicked-Ising chain.

    ``observable`` is ``"extensive"`` (sum_i Z_i -> O(N) terms after one layer)
    or ``"local"`` (single mid-chain Z).
    """
    import numpy as np

    theta = np.pi / 4
    coupling = np.pi / 4
    edges = [(i, i + 1) for i in range(num_qubits - 1)]

    gate_angles: list[tuple[ExpGate, float]] = []
    for i in range(num_qubits):
        gate_angles.append(
            (ExpGate(PauliOperator({Pauli("X", (i,)): 1.0}, num_qubits=num_qubits)),
             -theta / 2)
        )
    for i, j in edges:
        gate_angles.append(
            (ExpGate(PauliOperator({Pauli("ZZ", (i, j)): 1.0}, num_qubits=num_qubits)),
             -coupling)
        )
    circuit = Circuit(
        gates=tuple(g for g, _ in gate_angles),
        parameters=tuple(a for _, a in gate_angles),
        initial_state=[],
    )
    def _z(i: int) -> str:
        return "I" * i + "Z" + "I" * (num_qubits - i - 1)

    if observable == "extensive":
        observable_op = PauliOperator(
            {_z(i): 1.0 for i in range(num_qubits)}, num_qubits=num_qubits
        )
    else:
        observable_op = PauliOperator(
            {_z(num_qubits // 2): 1.0}, num_qubits=num_qubits
        )

    def factory():
        return PauliPropagator(
            observable_op,
            circuit.initial_state,
            cutoff=cutoff,  # support/weight cutoff == Julia truncateweight
            lower_atol=lower_atol,
        )

    return factory, circuit


# --------------------------------------------------------------------------- #


def save_result(output_path, record) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=["majorana", "pauli"], required=True)
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--observable", choices=["extensive", "local"],
                        default="extensive")
    parser.add_argument("--layers", type=int, default=1,
                        help="number of layers to apply (default 1)")
    parser.add_argument("--lower-atol", type=float, default=1e-8)
    parser.add_argument("--rounds", type=int, default=3,
                        help="fresh rebuild+apply repetitions; min time is kept")
    parser.add_argument("--out", "-o", required=True,
                        help="JSONL results file (appended to)")
    args = parser.parse_args()

    builder = build_majorana if args.basis == "majorana" else build_pauli
    factory, circuit = builder(
        args.num_qubits, args.cutoff, args.lower_atol, args.observable
    )

    best = float("inf")
    num_terms = 0
    memory_bytes = 0
    expectation = float("nan")
    for _ in range(max(1, args.rounds)):
        sim = factory()  # fresh operator each round (propagate mutates in place)
        t0 = perf_counter()
        for _ in range(args.layers):
            sim.propagate(circuit)
        dt = perf_counter() - t0
        best = min(best, dt)
        num_terms = sim.size()
        memory_bytes = sim._simulator.operator_memory_bytes()
        expectation = float(sim.expectation_value())
        del sim

    record = {
        "engine": "monoprop",
        "basis": args.basis,
        "num_qubits": args.num_qubits,
        "cutoff": args.cutoff,
        "observable": args.observable,
        "layers": args.layers,
        "lower_atol": args.lower_atol,
        "num_threads": os.environ.get("monoprop_NUM_THREADS", "not set"),
        "num_terms": int(num_terms),
        "memory_bytes": int(memory_bytes),
        "bytes_per_term": (memory_bytes / num_terms) if num_terms else 0.0,
        "seconds": best,
        "expectation": expectation,
    }
    print(
        f"[monoprop/{args.basis}] N={args.num_qubits} cutoff={args.cutoff} "
        f"terms={num_terms} mem={memory_bytes / 1024**2:.2f}MB "
        f"b/term={record['bytes_per_term']:.1f} t={best:.4f}s exp={expectation:.6g}"
    )
    save_result(args.out, record)


if __name__ == "__main__":
    main()
