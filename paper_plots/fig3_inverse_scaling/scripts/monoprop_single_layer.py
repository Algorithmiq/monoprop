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
import math
import os
import socket
from itertools import pairwise
from pathlib import Path
from time import perf_counter, process_time

from monoprop import Circuit, ExpGate, MajoranaPropagator, Pauli, PauliPropagator
from monoprop import __version__ as monoprop_version
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


def _checkerboard_classes(lx: int, ly: int) -> list[list[tuple[int, int]]]:
    """0-indexed open-boundary bonds of an ``lx`` x ``ly`` lattice, site ``y * lx + x``.

    The bonds come in four classes, each a matching (no two bonds share a site): horizontal
    bonds from even ``x``, from odd ``x``, then vertical bonds from even ``y``, from odd ``y``;
    row-major within a class. Gates within a class commute, so a layer moves a mode at most
    two sites along each axis and the operator's light cone grows by a fixed radius per
    layer. Ordering the bonds row by row instead lets one layer's sequence of
    non-commuting hops carry a mode across the whole lattice, and the term count then grows
    with a power of the lattice size rather than linearly in it. The MajoranaPropagation.jl
    driver builds the same order.
    """
    horizontal = [
        [(y * lx + x, y * lx + x + 1) for y in range(ly) for x in range(p, lx - 1, 2)]
        for p in (0, 1)
    ]
    vertical = [
        [(y * lx + x, (y + 1) * lx + x) for y in range(p, ly - 1, 2) for x in range(lx)]
        for p in (0, 1)
    ]
    return [*horizontal, *vertical]


def _checkerboard_topology(lx: int, ly: int) -> list[tuple[int, int]]:
    """The bonds of :func:`_checkerboard_classes`, class after class."""
    return [bond for group in _checkerboard_classes(lx, ly) for bond in group]


def _hubbard_terms(
    num_sites: int,
    hopping: float,
    interaction: float,
    num_modes: int,
    topology: list[tuple[int, int]] | None = None,
) -> list[FermiOperator]:
    """First-order Trotter terms of the Hubbard model (mu = 0) on ``topology``.

    The default topology is the 1D bricklayer chain. Every term carries the full
    ``num_modes`` width, not just the width of its own support: a gate's generator must
    match the circuit's ``system_size``.
    """
    terms: list[FermiOperator] = []
    topo = _bricklayer_topology(num_sites) if topology is None else topology
    for spin in ("up", "down"):
        for left_site, right_site in topo:
            left, right = _mode(left_site, spin), _mode(right_site, spin)
            terms.append(
                FermiOperator(
                    terms=[((left, "+"), (right, "-")), ((right, "+"), (left, "-"))],
                    coefficients=[-hopping, -hopping],
                    num_modes=num_modes,
                )
            )
    for site in range(num_sites):
        up, down = _mode(site, "up"), _mode(site, "down")
        terms.append(
            FermiOperator(
                terms=[((up, "+"), (up, "-"), (down, "+"), (down, "-"))],
                coefficients=[interaction],
                num_modes=num_modes,
            )
        )
    return terms


def _neel_occupied(num_sites: int, start_spin: str = "up") -> list[int]:
    other = "down" if start_spin == "up" else "up"
    return [_mode(s, start_spin if s % 2 == 0 else other) for s in range(num_sites)]


def build_majorana(
    num_qubits: int, cutoff: int, lower_atol: float, observable: str, comm=None
):
    """Return (propagator_factory, circuit) for a one-layer Hubbard problem.

    ``num_qubits`` must be even; ``num_sites = num_qubits // 2``. ``observable``
    is ``"extensive"`` (total spin-up number, sum over all sites -> O(N) terms
    after one layer, the scaling-relevant choice) or ``"local"`` (single
    mid-site number operator).

    ``comm`` is forwarded to the propagator. In an MPI-enabled build ``None``
    means ``MPI_COMM_WORLD``, not "serial", so a caller wanting one independent
    replica per rank must pass ``MPI.COMM_SELF`` explicitly. The default is left
    at ``None`` so the serial, non-MPI build is unaffected.
    """
    if num_qubits % 2:
        raise ValueError("majorana num_qubits must be even (2 per site)")
    num_sites = num_qubits // 2
    hopping, interaction, dt = 1.0, 1.5, 0.07
    gates = [
        ExpGate(t) for t in _hubbard_terms(num_sites, hopping, interaction, num_qubits)
    ]
    circuit = Circuit(
        gates=gates,
        system_size=num_qubits,
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
            comm=comm,
        )

    return factory, circuit


def build_hubbard2d(
    lx: int, ly: int, cutoff: int, comm=None, cutoff_type: str = "support"
):
    """Return (propagator_factory, circuit, steps) for one Trotter step of the 2D Hubbard model.

    An open ``lx`` x ``ly`` lattice of spinful sites, ``2 * lx * ly`` modes, ``t = 1``,
    ``U = 1.5``, ``dt = 0.07``: hop-up on every bond, then hop-down on every bond, then
    ``n_up n_down`` on every site, with the bonds in the order of
    :func:`_checkerboard_topology`. The state is the Neel checkerboard (spin up where
    ``x + y`` is even) and the observable is the staggered magnetisation
    ``sum_s (-1)^(x+y) (n_up - n_down)``, which is extensive and not conserved.

    Truncation is the support cutoff alone, with no coefficient tolerance
    (``lower_atol = 0``): a monomial survives if it touches at most
    ``cutoff`` modes or is fully paired. MajoranaPropagation.jl has no native truncation
    with that rule, so its driver passes the same rule as a ``customtruncfunc``.

    ``ExpGate`` applies ``exp(+i dt H)``; MajoranaPropagation.jl's ``FermionicRotation``
    applies ``exp(-i theta O)``, so its driver takes ``theta = +t dt`` for hopping and
    ``theta = -U dt`` for the interaction, the identical circuit.

    ``steps`` splits the layer into its nine commuting gate groups (hop-up per bond class,
    hop-down per bond class, the interaction), as circuits in the order they propagate: the
    last group first, as the Heisenberg picture applies a circuit. Propagating them one by
    one is exactly propagating ``circuit``, and timing them one by one resolves the term
    count's thousandfold rise within the second layer.
    """
    num_sites = lx * ly
    num_modes = 2 * num_sites
    hopping, interaction, dt = 1.0, 1.5, 0.07
    topology = _checkerboard_topology(lx, ly)
    gates = [
        ExpGate(t)
        for t in _hubbard_terms(num_sites, hopping, interaction, num_modes, topology)
    ]

    def parity(site: int) -> int:
        return (site % lx + site // lx) % 2

    occupied = [_mode(s, "up" if parity(s) == 0 else "down") for s in range(num_sites)]
    circuit = Circuit(
        gates=gates,
        system_size=num_modes,
        parameters=[dt] * len(gates),
        initial_state=occupied,
    )
    sizes = [len(c) for c in _checkerboard_classes(lx, ly)] * 2 + [num_sites]
    bounds = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    steps = [
        Circuit(
            gates=gates[lo:hi],
            system_size=num_modes,
            parameters=[dt] * (hi - lo),
            initial_state=occupied,
        )
        for lo, hi in reversed(list(pairwise(bounds)))
        if hi > lo
    ]
    terms, coeffs = [], []
    for s in range(num_sites):
        sign = 1.0 if parity(s) == 0 else -1.0
        terms.append(((_mode(s, "up"), "+"), (_mode(s, "up"), "-")))
        coeffs.append(sign)
        terms.append(((_mode(s, "down"), "+"), (_mode(s, "down"), "-")))
        coeffs.append(-sign)
    observable_op = FermiOperator(terms=terms, coefficients=coeffs, num_modes=num_modes)

    def factory():
        return MajoranaPropagator(
            observable_op,
            circuit.initial_state,
            cutoff=cutoff,
            cutoff_type=cutoff_type,
            lower_atol=0.0,
            comm=comm,
        )

    return factory, circuit, steps


# --------------------------------------------------------------------------- #
# Pauli / kicked-Ising chain (one X-layer + one ZZ-layer == one layer)
# --------------------------------------------------------------------------- #


def build_pauli(
    num_qubits: int,
    cutoff: int,
    lower_atol: float,
    observable: str,
    comm=None,
    active_window: int | None = None,
):
    """Return (propagator_factory, circuit) for a one-layer kicked-Ising chain.

    ``observable`` is ``"extensive"`` (sum_i Z_i -> O(N) terms after one layer)
    or ``"local"`` (single mid-chain Z).

    ``active_window`` confines the whole model -- every gate and every observable term --
    to qubits ``0..active_window-1``, while the register stays ``num_qubits`` wide. The
    remaining qubits are idle spectators: nothing acts on them and nothing in the operator
    ever touches them. That holds the term count, the gate count, the term supports and the
    expectation value *exactly* fixed as ``num_qubits`` grows, so a sweep over
    ``num_qubits`` at fixed ``active_window`` measures per-term cost in the width of the
    mode space and nothing else.

    This is the point of the knob. In the full-width model both the term count and the gate
    count grow with ``num_qubits``, so a runtime exponent there mixes three different
    N-dependencies and cannot isolate the per-term one. It also gives a free correctness
    check: the expectation value is an invariant of the sweep, so a drift in it means the
    padding is not actually idle.

    See :func:`build_majorana` for the ``comm`` semantics -- in particular that
    ``None`` means ``MPI_COMM_WORLD`` in an MPI build.
    """
    active = num_qubits if active_window is None else active_window
    if not 1 <= active <= num_qubits:
        msg = (
            f"active_window must be in 1..num_qubits, got {active} with N={num_qubits}"
        )
        raise ValueError(msg)
    # Angle convention, stated once because the two engines differ and getting it wrong is
    # silent: PauliPropagation.jl's PauliRotation(P) at angle t is exp(-i t/2 P), while
    # monoprop's ExpGate(P) with parameter a is exp(+i a P). Matching Julia therefore means
    # a = -t/2 in BOTH layers. (The Rzz layer used to pass -coupling, i.e. twice Julia's
    # angle, so the two engines propagated different circuits.)
    theta = math.pi / 4
    coupling = math.pi / 4
    edges = [(i, i + 1) for i in range(active - 1)]

    gate_angles: list[tuple[ExpGate, float]] = [
        (
            ExpGate(PauliOperator({Pauli("X", (i,)): 1.0}, num_qubits=num_qubits)),
            -theta / 2,
        )
        for i in range(active)
    ]
    gate_angles += [
        (
            ExpGate(PauliOperator({Pauli("ZZ", (i, j)): 1.0}, num_qubits=num_qubits)),
            -coupling / 2,
        )
        for i, j in edges
    ]
    circuit = Circuit(
        gates=tuple(g for g, _ in gate_angles),
        system_size=num_qubits,
        parameters=tuple(a for _, a in gate_angles),
        initial_state=[],
    )

    def _z(i: int) -> str:
        return "I" * i + "Z" + "I" * (num_qubits - i - 1)

    if observable == "extensive":
        observable_op = PauliOperator(
            {_z(i): 1.0 for i in range(active)}, num_qubits=num_qubits
        )
    else:
        observable_op = PauliOperator({_z(active // 2): 1.0}, num_qubits=num_qubits)

    def factory():
        return PauliPropagator(
            observable_op,
            circuit.initial_state,
            cutoff=cutoff,  # support/weight cutoff == Julia truncateweight
            lower_atol=lower_atol,
            comm=comm,
        )

    return factory, circuit


# --------------------------------------------------------------------------- #


# Rounds after the second stop once this much propagation has been timed.
ROUND_BUDGET_S = 60.0


def keep_timing(
    rounds_done: int, timed: float, last: float, time_limit: float | None
) -> bool:
    """Whether to time another round; every Fig. 3 driver applies this same rule.

    The first timed round of a fresh process can be far slower than the next (the first
    PauliPropagation.jl round at N=32 took 1.9x the second), so a point gets a second round
    whenever one fits, and further rounds while under ``ROUND_BUDGET_S``. No round starts that
    would be expected to end past ``time_limit``.
    """
    if time_limit is not None and timed + last > time_limit:
        return False
    return rounds_done < 2 or timed < ROUND_BUDGET_S


def save_result(output_path, record) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _build_model(args, parser):
    """(factory, circuit, steps) for the parsed arguments; ``steps`` is None but for hubbard2d."""
    if args.model == "hubbard2d":
        if args.basis != "majorana" or args.lx is None or args.ly is None:
            parser.error("--model hubbard2d needs --basis majorana, --lx and --ly")
        if args.num_qubits != 2 * args.lx * args.ly:
            parser.error("--num-qubits must equal 2 * lx * ly for --model hubbard2d")
        if args.lower_atol != 0:
            parser.error(
                "--model hubbard2d is truncated by the cutoff alone: --lower-atol 0"
            )
        return build_hubbard2d(
            args.lx, args.ly, args.cutoff, cutoff_type=args.cutoff_type
        )
    if args.basis == "majorana":
        if args.active_window is not None:
            parser.error("--active-window is implemented for the Pauli basis only")
        return (
            *build_majorana(
                args.num_qubits, args.cutoff, args.lower_atol, args.observable
            ),
            None,
        )
    factory, circuit = build_pauli(
        args.num_qubits,
        args.cutoff,
        args.lower_atol,
        args.observable,
        active_window=args.active_window,
    )
    return factory, circuit, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=["majorana", "pauli"], required=True)
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument(
        "--observable", choices=["extensive", "local"], default="extensive"
    )
    parser.add_argument(
        "--model",
        choices=["chain", "hubbard2d"],
        default="chain",
        help="chain: the kicked-Ising (pauli) or 1D Hubbard (majorana) chain; hubbard2d: the "
        "--lx x --ly Hubbard lattice, majorana basis, --num-qubits = 2 * lx * ly modes",
    )
    parser.add_argument(
        "--cutoff-type",
        choices=["support", "length"],
        default="support",
        help="hubbard2d only: bound the modes a monomial touches (support) or its Majorana "
        "count (length); fully paired monomials are kept under either",
    )
    parser.add_argument("--lx", type=int, default=None)
    parser.add_argument("--ly", type=int, default=None)
    parser.add_argument(
        "--layers", type=int, default=1, help="number of layers to apply (default 1)"
    )
    parser.add_argument("--lower-atol", type=float, default=1e-8)
    parser.add_argument(
        "--active-window",
        type=int,
        default=None,
        help="confine the model to qubits 0..M-1 and pad the register to --num-qubits with "
        "idle spectator qubits, holding terms, gates and the expectation value fixed as N "
        "grows (Pauli basis only; default: the full width)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="at most this many fresh rebuild+apply repetitions; min time is kept",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="seconds of timed propagation allowed for the whole point: no round is started "
        "that would be expected to end past it (default: no limit)",
    )
    parser.add_argument(
        "--out", "-o", required=True, help="JSONL results file (appended to)"
    )
    args = parser.parse_args()

    factory, circuit, steps = _build_model(args, parser)

    best = float("inf")
    best_cpu = float("nan")
    num_terms = 0
    memory_bytes = 0
    expectation = float("nan")
    # Per step (one propagate call: a layer, or a gate group of one for hubbard2d): the
    # minimum over rounds of its time, and the terms it leaves. Recorded for per-step analysis;
    # the figure plots the whole-run minimum above.
    steps = [circuit] if steps is None else steps
    step_seconds = [float("inf")] * (args.layers * len(steps))
    step_terms: list[int] | None = None
    initial_terms = 0
    rounds_done, timed = 0, 0.0
    while rounds_done < max(1, args.rounds):
        sim = factory()  # fresh operator each round (propagate mutates in place)
        initial_terms = sim.size()
        terms_now = []
        c0, t0 = process_time(), perf_counter()
        for _ in range(args.layers):
            for step in steps:
                tl = perf_counter()
                sim.propagate(step)
                i = len(terms_now)
                step_seconds[i] = min(step_seconds[i], perf_counter() - tl)
                terms_now.append(sim.size())
        dt, cpu = perf_counter() - t0, process_time() - c0
        if step_terms is not None and terms_now != step_terms:
            raise RuntimeError("per-step term counts differ between rounds")
        step_terms = terms_now
        if dt < best:
            best, best_cpu = dt, cpu
        num_terms = sim.size()
        memory_bytes = sim._simulator.operator_memory_bytes()
        expectation = float(sim.expectation_value())
        del sim
        rounds_done += 1
        timed += dt
        if not keep_timing(rounds_done, timed, dt, args.time_limit):
            break

    record = {
        "engine": "monoprop",
        "basis": args.basis,
        "model": "kicked_ising"
        if args.basis == "pauli"
        else ("hubbard2d" if args.model == "hubbard2d" else "hubbard1d"),
        "lattice": [args.lx, args.ly] if args.model == "hubbard2d" else None,
        "num_qubits": args.num_qubits,
        "cutoff": args.cutoff,
        "cutoff_type": args.cutoff_type if args.model == "hubbard2d" else None,
        "observable": args.observable,
        "layers": args.layers,
        "lower_atol": args.lower_atol,
        "num_threads": os.environ.get("monoprop_NUM_THREADS", "not set"),  # noqa: SIM112
        "num_terms": int(num_terms),
        "memory_bytes": int(memory_bytes),
        "bytes_per_term": (memory_bytes / num_terms) if num_terms else 0.0,
        "seconds": best,
        # The number of timed repetitions the reported minimum was taken over. Recorded
        # because a min-of-1 and a min-of-10 are different measurements, and the shipped
        # data used to carry no way to tell them apart.
        "rounds": rounds_done,
        "max_rounds": max(1, args.rounds),
        "time_limit": args.time_limit,
        "expectation": expectation,
        # Provenance. The shipped Leonardo data carries none of this, so a record from it
        # cannot be audited or told apart from a workstation run; every new record can.
        "active_window": args.active_window,
        "gates": len(circuit.gates) * args.layers,
        "gates_per_layer": len(circuit.gates),
        "initial_terms": int(initial_terms),
        "layer_terms": step_terms[len(steps) - 1 :: len(steps)],
        # A layer's time is the sum of its steps' minima.
        "layer_seconds": [
            sum(step_seconds[i : i + len(steps)])
            for i in range(0, len(step_seconds), len(steps))
        ],
        "step_gates": [len(step.gates) for step in steps] * args.layers,
        "step_terms": step_terms,
        "step_seconds": step_seconds,
        "cpu_seconds": best_cpu,
        "busy_cores": (best_cpu / best) if best else float("nan"),
        "host": socket.gethostname(),
        "monoprop_version": monoprop_version,
        "library_version": monoprop_version,
    }
    print(
        f"[monoprop/{args.basis}] N={args.num_qubits} cutoff={args.cutoff} "
        f"terms={num_terms} mem={memory_bytes / 1024**2:.2f}MB "
        f"b/term={record['bytes_per_term']:.1f} t={best:.4f}s exp={expectation:.6g}"
    )
    save_result(args.out, record)


if __name__ == "__main__":
    main()
