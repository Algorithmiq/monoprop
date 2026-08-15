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

"""Model builders shared across the monoprop benchmark suite.

Import-only (no pytest) so the builders are reusable from scripts or notebooks:
:func:`make_random_problem` (configurable random benchmarks),
:func:`build_hubbard_problem` (120-qubit Hubbard), and
:func:`build_kicked_ising_problem` (127-qubit Pauli-basis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from monoprop import (
    Circuit,
    ExpGate,
    MajoranaPropagator,
    Pauli,
    PauliPropagator,
)
from monoprop.circuit import expand_monomials
from monoprop.fermi import (
    FermiOperator,
    MajoranaOperator,
)
from monoprop.pauli import PauliOperator

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")

Built = tuple[MajoranaPropagator, Circuit]


def barriered(fn: Callable[..., _T], comm: Any | None) -> Callable[..., _T]:
    """Wrap ``fn`` so the measured time ends only when every rank has finished.

    Only the *exit* barrier belongs in the timed region. It is what makes each rank's
    measurement the makespan -- the slowest rank's finish. Pair this with
    :func:`barrier_setup`, which performs the entry barrier untimed.

    The entry barrier used to live here, and it was charging every rank's measurement
    with the skew in the *preceding* setup rather than with any work under test.
    Measured directly at 8 ranks, that wait was 0.000/0.098/0.059/0.257/0.000/0.136 s
    across six rounds of ``build_graph`` -- against a ~0.5 s operation. It accounted for
    roughly 1.44x of the 2.86x spread pytest-benchmark reported on ``build_graph``, and
    is why the reported remainder (wall minus instrumented layer time) correlated with
    wall at +0.987 while the actual layer time correlated at +0.176. The engine was
    steady; the instrument was not. (The residual, ~2.86x/1.44x, is NOT explained by
    this and remains open -- suspects are pytest-benchmark's own per-round machinery,
    the ``op_memory`` window, and the ``fresh`` cell running two ops in one process.)

    A serial run returns ``fn`` unchanged, so there is no overhead.

    Args:
        fn: The callable to wrap.
        comm: An MPI communicator, or ``None`` for a serial run.
    """
    if comm is None or comm.Get_size() == 1:
        return fn

    def wrapped(*args: object, **kwargs: object) -> _T:
        result = fn(*args, **kwargs)
        comm.Barrier()
        return result

    return wrapped


def barrier_setup(
    comm: Any | None, setup: Callable[[], _T] | None = None
) -> Callable[[], _T] | None:
    """Build a ``pedantic(setup=...)`` callable that leaves the ranks synchronised.

    ``pytest-benchmark``'s ``pedantic`` runs ``setup`` before each round and does not
    time it, which is exactly where the entry barrier has to go: the ranks still start
    the timed call together, but nobody's measurement absorbs the wait for the slowest
    rank to finish setting up.

    Args:
        comm: An MPI communicator, or ``None`` for a serial run.
        setup: The round's existing setup, or ``None`` if it has none.

    Returns:
        A setup callable, or ``None`` when there is nothing to do (serial run with no
        setup), which ``pedantic`` accepts.
    """
    if comm is None or comm.Get_size() == 1:
        return setup

    def wrapped_setup() -> _T:
        result = setup() if setup is not None else None
        comm.Barrier()
        return result  # type: ignore[return-value]

    return wrapped_setup


@dataclass(frozen=True, slots=True)
class RandomProblem:
    """A randomly generated observable and circuit ready for a propagator.

    Attributes:
        observable: Random Hermitian Majorana observable.
        circuit: Monomial circuit of random fixed-length Majorana generators.
        cutoff: Truncation cutoff to use when constructing the propagator.
    """

    observable: MajoranaOperator
    circuit: Circuit
    cutoff: int

    @property
    def parameter_mapping(self) -> np.ndarray:
        """Return the circuit's per-monomial parameter-index mapping."""
        _, _, mapping, _ = expand_monomials(
            self.circuit.gates, self.circuit.resolved_mapping
        )
        return np.asarray(mapping, dtype=int)

    @property
    def gen_coeffs(self) -> np.ndarray:
        """Return the circuit's per-monomial generator coefficients."""
        _, coeffs, _, _ = expand_monomials(
            self.circuit.gates, self.circuit.resolved_mapping
        )
        return np.asarray(coeffs, dtype=float)

    @property
    def parameters(self) -> np.ndarray:
        """Return the circuit's parameter values."""
        return np.asarray(self.circuit.parameters, dtype=float)


# Above this ratio of distinct monomials to requested terms, collisions are rare enough
# that a flat margin beats computing the coupon-collector estimate in floating point,
# where the huge binomials involved are not representable.
_SPARSE_RATIO = 100

# Rows are materialized in blocks so the transient list of freshly-boxed ints stays
# bounded; a single ``tolist()`` over 24M rows is itself a multi-GiB spike.
_TUPLE_CHUNK = 1 << 20


def _unique_rows(rows: np.ndarray, num_majorana_indices: int) -> np.ndarray:
    """Return ``rows`` with duplicate monomials dropped, draw order preserved.

    Each row is already sorted ascending, so a row is a monomial and two equal rows are
    the same monomial. When the indices fit, the row is packed into one ``int64`` and
    uniqueness becomes a 1-D sort instead of a lexsort over a void view of the whole
    array -- an order of magnitude cheaper at 24M rows, which is the size this exists for.
    """
    bits = max(int(num_majorana_indices - 1).bit_length(), 1)
    if bits * rows.shape[1] <= 63:
        keys = np.zeros(len(rows), dtype=np.int64)
        for col in range(rows.shape[1]):
            keys = (keys << bits) | rows[:, col].astype(np.int64)
        _, first = np.unique(keys, return_index=True)
    else:
        _, first = np.unique(rows, axis=0, return_index=True)
    # np.unique returns first occurrences in *sorted* order; re-sorting the indices puts
    # the survivors back in draw order, so truncating to num_terms stays seed-stable.
    return rows[np.sort(first)]


def _draw_size(want: int, distinct: int) -> int:
    """Return how many rows to draw so that ~``want`` of them are distinct monomials.

    Drawing ``m`` rows uniformly from ``distinct`` possibilities leaves
    ``distinct * (1 - (1 - 1/distinct)**m)`` expected distinct ones; inverting that gives
    the draw size. The margin is what keeps the top-up loop to a single pass.
    """
    if distinct > want * _SPARSE_RATIO:
        return want + max(16, want // 50)
    # Clamped below 1: asking for the whole space is coupon collecting, whose exact draw
    # count is unbounded in expectation, so aim just short of it and let the caller top up.
    ratio = min(want / distinct, 1 - 1 / (2 * distinct))
    return int(-distinct * math.log1p(-ratio)) + 16


def _draw_monomials(
    rng: np.random.Generator, size: int, length: int, num_majorana_indices: int
) -> np.ndarray:
    """Return ``size`` sorted rows of ``length`` pairwise-distinct indices."""
    rows = np.sort(rng.integers(0, num_majorana_indices, size=(size, length)), axis=1)
    if length == 1:
        return rows
    # Redraw rows that repeat an index, rather than permuting per row: with
    # length << num_majorana_indices a repeat is a ~1% event, so a couple of redraws beat
    # materializing a (size, num_majorana_indices) key matrix -- ~96 GiB at 24M rows.
    repeated = (rows[:, :-1] == rows[:, 1:]).any(axis=1)
    while repeated.any():
        rows[repeated] = np.sort(
            rng.integers(0, num_majorana_indices, size=(int(repeated.sum()), length)),
            axis=1,
        )
        repeated = (rows[:, :-1] == rows[:, 1:]).any(axis=1)
    return rows


def _random_terms(
    rng: np.random.Generator,
    num_terms: int,
    length: int,
    num_majorana_indices: int,
) -> list[tuple[int, ...]]:
    """Return ``num_terms`` distinct sorted Majorana monomials of ``length`` indices.

    Drawn as a batch rather than one ``rng.choice`` per term. The per-term form cost
    ~10 us/term, i.e. ~4 minutes and several GiB of transient heap at the 24M-term size
    the 100M-term A/B needs, on every rank of every run. This changes the RNG stream, so
    term sets are not comparable with runs predating it -- an A/B stays valid because both
    arms import this one module.
    """
    if length > num_majorana_indices:
        msg = (
            f"Cannot draw {length} distinct Majorana indices from "
            f"{num_majorana_indices} available indices."
        )
        raise ValueError(msg)

    distinct = math.comb(num_majorana_indices, length)
    if num_terms > distinct:
        msg = (
            f"Only {distinct} distinct monomials of length {length} exist over "
            f"{num_majorana_indices} indices; {num_terms} requested."
        )
        raise ValueError(msg)

    # Top up until enough distinct monomials survive. One pass is nearly always enough --
    # _draw_size already inflates the request by the expected duplicate rate -- but the
    # loop is what makes the function correct at every density rather than only the sparse
    # one the 24M-term case sits at.
    rows: np.ndarray | None = None
    while rows is None or len(rows) < num_terms:
        want = num_terms - (0 if rows is None else len(rows))
        batch = _draw_monomials(
            rng, _draw_size(want, distinct), length, num_majorana_indices
        )
        rows = batch if rows is None else np.concatenate((rows, batch))
        rows = _unique_rows(rows, num_majorana_indices)
    rows = rows[:num_terms]

    # The bypass in _random_majorana_operator is only sound while every row is strictly
    # ascending. Vectorized, so this costs nothing next to the draw it guards.
    if length > 1 and not bool((rows[:, :-1] < rows[:, 1:]).all()):
        msg = "internal: drawn monomials are not strictly ascending"
        raise AssertionError(msg)

    # Intern the indices. ``tolist()`` boxes every element into its own int object, and at
    # 24M x 4 that is ~3 GiB of int objects that live as long as the problem does; mapping
    # through a lookup list makes every tuple share one object per index value instead.
    lookup = [*range(num_majorana_indices)]
    terms: list[tuple[int, ...]] = []
    for start in range(0, len(rows), _TUPLE_CHUNK):
        terms.extend(
            tuple(map(lookup.__getitem__, row))
            for row in rows[start : start + _TUPLE_CHUNK].tolist()
        )
    return terms


def _random_majorana_operator(
    terms: list[tuple[int, ...]],
    coefficients: list[float],
    num_modes: int,
) -> MajoranaOperator:
    """Build a :class:`MajoranaOperator` directly from already-canonical terms.

    ``MajoranaOperator.__init__`` re-derives each term's sign through
    ``Majorana.from_unsorted`` and accumulates into a second dict. That is required in
    general, but :func:`_random_terms` already returns sorted, distinct-index, pairwise
    distinct monomials, so the canonicalization is a no-op that costs a full duplicate of
    a 24M-entry mapping plus one ``Majorana`` object per term. Bypassing it is what keeps
    the 100M-term working point inside a node's memory.
    """
    operator = MajoranaOperator.__new__(MajoranaOperator)
    operator.num_modes = num_modes
    operator.terms = dict(zip(terms, coefficients, strict=True))
    return operator


def make_random_problem(
    *,
    gen_length: int = 4,
    obs_terms: int = 10000,
    num_generators: int = 100,
    num_modes: int = 128,
    cutoff: int = 6,
    seed: int | np.random.Generator | None = None,
) -> RandomProblem:
    """Build a random observable and generator circuit for benchmarking.

    Args:
        gen_length: Number of Majorana operators in each random generator.
        obs_terms: Number of terms in the random observable.
        num_generators: Number of random generators in the circuit.
        num_modes: Number of fermionic modes (Majorana indices = ``2 * num_modes``).
        cutoff: Truncation cutoff carried on the returned problem.
        seed: Seed or ``Generator`` for the RNG; ``None`` (the default) draws
            fresh entropy. Fix it for a reproducible problem.

    Returns:
        A :class:`RandomProblem` bundling the observable, circuit, and cutoff.
    """
    rng = np.random.default_rng(seed)
    num_majorana_indices = 2 * num_modes

    obs_majoranas = _random_terms(rng, obs_terms, gen_length, num_majorana_indices)
    obs_coeffs = rng.standard_normal(obs_terms).tolist()  # Hermitian -> real
    observable = _random_majorana_operator(obs_majoranas, obs_coeffs, num_modes)

    gen_majoranas = _random_terms(rng, num_generators, gen_length, num_majorana_indices)
    gen_coeffs = rng.standard_normal(num_generators).tolist()
    # One free parameter per generator; small angles keep the evolution stable.
    parameters = (0.1 * rng.standard_normal(num_generators)).tolist()
    param_inds = list(range(num_generators))

    circuit = Circuit.from_dense_arrays(
        majoranas=gen_majoranas,
        gen_coeffs=gen_coeffs,
        param_inds=param_inds,
        system_size=num_modes,
        parameters=parameters,
        initial_state=[],
    )
    return RandomProblem(observable=observable, circuit=circuit, cutoff=cutoff)


def build_random_propagator(
    problem: RandomProblem,
    *,
    comm: Any | None = None,
    lower_atol: float | None = None,
    schrodinger: bool = False,
) -> Built:
    """Construct a propagator + gates for a random problem (optionally MPI-aware).

    Args:
        problem: The random observable/circuit problem.
        comm: Optional MPI communicator (``None`` for a serial run).
        lower_atol: Optional coefficient-truncation tolerance.
        schrodinger: If ``True``, build in the Schrödinger picture with
            ``schrodinger_cutoff = problem.cutoff + 2``; otherwise Heisenberg.

    Returns:
        ``(propagator, circuit)`` ready for build_graph / propagate.
    """
    propagator = MajoranaPropagator(
        problem.observable,
        problem.circuit.initial_state,
        cutoff=problem.cutoff,
        schrodinger_cutoff=problem.cutoff + 2 if schrodinger else None,
        lower_atol=lower_atol,
        comm=comm,
    )
    return propagator, problem.circuit


@dataclass(frozen=True, slots=True)
class HubbardConfig:
    """Configuration for the static Hubbard benchmark (sandbox default input)."""

    num_sites: int = 60
    hopping: float = 1.0
    interaction: float = -2.0
    chemical_potential: float = 0.0
    trotter_dt: float = 0.2
    trotter_steps: int = 29
    observable_site: int = 46
    observable_spin: str = "up"
    neel_start_spin: str = "down"
    cutoff: int = 6
    lower_atol: float = 1e-4

    @property
    def num_qubits(self) -> int:
        """Return the number of spin orbitals (two per site)."""
        return 2 * self.num_sites


def _mode(site: int, spin: str) -> int:
    """Return the interleaved mode index for a site and spin."""
    return 2 * site if spin == "up" else 2 * site + 1


def _hubbard_fermion_terms(config: HubbardConfig) -> list[FermiOperator]:
    """Return the ordered first-order Trotter terms for the 1D Hubbard model."""
    terms: list[FermiOperator] = []

    for site in range(config.num_sites - 1):
        for spin in ("up", "down"):
            left, right = _mode(site, spin), _mode(site + 1, spin)
            op_terms = [((left, "+"), (right, "-")), ((right, "+"), (left, "-"))]
            terms.append(
                FermiOperator(
                    terms=op_terms,
                    coefficients=[-config.hopping, -config.hopping],
                    num_modes=config.num_qubits,
                )
            )

    for site in range(config.num_sites):
        up, down = _mode(site, "up"), _mode(site, "down")
        terms.append(
            FermiOperator(
                terms=[((up, "+"), (up, "-"), (down, "+"), (down, "-"))],
                coefficients=[config.interaction],
                num_modes=config.num_qubits,
            )
        )

    for site in range(config.num_sites):
        for spin in ("up", "down"):
            m = _mode(site, spin)
            terms.append(
                FermiOperator(
                    terms=[((m, "+"), (m, "-"))],
                    coefficients=[-config.chemical_potential],
                    num_modes=config.num_qubits,
                )
            )

    return terms


def _neel_occupied_modes(num_sites: int, start_spin: str) -> list[int]:
    """Return the occupied modes for the half-filled Néel product state."""
    other = "down" if start_spin == "up" else "up"
    return [
        _mode(site, start_spin if site % 2 == 0 else other) for site in range(num_sites)
    ]


def build_hubbard_problem(
    config: HubbardConfig | None = None,
    *,
    comm: Any | None = None,
) -> Built:
    """Build the static Hubbard propagator (120 qubits, sandbox default input).

    Args:
        config: Hubbard configuration; defaults to the sandbox default input.
        comm: Optional MPI communicator (``None`` for a serial run).

    Returns:
        ``(propagator, circuit)`` for one Trotter step, re-applied by the driver.
    """
    config = config or HubbardConfig()
    fermi_gates = [ExpGate(term) for term in _hubbard_fermion_terms(config)]
    parameters = [config.trotter_dt] * len(fermi_gates)
    occupied = _neel_occupied_modes(config.num_sites, config.neel_start_spin)
    circuit = Circuit(
        gates=fermi_gates,
        parameters=parameters,
        initial_state=occupied,
        system_size=config.num_qubits,
    )

    observable = FermiOperator(
        terms=[
            (
                (_mode(config.observable_site, config.observable_spin), "+"),
                (_mode(config.observable_site, config.observable_spin), "-"),
            )
        ],
        coefficients=[1.0],
        num_modes=config.num_qubits,
    )

    propagator = MajoranaPropagator(
        observable,
        circuit.initial_state,
        cutoff=config.cutoff,
        cutoff_type="length",
        lower_atol=config.lower_atol,
        comm=comm,
    )
    return propagator, circuit


# IBM Eagle 127-qubit heavy-hex coupling map (i < j for all pairs).
HEAVY_HEX_TOPOLOGY: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (4, 15),
    (8, 16),
    (12, 17),
    (14, 18),
    (15, 22),
    (16, 26),
    (17, 30),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (20, 33),
    (24, 34),
    (28, 35),
    (32, 36),
    (33, 39),
    (34, 43),
    (35, 47),
    (36, 51),
    (37, 38),
    (38, 39),
    (39, 40),
    (40, 41),
    (41, 42),
    (42, 43),
    (43, 44),
    (44, 45),
    (45, 46),
    (46, 47),
    (47, 48),
    (48, 49),
    (49, 50),
    (50, 51),
    (37, 52),
    (41, 53),
    (45, 54),
    (49, 55),
    (52, 56),
    (53, 60),
    (54, 64),
    (55, 68),
    (56, 57),
    (57, 58),
    (58, 59),
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
    (65, 66),
    (66, 67),
    (67, 68),
    (68, 69),
    (69, 70),
    (58, 71),
    (62, 72),
    (66, 73),
    (70, 74),
    (71, 77),
    (72, 81),
    (73, 85),
    (74, 89),
    (75, 76),
    (76, 77),
    (77, 78),
    (78, 79),
    (79, 80),
    (80, 81),
    (81, 82),
    (82, 83),
    (83, 84),
    (84, 85),
    (85, 86),
    (86, 87),
    (87, 88),
    (88, 89),
    (75, 90),
    (79, 91),
    (83, 92),
    (87, 93),
    (90, 94),
    (91, 98),
    (92, 102),
    (93, 106),
    (94, 95),
    (95, 96),
    (96, 97),
    (97, 98),
    (98, 99),
    (99, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    (103, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    (107, 108),
    (96, 109),
    (100, 110),
    (104, 111),
    (108, 112),
    (109, 114),
    (110, 118),
    (111, 122),
    (112, 126),
    (113, 114),
    (114, 115),
    (115, 116),
    (116, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    (120, 121),
    (121, 122),
    (122, 123),
    (123, 124),
    (124, 125),
    (125, 126),
]


@dataclass(frozen=True, slots=True)
class KickedIsingConfig:
    """Configuration for the static Pauli-basis (kicked-Ising) benchmark."""

    num_qubits: int = 127
    num_layers: int = 20
    observable_qubit: int = 62
    theta: float = np.pi / 4
    coupling: float = np.pi / 4
    cutoff: int = 8
    lower_atol: float = 1e-4


def _xlayer(num_qubits: int, angle: float) -> list[tuple[ExpGate, float]]:
    """Single-qubit X rotations exp(-i·angle·X) on all qubits, with their angles."""
    return [
        (ExpGate(PauliOperator({Pauli("X", (i,)): 1.0}, num_qubits=num_qubits)), -angle)
        for i in range(num_qubits)
    ]


def _zzlayer(
    angle: float, topology: list[tuple[int, int]], num_qubits: int
) -> list[tuple[ExpGate, float]]:
    """Two-qubit ZZ interactions exp(-i·angle·ZZ) on every edge, with their angles."""
    return [
        (
            ExpGate(PauliOperator({Pauli("ZZ", (i, j)): 1.0}, num_qubits=num_qubits)),
            -angle,
        )
        for i, j in topology
    ]


def build_kicked_ising_problem(
    config: KickedIsingConfig | None = None,
    *,
    comm: Any | None = None,
) -> Built:
    """Build the static kicked-Ising propagator (127 qubits, Pauli basis).

    Args:
        config: Kicked-Ising configuration; defaults to the tutorial settings.
        comm: Optional MPI communicator (``None`` for a serial run).

    Returns:
        ``(propagator, circuit)`` ready for in-place propagation.
    """
    config = config or KickedIsingConfig()
    gate_angles: list[tuple[ExpGate, float]] = []
    for _ in range(config.num_layers):
        gate_angles.extend(_xlayer(config.num_qubits, config.theta / 2))
        gate_angles.extend(
            _zzlayer(config.coupling, HEAVY_HEX_TOPOLOGY, config.num_qubits)
        )
    circuit = Circuit(
        gates=tuple(gate for gate, _ in gate_angles),
        parameters=tuple(angle for _, angle in gate_angles),
        initial_state=[],
        system_size=config.num_qubits,
    )

    obs_str = (
        "I" * config.observable_qubit
        + "Z"
        + "I" * (config.num_qubits - config.observable_qubit - 1)
    )
    observable = PauliOperator({obs_str: 1.0}, num_qubits=config.num_qubits)

    propagator = PauliPropagator(
        observable,
        circuit.initial_state,
        cutoff=config.cutoff,
        lower_atol=config.lower_atol,
        upper_atol=None,
        comm=comm,
    )
    return propagator, circuit


# Fixed-model registry: model -> (config class, builder, steps-per-run). Steps
# derive from the config: Hubbard re-applies its one-step circuit ``trotter_steps``
# times; the Pauli circuit already holds all layers, so one propagate suffices.
# The benchmark suite and the conftest CLI options are both built from this.
MODELS: dict[str, tuple[type, Callable[..., Built], Callable]] = {
    "hubbard": (
        HubbardConfig,
        build_hubbard_problem,
        lambda config: config.trotter_steps,
    ),
    "pauli": (KickedIsingConfig, build_kicked_ising_problem, lambda _config: 1),
}
