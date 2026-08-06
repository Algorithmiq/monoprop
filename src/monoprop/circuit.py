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

r"""Authoring types for Majorana/qubit circuits.

An [ExpGate][] is the exponential of a generator *operator* -- a
[MajoranaOperator][monoprop.majorana.MajoranaOperator], [PauliOperator][monoprop.pauli.PauliOperator], or
[FermiOperator][monoprop.fermi.FermiOperator] (converted to Majorana form on construction). A bare term
is rejected: only an operator carries the system ``num_modes`` / ``num_qubits``, and its type
fixes the gate's family and normalization. A [Circuit][] is an ordered sequence of such
gates, all of one family, each gate carrying one variational angle. The propagators dispatch on
[Circuit.family][]: [MajoranaPropagator][monoprop.majorana_propagator.MajoranaPropagator] takes a
Majorana/fermionic circuit, [PauliPropagator][monoprop.pauli_propagator.PauliPropagator] a qubit one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .conversion_utils import (
    _pauli_to_local_slots,
)
from .majorana import MajoranaOperator
from .pauli import PauliOperator
from .utils import _validate_system_size

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .fermi import FermiOperator


#: A gate's generator family; a fermionic generator is converted, so there is no ``"fermi"``.
GateFamily = Literal["pauli", "majorana"]
#: The family of a circuit; ``"empty"`` when it has no gates.
CircuitFamily = Literal["pauli", "majorana", "empty"]


def _is_identity_gate(gate: ExpGate) -> bool:
    return all(coeff == 0 for coeff in gate.generator.terms.values())


class ExpGate:
    r"""The exponential of a generator: one variational gate, abstract over the family.

    Applies $e^{+i\theta H}$ for driving angle $\theta$ and Hermitian generator
    $H$. Note the **positive** sign: qiskit's ``PauliEvolutionGate`` and ``r<P>`` rotations
    use $e^{-itH}$, so [monoprop.qiskit_conversion][] negates the generator both ways.
    Every family supplies the **Hermitian** generator -- for a Majorana one that means the
    observable convention: imaginary coefficient for a weight-2 monomial, real for weight-4.

    Attributes:
        generator: The generator operator (a ``FermiOperator`` is stored converted to Majorana).
        index: The variational-angle index, or ``None`` for the identity mapping.
        family: ``"pauli"`` or ``"majorana"``, inferred from the generator type.
    """

    __slots__ = ("_atol", "_structural", "family", "generator", "index")

    def __init__(
        self,
        generator: MajoranaOperator | PauliOperator | FermiOperator,
        index: int | None = None,
        *,
        atol: float = 1e-8,
        _structural: bool = False,
    ) -> None:
        """Wrap a generator operator; its type selects the family and normalization convention.

        Args:
            generator: A [MajoranaOperator][monoprop.majorana.MajoranaOperator],
                [PauliOperator][monoprop.pauli.PauliOperator], or
                [FermiOperator][monoprop.fermi.FermiOperator]. A bare ``Majorana`` / ``Pauli`` term is
                *not* accepted; wrap it, e.g. ``MajoranaOperator({(0, 1): 1j}, num_modes)`` --
                the Hermitian convention makes a weight-2 coefficient imaginary.
            index: The variational-angle index, or ``None`` for the identity mapping (see
                [Circuit][]).
            atol: Generator terms with ``|coeff| <= atol`` are dropped.
            _structural: Internal. Set by ``_structural_gate`` when the generator already
                carries the real structural coefficients ``g`` (the wire/dense format), so
                ``_gate_layers`` passes them through unnormalized.

        Raises:
            TypeError: If ``generator`` is not one of the three operator types.
            ValueError: If the generator's terms do not all pairwise commute (a single
                exponential cannot stand in for non-commuting generators).
        """
        if isinstance(generator, PauliOperator):
            family: GateFamily = "pauli"
        elif isinstance(generator, MajoranaOperator):
            family = "majorana"
        elif hasattr(generator, "get_majorana_operator"):
            # Duck-typed to avoid a circular import of FermiOperator.
            generator = generator.get_majorana_operator()
            family = "majorana"
        else:
            raise TypeError(
                f"ExpGate generator must be a MajoranaOperator, PauliOperator, or FermiOperator "
                f"(an operator object carrying its mode/qubit count), not a bare term; got "
                f"{type(generator).__name__}."
            )

        self.generator = self._truncated_term(generator, atol)

        if not self.generator.all_pairwise_commute():
            raise ValueError(
                "The provided generator must be composed of commuting terms."
            )

        self.index = None if index is None else int(index)
        self.family = family
        self._structural = _structural
        # Kept so a clone (_with_index, used by Circuit.__add__) re-truncates at the SAME
        # tolerance; re-truncating at the default would silently drop terms the author kept.
        self._atol = atol

    def _truncated_term(
        self, generator: PauliOperator | MajoranaOperator, atol: float
    ) -> PauliOperator | MajoranaOperator:
        """Return a copy of ``generator`` with terms of magnitude ``<= atol`` dropped."""
        if isinstance(generator, PauliOperator):
            terms = {p: c for p, c in generator.terms.items() if abs(c) > atol}
            return PauliOperator(terms, generator.num_qubits)

        terms = {m: c for m, c in generator.terms.items() if abs(c) > atol}
        return MajoranaOperator(terms, generator.num_modes)

    @classmethod
    def _structural_gate(
        cls, generator: MajoranaOperator, index: int | None
    ) -> ExpGate:
        """Build a wire/dense-format gate whose coefficients are *already* structural ``g``."""
        return cls(generator, index=index, _structural=True)

    @classmethod
    def _with_index(cls, gate: ExpGate, index: int | None) -> ExpGate:
        """Clone ``gate`` with a new ``index``, preserving its ``atol`` and ``_structural`` flag."""
        return cls(
            gate.generator,
            index=index,
            atol=gate._atol,
            _structural=gate._structural,
        )

    def __eq__(self, other: object) -> bool:
        """Equal when the generator, parameter index, family, and structural flag all match."""
        if not isinstance(other, ExpGate):
            return NotImplemented
        return (
            self.generator == other.generator
            and self.index == other.index
            and self.family == other.family
            and self._structural == other._structural
        )

    __hash__ = None  # type: ignore[assignment]  # value-equal but not hashable (mutable generator)

    def __repr__(self) -> str:
        """Return a string representation such as ``ExpGate(<generator>, index=0)``."""
        return f"{self.__class__.__name__}({self.generator}, index={self.index})"

    @property
    def system_size(self) -> int:
        """Number of modes/qubits the generator acts on.

        Reads ``num_modes`` off a [MajoranaOperator][monoprop.majorana.MajoranaOperator]
        generator, or ``num_qubits`` off a [PauliOperator][monoprop.pauli.PauliOperator]
        generator.

        Returns:
            Number of modes/qubits
        """
        if isinstance(self.generator, PauliOperator):
            return self.generator.num_qubits
        return self.generator.num_modes


class Circuit:
    """A variational circuit: an ordered sequence of exponential gates, angles, and a state.

    All gates must share a family (see [ExpGate][]). Empty ``parameters`` means unbound; a
    bound circuit needs exactly [n_parameters][] values.

    The per-gate ``index`` values give the parameter mapping: with none set, each gate gets its
    own angle in order; otherwise every gate must set a contiguous ``0..n-1`` index, and gates
    sharing an index share an angle.

    Attributes:
        gates: The ordered exponential gates.
        parameters: The angle values, or empty for an unbound circuit.
        initial_state: The reference state (occupied mode / qubit indices).
        system_size: System width (number of fermionic modes / qubits).
        family: The gate family -- ``"pauli"``, ``"majorana"``, or ``"empty"`` -- computed at
            construction; the propagators dispatch on it.
    """

    def __init__(
        self,
        gates: Sequence[ExpGate],
        system_size: int,
        parameters: Sequence[float] = (),
        initial_state: Sequence[int] | None = None,
    ) -> None:
        """Build the circuit, dropping identity gates and validating family/mapping/params.

        Args:
            gates: The ordered exponential gates.
            initial_state: The reference state (occupied mode / qubit indices).
            system_size: Number of modes/qubits for the circuit.
            parameters: The angle values, or empty for an unbound circuit.

        Raises:
            ValueError: On duplicate initial-state indices, a bad parameter mapping, or a
                bound circuit whose parameter count does not match [n_parameters][]; also if
                a gate's operator width differs from ``system_size``.
            TypeError: On a non-[ExpGate][] gate or a mix of qubit and Majorana gate families.
        """
        gates = tuple(gates)
        parameters = tuple(float(v) for v in parameters)

        self._state_given = initial_state is not None
        initial_state = tuple(int(i) for i in initial_state or ())
        system_size = _validate_system_size(system_size, argument_name="system_size")
        self._validate_initial_state(initial_state, system_size)
        self._validate_gate_system_size(gates, system_size)
        # Checked first: the identity-drop below reads gate attributes, so a non-ExpGate must
        # fail here with a clear TypeError rather than an opaque AttributeError.

        default_mapping = all(gate.index is None for gate in gates)
        if default_mapping and any(_is_identity_gate(gate) for gate in gates):
            kept = [i for i, gate in enumerate(gates) if not _is_identity_gate(gate)]
            if parameters and len(parameters) != len(gates):
                raise ValueError(
                    f"parameters has {len(parameters)} values but the circuit has "
                    f"{len(gates)} gates."
                )
            if parameters:
                parameters = tuple(parameters[i] for i in kept)
            gates = tuple(gates[i] for i in kept)

        self.gates = gates
        self.parameters = parameters
        self.initial_state = initial_state
        self.system_size = system_size
        #: The gate family, computed from the (validated) gates; the propagators dispatch on it.
        self.family = self._resolve_family(gates)

        self.resolved_mapping  # validates the per-gate index scheme
        if self.parameters and len(self.parameters) != self.n_parameters:
            raise ValueError(
                f"parameters has {len(self.parameters)} values but the circuit has "
                f"{self.n_parameters} parameters."
            )

    @staticmethod
    def _validate_initial_state(
        initial_state: tuple[int, ...], system_size: int
    ) -> None:
        if len(set(initial_state)) != len(initial_state):
            raise ValueError("Duplicate indices in initial state")
        if any(i < 0 or i >= system_size for i in initial_state):
            raise ValueError(
                f"initial_state entries must be in 0..{system_size - 1}; got {list(initial_state)}."
            )

    @staticmethod
    def _validate_gate_system_size(gates: Sequence[ExpGate], system_size: int) -> None:
        for gate in gates:
            if not isinstance(gate, ExpGate):
                raise TypeError(
                    f"Circuit gates must be ExpGate; got {type(gate).__name__}."
                )
            gate_size = gate.system_size
            if gate_size != system_size:
                raise ValueError(
                    f"Gate generator width {gate_size} does not match circuit system_size={system_size}."
                )

    def __eq__(self, other: object) -> bool:
        """Equal when gates, parameters, and initial state match (``family`` is derived)."""
        if not isinstance(other, Circuit):
            return NotImplemented
        return (
            self.gates == other.gates
            and self.parameters == other.parameters
            and self.initial_state == other.initial_state
            and self.system_size == other.system_size
        )

    __hash__ = None  # type: ignore[assignment]  # value-equal but not hashable (mutable gates)

    def __repr__(self) -> str:
        """Return a string representation listing the gates, parameters, and initial state."""
        return (
            f"{self.__class__.__name__}(gates={self.gates!r}, "
            f"parameters={self.parameters!r}, initial_state={self.initial_state!r}, "
            f"system_size={self.system_size!r})"
        )

    @staticmethod
    def _resolve_family(gates: Sequence[ExpGate]) -> CircuitFamily:
        """Return the family ``"pauli"``/``"majorana"``/``"empty"``, rejecting a mixed circuit."""
        has_pauli = any(gate.family == "pauli" for gate in gates)
        has_majorana = any(gate.family == "majorana" for gate in gates)
        if has_pauli and has_majorana:
            raise TypeError(
                "A circuit cannot mix qubit and Majorana/fermionic gates; build separate "
                "circuits per family."
            )
        if has_pauli:
            return "pauli"
        if has_majorana:
            return "majorana"
        return "empty"

    @property
    def resolved_mapping(self) -> tuple[int, ...]:
        """Per-gate angle index, derived from each gate's ``index`` (see [Circuit][])."""
        indices = [gate.index for gate in self.gates]
        if all(i is None for i in indices):
            return tuple(range(len(self.gates)))
        if any(i is None for i in indices):
            raise ValueError(
                "Either every gate must set `index`, or none of them must (mixing an "
                "explicit index with the default would be ambiguous)."
            )
        mapping = tuple(int(i) for i in indices)  # type: ignore[arg-type]
        validate_parameter_mapping(mapping, len(self.gates), "gates")
        return mapping

    @property
    def n_parameters(self) -> int:
        """Number of distinct variational angles the circuit references."""
        mapping = self.resolved_mapping
        return max(mapping) + 1 if mapping else 0

    def __len__(self) -> int:
        """Number of gates."""
        return len(self.gates)

    def __iter__(self) -> Iterator[ExpGate]:
        """Iterate over the gates in application order."""
        return iter(self.gates)

    def __add__(self, other: Circuit) -> Circuit:
        """Concatenate two circuits of the same family, appending ``other``'s angles.

        ``other``'s angle indices are shifted up by ``self.n_parameters``, so the two halves keep
        independent angles and every gate in the result gets an explicit ``index``. Prefer one
        [MonomialPropagator.build_graph][monoprop.monomial_propagator.MonomialPropagator.build_graph] call over
        incremental multi-call building, whose ordering is picture-dependent.
        """
        if not isinstance(other, Circuit):
            return NotImplemented
        if "empty" not in (self.family, other.family) and self.family != other.family:
            raise TypeError(
                f"Cannot concatenate a {self.family}-family circuit with a "
                f"{other.family}-family one; the gate families differ."
            )
        if (
            self._state_given
            and other._state_given
            and self.initial_state != other.initial_state
        ):
            raise ValueError(
                "Cannot concatenate circuits with different initial states."
            )
        if self.system_size != other.system_size:
            raise ValueError(
                f"Cannot concatenate circuits with different system_size: "
                f"{self.system_size} != {other.system_size}."
            )
        offset = self.n_parameters
        left = tuple(
            ExpGate._with_index(gate, index)
            for gate, index in zip(self.gates, self.resolved_mapping, strict=True)
        )
        right = tuple(
            ExpGate._with_index(gate, index + offset)
            for gate, index in zip(other.gates, other.resolved_mapping, strict=True)
        )
        if self._state_given:
            state = self.initial_state
        elif other._state_given:
            state = other.initial_state
        else:
            state = None
        return Circuit(
            gates=left + right,
            parameters=tuple(self.parameters) + tuple(other.parameters),
            initial_state=state,
            system_size=self.system_size,
        )

    @classmethod
    def from_dense_arrays(
        cls,
        majoranas: Sequence[Sequence[int]],
        gen_coeffs: Sequence[float],
        param_inds: Sequence[int],
        system_size: int,
        parameters: Sequence[float] = (),
        initial_state: Sequence[int] | None = None,
    ) -> Circuit:
        """Build a Majorana circuit from flat, per-monomial dense arrays.

        The native dense/wire format (also the on-disk msgpack-fixture layout): consecutive
        monomials sharing a ``param_ind`` become one Majorana [ExpGate][] with that
        ``param_ind`` as its ``index``, so weight-tying is preserved and the expanded engine
        arrays stay identical.

        Args:
            majoranas: One Majorana-index sequence per monomial.
            gen_coeffs: Generator coefficient per monomial.
            param_inds: Variational-angle index per monomial (contiguous runs group into
                gates).
            system_size: Number of fermionic modes in the system.
            parameters: Optional angle values.
            initial_state: Reference state (occupied mode indices), or ``None`` to defer to the
                propagator's. ``()`` is the explicit vacuum, exactly as in the constructor -- the
                wire format has no third state, so a caller round-tripping it decides which of the
                two an absent field means.
        """
        indices = [int(p) for p in param_inds]
        system_size = _validate_system_size(system_size, argument_name="system_size")
        gates: list[ExpGate] = []
        current_index: int | None = None
        current_majoranas: list[tuple[int, ...]] = []
        current_coeffs: list[complex] = []

        def _flush() -> None:
            gates.append(
                ExpGate._structural_gate(
                    MajoranaOperator._from_terms(
                        current_majoranas, current_coeffs, num_modes=system_size
                    ),
                    index=current_index,
                )
            )

        for mono, coeff, pidx in zip(majoranas, gen_coeffs, indices, strict=True):
            if current_majoranas and pidx != current_index:
                _flush()
                current_majoranas = []
                current_coeffs = []
            current_index = pidx
            majorana = tuple(int(i) for i in mono)
            if any(i < 0 for i in majorana):
                raise ValueError(
                    f"Majorana indices must be non-negative; got {majorana}."
                )
            if majorana and max(majorana) >= 2 * system_size:
                raise ValueError(
                    f"Majorana term {majorana} acts on an index >= 2*system_size={2 * system_size}."
                )
            current_majoranas.append(majorana)
            current_coeffs.append(complex(float(coeff)))
        if current_majoranas:
            _flush()

        return cls(
            gates=tuple(gates),
            parameters=tuple(float(p) for p in parameters),
            initial_state=tuple(int(i) for i in initial_state or ()),
            system_size=system_size,
        )


def validate_parameter_mapping(
    mapping: Sequence[int], expected_len: int, unit: str = "gates"
) -> None:
    """Validate that a parameter mapping has the right length and contiguous indices.

    Args:
        mapping: Per-unit angle indices to validate.
        expected_len: Number of entries the mapping must have.
        unit: Noun naming what each entry covers (``"gates"``, ``"graph layers"``), used only in
            the error message.

    Raises:
        ValueError: If the length does not match ``expected_len``, or the indices are not
            contiguous ``0..max`` (a gap would silently invent a phantom parameter).
    """
    if len(mapping) != expected_len:
        raise ValueError(
            f"parameter_mapping has {len(mapping)} entries but there are "
            f"{expected_len} {unit}."
        )
    if mapping and set(mapping) != set(range(max(mapping) + 1)):
        raise ValueError(
            "parameter_mapping indices must be contiguous 0..n-1 with no gaps; "
            f"got {sorted(set(mapping))}."
        )


#: An imaginary residue above this is a non-Hermitian generator, not conversion roundoff.
_GENERATOR_HERMITICITY_ATOL = 1e-9


def _real_generator_coefficient(majorana: Sequence[int], value: complex) -> float:
    """Return the real part of a structural generator coefficient, rejecting a non-Hermitian one."""
    value = complex(value)
    if abs(value.imag) > _GENERATOR_HERMITICITY_ATOL:
        raise ValueError(
            f"Gate generator is not Hermitian: monomial {tuple(majorana)} yields the "
            f"structural generator coefficient {value}, whose imaginary part is not "
            f"negligible. A Majorana generator must be Hermitian -- a weight-2 monomial takes "
            f"an imaginary coefficient, weight-4 real -- and a Pauli generator must be a "
            f"Hermitian Pauli operator (real coefficients)."
        )
    return float(value.real)


def _antihermitian_gen_coeff(majorana: Sequence[int], coeff: complex) -> float:
    r"""Antihermitian-normalize a raw Majorana-product coefficient to a real ``g``.

    Divides out the Hermitian phase ``(1j)**(w(w-1)/2)`` of the weight-``w`` monomial to get the
    structural coefficient of the antihermitian generator the engine rotates by, and negates it --
    the sign is what makes the rotation come out as the ``exp(+i*theta*H)`` [ExpGate][] documents.
    """
    weight = len(majorana)
    gen = -coeff / (1j) ** (weight * (weight - 1) / 2)
    return _real_generator_coefficient(majorana, gen)


def _gate_layers(gate: ExpGate, num_qubits: int) -> list[tuple[tuple[int, ...], float]]:
    r"""Expand one gate into ``(majorana, gen_coeff)`` layers, in application order.

    A ``"pauli"``-family [ExpGate][] places each [Pauli][monoprop.pauli.Pauli] term on
    its qubits within the ``num_qubits``-wide system, Jordan-Wigner maps it, and
    antihermitian-normalizes (one layer per term). A ``"majorana"``-family [ExpGate][] carries
    the Hermitian generator, so its [MajoranaOperator][monoprop.majorana.MajoranaOperator] terms are
    antihermitian-normalized the same way (the $i^{\binom{w}{2}}$ phase divided out) -- unless
    the gate is flagged [ExpGate._structural][] (the wire/dense format), whose coefficients are
    already the structural ``g`` and are used directly.
    """
    # A Pauli-family gate holds a PauliOperator; every other family a MajoranaOperator (so the
    # ``isinstance`` narrows the fall-through arm to MajoranaOperator).
    generator = gate.generator
    if isinstance(generator, PauliOperator):
        layers: list[tuple[tuple[int, ...], float]] = []
        for pauli, coeff in generator.terms.items():
            # PauliOperator only bounds-checks against its own num_qubits (possibly None or
            # larger), so a too-wide generator would pack slots past the end of the monomial.
            if pauli.qubits and pauli.qubits[-1] >= num_qubits:
                raise ValueError(
                    f"Gate generator term {pauli} acts on a qubit index >= the system's "
                    f"num_qubits={num_qubits}."
                )
            slots = _pauli_to_local_slots(pauli.string, pauli.qubits)
            layers.append((slots, _real_generator_coefficient(slots, coeff)))
        return layers

    if gate._structural:
        return [
            (mono, _real_generator_coefficient(mono, c))
            for mono, c in generator.terms.items()
        ]
    return [
        (mono, _antihermitian_gen_coeff(mono, c)) for mono, c in generator.terms.items()
    ]


def expand_monomials(
    gates: Sequence[ExpGate],
    mapping: Sequence[int],
    num_qubits: int,
) -> tuple[list[tuple[int, ...]], list[float], list[int], list[int]]:
    """Flatten gates + an already-resolved per-gate mapping into per-monomial arrays.

    Args:
        gates: [ExpGate][] gates, in application order.
        mapping: The angle index driving each gate (one entry per gate).
        num_qubits: System qubit count, required to place Pauli generators; unused for Majorana.

    Returns:
        ``(majoranas, gen_coeffs, parameter_mapping, gate_indices)`` for the C++ engine, expanded
        per monomial. ``gate_indices[i]`` is the local 0-based index of the authoring gate, so
        the engine can recover gate boundaries; a multi-term gate's monomials share one index.
    """
    num_qubits = _validate_system_size(num_qubits, argument_name="num_qubits")
    majoranas: list[tuple[int, ...]] = []
    gen_coeffs: list[float] = []
    per_monomial: list[int] = []
    gate_indices: list[int] = []
    for gate_index, (gate, param) in enumerate(zip(gates, mapping, strict=True)):
        # A gate whose every term fell below its atol expands to nothing, which would hole the
        # contiguous gate_indices runs the engine requires and orphan the gate's parameter slot.
        # Emit the identity instead: the empty monomial with a zero coefficient rotates by zero.
        layers = _gate_layers(gate, num_qubits) or [((), 0.0)]
        for majorana, gen_coeff in layers:
            majoranas.append(majorana)
            gen_coeffs.append(gen_coeff)
            per_monomial.append(param)
            gate_indices.append(gate_index)
    return majoranas, gen_coeffs, per_monomial, gate_indices
