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

The authoring model is four layers: a **term** (:class:`~monoprop.majorana.Majorana` /
:class:`~monoprop.pauli.Pauli`) is the atom; an **operator**
(:class:`~monoprop.majorana.MajoranaOperator` / :class:`~monoprop.pauli.PauliOperator`)
is a weighted sum of terms that also carries the system ``num_modes`` / ``num_qubits``; an
**exponential gate** wraps a generator *operator* that gets exponentiated; and a **circuit**
is an ordered sequence of such gates.

A gate is an explicit exponential of a generator *operator* -- :class:`ExpGate` accepts only
operator objects (never a bare term), because those carry the system size. There is a
**single** :class:`ExpGate` gate type; it *abstracts over the family* the same way :class:`Circuit`
does -- the **generator type** it is handed decides how it is normalized:

- a :class:`~monoprop.majorana.MajoranaOperator` is a native Majorana generator carrying
  the *Hermitian* operator (same coefficient convention as an observable: imaginary for a
  weight-2 monomial, real for weight-4); each term is antihermitian-normalized when the circuit
  is ingested, dividing out the Hermitian phase to the structural coefficient the engine rotates
  by;
- a :class:`~monoprop.pauli.PauliOperator` is a qubit generator; each term is
  Jordan-Wigner mapped and antihermitian-normalized when the circuit is ingested;
- a :class:`~monoprop.fermi.FermiOperator` is a fermionic generator; it is converted to
  its (Hermitian) Majorana form in :class:`ExpGate`.

There is likewise a **single** :class:`Circuit` type. The gate objects carry the family, so
one circuit can be authored from Majorana/fermionic gates *or* qubit gates, and the circuit
validates that its gates are a single, consistent family (the two cannot be mixed). Each gate
is the unit of parameterization: one gate is driven by exactly one variational angle, named
by its ``index`` (``None`` on every gate => each gate gets its own angle in order;
repeat an index to tie gates to a shared angle). A multi-term generator is a single
exponential driven by a single angle.

The propagators check the circuit's :attr:`Circuit.family`:
:class:`~monoprop.majorana_propagator.MajoranaPropagator` consumes a Majorana/fermionic
circuit, :class:`~monoprop.pauli_propagator.PauliPropagator` a qubit circuit.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

from .conversion_utils import _extend_pauli_string, _pauli_to_majorana
from .majorana import MajoranaOperator
from .pauli import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .fermi import FermiOperator


#: The family a gate's generator belongs to, inferred from its generator type. A
#: :class:`~monoprop.fermi.FermiOperator` generator is converted to its Majorana form in
#: :meth:`ExpGate.__init__`, so it becomes a ``"majorana"`` gate -- there is no ``"fermi"`` family.
GateFamily = Literal["pauli", "majorana"]
#: The family of a circuit; ``"empty"`` when it has no gates.
CircuitFamily = Literal["pauli", "majorana", "empty"]


class ExpGate:
    r"""The exponential of a generator: one variational gate, abstract over the family.

    A single gate type serves every family; the generator must be an *operator object* (it
    carries the system size), and its **type** decides how it is normalized (mirroring how a
    single :class:`Circuit` dispatches on its gates):

    - :class:`~monoprop.majorana.MajoranaOperator` -- a Majorana generator carrying the
      *Hermitian* operator (its coefficients follow the same convention as an observable:
      imaginary for a weight-2 monomial, real for weight-4); it is antihermitian-normalized --
      the Hermitian phase $i^{\binom{w}{2}}$ divided out -- by :func:`_gate_layers` when
      the circuit is ingested. A coefficient that leaves a non-negligible imaginary residue
      after normalization is rejected as non-Hermitian.
    - :class:`~monoprop.pauli.PauliOperator` -- a qubit generator; each Pauli term is
      Jordan-Wigner mapped and antihermitian-normalized by :func:`expand_monomials` when the
      circuit is ingested, using the propagator's qubit count.
    - :class:`~monoprop.fermi.FermiOperator` -- a fermionic generator; converted to its
      (Hermitian) Majorana form by :meth:`get_majorana_operator` right here in ``__init__``, so
      the gate *is* a ``"majorana"`` gate from then on. The fermionic-to-Majorana mapping already
      carries the factors of 1/2 and the phases, so the resulting coefficients are
      exactly the Hermitian convention above -- no separate fermionic normalization is needed.

    All three families thus take the **Hermitian** generator and normalize it identically; the
    only exception is the internal wire/dense format (:meth:`Circuit.from_dense_arrays`), whose
    coefficients are *already* the real structural ``g`` and are flagged so :func:`_gate_layers`
    passes them through unchanged.

    Attributes:
        generator: The generator operator (a ``MajoranaOperator`` or ``PauliOperator``; a
            ``FermiOperator`` is stored in its converted ``MajoranaOperator`` form).
        index: The variational-angle index driving this gate, or ``None`` for the identity
            mapping (see :class:`Circuit`).
        family: The generator family -- ``"pauli"`` or ``"majorana"`` -- inferred from the
            generator type at construction (a fermionic generator becomes ``"majorana"``).
    """

    __slots__ = ("_structural", "family", "generator", "index")

    def __init__(
        self,
        generator: MajoranaOperator | PauliOperator | FermiOperator,
        index: int | None = None,
        *,
        atol: float = 1e-8,
        _structural: bool = False,
    ) -> None:
        """Wrap a generator operator; its type selects the family and normalization convention.

        The generator must be an *operator object* -- a
        :class:`~monoprop.majorana.MajoranaOperator`,
        :class:`~monoprop.pauli.PauliOperator`, or
        :class:`~monoprop.fermi.FermiOperator` -- because those carry the system
        ``num_modes`` / ``num_qubits``. A bare :class:`~monoprop.majorana.Majorana` /
        :class:`~monoprop.pauli.Pauli` term is *not* accepted; wrap it in the
        corresponding operator (e.g. ``MajoranaOperator({(0, 1): 1j}, num_modes)`` -- a Majorana
        generator carries the Hermitian operator, so a weight-2 coefficient is imaginary).

        ``atol`` is the tolerance for rejecting a non-Hermitian Majorana or Pauli terms; if the corresponding
        coefficients are below the threshold in the absolute values, they are discarded.


        ``_structural`` is internal: :meth:`_structural_gate` sets it when the generator already
        carries the real structural coefficients ``g`` (the wire/dense format), so
        :func:`_gate_layers` passes them through rather than antihermitian-normalizes them.
        """
        if isinstance(generator, PauliOperator):
            family: GateFamily = "pauli"
        elif isinstance(generator, MajoranaOperator):
            family = "majorana"
        elif hasattr(generator, "get_majorana_operator"):
            # A fermionic generator (e.g. FermiOperator): convert to its Majorana form now. The
            # mapping already carries the factors of 1/2 and the phases, so the coefficients come
            # out in the Hermitian convention -- from here it is just a native Majorana gate that
            # _gate_layers antihermitian-normalizes like any other. Duck-typed to avoid a circular
            # import of FermiOperator.
            generator = generator.get_majorana_operator()
            family = "majorana"
        else:
            raise TypeError(
                f"ExpGate generator must be a MajoranaOperator, PauliOperator, or FermiOperator "
                f"(an operator object carrying its mode/qubit count), not a bare term; got "
                f"{type(generator).__name__}."
            )

        self.generator = self._truncated_term(generator, atol)

        if isinstance(self.generator, PauliOperator):
            _validate_commuting_pauli_generator(self.generator)
        elif isinstance(self.generator, MajoranaOperator):
            _validate_commuting_majorana_generator(self.generator)

        self.index = None if index is None else int(index)
        self.family = family
        # Authored generators (including converted fermionic ones) carry the Hermitian operator;
        # _gate_layers normalizes them. Only the wire/dense path sets _structural=True.
        self._structural = _structural

    def _truncated_term(
        self, generator: PauliOperator | MajoranaOperator, atol: float
    ) -> PauliOperator | MajoranaOperator:
        """Return a generator with terms below the threshold dropped.

        The threshold is the same as :meth:`Circuit.from_dense_arrays` uses to drop
        negligible terms when converting a FermiOperator to its Majorana form.
        """
        if isinstance(generator, PauliOperator):
            terms = {p: c for p, c in generator.terms.items() if abs(c) > atol}
            return PauliOperator(terms, generator.num_qubits)

        terms = {m: c for m, c in generator.terms.items() if abs(c) > atol}
        return MajoranaOperator(terms, generator.num_modes)

    @classmethod
    def _structural_gate(
        cls, generator: MajoranaOperator, index: int | None
    ) -> ExpGate:
        """Build a Majorana gate whose coefficients are *already* structural ``g``.

        For the wire/dense format (:meth:`Circuit.from_dense_arrays`) the generator's
        coefficients are already the real structural generator coefficients, so
        :func:`_gate_layers` must pass them through rather than antihermitian-normalize them.
        """
        return cls(generator, index=index, _structural=True)

    @classmethod
    def _with_index(cls, gate: ExpGate, index: int | None) -> ExpGate:
        """Clone ``gate`` with a new ``index``, preserving its family and ``_structural`` flag.

        Used by :meth:`Circuit.__add__`; a plain ``ExpGate(gate.generator, index)`` would reset
        ``_structural`` to ``False`` and re-normalize an already-structural (dense) generator.
        """
        return cls(gate.generator, index=index, _structural=gate._structural)

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


class Circuit:
    """A variational circuit: an ordered sequence of exponential gates, angles, and a state.

    A **single** circuit type serves every gate family, built from the single :class:`ExpGate`
    gate. The gates carry the family: a Majorana :class:`ExpGate` for Majorana/fermionic problems
    (consumed by :class:`~monoprop.majorana_propagator.MajoranaPropagator`) and a Pauli
    :class:`ExpGate` for qubit problems (consumed by
    :class:`~monoprop.pauli_propagator.PauliPropagator`). The two families cannot be mixed in one
    circuit -- construction rejects it. A fermionic generator is converted to its Majorana form
    in :meth:`ExpGate.__init__`, so every gate is already ``"pauli"`` or ``"majorana"``.

    Bundles everything the propagator needs to build or evaluate an evolution:

    - ``gates``: the ordered exponential gates. Each gate is the unit of parameterization --
      one gate is driven by one angle, named by its ``param`` index.
    - ``parameters``: the angle *values* (a point in parameter space). Empty means unbound --
      author the structure now and supply values at evaluation time.
    - ``initial_state``: the reference Slater determinant / computational-basis state.

    The per-gate ``param`` indices give the parameter mapping: if *no* gate sets ``param``,
    each gate gets its own angle in order (the identity mapping); if *any* gate sets it, *all*
    must, the indices must be contiguous ``0..n-1``, and gates sharing an index share an angle.

    Compose circuits with ``+`` (temporal concatenation within the same gate family; the right
    operand's angles are appended on a fresh axis). A *bound* circuit is self-consistent: when
    ``parameters`` is non-empty its length must equal :attr:`n_parameters`.

    Attributes:
        gates: The ordered exponential gates.
        parameters: The angle values, or empty for an unbound circuit.
        initial_state: The reference state (occupied mode / qubit indices).
        family: The gate family -- ``"pauli"``, ``"majorana"``, or ``"empty"`` -- computed at
            construction; the propagators dispatch on it.
    """

    def __init__(
        self,
        gates: Sequence[ExpGate] = (),
        parameters: Sequence[float] = (),
        initial_state: Sequence[int] = (),
    ) -> None:
        """Build the circuit, dropping identity gates and validating family/mapping/params.

        Args:
            gates: The ordered exponential gates.
            parameters: The angle values, or empty for an unbound circuit.
            initial_state: The reference state (occupied mode / qubit indices).

        Raises:
            ValueError: On duplicate initial-state indices, a bad parameter mapping, or a
                bound circuit whose parameter count does not match :attr:`n_parameters`.
            TypeError: On a non-:class:`ExpGate` gate or a mix of qubit and Majorana gate families.
        """
        gates = tuple(gates)
        parameters = tuple(float(v) for v in parameters)
        initial_state = tuple(int(i) for i in initial_state)
        if len(set(initial_state)) != len(initial_state):
            raise ValueError("Duplicate indices in initial state")

        # Validate gate types up front: the identity-drop below reads gate.index/.generator,
        # so a non-ExpGate gate must be rejected with a clear TypeError first rather than crashing
        # with an opaque AttributeError.
        for gate in gates:
            if not isinstance(gate, ExpGate):
                raise TypeError(
                    f"Circuit gates must be ExpGate; got {type(gate).__name__}."
                )

        def _is_identity_gate(gate: ExpGate) -> bool:
            return all(coeff == 0 for coeff in gate.generator.terms.values())

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
        #: The gate family, computed from the (validated) gates; the propagators dispatch on it.
        self.family = self._resolve_family(gates)

        self.resolved_mapping  # validates the per-gate param scheme
        if self.parameters and len(self.parameters) != self.n_parameters:
            raise ValueError(
                f"parameters has {len(self.parameters)} values but the circuit has "
                f"{self.n_parameters} parameters."
            )

    def __eq__(self, other: object) -> bool:
        """Equal when gates, parameters, and initial state match (``family`` is derived)."""
        if not isinstance(other, Circuit):
            return NotImplemented
        return (
            self.gates == other.gates
            and self.parameters == other.parameters
            and self.initial_state == other.initial_state
        )

    __hash__ = None  # type: ignore[assignment]  # value-equal but not hashable (mutable gates)

    def __repr__(self) -> str:
        """Return a string representation listing the gates, parameters, and initial state."""
        return (
            f"{self.__class__.__name__}(gates={self.gates!r}, "
            f"parameters={self.parameters!r}, initial_state={self.initial_state!r})"
        )

    @staticmethod
    def _resolve_family(gates: Sequence[ExpGate]) -> CircuitFamily:
        """Return the family ``"pauli"``/``"majorana"``/``"empty"`` and reject a mixed circuit.

        Gates are already known to be :class:`ExpGate` (:meth:`__init__` validates that first).
        Rejects any mix of qubit and Majorana/fermionic gates. Computed once at construction
        and stored on :attr:`family`; the propagators dispatch on it (a fermionic generator is
        already in Majorana form, converted in :meth:`ExpGate`).
        """
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
        """Per-gate angle index, derived from each gate's ``index``.

        With no gate setting ``index`` this is the identity ``0..n-1`` (each gate its own
        angle). Otherwise every gate must set ``index`` and the indices must be contiguous.
        """
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

        The result applies ``self``'s gates then ``other``'s; ``other``'s angle indices are
        shifted up by ``self.n_parameters`` so the two halves keep independent angles (both
        halves' gates get explicit ``param`` indices in the result). Build the whole thing in
        a single :meth:`~monoprop.MajoranaPropagator.build_graph` call to avoid the
        picture-dependent ordering of incremental multi-call building.

        The two circuits must share a gate family (both qubit, or both Majorana/fermionic).
        """
        if not isinstance(other, Circuit):
            return NotImplemented
        if "empty" not in (self.family, other.family) and self.family != other.family:
            raise TypeError(
                f"Cannot concatenate a {self.family}-family circuit with a "
                f"{other.family}-family one; the gate families differ."
            )
        if (
            self.initial_state
            and other.initial_state
            and self.initial_state != other.initial_state
        ):
            raise ValueError(
                "Cannot concatenate circuits with different initial states."
            )
        offset = self.n_parameters
        # Preserve each gate's _structural flag: a dense (wire-format) gate carries structural
        # coefficients that must not be antihermitian-normalized again.
        left = tuple(
            ExpGate._with_index(gate, index)
            for gate, index in zip(self.gates, self.resolved_mapping, strict=True)
        )
        right = tuple(
            ExpGate._with_index(gate, index + offset)
            for gate, index in zip(other.gates, other.resolved_mapping, strict=True)
        )
        return Circuit(
            gates=left + right,
            parameters=tuple(self.parameters) + tuple(other.parameters),
            initial_state=self.initial_state or other.initial_state,
        )

    @classmethod
    def from_dense_arrays(
        cls,
        majoranas: Sequence[Sequence[int]],
        gen_coeffs: Sequence[float],
        param_inds: Sequence[int],
        parameters: Sequence[float] = (),
        initial_state: Sequence[int] = (),
    ) -> Circuit:
        """Build a Majorana circuit from flat, per-monomial dense arrays.

        This is the native dense/wire format (also the on-disk msgpack-fixture layout):
        consecutive monomials sharing a ``param_ind`` become one Majorana :class:`ExpGate` whose
        generator is a :class:`~monoprop.majorana.MajoranaOperator` carrying those
        monomials with their (structural) generator coefficients, and each gate's
        ``param_ind`` becomes that gate's ``index``, so weight-tying is preserved and the
        expanded engine arrays stay identical to the original.

        Args:
            majoranas: One Majorana-index sequence per monomial.
            gen_coeffs: Generator coefficient per monomial.
            param_inds: Variational-angle index per monomial (contiguous runs group into
                gates).
            parameters: Optional angle values.
            initial_state: Optional reference state (occupied mode indices).

        Returns:
            A :class:`Circuit` carrying the grouped gates, angle values, and initial state.
        """
        indices = [int(p) for p in param_inds]
        gates: list[ExpGate] = []
        current_index: int | None = None
        current_majoranas: list[tuple[int, ...]] = []
        current_coeffs: list[complex] = []

        def _flush() -> None:
            # Dense arrays are the wire format: the coefficients are already the structural
            # generator coefficients g, so build a structural gate that skips normalization.
            gates.append(
                ExpGate._structural_gate(
                    MajoranaOperator._from_terms(
                        current_majoranas, current_coeffs, num_modes=None
                    ),
                    index=current_index,
                )
            )

        for maj, coeff, pidx in zip(majoranas, gen_coeffs, indices, strict=True):
            if current_majoranas and pidx != current_index:
                _flush()
                current_majoranas = []
                current_coeffs = []
            current_index = pidx
            current_majoranas.append(tuple(int(i) for i in maj))
            current_coeffs.append(complex(float(coeff)))
        if current_majoranas:
            _flush()

        return cls(
            gates=tuple(gates),
            parameters=tuple(float(p) for p in parameters),
            initial_state=tuple(int(i) for i in initial_state),
        )


def validate_parameter_mapping(
    mapping: Sequence[int], expected_len: int, unit: str = "gates"
) -> None:
    """Validate that a parameter mapping has the right length and contiguous indices.

    Args:
        mapping: Per-unit angle indices to validate.
        expected_len: Number of entries the mapping must have.
        unit: Noun naming what each entry covers (e.g. ``"gates"`` or ``"graph layers"``),
            used only in the length-mismatch error message.

    Raises:
        ValueError: If the mapping length does not match ``expected_len`` or its indices are
            not contiguous ``0..max`` (an index gap would silently invent a phantom
            parameter).
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


#: A generator coefficient with an imaginary part above this tolerance is rejected as
#: non-Hermitian (the imaginary residue of an exact conversion is at machine precision).
_GENERATOR_HERMITICITY_ATOL = 1e-9


def _real_generator_coefficient(majorana: Sequence[int], value: complex) -> float:
    """Return the real generator coefficient, rejecting a non-Hermitian generator.

    ``value`` is the structural generator coefficient a monomial contributes (after any
    antihermitian normalization). A non-negligible imaginary part means the gate's
    generator is not Hermitian, so exponentiating it would not give a valid rotation --
    fail loudly rather than silently discarding the imaginary part.
    """
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

    A physical generator's coefficient on the raw product $m_{i_1}...m_{i_w}$ is
    turned into the real structural coefficient of the antihermitian generator the engine
    rotates by, dividing out the Hermitian phase $1j^{\binom{w}{2}}$. Raises ``ValueError``
    if the result is not real (i.e. the generator is not Hermitian).
    """
    weight = len(majorana)
    gen = -coeff / (1j) ** (weight * (weight - 1) / 2)
    return _real_generator_coefficient(majorana, gen)


def _paulis_commute(p1: Pauli, p2: Pauli) -> bool:
    """Whether two Pauli terms commute as operators.

    Two Paulis anticommute iff they act with *different* non-identity letters on an odd number
    of shared qubits (:class:`~monoprop.pauli.Pauli` drops identity letters on
    construction, so every letter here is non-trivial).
    """
    op1 = dict(zip(p1.qubits, p1.string, strict=True))
    op2 = dict(zip(p2.qubits, p2.string, strict=True))
    anticommuting = sum(1 for q in op1.keys() & op2.keys() if op1[q] != op2[q])
    return anticommuting % 2 == 0


def _validate_commuting_pauli_generator(generator: PauliOperator) -> None:
    r"""Reject a multi-term Pauli generator whose terms do not pairwise commute.

    A gate is a single exponential of its generator, but :func:`_gate_layers` realizes a
    multi-term generator as a *product* of one rotation per term
    $\exp(\theta g_1 P_1) \cdot \exp(\theta g_2 P_2) \cdot ...$. That product equals
    $\exp(\theta \sum_i g_i P_i)$ only when the Pauli terms mutually commute; otherwise the
    evolution would be silently Trotterized. Fail loudly instead (mirroring the check the old
    ``PauliEvGate`` enforced).

    Raises:
        ValueError: If any two terms of ``generator`` anticommute.
    """
    for p1, p2 in itertools.combinations(generator.terms, 2):
        if not _paulis_commute(p1, p2):
            raise ValueError(
                "A multi-term Pauli gate generator must have mutually commuting terms so the "
                f"gate is a single exponential of their sum; {p1} and {p2} anticommute."
            )


def _majoranas_commute(m1: Sequence[int], m2: Sequence[int]) -> bool:
    """Whether two Majorana monomials commute as operators.

    For canonicalized monomials with distinct indices, swapping the products contributes
    the sign ``(-1)**(len(m1)*len(m2) - |set(m1) & set(m2)|)``.
    """
    n_common = len(set(m1) & set(m2))
    return ((len(m1) * len(m2) - n_common) % 2) == 0


def _validate_commuting_majorana_generator(generator: MajoranaOperator) -> None:
    r"""Reject a multi-term Majorana generator whose terms do not pairwise commute.

    A gate is a single exponential of its generator, but :func:`_gate_layers` realizes a
    multi-term generator as a product of one rotation per term. That product equals
    $\exp(\theta \sum_i g_i M_i)$ only when the Majorana monomials mutually commute;
    otherwise the evolution would be silently Trotterized.

    Raises:
        ValueError: If any two terms of ``generator`` anticommute.
    """
    for m1, m2 in itertools.combinations(generator.terms, 2):
        if not _majoranas_commute(m1, m2):
            raise ValueError(
                "A multi-term Majorana gate generator must have mutually commuting terms "
                "so the gate is a single exponential of their sum; "
                f"{tuple(m1)} and {tuple(m2)} anticommute."
            )


def _gate_layers(
    gate: ExpGate, num_qubits: int | None
) -> list[tuple[tuple[int, ...], float]]:
    r"""Expand one gate into ``(majorana, gen_coeff)`` layers, in application order.

    A ``"pauli"``-family :class:`ExpGate` places each :class:`~monoprop.pauli.Pauli` term on
    its qubits within the ``num_qubits``-wide system, Jordan-Wigner maps it, and
    antihermitian-normalizes (one layer per term). A ``"majorana"``-family :class:`ExpGate` carries
    the Hermitian generator, so its :class:`~monoprop.majorana.MajoranaOperator` terms are
    antihermitian-normalized the same way (the $i^{\binom{w}{2}}$ phase divided out) -- unless
    the gate is flagged :attr:`ExpGate._structural` (the wire/dense format), whose coefficients are
    already the structural ``g`` and are used directly.
    """
    # A Pauli-family gate holds a PauliOperator; every other family a MajoranaOperator (so the
    # ``isinstance`` narrows the fall-through arm to MajoranaOperator).
    generator = gate.generator
    if isinstance(generator, PauliOperator):
        if num_qubits is None:
            raise ValueError("num_qubits is required to expand a Pauli gate.")
        layers: list[tuple[tuple[int, ...], float]] = []
        for pauli, coeff in generator.terms.items():
            extended = _extend_pauli_string(pauli.string, pauli.qubits, num_qubits)
            majorana, jw_coeff = _pauli_to_majorana(extended)
            layers.append(
                (tuple(majorana), _antihermitian_gen_coeff(majorana, coeff * jw_coeff))
            )
        return layers

    if gate._structural:
        return [
            (maj, _real_generator_coefficient(maj, c))
            for maj, c in generator.terms.items()
        ]
    return [
        (maj, _antihermitian_gen_coeff(maj, c)) for maj, c in generator.terms.items()
    ]


def expand_monomials(
    gates: Sequence[ExpGate],
    mapping: Sequence[int],
    num_qubits: int | None = None,
) -> tuple[list[tuple[int, ...]], list[float], list[int], list[int]]:
    """Flatten gates + an already-resolved per-gate mapping into per-monomial arrays.

    Args:
        gates: :class:`ExpGate` gates, in application order.
        mapping: The angle index driving each gate (one entry per gate).
        num_qubits: System qubit count, required to place Pauli-family generators; unused for
            native Majorana generators.

    Returns:
        A tuple ``(majoranas, gen_coeffs, parameter_mapping, gate_indices)`` for the C++
        engine, expanded per monomial. ``gate_indices[i]`` is the (local, 0-based) index of
        the authoring gate monomial ``i`` came from, so the engine can recover gate
        boundaries; monomials from a multi-term gate share one gate index.
    """
    majoranas: list[tuple[int, ...]] = []
    gen_coeffs: list[float] = []
    per_monomial: list[int] = []
    gate_indices: list[int] = []
    for gate_index, (gate, param) in enumerate(zip(gates, mapping, strict=True)):
        for majorana, gen_coeff in _gate_layers(gate, num_qubits):
            majoranas.append(majorana)
            gen_coeffs.append(gen_coeff)
            per_monomial.append(param)
            gate_indices.append(gate_index)
    return majoranas, gen_coeffs, per_monomial, gate_indices
