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

"""Authoring types for Majorana/qubit circuits.

The authoring model is four layers: a **term** (:class:`~monoprop.majorana_data.Majorana` /
:class:`~monoprop.pauli_data.Pauli`) is the atom; an **operator**
(:class:`~monoprop.majorana_data.MajoranaOperator` / :class:`~monoprop.pauli_data.PauliOperator`)
is a weighted sum of terms; an **exponential gate** wraps a generator (a term or an operator)
that gets exponentiated; and a **circuit** is an ordered sequence of such gates.

A gate is an explicit exponential of a generator. There is a **single** :class:`Exp` gate
type; it *abstracts over the family* the same way :class:`Circuit` does -- the **generator
type** it is handed decides how it is normalized:

- a :class:`~monoprop.majorana_data.Majorana` term or
  :class:`~monoprop.majorana_data.MajoranaOperator` is a native Majorana generator whose
  coefficients are the *structural* generator coefficients ``g`` used directly (the layer
  angle is ``parameters[param] * g``);
- a :class:`~monoprop.pauli_data.Pauli` term or :class:`~monoprop.pauli_data.PauliOperator`
  is a qubit generator; each term is Jordan-Wigner mapped and antihermitian-normalized when
  the circuit is ingested;
- a :class:`~monoprop.fermi_data.FermiOperator` is a fermionic generator; it is
  antihermitian-normalized into a structural Majorana :class:`Exp` at :class:`Circuit`
  construction.

There is likewise a **single** :class:`Circuit` type. The gate objects carry the family, so
one circuit can be authored from Majorana/fermionic gates *or* qubit gates, and the circuit
validates that its gates are a single, consistent family (the two cannot be mixed). Each gate
is the unit of parameterization: one gate is driven by exactly one variational angle, named
by its ``param`` index (``None`` on every gate => each gate gets its own angle in order;
repeat an index to tie gates to a shared angle). A multi-term generator is a single
exponential driven by a single angle.

The propagators check the circuit's :attr:`Circuit.family`:
:class:`~monoprop.majorana_propagator.MajoranaPropagator` consumes a Majorana/fermionic
circuit, :class:`~monoprop.pauli_propagator.PauliPropagator` a qubit circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .conversion_utils import _extend_pauli_string, _pauli_to_fermi
from .majorana_data import Majorana, MajoranaOperator
from .pauli_data import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .fermi_data import FermiOperator


@dataclass(frozen=True, slots=True, init=False)
class Exp:
    """The exponential of a generator: one variational gate, abstract over the family.

    A single gate type serves every family; the **generator type** decides how the generator
    is normalized (mirroring how a single :class:`Circuit` dispatches on its gates):

    - :class:`~monoprop.majorana_data.Majorana` / :class:`~monoprop.majorana_data.MajoranaOperator`
      -- a native Majorana generator; its coefficients are the *structural* generator
      coefficients ``g`` (the layer angle is ``parameters[param] * g``), and a non-negligible
      imaginary part is rejected as non-Hermitian.
    - :class:`~monoprop.pauli_data.Pauli` / :class:`~monoprop.pauli_data.PauliOperator` -- a
      qubit generator; each Pauli term is Jordan-Wigner mapped and antihermitian-normalized by
      :func:`expand_monomials` when the circuit is ingested, using the propagator's qubit count.
    - :class:`~monoprop.fermi_data.FermiOperator` -- a fermionic generator; held in raw Majorana
      form and antihermitian-normalized into a structural Majorana ``Exp`` at :class:`Circuit`
      construction, so a circuit's ``gates`` never contain a ``"fermi"``-family gate.

    Attributes:
        generator: The generator operator (a ``MajoranaOperator`` or ``PauliOperator``).
        param: The variational-angle index driving this gate, or ``None`` for the identity
            mapping (see :class:`Circuit`).
        family: The generator family -- ``"pauli"``, ``"majorana"``, or ``"fermi"`` -- inferred
            from the generator type at construction.
    """

    generator: MajoranaOperator | PauliOperator
    param: int | None
    family: str

    def __init__(
        self,
        generator: Majorana | MajoranaOperator | Pauli | PauliOperator | FermiOperator,
        param: int | None = None,
    ) -> None:
        """Wrap a generator; its type selects the family and normalization convention."""
        if isinstance(generator, Pauli):
            generator = PauliOperator({generator: 1.0})
            family = "pauli"
        elif isinstance(generator, PauliOperator):
            family = "pauli"
        elif isinstance(generator, Majorana):
            generator = MajoranaOperator({generator: 1.0})
            family = "majorana"
        elif isinstance(generator, MajoranaOperator):
            family = "majorana"
        elif hasattr(generator, "get_majorana_operator"):
            # A fermionic generator (e.g. FermiOperator): raw Majorana-product coefficients,
            # antihermitian-normalized to structural Majorana coefficients at Circuit
            # construction. Duck-typed to avoid a circular import of FermiOperator.
            generator = generator.get_majorana_operator()
            family = "fermi"
        else:
            raise TypeError(
                f"Exp generator must be a Majorana/MajoranaOperator, Pauli/PauliOperator, or "
                f"FermiOperator; got {type(generator).__name__}."
            )
        object.__setattr__(self, "generator", generator)
        object.__setattr__(self, "param", None if param is None else int(param))
        object.__setattr__(self, "family", family)


def _fermi_exp_to_majorana_exp(gate: Exp) -> Exp:
    """Normalize a ``"fermi"``-family :class:`Exp` into a structural Majorana :class:`Exp`.

    The generator's monomials are antihermitian-normalized to real structural coefficients
    and packed into a :class:`~monoprop.majorana_data.MajoranaOperator`; zero-coefficient
    (identity) monomials are dropped by the operator's construction threshold. The gate's
    ``param`` is preserved.
    """
    majoranas: list[tuple[int, ...]] = []
    coefficients: list[complex] = []
    for monomial, coefficient in gate.generator.terms.items():
        # Validates the fermionic generator is Hermitian (raises otherwise).
        majoranas.append(tuple(monomial))
        coefficients.append(complex(_antihermitian_gen_coeff(monomial, coefficient)))
    return Exp(
        MajoranaOperator._from_terms(majoranas, coefficients, gate.generator.num_modes),
        param=gate.param,
    )


@dataclass(frozen=True)
class Circuit:
    """A variational circuit: an ordered sequence of exponential gates, angles, and a state.

    A **single** circuit type serves every gate family, built from the single :class:`Exp`
    gate. The gates carry the family: a Majorana or fermionic :class:`Exp` for
    Majorana/fermionic problems (consumed by
    :class:`~monoprop.majorana_propagator.MajoranaPropagator`) and a Pauli :class:`Exp` for
    qubit problems (consumed by :class:`~monoprop.pauli_propagator.PauliPropagator`). The two
    families cannot be mixed in one circuit -- construction rejects it. A ``"fermi"``-family
    :class:`Exp` gate is normalized into a structural Majorana :class:`Exp` at construction, so
    :attr:`gates` never contains a fermionic gate; the originals are kept on
    :attr:`fermi_generators`.

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
        gates: The ordered exponential gates (fermionic gates normalized to structural
            Majorana :class:`Exp`).
        parameters: The angle values, or empty for an unbound circuit.
        initial_state: The reference state (occupied mode / qubit indices).
        family: The gate family -- ``"pauli"``, ``"majorana"``, or ``"empty"`` -- computed at
            construction; the propagators dispatch on it.
        fermi_generators: The original fermionic :class:`Exp` gates, if any were given (before
            normalization to a structural Majorana :class:`Exp`).
    """

    gates: tuple[Exp, ...] = ()
    parameters: tuple[float, ...] = ()
    initial_state: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize fermionic gates, validate the family, mapping, and params."""
        raw = tuple(self.gates)
        contains_fermi = any(getattr(gate, "family", None) == "fermi" for gate in raw)
        gates = tuple(
            _fermi_exp_to_majorana_exp(gate)
            if getattr(gate, "family", None) == "fermi"
            else gate
            for gate in raw
        )
        parameters = tuple(float(v) for v in self.parameters)
        initial_state = tuple(int(i) for i in self.initial_state)
        if len(set(initial_state)) != len(initial_state):
            raise ValueError("Duplicate indices in initial state")

        fermi_generators: tuple[Exp, ...] = ()
        if contains_fermi and all(gate.param is None for gate in gates):
            # Fermionic convenience: under the default mapping, drop identity generators
            # (e.g. a zero chemical potential) and their aligned parameter.
            kept = [i for i, gate in enumerate(gates) if gate.generator.terms]
            gates = tuple(gates[i] for i in kept)
            if parameters:
                parameters = tuple(parameters[i] for i in kept)
            fermi_generators = tuple(raw[i] for i in kept if raw[i].family == "fermi")
        elif contains_fermi:
            fermi_generators = tuple(g for g in raw if g.family == "fermi")

        object.__setattr__(self, "gates", gates)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "fermi_generators", fermi_generators)
        object.__setattr__(self, "family", self._resolve_family(gates))

        self.resolved_mapping  # validates the per-gate param scheme
        if self.parameters and len(self.parameters) != self.n_parameters:
            raise ValueError(
                f"parameters has {len(self.parameters)} values but the circuit has "
                f"{self.n_parameters} parameters."
            )

    @staticmethod
    def _resolve_family(gates: Sequence[Exp]) -> str:
        """Validate the gates and return the family: ``"pauli"``/``"majorana"``/``"empty"``.

        Rejects non-:class:`Exp` gates and any mix of qubit and Majorana/fermionic gates.
        Computed once at construction and stored on :attr:`family`; the propagators dispatch
        on it (a ``"fermi"``-family gate is normalized to a ``"majorana"`` one before this runs).
        """
        for gate in gates:
            if not isinstance(gate, Exp):
                raise TypeError(
                    f"Circuit gates must be Exp; got {type(gate).__name__}."
                )
        has_pauli = any(gate.family == "pauli" for gate in gates)
        has_majorana = any(gate.family == "majorana" for gate in gates)
        if has_pauli and has_majorana:
            raise TypeError(
                "A circuit cannot mix qubit and Majorana/fermionic gates; build separate "
                "circuits per family."
            )
        return "pauli" if has_pauli else "majorana" if has_majorana else "empty"

    @property
    def resolved_mapping(self) -> tuple[int, ...]:
        """Per-gate angle index, derived from each gate's ``param``.

        With no gate setting ``param`` this is the identity ``0..n-1`` (each gate its own
        angle). Otherwise every gate must set ``param`` and the indices must be contiguous.
        """
        params = [gate.param for gate in self.gates]
        if all(p is None for p in params):
            return tuple(range(len(self.gates)))
        if any(p is None for p in params):
            raise ValueError(
                "Either every gate must set `param`, or none of them must (mixing an "
                "explicit index with the default would be ambiguous)."
            )
        mapping = tuple(int(p) for p in params)  # type: ignore[arg-type]
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

    def __iter__(self) -> Iterator[Exp]:
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
        left = tuple(
            Exp(gate.generator, param=index)
            for gate, index in zip(self.gates, self.resolved_mapping, strict=True)
        )
        right = tuple(
            Exp(gate.generator, param=index + offset)
            for gate, index in zip(other.gates, other.resolved_mapping, strict=True)
        )
        return Circuit(
            gates=left + right,
            parameters=self.parameters + other.parameters,
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
        consecutive monomials sharing a ``param_ind`` become one Majorana :class:`Exp` whose
        generator is a :class:`~monoprop.majorana_data.MajoranaOperator` carrying those
        monomials with their (structural) generator coefficients, and each gate's
        ``param_ind`` becomes that gate's ``param``, so weight-tying is preserved and the
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
        gates: list[Exp] = []
        current_index: int | None = None
        current_majoranas: list[tuple[int, ...]] = []
        current_coeffs: list[complex] = []

        def _flush() -> None:
            gates.append(
                Exp(
                    MajoranaOperator._from_terms(
                        current_majoranas, current_coeffs, num_modes=None
                    ),
                    param=current_index,
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
            f"Gate generator is not Hermitian: monomial {tuple(majorana)} contributes the "
            f"generator coefficient {value}, whose imaginary part is not negligible. A "
            f"Majorana gate generator must carry real (structural) coefficients, and a "
            f"Pauli gate generator must be a Hermitian Pauli operator (real coefficients)."
        )
    return float(value.real)


def _antihermitian_gen_coeff(majorana: Sequence[int], coeff: complex) -> float:
    """Antihermitian-normalize a raw Majorana-product coefficient to a real ``g``.

    A physical generator's coefficient on the raw product ``gamma_{i_1}...gamma_{i_w}`` is
    turned into the real structural coefficient of the antihermitian generator the engine
    rotates by, dividing out the Hermitian phase ``(1j)**(w(w-1)/2)``. Raises ``ValueError``
    if the result is not real (i.e. the generator is not Hermitian).
    """
    weight = len(majorana)
    gen = -coeff / (1j) ** (weight * (weight - 1) / 2)
    return _real_generator_coefficient(majorana, gen)


def _gate_layers(
    gate: Exp, num_qubits: int | None
) -> list[tuple[tuple[int, ...], float]]:
    """Expand one gate into ``(majorana, gen_coeff)`` layers, in application order.

    A ``"pauli"``-family :class:`Exp` places each :class:`~monoprop.pauli_data.Pauli` term on
    its qubits within the ``num_qubits``-wide system, Jordan-Wigner maps it, and
    antihermitian-normalizes (one layer per term). A ``"majorana"``-family :class:`Exp`
    contributes its :class:`~monoprop.majorana_data.MajoranaOperator` terms directly -- the
    coefficients are already the structural generator coefficients.
    """
    if gate.family == "pauli":
        if num_qubits is None:
            raise ValueError("num_qubits is required to expand a Pauli gate.")
        layers: list[tuple[tuple[int, ...], float]] = []
        for pauli, coeff in gate.generator.terms.items():
            extended = _extend_pauli_string(pauli.string, pauli.qubits, num_qubits)
            majorana, jw_coeff = _pauli_to_fermi(extended)
            layers.append(
                (tuple(majorana), _antihermitian_gen_coeff(majorana, coeff * jw_coeff))
            )
        return layers

    return [
        (maj, _real_generator_coefficient(maj, c))
        for maj, c in gate.generator.terms.items()
    ]


def expand_monomials(
    gates: Sequence[Exp],
    mapping: Sequence[int],
    num_qubits: int | None = None,
) -> tuple[list[tuple[int, ...]], list[float], list[int], list[int]]:
    """Flatten gates + an already-resolved per-gate mapping into per-monomial arrays.

    Args:
        gates: :class:`Exp` gates, in application order.
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
