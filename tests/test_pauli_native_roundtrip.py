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

"""Unit tests for the native Pauli surface: symplectic ingest, decode round-trip, introspection.

Covers the gamma-slot encoding of :meth:`PauliOperator.get_symplectic_terms` (the single source
of truth ``X_q -> slot 2q``, ``Y_q -> slot 2q+1``, ``Z_q -> both``), the inverse decode via
:meth:`PauliPropagator.evolved_pauli_operator`, ``update_initial_pauli_operator``, the ``basis``
introspection property, a non-block-multiple qubit width (33 qubits -> 64-mode storage core,
exercising logical-mode masking), and rejection of a bad ``engine_basis`` string.
"""

from __future__ import annotations

import numpy as np
import pytest

from monoprop import Circuit, Exp, Pauli, PauliOperator, PauliPropagator


class TestGetSymplecticTerms:
    """The symplectic (gamma-slot) ingest of a PauliOperator."""

    def test_x_on_slot_2q(self):
        assert PauliOperator({"X": 1.0}, num_qubits=1).get_symplectic_terms() == {
            (0,): 1.0
        }
        # X on qubit 2 -> slot 4.
        assert PauliOperator(
            {Pauli("X", (2,)): 1.0}, num_qubits=3
        ).get_symplectic_terms() == {(4,): 1.0}

    def test_y_on_slot_2q_plus_1_coefficient_unchanged(self):
        # Y carries no Jordan-Wigner phase in the native encoding: the coefficient is stored
        # verbatim on slot 2q+1.
        assert PauliOperator({"Y": 2.5}, num_qubits=1).get_symplectic_terms() == {
            (1,): 2.5
        }

    def test_z_on_both_slots(self):
        assert PauliOperator({"Z": 1.0}, num_qubits=1).get_symplectic_terms() == {
            (0, 1): 1.0
        }

    def test_multi_qubit_mixed(self):
        # X on qubit 0 -> slot 0; Y on qubit 1 -> slot 3.
        assert PauliOperator(
            {Pauli("XY", (0, 1)): 1.0}, num_qubits=2
        ).get_symplectic_terms() == {(0, 3): 1.0}
        # Z on qubit 0 -> {0,1}; X on qubit 1 -> {2}.
        assert PauliOperator(
            {Pauli("ZX", (0, 1)): -0.5}, num_qubits=2
        ).get_symplectic_terms() == {(0, 1, 2): -0.5}

    def test_identity_term_maps_to_empty_key(self):
        assert PauliOperator({"I": 2.5}, num_qubits=1).get_symplectic_terms() == {
            (): 2.5
        }

    def test_hermiticity_rejection_on_complex_coeff(self):
        op = PauliOperator({"X": 1.0j}, num_qubits=1)
        with pytest.raises(ValueError, match="Hermitian"):
            op.get_symplectic_terms()

    def test_tiny_imaginary_residue_allowed(self):
        # A machine-precision imaginary residue is tolerated (real part is stored).
        op = PauliOperator({"Z": 1.0 + 1e-13j}, num_qubits=1)
        assert op.get_symplectic_terms() == {(0, 1): pytest.approx(1.0)}


class TestEvolvedPauliOperatorRoundtrip:
    """Ingest an observable, evolve with no gates, decode back to the same PauliOperator."""

    @pytest.mark.parametrize(
        "terms",
        [
            {"X": 1.0},
            {"Y": -2.5},
            {"Z": 0.5},
            {"XY": 0.5, "ZZ": -0.3, "YX": 0.7, "IZ": 0.2, "YY": -0.15},
            {"XYZ": 0.4, "YYY": -0.6, "ZIZ": 0.25, "IYX": 0.5},
        ],
    )
    def test_zero_gate_roundtrip(self, serial_comm, terms):
        num_qubits = len(next(iter(terms)))
        observable = PauliOperator(terms, num_qubits=num_qubits)
        propagator = PauliPropagator(
            observable,
            [],
            cutoff=num_qubits,
            comm=serial_comm,
            engine_basis="pauli",
        )
        decoded = propagator.evolved_pauli_operator()
        assert decoded.num_qubits == num_qubits
        assert decoded.isclose(observable, rtol=0.0, atol=1e-12)

    def test_roundtrip_includes_identity_term(self, serial_comm):
        observable = PauliOperator({"XY": 0.5, "II": 1.25}, num_qubits=2)
        propagator = PauliPropagator(
            observable, [], cutoff=2, comm=serial_comm, engine_basis="pauli"
        )
        decoded = propagator.evolved_pauli_operator()
        # The identity term round-trips as the empty-string Pauli term.
        assert decoded.isclose(observable, rtol=0.0, atol=1e-12)

    def test_majorana_jw_mode_raises(self, serial_comm):
        propagator = PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=2),
            [],
            cutoff=2,
            comm=serial_comm,
            engine_basis="majorana-jw",
        )
        with pytest.raises(NotImplementedError):
            propagator.evolved_pauli_operator()


class TestUpdateInitialPauliOperator:
    """Re-weighting the native initial operator's existing terms."""

    def test_update_reflected_in_evolved(self, serial_comm):
        observable = PauliOperator({"XY": 1.0, "ZZ": -0.5}, num_qubits=2)
        propagator = PauliPropagator(
            observable, [], cutoff=2, comm=serial_comm, engine_basis="pauli"
        )
        propagator.update_initial_pauli_operator(
            PauliOperator({"XY": 3.0, "ZZ": 4.0}, num_qubits=2)
        )
        decoded = propagator.evolved_pauli_operator()
        assert decoded.isclose(
            PauliOperator({"XY": 3.0, "ZZ": 4.0}, num_qubits=2), rtol=0.0, atol=1e-12
        )

    def test_majorana_jw_mode_raises(self, serial_comm):
        propagator = PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=2),
            [],
            cutoff=2,
            comm=serial_comm,
            engine_basis="majorana-jw",
        )
        with pytest.raises(NotImplementedError):
            propagator.update_initial_pauli_operator(
                PauliOperator({"ZZ": 2.0}, num_qubits=2)
            )


class TestBasisIntrospection:
    """The read-only ``basis`` property reports the engine backing."""

    def test_native_reports_pauli(self, serial_comm):
        propagator = PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=2),
            [],
            cutoff=2,
            comm=serial_comm,
            engine_basis="pauli",
        )
        assert propagator.basis == "pauli"

    def test_jw_reports_majorana(self, serial_comm):
        propagator = PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=2),
            [],
            cutoff=2,
            comm=serial_comm,
            engine_basis="majorana-jw",
        )
        assert propagator.basis == "majorana"


def test_bad_engine_basis_raises(serial_comm):
    with pytest.raises(ValueError, match="engine_basis"):
        PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=2),
            [],
            cutoff=2,
            comm=serial_comm,
            engine_basis="bogus",
        )


def test_non_block_multiple_width_masks_high_modes(serial_comm):
    """33 qubits dispatches to a 64-mode storage core; logical masking must stay correct.

    Exercises the logical_num_modes masking path (storage block 64, logical 33) and confirms
    the native arm still matches the Jordan-Wigner reference and decodes within [0, 33).
    """
    num_qubits = 33
    observable = PauliOperator(
        {Pauli("Z", (0,)): 1.0, Pauli("Z", (num_qubits - 1,)): 0.5}, num_qubits=num_qubits
    )
    gate_angles = [
        (Exp(PauliOperator({Pauli("X", (0,)): 1.0}, num_qubits=num_qubits)), -0.3),
        (Exp(PauliOperator({Pauli("X", (num_qubits - 1,)): 1.0}, num_qubits=num_qubits)), -0.4),
        (
            Exp(PauliOperator({Pauli("ZZ", (0, num_qubits - 1)): 1.0}, num_qubits=num_qubits)),
            -0.2,
        ),
    ]
    circuit = Circuit(
        gates=tuple(gate for gate, _ in gate_angles),
        parameters=tuple(angle for _, angle in gate_angles),
        initial_state=[],
    )

    def _build(engine_basis):
        propagator = PauliPropagator(
            observable, [], cutoff=4, comm=serial_comm, engine_basis=engine_basis
        )
        propagator.build_graph(circuit)
        return propagator

    native = _build("pauli")
    jw = _build("majorana-jw")
    assert np.isclose(
        native.expectation_value(circuit.parameters),
        jw.expectation_value(circuit.parameters),
        rtol=1e-12,
        atol=1e-12,
    )
    assert native.size() == jw.size()

    decoded = native.evolved_pauli_operator(circuit.parameters)
    assert decoded.num_qubits == num_qubits
    for pauli in decoded.terms:
        assert all(0 <= q < num_qubits for q in pauli.qubits)
