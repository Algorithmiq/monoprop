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

"""Tests for shared validation and basis-change helpers in monoprop.utils."""

from __future__ import annotations

import pytest

from monoprop.utils import (
    _validate_system_size,
    jordan_wigner_basis_change,
    validate_basis_change,
)


def test_validate_system_size_accepts_zero_and_positive() -> None:
    """Zero and positive ints are valid system sizes and round-trip as plain ``int``."""
    assert _validate_system_size(0, argument_name="num_modes") == 0
    result = _validate_system_size(5, argument_name="num_modes")
    assert result == 5
    assert type(result) is int


def test_validate_system_size_rejects_negative() -> None:
    """A negative size is a value error, not a type error, and names the argument."""
    with pytest.raises(ValueError, match="num_modes must be non-negative; got -1"):
        _validate_system_size(-1, argument_name="num_modes")


def test_validate_system_size_rejects_bool() -> None:
    """``bool`` is a ``int`` subclass in Python, so it needs an explicit reject.

    Without the ``isinstance(size, bool)`` guard, ``True``/``False`` would silently pass
    the ``isinstance(size, int)`` check and be accepted as ``1``/``0``.
    """
    with pytest.raises(TypeError, match=r"num_qubits must be an integer \(not bool\)"):
        _validate_system_size(True, argument_name="num_qubits")  # noqa: FBT003
    with pytest.raises(TypeError, match=r"num_qubits must be an integer \(not bool\)"):
        _validate_system_size(False, argument_name="num_qubits")  # noqa: FBT003


def test_validate_system_size_rejects_non_integer() -> None:
    """Non-integer types report the offending type name in the message."""
    with pytest.raises(
        TypeError, match=r"system_size must be an integer \(not float\)"
    ):
        _validate_system_size(2.5, argument_name="system_size")
    with pytest.raises(TypeError, match=r"system_size must be an integer \(not str\)"):
        _validate_system_size("4", argument_name="system_size")


def test_jordan_wigner_basis_change_single_qubit() -> None:
    """One qubit has no ``Z`` prefix, so the two Majoranas map directly to ``X_0``/``Y_0``."""
    assert jordan_wigner_basis_change(1) == [[0], [1]]


def test_jordan_wigner_basis_change_multi_qubit_has_growing_z_string() -> None:
    """Each successive qubit's images carry a longer ``Z`` prefix over the lower slots.

    ``m_{2i}`` is ``X_i`` dressed by ``Z`` on slots ``[0, 2i)``, and ``m_{2i+1}`` is the same
    prefix with ``Y_i`` in place of ``X_i``, per the docstring's slot-support convention.
    """
    basis = jordan_wigner_basis_change(3)
    assert len(basis) == 6
    assert basis[0] == [0]
    assert basis[1] == [1]
    assert basis[2] == [0, 1, 2]
    assert basis[3] == [0, 1, 3]
    assert basis[4] == [0, 1, 2, 3, 4]
    assert basis[5] == [0, 1, 2, 3, 5]


def test_jordan_wigner_basis_change_zero_qubits_is_empty() -> None:
    """The degenerate zero-qubit system has no Majoranas to map."""
    assert jordan_wigner_basis_change(0) == []


def test_validate_basis_change_accepts_none() -> None:
    """``None`` means "use the engine's native basis" and always passes, for any mode count."""
    assert validate_basis_change(None, 4) is None
    assert validate_basis_change(None, 0) is None


def test_validate_basis_change_accepts_matching_length() -> None:
    """A basis change with exactly ``2 * num_modes`` entries is accepted."""
    basis_change = jordan_wigner_basis_change(3)
    assert validate_basis_change(basis_change, 3) is None


def test_validate_basis_change_rejects_wrong_length() -> None:
    """A basis change whose length disagrees with ``2 * num_modes`` is rejected.

    The error message reports both the expected and actual lengths so callers can see how the
    mismatch arose (e.g. built for the wrong mode count).
    """
    basis_change = jordan_wigner_basis_change(2)  # length 4
    with pytest.raises(
        ValueError, match=r"Basis change must have length 6, but got 4\."
    ):
        validate_basis_change(basis_change, 3)


def test_validate_basis_change_rejects_empty_list_for_nonzero_modes() -> None:
    """An empty basis change is only valid for a zero-mode system, not a nonzero one."""
    with pytest.raises(
        ValueError, match=r"Basis change must have length 2, but got 0\."
    ):
        validate_basis_change([], 1)
