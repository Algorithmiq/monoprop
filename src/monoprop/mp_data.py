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

"""Module for handling data related to Many-Body Simulations (MP) in monoprop.

This module provides the MPData class for representing and managing
Majorana operators and parameters used in quantum simulations. It handles
serialization to/from JSON and MsgPack binary formats with efficient complex
number representation using separate real/imaginary arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from msgpack import packb, unpackb
from packaging.version import Version

from . import __version__


@dataclass
class MPData:
    """Data class for Majorana operators and parameters for MP.

    This class handles the transfer of data to the C++ MonoProp version. In this representation,
    we represent the majoranas as hermitian operators, i.e. sorted tuples of integers. To make
    them anti-hermitian we multiply them by (1j).

    The unitary evolution is given by
    U = (exp(i g[k-1] p[k-1] m[k-1] / 2 )... exp(i g[1] p[1] m[1] / 2 ) exp(i g[0] p[0] m[0] / 2 ))
    where g[i] are the gen_coeffs, p[i] are the parameters, and m[i] are the Hermitian majoranas.
    """

    majoranas: list[tuple[int, ...]]
    """Hermitian Majorana operators to evolve the system with.

    A hermitian Majorana operator is represented as a sorted tuple of integers,
    where each integer represents a Majorana mode. To be Hermitian, the tuple must be sorted
    in ascending order and contain no duplicates, and be represented in terms of the majoranas m_i
    as:

    t = (i, j, k, ...) = (1j)**(len(t)**2 - len(t))/2 * m_i * m_j * m_k * ...
    """
    gen_coeffs: np.ndarray
    """Real coefficients that account for the sign from the making the Majorana operators antihermitian."""
    param_inds: np.ndarray
    """Indices mapping each Majorana operator to its corresponding parameter."""
    parameters: np.ndarray
    """Parameters for the Heisenberg evolution."""
    fermionic_hamiltonian: dict[tuple[int, ...], complex]
    """Fermionic Hamiltonian in Majorana representation.

    NOTE: The keys of this dictionary are tuples representing non-Hermitian Majorana products, and
    the values are complex coefficients.
    """
    hartree_fock: list[int]
    """Hartree-Fock state as a list of occupied modes."""
    num_modes: int
    """Number of modes in the system."""
    ansatz_energy: float
    """Expectation value computed with the circuit and the simulated cutoff."""
    ansatz_gradient: np.ndarray
    """Gradient computed with the circuit and the simulated cutoff."""
    actual_energy: float
    """Actual expectation value."""
    actual_gradient: np.ndarray
    """Actual gradient."""
    evolved_hamiltonian: dict[tuple[int, ...], complex]
    """Evolved Hamiltonian after applying the circuit."""
    tag: str = ""
    """Optional tag for identifying the data."""
    fermionic_pool: list[list[tuple[tuple[int, ...], complex]]] | None = None
    """Fermionic pool."""
    commutator_gradient: np.ndarray | None = None
    """Commutator gradient."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for serialization.
        """
        return {
            "majoranas": self.majoranas,
            "gen_coeffs": self.gen_coeffs.tolist(),
            "param_inds": self.param_inds.tolist(),
            "parameters": self.parameters.tolist(),
            "fermionic_hamiltonian": _store_dict_tuple_to_complex(
                self.fermionic_hamiltonian
            ),
            "hartree_fock": self.hartree_fock,
            "num_modes": self.num_modes,
            "ansatz_energy": self.ansatz_energy,
            "ansatz_gradient": self.ansatz_gradient.tolist(),
            "actual_energy": self.actual_energy,
            "monoprop_version": __version__,
            "actual_gradient": self.actual_gradient.tolist(),
            "evolved_hamiltonian": _store_dict_tuple_to_complex(
                self.evolved_hamiltonian
            ),
            "tag": self.tag,
            "fermionic_pool": _store_fermionic_pool(self.fermionic_pool)
            if self.fermionic_pool is not None
            else None,
            "commutator_gradient": self.commutator_gradient.tolist()
            if self.commutator_gradient is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MPData:
        """Create MPData from dictionary.

        Args:
            data: Dictionary containing the serialized data in the new format.

        Returns:
            MPData instance.
        """
        # If no version was stored, treat the payload as coming from the current major version.
        save_version = str(data.get("monoprop_version", __version__))
        if __version__ and Version(save_version).major != Version(__version__).major:
            raise ValueError(
                f"Version mismatch: expected {__version__}, got {save_version}"
            )

        raw_commutator_gradient = data.get("commutator_gradient")
        commutator_gradient = (
            None
            if raw_commutator_gradient is None
            else np.asarray(raw_commutator_gradient)
        )

        raw_fermionic_pool = data.get("fermionic_pool")
        fermionic_pool = (
            None
            if raw_fermionic_pool is None
            else _load_fermionic_pool(raw_fermionic_pool)
        )

        return cls(
            majoranas=[tuple(maj) for maj in data["majoranas"]],
            gen_coeffs=np.asarray(data["gen_coeffs"]),
            param_inds=np.asarray(data["param_inds"], dtype=int),
            parameters=np.asarray(data["parameters"]),
            fermionic_hamiltonian=_load_dict_tuple_to_complex(
                data["fermionic_hamiltonian"]
            ),
            hartree_fock=data["hartree_fock"],
            num_modes=data["num_modes"],
            ansatz_energy=data["ansatz_energy"],
            ansatz_gradient=np.asarray(data["ansatz_gradient"]),
            actual_energy=data["actual_energy"],
            actual_gradient=np.asarray(data["actual_gradient"]),
            evolved_hamiltonian=_load_dict_tuple_to_complex(
                data["evolved_hamiltonian"]
            ),
            tag=data.get("tag", ""),
            fermionic_pool=fermionic_pool,
            commutator_gradient=commutator_gradient,
        )

    def to_msgpack(self, filepath: str | Path) -> None:
        """Serialize to MsgPack binary format.

        Args:
            filepath: File path to save the MsgPack payload to.
        """
        with Path(filepath).open("wb") as fh:
            fh.write(packb(self.to_dict()))

    @classmethod
    def from_msgpack(cls, filepath: str | Path) -> MPData:
        """Load from MsgPack file.

        Args:
            filepath: File path to load MsgPack from.

        Returns:
            MPData instance.
        """
        with Path(filepath).open("rb") as fh:
            return cls.from_dict(unpackb(fh.read()))


def _store_dict_tuple_to_complex(
    data: dict[tuple[int, ...], complex],
) -> dict[str, Any]:
    keys = []
    real = []
    imag = []
    for key, value in data.items():
        keys.append(list(key))  # Convert tuple to list
        real.append(value.real)
        imag.append(value.imag)
    return {"keys": keys, "vals": {"real": real, "imag": imag}}


def _load_dict_tuple_to_complex(data: dict[str, Any]) -> dict[tuple[int, ...], complex]:
    keys = data["keys"]
    real = data["vals"]["real"]
    imag = data["vals"]["imag"]
    return {tuple(k): complex(r, i) for k, r, i in zip(keys, real, imag, strict=True)}


def _store_fermionic_pool(
    pool: list[list[tuple[tuple[int, ...], complex]]],
) -> dict[str, Any]:
    generators = []
    real_coeffs = []
    imag_coeffs = []

    for operator in pool:
        op_generators = []
        for indices, coeff in operator:
            op_generators.append(
                list(indices)
            )  # Convert tuple to list for serialization
            real_coeffs.append(coeff.real)
            imag_coeffs.append(coeff.imag)
        generators.append(op_generators)

    return {
        "generators": generators,
        "coefficients": {"real": real_coeffs, "imag": imag_coeffs},
    }


def _load_fermionic_pool(
    data: dict[str, Any],
) -> list[list[tuple[tuple[int, ...], complex]]]:
    pool: list[list[tuple[tuple[int, ...], complex]]] = []
    real_coeffs = data["coefficients"]["real"]
    imag_coeffs = data["coefficients"]["imag"]

    coeff_idx = 0
    for generators in data["generators"]:
        item = []
        for indices in generators:
            coeff = complex(real_coeffs[coeff_idx], imag_coeffs[coeff_idx])
            item.append((tuple(indices), coeff))
            coeff_idx += 1
        pool.append(item)
    return pool
