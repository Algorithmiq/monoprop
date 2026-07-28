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

"""Module for converting integrals to fermionic format."""

from __future__ import annotations

import itertools as it
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from monoprop.fermi import FermiOperator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import ndarray


def _index(
    quad: tuple[int, int, int, int], shift: tuple[int, int, int, int], num_orbs: int
) -> tuple[tuple[int, str], ...]:
    """Turn an orbital quadruple into the ``c+ c+ c- c-`` term, shifting beta spins by ``num_orbs``."""
    exc = ("+", "+", "-", "-")
    new_quad = tuple(
        q + (num_orbs if s else 0) for q, s in zip(quad, shift, strict=True)
    )
    return tuple(zip(new_quad, exc, strict=True))


def _iter_integrals_to_fermion(
    h0: float, h1: ndarray, h2: ndarray
) -> Iterator[tuple[tuple[tuple[int, str], ...], float]]:
    """Yield the ``(fermionic term, coefficient)`` pairs of a zero-, one- and two-body integral set."""
    num_orbs = h1.shape[1]
    # zero-body
    yield (), h0

    # one-body electronic
    for p, q in it.product(range(num_orbs), repeat=2):
        if h1[0, p, q]:
            yield ((p, "+"), (q, "-")), h1[0, p, q]
        if h1[1, p, q]:
            yield ((p + num_orbs, "+"), (q + num_orbs, "-")), h1[1, p, q]

    # two-body electron-electron
    quad_comb: tuple[tuple[int, int], tuple[int, int]]
    quad: tuple[int, int, int, int]

    # aa and bb
    hh = h2.transpose([0, 1, 3, 4, 2])
    for quad_comb in it.product(  # type: ignore[assignment]
        it.combinations(range(num_orbs), r=2), repeat=2
    ):
        quad = (*quad_comb[0], *quad_comb[1])
        for hh_first_ind, shift in zip(
            [0, 2], [(0, 0, 0, 0), (1, 1, 1, 1)], strict=True
        ):
            coeff = sum(
                sign * hh[hh_first_ind, *(quad[i] for i in perm)]
                for perm, sign in [
                    ([0, 1, 2, 3], 1),
                    ([0, 1, 3, 2], -1),
                    ([1, 0, 2, 3], -1),
                    ([1, 0, 3, 2], 1),
                ]
            )
            if coeff:
                yield _index(quad, shift, num_orbs), coeff / 2

    # ab
    for quad in it.product(range(num_orbs), repeat=4):  # type: ignore[assignment]
        coeff = hh[1, *quad]
        if coeff:
            yield _index(quad, (0, 1, 1, 0), num_orbs), coeff


def integrals_to_fermion(h0: float, h1: ndarray, h2: ndarray) -> FermiOperator:
    r"""Convert molecular integrals into a fermionic Hamiltonian.

    integrals should describe an unrestricted molecular Hamiltonian in a spatial-orbital basis.

    Args:
        h0: is the scalar (core) energy containing the nuclear
            repulsion energy and any frozen-core or inactive-space
            contributions.
        h1: is an array of shape ``(2, n_orb, n_orb)`` containing
            the alpha- and beta-spin one-electron integrals, corresponding
            to the kinetic-energy operator and electron-nucleus attraction
            operator (plus any effective one-electron contributions from
            frozen orbitals or inactive electrons).
        h2: is an array of shape
            ``(3, n_orb, n_orb, n_orb, n_orb)`` containing the alpha-alpha,
            beta-beta, and alpha-beta electron-repulsion integrals in
            chemist's notation,
            $(pq|rs)=\int \phi_p(r_1)\phi_q(r_1) r_{12}^{-1}\phi_r(r_2)\phi_s(r_2)\mathrm{d}r_1\mathrm{d}r_2$

    Returns:
        Hamiltonian in FermiOperator format.
    """
    terms = defaultdict(complex)
    for ind, coeff in _iter_integrals_to_fermion(h0, h1, h2):
        if np.isclose(coeff, 0, atol=1e-12):
            continue
        terms[ind] += coeff

    return FermiOperator.from_dict(terms)
