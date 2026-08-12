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

"""A fixture wide enough to run the row store that ships for wide systems.

Every checked-in msgpack fixture is 28 modes or fewer, so all of them store monomials in a single
32-mode block -- below every sparse-row crossover. The support-form row store could therefore be
compiled, forced on with ``monoprop_ROW_STORE=sparse``, pass the whole suite, and still never have seen
the width it exists for. These tests close that gap with ``WIDE_EMBEDDING``: the LiH fixture's 12 modes
relabelled into a 90-mode system, which stores at 96 modes -- three 64-bit words per monomial, an active
window six modes short of its storage, and mode positions that straddle both storage-word boundaries.

The embedding is also the oracle. Relabelling modes monotonically is a canonical transformation, so the
wide run owes the narrow run's evolved operator term for term under the map, and the fixture's exact
energy and gradient still apply -- no second reference calculation, and no second checked-in blob.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from monoprop import MajoranaPropagator
from tests.cases import (
    WIDE_EMBEDDING,
    CasesWideFermionicProblem,
    FermionicProblem,
    load_problem,
)

DATA = Path(__file__).parent / "data"
_SOURCE = "lih_fermionic_spin_exact"

# The source fixture's own width. The embedding only relabels, so the physics stays inside 12 modes and
# no evolved term can carry more than 24 Majorana indices -- which is what makes a cutoff of 2 * 12
# untruncated for the 90-mode problem, and its reference values exact.
_SOURCE_MODES = 12


def _propagate(
    problem: FermionicProblem, cutoff: int, cutoff_type: str = "length", comm=None
) -> MajoranaPropagator:
    prop = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=cutoff,
        cutoff_type=cutoff_type,
        comm=comm,
    )
    prop.propagate(problem.monomial_circuit.to_circuit())
    return prop


@parametrize_with_cases("problem", cases=CasesWideFermionicProblem)
def test_the_wide_case_sits_above_the_shipped_sparse_crossover(
    problem: FermionicProblem, serial_comm
) -> None:
    """The premise the rest of this module rests on: 90 logical modes stored at 96.

    96 is at or above the crossover a released wheel is built with
    (``monoprop_SPARSE_ROW_MIN_MODES`` defaults to 96, and to 1024 only when ``-march=native`` moves
    it), so on a wheel this case reaches the support-form store on its own. A dev build with arch flags
    on selects dense rows at this width and reaches the other backend through
    ``monoprop_ROW_STORE=sparse`` (``just test-sparse-rows``) -- pinned below, so that run cannot
    quietly have used dense rows after all.
    """
    prop = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=4,
        comm=serial_comm,
    )
    assert problem.n_modes == 90
    assert prop.num_modes == 90
    assert prop._simulator.storage_num_modes == 96
    # SIM112 asks for an uppercase name; monoprop spells its variables this way (see EnvConfig.h).
    if os.environ.get("monoprop_ROW_STORE") == "sparse":  # noqa: SIM112
        assert prop._simulator.rows_are_sparse


@parametrize_with_cases("problem", cases=CasesWideFermionicProblem)
def test_wide_run_reproduces_the_exact_energy_and_gradient(
    problem: FermionicProblem, comm
) -> None:
    """The fixture's exact values, reached at a width no fixture on disk has.

    The cutoff is ``2 * _SOURCE_MODES``, not ``2 * problem.n_modes``: both are untruncated for this
    problem, and the smaller one says why -- the embedded physics cannot leave its 12 modes.
    """
    circuit = problem.monomial_circuit.to_circuit()
    prop = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * _SOURCE_MODES,
        comm=comm,
    )
    prop.build_graph(circuit)
    np.testing.assert_allclose(prop.expval(circuit), problem.exact_expval, atol=1e-9)
    np.testing.assert_allclose(prop.grad(circuit), problem.exact_gradient, atol=1e-9)


@pytest.mark.parametrize("cutoff_type", ["length", "support"])
@pytest.mark.parametrize("cutoff", [4, 6])
def test_truncated_wide_run_matches_the_narrow_run_term_by_term(
    cutoff: int, cutoff_type: str, serial_comm
) -> None:
    """The sharpest check the embedding affords, and the one truncation makes non-trivial.

    Which terms survive is invariant under the relabelling: both cutoffs count something the map
    preserves -- Majorana indices for ``length``, occupied modes for ``support`` -- so the wide operator
    must be the narrow one relabelled, coefficient for coefficient. A term the wide run kept and the
    narrow one dropped would mean the surviving set depends on the storage width.

    Coefficients are compared to a tolerance rather than bit-wise: width enters a monomial's hash, so it
    enters probe order and hence the order coefficients accumulate in. (They do come out identical here;
    that is not something the engine promises.)
    """
    narrow = _propagate(
        load_problem(DATA / f"{_SOURCE}.msgpack"), cutoff, cutoff_type, serial_comm
    )
    wide = _propagate(
        load_problem(DATA / f"{_SOURCE}.msgpack", WIDE_EMBEDDING),
        cutoff,
        cutoff_type,
        serial_comm,
    )

    narrow_terms = narrow.evolved_operator(atol=0.0).terms
    wide_terms = wide.evolved_operator(atol=0.0).terms
    mapped = {
        tuple(WIDE_EMBEDDING.majorana(i) for i in key): coeff
        for key, coeff in narrow_terms.items()
    }
    assert len(mapped) == len(narrow_terms)  # the map is injective, so nothing merged
    assert set(mapped) == set(wide_terms)
    for key, coeff in mapped.items():
        assert wide_terms[key] == pytest.approx(coeff, rel=1e-12, abs=1e-15)
    assert wide.expectation_value() == pytest.approx(
        narrow.expectation_value(), rel=1e-12
    )
