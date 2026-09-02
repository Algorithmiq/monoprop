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

"""``monoprop_DROP_SILENT_RECORDS``: what the approximation costs, on literal values.

The knob is parsed once per process, so each arm runs in its own interpreter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

# Anything that would make the child join this process's MPI world (or inherit its rank layout).
_LAUNCHER = re.compile(r"^(PMI|PMIX|OMPI_|MPI_|SLURM_|PRTE_)")

# One tracked pair under one gate: nu = (0, 1) with a coefficient far below ``lower_atol``, and its
# partner mu = nu ^ G = (0, 2, 3, 4) at 1.0. The pair rotates because mu emits, so the exact protocol
# applies BOTH adds; with silent records dropped, nu sends nothing and mu loses nu's contribution.
_NU = (0, 1)
_MU = (0, 2, 3, 4)
_A = 1e-4  # nu's coefficient: below the 1e-3 threshold, so nu never rotates on its own
_B = 1.0  # mu's coefficient: rotates
_LOWER_ATOL = 1e-3
# theta = pi/6, so cos(2 theta) = 1/2 and sin(2 theta) = -sqrt(3)/2. A unit nu contributes exactly
# sin(2 theta) to mu in the dict's encoding (test_coeff_trunc pins that same rotation), and the layer is
# linear in the coefficients, so mu's exact value is _B * COS + _A * SIN and the dropped term is _A * SIN.
# MajoranaOperator.terms rounds its coefficients (12 significant digits), which is why the tolerances
# below are absolute 1e-11 on a coefficient and relative on the difference between the two arms.
_COS = 0.5
_SIN = -0.8660254037844386

# The child takes nu's coefficient, mu's, and lower_atol on argv, so the script is a literal.
_SCRIPT = """
import json, sys
import numpy as np
import monoprop
from monoprop import Circuit, MajoranaPropagator
from monoprop.majorana import MajoranaOperator

a, b, lower_atol = (float(x) for x in sys.argv[1:4])
n_modes = 5
op = MajoranaOperator({(0, 1): 1.0j * a, (0, 2, 3, 4): b}, num_modes=n_modes)
circuit = Circuit.from_dense_arrays(
    initial_state=[],
    majoranas=[(1, 2, 3, 4)],
    parameters=[np.pi / 6],
    gen_coeffs=[1.0],
    param_inds=[0],
    system_size=n_modes,
)
mp = MajoranaPropagator(op, circuit.initial_state, cutoff=4, lower_atol=lower_atol, comm=None)
mp.propagate(circuit)
out = {" ".join(str(i) for i in k): [float(np.real(v)), float(np.imag(v))]
       for k, v in mp.evolved_operator().terms.items()}
world = 1
if monoprop.has_mpi:
    from mpi4py import MPI

    world = MPI.COMM_WORLD.size
json.dump({"world": world, "terms": out}, sys.stdout)
"""


def _run(drop: str | None) -> dict[tuple[int, ...], complex]:
    """Propagate the pair in a fresh interpreter with the knob set to ``drop``.

    The child must be a one-rank world: the coefficients below are the whole operator's, and a
    partitioned world would hand each rank only its own terms. Under ``--with-mpi`` this test's own
    process is a rank of a larger world, so the launcher's environment is stripped from the child.
    """
    env = {k: v for k, v in os.environ.items() if not _LAUNCHER.match(k)}
    env.pop("monoprop_DROP_SILENT_RECORDS", None)
    if drop is not None:
        env["monoprop_DROP_SILENT_RECORDS"] = drop
    env["monoprop_PARTITIONS"] = "off"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT, repr(_A), repr(_B), repr(_LOWER_ATOL)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"child failed: {proc.stderr}"
    out = json.loads(proc.stdout)
    assert out["world"] == 1, f"child joined a {out['world']}-rank world"
    return {
        tuple(int(i) for i in key.split()): complex(re, im)
        for key, (re, im) in out["terms"].items()
    }


@pytest.mark.parametrize("off", ["0", None])
def test_exact_protocol_applies_both_adds(off: str | None) -> None:
    """Unset and ``0`` are the exact protocol: mu carries nu's contribution."""
    terms = _run(off)
    assert terms[_MU] == pytest.approx((_B * _COS) + (_A * _SIN), abs=1e-11)


def test_dropping_silent_records_costs_the_below_atol_contribution() -> None:
    """With the knob on, mu loses exactly ``_A * _SIN`` and nu is untouched."""
    exact = _run("0")
    dropped = _run("1")
    assert dropped[_MU] == pytest.approx(_B * _COS, abs=1e-11)
    # The whole difference between the arms is nu's dropped contribution.
    assert (exact[_MU] - dropped[_MU]).real == pytest.approx(_A * _SIN, rel=1e-6, abs=0)
    # The error is bounded by the threshold that already stopped nu from rotating on its own.
    assert abs(exact[_MU] - dropped[_MU]) < _LOWER_ATOL
    # nu's own half comes from mu's record, which the knob does not drop, so nu is bit-identical.
    assert dropped[_NU] == exact[_NU]
    assert set(dropped) == set(exact)
