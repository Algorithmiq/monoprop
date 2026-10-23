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

"""Backend selection: ``comm=None`` runs on the serial extension, a communicator on the MPI one."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import monoprop
from monoprop import Circuit, MajoranaOperator, MajoranaPropagator, _backend
from monoprop.exceptions import (
    MPIAbiMismatchError,
    MPIUnavailableError,
    MPIVersionError,
)
from tests.mpi_support import HAS_MPI_BACKEND, MPI

requires_mpi_backend = pytest.mark.skipif(
    not HAS_MPI_BACKEND, reason="needs mpi4py and an MPI-enabled monoprop"
)

N_MODES = 6


class _FakeMPIError(Exception):
    pass


def _fake_mpi(origin, *, library, abi_version, vendor=("Open MPI", (5, 0, 11))):
    """Stand-in for ``mpi4py.MPI`` with the attributes detection and the version check read."""

    def get_abi_version():
        if abi_version is None:
            raise NotImplementedError
        return abi_version

    return SimpleNamespace(
        __spec__=SimpleNamespace(origin=origin),
        Exception=_FakeMPIError,
        Get_abi_version=get_abi_version,
        Get_library_version=lambda: library,
        get_vendor=lambda: vendor,
    )


@pytest.mark.parametrize(
    ("origin", "library", "abi_version", "expected"),
    [
        pytest.param(
            "/site/mpi4py/MPI.mpich.abi3.so",
            "Open MPI v5.0.11",
            (-1, -1),
            "mpich",
            id="wheel-name-wins",
        ),
        pytest.param(
            "/site/mpi4py/MPI.openmpi.abi3.so",
            "MPICH Version: 5.0.2",
            (-1, -1),
            "openmpi",
            id="wheel-openmpi",
        ),
        pytest.param(
            "/site/mpi4py/MPI.mpiabi.cpython-312-x86_64-linux-gnu.so",
            "MPICH Version: 5.0.2",
            (1, 0),
            "mpiabi",
            id="wheel-mpiabi",
        ),
        pytest.param(
            "/conda/mpi4py/MPI.cpython-312-x86_64-linux-gnu.so",
            "Open MPI v4.1.6, package: Open MPI",
            (-1, -1),
            "openmpi",
            id="source-openmpi",
        ),
        pytest.param(
            "/conda/mpi4py/MPI.cpython-312-darwin.so",
            "MPICH Version: 4.2.0",
            (-1, -1),
            "mpich",
            id="source-mpich",
        ),
        pytest.param(
            "/opt/mpi4py/MPI.abi3.so",
            "Intel(R) MPI Library 2021.19 for Linux* OS",
            None,
            "mpich",
            id="source-intel",
        ),
        pytest.param(
            "/opt/mpi4py/MPI.abi3.so",
            "CRAY MPICH version 8.1.29",
            (0, 0),
            "mpich",
            id="source-cray",
        ),
        pytest.param(
            "/opt/mpi4py/MPI.abi3.so",
            "MPICH Version: 5.0.2",
            (1, 0),
            "mpiabi",
            id="source-standard-abi",
        ),
        pytest.param(None, "Open MPI v5.0.11", None, "openmpi", id="no-origin"),
    ],
)
def test_detect_mpi4py_abi(origin, library, abi_version, expected) -> None:
    fake = _fake_mpi(origin, library=library, abi_version=abi_version)
    assert _backend.detect_mpi4py_abi(fake) == expected


def test_detect_mpi4py_abi_survives_an_mpi_error() -> None:
    fake = _fake_mpi("/x/MPI.abi3.so", library="Open MPI v5.0.11", abi_version=(-1, -1))

    def failing():
        raise _FakeMPIError

    fake.Get_abi_version = failing
    assert _backend.detect_mpi4py_abi(fake) == "openmpi"


def test_comm_none_is_served_by_the_serial_module() -> None:
    assert _backend.core_module(None) is monoprop._core


def test_communicator_without_mpi_support_raises(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "monoprop._core_mpi", raising=False)
    monkeypatch.setattr(_backend, "MPI_ABIS", ())
    with pytest.raises(MPIUnavailableError, match="no MPI support"):
        _backend.core_module(object())


def test_mpi4py_abi_outside_the_installation_raises(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "monoprop._core_mpi", raising=False)
    monkeypatch.setattr(_backend, "MPI_ABIS", ("openmpi",))
    fake = _fake_mpi(
        "/site/mpi4py/MPI.mpich.abi3.so",
        library="MPICH Version: 5.0.2",
        abi_version=(-1, -1),
    )
    monkeypatch.setattr(_backend, "_mpi4py", lambda: fake)
    with pytest.raises(MPIAbiMismatchError, match=r"'mpich'.*supports openmpi"):
        _backend.core_module(object())


def test_mpi_below_its_floor_is_rejected_before_loading(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "monoprop._core_mpi", raising=False)
    monkeypatch.setattr(_backend, "MPI_ABIS", ("openmpi",))
    fake = _fake_mpi(
        "/site/mpi4py/MPI.openmpi.abi3.so",
        library="Open MPI v4.0.7",
        abi_version=(-1, -1),
        vendor=("Open MPI", (4, 0, 7)),
    )
    monkeypatch.setattr(_backend, "_mpi4py", lambda: fake)

    def unexpected_load(abi):
        raise AssertionError(abi)

    monkeypatch.setattr(_backend, "_load_mpi_core", unexpected_load)
    with pytest.raises(MPIVersionError, match=r"Open MPI >= 4\.1, found 4\.0\.7"):
        _backend.core_module(object())


@pytest.mark.parametrize(
    ("vendor", "version"),
    [
        ("Open MPI", (4, 1, 0)),
        ("Open MPI", (5, 0, 11)),
        ("MPICH", (4, 0, 0)),
        ("Intel MPI", (2021, 1, 1)),
        ("MVAPICH", (3, 0, 0)),
        ("Cray MPICH", (8, 1, 29)),
    ],
)
def test_mpi_at_or_above_its_floor_is_accepted(vendor, version) -> None:
    _backend.check_mpi_version(vendor, version)


@pytest.mark.parametrize(
    ("vendor", "version", "floor"),
    [
        ("Open MPI", (4, 0, 7), "4.1"),
        ("MPICH", (3, 4, 3), "4.0"),
        ("Intel MPI", (2019, 12, 0), "2021.0"),
        ("MVAPICH", (2, 3, 7), "3.0"),
        ("Cray MPICH", (8, 0, 16), "8.1"),
    ],
)
def test_mpi_below_its_floor_raises(vendor, version, floor) -> None:
    with pytest.raises(MPIVersionError, match=f"{vendor} >= {floor}"):
        _backend.check_mpi_version(vendor, version)


def test_unrecognised_mpi_only_warns() -> None:
    with pytest.warns(RuntimeWarning, match="not been validated"):
        _backend.check_mpi_version("unknown", (0, 0, 0))


def test_serial_module_rejects_a_communicator() -> None:
    with pytest.raises(TypeError, match="serial backend"):
        monoprop._core.MonomialPropagator032(
            initial_operator={(0,): 1.0}, cutoff=3, initial_state=[], comm=object()
        )


def test_import_does_not_import_mpi4py() -> None:
    code = "import sys, monoprop; sys.exit('mpi4py' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603


def _evolved(comm):
    propagator = MajoranaPropagator(
        MajoranaOperator({(0,): 1.0}, N_MODES), [], cutoff=N_MODES // 2, comm=comm
    )
    propagator.propagate(
        Circuit.from_dense_arrays(
            majoranas=[(5,)],
            gen_coeffs=[-1.0],
            param_inds=[0],
            parameters=[1.0],
            system_size=N_MODES,
        )
    )
    return propagator, propagator.evolved_operator().terms


@requires_mpi_backend
def test_serial_and_mpi_backends_coexist_and_agree() -> None:
    serial, serial_terms = _evolved(None)
    distributed, mpi_terms = _evolved(MPI.COMM_SELF)
    assert type(serial._simulator._core).__module__ == "monoprop._core"
    assert type(distributed._simulator._core).__module__ == "monoprop._core_mpi"
    assert serial_terms.keys() == mpi_terms.keys()
    for term, value in serial_terms.items():
        assert np.isclose(value, mpi_terms[term])


@requires_mpi_backend
def test_mpi_module_rejects_comm_none() -> None:
    mpi_core = _backend.core_module(MPI.COMM_SELF)
    with pytest.raises(TypeError, match="mpi4py communicator"):
        mpi_core.MonomialPropagator032(
            initial_operator={(0,): 1.0}, cutoff=3, initial_state=[], comm=None
        )


@requires_mpi_backend
def test_mpi_module_file_matches_mpi4py() -> None:
    mpi_core = _backend.core_module(MPI.COMM_SELF)
    abi = _backend.detect_mpi4py_abi(MPI)
    assert Path(mpi_core.__file__).name.startswith(f"_core_mpi.{abi}.")


@requires_mpi_backend
def test_mpi_module_api_matches_the_serial_module() -> None:
    """Drift guard: the MPI modules carry no stub, so their API must equal _core's."""
    stubgen = pytest.importorskip("nanobind.stubgen")

    def render(module):
        generator = stubgen.StubGen(module)
        generator.put(module)
        return generator.get()

    mpi_core = _backend.core_module(MPI.COMM_SELF)
    assert render(mpi_core).replace("_core_mpi", "_core") == render(monoprop._core)
