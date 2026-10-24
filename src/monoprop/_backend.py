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

"""Pick the extension module that serves a propagator.

``comm=None`` is served by the serial ``monoprop._core``. A communicator is served by
``monoprop._core_mpi``, loaded on first use from the file built for the MPI ABI family mpi4py uses:
mpi4py hands over ``MPI_Comm`` handles whose type depends on that family.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from monoprop import _core
from monoprop._mpi_abis import ABIS as _BUILT_MPI_ABIS
from monoprop._mpi_abis import MIN_VERSIONS as _MIN_VERSIONS
from monoprop._mpi_abis import MODULE_FILES as _MODULE_FILES
from monoprop.exceptions import (
    MPIAbiMismatchError,
    MPIUnavailableError,
    MPIVersionError,
)

if TYPE_CHECKING:
    from types import ModuleType

# Re-exported as the documented `monoprop.MPI_ABIS`; read here (not there) so tests can monkeypatch it.
MPI_ABIS: tuple[str, ...] = _BUILT_MPI_ABIS

_MPI_MODULE_NAME = "monoprop._core_mpi"
# mpi4py wheels ship one extension per ABI family, e.g. MPI.mpich.abi3.so
_MPI4PY_VARIANT = re.compile(r"^MPI\.(?P<abi>mpich|openmpi|mpiabi)\.")


def core_module(comm: object | None) -> ModuleType:
    """Return the extension module that serves ``comm``.

    Args:
        comm: ``None`` for a serial propagator, otherwise an mpi4py communicator.

    Returns:
        ``monoprop._core`` for ``None``, otherwise the MPI extension module.

    Raises:
        MPIUnavailableError: this installation was built without MPI.
        MPIAbiMismatchError: mpi4py uses an MPI ABI family this installation was not built for.
        MPIVersionError: the MPI library mpi4py uses is older than monoprop supports.
    """
    if comm is None:
        return _core
    loaded = sys.modules.get(_MPI_MODULE_NAME)
    if loaded is not None:
        return loaded
    if not MPI_ABIS:
        msg = (
            "this monoprop installation has no MPI support (monoprop.MPI_ABIS is empty); "
            "pass comm=None or install an MPI-enabled monoprop"
        )
        raise MPIUnavailableError(msg)
    mpi = _mpi4py()
    abi = detect_mpi4py_abi(mpi)
    if abi not in MPI_ABIS:
        msg = (
            f"mpi4py is using the {abi!r} MPI ABI, but this monoprop installation supports "
            f"{', '.join(MPI_ABIS)}"
        )
        raise MPIAbiMismatchError(msg)
    check_mpi_version(*mpi.get_vendor())
    return _load_mpi_core(abi)


def check_mpi_version(vendor: str, version: tuple[int, ...]) -> None:
    """Reject an MPI implementation older than monoprop supports.

    Args:
        vendor: The implementation, named as ``mpi4py.MPI.get_vendor()`` names it.
        version: Its version, as ``mpi4py.MPI.get_vendor()`` reports it.

    Raises:
        MPIVersionError: ``version`` is below monoprop's floor for ``vendor``. An implementation
            without a floor only emits a ``RuntimeWarning``.
    """
    floor = _MIN_VERSIONS.get(vendor)
    if floor is None:
        supported = ", ".join(
            f"{name} >= {_dotted(low)}" for name, low in _MIN_VERSIONS.items()
        )
        msg = (
            f"monoprop has not been validated with the MPI implementation {vendor!r} "
            f"{_dotted(version)}; it supports {supported}"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return
    if tuple(version) < floor:
        msg = f"monoprop needs {vendor} >= {_dotted(floor)}, found {_dotted(version)}"
        raise MPIVersionError(msg)


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))


def _mpi4py() -> ModuleType:
    # only callers holding a communicator get here, so mpi4py is already loaded
    from mpi4py import MPI  # noqa: PLC0415

    return MPI


def detect_mpi4py_abi(mpi: ModuleType) -> str:
    """Return the MPI ABI family of the MPI library mpi4py uses.

    Args:
        mpi: The ``mpi4py.MPI`` module.

    Returns:
        ``"mpich"``, ``"openmpi"`` or ``"mpiabi"``.
    """
    origin = getattr(getattr(mpi, "__spec__", None), "origin", None)
    if origin:
        match = _MPI4PY_VARIANT.match(Path(origin).name)
        if match is not None:
            return match["abi"]
    # conda-forge and source builds of mpi4py carry no ABI in the file name: ask the library
    try:
        abi_major = mpi.Get_abi_version()[0]
    except (NotImplementedError, mpi.Exception):
        abi_major = 0
    if abi_major > 0:
        return "mpiabi"
    if mpi.Get_library_version().startswith("Open MPI"):
        return "openmpi"
    # MPICH, Intel MPI, MVAPICH and Cray MPICH share the MPICH ABI
    return "mpich"


def _load_mpi_core(abi: str) -> ModuleType:
    serial = Path(_core.__file__)
    # e.g. ".abi3.so": the MPI modules come from the same build, hence the same suffix
    path = serial.with_name(_MODULE_FILES[abi] + serial.name.removeprefix("_core"))
    spec = importlib.util.spec_from_file_location(_MPI_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load the {abi!r} MPI backend from {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MPI_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_MPI_MODULE_NAME]
        raise
    return module
