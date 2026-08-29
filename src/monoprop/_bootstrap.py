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

"""Binds ``monoprop._core`` to the ISA variant of the engine that this CPU can run.

A wheel built as a fat binary carries the compiled engine once per x86-64 ISA tier, under
``monoprop/_variants/<tier>/``, and no ``monoprop/_core`` of its own. Importing this module picks one
and registers it as ``monoprop._core``, so every other import in the package is unaffected.

A single-ISA build -- any source build, and every non-x86-64 wheel -- ships ``monoprop/_core``
directly. There is then nothing to choose and importing this module does nothing at all.

This module is imported for its side effect, which is why the name sorts ahead of ``_core``: the
selection has to be in place before the first ``from ._core import ...`` runs, and import sorters
order the package's imports alphabetically.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_CORE_MODULE = "monoprop._core"
_VARIANTS_DIRNAME = "_variants"

#: Pin the variant instead of probing the CPU. Refused if the named variant is absent or unrunnable,
#: because the point of pinning one is to know which one ran -- benchmarking a tier, reproducing a
#: report, bisecting a codegen difference.
VARIANT_ENV_VAR = "monoprop_VARIANT"


def _variants_root() -> Path | None:
    """The ``_variants`` directory, or ``None`` if this build has none.

    Searched along the package's ``__path__`` rather than next to this file, because an editable
    install has two entries there: the source tree, holding the Python modules, and the CMake install
    tree, holding everything compiled. ``_variants`` is only ever in the second.
    """
    package = sys.modules.get(__package__ or "monoprop")
    search = list(
        getattr(package, "__path__", None) or [str(Path(__file__).resolve().parent)]
    )
    for entry in search:
        candidate = Path(entry) / _VARIANTS_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def _module_path(directory: Path) -> Path | None:
    """The compiled core inside one variant directory, or ``None`` if there is none."""
    # Both spellings occur: a stable-ABI build writes _core.abi3.so, a version-specific one writes
    # _core.cpython-<ver>-<plat>.so. EXTENSION_SUFFIXES holds whichever this interpreter accepts.
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = directory / f"_core{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _probe() -> ModuleType | None:
    try:
        # Deliberately not a top-level import: on a single-ISA build the probe is not shipped at all,
        # and importing this module must still work there.
        from . import _isa  # noqa: PLC0415
    except ImportError:
        return None
    return _isa


def available_variants() -> tuple[str, ...]:
    """ISA variants installed alongside this package, best first.

    Empty for a single-ISA build, which is the shape of every source build.
    """
    root = _variants_root()
    if root is None:
        return ()
    on_disk = {
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and _module_path(entry) is not None
    }
    probe = _probe()
    known = tuple(probe.known_variants()) if probe is not None else ()
    ordered = [name for name in known if name in on_disk]
    # Anything on disk the probe does not know about is still reported, so a mismatch between the two
    # shows up as an odd listing rather than as a variant that silently never gets picked.
    ordered.extend(sorted(on_disk.difference(known)))
    return tuple(ordered)


def supported_variants() -> tuple[str, ...]:
    """ISA variants this CPU should be given, best first, whether installed or not.

    Not the same as the ones it *can* execute -- see :func:`runnable_variants`. Two variants differ
    only in vector width, which is a tuning question and not a capability one, so a CPU that can run
    512-bit code but is measurably better off without it does not offer that variant here.
    """
    probe = _probe()
    return tuple(probe.supported_variants()) if probe is not None else ()


def runnable_variants() -> tuple[str, ...]:
    """ISA variants the running CPU has the instructions for, best first.

    A superset of :func:`supported_variants`, and the one a ``monoprop_VARIANT`` pin is checked
    against: pinning a variant this CPU merely would not have chosen is the whole point of pinning,
    while pinning one it cannot execute is a SIGILL somewhere inside the scan.
    """
    probe = _probe()
    return tuple(probe.runnable_variants()) if probe is not None else ()


def _select(available: tuple[str, ...]) -> str:
    supported = supported_variants()
    requested = os.environ.get(VARIANT_ENV_VAR)

    if requested:
        if requested not in available:
            raise RuntimeError(
                f"{VARIANT_ENV_VAR}={requested!r} is not installed; "
                f"available variants: {', '.join(available)}"
            )
        # Runnable, not supported: a pin this CPU merely would not have chosen is honoured, since
        # comparing it against the one that would have been is the reason for pinning.
        runnable = runnable_variants()
        if runnable and requested not in runnable:
            raise RuntimeError(
                f"{VARIANT_ENV_VAR}={requested!r} needs instructions this CPU does not have; "
                f"runnable variants: {', '.join(runnable)}"
            )
        return requested

    if not supported:
        raise ImportError(
            "monoprop was built as a fat binary but its CPU probe (monoprop._isa) is missing or "
            "unusable, so no ISA variant can be selected. This is a broken installation; "
            f"reinstall, or pin a variant with {VARIANT_ENV_VAR}."
        )

    for variant in supported:
        if variant in available:
            return variant

    raise ImportError(
        f"none of the installed ISA variants ({', '.join(available)}) can run on this CPU, "
        f"which supports {', '.join(supported)}"
    )


def _load(variant: str) -> ModuleType:
    root = _variants_root()
    path = None if root is None else _module_path(root / variant)
    if (
        path is None
    ):  # pragma: no cover - _select only returns variants that have a module
        raise ImportError(f"ISA variant {variant!r} has no compiled core")

    spec = importlib.util.spec_from_file_location(_CORE_MODULE, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load ISA variant {variant!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    # Registered before execution, as extension modules expect, and removed again on failure so a
    # retry does not find a half-initialized module.
    sys.modules[_CORE_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_CORE_MODULE]
        raise
    return module


def install_core() -> str:
    """Register the selected variant as ``monoprop._core``; return its id, or ``""``.

    Returns the empty string for a single-ISA build, where the normal import machinery finds
    ``monoprop._core`` on its own and there is nothing to select.
    """
    if _CORE_MODULE in sys.modules:
        return str(getattr(sys.modules[_CORE_MODULE], "__variant__", ""))

    available = available_variants()
    if not available:
        return ""

    variant = _select(available)
    module = _load(variant)
    # Also as an attribute of the parent package, which is what the import system would have done and
    # what `monoprop._core` attribute access relies on.
    parent = sys.modules.get("monoprop")
    if parent is not None:
        parent._core = module  # type: ignore[attr-defined]
    return variant


selected_variant = install_core()
"""ISA variant bound to ``monoprop._core``, or ``""`` for a single-ISA build."""
