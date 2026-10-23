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

"""Fresh-process probe for ``tests/test_mpi_thread_support.py``.

Run as ``python mpi_thread_support_probe.py LEVEL MODE [OUT]``, directly or under an MPI launcher. It configures
``mpi4py`` to initialize MPI at ``LEVEL`` (``single``, ``funneled``, ``serialized`` or ``multiple``) and imports
``mpi4py.MPI`` *before* ``monoprop``, as a host application would, so the library finds MPI already initialized and
must work with the level actually provided. Modes:

- ``operations``: construct, build a graph, evaluate energies, gradients and retained functionals on the world
  communicator, and the same on ``MPI.COMM_SELF`` as a single-process reference.
- ``construct``: only construct (the insufficient-level refusal).
- ``wrong-thread``: construct and build on the initializing thread, then evaluate from another Python thread.

Each rank writes one JSON document (``OUT/rank<r>.json`` when ``OUT`` is given, else the last stdout line) carrying the
requested and the provided level. Not collected by pytest: the file name has no ``test_`` prefix.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import mpi4py

level, mode = sys.argv[1], sys.argv[2]
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
mpi4py.rc.thread_level = level

from mpi4py import MPI  # noqa: E402

_LEVELS = {
    MPI.THREAD_SINGLE: "single",
    MPI.THREAD_FUNNELED: "funneled",
    MPI.THREAD_SERIALIZED: "serialized",
    MPI.THREAD_MULTIPLE: "multiple",
}
rank = MPI.COMM_WORLD.Get_rank()
doc: dict[str, object] = {
    "requested": level,
    "provided": _LEVELS[MPI.Query_thread()],
    "rank": rank,
    "ranks": MPI.COMM_WORLD.Get_size(),
}

import monoprop  # noqa: E402
from monoprop import (  # noqa: E402
    Circuit,
    ExpGate,
    MajoranaOperator,
    MajoranaPropagator,
)

num_modes = 6
operator = MajoranaOperator(
    {(0, j): 0.1j * j for j in range(2, 2 * num_modes)}
    | {(1, j): 0.05j * j for j in range(2, 2 * num_modes)},
    num_modes,
)
circuit = Circuit(
    [
        ExpGate(MajoranaOperator({(0, 1): 1j}, num_modes), index=0),
        ExpGate(MajoranaOperator({(2, 3): 1j}, num_modes), index=1),
        ExpGate(MajoranaOperator({(1, 4): 1j}, num_modes), index=2),
    ],
    num_modes,
)
params = [0.3, 0.2, 0.1]


def _propagator(comm: object) -> MajoranaPropagator:
    return MajoranaPropagator(operator, [0, 1], cutoff=2 * num_modes, comm=comm)


def _evaluate(comm: object) -> dict[str, object]:
    propagator = _propagator(comm)
    propagator.build_graph(circuit)
    energy = propagator.expectation_value(params)
    value, gradient = propagator.expectation_value_and_gradient(params)
    functional = propagator.expectation_value_and_gradient_functional()
    f_value, f_gradient = functional(params)
    return {
        "energy": float(energy),
        "value": float(value),
        "gradient": [float(g) for g in gradient],
        "functional": float(f_value),
        "functional_gradient": [float(g) for g in f_gradient],
    }


def _write() -> None:
    text = json.dumps(doc)
    if out_dir is not None:
        (out_dir / f"rank{rank}.json").write_text(text)
    else:
        print(text, flush=True)  # noqa: T201


doc["has_mpi"] = monoprop.has_mpi
if mode not in ("operations", "construct", "wrong-thread"):
    raise SystemExit(f"unknown mode {mode!r}")
try:
    if mode == "operations":
        doc["world"] = _evaluate(MPI.COMM_WORLD)
        doc["self"] = _evaluate(MPI.COMM_SELF)
    elif mode == "construct":
        doc["size"] = _propagator(MPI.COMM_WORLD)._simulator.size()
    elif mode == "wrong-thread":
        propagator = _propagator(MPI.COMM_WORLD)
        propagator.build_graph(circuit)
        doc["main_thread_energy"] = float(propagator.expectation_value(params))
        _write()
        # The library fails fast, locally, before any MPI call from this thread: nothing below may run.
        worker = threading.Thread(target=lambda: propagator.expectation_value(params))
        worker.start()
        worker.join()
        doc["completed"] = True
except Exception as error:  # noqa: BLE001 - reported to the parent, which decides
    doc["error"] = f"{type(error).__name__}: {error}"
_write()
