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

"""Count the terms the operator holds when each gate of a Fig. 3 circuit is applied.

The cost panels divide a run's time by the work it did, gate by gate: a gate scans (dense codes)
or indexes into (monoprop) the operator as it stands when the gate arrives, so the work is

    terms held = sum over every gate of the run of the operator's size just before it,

not the gate count times the final size, which overstates it wherever the operator is still
growing (and so by more on the large Hubbard lattices than on the small ones, which the cutoff
saturates within two or three layers). This replays the timed circuit with monoprop one authoring
gate per propagate call, untimed, in the order the timed run applies it, and checks that every
layer (kicked Ising) or gate group (Hubbard) ends with the term count the timed run recorded.
The term set is the same for every engine, so one count serves all of them.

    python scripts/count_held.py --basis pauli --num-qubits 64 --cutoff 6 --layers 7 -o held.jsonl
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from time import perf_counter

import monoprop_single_layer as single
from monoprop import __version__ as monoprop_version
from monoprop.circuit import Circuit


def _single_gates(step):
    """One circuit per authoring gate of ``step``, in the order its propagate call applies
    them: the Heisenberg picture applies a circuit's last gate first."""
    return [
        Circuit(
            gates=(g,),
            system_size=step.system_size,
            parameters=(p,),
            initial_state=step.initial_state,
        )
        for g, p in reversed(list(zip(step.gates, step.parameters, strict=True)))
    ]


def held(factory, steps, layers: int) -> tuple[int, int, list[int], list[int]]:
    """(terms held summed over every gate, gates, held per layer, terms after each step)."""
    sim = factory()
    split = [_single_gates(step) for step in steps]
    total = gates = 0
    per_layer, step_terms = [], []
    for _ in range(layers):
        layer = 0
        for singles in split:
            for one in singles:
                layer += sim.size()
                sim.propagate(one)
            gates += len(singles)
            step_terms.append(sim.size())
        per_layer.append(layer)
        total += layer
    return total, gates, per_layer, step_terms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", choices=["pauli", "majorana"], required=True)
    ap.add_argument("--num-qubits", type=int, required=True)
    ap.add_argument("--cutoff-type", choices=["support", "length"], default="support")
    ap.add_argument("--boundary", choices=["open", "periodic"], default="open")
    ap.add_argument("--lx", type=int, default=None)
    ap.add_argument("--ly", type=int, default=None)
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out", "-o", required=True)
    args = ap.parse_args()

    if args.basis == "majorana":
        factory, _, steps = single.build_hubbard2d(
            args.lx,
            args.ly,
            args.cutoff,
            cutoff_type=args.cutoff_type,
            boundary=args.boundary,
        )
    else:
        factory, circuit = single.build_pauli(
            args.num_qubits, args.cutoff, 0.0, "extensive"
        )
        steps = [circuit]
    t0 = perf_counter()
    total, gates, per_layer, step_terms = held(factory, steps, args.layers)
    record = {
        "engine": "held",
        "basis": args.basis,
        "model": "kicked_ising" if args.basis == "pauli" else "hubbard2d",
        "lattice": [args.lx, args.ly] if args.basis == "majorana" else None,
        "num_qubits": args.num_qubits,
        "cutoff": args.cutoff,
        "cutoff_type": args.cutoff_type if args.basis == "majorana" else None,
        "boundary": args.boundary if args.basis == "majorana" else None,
        "layers": args.layers,
        "lower_atol": 0.0,
        "num_terms": step_terms[-1],
        "gates": gates,
        "held_terms": total,
        "layer_held_terms": per_layer,
        "step_terms": step_terms,
        "count_seconds": perf_counter() - t0,
        "host": socket.gethostname(),
        "monoprop_version": monoprop_version,
    }
    print(
        f"[held/{args.basis}] N={args.num_qubits} L={args.layers} K={record['num_terms']} "
        f"held/(gates*K)={total / (gates * record['num_terms']):.3f} "
        f"t={record['count_seconds']:.1f}s"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
