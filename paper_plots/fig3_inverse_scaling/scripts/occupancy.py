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

"""Measure how evenly the propagated operator is spread over the anticommutation index.

The O(K/N) bound assumes an evenly supported operator: the longest list of the index is within
a constant factor c of the mean list (paper, Definition ``def:even-occupancy``). List j holds the
terms that contain Majorana index j, of which there are 2N (monoprop stores qubit q of a Pauli
string as indices 2q and 2q+1), so the list lengths are the occurrence counts of each index over
the exported operator. After every layer this writes one row: the term count, the list lengths
(all 2N of them), and c = max / mean with the min, coefficient of variation and 99th percentile
beside it. Untimed; one run at the largest depth gives every shallower depth too.

The export holds the operator as Python tuples, ~1 KB per term at its peak; a layer whose
operator exceeds ``--max-terms`` ends the run.
"""

from __future__ import annotations

import argparse
import json
import socket
from itertools import chain
from pathlib import Path
from time import perf_counter

import monoprop_single_layer as single
import numpy as np
from monoprop import __version__ as monoprop_version

CHUNK = 4_000_000


def list_lengths(sim, num_positions: int) -> tuple[int, np.ndarray]:
    """(terms, length of each of the 2N index lists) of the propagator's current operator."""
    keys = [k for k in sim._simulator.evolved_operator([], 0.0) if k]
    lengths = np.zeros(num_positions, dtype=np.int64)
    for lo in range(0, len(keys), CHUNK):
        chunk = keys[lo : lo + CHUNK]
        flat = np.fromiter(chain.from_iterable(chunk), dtype=np.int64)
        lengths += np.bincount(flat, minlength=num_positions)
    return len(keys), lengths


def summary(lengths: np.ndarray) -> dict:
    mean = float(lengths.mean())
    return {
        "list_mean": mean,
        "list_max": int(lengths.max()),
        "list_min": int(lengths.min()),
        "c_max_over_mean": float(lengths.max() / mean),
        "min_over_mean": float(lengths.min() / mean),
        "cv": float(lengths.std() / mean),
        "p99_over_mean": float(np.percentile(lengths, 99) / mean),
        "argmax_position": int(lengths.argmax()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", choices=["pauli", "majorana"], required=True)
    ap.add_argument("--num-qubits", type=int, required=True)
    ap.add_argument("--cutoff-type", choices=["support", "length"], default="support")
    ap.add_argument("--boundary", choices=["open", "periodic"], default="open")
    ap.add_argument("--lx", type=int, default=None)
    ap.add_argument("--ly", type=int, default=None)
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument(
        "--layers", type=int, required=True, help="the deepest layer to reach"
    )
    ap.add_argument("--max-terms", type=float, default=1.6e8)
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
    sim = factory()
    num_positions = 2 * args.num_qubits
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for layer in range(1, args.layers + 1):
        for step in steps:
            sim.propagate(step)
        size = sim.size()
        if size > args.max_terms:
            print(
                f"[occupancy] stop: L={layer} holds {size} terms > {args.max_terms:.3g}"
            )
            break
        t0 = perf_counter()
        num_terms, lengths = list_lengths(sim, num_positions)
        if num_terms not in (size, size - 1):
            raise RuntimeError(f"exported {num_terms} terms, propagator holds {size}")
        record = {
            "engine": "occupancy",
            "basis": args.basis,
            "model": "kicked_ising" if args.basis == "pauli" else "hubbard2d",
            "lattice": [args.lx, args.ly] if args.basis == "majorana" else None,
            "num_qubits": args.num_qubits,
            "cutoff": args.cutoff,
            "cutoff_type": args.cutoff_type if args.basis == "majorana" else None,
            "boundary": args.boundary if args.basis == "majorana" else None,
            "layers": layer,
            "lower_atol": 0.0,
            "num_terms": int(size),
            **summary(lengths),
            "list_lengths": lengths.tolist(),
            "export_seconds": perf_counter() - t0,
            "host": socket.gethostname(),
            "monoprop_version": monoprop_version,
        }
        print(
            f"[occupancy/{args.basis}] N={args.num_qubits} L={layer} K={size} "
            f"c={record['c_max_over_mean']:.3f} cv={record['cv']:.3f} "
            f"min/mean={record['min_over_mean']:.3f}"
        )
        with Path(args.out).open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
