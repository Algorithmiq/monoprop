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

"""Count, per gate, the terms of the propagated operator that anticommute with it.

The work-per-gate strip of Fig. 3 sets two numbers side by side for every point of the timed
sweeps: the K terms that a dense code tests against each gate, and the terms that actually
anticommute with the gate, which are the only ones an anticommutation index visits. The second is
measured here, untimed, on the operator monoprop holds after the last layer:

    anticommuting per gate = (1 / G) * sum over the G gates of one layer of |anti(gate)|,

an exact count over the exported operator. Beside it, the index entries the same gates read:

    index reads per gate = (1 / G) * sum over the layer's scans of the lengths of the lists folded,

where each scan (one generator monomial) XOR-folds the lists of G's positions (Majorana) or of
J(G) (Pauli) in full (scripts/read_weights.py); the pivot list, read only where the fold is
nonzero, is left out. A list held as a dense bitmap (over 1/64 of the terms) is read as K/64
words rather than entry by entry; the count is of entries either way. A term anticommutes with a multi-term generator (a
hopping gate's two bilinears, the interaction's three diagonal monomials) when it anticommutes
with any of them. The rules for the two models' gates are written out per term and evaluated
in numpy over chunks of the exported operator; they were checked against a gate-by-gate
brute force on small lattices.

monoprop stores qubit q of a Pauli string as Majorana indices 2q (X), 2q+1 (Y) or both (Z).
"""

from __future__ import annotations

import argparse
import json
import resource
import socket
from itertools import chain
from pathlib import Path
from time import perf_counter

import monoprop_single_layer as single
import numpy as np
import read_weights
from monoprop import __version__ as monoprop_version

CHUNK = 4_000_000


def _flatten(keys):
    """(flat indices, term id per index) of a chunk of sorted index tuples."""
    lens = np.fromiter(map(len, keys), dtype=np.int64, count=len(keys))
    flat = np.fromiter(chain.from_iterable(keys), dtype=np.int64, count=int(lens.sum()))
    return flat, np.repeat(np.arange(len(keys), dtype=np.int64), lens)


def _pauli_chunk(keys, num_qubits: int) -> int:
    """Sum over the chunk's terms of the kicked-Ising gates each anticommutes with.

    X_q anticommutes with a term carrying Y or Z on q, i.e. index 2q+1; Z_q Z_{q+1} with a term
    whose X-or-Y content (index 2q) is on exactly one of q, q+1.
    """
    flat, seg = _flatten(keys)
    odd = (flat & 1) == 1
    total = int(odd.sum())
    xq, xs = flat[~odd] >> 1, seg[~odd]
    degree = 2 - (xq == 0).astype(np.int64) - (xq == num_qubits - 1).astype(np.int64)
    adjacent = (xs[1:] == xs[:-1]) & (xq[1:] - xq[:-1] == 1)
    return total + int(degree.sum()) - 2 * int(adjacent.sum())


def _hubbard_chunk(keys, lx: int, ly: int) -> int:
    """Sum over the chunk's terms of the Hubbard gates each anticommutes with.

    Mode m = 2 * site + spin holds Majorana indices 2m (e) and 2m+1 (o). The interaction on a
    site anticommutes with a term that leaves either of the site's modes unpaired. A hopping gate
    on modes (a, b) has generator terms {2a, 2b+1} and {2a+1, 2b}, so it anticommutes with a term
    when e_a != o_b or o_a != e_b.
    """
    flat, seg = _flatten(keys)
    mode = flat >> 1
    num_modes = 2 * lx * ly
    key = seg * num_modes + mode
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, len(key)))
    ukey, umode, useg = key[starts], mode[starts], seg[starts]
    single_odd = (flat[starts] & 1) == 1
    e = (counts == 2) | ~single_odd
    o = (counts == 2) | single_odd
    unpaired = counts == 1
    total = len(np.unique(useg[unpaired] * (num_modes // 2) + (umode[unpaired] >> 1)))
    site, spin = umode >> 1, umode & 1
    x, y = site % lx, site // lx
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ok = (x + dx >= 0) & (x + dx < lx) & (y + dy >= 0) & (y + dy < ly)
        a = umode[ok]
        b = 2 * ((y[ok] + dy) * lx + x[ok] + dx) + spin[ok]
        bkey = useg[ok] * num_modes + b
        pos = np.searchsorted(ukey, bkey)
        pos_c = np.minimum(pos, len(ukey) - 1)
        present = ukey[pos_c] == bkey
        e_b = present & e[pos_c]
        o_b = present & o[pos_c]
        anti = (e[ok] != o_b) | (o[ok] != e_b)
        total += int((anti & (~present | (a < b))).sum())
    return total


def count(sim, args) -> tuple[int, int, np.ndarray]:
    """(terms, sum over one layer's gates of the terms anticommuting with each, the length of
    every index list)."""
    keys = [k for k in sim._simulator.evolved_operator([], 0.0) if k]
    total = 0
    lists = np.zeros(2 * args.num_qubits, dtype=np.int64)
    for lo in range(0, len(keys), CHUNK):
        chunk = [tuple(sorted(k)) for k in keys[lo : lo + CHUNK]]
        lists += np.bincount(_flatten(chunk)[0], minlength=len(lists))
        if args.basis == "pauli":
            total += _pauli_chunk(chunk, args.num_qubits)
        else:
            total += _hubbard_chunk(chunk, args.lx, args.ly)
    return len(keys), total, lists


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
        factory, circuit, steps = single.build_hubbard2d(
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
    for _ in range(args.layers):
        for step in steps:
            sim.propagate(step)
    size = sim.size()
    t0 = perf_counter()
    num_terms, total, lists = count(sim, args)
    gates = len(circuit.gates)
    layer_gates = [g for step in steps for g in step.gates]
    if len(layer_gates) != gates:
        raise RuntimeError(f"steps hold {len(layer_gates)} gates, the layer {gates}")
    fold = np.asarray(
        read_weights.weights(layer_gates, args.num_qubits, args.basis == "pauli")[
            "fold"
        ]
    )
    # The exported operator holds every term but the identity, which sim.size() may count.
    if num_terms not in (size, size - 1):
        raise RuntimeError(f"exported {num_terms} terms, propagator holds {size}")
    record = {
        "engine": "count",
        "basis": args.basis,
        "model": "kicked_ising" if args.basis == "pauli" else "hubbard2d",
        "lattice": [args.lx, args.ly] if args.basis == "majorana" else None,
        "num_qubits": args.num_qubits,
        "cutoff": args.cutoff,
        "cutoff_type": args.cutoff_type if args.basis == "majorana" else None,
        "boundary": args.boundary if args.basis == "majorana" else None,
        "layers": args.layers,
        "lower_atol": 0.0,
        "num_terms": int(size),
        "gates_per_layer": gates,
        "anticommuting_per_gate": total / gates,
        "index_reads_per_gate": int(fold @ lists) / gates,
        "count_seconds": perf_counter() - t0,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "host": socket.gethostname(),
        "monoprop_version": monoprop_version,
    }
    print(
        f"[count/{args.basis}] N={args.num_qubits} L={args.layers} K={size} "
        f"anti/gate={record['anticommuting_per_gate']:.4g} "
        f"({record['anticommuting_per_gate'] / size:.3g} of K) "
        f"reads/gate={record['index_reads_per_gate'] / size:.3g} of K t={record['count_seconds']:.1f}s"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
