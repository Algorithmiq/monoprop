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

"""Count how often one layer of each Fig. 3 circuit reads each index list.

monoprop applies a gate one generator monomial at a time, and each monomial G scans the index
by XOR-folding whole lists: the lists of G's own positions in the Majorana basis, and of
J(G), each slot swapped with its qubit partner (X_q <-> Y_q), in the Pauli one. The
scan also reads the pivot list (G's highest slot) at the words where the fold is nonzero, so
only partly. This writes, per model and N, how many scans of one layer fold each list in
full (``fold``) and take it as pivot (``pivot``), from the gates the campaign builds (the open Hubbard lattice).

    python scripts/read_weights.py -o data/read_weights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import monoprop_single_layer as single
import numpy as np
from monoprop.circuit import _gate_layers

NS = (32, 64, 128, 256, 512, 1024)
# The square Hubbard lattices of merge_octave_runs.HUBBARD_NS.
LATTICE = {
    32: (4, 4),
    72: (6, 6),
    128: (8, 8),
    242: (11, 11),
    512: (16, 16),
    968: (22, 22),
}


def weights(gates, num_qubits: int, pauli: bool) -> dict:
    fold = np.zeros(2 * num_qubits, dtype=np.int64)
    pivot = np.zeros(2 * num_qubits, dtype=np.int64)
    scans = 0
    for gate in gates:
        for monomial, _ in _gate_layers(gate, num_qubits):
            if not monomial:
                continue  # the identity term of a gate is skipped, with no scan
            scans += 1
            for s in monomial:
                fold[s ^ 1 if pauli else s] += 1
            pivot[max(monomial)] += 1
    return {"scans": scans, "fold": fold.tolist(), "pivot": pivot.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "-o", required=True)
    args = ap.parse_args()
    out = {}
    for n in NS:
        _, circuit = single.build_pauli(n, 6, 0.0, "extensive")
        out[f"pauli:{n}"] = weights(circuit.gates, n, pauli=True)
    for n, (lx, ly) in LATTICE.items():
        _, _, steps = single.build_hubbard2d(
            lx, ly, 4, cutoff_type="length", boundary="open"
        )
        out[f"majorana:{n}"] = weights(
            [g for step in steps for g in step.gates], n, pauli=False
        )
    for key, w in out.items():
        fold = np.asarray(w["fold"])
        print(
            f"{key}: {w['scans']} scans, fold reads per list "
            f"{fold.min()}-{fold.max()} (mean {fold.mean():.2f})"
        )
    Path(args.out).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
