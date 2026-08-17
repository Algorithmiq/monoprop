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

"""Fingerprint an evolved operator bit-exactly, for comparing two arms of an env knob.

WHY BIT-IDENTITY AND NOT A TOLERANCE. This was written for ``monoprop_COMPACT_RECORD``,
which changed how a query was *encoded* on the wire while changing no routing decision and
no record order -- and record order is the floating-point accumulation order, since
``Resolve.h`` mints each miss's term index in ``(sender, record)`` order. So its two arms
had to agree bit for bit, and agreement at 1e-7 would have been a *failure*: it would mean
something reordered, and an unintended reordering is one nobody has reasoned about. That
knob is gone (the compact record is now the only format), but the standard is not specific
to it: it is the right bar for any change that reorganises how a term is carried or handled
without changing which term it is, or in which order.

WHY THE OPERATOR AND NOT THE ENERGY. The energy is a single scalar and, at some
configurations of the random problem, it is exactly ``0.0`` -- at which point the two arms
agree trivially and the comparison proves nothing. That is the same failure shape as an
instrument that never fired. The fingerprint here is therefore the whole evolved operator:
every term, in enumeration order, with both halves of its coefficient hashed as raw IEEE-754
bits. It is sensitive to the term set, to the coefficients, and to the order.

Per-rank digests are reported separately and are NOT merged. Under MPI the operator is
partitioned, so a change that merely moved terms between ranks would leave a merged digest
alone; comparing the ranks elementwise is the stronger check, and this knob must not move a
single term.

The knob is read once into a cached ``Settings`` singleton, so the arms cannot share a
process. Run this twice under different environments and diff the two outputs.

    srun ... python hpc/deucalion/tools/bitident.py --mode graph --out arm_on.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benches"))

import numpy as np  # noqa: E402

from _builders import (  # noqa: E402
    KickedIsingConfig,
    build_kicked_ising_problem,
    build_random_propagator,
    make_random_problem,
)


def _bits(x: float) -> str:
    """Return the IEEE-754 double's 64 bits as hex, for exact string comparison."""
    return "0x" + struct.pack(">d", float(x)).hex()


def _fingerprint(terms) -> tuple[str, int]:
    """Digest a ``{key: coefficient}`` mapping in ITERATION order, coefficients bit-exact.

    ``str(key)`` rather than the key itself because the two operator families hand back
    different key types -- a tuple of Majorana indices, or a ``Pauli`` -- and both have a
    deterministic, injective string form. Hashing floats through ``repr`` would round-trip
    correctly but hides which of two arms produced a subnormal; the raw bits do not.
    """
    h = hashlib.sha256()
    n = 0
    for key, coeff in terms.items():
        c = complex(coeff)
        h.update(str(key).encode())
        h.update(b"\x00")
        h.update(struct.pack(">dd", c.real, c.imag))
        n += 1
    return h.hexdigest(), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-modes", type=int, default=250)
    ap.add_argument("--cutoff", type=int, default=6)
    ap.add_argument("--obs-terms", type=int, default=250000)
    ap.add_argument("--num-generators", type=int, default=100)
    ap.add_argument("--gen-length", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument(
        "--picture", choices=["heisenberg", "schrodinger"], default="heisenberg"
    )
    # Which sink the layer build runs under. `graph` -> GraphSink (plain incoming layout);
    # `propagate` -> ContractSink, whose incoming layout is the FUSED one. They are
    # different codec paths, so covering only one proves nothing about the other -- and the
    # fused path is the one where a mis-strided read is a sign flip on a coefficient rather
    # than a crash. `pauli` additionally exercises the Pauli rotation sign, which unlike
    # Majorana's needs the product monomial.
    ap.add_argument("--mode", choices=["graph", "propagate", "pauli"], default="graph")
    ap.add_argument("--pauli-layers", type=int, default=0, help="0 = the config default")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    comm = None
    rank, size = 0, 1
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except ImportError:
        pass

    gradient = None
    if args.mode == "pauli":
        config = KickedIsingConfig()
        if args.pauli_layers:
            config = dataclasses.replace(config, num_layers=args.pauli_layers)
        propagator, circuit = build_kicked_ising_problem(config, comm=comm)
        propagator.propagate(circuit)
        value = propagator.expectation_value()
        terms = propagator.evolved_operator(atol=0.0).terms
    else:
        problem = make_random_problem(
            gen_length=args.gen_length,
            obs_terms=args.obs_terms,
            num_generators=args.num_generators,
            num_modes=args.num_modes,
            cutoff=args.cutoff,
            seed=args.seed,
        )
        propagator, circuit = build_random_propagator(
            problem, comm=comm, schrodinger=args.picture == "schrodinger"
        )
        params = np.asarray(problem.parameters, dtype=float)
        if args.mode == "propagate":
            propagator.propagate(circuit)
            value = propagator.expectation_value()
            terms = propagator.evolved_operator(atol=0.0).terms
        else:
            propagator.build_graph(circuit)
            value, gradient = propagator.expectation_value_and_gradient_functional()(
                params
            )
            gradient = np.asarray(gradient, dtype=float)
            terms = propagator.evolved_operator(params, atol=0.0).terms

    digest, nterms = _fingerprint(terms)
    del terms

    # Per-rank term counts are part of the fingerprint: a change that moved terms between
    # ranks while leaving the total alone would not show up in the total.
    local = (propagator.size(), nterms, digest)
    allrows = comm.allgather(local) if comm is not None else [local]

    lines = [
        "mode %s" % args.mode,
        "ranks %d" % size,
        "size_total %d" % sum(r[0] for r in allrows),
        "terms_total %d" % sum(r[1] for r in allrows),
        "energy %s %s" % (_bits(value), float(value).hex()),
    ]
    for i, (sz, nt, dg) in enumerate(allrows):
        lines.append("rank[%d] size=%d terms=%d sha256=%s" % (i, sz, nt, dg))
    if gradient is not None:
        lines.append("grad_len %d" % len(gradient))
        for i, g in enumerate(gradient):
            lines.append("grad[%d] %s" % (i, _bits(g)))

    # A fingerprint of a zero operator is a fingerprint of nothing, and it agrees with
    # itself across any two arms. Refusing here is what keeps "the arms match" from being
    # reported when there was nothing to match.
    if sum(r[1] for r in allrows) == 0:
        sys.stderr.write("bitident: the evolved operator is EMPTY -- nothing compared\n")
        return 2

    text = "\n".join(lines) + "\n"
    if rank == 0:
        sys.stdout.write(text)
        if args.out:
            pathlib.Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
