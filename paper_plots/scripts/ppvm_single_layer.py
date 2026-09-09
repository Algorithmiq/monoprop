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

"""Single-layer scaling benchmark for QuEra's ppvm, the third engine in the comparison.

The counterpart of ``monoprop_single_layer.py`` and ``julia_pauli_single_layer.jl``: same
kicked-Ising chain, same extensive observable, same support cutoff, same record schema, so
the three files can be plotted against each other without a translation step.

ppvm propagates through a serial indexmap ``PauliSum`` and has no parallel path, so it is a
single-thread point by construction; ``RAYON_NUM_THREADS=1`` only bounds an idle pool.

Angle convention: ppvm's ``rx(q, t)`` and ``rzz(i, k, t)`` take the standard rotation angle,
``exp(-i t/2 P)`` -- verified against ``cos(t)`` on the Heisenberg-evolved ``Z``. That is
PauliPropagation.jl's convention too, so both take ``theta`` directly where monoprop's
``ExpGate`` needs ``-theta/2``.

ppvm exposes no memory accounting of its own, so ``memory_bytes`` is the process high-water
mark measured across the timed call, which is not commensurable with monoprop's operator
accounting or Julia's ``Base.summarysize``. It is recorded for the trend, and the metric it
came from is recorded beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
from pathlib import Path
from time import perf_counter, process_time


def build_terms(active: int) -> list[str]:
    """The extensive observable sum_i Z_i over the active window, in ppvm's term syntax."""
    return [f"Z{i}" for i in range(active)]


def peak_rss_bytes() -> int:
    """Process high-water mark. Linux reports KiB, macOS bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if os.uname().sysname == "Darwin" else peak * 1024


def run(num_qubits: int, active: int, cutoff: int, lower_atol: float, layers: int):
    """Apply `layers` kicked-Ising layers and return (seconds, cpu, terms, expectation)."""
    from ppvm import PauliSum  # noqa: PLC0415

    theta = math.pi / 4
    coupling = math.pi / 4
    edges = [(i, i + 1) for i in range(active - 1)]

    pauli_sum = PauliSum.new(
        n_qubits=num_qubits,
        terms=build_terms(active),
        min_abs_coeff=lower_atol,
        max_pauli_weight=cutoff,
    )
    # Gate order is REVERSED against the circuit the other two drivers declare. They hand
    # `propagate` a circuit of [Rx layer, Rzz layer] and evolve in the Heisenberg picture,
    # which applies U's gates back to front, so the Rzz layer acts first. ppvm has no
    # circuit object -- each call conjugates the sum immediately -- so the reversal has to
    # be written out here. Getting it wrong is nearly silent: the expectation value agrees
    # to 15 digits either way for this observable, and only the term count gives it away
    # (12090 forward against the 10122 the other two engines both report at N=32, c4).
    c0, t0 = process_time(), perf_counter()
    for _ in range(layers):
        for i, k in edges:
            pauli_sum.rzz(i, k, coupling)
        for i in range(active):
            pauli_sum.rx(i, theta)
    seconds, cpu = perf_counter() - t0, process_time() - c0
    return seconds, cpu, len(pauli_sum), float(pauli_sum.overlap_with_zero())


def save_result(output_path, record) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument(
        "--layers", type=int, default=1, help="number of layers to apply (default 1)"
    )
    parser.add_argument("--lower-atol", type=float, default=1e-8)
    parser.add_argument(
        "--active-window",
        type=int,
        default=None,
        help="confine the model to qubits 0..M-1 and pad the register to --num-qubits with "
        "idle spectator qubits (default: the full width)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="fresh rebuild+apply repetitions; min time is kept",
    )
    parser.add_argument(
        "--out", "-o", required=True, help="JSONL results file (appended to)"
    )
    args = parser.parse_args()

    active = args.num_qubits if args.active_window is None else args.active_window
    if not 1 <= active <= args.num_qubits:
        parser.error(
            f"--active-window must be in 1..{args.num_qubits}, got {args.active_window}"
        )

    best = float("inf")
    best_cpu = float("nan")
    num_terms = 0
    expectation = float("nan")
    for _ in range(max(1, args.rounds)):
        seconds, cpu, num_terms, expectation = run(
            args.num_qubits, active, args.cutoff, args.lower_atol, args.layers
        )
        if seconds < best:
            best, best_cpu = seconds, cpu

    memory_bytes = peak_rss_bytes()
    gates = (active + max(0, active - 1)) * args.layers
    record = {
        "engine": "ppvm",
        "basis": "pauli",
        "num_qubits": args.num_qubits,
        "cutoff": args.cutoff,
        "observable": "extensive",
        "layers": args.layers,
        "lower_atol": args.lower_atol,
        "num_threads": os.environ.get("RAYON_NUM_THREADS", "not set"),
        "num_terms": int(num_terms),
        "memory_bytes": int(memory_bytes),
        "bytes_per_term": (memory_bytes / num_terms) if num_terms else 0.0,
        "memory_metric": "peak process RSS (not the operator; see the module docstring)",
        "seconds": best,
        "expectation": expectation,
        "active_window": args.active_window,
        "gates": gates,
        "cpu_seconds": best_cpu,
        "busy_cores": (best_cpu / best) if best else float("nan"),
        "host": socket.gethostname(),
        "library_version": _ppvm_version(),
    }
    print(
        f"[ppvm/pauli] N={args.num_qubits} cutoff={args.cutoff} terms={num_terms} "
        f"t={best:.4f}s busy={record['busy_cores']:.2f} exp={expectation:.6f}"
    )
    save_result(args.out, record)


def _ppvm_version() -> str:
    """Version plus the installed git commit, because the version alone is ambiguous.

    The engine is installed from QuEra's git pin and self-reports "0.1.0" -- which is also
    the version of an unrelated `ppvm` on PyPI (a Windows Python version manager) that
    installs cleanly under the same name. A record saying only "0.1.0" therefore does not
    identify what ran, so append the commit pip recorded for the install.
    """
    import json  # noqa: PLC0415
    from importlib.metadata import PackageNotFoundError, distribution  # noqa: PLC0415

    try:
        dist = distribution("ppvm")
    except PackageNotFoundError:
        return "unknown"
    ver = dist.version
    try:
        raw = dist.read_text("direct_url.json")
        commit = json.loads(raw)["vcs_info"]["commit_id"] if raw else None
    except (OSError, ValueError, KeyError, TypeError):
        commit = None
    return f"{ver}+git.{commit[:12]}" if commit else ver


if __name__ == "__main__":
    main()
