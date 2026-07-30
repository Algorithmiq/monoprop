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

"""Sweep the TFIM lattice size and record each backend's total time and final memory.

Default ladder: square grids L x L for L = 6, 8, ... 18 — the side grows by two qubits
per point, 36 to 324 qubits.

Each (size, backend) point runs as its own subprocess under a wall-clock budget. A point
that exceeds the budget, crashes, or is OOM-killed is recorded with a non-ok status and
the sweep continues; by default a backend is then skipped at every larger size, since
cost is monotone in size (pass --no-skip-after-fail to keep trying).

    python run_scaling.py --output results/scaling.jsonl --timeout 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import backends as backend_mod

HERE = Path(__file__).parent


def _record_failure(
    output: Path,
    backend: str,
    nx: int,
    ny: int,
    status: str,
    elapsed: float,
    detail: str,
) -> None:
    """Write a placeholder so a DNF point is visible in the data, not just absent."""
    record = {
        "backend": backend,
        "label": backend_mod.LABELS[backend],
        "nx": nx,
        "ny": ny,
        "num_qubits": nx * ny,
        "status": status,
        "elapsed_s": elapsed,
        "detail": detail,
    }
    with output.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _final_terms(output: Path, backend: str, nx: int, ny: int) -> int | None:
    """monoprop's final term count for this size, if it completed."""
    if not output.exists():
        return None
    with output.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (
                r.get("backend") == backend
                and r.get("nx") == nx
                and r.get("ny") == ny
                and r.get("status") == "ok"
            ):
                return r["final_num_terms"]
    return None


def _parse_threads(pairs: list[str]) -> list[tuple[str, int]]:
    out = []
    for pair in pairs:
        backend, _, count = pair.partition("=")
        if not count.isdigit() or int(count) < 1:
            raise SystemExit(f"--threads expects BACKEND=N with N >= 1, got {pair!r}")
        out.append((backend, int(count)))
    return out


def _child_env(backend: str, cap: int | None) -> dict[str, str] | None:
    """The environment for one backend's subprocess, or None to inherit unchanged."""
    if cap is None:
        return None
    env = dict(os.environ)
    for var in backend_mod.THREAD_VARS.get(backend, ()):
        env[var] = str(cap)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(range(6, 19, 2)),
        help="Side lengths L of the L x L grids to run (default: 6 8 10 12 14 16 18).",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=None,
        help="Fix the second dimension instead of sweeping square grids, so --sizes "
        "becomes nx of an nx x NY strip.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(backend_mod.ALL_BACKENDS),
        choices=list(backend_mod.ALL_BACKENDS),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Wall-clock budget per (size, backend) point, in seconds.",
    )
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument(
        "--no-skip-after-fail",
        action="store_true",
        help="Keep running a backend at larger sizes after it fails one.",
    )
    parser.add_argument(
        "--qiskit-max-terms",
        type=int,
        default=None,
        help="Fallback term budget for Qiskit when monoprop has no result at that size.",
    )
    parser.add_argument("--julia-bin", default=os.environ.get("JULIA_BIN", "julia"))
    parser.add_argument(
        "--julia-project",
        default=os.environ.get("JULIA_PROJ", str(HERE)),
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        default=[],
        metavar="BACKEND=N",
        help="Per-backend thread cap, e.g. --threads monoprop=56 juliapp=28. Throughput is "
        "not monotone in thread count for every backend, so the fair comparison runs each "
        "at its own measured optimum rather than at the node's core count; repeat a small "
        "sweep at a few values to find it.",
    )
    args = parser.parse_args()

    thread_caps = dict(_parse_threads(args.threads))
    unknown = set(thread_caps) - set(backend_mod.ALL_BACKENDS)
    if unknown:
        parser.error(f"--threads names unknown backend(s): {sorted(unknown)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # monoprop first at each size: the Qiskit backend needs its term count as a budget.
    ordered = sorted(args.backends, key=lambda b: (b != "monoprop", b))
    failed: set[str] = set()

    caps = ", ".join(f"{b}={n}" for b, n in sorted(thread_caps.items())) or "none"
    print(
        f"sweep: sizes={args.sizes} ny={args.ny or 'L (square)'} "
        f"backends={ordered} timeout={args.timeout:g}s thread-caps: {caps} "
        f"-> {args.output}",
        flush=True,
    )

    for size in args.sizes:
        nx, ny = size, args.ny or size
        for backend in ordered:
            if backend in failed and not args.no_skip_after_fail:
                print(
                    f"SKIP {backend} at {nx}x{ny}: it already failed at a smaller size",
                    flush=True,
                )
                continue

            if backend == backend_mod.JULIA_BACKEND:
                cmd = [
                    args.julia_bin,
                    f"--project={args.julia_project}",
                    str(HERE / "run_scaling.jl"),
                    "--nx",
                    str(nx),
                    "--ny",
                    str(ny),
                    "--output",
                    str(args.output),
                ]
                if args.step_max is not None:
                    cmd += ["--step-max", str(args.step_max)]
            else:
                cmd = [
                    sys.executable,
                    str(HERE / "run_one.py"),
                    "--backend",
                    backend,
                    "--nx",
                    str(nx),
                    "--ny",
                    str(ny),
                    "--output",
                    str(args.output),
                ]
                if args.step_max is not None:
                    cmd += ["--step-max", str(args.step_max)]
                if backend == "qiskit":
                    budget = (
                        _final_terms(args.output, "monoprop", nx, ny)
                        or args.qiskit_max_terms
                    )
                    if budget is None:
                        print(
                            f"SKIP qiskit at {nx}x{ny}: no monoprop term count for this "
                            "size and no --qiskit-max-terms fallback",
                            flush=True,
                        )
                        continue
                    cmd += ["--max-terms", str(budget)]

            cap = thread_caps.get(backend)
            suffix = f", {cap} threads" if cap else ""
            print(
                f"\n=== {backend} at {nx}x{ny} ({nx * ny} qubits{suffix}) ===",
                flush=True,
            )
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=HERE,
                    timeout=args.timeout,
                    check=False,
                    env=_child_env(backend, cap),
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                print(f"TIMEOUT after {elapsed:.0f}s", flush=True)
                _record_failure(
                    args.output,
                    backend,
                    nx,
                    ny,
                    "timeout",
                    elapsed,
                    f"exceeded {args.timeout:g}s",
                )
                failed.add(backend)
                continue
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                # Negative return codes are signals: -9 is the OOM killer or Slurm.
                status = "killed" if proc.returncode < 0 else "error"
                print(
                    f"{status.upper()} (rc={proc.returncode}) after {elapsed:.0f}s",
                    flush=True,
                )
                _record_failure(
                    args.output,
                    backend,
                    nx,
                    ny,
                    status,
                    elapsed,
                    f"return code {proc.returncode}",
                )
                failed.add(backend)

    print(f"\nsweep done -> {args.output}", flush=True)
    if failed:
        print(f"backends that did not finish every size: {sorted(failed)}", flush=True)


if __name__ == "__main__":
    main()
