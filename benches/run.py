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

"""Run the monoprop benchmark suite and write a combined Markdown report.

This is the single entry point for the benchmarks. For one run *label* it runs a
single sweep recording both timing (``pytest-benchmark``) and peak physical
memory (PSS, recorded by the suite itself at no measurable cost), records the run
configuration, then regenerates ``REPORT.md``. Run it again with a different
configuration (more ranks, different thread count, …) and the new label is added
as another column so configurations sit side by side.

Configuration is deliberately external and explicit (KISS):

- **MPI** — ``--ranks N`` runs under ``mpiexec -n N``; ``--mpiexec-args`` passes
  anything else (core pinning via ``--bind-to core``, ``--map-by socket``, …).
- **Threads** — ``--env monoprop_NUM_THREADS=K`` sets monoprop's oneTBB worker
  count; it is the only thread variable monoprop reads (at import). Per-process
  pinning is via ``mpiexec --bind-to`` (see MPI above).
- **Label** — ``--label`` names the run (default ``np<ranks>``); use a custom
  label to compare thread/pinning variants at the same rank count.

Examples::

    uv run --group bench python benches/run.py                       # serial (np1)
    uv run --group bench python benches/run.py --ranks 4             # 4 ranks (np4)
    uv run --group bench python benches/run.py \
        --ranks 4 --mpiexec-args="--bind-to core" \
        --env monoprop_NUM_THREADS=2 \
        --label r4t2-pinned                                          # custom config
    uv run --group bench python benches/run.py --num-modes 64 --bench-rounds 10

Anything not recognised below is forwarded verbatim to pytest.
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import report

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"

# Environment variable recorded (if set) in each run's metadata for provenance.
# monoprop_NUM_THREADS is the only knob that affects monoprop (its oneTBB worker
# count); it is read at import.
TRACKED_ENV = ("monoprop_NUM_THREADS",)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split known driver options from arguments forwarded to pytest."""
    parser = argparse.ArgumentParser(
        prog="benches/run.py",
        description=__doc__,
        add_help=True,
        allow_abbrev=False,  # so forwarded pytest options are never partially matched
    )
    parser.add_argument(
        "--ranks", type=int, default=1, help="MPI ranks (1 = serial; default 1)."
    )
    parser.add_argument(
        "--mpiexec-args",
        default="",
        help="Extra arguments passed to mpiexec (e.g. '--bind-to core').",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for the run (repeatable), e.g. monoprop_NUM_THREADS=2.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Run label / report column (default: np<ranks>).",
    )
    parser.add_argument(
        "--skip-mpi-check",
        action="store_true",
        help="Skip the preflight that verifies MPI distribution (ranks > 1).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="Output directory."
    )
    return parser.parse_known_args(argv)


def _parse_env(pairs: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings into a dict, erroring on a missing ``=``."""
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            msg = f"--env expects KEY=VALUE, got {pair!r}"
            raise SystemExit(msg)
        key, value = pair.split("=", 1)
        env[key] = value
    return env


def _monoprop_version() -> str:
    """Return the installed monoprop version, or 'unknown'."""
    try:
        return version("monoprop")
    except PackageNotFoundError:
        return "unknown"


def _write_metadata(
    results_dir: Path,
    label: str,
    args: argparse.Namespace,
    launcher: list[str],
    overrides: dict[str, str],
    run_env: dict[str, str],
    pytest_args: list[str],
) -> None:
    """Record how this run was configured so the report can show it."""
    tracked = {k: run_env[k] for k in TRACKED_ENV if k in run_env}
    tracked.update(overrides)  # always show what the user explicitly set
    meta = {
        "label": label,
        "ranks": args.ranks,
        "launcher": shlex.join(launcher) if launcher else "serial",
        "env": tracked,
        "pytest_args": pytest_args,
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "monoprop_version": _monoprop_version(),
    }
    report.write_json(results_dir / f"meta-{label}.json", meta)


def _launch(prefix: list[str], pytest_args: list[str], run_env: dict[str, str]) -> None:
    """Run pytest (optionally under mpiexec) and raise if it fails."""
    cmd = [*prefix, sys.executable, "-m", "pytest", str(BENCH_DIR), *pytest_args]
    print(f"$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=run_env)


def _mpi_build_ok(run_env: dict[str, str]) -> bool:
    """Check that the loaded ``monoprop`` extension was built with MPI.

    Catches the common failure where the loaded ``monoprop`` extension was built
    without MPI (so every rank holds the full operator and the run OOMs). The
    extension reports this via ``monoprop.has_mpi``, so a single import suffices --
    no ``mpiexec`` and no probe problem needed.
    """
    cmd = [sys.executable, str(BENCH_DIR / "_mpi_check.py")]
    print(f"$ {shlex.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False, env=run_env).returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Run the timing and memory passes for one label, then write the report."""
    args, pytest_args = _parse_args(sys.argv[1:] if argv is None else argv)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or f"np{args.ranks}"
    overrides = _parse_env(args.env)
    run_env = {**os.environ, **overrides}
    # Let the pytest session (conftest) record the resolved hyperparameters and
    # operator sizes for this label, keyed and located by these variables.
    run_env["MONOPROP_BENCH_LABEL"] = label
    run_env["MONOPROP_BENCH_RESULTS"] = str(results_dir)

    prefix: list[str] = []
    if args.ranks > 1:
        # Forward overridden vars and the recording variables to the ranks
        # (OpenMPI), then add user mpiexec args.
        forwarded_keys = (*overrides, "MONOPROP_BENCH_LABEL", "MONOPROP_BENCH_RESULTS")
        forwards = [tok for key in forwarded_keys for tok in ("-x", key)]
        prefix = [
            "mpiexec",
            "--allow-run-as-root",
            "-n",
            str(args.ranks),
            *forwards,
            *shlex.split(args.mpiexec_args),
        ]

    # Fail fast if an MPI run would not actually distribute (e.g. a non-MPI build
    # got loaded): otherwise every rank holds the full operator and the run OOMs.
    if args.ranks > 1 and not args.skip_mpi_check and not _mpi_build_ok(run_env):
        sys.stderr.write(
            "ERROR: MPI build preflight failed (see [mpi-check] above). "
            "Aborting before the heavy passes. Rebuild the MPI extension with "
            "`just bench-build-mpi`, or pass --skip-mpi-check to override.\n"
        )
        return 1

    # ``-o filterwarnings=default`` overrides the project-wide ``error`` filter so
    # benchmark-plugin warnings do not fail the run.
    common = ["-o", "filterwarnings=default", *pytest_args]

    _write_metadata(results_dir, label, args, prefix, overrides, run_env, pytest_args)

    sweep_args = [*common, "--benchmark-json", str(results_dir / f"time-{label}.json")]
    _launch(prefix, sweep_args, run_env)

    report.main([str(results_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
