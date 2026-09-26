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

"""Tests for the rank-local OpenMP parity evidence: benchmark overlay and parity driver.

The benchmark-suite tests launch ``pytest benches`` at tiny sizes in fresh processes; they check
the evidence fields the suite writes, not performance. Everything else exercises
``tools/benchmark-rank-local-openmp.py`` on synthetic artifacts that follow the real schema. None
of it is runtime evidence or a parity claim.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import monoprop

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "tools/benchmark-rank-local-openmp.py"

# Tiny sizes for every benchmark family; the fixed models are shrunk through their own overrides.
_TINY = [
    "--num-generators=8",
    "--num-modes=8",
    "--cutoff=6",
    "--obs-terms=16",
    "--bench-rounds=1",
    "--hubbard-num-sites=4",
    "--hubbard-observable-site=1",
    "--hubbard-trotter-steps=2",
    "--pauli-num-layers=1",
    "--pauli-cutoff=2",
    "--pauli-lower-atol=0.1",
]
_FAMILIES = ("build_graph", "propagate", "energy", "gradient")

# These tests launch their own processes (pytest, the driver, mpiexec). Inside a multi-rank MPI test
# job every rank would repeat those launches nested in the parent job, so they run outside one.
_IN_MPI_JOB = (
    int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))) > 1
)
_launches = pytest.mark.skipif(
    _IN_MPI_JOB, reason="launches its own processes; run outside a multi-rank MPI job"
)


def _bench_env(**extra: str) -> dict[str, str]:
    """Return a clean launch environment for a benchmark subprocess."""
    # Launcher variables are dropped so a child never joins an enclosing MPI job.
    dropped = ("monoprop_", "OMP_", "PYTEST_", "OMPI_", "PMIX_", "PRTE_", "PMI_")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(dropped)
    }
    env.update(extra)
    return env


def _run_benches(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``pytest benches`` in a fresh process from the repository root."""
    return subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
        [sys.executable, "-m", "pytest", "benches", "-p", "no:cacheprovider", *args],
        cwd=_ROOT,
        env=env if env is not None else _bench_env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _needs_bench_tools() -> None:
    pytest.importorskip("monoprop_bench_tools")
    pytest.importorskip("pytest_benchmark")


@_launches
def test_suite_records_independent_exactness_for_all_four_families(
    tmp_path: Path,
) -> None:
    _needs_bench_tools()
    results = tmp_path / "results"
    results.mkdir()
    env = _bench_env(
        monoprop_BENCH_LABEL="tiny",
        monoprop_BENCH_RESULTS=str(results),
        monoprop_PARTITIONS="1",
        monoprop_NUM_THREADS="1",
    )
    proc = _run_benches(
        *_TINY,
        f"--benchmark-json={results / 'time-tiny.json'}",
        env=env,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    data = json.loads((results / "tiny.json").read_text())

    nodes = [
        f"bench_random.py::test_random_{family}[{picture}]"
        for family in _FAMILIES
        for picture in ("heisenberg", "schrodinger")
    ] + [
        f"bench_models.py::test_model_{family}[{model}]"
        for family in _FAMILIES
        for model in ("hubbard", "pauli")
    ]
    for node in nodes:
        for section in ("opmempeak", "opmembase", "opmemdelta"):
            assert set(data[section][node]) == {"sum", "max"}, (section, node)
        assert data["opmemexact"][node] is True, node
        assert data["memhwmexact"][node] is True, node
        assert data["memhwm"][node] >= data["opmempeak"][node]["max"] > 0, node
    assert data["meta"]["has_mpi"] is monoprop.has_mpi
    assert len(data["meta"]["monoprop_core_sha256"]) == 64


def _mpi4py_trap(tmp_path: Path) -> tuple[Path, Path]:
    """Return a PYTHONPATH entry whose ``mpi4py`` records any import, and that marker file."""
    marker = tmp_path / "mpi4py-imported"
    package = tmp_path / "trap" / "mpi4py"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "raise RuntimeError('mpi4py must not be imported in an MPI-off run')\n"
    )
    return package.parent, marker


@_launches
@pytest.mark.skipif(monoprop.has_mpi, reason="needs an MPI-off extension")
def test_mpi_off_suite_never_imports_mpi4py(tmp_path: Path) -> None:
    _needs_bench_tools()
    trap, marker = _mpi4py_trap(tmp_path)
    env = _bench_env(PYTHONPATH=os.pathsep.join([str(trap), *sys.path]))
    proc = _run_benches("--collect-only", "-q", *_TINY, env=env)

    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert not marker.exists()


@_launches
@pytest.mark.skipif(monoprop.has_mpi, reason="needs an MPI-off extension")
@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (["--runtime-shape=partitions"], {"monoprop_NUM_THREADS": "2"}, "partitions"),
        (
            ["--runtime-shape=openmp"],
            {"monoprop_NUM_THREADS": "2", "monoprop_PARTITIONS": "2"},
            "monoprop_PARTITIONS",
        ),
        (["--runtime-shape=openmp"], {"monoprop_NUM_THREADS": "0"}, "positive"),
        (
            ["--runtime-shape=partitions", "--build-mode=mpi"],
            {"monoprop_PARTITIONS": "2", "monoprop_NUM_THREADS": "2"},
            "has_mpi",
        ),
        (["--build-mode=mpi"], {}, "has_mpi"),
    ],
)
def test_suite_rejects_contradictory_declarations(
    args: list[str], env: dict[str, str], message: str
) -> None:
    _needs_bench_tools()
    proc = _run_benches("--collect-only", "-q", *args, env=_bench_env(**env))

    assert proc.returncode != 0
    assert message in proc.stdout + proc.stderr


@_launches
@pytest.mark.skipif(monoprop.has_mpi, reason="needs an MPI-off extension")
@pytest.mark.parametrize(
    ("args", "env"),
    [
        (
            ["--runtime-shape=partitions", "--build-mode=mpi-off"],
            {"monoprop_PARTITIONS": "2", "monoprop_NUM_THREADS": "2"},
        ),
        (["--runtime-shape=openmp"], {"monoprop_NUM_THREADS": "2"}),
        (["--runtime-shape=openmp"], {"OMP_NUM_THREADS": "2"}),
        (["--build-mode=mpi-off"], {}),
        ([], {}),
    ],
)
def test_suite_accepts_valid_declarations(args: list[str], env: dict[str, str]) -> None:
    _needs_bench_tools()
    proc = _run_benches("--collect-only", "-q", *args, env=_bench_env(**env))

    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


@_launches
@pytest.mark.skipif(
    not monoprop.has_mpi or shutil.which("mpiexec") is None,
    reason="needs an MPI extension and mpiexec",
)
@pytest.mark.parametrize(
    ("args", "env", "ok"),
    [
        # The legacy guard alone would reject this candidate shape above one rank.
        (["--runtime-shape=openmp"], {"monoprop_NUM_THREADS": "1"}, True),
        (
            ["--runtime-shape=partitions"],
            {"monoprop_PARTITIONS": "1", "monoprop_NUM_THREADS": "1"},
            True,
        ),
        (["--runtime-shape=partitions"], {"monoprop_NUM_THREADS": "1"}, False),
        ([], {}, False),
    ],
)
def test_multi_rank_shape_preflight(
    args: list[str],
    env: dict[str, str],
    ok: bool,  # noqa: FBT001 - parametrized
) -> None:
    _needs_bench_tools()
    exported = [flag for key in env for flag in ("-x", key)]
    proc = subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
        [
            shutil.which("mpiexec"),
            "--map-by",
            ":OVERSUBSCRIBE",
            "-n",
            "2",
            *exported,
            sys.executable,
            "-m",
            "pytest",
            "benches",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            *args,
        ],
        cwd=_ROOT,
        env=_bench_env(**env),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert (proc.returncode == 0) is ok, proc.stdout[-4000:] + proc.stderr[-4000:]


# --------------------------------------------------------------------------------- comparator
#
# The plan's miniature inventory: two empty-map energy fixtures following the real schema,
# reference loading and provenance checks. No synthetic-mode bypass exists in the driver.


@_launches
def test_comparison_does_not_hide_one_regression(tmp_path):
    pytest.importorskip("monoprop_bench_tools")
    script = (
        Path(__file__).resolve().parents[1] / "tools/benchmark-rank-local-openmp.py"
    )

    def encoded(value):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()

    def digest(value):
        return hashlib.sha256(encoded(value)).hexdigest()

    def save(name, value):
        path = tmp_path / name
        path.write_bytes(encoded(value))
        return str(path)

    config = {
        "gen_length": 4,
        "obs_terms": 16,
        "num_generators": 8,
        "num_modes": 8,
        "cutoff": 6,
        "seed": 0,
        "lower_atol": None,
    }
    parameters = [0.0] * 8
    identity = {
        "has_mpi": False,
        "ranks": 1,
        "threads": 2,
        "index_bits": 32,
        "cpu_allocation": "test-cpus",
        "compiler_flags": "test",
        "config": config,
        "routing": {"mode": "linear", "bits": 0, "seed": 0},
    }
    expected, entries = [], {}
    for name in ("build", "replay"):
        node = f"energy-fixture-{name}"
        entries[node] = {
            "operation": "energy",
            "measurement_kind": "pytest",
            "basis": "majorana",
            "picture": "heisenberg",
            "config": config,
            "parameters": parameters,
            "pare_threshold": None,
        }
        expected.append(
            {
                "id": name,
                "profile": "tiny",
                "node_id": node,
                "operation": "energy",
                "measurement_kind": "pytest",
                "identity": identity,
                "config_digest": digest(identity["config"]),
                "parameters_digest": digest(parameters),
            }
        )
    workloads = {"schema_version": 1, "profiles": {"tiny": entries}}
    campaign = {
        "schema_version": 1,
        "workloads_path": save("workloads.json", workloads),
        "workloads_sha256": digest(workloads),
        "cells": expected,
    }
    campaign_path = save("campaign.json", campaign)
    cells = []
    for target, candidate_time in zip(expected, (10.1, 5.0), strict=True):
        name = target["id"]
        cell = {"id": name}
        for arm, elapsed in (("baseline", 10.0), ("candidate", candidate_time)):
            shape = "partitions" if arm == "baseline" else "openmp"
            binary_hash = ("b" if arm == "baseline" else "c") * 64
            settings = {
                "monoprop_NUM_THREADS": "2",
                "OMP_NUM_THREADS": "2",
                "OMP_DYNAMIC": "FALSE",
            }
            if arm == "baseline":
                settings.update(monoprop_PARTITIONS="2")
            placement = {
                "schema_version": 1,
                "artifact_kind": "placement",
                "arm": arm,
                "cell_id": name,
                "binary_hash": binary_hash,
                "identity": identity,
                "runtime_shape": shape,
                "declared": {
                    "ranks": 1,
                    "threads": 2,
                    "cpu_allocation": "test-cpus",
                    "settings": settings,
                },
                "observed": {
                    "has_mpi": False,
                    "ranks": 1,
                    "threads_per_rank": [2],
                    "cpu_allocation": "test-cpus",
                },
            }
            placement_path = save(f"{name}-{arm}-placement.json", placement)
            common = {
                "schema_version": 1,
                "campaign_sha256": digest(campaign),
                "arm": arm,
                "has_mpi": False,
                "cell_id": name,
                "profile": target["profile"],
                "node_id": target["node_id"],
                "operation": "energy",
                "binary_hash": binary_hash,
                "identity": identity,
                "config_digest": target["config_digest"],
                "parameters_digest": target["parameters_digest"],
                "placement_path": placement_path,
                "placement_sha256": digest(placement),
            }
            validation = {
                **common,
                "artifact_kind": "validation",
                "run_id": f"{name}-{arm}-validation",
                "basis": "majorana",
                "picture": "heisenberg",
                "energy": 0.0,
                "global_term_count": 0,
                "graph_layers": 0,
                "term_files": [],
            }
            validation_path = save(f"{name}-{arm}-validation.json", validation)
            cell[arm] = []
            for i in range(5):
                sample_id = f"{name}-{arm}-{i}"
                timed = {
                    **common,
                    "artifact_kind": "timed",
                    "sample_id": sample_id,
                    "run_id": f"{sample_id}-timed",
                    "runtime_seconds": elapsed,
                    "peak_sum_bytes": 1000,
                    "peak_max_bytes": 1000,
                    "op_exact": True,
                    "outer_peak_sum_bytes": 1200,
                    "outer_peak_max_bytes": 1200,
                    "outer_exact": True,
                    "operation_floor_sum_bytes": 800,
                    "operation_floor_max_bytes": 800,
                    "operation_delta_sum_bytes": 200,
                    "operation_delta_max_bytes": 200,
                }
                construction = {
                    **common,
                    "artifact_kind": "construction",
                    "sample_id": sample_id,
                    "run_id": f"{sample_id}-construction",
                    "construction_exact": True,
                    "construction_peak_sum_bytes": 1200,
                    "construction_peak_max_bytes": 1200,
                    "construction_floor_sum_bytes": 800,
                    "construction_floor_max_bytes": 800,
                    "construction_delta_sum_bytes": 400,
                    "construction_delta_max_bytes": 400,
                }
                cell[arm].append(
                    {
                        "sample_id": sample_id,
                        "timed_path": save(f"{sample_id}-timed.json", timed),
                        "construction_path": save(
                            f"{sample_id}-construction.json", construction
                        ),
                        "validation_path": validation_path,
                    }
                )
        cells.append(cell)
    manifest = save(
        "manifest.json",
        {"schema_version": 1, "campaign_sha256": digest(campaign), "cells": cells},
    )
    output = tmp_path / "comparison.json"
    result = subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
        [
            sys.executable,
            str(script),
            "compare",
            "--campaign",
            campaign_path,
            "--manifest",
            manifest,
            "--output",
            str(output),
        ],
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text())["failed_cells"] == ["build"]


# ------------------------------------------------------------------- evidence builder (negatives)


def _driver():
    """Import the driver script as a module (it is a tool, not a package)."""
    pytest.importorskip("monoprop_bench_tools")
    spec = importlib.util.spec_from_file_location("rank_local_driver", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _encoded(value, *, allow_nan: bool = False) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=allow_nan
    ).encode()


def _digest(value) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


_RANDOM = {
    "gen_length": 4,
    "obs_terms": 16,
    "num_generators": 8,
    "num_modes": 8,
    "cutoff": 6,
    "seed": 0,
    "lower_atol": None,
}


def _write_gz(path: Path, text: str) -> None:
    """Write ``text`` as a gzip member, as the driver writes term files."""
    with gzip.open(path, "wt") as handle:
        handle.write(text)


class Evidence:
    """Writes a complete, valid campaign/manifest; tests then break one thing at a time."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cells: list[dict] = []
        self.entries: dict[str, dict] = {"tiny": {}, "large": {}}

    def save(self, name: str, value) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Finite documents encode exactly as their canonical digest; non-finite ones exist only
        # to be rejected.
        path.write_bytes(_encoded(value, allow_nan=True))
        return str(path)

    def add_cell(
        self,
        cell_id: str,
        *,
        profile: str = "tiny",
        node: str = "bench_random.py::test_random_energy[heisenberg]",
        operation: str = "energy",
        has_mpi: bool = False,
        ranks: int = 1,
        threads: int = 2,
        allocation: str = "0-1",
        routing: dict | None = None,
        threshold: float | None = None,
        kind: str = "pytest",
    ) -> dict:
        entry = {
            "operation": operation,
            "measurement_kind": kind,
            "basis": "majorana",
            "picture": "heisenberg",
            "config": _RANDOM,
            "parameters": [0.5] * 8,
            "pare_threshold": threshold,
        }
        if operation == "pare_functional_construct":
            entry["functional"] = "expectation_value_and_gradient_functional"
        self.entries[profile][node] = entry
        identity = {
            "has_mpi": has_mpi,
            "ranks": ranks,
            "threads": threads,
            "index_bits": 32,
            "cpu_allocation": allocation,
            "compiler_flags": "-O3",
            "config": _RANDOM,
            "routing": routing
            or {"mode": "linear", "bits": ranks.bit_length() - 1, "seed": 7},
        }
        cell = {
            "id": cell_id,
            "profile": profile,
            "node_id": node,
            "operation": operation,
            "measurement_kind": kind,
            "identity": identity,
            "config_digest": _digest(_RANDOM),
            "parameters_digest": _digest(entry["parameters"]),
        }
        self.cells.append(cell)
        return cell

    def write_campaign(self) -> str:
        workloads = {"schema_version": 1, "profiles": self.entries}
        self.campaign = {
            "schema_version": 1,
            "workloads_path": "workloads.json",
            "workloads_sha256": _digest(workloads),
            "cells": self.cells,
        }
        self.save("workloads.json", workloads)
        self.campaign_path = self.save("campaign.json", self.campaign)
        return self.campaign_path

    def placement(self, cell: dict, which: str, /, **override) -> str:
        arm = which
        identity = cell["identity"]
        threads = str(identity["threads"])
        settings = {"monoprop_NUM_THREADS": threads, "OMP_NUM_THREADS": threads}
        if arm == "baseline":
            settings["monoprop_PARTITIONS"] = threads
        value = {
            "schema_version": 1,
            "artifact_kind": "placement",
            "arm": arm,
            "cell_id": cell["id"],
            "binary_hash": self.binary(arm, identity),
            "identity": identity,
            "runtime_shape": "partitions" if arm == "baseline" else "openmp",
            "declared": {
                "ranks": identity["ranks"],
                "threads": identity["threads"],
                "cpu_allocation": identity["cpu_allocation"],
                "settings": settings,
            },
            "observed": {
                "has_mpi": identity["has_mpi"],
                "ranks": identity["ranks"],
                "threads_per_rank": [identity["threads"]] * identity["ranks"],
                "cpu_allocation": identity["cpu_allocation"],
            },
        }
        for key, item in override.items():
            if key in ("declared", "observed"):
                value[key] = {**value[key], **item}
            else:
                value[key] = item
        return self.save(f"{cell['id']}/{arm}/placement.json", value)

    @staticmethod
    def binary(arm: str, identity: dict) -> str:
        tag = ("b" if arm == "baseline" else "c") + (
            "m" if identity["has_mpi"] else "o"
        )
        return (tag * 32)[:64]

    def common(self, cell: dict, arm: str, placement: str) -> dict:
        return {
            "schema_version": 1,
            "campaign_sha256": _digest(self.campaign),
            "arm": arm,
            "has_mpi": cell["identity"]["has_mpi"],
            "cell_id": cell["id"],
            "profile": cell["profile"],
            "node_id": cell["node_id"],
            "operation": cell["operation"],
            "binary_hash": self.binary(arm, cell["identity"]),
            "identity": cell["identity"],
            "config_digest": cell["config_digest"],
            "parameters_digest": cell["parameters_digest"],
            "placement_path": placement,
            "placement_sha256": _digest(json.loads(Path(placement).read_bytes())),
        }

    def terms(self, cell: dict, arm: str, ranks: list[list[tuple]]) -> list[str]:
        paths = []
        for rank, terms in enumerate(ranks):
            path = self.root / cell["id"] / arm / f"terms-{rank}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_gz(
                path,
                "".join(
                    json.dumps({"key": list(key), "real": real, "imag": imag}) + "\n"
                    for key, real, imag in sorted(terms)
                ),
            )
            paths.append(str(path))
        return paths

    def arm_samples(
        self,
        cell: dict,
        arm: str,
        *,
        n: int = 5,
        runtime: float = 10.0,
        peak: float = 1000.0,
        construction: float = 1200.0,
        outer: float = 1500.0,
        timed_extra: dict | None = None,
        construction_extra: dict | None = None,
        validation_extra: dict | None = None,
        placement: str | None = None,
    ) -> list[dict]:
        placement = placement or self.placement(cell, arm)
        common = self.common(cell, arm, placement)
        validation = {
            **common,
            "artifact_kind": "validation",
            "run_id": f"{cell['id']}-{arm}-validation",
            "basis": "majorana",
            "picture": "heisenberg",
            "energy": -1.25,
            "gradient": [0.1] * 8,
            "global_term_count": 0,
            "graph_layers": 3,
            "term_files": [],
            **(validation_extra or {}),
        }
        validation_path = self.save(f"{cell['id']}/{arm}/validation.json", validation)
        samples = []
        for i in range(n):
            sample_id = f"s{i:02d}"
            timed = {
                **common,
                "artifact_kind": "timed",
                "sample_id": sample_id,
                "run_id": f"{cell['id']}-{arm}-{sample_id}-timed",
                "runtime_seconds": runtime,
                "peak_sum_bytes": peak,
                "peak_max_bytes": peak,
                "op_exact": True,
                "outer_peak_sum_bytes": outer,
                "outer_peak_max_bytes": outer,
                "outer_exact": True,
                **(timed_extra or {}),
            }
            built = {
                **common,
                "artifact_kind": "construction",
                "sample_id": sample_id,
                "run_id": f"{cell['id']}-{arm}-{sample_id}-construction",
                "construction_exact": True,
                "construction_peak_sum_bytes": construction,
                "construction_peak_max_bytes": construction,
                **(construction_extra or {}),
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "timed_path": self.save(
                        f"{cell['id']}/{arm}/{sample_id}/timed.json", timed
                    ),
                    "construction_path": self.save(
                        f"{cell['id']}/{arm}/{sample_id}/construction.json", built
                    ),
                    "validation_path": validation_path,
                }
            )
        return samples

    def manifest(self, cells: list[dict], campaign_sha: str | None = None) -> str:
        return self.save(
            "manifest.json",
            {
                "schema_version": 1,
                "campaign_sha256": campaign_sha or _digest(self.campaign),
                "cells": cells,
            },
        )

    def compare(self, manifest: str) -> tuple[int, dict]:
        driver = _driver()
        output = self.root / "comparison.json"
        code = driver.main(
            [
                "compare",
                "--campaign",
                self.campaign_path,
                "--manifest",
                manifest,
                "--output",
                str(output),
            ]
        )
        return code, json.loads(output.read_text())

    def passing(self, cell: dict, **arms: dict) -> dict:
        return {
            "id": cell["id"],
            "baseline": self.arm_samples(cell, "baseline", **arms.get("baseline", {})),
            "candidate": self.arm_samples(
                cell, "candidate", runtime=9.0, **arms.get("candidate", {})
            ),
        }


def _one_cell(tmp_path: Path, **cell_kwargs) -> tuple[Evidence, dict]:
    evidence = Evidence(tmp_path)
    cell = evidence.add_cell("tiny-energy", **cell_kwargs)
    evidence.write_campaign()
    return evidence, cell


def test_complete_passing_campaign_exits_zero(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    code, report = evidence.compare(evidence.manifest([evidence.passing(cell)]))

    assert code == 0, report
    assert report["passed"] is True
    gates = report["cells"]["tiny-energy"]["gates"]
    assert set(gates) == {
        "runtime",
        "operation_peak_sum",
        "operation_peak_max",
        "construction_peak_sum",
        "construction_peak_max",
    }
    assert gates["runtime"]["ratio"] == pytest.approx(0.9)


def test_a_whole_cell_missing_from_both_arms_fails_completeness(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path)
    kept = evidence.add_cell("tiny-energy")
    evidence.add_cell(
        "large-mpi-off-full",
        profile="large",
        node="bench_random.py::test_random_gradient[heisenberg]",
        operation="gradient",
        threads=96,
        allocation="0-95",
    )
    evidence.write_campaign()
    code, report = evidence.compare(evidence.manifest([evidence.passing(kept)]))

    assert code == 1
    assert report["missing_cells"] == ["large-mpi-off-full"]


@pytest.mark.parametrize("n", [4, 6])
def test_wrong_sample_counts_are_incomplete(tmp_path: Path, n: int) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline", n=n),
                "candidate": evidence.arm_samples(cell, "candidate", n=n),
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 1
    assert "tiny-energy" in report["incomplete_cells"]


def test_unequal_arm_counts_are_incomplete(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline", n=10),
                "candidate": evidence.arm_samples(cell, "candidate", n=5),
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 1
    assert "tiny-energy" in report["incomplete_cells"]


def test_ten_samples_after_repetition_are_judged_on_all_ten(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline", n=10),
                "candidate": evidence.arm_samples(
                    cell, "candidate", n=10, runtime=10.5
                ),
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 1
    assert report["failed_cells"] == ["tiny-energy"]
    assert report["cells"]["tiny-energy"]["needs_additional_samples"] == 0


def test_first_failure_requests_five_more_observations(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest([evidence.passing(cell, candidate={"peak": 1001.0})])
    code, report = evidence.compare(manifest)

    assert code == 1
    result = report["cells"]["tiny-energy"]
    assert result["gates"]["operation_peak_sum"]["passed"] is False
    assert result["needs_additional_samples"] == 5


@pytest.mark.parametrize(
    ("arm_kwargs", "gate"),
    [
        ({"runtime": 10.01}, "runtime"),
        ({"peak": 1000.5}, "operation_peak_max"),
        ({"construction": 1200.5}, "construction_peak_sum"),
        ({"construction": 1200.5}, "construction_peak_max"),
    ],
)
def test_each_of_the_five_gates_fails_on_its_own(
    tmp_path: Path, arm_kwargs: dict, gate: str
) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": evidence.arm_samples(cell, "candidate", **arm_kwargs),
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 1
    assert report["cells"]["tiny-energy"]["gates"][gate]["passed"] is False


def test_sum_and_max_peaks_are_separate_gates(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path, has_mpi=True, ranks=2, allocation="0-3")
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": evidence.arm_samples(
                    cell,
                    "candidate",
                    timed_extra={"peak_sum_bytes": 900, "peak_max_bytes": 1100},
                ),
            }
        ]
    )
    code, report = evidence.compare(manifest)
    gates = report["cells"]["tiny-energy"]["gates"]

    assert code == 1
    assert gates["operation_peak_sum"]["passed"] is True
    assert gates["operation_peak_max"]["passed"] is False


def test_outer_only_regression_is_a_diagnostic(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell, candidate={"outer": 9000.0, "timed_extra": {"outer_exact": None}}
            )
        ]
    )
    code, report = evidence.compare(manifest)
    diagnostics = report["cells"]["tiny-energy"]["diagnostics"]

    assert code == 0
    assert diagnostics["outer_peak_sum"]["ratio"] == pytest.approx(6.0)
    assert diagnostics["outer_peak_sum"]["exactness"] == "unknown"


def test_outer_non_exact_is_labelled_without_failing(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [evidence.passing(cell, baseline={"timed_extra": {"outer_exact": False}})]
    )
    code, report = evidence.compare(manifest)

    assert code == 0
    assert (
        report["cells"]["tiny-energy"]["diagnostics"]["outer_peak_max"]["exactness"]
        == "non-exact"
    )


@pytest.mark.parametrize(
    ("where", "flag", "value"),
    [
        ("timed_extra", "op_exact", False),
        ("timed_extra", "op_exact", None),
        ("construction_extra", "construction_exact", False),
        ("construction_extra", "construction_exact", None),
    ],
)
def test_inexact_required_windows_fail(
    tmp_path: Path, where: str, flag: str, value
) -> None:
    evidence, cell = _one_cell(tmp_path)
    extra = {flag: value}
    manifest = evidence.manifest([evidence.passing(cell, candidate={where: extra})])
    code, report = evidence.compare(manifest)

    assert code == 1
    assert report["failed_cells"] == ["tiny-energy"]
    assert any(flag in reason for reason in report["cells"]["tiny-energy"]["reasons"])


def test_absent_exactness_flag_fails(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate", runtime=9.0)
    for sample in samples:
        path = Path(sample["timed_path"])
        artifact = json.loads(path.read_text())
        del artifact["op_exact"]
        path.write_text(json.dumps(artifact))
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )
    code, _report = evidence.compare(manifest)

    assert code == 1


@pytest.mark.parametrize("value", [None, float("nan"), 0, -1.0])
def test_missing_or_non_finite_measurements_are_malformed(
    tmp_path: Path, value
) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    path = Path(samples[2]["timed_path"])
    artifact = json.loads(path.read_text())
    artifact["runtime_seconds"] = value
    path.write_text(json.dumps(artifact))
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 2
    assert report["errors"]


def test_duplicate_sample_ids_are_rejected(tmp_path: Path) -> None:
    # Two genuinely distinct runs that both claim sample s00.
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    for kind in ("timed_path", "construction_path"):
        path = Path(samples[1][kind])
        artifact = json.loads(path.read_text())
        artifact["sample_id"] = samples[0]["sample_id"]
        path.write_text(json.dumps(artifact))
    samples[1] = {**samples[1], "sample_id": samples[0]["sample_id"]}
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 2
    assert "twice" in report["errors"][0]


def test_five_references_to_one_run_are_not_five_samples(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    reused = [{**samples[0], "sample_id": f"s{i:02d}"} for i in range(5)]
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": reused,
            }
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 2
    assert "already used" in report["errors"][0]


def test_copied_artifacts_reusing_a_run_id_are_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    copy = Path(samples[0]["timed_path"]).read_text()
    duplicate = tmp_path / "copy-timed.json"
    duplicate.write_text(copy)
    artifact = json.loads(copy)
    artifact["sample_id"] = samples[1]["sample_id"]
    duplicate.write_text(json.dumps(artifact))
    samples[1] = {**samples[1], "timed_path": str(duplicate)}
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_measurement_reused_across_cells_is_rejected(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path)
    first = evidence.add_cell("tiny-energy")
    second = evidence.add_cell(
        "tiny-gradient",
        node="bench_random.py::test_random_gradient[heisenberg]",
        operation="gradient",
    )
    evidence.write_campaign()
    one = evidence.passing(first)
    two = evidence.passing(second)
    two["candidate"][0] = {
        **two["candidate"][0],
        "timed_path": one["candidate"][0]["timed_path"],
    }
    code, report = evidence.compare(evidence.manifest([one, two]))

    assert code == 2
    assert "already used" in report["errors"][0]


def test_mismatched_timed_and_construction_pair_is_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    samples[0], samples[1] = (
        {**samples[0], "construction_path": samples[1]["construction_path"]},
        {**samples[1], "construction_path": samples[0]["construction_path"]},
    )
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "config_digest",
        "parameters_digest",
        "node_id",
        "binary_hash",
        "has_mpi",
        "cell_id",
    ],
)
def test_mismatched_identity_fields_are_rejected(tmp_path: Path, field: str) -> None:
    evidence, cell = _one_cell(tmp_path)
    changed = {
        "identity": {**cell["identity"], "threads": 4},
        "config_digest": "0" * 64,
        "parameters_digest": "0" * 64,
        "node_id": "bench_random.py::test_random_gradient[heisenberg]",
        "binary_hash": "f" * 64,
        "has_mpi": True,
        "cell_id": "other-cell",
    }[field]
    manifest = evidence.manifest(
        [evidence.passing(cell, candidate={"construction_extra": {field: changed}})]
    )

    assert evidence.compare(manifest)[0] == 2


def test_unequal_index_width_is_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    identity = {**cell["identity"], "index_bits": 64}
    manifest = evidence.manifest(
        [evidence.passing(cell, candidate={"timed_extra": {"identity": identity}})]
    )

    assert evidence.compare(manifest)[0] == 2


def test_calibration_output_is_not_acceptance_evidence(tmp_path: Path) -> None:
    # Pilot artifacts belong to a different (draft) campaign digest.
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(
        cell, "candidate", timed_extra={"campaign_sha256": "a" * 64}
    )
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_manifest_for_another_campaign_is_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest([evidence.passing(cell)], campaign_sha="d" * 64)

    assert evidence.compare(manifest)[0] == 2


def test_extra_and_duplicate_manifest_cells_are_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    good = evidence.passing(cell)
    assert evidence.compare(evidence.manifest([good, good]))[0] == 2
    assert (
        evidence.compare(evidence.manifest([good, {**good, "id": "not-in-inventory"}]))[
            0
        ]
        == 2
    )


def test_campaign_with_duplicate_cell_ids_is_rejected(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path)
    cell = evidence.add_cell("tiny-energy")
    evidence.cells.append(dict(cell))
    evidence.write_campaign()

    assert evidence.compare(evidence.manifest([]))[0] == 2


def test_tampered_workloads_file_is_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest([evidence.passing(cell)])
    (tmp_path / "workloads.json").write_text("{}")

    assert evidence.compare(manifest)[0] == 2


def test_mpi_off_cell_declared_at_several_ranks_is_rejected(tmp_path: Path) -> None:
    evidence, _cell = _one_cell(tmp_path, has_mpi=False, ranks=2)

    assert evidence.compare(evidence.manifest([]))[0] == 2


def test_linear_routing_at_three_ranks_is_rejected(tmp_path: Path) -> None:
    evidence, _cell = _one_cell(
        tmp_path,
        has_mpi=True,
        ranks=3,
        routing={"mode": "linear", "bits": 1, "seed": 7},
    )

    assert evidence.compare(evidence.manifest([]))[0] == 2


def test_d1_mpi_enabled_samples_cannot_stand_in_for_mpi_off_cells(
    tmp_path: Path,
) -> None:
    # On a one-domain host the MPI-enabled R=1,T=C cell has the same geometry as the required
    # MPI-off full-machine cell; the inventory keeps both and evidence cannot be shared.
    evidence = Evidence(tmp_path)
    mpi_off = evidence.add_cell(
        "reference-mpi-off-full", has_mpi=False, threads=96, allocation="0-95"
    )
    mpi_on = evidence.add_cell(
        "reference-l2b", has_mpi=True, threads=96, allocation="0-95"
    )
    evidence.write_campaign()
    on = evidence.passing(mpi_on)
    borrowed = {
        "id": mpi_off["id"],
        "baseline": on["baseline"],
        "candidate": on["candidate"],
    }

    assert evidence.compare(evidence.manifest([on, borrowed]))[0] == 2


def test_campaign_listing_the_same_cell_twice_under_two_ids_is_rejected(
    tmp_path: Path,
) -> None:
    evidence = Evidence(tmp_path)
    evidence.add_cell("a")
    evidence.add_cell("b")
    evidence.write_campaign()

    assert evidence.compare(evidence.manifest([]))[0] == 2


@pytest.mark.parametrize(
    "override",
    [
        {"runtime_shape": "openmp"},
        {"arm": "candidate"},
        {"declared": {"threads": 1}},
        {"declared": {"settings": {"monoprop_NUM_THREADS": "2"}}},
        {"observed": {"threads_per_rank": [1]}},
        {"observed": {"has_mpi": True}},
        {"observed": {"cpu_allocation": "0"}},
        {"binary_hash": "e" * 64},
    ],
)
def test_invalid_placement_evidence_is_rejected(tmp_path: Path, override: dict) -> None:
    evidence, cell = _one_cell(tmp_path)
    bad = evidence.placement(cell, "baseline", **override)
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline", placement=bad),
                "candidate": evidence.arm_samples(cell, "candidate"),
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_candidate_placement_with_leftover_partitions_is_rejected(
    tmp_path: Path,
) -> None:
    evidence, cell = _one_cell(tmp_path)
    bad = evidence.placement(
        cell,
        "candidate",
        declared={
            "settings": {"monoprop_NUM_THREADS": "2", "monoprop_PARTITIONS": "2"}
        },
    )
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": evidence.arm_samples(cell, "candidate", placement=bad),
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_missing_placement_provenance_is_malformed(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    Path(
        json.loads(Path(samples[0]["timed_path"]).read_text())["placement_path"]
    ).unlink()
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_absent_validation_is_malformed(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    samples = evidence.arm_samples(cell, "candidate")
    Path(samples[0]["validation_path"]).unlink()
    manifest = evidence.manifest(
        [
            {
                "id": cell["id"],
                "baseline": evidence.arm_samples(cell, "baseline"),
                "candidate": samples,
            }
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_validation_for_another_binary_is_rejected(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell, candidate={"validation_extra": {"binary_hash": "9" * 64}}
            )
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def _numerical(
    tmp_path: Path, baseline: dict, candidate: dict, **cell_kwargs
) -> tuple[int, dict]:
    evidence, cell = _one_cell(tmp_path, **cell_kwargs)
    for extra, arm in ((baseline, "baseline"), (candidate, "candidate")):
        if "terms" in extra:
            ranks = extra.pop("terms")
            extra["term_files"] = evidence.terms(cell, arm, ranks)
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                baseline={"validation_extra": baseline},
                candidate={"validation_extra": candidate},
            )
        ]
    )
    return evidence.compare(manifest)


_TERMS = [
    [((), 0.5, 0.0), ((0, 1), 0.25, 0.0)],
    [((), 0.5, 0.0), ((2, 3), -0.125, 0.0)],
]


def test_matching_global_maps_pass_across_different_ownership(tmp_path: Path) -> None:
    # Candidate ownership differs; the replicated Heisenberg identity is kept once per arm.
    moved = [
        [((), 0.5, 0.0), ((2, 3), -0.125, 0.0), ((0, 1), 0.25, 0.0)],
        [((), 0.5, 0.0)],
    ]
    code, report = _numerical(
        tmp_path,
        {"terms": [list(t) for t in _TERMS], "global_term_count": 3},
        {"terms": moved, "global_term_count": 3},
        has_mpi=True,
        ranks=2,
        allocation="0-3",
    )

    assert code == 0, report
    assert (
        report["cells"]["tiny-energy"]["numerical"]["pairs"][0]["terms"]["common_terms"]
        == 3
    )


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ({"energy": -1.25}, {"energy": -1.2501}),
        ({"gradient": [0.1] * 8}, {"gradient": [0.1] * 7 + [0.11]}),
        ({"gradient": [0.1] * 8}, {"gradient": [0.1] * 7}),
        ({"energy": -1.25}, {"energy": float("inf")}),
        (
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 1},
            {
                "terms": [[((0, 1), 0.25, 0.0), ((4, 5), 0.0, 0.0)]],
                "global_term_count": 2,
            },
        ),
        (
            {
                "terms": [[((0, 1), 0.25, 0.0), ((4, 5), 0.0, 0.0)]],
                "global_term_count": 2,
            },
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 1},
        ),
        (
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 1},
            {"terms": [[((0, 1), 0.2500001, 0.0)]], "global_term_count": 1},
        ),
        (
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 1},
            {
                "terms": [[((0, 1), 0.25, 0.0)], [((0, 1), 0.25, 0.0)]],
                "global_term_count": 1,
            },
        ),
        (
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 1},
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 2},
        ),
        (
            {"terms": [[((0, 1), 0.25, 0.0)]], "global_term_count": 1},
            {"term_files": []},
        ),
    ],
    ids=[
        "energy",
        "gradient-entry",
        "gradient-length",
        "non-finite",
        "extra-term",
        "missing-term",
        "value",
        "duplicate-ownership",
        "count",
        "one-arm-map",
    ],
)
def test_numerical_mismatches_fail_the_cell(
    tmp_path: Path, baseline: dict, candidate: dict
) -> None:
    code, report = _numerical(tmp_path, baseline, candidate)

    assert code == 1, report
    assert report["cells"]["tiny-energy"]["numerical"]["ok"] is False


def test_replicated_identity_disagreement_fails(tmp_path: Path) -> None:
    bad = [[((), 0.5, 0.0), ((0, 1), 0.25, 0.0)], [((), 0.75, 0.0)]]
    code, report = _numerical(
        tmp_path,
        {"terms": [list(t) for t in _TERMS], "global_term_count": 3},
        {"terms": bad, "global_term_count": 2},
        has_mpi=True,
        ranks=2,
        allocation="0-3",
    )

    assert code == 1
    assert "identity" in report["cells"]["tiny-energy"]["numerical"]["reason"]


def test_schrodinger_identity_is_an_ordinary_owned_term(tmp_path: Path) -> None:
    doubled = [[((), 0.5, 0.0)], [((), 0.5, 0.0)]]
    code, report = _numerical(
        tmp_path,
        {"terms": doubled, "global_term_count": 1, "picture": "schrodinger"},
        {
            "terms": [list(t) for t in doubled],
            "global_term_count": 1,
            "picture": "schrodinger",
        },
        has_mpi=True,
        ranks=2,
        allocation="0-3",
    )

    assert code == 1
    assert "owned by 2 ranks" in report["cells"]["tiny-energy"]["numerical"]["reason"]


def test_unsorted_term_file_is_malformed(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    files = evidence.terms(
        cell, "baseline", [[((0, 1), 0.25, 0.0), ((2, 3), 0.5, 0.0)]]
    )
    _write_gz(
        Path(files[0]),
        '{"key": [2, 3], "real": 0.5, "imag": 0.0}\n{"key": [0, 1], "real": 0.25, "imag": 0.0}\n',
    )
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                baseline={
                    "validation_extra": {"term_files": files, "global_term_count": 2}
                },
                candidate={
                    "validation_extra": {"term_files": files, "global_term_count": 2}
                },
            )
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_pared_evidence_must_not_carry_term_files(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path, threshold=1e-10)
    files = evidence.terms(cell, "baseline", [[((0, 1), 0.25, 0.0)]])
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                baseline={
                    "validation_extra": {"term_files": files, "pare_threshold": 1e-10}
                },
                candidate={"validation_extra": {"pare_threshold": 1e-10}},
            )
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_failed_graph_replay_check_fails_the_cell(tmp_path: Path) -> None:
    evidence, cell = _one_cell(
        tmp_path,
        node="bench_random.py::test_random_propagate[heisenberg]",
        operation="propagate",
    )
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                candidate={
                    "validation_extra": {
                        "graph_replay_check": {"applicable": True, "ok": False}
                    }
                },
            )
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 1
    assert "replay" in report["cells"]["tiny-energy"]["numerical"]["reason"]


def test_pare_construction_validates_the_callable_gradient(tmp_path: Path) -> None:
    evidence, cell = _one_cell(
        tmp_path,
        node="driver::pare_functional_construct[heisenberg]",
        operation="pare_functional_construct",
        threshold=1e-10,
        kind="driver",
    )
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                baseline={"validation_extra": {"pare_threshold": 1e-10}},
                candidate={
                    "validation_extra": {"pare_threshold": 1e-10, "gradient": None}
                },
            )
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 1
    assert "gradient" in report["cells"]["tiny-energy"]["numerical"]["reason"]


# ------------------------------------------------------------- measurement routines (with fakes)


class _FakeWindow:
    """Records open/close in the shared event log; fixed byte counts."""

    def __init__(self, log: list[str], name: str, *, settle: bool = True) -> None:
        self.log, self.name, self.settle = log, name, settle
        self.peak_bytes, self.baseline_bytes, self.exact = 2000, 500, True

    @property
    def delta_bytes(self) -> int:
        return self.peak_bytes - self.baseline_bytes

    def start(self) -> None:
        self.log.append(f"{self.name}:start(settle={self.settle})")

    def stop(self) -> None:
        self.log.append(f"{self.name}:stop")


class _FakeCallable:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def __call__(self, parameters):
        self.log.append("EVALUATED")
        return -1.0, [0.0] * len(parameters)

    def __del__(self) -> None:
        self.log.append("callable-dead")


class _FakePropagator:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        simulator = type("Sim", (), {})()
        simulator.operator_memory_bytes = lambda: 10
        simulator.graph_memory_bytes = lambda: 20
        self._simulator = simulator

    def build_graph(self, _circuit) -> None:
        self.log.append("build_graph")

    def propagate(self, _circuit) -> None:
        self.log.append("propagate")

    def expectation_value_and_gradient_functional(self, threshold):
        self.log.append(f"functional({threshold})")
        return _FakeCallable(self.log)

    def expectation_value_functional(self, threshold):
        self.log.append(f"energy-functional({threshold})")
        return _FakeCallable(self.log)


def _fake_context(driver):
    return driver.Context(
        comm=None,
        rank=0,
        size=1,
        has_mpi=False,
        binary_path="x",
        binary_hash="0" * 64,
        declared={},
        cpu_allocation="0",
        placement_path="p",
        placement_sha256="1" * 64,
    )


def _fake_problem(driver, log: list[str], steps: int = 2):
    return driver.Problem(
        make_propagator=lambda: (
            log.append("propagator-built") or _FakePropagator(log),
            "circuit",
        ),
        steps=steps,
        parameters=[0.1, 0.2],
    )


def test_pare_construction_times_only_the_callable_and_keeps_it_alive() -> None:
    driver = _driver()
    log: list[str] = []
    windows = iter(["outer", "operation"])
    ticks = iter([100.0, 103.5])

    def clock() -> float:
        log.append("clock")
        return next(ticks)

    result = driver.timed_pare_construct(
        _fake_problem(driver, log),
        _fake_context(driver),
        window_factory=lambda **kw: _FakeWindow(log, next(windows), **kw),
        clock=clock,
        events=log,
    )

    assert result["runtime_seconds"] == pytest.approx(3.5)
    assert "EVALUATED" not in log
    order = [
        "outer:start(settle=True)",
        "propagator-built",
        "build_graph",
        "graph-built",
        "operation:start(settle=False)",
        "clock",
        "functional(1e-10)",
        "clock",
        "operation:stop",
        "outer:stop",
        "callable-dead",
        "callable-released",
    ]
    positions = [log.index(event) for event in order[:6]] + [
        len(log) - 1 - log[::-1].index(event) for event in order[6:]
    ]
    assert positions == sorted(positions), log
    assert log.count("build_graph") == 2  # every step before timing starts
    assert result["op_exact"] is True
    assert result["outer_exact"] is True
    assert result["peak_sum_bytes"] == result["peak_max_bytes"] == 2000


def test_construction_window_opens_before_inputs_and_outlives_outputs() -> None:
    driver = _driver()
    log: list[str] = []
    spec = driver.parse_node("bench_random.py::test_random_gradient[heisenberg]")

    def build():
        log.append("inputs-built")
        return _fake_problem(driver, log, steps=1)

    result = driver.construction_worker(
        spec,
        {"pare_threshold": None},
        _fake_context(driver),
        window_factory=lambda **kw: _FakeWindow(log, "construction", **kw),
        build=build,
        events=log,
    )

    assert log.index("construction:start(settle=True)") < log.index("inputs-built")
    assert log.index("EVALUATED") < log.index("construction:stop")
    assert log.index("construction:stop") < log.index("callable-dead")
    assert result["construction_exact"] is True
    assert "runtime_seconds" not in result  # the untimed worker invents no timing


def test_pare_construction_worker_builds_but_never_evaluates_the_callable() -> None:
    driver = _driver()
    log: list[str] = []
    spec = driver.parse_node("driver::pare_functional_construct[schrodinger]")
    driver.construction_worker(
        spec,
        {"pare_threshold": 1e-10},
        _fake_context(driver),
        window_factory=lambda **kw: _FakeWindow(log, "construction", **kw),
        build=lambda: _fake_problem(driver, log, steps=1),
        events=log,
    )

    assert "functional(1e-10)" in log
    assert "EVALUATED" not in log
    assert log.index("construction:stop") < log.index("callable-dead")


def test_propagate_construction_runs_every_step_without_a_graph() -> None:
    driver = _driver()
    log: list[str] = []
    spec = driver.parse_node("bench_models.py::test_model_propagate[hubbard]")
    driver.construction_worker(
        spec,
        {"pare_threshold": None},
        _fake_context(driver),
        window_factory=lambda **kw: _FakeWindow(log, "construction", **kw),
        build=lambda: _fake_problem(driver, log, steps=3),
        events=log,
    )

    assert log.count("propagate") == 3
    assert "build_graph" not in log


# ------------------------------------------------------------------ workload and CLI identity


def test_driver_has_exactly_three_modes() -> None:
    driver = _driver()
    parser = driver.build_parser()
    modes = next(a for a in parser._actions if a.dest == "mode")

    assert set(modes.choices) == {"observe", "validate", "compare"}
    observe = modes.choices["observe"]
    required = {a.dest for a in observe._actions if a.required}
    assert required == {
        "arm",
        "runtime_shape",
        "campaign",
        "cell_id",
        "sample_id",
        "placement",
        "output",
    }
    validate = modes.choices["validate"]
    assert "sample_id" not in {a.dest for a in validate._actions}
    assert "whole_process" not in {a.dest for a in validate._actions}


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            "bench_random.py::test_random_energy[schrodinger]",
            ("random", "energy", "schrodinger", "pytest"),
        ),
        (
            "bench_models.py::test_model_propagate[pauli]",
            ("pauli", "propagate", "heisenberg", "pytest"),
        ),
        (
            "driver::pare_functional_construct[heisenberg]",
            ("random", "pare_functional_construct", "heisenberg", "driver"),
        ),
    ],
)
def test_node_ids_resolve_to_historical_benchmarks(node: str, expected: tuple) -> None:
    spec = _driver().parse_node(node)

    assert (
        spec.family,
        spec.operation,
        spec.picture,
        spec.measurement_kind,
    ) == expected


@pytest.mark.parametrize(
    "node",
    [
        "bench_random.py::test_random_inplace[heisenberg]",
        "bench_models.py::test_random_energy[hubbard]",
        "driver::pare_functional_construct[hubbard]",
        "bench_random.py::test_random_energy",
    ],
)
def test_unknown_node_ids_are_rejected(node: str) -> None:
    driver = _driver()
    with pytest.raises(driver.EvidenceError):
        driver.parse_node(node)


def _pare_entry(**changes) -> dict:
    entry = {
        "operation": "pare_functional_construct",
        "measurement_kind": "driver",
        "functional": "expectation_value_and_gradient_functional",
        "basis": "majorana",
        "picture": "heisenberg",
        "config": _RANDOM,
        "parameters": [0.1] * 8,
        "pare_threshold": 1e-10,
    }
    entry.update(changes)
    return entry


@pytest.mark.parametrize(
    "changes",
    [
        {"pare_threshold": 1e-9},
        {"pare_threshold": None},
        {"functional": "expectation_value_functional"},
        {"measurement_kind": "pytest"},
        {"picture": "schrodinger"},
        {"basis": "pauli"},
        {"parameters": [float("nan")]},
    ],
)
def test_pare_construction_manifest_identity_is_exact(changes: dict) -> None:
    driver = _driver()
    with pytest.raises(driver.EvidenceError):
        driver.check_workload_entry(
            "tiny-pared",
            "driver::pare_functional_construct[heisenberg]",
            _pare_entry(**changes),
        )


def test_pare_construction_manifest_entry_is_accepted() -> None:
    driver = _driver()
    spec = driver.check_workload_entry(
        "tiny-pared", "driver::pare_functional_construct[heisenberg]", _pare_entry()
    )

    assert spec.operation == "pare_functional_construct"


def test_pytest_flags_reproduce_the_workload_config() -> None:
    driver = _driver()
    spec = driver.parse_node("bench_models.py::test_model_energy[hubbard]")
    entry = {
        "config": {"num_sites": 4, "lower_atol": 0.0001, "observable_spin": "up"},
        "pare_threshold": 1e-10,
    }

    assert driver.pytest_flags(spec, entry) == [
        "--hubbard-num-sites=4",
        "--hubbard-lower-atol=0.0001",
        "--hubbard-observable-spin=up",
        "--pare-threshold=1e-10",
    ]


def test_random_pytest_cells_must_use_the_benchmark_default_lower_atol() -> None:
    driver = _driver()
    entry = {
        "operation": "energy",
        "measurement_kind": "pytest",
        "basis": "majorana",
        "picture": "heisenberg",
        "config": {**_RANDOM, "lower_atol": 1e-6},
        "parameters": [0.1] * 8,
        "pare_threshold": None,
    }
    with pytest.raises(driver.EvidenceError, match="lower_atol"):
        driver.check_workload_entry(
            "tiny", "bench_random.py::test_random_energy[heisenberg]", entry
        )


# --------------------------------------------------------------- end to end on the real engine
#
# Baseline-arm artifacts only: a baseline binary is never relabelled as candidate evidence, even
# in a test. These runs check schema and provenance, not performance.

_E2E_NODES = {
    "e2e-energy": ("tiny", "bench_random.py::test_random_energy[heisenberg]", None),
    "e2e-propagate": (
        "tiny",
        "bench_random.py::test_random_propagate[heisenberg]",
        None,
    ),
    "e2e-pare": ("tiny-pared", "driver::pare_functional_construct[schrodinger]", 1e-10),
    "e2e-hubbard": ("tiny", "bench_models.py::test_model_build_graph[hubbard]", None),
}
_TINY_HUBBARD = {"num_sites": 4, "observable_site": 1, "trotter_steps": 2}
_TINY_RANDOM = {**_RANDOM, "obs_terms": 16}


def _e2e_campaign(root: Path, *, has_mpi: bool = False) -> tuple[str, dict]:
    # Optional workspace dependency, absent when the suite tests a built wheel.
    from monoprop_bench_tools.models import (  # noqa: PLC0415
        HubbardConfig,
        build_hubbard_problem,
        make_random_problem,
    )

    driver = _driver()
    profiles: dict[str, dict] = {"tiny": {}, "tiny-pared": {}}
    cells = []
    allocation = driver.format_cpu_list(os.sched_getaffinity(0))
    for cell_id, (profile, node, threshold) in _E2E_NODES.items():
        spec = driver.parse_node(node)
        if spec.family == "random":
            config = dict(_TINY_RANDOM)
            problem = make_random_problem(
                **{k: v for k, v in config.items() if k != "lower_atol"}
            )
            parameters = [float(p) for p in problem.parameters]
        else:
            hubbard = HubbardConfig(**_TINY_HUBBARD)
            config = dataclasses.asdict(hubbard)
            # Rank-local construction: an MPI build must not be handed the world implicitly.
            _propagator, circuit = build_hubbard_problem(hubbard, comm=_serial_comm())
            parameters = [float(p) for p in circuit.parameters] * hubbard.trotter_steps
        entry = {
            "operation": spec.operation,
            "measurement_kind": spec.measurement_kind,
            "basis": "majorana",
            "picture": spec.picture,
            "config": config,
            "parameters": parameters,
            "pare_threshold": threshold,
        }
        if spec.measurement_kind == "driver":
            entry["functional"] = "expectation_value_and_gradient_functional"
        profiles[profile][node] = entry
        identity = {
            "has_mpi": has_mpi,
            "ranks": 1,
            "threads": 1,
            "index_bits": 32,
            "cpu_allocation": allocation,
            "compiler_flags": "recorded-by-operator",
            "config": config,
            "routing": {"mode": "linear", "bits": 0, "seed": driver.DEFAULT_ROUTE_SEED},
        }
        cells.append(
            {
                "id": cell_id,
                "profile": profile,
                "node_id": node,
                "operation": spec.operation,
                "measurement_kind": spec.measurement_kind,
                "identity": identity,
                "config_digest": _digest(config),
                "parameters_digest": _digest(parameters),
            }
        )
    workloads = {"schema_version": 1, "profiles": profiles}
    (root / "workloads.json").write_bytes(_encoded(workloads))
    campaign = {
        "schema_version": 1,
        "workloads_path": "workloads.json",
        "workloads_sha256": _digest(workloads),
        "cells": cells,
    }
    path = root / "campaign.json"
    path.write_bytes(_encoded(campaign))
    binary = hashlib.sha256(Path(monoprop._core.__file__).read_bytes()).hexdigest()
    placements = {}
    for cell in cells:
        placement = {
            "schema_version": 1,
            "artifact_kind": "placement",
            "arm": "baseline",
            "cell_id": cell["id"],
            "binary_hash": binary,
            "identity": cell["identity"],
            "runtime_shape": "partitions",
            "declared": {
                "ranks": 1,
                "threads": 1,
                "cpu_allocation": allocation,
                "settings": dict(_E2E_SETTINGS),
            },
            "observed": {
                "has_mpi": has_mpi,
                "ranks": 1,
                "threads_per_rank": [1],
                "cpu_allocation": allocation,
            },
        }
        placement_path = root / f"placement-{cell['id']}.json"
        placement_path.write_bytes(_encoded(placement))
        placements[cell["id"]] = str(placement_path)
    return str(path), placements


def _serial_comm():
    """Return COMM_SELF for an MPI extension, ``None`` otherwise (never imports mpi4py MPI-off)."""
    if not monoprop.has_mpi:
        return None
    from mpi4py import MPI  # noqa: PLC0415 - gated on the build mode

    return MPI.COMM_SELF


_E2E_SETTINGS = {
    "monoprop_NUM_THREADS": "1",
    "monoprop_PARTITIONS": "1",
    "OMP_NUM_THREADS": "1",
}


def _driver_run(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
        [sys.executable, str(_SCRIPT), *args],
        cwd=_ROOT,
        env=env if env is not None else _bench_env(**_E2E_SETTINGS),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _observe_args(
    campaign: str, placements: dict, cell: str, out: Path, *extra: str
) -> list[str]:
    return [
        "--arm",
        "baseline",
        "--runtime-shape",
        "partitions",
        "--campaign",
        campaign,
        "--cell-id",
        cell,
        "--placement",
        placements[cell],
        "--output",
        str(out),
        *extra,
    ]


@_launches
@pytest.mark.skipif(monoprop.has_mpi, reason="MPI-off end-to-end run")
def test_observe_and_validate_write_complete_joined_artifacts(tmp_path: Path) -> None:
    _needs_bench_tools()
    driver = _driver()
    campaign, placements = _e2e_campaign(tmp_path)
    out = tmp_path / "results"
    loaded = driver.load_campaign(Path(campaign))
    placement_cache: dict = {}
    for cell_id in _E2E_NODES:
        base = _observe_args(campaign, placements, cell_id, out)
        for extra in ([], ["--whole-process"]):
            proc = _driver_run("observe", *base, "--sample-id", "s00", *extra)
            assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        proc = _driver_run("validate", *base)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

        cell = loaded.cells[cell_id]
        timed = json.loads(
            (out / "baseline" / cell_id / "s00" / "timed.json").read_text()
        )
        built = json.loads(
            (out / "baseline" / cell_id / "s00" / "construction.json").read_text()
        )
        (validation_path,) = (out / "baseline" / cell_id).glob("validation-*.json")
        validation = json.loads(validation_path.read_text())
        expected = {
            "arm": "baseline",
            "cell_id": cell_id,
            "campaign_sha256": loaded.sha256,
            "binary_hash": timed["binary_hash"],
            "profile": cell["profile"],
            "node_id": cell["node_id"],
            "operation": cell["operation"],
            "identity": cell["identity"],
            "config_digest": cell["config_digest"],
            "parameters_digest": cell["parameters_digest"],
            "has_mpi": False,
            "placement_sha256": timed["placement_sha256"],
        }
        for artifact in (timed, built, validation):
            driver._check_join(artifact, expected, cell_id, placement_cache)
            assert artifact["has_mpi"] is False
        assert timed["runtime_seconds"] > 0
        assert timed["op_exact"] is True
        assert timed["outer_exact"] is True
        assert timed["peak_sum_bytes"] == timed["peak_max_bytes"] > 0
        assert built["construction_exact"] is True
        assert built["construction_peak_sum_bytes"] > 0
        assert "runtime_seconds" not in built
        assert driver.compare_validations(validation, validation)["ok"] is True
        if cell_id == "e2e-pare":
            assert validation["term_files"] == []
            assert len(validation["gradient"]) == len(loaded.entry(cell)["parameters"])
        else:
            assert validation["global_term_count"] > 0
            assert len(validation["term_files"]) == 1
            assert validation["term_files"][0].endswith(".jsonl.gz")
        if cell_id == "e2e-propagate":
            assert validation["graph_replay_check"]["applicable"] is True
            assert validation["graph_replay_check"]["ok"] is True
        if cell_id in ("e2e-energy", "e2e-hubbard"):
            assert timed["raw_dir"].endswith("s00/timed")

    # Artifacts are created exclusively: rerunning a sample cannot overwrite it.
    again = _driver_run(
        "observe",
        *_observe_args(campaign, placements, "e2e-energy", out),
        "--sample-id",
        "s00",
    )
    assert again.returncode == 2
    assert "never overwritten" in again.stderr


@_launches
@pytest.mark.skipif(monoprop.has_mpi, reason="MPI-off end-to-end run")
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"arm": "candidate"}, "arm"),
        ({"env": {"monoprop_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}}, "partitions"),
        ({"env": {**_E2E_SETTINGS, "OMP_PLACES": "cores"}}, "settings differ"),
        ({"env": {**_E2E_SETTINGS, "monoprop_ROUTING": "splitmix"}}, "routing"),
        ({"has_mpi": True}, "has_mpi"),
        ({"cell": "not-a-cell"}, "no cell"),
    ],
)
def test_observe_rejects_contradictory_evidence(
    tmp_path: Path, change: dict, message: str
) -> None:
    _needs_bench_tools()
    campaign, placements = _e2e_campaign(tmp_path, has_mpi=change.get("has_mpi", False))
    cell = change.get("cell", "e2e-energy")
    args = _observe_args(
        campaign, {**placements, cell: placements["e2e-energy"]}, cell, tmp_path / "r"
    )
    if "arm" in change:
        args[1] = change["arm"]
    env = _bench_env(**change.get("env", _E2E_SETTINGS))
    proc = _driver_run("observe", *args, "--sample-id", "s00", env=env)

    assert proc.returncode == 2
    assert message in proc.stderr
    assert not (tmp_path / "r" / "baseline" / cell / "s00" / "timed.json").exists()


@_launches
@pytest.mark.skipif(
    not monoprop.has_mpi or shutil.which("mpiexec") is None,
    reason="needs an MPI extension and mpiexec",
)
def test_two_rank_observe_and_validate_merge_the_replicated_identity(
    tmp_path: Path,
) -> None:
    _needs_bench_tools()
    driver = _driver()
    campaign_path, placements = _e2e_campaign(tmp_path, has_mpi=True)
    # Re-issue the energy cell at R=2 (linear, one rank bit) over the whole host mask.
    campaign = json.loads(Path(campaign_path).read_text())
    cell = next(c for c in campaign["cells"] if c["id"] == "e2e-energy")
    allocation = driver.format_cpu_list(range(os.cpu_count() or 1))
    cell["identity"] = {
        **cell["identity"],
        "ranks": 2,
        "cpu_allocation": allocation,
        "routing": {"mode": "linear", "bits": 1, "seed": driver.DEFAULT_ROUTE_SEED},
    }
    campaign["cells"] = [cell]
    Path(campaign_path).write_bytes(_encoded(campaign))
    placement = json.loads(Path(placements["e2e-energy"]).read_text())
    placement["identity"] = cell["identity"]
    placement["declared"] |= {"ranks": 2, "cpu_allocation": allocation}
    placement["observed"] |= {
        "ranks": 2,
        "threads_per_rank": [1, 1],
        "cpu_allocation": allocation,
    }
    Path(placements["e2e-energy"]).write_bytes(_encoded(placement))
    out = tmp_path / "results"
    base = _observe_args(campaign_path, placements, "e2e-energy", out)
    launcher = [shutil.which("mpiexec"), "--bind-to", "none", "-n", "2"]
    exported = [flag for key in _E2E_SETTINGS for flag in ("-x", key)]
    for args in (["observe", *base, "--sample-id", "s00"], ["validate", *base]):
        proc = subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
            [*launcher, *exported, sys.executable, str(_SCRIPT), *args],
            cwd=_ROOT,
            env=_bench_env(**_E2E_SETTINGS),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    timed = json.loads(
        (out / "baseline" / "e2e-energy" / "s00" / "timed.json").read_text()
    )
    (validation_path,) = (out / "baseline" / "e2e-energy").glob("validation-*.json")
    validation = json.loads(validation_path.read_text())
    assert timed["has_mpi"] is True
    assert timed["observed"]["ranks"] == 2
    assert timed["peak_sum_bytes"] > timed["peak_max_bytes"]
    assert len(validation["term_files"]) == 2
    # The identity sits on both ranks' files but counts once in the global map.
    merged = list(driver.merged_terms(validation["term_files"], "heisenberg"))
    assert [key for key, _r, _i in merged].count(()) == 1
    assert validation["global_term_count"] == len(merged)
    assert driver.compare_validations(validation, validation)["ok"] is True


def test_term_lines_encode_exactly_as_json(tmp_path: Path) -> None:
    driver = _driver()
    terms = {
        (): complex(0.0, -0.0),
        (0, 1, 2, 6): complex(-2.9288912e-05, 1e300),
        (3, 250): complex(0.1 + 0.2, -123456789.125),
    }
    path = tmp_path / "terms.jsonl.gz"
    assert driver._write_terms(path, terms) == (3, True)
    expected = "".join(
        json.dumps({"key": list(key), "real": terms[key].real, "imag": terms[key].imag})
        + "\n"
        for key in sorted(terms)
    )
    # A gzip member whose decompressed lines are exactly the JSON encoding.
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert gzip.decompress(path.read_bytes()).decode() == expected
    # Reproducible bytes: MTIME (header bytes 4-7) is zero and no file name is stored.
    assert path.read_bytes()[4:8] == bytes(4)
    assert b"terms" not in path.read_bytes()[:32]
    again = tmp_path / "again.jsonl.gz"
    driver._write_terms(again, terms)
    assert again.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("name", ["terms.jsonl", "terms.txt"])
def test_term_files_must_be_gzip_json_lines(tmp_path: Path, name: str) -> None:
    driver = _driver()
    with pytest.raises(driver.EvidenceError, match=r"\.jsonl\.gz"):
        driver._write_terms(tmp_path / name, {(0, 1): 0.5 + 0j})


def test_uncompressed_term_file_is_malformed(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"key": [0, 1], "real": 0.25, "imag": 0.0}\n')
    extra = {"term_files": [str(plain)], "global_term_count": 1}
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                baseline={"validation_extra": extra},
                candidate={"validation_extra": extra},
            )
        ]
    )
    code, report = evidence.compare(manifest)

    assert code == 2
    assert ".jsonl.gz" in report["errors"][0]


def test_corrupt_gzip_term_file_is_malformed(tmp_path: Path) -> None:
    evidence, cell = _one_cell(tmp_path)
    (files,) = [evidence.terms(cell, "baseline", [[((0, 1), 0.25, 0.0)]])]
    data = Path(files[0]).read_bytes()
    Path(files[0]).write_bytes(data[: len(data) // 2])
    extra = {"term_files": files, "global_term_count": 1}
    manifest = evidence.manifest(
        [
            evidence.passing(
                cell,
                baseline={"validation_extra": extra},
                candidate={"validation_extra": extra},
            )
        ]
    )

    assert evidence.compare(manifest)[0] == 2


def test_replay_stream_comparison_needs_no_second_file(tmp_path: Path) -> None:
    driver = _driver()
    own = tmp_path / "rank-00000.jsonl.gz"
    driver._write_terms(own, {(0, 1): 0.5 + 0j, (2, 3): -0.25 + 0j})

    same = driver.compare_sorted_streams(
        driver._read_terms(own),
        driver._sorted_items({(2, 3): -0.25 + 0j, (0, 1): 0.5 + 0j}),
    )
    moved = driver.compare_sorted_streams(
        driver._read_terms(own),
        driver._sorted_items({(0, 1): 0.5 + 0j, (4, 5): -0.25 + 0j}),
    )

    assert same["ok"] is True
    assert moved["ok"] is False
    assert list(tmp_path.iterdir()) == [own]
