#!/usr/bin/env python3
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

"""Evidence driver for the rank-local OpenMP parity campaign.

A thin execution/validation wrapper around the existing benchmark suite (``benches/``), not a
benchmark framework. It has exactly three modes:

``observe``
    One fresh-process measurement of one campaign cell. The default writes
    ``DIR/ARM/CELL/SAMPLE/timed.json``: a historical pytest node run in this process (raw pytest
    artifacts kept under ``SAMPLE/timed/``), or the driver-owned
    ``driver::pare_functional_construct[<picture>]`` measurement. ``--whole-process`` instead
    writes ``DIR/ARM/CELL/SAMPLE/construction.json`` from a separate, untimed process that opens
    one memory window before any input is built.
``validate``
    A separate untimed process that rebuilds the cell's workload and records its numerical
    products (energy, full gradient, decoded per-rank term files) as
    ``DIR/ARM/CELL/validation-<run>.json``.
``compare``
    Joins baseline/candidate evidence against a frozen campaign inventory and applies the five
    parity gates -- runtime, operation peak sum/max, construction peak sum/max -- each median
    ratio <= 1.00, plus the numerical comparison of validation products. Exit 0: every cell
    passes; 1: an unmet gate or incomplete coverage; 2: malformed or incompatible evidence.

The launcher, allocation, binding and environment are external: ``observe`` and ``validate``
run under whatever ``mpiexec`` (MPI builds) or direct launch (MPI-off builds) supplies them, and
check what they observe against the campaign cell and the placement evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import re
import socket
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]

ARMS = ("baseline", "candidate")
ARM_SHAPES = {"baseline": "partitions", "candidate": "openmp"}
OPERATIONS = (
    "build_graph",
    "propagate",
    "energy",
    "gradient",
    "pare_functional_construct",
)
MEASUREMENT_KINDS = ("pytest", "driver")
ROUTING_MODES = ("linear", "splitmix")
PARE_FUNCTIONAL = "expectation_value_and_gradient_functional"
PARE_THRESHOLD = 1e-10

#: Default of ``monoprop_ROUTE_SEED`` (``routing::kDefaultSeed``).
DEFAULT_ROUTE_SEED = 0x5DEE_CE66_D0C6_2517

#: Observations per arm/cell: five initially, ten after the prescribed repetition.
INITIAL_SAMPLES = 5
REPEATED_SAMPLES = 10

#: Term files are gzip-compressed JSON Lines; level 1 cuts them ~4-16x at several hundred MB/s.
TERM_SUFFIX = ".jsonl.gz"
TERM_GZIP_LEVEL = 1

#: ``test_utils::near`` from cpp/tests/TestUtilities.h.in, applied componentwise.
ABS_TOL = 1e-9
REL_TOL = 1e-7

EXIT_PASS = 0
EXIT_UNMET = 1
EXIT_MALFORMED = 2

#: The five parity gates: (name, artifact kind, field).
GATES = (
    ("runtime", "timed", "runtime_seconds"),
    ("operation_peak_sum", "timed", "peak_sum_bytes"),
    ("operation_peak_max", "timed", "peak_max_bytes"),
    ("construction_peak_sum", "construction", "construction_peak_sum_bytes"),
    ("construction_peak_max", "construction", "construction_peak_max_bytes"),
)
#: Reported, never gated: the setup-spanning outer window of the timed process.
DIAGNOSTICS = (
    ("outer_peak_sum", "outer_peak_sum_bytes"),
    ("outer_peak_max", "outer_peak_max_bytes"),
)

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RANDOM_FIELDS = (
    "gen_length",
    "obs_terms",
    "num_generators",
    "num_modes",
    "cutoff",
    "seed",
    "lower_atol",
)
_PICTURES = ("heisenberg", "schrodinger")
_NODE = re.compile(
    r"^(?:bench_(?P<file>random|models)\.py::test_(?P<family>random|model)_(?P<op>[a-z_]+)"
    r"|driver::(?P<driver>pare_functional_construct))\[(?P<param>[a-z]+)\]$"
)


class EvidenceError(Exception):
    """Malformed, incompatible or unverifiable evidence (exit 2)."""


# ------------------------------------------------------------------------------------ encoding


def canonical(value: object) -> bytes:
    """Return the canonical JSON encoding that configuration digests hash."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: object) -> str:
    """Return the SHA-256 of ``value``'s canonical JSON encoding."""
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the exact bytes of ``path``."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def load_json(path: Path, what: str) -> Any:  # noqa: ANN401 - arbitrary JSON
    """Return the JSON document at ``path``, raising :class:`EvidenceError` if unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"cannot read {what} {path}: {exc}"
        raise EvidenceError(msg) from exc


def require_slug(value: object, what: str) -> str:
    """Return ``value`` if it is a path-safe slug."""
    if not isinstance(value, str) or not _SLUG.match(value):
        msg = f"{what} {value!r} is not a path-safe slug ([A-Za-z0-9._-], no leading symbol)"
        raise EvidenceError(msg)
    return value


def _require(mapping: object, keys: Iterable[str], what: str) -> dict[str, Any]:
    """Return ``mapping`` if it is a dict containing every key in ``keys``."""
    if not isinstance(mapping, dict):
        msg = f"{what} is not a JSON object"
        raise EvidenceError(msg)
    missing = [key for key in keys if key not in mapping]
    if missing:
        msg = f"{what} lacks {missing}"
        raise EvidenceError(msg)
    return mapping


def near(a: float, b: float) -> bool:
    """Return whether ``a`` and ``b`` agree under ``test_utils::near``."""
    return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b))


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Return a kernel-style CPU list (``0-3,8``) for ``cpus``."""
    ordered = sorted(set(cpus))
    runs: list[str] = []
    start = prev = None
    for cpu in ordered:
        if start is None:
            start = prev = cpu
        elif cpu == prev + 1:
            prev = cpu
        else:
            runs.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = cpu
    if start is not None:
        runs.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(runs)


# ----------------------------------------------------------------------------------- workloads


@dataclass(frozen=True)
class NodeSpec:
    """What a historical pytest node id or a driver node id denotes."""

    node_id: str
    family: str  # "random" or a fixed-model name
    operation: str
    picture: str
    measurement_kind: str

    @property
    def pytest_path(self) -> str:
        """Return the pytest selector for a historical node."""
        return f"benches/{self.node_id}"


def parse_node(node_id: str) -> NodeSpec:
    """Return the family, operation and picture of a campaign node id."""
    match = _NODE.match(node_id) if isinstance(node_id, str) else None
    if match is None:
        msg = f"unrecognised node id {node_id!r}"
        raise EvidenceError(msg)
    param = match["param"]
    if match["driver"]:
        if param not in _PICTURES:
            msg = f"driver node {node_id!r} must name a picture"
            raise EvidenceError(msg)
        return NodeSpec(node_id, "random", match["driver"], param, "driver")
    operation = match["op"]
    if operation not in OPERATIONS[:4] or (match["file"] == "random") != (
        match["family"] == "random"
    ):
        msg = f"unrecognised node id {node_id!r}"
        raise EvidenceError(msg)
    if match["family"] == "random":
        if param not in _PICTURES:
            msg = f"random node {node_id!r} must name a picture"
            raise EvidenceError(msg)
        return NodeSpec(node_id, "random", operation, param, "pytest")
    # The fixed models are Heisenberg-only (bench_models.py).
    return NodeSpec(node_id, param, operation, "heisenberg", "pytest")


def _basis_of(family: str) -> str:
    return "pauli" if family == "pauli" else "majorana"


def check_workload_schema(profile: str, node_id: str, entry: object) -> dict[str, Any]:
    """Validate the schema of one ``profiles[profile][node_id]`` workload entry."""
    what = f"workload {profile}/{node_id}"
    keys = ("operation", "measurement_kind", "basis", "picture", "config", "parameters")
    entry = _require(entry, (*keys, "pare_threshold"), what)
    allowed = {
        "operation": OPERATIONS,
        "measurement_kind": MEASUREMENT_KINDS,
        "basis": ("majorana", "pauli"),
        "picture": _PICTURES,
    }
    for key, values in allowed.items():
        if entry[key] not in values:
            msg = f"{what}: {key} {entry[key]!r} is not one of {values}"
            raise EvidenceError(msg)
    driver = entry["operation"] == "pare_functional_construct"
    if driver != (entry["measurement_kind"] == "driver"):
        msg = f"{what}: only pare_functional_construct is a driver measurement"
        raise EvidenceError(msg)
    if not isinstance(entry["config"], dict):
        msg = f"{what} config is not an object"
        raise EvidenceError(msg)
    params = entry["parameters"]
    if not isinstance(params, list) or not all(_finite_number(p) for p in params):
        msg = f"{what} parameters must be a list of finite numbers"
        raise EvidenceError(msg)
    threshold = entry["pare_threshold"]
    if entry["operation"] in ("build_graph", "propagate") and threshold is not None:
        msg = f"{what}: {entry['operation']} has no pare threshold"
        raise EvidenceError(msg)
    if driver:
        if entry.get("functional") != PARE_FUNCTIONAL or threshold != PARE_THRESHOLD:
            msg = (
                f"{what} must set functional={PARE_FUNCTIONAL!r} and "
                f"pare_threshold={PARE_THRESHOLD}"
            )
            raise EvidenceError(msg)
    elif threshold is not None and threshold != PARE_THRESHOLD:
        msg = f"{what}: pared profiles use pare_threshold={PARE_THRESHOLD}"
        raise EvidenceError(msg)
    return entry


def check_workload_entry(profile: str, node_id: str, entry: object) -> NodeSpec:
    """Validate that a workload entry is runnable: its node id resolves and agrees with it."""
    what = f"workload {profile}/{node_id}"
    entry = check_workload_schema(profile, node_id, entry)
    spec = parse_node(node_id)
    expected = {
        "operation": spec.operation,
        "measurement_kind": spec.measurement_kind,
        "basis": _basis_of(spec.family),
        "picture": spec.picture,
    }
    for key, value in expected.items():
        if entry[key] != value:
            msg = f"{what} has {key}={entry[key]!r}, but its node id implies {value!r}"
            raise EvidenceError(msg)
    # bench_random.py builds every random propagator with the default lower_atol=None.
    random_pytest = spec.family == "random" and spec.measurement_kind == "pytest"
    if random_pytest and entry["config"].get("lower_atol") is not None:
        msg = f"{what}: the random benchmarks build with lower_atol=None"
        raise EvidenceError(msg)
    return spec


def pytest_flags(spec: NodeSpec, entry: dict[str, Any]) -> list[str]:
    """Return the benchmark-suite CLI options that reproduce ``entry``."""
    config = entry["config"]
    flags: list[str] = []
    if spec.family == "random":
        unknown = set(config) - set(_RANDOM_FIELDS)
        if unknown:
            msg = f"random config has unknown fields {sorted(unknown)}"
            raise EvidenceError(msg)
        for name in _RANDOM_FIELDS:
            if name == "lower_atol":
                continue
            flags.append(f"--{name.replace('_', '-')}={config[name]}")
    else:
        for name, value in config.items():
            text = repr(value) if isinstance(value, float) else str(value)
            flags.append(f"--{spec.family}-{name.replace('_', '-')}={text}")
    if entry["pare_threshold"] is not None:
        flags.append(f"--pare-threshold={entry['pare_threshold']!r}")
    return flags


# ------------------------------------------------------------------------------------ campaign


@dataclass
class Campaign:
    """A frozen campaign inventory and its digest-checked workload file."""

    path: Path
    sha256: str
    workloads_path: Path
    workloads_sha256: str
    workloads: dict[str, Any]
    cells: dict[str, dict[str, Any]]

    def entry(self, cell: dict[str, Any]) -> dict[str, Any]:
        """Return the workload entry a cell resolves to."""
        return self.workloads["profiles"][cell["profile"]][cell["node_id"]]


def _check_identity(identity: object, what: str) -> dict[str, Any]:
    keys = (
        "has_mpi",
        "ranks",
        "threads",
        "index_bits",
        "cpu_allocation",
        "compiler_flags",
    )
    identity = _require(identity, (*keys, "config", "routing"), what)
    if not isinstance(identity["has_mpi"], bool):
        msg = f"{what}: has_mpi must be a boolean"
        raise EvidenceError(msg)
    for key in ("ranks", "threads"):
        value = identity[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            msg = f"{what}: {key} must be a positive integer"
            raise EvidenceError(msg)
    if identity["index_bits"] != 32:
        msg = f"{what}: index_bits must be 32 (fixed uint32_t TermIndex)"
        raise EvidenceError(msg)
    if not identity["has_mpi"] and identity["ranks"] != 1:
        msg = f"{what}: an MPI-off cell has R=1, not {identity['ranks']}"
        raise EvidenceError(msg)
    routing = _require(identity["routing"], ("mode", "bits", "seed"), f"{what} routing")
    if routing["mode"] not in ROUTING_MODES:
        msg = f"{what}: routing mode must be one of {ROUTING_MODES}"
        raise EvidenceError(msg)
    if routing != expected_routing(routing["mode"], identity["ranks"], routing["seed"]):
        msg = f"{what}: routing {routing} is not what mode/ranks/seed derive"
        raise EvidenceError(msg)
    return identity


def expected_routing(mode: str, ranks: int, seed: object) -> dict[str, Any]:
    """Return the routing identity the engine derives for ``mode`` at ``ranks``.

    Linear routing takes log2(R) rank bits and needs power-of-two R; splitmix and R=1 have none.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 64:
        msg = f"routing seed {seed!r} is not a uint64"
        raise EvidenceError(msg)
    if mode == "linear" and ranks & (ranks - 1):
        msg = f"linear routing cannot route R={ranks}; successful R={ranks} cells use splitmix"
        raise EvidenceError(msg)
    bits = ranks.bit_length() - 1 if mode == "linear" and ranks > 1 else 0
    return {"mode": mode, "bits": bits, "seed": seed}


def _load_workloads(
    campaign_path: Path, data: dict[str, Any]
) -> tuple[Path, str, dict]:
    """Return the digest-checked workload file a campaign references."""
    workloads_path = (campaign_path.parent / data["workloads_path"]).resolve()
    if not workloads_path.is_file():
        msg = f"campaign workloads file {workloads_path} does not exist"
        raise EvidenceError(msg)
    workloads_sha = file_sha256(workloads_path)
    if workloads_sha != data["workloads_sha256"]:
        msg = f"workloads file {workloads_path} does not match the campaign's workloads_sha256"
        raise EvidenceError(msg)
    workloads = _require(
        load_json(workloads_path, "workloads"),
        ("schema_version", "profiles"),
        "workloads",
    )
    profiles = workloads["profiles"]
    if workloads["schema_version"] != SCHEMA_VERSION or not isinstance(profiles, dict):
        msg = "workloads file has an unsupported schema"
        raise EvidenceError(msg)
    return workloads_path, workloads_sha, workloads


def _check_cell(raw: object, workloads: dict[str, Any]) -> dict[str, Any]:
    """Validate one campaign cell against its workload entry."""
    keys = ("id", "profile", "node_id", "operation", "measurement_kind", "identity")
    cell = _require(raw, (*keys, "config_digest", "parameters_digest"), "campaign cell")
    what = f"cell {require_slug(cell['id'], 'cell id')}"
    entry = workloads["profiles"].get(cell["profile"], {}).get(cell["node_id"])
    if entry is None:
        msg = f"{what}: workloads have no {cell['profile']}/{cell['node_id']}"
        raise EvidenceError(msg)
    # Runnability of the node id is checked where a node runs (observe/validate).
    check_workload_schema(cell["profile"], cell["node_id"], entry)
    identity = _check_identity(cell["identity"], what)
    expected = {
        "operation": entry["operation"],
        "measurement_kind": entry["measurement_kind"],
        "config_digest": digest(entry["config"]),
        "parameters_digest": digest(entry["parameters"]),
    }
    for key, value in expected.items():
        if cell[key] != value:
            msg = f"{what}: {key} {cell[key]!r} does not match its workload entry"
            raise EvidenceError(msg)
    if identity["config"] != entry["config"]:
        msg = f"{what}: identity config differs from the workload config"
        raise EvidenceError(msg)
    return cell


def load_campaign(path: Path) -> Campaign:
    """Load and verify a campaign inventory and its workload file."""
    path = path.resolve()
    data = _require(
        load_json(path, "campaign"),
        ("schema_version", "workloads_path", "workloads_sha256", "cells"),
        "campaign",
    )
    if data["schema_version"] != SCHEMA_VERSION:
        msg = f"campaign schema_version {data['schema_version']!r} is not {SCHEMA_VERSION}"
        raise EvidenceError(msg)
    workloads_path, workloads_sha, workloads = _load_workloads(path, data)
    if not isinstance(data["cells"], list) or not data["cells"]:
        msg = "campaign lists no cells"
        raise EvidenceError(msg)
    cells: dict[str, dict[str, Any]] = {}
    seen: dict[str, str] = {}
    for raw in data["cells"]:
        cell = _check_cell(raw, workloads)
        if cell["id"] in cells:
            msg = f"campaign lists cell id {cell['id']!r} twice"
            raise EvidenceError(msg)
        # Duplicates collapse only within one build mode and complete identity; a D=1
        # MPI-enabled cell never stands in for the MPI-off one: has_mpi is part of this key.
        key = digest([cell["profile"], cell["node_id"], cell["identity"]])
        if key in seen:
            msg = f"cell {cell['id']} duplicates cell {seen[key]} (same profile, node, identity)"
            raise EvidenceError(msg)
        seen[key] = cell["id"]
        cells[cell["id"]] = cell
    return Campaign(
        path=path,
        sha256=file_sha256(path),
        workloads_path=workloads_path,
        workloads_sha256=workloads_sha,
        workloads=workloads,
        cells=cells,
    )


# ----------------------------------------------------------------------------------- placement


def check_placement(
    placement: object,
    *,
    arm: str,
    cell_id: str,
    binary_hash: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate external placement evidence for one cell/arm/binary.

    The runtime shape is fixed by the arm; the declared settings must resolve the campaign's
    thread count through the shared preflight, and the observed shape must be the full team on
    every rank of the campaign allocation. A limited team cannot certify a full-team cell.
    """
    keys = (
        "schema_version",
        "artifact_kind",
        "arm",
        "cell_id",
        "binary_hash",
        "identity",
    )
    placement = _require(
        placement,
        (*keys, "runtime_shape", "declared", "observed"),
        "placement evidence",
    )
    what = f"placement evidence for {arm}/{cell_id}"
    checks = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "placement",
        "arm": arm,
        "cell_id": cell_id,
        "binary_hash": binary_hash,
        "identity": identity,
        "runtime_shape": ARM_SHAPES[arm],
    }
    for key, value in checks.items():
        if placement[key] != value:
            msg = f"{what}: {key} does not match (got {placement[key]!r})"
            raise EvidenceError(msg)
    _check_declared_placement(placement["declared"], arm, identity, what)
    _check_observed_placement(placement["observed"], identity, what)
    return placement


def _check_declared_placement(
    declared: object, arm: str, identity: dict[str, Any], what: str
) -> None:
    """Require the declared launch shape and settings to resolve the campaign cell's shape."""
    from monoprop_bench_tools.preflight import (  # noqa: PLC0415 - optional at import
        PreflightError,
        declared_shape,
    )

    declared = _require(
        declared, ("ranks", "threads", "cpu_allocation", "settings"), what
    )
    for key in ("ranks", "threads", "cpu_allocation"):
        if declared[key] != identity[key]:
            msg = f"{what}: declared {key} {declared[key]!r} differs from {identity[key]!r}"
            raise EvidenceError(msg)
    if not isinstance(declared["settings"], dict):
        msg = f"{what}: declared settings must be an object"
        raise EvidenceError(msg)
    try:
        declared_shape(
            ARM_SHAPES[arm],
            declared["settings"],
            ranks=identity["ranks"],
            has_mpi=identity["has_mpi"],
            expected_threads=identity["threads"],
        )
    except PreflightError as exc:
        msg = f"{what}: {exc}"
        raise EvidenceError(msg) from exc


def _check_observed_placement(
    observed: object, identity: dict[str, Any], what: str
) -> None:
    """Require the observed shape to be the full team on every rank of the cell's allocation."""
    observed = _require(
        observed, ("has_mpi", "ranks", "threads_per_rank", "cpu_allocation"), what
    )
    for key in ("has_mpi", "ranks", "cpu_allocation"):
        if observed[key] != identity[key] or type(observed[key]) is not type(
            identity[key]
        ):
            msg = f"{what}: observed {key} {observed[key]!r} differs from {identity[key]!r}"
            raise EvidenceError(msg)
    teams = observed["threads_per_rank"]
    if not isinstance(teams, list) or len(teams) != identity["ranks"]:
        msg = f"{what}: threads_per_rank must list one team size per rank"
        raise EvidenceError(msg)
    if any(team != identity["threads"] for team in teams):
        msg = (
            f"{what}: observed teams {teams} are not the full {identity['threads']}-thread "
            "team; a limited-team diagnostic cannot certify this cell"
        )
        raise EvidenceError(msg)


# ---------------------------------------------------------------------------- process context


@dataclass
class Context:
    """The observed process: communicator, build mode, binary and declared shape."""

    comm: Any
    rank: int
    size: int
    has_mpi: bool
    binary_path: str
    binary_hash: str
    declared: dict[str, Any]
    cpu_allocation: str
    placement_path: str
    placement_sha256: str

    def barrier(self) -> None:
        """Synchronize ranks (serial: no-op)."""
        if self.comm is not None and self.size > 1:
            self.comm.Barrier()

    def reduce(self, value: float, op: str) -> float:
        """Reduce ``value`` over ranks with ``op`` (``sum``, ``max`` or ``min``)."""
        if self.comm is None or self.size == 1:
            return value
        mpi = sys.modules["mpi4py.MPI"]
        return self.comm.allreduce(
            value, op={"sum": mpi.SUM, "max": mpi.MAX, "min": mpi.MIN}[op]
        )

    def gather(self, value: object) -> list[Any]:
        """Gather ``value`` to rank 0 (other ranks receive ``[]``)."""
        if self.comm is None or self.size == 1:
            return [value]
        gathered = self.comm.gather(value, root=0)
        return gathered if gathered is not None else []

    def spread(self, value: int) -> dict[str, int]:
        """Return ``value`` reduced to its rank sum and maximum."""
        return {
            "sum": int(self.reduce(value, "sum")),
            "max": int(self.reduce(value, "max")),
        }

    def all_true(self, *, flag: bool) -> bool:
        """Return whether ``flag`` holds on every rank."""
        return self.reduce(int(flag), "min") == 1


def _routing_from_env(ranks: int) -> dict[str, Any]:
    """Return the routing identity the launch environment selects (engine defaults applied)."""
    # The library's own variable names; unset or empty selects the engine defaults.
    mode = os.environ.get("monoprop_ROUTING") or "linear"  # noqa: SIM112
    seed_text = os.environ.get("monoprop_ROUTE_SEED") or str(DEFAULT_ROUTE_SEED)  # noqa: SIM112
    if mode not in ROUTING_MODES or not seed_text.isdigit():
        msg = (
            f"invalid routing environment monoprop_ROUTING={mode!r} seed={seed_text!r}"
        )
        raise EvidenceError(msg)
    return expected_routing(mode, ranks, int(seed_text))


def open_context(args: argparse.Namespace, cell: dict[str, Any]) -> Context:
    """Import the extension, gate MPI on its build mode and check the declared shape.

    Raises:
        EvidenceError: If the process contradicts the cell, its arm, or the placement evidence.
    """
    from monoprop_bench_tools.preflight import (  # noqa: PLC0415
        PreflightError,
        declared_shape,
        import_mpi,
        require_arm_shape,
        settings_snapshot,
    )

    import monoprop  # noqa: PLC0415

    identity = cell["identity"]
    try:
        require_arm_shape(args.arm, args.runtime_shape)
    except PreflightError as exc:
        raise EvidenceError(str(exc)) from exc
    has_mpi = getattr(monoprop, "has_mpi", None)
    # Gated before any mpi4py import: an MPI-off process must never initialize MPI.
    mpi = import_mpi(has_mpi=bool(has_mpi))
    comm = mpi.COMM_WORLD if mpi is not None else None
    rank = 0 if comm is None else comm.Get_rank()
    size = 1 if comm is None else comm.Get_size()
    try:
        declared = declared_shape(
            args.runtime_shape,
            os.environ,
            ranks=size,
            has_mpi=has_mpi,
            expected_has_mpi=identity["has_mpi"],
            expected_ranks=identity["ranks"],
            expected_threads=identity["threads"],
        )
    except PreflightError as exc:
        raise EvidenceError(str(exc)) from exc
    if _routing_from_env(size) != identity["routing"]:
        msg = f"routing environment {_routing_from_env(size)} differs from {identity['routing']}"
        raise EvidenceError(msg)

    binary = Path(monoprop._core.__file__)
    binary_hash = file_sha256(binary)
    local_cpus = sorted(os.sched_getaffinity(0))
    everyone = (
        local_cpus
        if comm is None
        else [c for cpus in comm.allgather(local_cpus) for c in cpus]
    )
    cpu_allocation = format_cpu_list(everyone)
    if cpu_allocation != identity["cpu_allocation"]:
        msg = f"process allocation {cpu_allocation!r} differs from {identity['cpu_allocation']!r}"
        raise EvidenceError(msg)

    placement_path = Path(args.placement).resolve()
    placement = check_placement(
        load_json(placement_path, "placement evidence"),
        arm=args.arm,
        cell_id=cell["id"],
        binary_hash=binary_hash,
        identity=identity,
    )
    if placement["declared"]["settings"] != settings_snapshot(os.environ):
        msg = "the launch settings differ from those the placement evidence declares"
        raise EvidenceError(msg)
    return Context(
        comm=comm,
        rank=rank,
        size=size,
        has_mpi=bool(has_mpi),
        binary_path=str(binary),
        binary_hash=binary_hash,
        declared=declared,
        cpu_allocation=cpu_allocation,
        placement_path=str(placement_path),
        placement_sha256=file_sha256(placement_path),
    )


def common_fields(
    kind: str,
    args: argparse.Namespace,
    campaign: Campaign,
    cell: dict[str, Any],
    ctx: Context,
) -> dict[str, Any]:
    """Return the provenance every artifact carries."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": kind,
        "run_id": f"{kind}-{uuid.uuid4().hex}",
        "arm": args.arm,
        "runtime_shape": args.runtime_shape,
        "cell_id": cell["id"],
        "campaign_path": str(campaign.path),
        "campaign_sha256": campaign.sha256,
        "workloads_sha256": campaign.workloads_sha256,
        "profile": cell["profile"],
        "node_id": cell["node_id"],
        "operation": cell["operation"],
        "measurement_kind": cell["measurement_kind"],
        "identity": cell["identity"],
        "config_digest": cell["config_digest"],
        "parameters_digest": cell["parameters_digest"],
        "binary_path": ctx.binary_path,
        "binary_hash": ctx.binary_hash,
        "has_mpi": ctx.has_mpi,
        "observed": {"ranks": ctx.size, "cpu_allocation": ctx.cpu_allocation},
        "declared": ctx.declared,
        "placement_path": ctx.placement_path,
        "placement_sha256": ctx.placement_sha256,
        "host": socket.gethostname(),
        "driver_sha256": file_sha256(Path(__file__)),
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Create ``path`` with ``value``; an existing artifact is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=1, allow_nan=False)
        handle.write("\n")


# -------------------------------------------------------------------------------- measurements


@dataclass
class Problem:
    """A workload rebuilt with the existing model builders."""

    make_propagator: Callable[[], tuple[Any, Any]]  # () -> (propagator, circuit)
    steps: int
    parameters: list[float]


def build_problem(spec: NodeSpec, entry: dict[str, Any], comm: Any) -> Problem:  # noqa: ANN401
    """Rebuild a workload's inputs and check its parameter vector against the manifest.

    Random inputs are generated here (before any propagator exists); fixed models construct their
    circuit inside ``make_propagator``, exactly as the benchmark fixtures do.
    """
    from monoprop_bench_tools.models import (  # noqa: PLC0415
        MODELS,
        build_random_propagator,
        make_random_problem,
    )

    config = entry["config"]
    if spec.family == "random":
        problem = make_random_problem(
            **{k: v for k, v in config.items() if k != "lower_atol"}
        )
        parameters = [float(p) for p in problem.parameters]
        schrodinger = spec.picture == "schrodinger"

        def make() -> tuple[Any, Any]:
            return build_random_propagator(
                problem,
                comm=comm,
                lower_atol=config["lower_atol"],
                schrodinger=schrodinger,
            )

        steps = 1
    else:
        config_cls, build_fn, steps_fn = MODELS[spec.family]
        model_config = config_cls(**config)
        steps = int(steps_fn(model_config))

        def make() -> tuple[Any, Any]:
            return build_fn(model_config, comm=comm)

        # The builders return the circuit only beside its (single-term) propagator.
        propagator, circuit = make()
        parameters = [float(p) for p in circuit.parameters] * steps
        del propagator, circuit
    if parameters != [float(p) for p in entry["parameters"]]:
        msg = f"{spec.node_id}: rebuilt parameters differ from the workload's parameter vector"
        raise EvidenceError(msg)
    return Problem(make_propagator=make, steps=steps, parameters=parameters)


def timed_pare_construct(
    problem: Problem,
    ctx: Context,
    *,
    window_factory: Callable[..., Any],
    clock: Callable[[], float] = time.perf_counter,
    events: list[str] | None = None,
) -> dict[str, Any]:
    """Measure ``expectation_value_and_gradient_functional(1e-10)`` on a ready graph.

    The outer window (the driver analogue of pytest's ``record_memory``) opens after the random
    inputs exist and before the propagator is built. The graph is complete before the operation
    window, the entry barrier and the timer; the timer covers only the functional construction and
    the completion barrier. The returned callable is never evaluated here and stays alive until
    both windows have closed.
    """
    log = events if events is not None else []
    outer = window_factory()
    outer.start()
    log.append("outer-open")
    propagator, circuit = problem.make_propagator()
    for _ in range(problem.steps):
        propagator.build_graph(circuit)
    log.append("graph-built")
    operation = window_factory(settle=False)
    operation.start()
    log.append("operation-open")
    ctx.barrier()  # untimed entry barrier
    start = clock()
    log.append("timer-start")
    functional = propagator.expectation_value_and_gradient_functional(PARE_THRESHOLD)
    ctx.barrier()  # completion is part of the measured makespan
    elapsed = clock() - start
    log.append("timer-stop")
    operation.stop()
    log.append("operation-close")
    outer.stop()
    log.append("outer-close")
    runtime = ctx.reduce(elapsed, "max")
    op_spread = {
        "peak": ctx.spread(operation.peak_bytes),
        "floor": ctx.spread(operation.baseline_bytes),
        "delta": ctx.spread(operation.delta_bytes),
    }
    outer_spread = ctx.spread(outer.peak_bytes)
    result = {
        "runtime_seconds": runtime,
        "peak_sum_bytes": op_spread["peak"]["sum"],
        "peak_max_bytes": op_spread["peak"]["max"],
        "op_exact": ctx.all_true(flag=operation.exact),
        "outer_peak_sum_bytes": outer_spread["sum"],
        "outer_peak_max_bytes": outer_spread["max"],
        "outer_exact": ctx.all_true(flag=outer.exact),
        "operation_floor_sum_bytes": op_spread["floor"]["sum"],
        "operation_floor_max_bytes": op_spread["floor"]["max"],
        "operation_delta_sum_bytes": op_spread["delta"]["sum"],
        "operation_delta_max_bytes": op_spread["delta"]["max"],
        "persistent_bytes": {
            "operator": int(
                ctx.reduce(propagator._simulator.operator_memory_bytes(), "sum")
            ),
            "graph": int(ctx.reduce(propagator._simulator.graph_memory_bytes(), "sum")),
        },
    }
    # Only now may the callable go: it had to outlive both windows.
    del functional
    log.append("callable-released")
    return result


def construction_worker(
    spec: NodeSpec,
    entry: dict[str, Any],
    ctx: Context,
    *,
    window_factory: Callable[..., Any],
    build: Callable[[], Problem],
    events: list[str] | None = None,
) -> dict[str, Any]:
    """Measure one whole construction, inputs included, in an untimed fresh process.

    One window opens before the random inputs are generated; the propagator, graph and callable
    are built and the operation runs; every output stays alive until the window closes.
    """
    log = events if events is not None else []
    window = window_factory()
    window.start()
    log.append("window-open")
    problem = build()
    propagator, circuit = problem.make_propagator()
    retained: list[Any] = [problem, propagator, circuit]
    threshold = entry["pare_threshold"]
    if spec.operation == "propagate":
        for _ in range(problem.steps):
            propagator.propagate(circuit)
    else:
        for _ in range(problem.steps):
            propagator.build_graph(circuit)
    if spec.operation == "energy":
        functional = propagator.expectation_value_functional(threshold)
        retained += [functional, functional(problem.parameters)]
    elif spec.operation == "gradient":
        functional = propagator.expectation_value_and_gradient_functional(threshold)
        retained += [functional, functional(problem.parameters)]
    elif spec.operation == "pare_functional_construct":
        retained.append(propagator.expectation_value_and_gradient_functional(threshold))
    log.append("operation-done")
    window.stop()
    log.append("window-close")
    peak, floor, delta = (
        ctx.spread(window.peak_bytes),
        ctx.spread(window.baseline_bytes),
        ctx.spread(window.delta_bytes),
    )
    result = {
        "construction_exact": ctx.all_true(flag=window.exact),
        "construction_peak_sum_bytes": peak["sum"],
        "construction_peak_max_bytes": peak["max"],
        "construction_floor_sum_bytes": floor["sum"],
        "construction_floor_max_bytes": floor["max"],
        "construction_delta_sum_bytes": delta["sum"],
        "construction_delta_max_bytes": delta["max"],
    }
    del retained
    log.append("outputs-released")
    return result


def timed_pytest(
    spec: NodeSpec,
    entry: dict[str, Any],
    args: argparse.Namespace,
    ctx: Context,
    raw: Path,
) -> dict[str, Any]:
    """Run one historical pytest node in this process and read back its raw artifacts."""
    import pytest  # noqa: PLC0415

    label = "timed"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ["monoprop_BENCH_LABEL"] = label  # noqa: SIM112 - the suite's names
    os.environ["monoprop_BENCH_RESULTS"] = str(raw)  # noqa: SIM112
    build_mode = "mpi" if ctx.has_mpi else "mpi-off"
    time_json = raw / f"time-{label}.json"
    argv = [
        spec.pytest_path,
        "-p",
        "no:cacheprovider",
        "-o",
        "filterwarnings=default",
        "--bench-rounds=1",
        f"--runtime-shape={args.runtime_shape}",
        f"--build-mode={build_mode}",
        *pytest_flags(spec, entry),
        f"--benchmark-json={time_json}",
    ]
    cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        status = int(pytest.main(argv))
    finally:
        os.chdir(cwd)
    if status != 0:
        msg = f"pytest exited {status} for {spec.node_id}; a skipped or failed node is not a pass"
        raise EvidenceError(msg)
    if ctx.rank != 0:
        return {}
    results = load_json(raw / f"{label}.json", "raw benchmark results")
    timings = load_json(time_json, "raw benchmark timings")
    benchmarks = [
        b for b in timings.get("benchmarks", []) if b["fullname"].endswith(spec.node_id)
    ]
    if len(benchmarks) != 1 or benchmarks[0]["stats"].get("rounds") != 1:
        msg = f"raw timings do not hold exactly one single-round run of {spec.node_id}"
        raise EvidenceError(msg)
    node = spec.node_id

    def field(section: str) -> Any:  # noqa: ANN401 - raw JSON value
        if node not in results.get(section, {}):
            msg = f"raw results lack {section}[{node}]"
            raise EvidenceError(msg)
        return results[section][node]

    peak, floor, delta = field("opmempeak"), field("opmembase"), field("opmemdelta")
    return {
        "runtime_seconds": benchmarks[0]["stats"]["mean"],
        "peak_sum_bytes": peak["sum"],
        "peak_max_bytes": peak["max"],
        "op_exact": field("opmemexact"),
        "outer_peak_sum_bytes": field("memhwm"),
        "outer_peak_max_bytes": field("memhwm_max"),
        "outer_exact": field("memhwmexact"),
        "operation_floor_sum_bytes": floor["sum"],
        "operation_floor_max_bytes": floor["max"],
        "operation_delta_sum_bytes": delta["sum"],
        "operation_delta_max_bytes": delta["max"],
        "persistent_bytes": results.get("opbytes", {}).get(node),
        "terms": results.get("opsize", {}).get(node, {}).get("terms"),
        "raw_dir": str(raw),
        "raw_meta": results.get("meta", {}),
    }


def observe(args: argparse.Namespace) -> int:
    """Run one timed or whole-construction observation of one campaign cell."""
    from monoprop_bench_tools.memory.cpu import HighWaterMark  # noqa: PLC0415

    campaign = load_campaign(Path(args.campaign))
    cell = campaign.cells.get(require_slug(args.cell_id, "cell id"))
    if cell is None:
        msg = f"campaign has no cell {args.cell_id!r}"
        raise EvidenceError(msg)
    sample = require_slug(args.sample_id, "sample id")
    kind = "construction" if args.whole_process else "timed"
    target = (
        Path(args.output).resolve() / args.arm / cell["id"] / sample / f"{kind}.json"
    )
    if target.exists():
        msg = f"{target} exists; artifacts are never overwritten"
        raise EvidenceError(msg)
    ctx = open_context(args, cell)
    common = common_fields(kind, args, campaign, cell, ctx)
    common["sample_id"] = sample
    entry = campaign.entry(cell)
    spec = parse_node(cell["node_id"])
    started = time.time()
    if args.whole_process:
        fields = construction_worker(
            spec,
            entry,
            ctx,
            window_factory=HighWaterMark,
            build=lambda: build_problem(spec, entry, ctx.comm),
        )
    elif spec.measurement_kind == "driver":
        problem = build_problem(spec, entry, ctx.comm)
        fields = timed_pare_construct(problem, ctx, window_factory=HighWaterMark)
    else:
        fields = timed_pytest(spec, entry, args, ctx, target.parent / "timed")
    if ctx.rank == 0:
        _write_exclusive(
            target,
            {
                **common,
                **fields,
                "started": started,
                "wall_seconds": time.time() - started,
            },
        )
    ctx.barrier()
    return EXIT_PASS


# ---------------------------------------------------------------------------------- validation


def _sorted_items(
    terms: dict[tuple[int, ...], complex],
) -> Iterator[tuple[tuple[int, ...], float, float]]:
    """Yield ``(key, real, imag)`` in sorted key order, rejecting non-finite coefficients."""
    for key in sorted(terms):
        value = complex(terms[key])
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            msg = f"non-finite coefficient for term {key}"
            raise EvidenceError(msg)
        yield key, value.real, value.imag


def _require_term_suffix(path: Path) -> None:
    if not path.name.endswith(TERM_SUFFIX):
        msg = f"term file {path} must be gzip-compressed JSON Lines ({TERM_SUFFIX})"
        raise EvidenceError(msg)


def _write_terms(path: Path, terms: dict[tuple[int, ...], complex]) -> tuple[int, bool]:
    """Write ``terms`` as sorted, gzip-compressed JSON Lines; return the count and identity flag.

    Lines are formatted directly rather than through ``json.dumps``: for finite floats ``%r`` is
    the same shortest round-trip repr, so the decompressed text is identical and faster to write.
    The gzip header carries no timestamp or file name, so equal maps give equal bytes.
    """
    _require_term_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=TERM_GZIP_LEVEL, mtime=0
        ) as packed,
        io.TextIOWrapper(packed, encoding="ascii", newline="\n") as handle,
    ):
        write = handle.write
        for key, real, imag in _sorted_items(terms):
            indices = ", ".join(map(str, key))
            write('{"key": [%s], "real": %r, "imag": %r}\n' % (indices, real, imag))  # noqa: UP031
    return len(terms), () in terms


def _evaluate(
    spec: NodeSpec,
    threshold: float | None,
    problem: Problem,
    propagator: Any,  # noqa: ANN401 - either propagator front-end
    circuit: object,
) -> tuple[dict[str, Any], list[float]]:
    """Run the operation and return its scalar products and the export parameter vector."""
    if spec.operation == "propagate":
        for _ in range(problem.steps):
            propagator.propagate(circuit)
        return {"energy": float(propagator.expectation_value())}, []
    for _ in range(problem.steps):
        propagator.build_graph(circuit)
    if spec.operation == "build_graph":
        value, gradient = propagator.expectation_value_and_gradient(problem.parameters)
    elif spec.operation == "energy":
        energy = propagator.expectation_value_functional(threshold)(problem.parameters)
        return {"energy": float(energy)}, problem.parameters
    else:
        functional = propagator.expectation_value_and_gradient_functional(threshold)
        value, gradient = functional(problem.parameters)
    return {
        "energy": float(value),
        "gradient": [float(g) for g in gradient],
    }, problem.parameters


def _export_terms(
    spec: NodeSpec,
    propagator: Any,  # noqa: ANN401 - either propagator front-end
    parameters: list[float],
    ctx: Context,
    term_dir: Path,
) -> dict[str, Any]:
    """Write this rank's decoded term map and return the gathered file list and global count."""
    terms = propagator._simulator.evolved_operator(list(parameters), 0.0)
    path = term_dir / f"rank-{ctx.rank:05d}{TERM_SUFFIX}"
    count, identity = _write_terms(path, terms)
    del terms
    gathered = ctx.gather((str(path), count, identity))
    if ctx.rank != 0:
        return {}
    counts = [c for _, c, _ in gathered]
    # The binding adds the Heisenberg core term on every rank; it counts once globally.
    copies = (
        sum(1 for *_, has in gathered if has) if spec.picture == "heisenberg" else 0
    )
    return {
        "term_files": [p for p, _, _ in gathered],
        "term_counts": counts,
        "global_term_count": sum(counts) - max(0, copies - 1),
    }


def _graph_replay_check(
    entry: dict[str, Any], problem: Problem, ctx: Context, term_dir: Path
) -> dict[str, Any]:
    """Compare graph-free propagation with a graph replay of the same inputs, per rank.

    Only meaningful without coefficient truncation, when both paths apply the cutoff alone. The
    replayed map is streamed against this rank's already written term file, not written again.
    """
    if entry["config"].get("lower_atol") is not None:
        return {"applicable": False}
    replay, circuit = problem.make_propagator()
    for _ in range(problem.steps):
        replay.build_graph(circuit)
    replayed = replay._simulator.evolved_operator(list(problem.parameters), 0.0)
    del replay
    own = term_dir / f"rank-{ctx.rank:05d}{TERM_SUFFIX}"
    local = compare_sorted_streams(_read_terms(own), _sorted_items(replayed))
    del replayed
    return {
        "applicable": True,
        "ok": ctx.all_true(flag=local["ok"]),
        "detail": ctx.gather(local),
    }


def validation_products(
    spec: NodeSpec,
    entry: dict[str, Any],
    problem: Problem,
    ctx: Context,
    term_dir: Path,
) -> dict[str, Any]:
    """Compute the operation's numerical products at the recorded parameters."""
    threshold = entry["pare_threshold"]
    unpared = threshold is None and spec.operation != "pare_functional_construct"
    propagator, circuit = problem.make_propagator()
    products, export_parameters = _evaluate(
        spec, threshold, problem, propagator, circuit
    )
    layers = int(propagator.graph_layers)
    if ctx.reduce(layers, "max") != ctx.reduce(layers, "min"):
        msg = "ranks disagree on the graph layer count"
        raise EvidenceError(msg)
    products |= {"graph_layers": layers, "term_files": []}
    if unpared:
        products |= _export_terms(spec, propagator, export_parameters, ctx, term_dir)
    else:
        products["global_term_count"] = int(ctx.reduce(propagator.size(), "sum"))
    del propagator
    if spec.operation == "propagate":
        products["graph_replay_check"] = _graph_replay_check(
            entry, problem, ctx, term_dir
        )
    return products


def validate(args: argparse.Namespace) -> int:
    """Rebuild a cell's workload in an untimed process and record its numerical products."""
    from monoprop_bench_tools.memory.cpu import HighWaterMark  # noqa: PLC0415

    campaign = load_campaign(Path(args.campaign))
    cell = campaign.cells.get(require_slug(args.cell_id, "cell id"))
    if cell is None:
        msg = f"campaign has no cell {args.cell_id!r}"
        raise EvidenceError(msg)
    ctx = open_context(args, cell)
    common = common_fields("validation", args, campaign, cell, ctx)
    run_id = (
        ctx.comm.bcast(common["run_id"], root=0) if ctx.size > 1 else common["run_id"]
    )
    common["run_id"] = run_id
    base = Path(args.output).resolve() / args.arm / cell["id"]
    target = base / f"validation-{run_id}.json"
    entry = campaign.entry(cell)
    spec = parse_node(cell["node_id"])
    started = time.time()
    with HighWaterMark() as window:
        problem = build_problem(spec, entry, ctx.comm)
        products = validation_products(
            spec, entry, problem, ctx, base / f"{run_id}-terms"
        )
    diagnostics = {
        "wall_seconds": ctx.reduce(time.time() - started, "max"),
        "peak_sum_bytes": int(ctx.reduce(window.peak_bytes, "sum")),
        "peak_max_bytes": int(ctx.reduce(window.peak_bytes, "max")),
        "exact": ctx.all_true(flag=window.exact),
    }
    if ctx.rank == 0:
        artifact = {
            **common,
            "basis": entry["basis"],
            "picture": entry["picture"],
            "pare_threshold": entry["pare_threshold"],
            "parameters": problem.parameters,
            **products,
            "diagnostics": diagnostics,
        }
        _write_exclusive(target, artifact)
    ctx.barrier()
    return EXIT_PASS


# ------------------------------------------------------------------------- numerical comparison


def _read_terms(path: Path) -> Iterator[tuple[tuple[int, ...], float, float]]:
    """Yield ``(key, real, imag)`` from a sorted term file, rejecting disorder and bad values."""
    _require_term_suffix(path)
    previous: tuple[int, ...] | None = None
    try:
        with gzip.open(path, "rt", encoding="ascii") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    key = tuple(int(i) for i in record["key"])
                    real, imag = float(record["real"]), float(record["imag"])
                except (ValueError, KeyError, TypeError) as exc:
                    msg = f"{path}:{number}: malformed term record"
                    raise EvidenceError(msg) from exc
                if previous is not None and key <= previous:
                    msg = f"{path}:{number}: term keys are not strictly increasing"
                    raise EvidenceError(msg)
                if not (math.isfinite(real) and math.isfinite(imag)):
                    msg = f"{path}:{number}: non-finite coefficient"
                    raise EvidenceError(msg)
                previous = key
                yield key, real, imag
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        msg = f"cannot read term file {path}: {exc}"
        raise EvidenceError(msg) from exc


def merged_terms(
    paths: list[str], picture: str
) -> Iterator[tuple[tuple[int, ...], float, float]]:
    """Merge per-rank term files into the global map.

    Ownership is unique: a key on two ranks fails, except the Heisenberg identity/core term, which
    the binding replicates on every rank; it is kept once after checking the copies agree. The
    Schrödinger identity is an ordinary owned term.
    """
    streams = [_read_terms(Path(p)) for p in paths]
    merged = heapq.merge(*streams, key=lambda term: term[0])
    pending: list[tuple[tuple[int, ...], float, float]] = []

    def flush() -> Iterator[tuple[tuple[int, ...], float, float]]:
        if not pending:
            return
        key = pending[0][0]
        if len(pending) > 1:
            if key != () or picture != "heisenberg":
                raise NumericalMismatchError(
                    f"term {list(key)} is owned by {len(pending)} ranks"
                )
            real, imag = pending[0][1], pending[0][2]
            if not all(near(r, real) and near(i, imag) for _, r, i in pending):
                raise NumericalMismatchError(
                    "replicated Heisenberg identity copies disagree"
                )
        yield pending[0]

    for term in merged:
        if pending and term[0] != pending[0][0]:
            yield from flush()
            pending.clear()
        pending.append(term)
    yield from flush()


class NumericalMismatchError(Exception):
    """The validation products of two arms disagree (a failed numerical check)."""


def compare_sorted_streams(
    left: Iterable[tuple[tuple[int, ...], float, float]],
    right: Iterable[tuple[tuple[int, ...], float, float]],
) -> dict[str, Any]:
    """Compare two sorted term streams: equal key sets, componentwise ``near`` values."""
    missing = extra = mismatched = count = 0
    worst = 0.0
    first: list[str] = []
    a_iter, b_iter = iter(left), iter(right)
    a, b = next(a_iter, None), next(b_iter, None)
    while a is not None or b is not None:
        if b is None or (a is not None and a[0] < b[0]):
            missing += 1
            first.append(f"only in first: {list(a[0])}")
            a = next(a_iter, None)
        elif a is None or b[0] < a[0]:
            extra += 1
            first.append(f"only in second: {list(b[0])}")
            b = next(b_iter, None)
        else:
            count += 1
            diff = max(abs(a[1] - b[1]), abs(a[2] - b[2]))
            worst = max(worst, diff)
            if not (near(a[1], b[1]) and near(a[2], b[2])):
                mismatched += 1
                first.append(f"value differs at {list(a[0])}")
            a, b = next(a_iter, None), next(b_iter, None)
    return {
        "ok": missing == extra == mismatched == 0,
        "common_terms": count,
        "only_first": missing,
        "only_second": extra,
        "value_mismatches": mismatched,
        "max_abs_diff": worst,
        "examples": first[:5],
    }


def _check_validation(artifact: dict[str, Any], what: str) -> None:
    """Require a validation artifact's products to be present and finite."""
    operation = artifact["operation"]
    pared = (
        artifact.get("pare_threshold") is not None
        or operation == "pare_functional_construct"
    )
    if not _finite_number(artifact.get("energy")):
        msg = f"{what}: energy is missing or non-finite"
        raise NumericalMismatchError(msg)
    if operation in ("build_graph", "gradient", "pare_functional_construct"):
        gradient = artifact.get("gradient")
        if not isinstance(gradient, list) or not all(
            _finite_number(g) for g in gradient
        ):
            msg = f"{what}: full gradient is missing or non-finite"
            raise NumericalMismatchError(msg)
    files = artifact.get("term_files")
    if not isinstance(files, list):
        msg = f"{what}: term_files must be a list"
        raise EvidenceError(msg)
    if pared and files:
        # A pared callable owns a subgraph, not an independently exported operator.
        msg = f"{what}: pared evidence must not carry term files"
        raise EvidenceError(msg)
    replay = artifact.get("graph_replay_check")
    if replay is not None and replay.get("applicable") and not replay.get("ok"):
        msg = f"{what}: graph replay disagrees with graph-free propagation"
        raise NumericalMismatchError(msg)


def _compare_gradients(baseline: dict[str, Any], candidate: dict[str, Any]) -> int:
    """Require equal-length, componentwise-near gradients; return their length."""
    left, right = baseline.get("gradient"), candidate.get("gradient")
    if left is None or right is None or len(left) != len(right):
        msg = "gradient missing or of different length"
        raise NumericalMismatchError(msg)
    bad = [
        i for i, (x, y) in enumerate(zip(left, right, strict=True)) if not near(x, y)
    ]
    if bad:
        msg = f"gradient differs at parameters {bad[:5]}"
        raise NumericalMismatchError(msg)
    return len(left)


def _compare_term_maps(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Merge each arm's per-rank term files and compare the two global maps."""
    if not (baseline["term_files"] and candidate["term_files"]):
        msg = "only one arm exported a term map"
        raise NumericalMismatchError(msg)
    result = compare_sorted_streams(
        merged_terms(baseline["term_files"], baseline["picture"]),
        merged_terms(candidate["term_files"], candidate["picture"]),
    )
    if not result["ok"]:
        msg = f"global term maps differ: {result['examples']}"
        raise NumericalMismatchError(msg)
    for side, artifact in (("baseline", baseline), ("candidate", candidate)):
        if artifact.get("global_term_count") != result["common_terms"]:
            msg = f"{side} global_term_count disagrees with its merged term files"
            raise NumericalMismatchError(msg)
    return result


def compare_validations(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare two arms' validation products of the identical operation.

    Both artifacts already joined on the same parameters digest.
    """
    for side, artifact in (("baseline", baseline), ("candidate", candidate)):
        _check_validation(artifact, f"{side} validation {artifact['run_id']}")
    if not near(baseline["energy"], candidate["energy"]):
        msg = f"energy {baseline['energy']!r} vs {candidate['energy']!r}"
        raise NumericalMismatchError(msg)
    detail: dict[str, Any] = {}
    if "gradient" in baseline or "gradient" in candidate:
        detail["gradient_length"] = _compare_gradients(baseline, candidate)
    if baseline["term_files"] or candidate["term_files"]:
        detail["terms"] = _compare_term_maps(baseline, candidate)
    return {"ok": True, **detail}


# ------------------------------------------------------------------------------------- compare


_JOIN_FIELDS = (
    "arm",
    "cell_id",
    "campaign_sha256",
    "binary_hash",
    "profile",
    "node_id",
    "operation",
    "identity",
    "config_digest",
    "parameters_digest",
    "has_mpi",
    "placement_sha256",
)


class _Evidence:
    """Loads referenced artifacts once and enforces cross-sample uniqueness."""

    def __init__(self) -> None:
        self.cache: dict[Path, dict[str, Any]] = {}
        self.measurement_paths: dict[Path, str] = {}
        self.run_ids: dict[str, str] = {}

    def load(self, path_text: object, kind: str, owner: str) -> dict[str, Any]:
        if not isinstance(path_text, str):
            msg = f"{owner}: {kind} path must be a string"
            raise EvidenceError(msg)
        path = Path(path_text).resolve()
        if kind in ("timed", "construction"):
            if path in self.measurement_paths:
                msg = (
                    f"{owner}: {path} is already used by {self.measurement_paths[path]}"
                )
                raise EvidenceError(msg)
            self.measurement_paths[path] = owner
        if path not in self.cache:
            artifact = _require(
                load_json(path, f"{kind} artifact"), ("artifact_kind",), str(path)
            )
            if (
                artifact["artifact_kind"] != kind
                or artifact.get("schema_version") != SCHEMA_VERSION
            ):
                msg = f"{path} is not a schema-{SCHEMA_VERSION} {kind} artifact"
                raise EvidenceError(msg)
            run_id = artifact.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                msg = f"{path} has no run_id"
                raise EvidenceError(msg)
            if run_id in self.run_ids:
                msg = f"{path} reuses run_id {run_id!r} of {self.run_ids[run_id]}"
                raise EvidenceError(msg)
            self.run_ids[run_id] = str(path)
            self.cache[path] = artifact
        return self.cache[path]


def _check_join(
    artifact: dict[str, Any],
    expected: dict[str, Any],
    what: str,
    placement_cache: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Require an artifact to belong to exactly this cell/arm/binary/configuration."""
    for key in _JOIN_FIELDS:
        if key not in artifact:
            msg = f"{what} lacks {key}"
            raise EvidenceError(msg)
        if artifact[key] != expected[key]:
            msg = f"{what}: {key} {artifact[key]!r} does not match {expected[key]!r}"
            raise EvidenceError(msg)
    path_text = artifact.get("placement_path")
    if not isinstance(path_text, str):
        msg = f"{what} lacks placement_path"
        raise EvidenceError(msg)
    key = (path_text, artifact["placement_sha256"])
    if key not in placement_cache:
        path = Path(path_text)
        if not path.is_file() or file_sha256(path) != artifact["placement_sha256"]:
            msg = f"{what}: placement evidence {path} is missing or does not match its hash"
            raise EvidenceError(msg)
        placement_cache[key] = check_placement(
            load_json(path, "placement evidence"),
            arm=expected["arm"],
            cell_id=expected["cell_id"],
            binary_hash=expected["binary_hash"],
            identity=expected["identity"],
        )


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _measurement(artifact: dict[str, Any], field: str, what: str) -> float:
    value = artifact.get(field)
    if not _finite_number(value) or value <= 0:
        msg = f"{what}: {field} is missing, non-finite or not positive"
        raise EvidenceError(msg)
    return float(value)


def compare(args: argparse.Namespace) -> int:
    """Join evidence against the frozen inventory and apply the five parity gates."""
    output = Path(args.output)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "failed_cells": [],
        "missing_cells": [],
        "incomplete_cells": {},
        "cells": {},
        "errors": [],
    }
    try:
        status = _compare(Path(args.campaign), Path(args.manifest), report)
    except EvidenceError as exc:
        report["errors"].append(str(exc))
        status = EXIT_MALFORMED
    report["passed"] = status == EXIT_PASS
    output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return status


def _load_manifest(path: Path, campaign: Campaign) -> dict[str, dict[str, Any]]:
    """Return the manifest's cells by id, rejecting foreign, duplicate or unknown cells."""
    manifest = _require(
        load_json(path, "manifest"),
        ("schema_version", "campaign_sha256", "cells"),
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        msg = "manifest has an unsupported schema_version"
        raise EvidenceError(msg)
    if manifest["campaign_sha256"] != campaign.sha256:
        msg = "manifest was assembled for a different campaign (digest mismatch)"
        raise EvidenceError(msg)
    if not isinstance(manifest["cells"], list):
        msg = "manifest cells must be a list"
        raise EvidenceError(msg)
    offered: dict[str, dict[str, Any]] = {}
    for raw in manifest["cells"]:
        cell = _require(raw, ("id", *ARMS), "manifest cell")
        if cell["id"] in offered:
            msg = f"manifest lists cell {cell['id']!r} twice"
            raise EvidenceError(msg)
        if cell["id"] not in campaign.cells:
            msg = f"manifest cell {cell['id']!r} is not in the frozen inventory"
            raise EvidenceError(msg)
        offered[cell["id"]] = cell
    return offered


def _compare(campaign_path: Path, manifest_path: Path, report: dict[str, Any]) -> int:
    campaign = load_campaign(campaign_path)
    report["campaign_sha256"] = campaign.sha256
    offered = _load_manifest(manifest_path, campaign)
    evidence = _Evidence()
    placements: dict[tuple[str, str], dict[str, Any]] = {}
    for cell_id, cell in campaign.cells.items():
        if cell_id not in offered:
            report["missing_cells"].append(cell_id)
            continue
        result = _compare_cell(campaign, cell, offered[cell_id], evidence, placements)
        report["cells"][cell_id] = result
        if result.get("incomplete"):
            report["incomplete_cells"][cell_id] = result["incomplete"]
        elif not result["passed"]:
            report["failed_cells"].append(cell_id)
    unmet = (
        report["missing_cells"] or report["incomplete_cells"] or report["failed_cells"]
    )
    return EXIT_UNMET if unmet else EXIT_PASS


def _join_arm(
    campaign: Campaign,
    cell: dict[str, Any],
    arm: str,
    samples: object,
    evidence: _Evidence,
    placements: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Load and join one arm's samples of one cell; return its timed/construction/validations."""
    cell_id = cell["id"]
    if not isinstance(samples, list):
        msg = f"manifest {cell_id}/{arm} samples must be a list"
        raise EvidenceError(msg)
    joined: dict[str, Any] = {"timed": [], "construction": [], "validations": {}}
    binary_hash: str | None = None
    sample_ids: set[str] = set()
    for index, raw in enumerate(samples):
        what = f"{cell_id}/{arm} sample {index}"
        keys = ("sample_id", "timed_path", "construction_path", "validation_path")
        sample = _require(raw, keys, what)
        sample_id = require_slug(sample["sample_id"], f"{what} sample id")
        if sample_id in sample_ids:
            msg = f"{cell_id}/{arm} lists sample id {sample_id!r} twice"
            raise EvidenceError(msg)
        sample_ids.add(sample_id)
        timed = evidence.load(sample["timed_path"], "timed", what)
        built = evidence.load(sample["construction_path"], "construction", what)
        validation = evidence.load(sample["validation_path"], "validation", what)
        binary_hash = binary_hash or timed.get("binary_hash")
        expected = {
            "arm": arm,
            "cell_id": cell_id,
            "campaign_sha256": campaign.sha256,
            "binary_hash": binary_hash,
            "profile": cell["profile"],
            "node_id": cell["node_id"],
            "operation": cell["operation"],
            "identity": cell["identity"],
            "config_digest": cell["config_digest"],
            "parameters_digest": cell["parameters_digest"],
            "has_mpi": cell["identity"]["has_mpi"],
            "placement_sha256": timed.get("placement_sha256"),
        }
        for artifact, kind in ((timed, "timed"), (built, "construction")):
            _check_join(artifact, expected, f"{what} {kind}", placements)
            if artifact.get("sample_id") != sample_id:
                msg = f"{what} {kind} artifact belongs to sample {artifact.get('sample_id')!r}"
                raise EvidenceError(msg)
        # Validation is reusable across samples of the same cell/arm/binary/configuration only.
        own = {**expected, "placement_sha256": validation.get("placement_sha256")}
        _check_join(validation, own, f"{what} validation", placements)
        joined["validations"][validation["run_id"]] = validation
        joined["timed"].append(timed)
        joined["construction"].append(built)
    return joined


def _gates(
    cell_id: str, per_arm: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Return the five median-ratio gates and the reasons any of them, or exactness, fails."""
    reasons: list[str] = []
    for arm in ARMS:
        for kind, flag in (
            ("timed", "op_exact"),
            ("construction", "construction_exact"),
        ):
            flags = [artifact.get(flag) for artifact in per_arm[arm][kind]]
            if any(f is not True for f in flags):
                state = "unknown" if any(f is None for f in flags) else "false"
                reasons.append(f"{arm} {flag} is {state} for some observations")
    gates: dict[str, Any] = {}
    for name, kind, field in GATES:
        medians = {
            arm: _median(
                [
                    _measurement(a, field, f"{cell_id}/{arm} {kind}")
                    for a in per_arm[arm][kind]
                ]
            )
            for arm in ARMS
        }
        ratio = medians["candidate"] / medians["baseline"]
        gates[name] = {
            "baseline_median": medians["baseline"],
            "candidate_median": medians["candidate"],
            "ratio": ratio,
            "passed": ratio <= 1.0,
        }
        if ratio > 1.0:
            reasons.append(f"{name} ratio {ratio:.4f} > 1.00")
    return gates, reasons


def _diagnostics(per_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return outer-peak ratios, labelled by the outer windows' own exactness; never gated."""
    flags = [a.get("outer_exact") for arm in ARMS for a in per_arm[arm]["timed"]]
    unknown = "unknown" if any(f is None for f in flags) else "non-exact"
    label = "exact" if all(f is True for f in flags) else unknown
    diagnostics: dict[str, Any] = {}
    for name, field in DIAGNOSTICS:
        medians: dict[str, float | None] = {}
        for arm in ARMS:
            values = [a.get(field) for a in per_arm[arm]["timed"]]
            finite = all(_finite_number(v) for v in values)
            medians[arm] = _median([float(v) for v in values]) if finite else None
        base, cand = medians["baseline"], medians["candidate"]
        diagnostics[name] = {
            "baseline_median": base,
            "candidate_median": cand,
            "ratio": cand / base if base and cand is not None else None,
            "exactness": label,
        }
    return diagnostics


def _numerical(per_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare every baseline validation with every candidate validation of the cell."""
    pairs = []
    try:
        for base in per_arm["baseline"]["validations"].values():
            for cand in per_arm["candidate"]["validations"].values():
                result = compare_validations(base, cand)
                pairs.append(
                    {"baseline": base["run_id"], "candidate": cand["run_id"], **result}
                )
    except NumericalMismatchError as exc:
        return {"ok": False, "reason": str(exc)}
    return {"ok": True, "pairs": pairs}


def _compare_cell(
    campaign: Campaign,
    cell: dict[str, Any],
    offered: dict[str, Any],
    evidence: _Evidence,
    placements: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Join, gate and numerically compare one campaign cell."""
    per_arm = {
        arm: _join_arm(campaign, cell, arm, offered[arm], evidence, placements)
        for arm in ARMS
    }
    counts = {arm: len(per_arm[arm]["timed"]) for arm in ARMS}
    result: dict[str, Any] = {"samples": counts, "passed": False}
    allowed = (INITIAL_SAMPLES, REPEATED_SAMPLES)
    if counts["baseline"] != counts["candidate"] or counts["baseline"] not in allowed:
        result["incomplete"] = (
            f"needs {INITIAL_SAMPLES} (or {REPEATED_SAMPLES} after repetition) distinct paired "
            f"observations per arm, got {counts}"
        )
        return result
    gates, reasons = _gates(cell["id"], per_arm)
    numerical = _numerical(per_arm)
    if not numerical["ok"]:
        reasons.append(f"numerical mismatch: {numerical['reason']}")
    ratio_failed = any(not gate["passed"] for gate in gates.values())
    first_round = counts["baseline"] == INITIAL_SAMPLES
    result |= {
        "gates": gates,
        "diagnostics": _diagnostics(per_arm),
        "numerical": numerical,
        "needs_additional_samples": (
            REPEATED_SAMPLES - INITIAL_SAMPLES if ratio_failed and first_round else 0
        ),
        "reasons": reasons,
        "passed": not reasons,
    }
    return result


# ------------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    """Return the driver's three-mode argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("observe", "validate"):
        mode = modes.add_parser(name)
        mode.add_argument("--arm", required=True, choices=ARMS)
        mode.add_argument(
            "--runtime-shape", required=True, choices=tuple(ARM_SHAPES.values())
        )
        mode.add_argument("--campaign", required=True)
        mode.add_argument("--cell-id", required=True)
        if name == "observe":
            mode.add_argument("--sample-id", required=True)
        mode.add_argument("--placement", required=True)
        mode.add_argument("--output", required=True)
        if name == "observe":
            mode.add_argument("--whole-process", action="store_true")
    mode = modes.add_parser("compare")
    mode.add_argument("--campaign", required=True)
    mode.add_argument("--manifest", required=True)
    mode.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one driver mode; malformed evidence exits 2."""
    args = build_parser().parse_args(argv)
    if args.mode == "compare":
        return compare(args)
    try:
        return observe(args) if args.mode == "observe" else validate(args)
    except EvidenceError as exc:
        print(f"benchmark-rank-local-openmp: {exc}", file=sys.stderr)  # noqa: T201
        return EXIT_MALFORMED


if __name__ == "__main__":
    raise SystemExit(main())
