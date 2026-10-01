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

"""Construction-time thread budgets, observed from fresh Python processes.

A propagator reads ``monoprop_NUM_THREADS`` once, when it is constructed, so every case launches its own
interpreter with the environment under test. In a legacy build only the one-store prototype (the low-level
``partitions=1`` argument on an ordinary communicator) captures a budget; the default and multi-partition paths
keep their legacy configuration. In a ``monoprop_SHARDED_OPENMP_PROTOTYPE`` build every propagator captures it (its
thread count is the shard count), and the legacy controls are rejected; ``tests/test_sharded_openmp.py`` covers that
build in depth.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from monoprop import _core

# Environment a launcher gives its ranks: a child that inherits it would try to join the parent's job.
_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PRTE_", "PMI_", "HYDRA_", "MPIR_")

_PROBE = r"""
import json
import sys

from monoprop import _core

args = json.loads(sys.argv[1])
try:
    sim = _core.MonomialPropagator032(
        initial_operator={(0, 1): 1j, (2, 3): 0.5j, (1, 2): 0.25j},
        cutoff=4,
        initial_state=[0, 1],
        **args,
    )
    sim.build_graph([[0, 2], [1, 3]], [0, 1], [1.0, 1.0])
    print(json.dumps({"ok": True, "energy": sim.expectation_value([0.3, 0.2])}))
except Exception as exc:  # noqa: BLE001 - the parent inspects whatever the constructor raised
    print(json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}))
"""


def _inside_multirank_job() -> bool:
    size = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or "1"
    return int(size) > 1


pytestmark = pytest.mark.skipif(
    _inside_multirank_job(),
    reason="launches fresh interpreters, which cannot run inside a multi-rank job",
)

SHARDED = (
    getattr(_core, "__runtime_identity__", None)
    == "monoprop-runtime=sharded-openmp-prototype"
)
legacy_only = pytest.mark.skipif(SHARDED, reason="legacy one-store/partition controls")
sharded_only = pytest.mark.skipif(not SHARDED, reason="needs a sharded prototype build")


def _probe(
    budget: str | None, extra_env: dict[str, str] | None = None, **args: object
) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
    env.pop("monoprop_NUM_THREADS", None)
    env.pop("monoprop_PARTITIONS", None)
    if budget is not None:
        env["monoprop_NUM_THREADS"] = budget
    env.update(extra_env or {})
    result = subprocess.run(  # noqa: S603 - trusted: this interpreter and a fixed probe
        [sys.executable, "-c", _PROBE, json.dumps(args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@legacy_only
@pytest.mark.parametrize(
    "budget", ["", "0", "-1", "abc", "2,3", " 3", "3 ", "+3", "2147483648"]
)
def test_prototype_rejects_an_invalid_budget(budget: str) -> None:
    outcome = _probe(budget, partitions=1)
    assert not outcome["ok"]
    assert outcome["type"] == "RuntimeError"
    assert "monoprop_NUM_THREADS" in outcome["message"]


@legacy_only
def test_prototype_accepts_valid_budgets_without_changing_results() -> None:
    energies = []
    for budget in ["1", "3", "0003", None]:
        outcome = _probe(budget, {"OMP_NUM_THREADS": "2"}, partitions=1)
        assert outcome["ok"], (budget, outcome)
        energies.append(outcome["energy"])
    # Threaded kernels give bitwise-identical results at every budget (and this problem is too small
    # to leave their serial paths anyway).
    assert all(e == energies[0] for e in energies)


@legacy_only
@pytest.mark.parametrize(
    ("args", "extra_env"),
    [
        pytest.param({"partitions": 2}, {}, id="two-partition-facade"),
        pytest.param(
            {"partitions": 0},
            {"monoprop_PARTITIONS": "off"},
            id="default-single-partition",
        ),
    ],
)
def test_legacy_paths_do_not_capture_the_budget(
    args: dict, extra_env: dict[str, str]
) -> None:
    # The same value the prototype rejects is never read on the legacy paths.
    outcome = _probe("abc", extra_env, **args)
    assert outcome["ok"], outcome


def test_no_thread_keyword_is_accepted() -> None:
    outcome = (
        _probe("1", num_threads=2)
        if SHARDED
        else _probe("1", partitions=1, num_threads=2)
    )
    assert not outcome["ok"]
    assert outcome["type"] == "TypeError"


@sharded_only
@pytest.mark.parametrize(
    "budget", ["", "0", "-1", "abc", "2,3", " 3", "3 ", "+3", "2147483648"]
)
def test_sharded_root_rejects_an_invalid_budget(budget: str) -> None:
    outcome = _probe(budget)
    assert not outcome["ok"]
    assert outcome["type"] == "RuntimeError"
    assert "monoprop_NUM_THREADS" in outcome["message"]


@sharded_only
def test_sharded_root_accepts_valid_budgets() -> None:
    # T changes the shard geometry, and with it the shard fold, so values agree to tolerance, not bitwise.
    energies = []
    for budget in ["1", "3", "0003", None]:
        outcome = _probe(budget, {"OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"})
        assert outcome["ok"], (budget, outcome)
        energies.append(outcome["energy"])
    assert all(e == pytest.approx(energies[0], rel=1e-12, abs=1e-12) for e in energies)


@sharded_only
@pytest.mark.parametrize(
    ("args", "extra_env"),
    [
        pytest.param({"partitions": 2}, {}, id="explicit-partitions"),
        pytest.param({"partitions": 1}, {}, id="one-store-prototype"),
        pytest.param({}, {"monoprop_PARTITIONS": "off"}, id="obsolete-selector"),
    ],
)
def test_sharded_root_rejects_legacy_controls(
    args: dict, extra_env: dict[str, str]
) -> None:
    outcome = _probe("1", extra_env, **args)
    assert not outcome["ok"], outcome
    assert outcome["type"] == "RuntimeError"
