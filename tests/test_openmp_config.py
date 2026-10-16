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
interpreter with the environment under test. Every propagator captures it (its thread count is the shard count),
strictly, falling back to the OpenMP default only when the variable is unset; the removed partition controls are
unsupported, not rejected configurations. ``tests/test_sharded_openmp.py`` covers the runtime in depth.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

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


def test_no_thread_keyword_is_accepted() -> None:
    outcome = _probe("1", num_threads=2)
    assert not outcome["ok"]
    assert outcome["type"] == "TypeError"


@pytest.mark.parametrize(
    "budget", ["", "0", "-1", "abc", "2,3", " 3", "3 ", "+3", "2147483648"]
)
def test_sharded_root_rejects_an_invalid_budget(budget: str) -> None:
    outcome = _probe(budget)
    assert not outcome["ok"]
    assert outcome["type"] == "RuntimeError"
    assert "monoprop_NUM_THREADS" in outcome["message"]


def test_sharded_root_accepts_valid_budgets() -> None:
    # T changes the shard geometry, and with it the shard fold, so values agree to tolerance, not bitwise.
    energies = []
    for budget in ["1", "3", "0003", None]:
        outcome = _probe(budget, {"OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"})
        assert outcome["ok"], (budget, outcome)
        energies.append(outcome["energy"])
    assert all(e == pytest.approx(energies[0], rel=1e-12, abs=1e-12) for e in energies)


@pytest.mark.parametrize("partitions", [0, 1, 2])
def test_sharded_root_has_no_partitions_argument(partitions: int) -> None:
    # The removed argument is an unsupported keyword, whatever its value -- not a rejected configuration.
    outcome = _probe("1", partitions=partitions)
    assert not outcome["ok"], outcome
    assert outcome["type"] == "TypeError"


@pytest.mark.parametrize("value", ["off", "auto", "1", "2", "0", "", "not-a-count"])
def test_sharded_root_ignores_the_obsolete_partition_variable(value: str) -> None:
    # Never read: the same budget gives the same answer, bit for bit, whatever the variable holds.
    exact_team = {"OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"}
    reference = _probe("2", exact_team)
    outcome = _probe("2", {**exact_team, "monoprop_PARTITIONS": value})
    assert reference["ok"], reference
    assert outcome == reference
