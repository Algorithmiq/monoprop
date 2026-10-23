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

"""Unit tests for the shared runtime-shape and build-mode preflight."""

from __future__ import annotations

import sys
import types

import pytest
from monoprop_bench_tools import preflight
from monoprop_bench_tools.preflight import PreflightError

_BASELINE = {"monoprop_PARTITIONS": "4", "monoprop_NUM_THREADS": "4"}


def _declare(shape: str, env: dict[str, str], **kwargs: object) -> dict:
    kwargs.setdefault("ranks", 1)
    kwargs.setdefault("has_mpi", False)
    return preflight.declared_shape(shape, env, **kwargs)  # type: ignore[arg-type]


def test_baseline_declares_both_legacy_thread_variables() -> None:
    shape = _declare("partitions", {**_BASELINE, "OMP_NUM_THREADS": "4"})

    assert shape["runtime_shape"] == "partitions"
    assert (shape["ranks"], shape["threads"]) == (1, 4)
    assert shape["settings"] == {**_BASELINE, "OMP_NUM_THREADS": "4"}


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"monoprop_NUM_THREADS": "4"},
        {"monoprop_PARTITIONS": "4"},
        {"monoprop_PARTITIONS": "4", "monoprop_NUM_THREADS": "2"},
        {"monoprop_PARTITIONS": "auto", "monoprop_NUM_THREADS": "4"},
    ],
)
def test_baseline_declaration_must_be_complete_and_consistent(env: dict) -> None:
    with pytest.raises(PreflightError, match="partitions"):
        _declare("partitions", env)


def test_candidate_explicit_thread_budget() -> None:
    shape = _declare("openmp", {"monoprop_NUM_THREADS": "8", "OMP_NUM_THREADS": "8"})

    assert (shape["threads"], shape["threads_source"]) == (8, "monoprop_NUM_THREADS")


def test_candidate_openmp_fallback_resolves_the_declared_budget() -> None:
    shape = _declare("openmp", {"OMP_NUM_THREADS": "6"})

    assert (shape["threads"], shape["threads_source"]) == (6, "OMP_NUM_THREADS")


@pytest.mark.parametrize("value", ["", "0", "-1", "abc", "2,3", " 3", "+3", "3 "])
def test_candidate_invalid_thread_override_is_rejected(value: str) -> None:
    with pytest.raises(PreflightError, match="monoprop_NUM_THREADS"):
        _declare("openmp", {"monoprop_NUM_THREADS": value, "OMP_NUM_THREADS": "4"})


def test_candidate_without_any_resolvable_budget_is_rejected() -> None:
    # A list-valued OMP_NUM_THREADS cannot declare one team size for the preflight.
    with pytest.raises(PreflightError, match="cannot establish"):
        _declare("openmp", {"OMP_NUM_THREADS": "4,2"})


def test_leftover_partition_setting_in_candidate_environment_is_rejected() -> None:
    env = {"monoprop_NUM_THREADS": "4", "monoprop_PARTITIONS": "4"}
    with pytest.raises(PreflightError, match="monoprop_PARTITIONS"):
        _declare("openmp", env)


def test_unknown_runtime_shape_is_rejected() -> None:
    with pytest.raises(PreflightError, match="runtime shape"):
        _declare("hybrid", _BASELINE)


def test_wrong_observed_ranks_or_threads_is_rejected() -> None:
    with pytest.raises(PreflightError, match="ranks"):
        _declare("partitions", _BASELINE, ranks=2, has_mpi=True, expected_ranks=1)
    with pytest.raises(PreflightError, match="threads"):
        _declare("partitions", _BASELINE, expected_threads=8)


def test_nonpositive_ranks_are_rejected() -> None:
    with pytest.raises(PreflightError, match="ranks"):
        _declare("partitions", _BASELINE, ranks=0)


def test_mpi_off_declaration_with_several_ranks_is_rejected() -> None:
    with pytest.raises(PreflightError, match="MPI-off"):
        _declare("partitions", _BASELINE, ranks=2, has_mpi=False)


def test_requested_build_mode_must_match_the_imported_binary() -> None:
    with pytest.raises(PreflightError, match="has_mpi"):
        _declare("partitions", _BASELINE, has_mpi=True, expected_has_mpi=False)
    with pytest.raises(PreflightError, match="has_mpi"):
        _declare("partitions", _BASELINE, has_mpi=False, expected_has_mpi=True)


def test_missing_build_mode_evidence_is_rejected() -> None:
    with pytest.raises(PreflightError, match="has_mpi"):
        _declare("partitions", _BASELINE, has_mpi=None)


@pytest.mark.parametrize(
    ("arm", "shape"), [("baseline", "openmp"), ("candidate", "partitions")]
)
def test_a_relabelled_arm_cannot_bypass_the_shape(arm: str, shape: str) -> None:
    with pytest.raises(PreflightError, match="arm"):
        preflight.require_arm_shape(arm, shape)


def test_matching_arm_and_shape_pass() -> None:
    preflight.require_arm_shape("baseline", "partitions")
    preflight.require_arm_shape("candidate", "openmp")


def test_mpi_off_never_imports_mpi4py(monkeypatch: pytest.MonkeyPatch) -> None:
    # A trap standing in for an installed mpi4py: importing it at all would initialize MPI.
    trap = types.ModuleType("mpi4py")

    def _explode(name: str) -> None:
        msg = f"attempted to import mpi4py.{name} in an MPI-off run"
        raise AssertionError(msg)

    trap.__getattr__ = _explode  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "mpi4py", trap)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", None)

    assert preflight.import_mpi(has_mpi=False) is None


def test_mpi_build_without_mpi4py_stays_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mpi4py", None)

    assert preflight.import_mpi(has_mpi=True) is None


def test_mpi_build_imports_mpi4py(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mpi = types.ModuleType("mpi4py.MPI")
    package = types.ModuleType("mpi4py")
    package.MPI = fake_mpi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mpi4py", package)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake_mpi)

    assert preflight.import_mpi(has_mpi=True) is fake_mpi


def test_settings_snapshot_keeps_only_runtime_configuration() -> None:
    env = {
        "monoprop_NUM_THREADS": "4",
        "OMP_PLACES": "cores",
        "HOME": "/root",
        "monoprop_BENCH_LABEL": "x",
        "monoprop_BENCH_RESULTS": "results/r",
        "monoprop_BENCH_ALLOW_BIG_GRAPH": "1",
    }

    assert preflight.settings_snapshot(env) == {
        "OMP_PLACES": "cores",
        "monoprop_BENCH_ALLOW_BIG_GRAPH": "1",
        "monoprop_NUM_THREADS": "4",
    }
