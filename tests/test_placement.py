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

"""Partition placement is reportable from Python, and refusing to pin is not silent.

Asking for more partitions than there are visible physical cores leaves every partition thread
unpinned, which measured 16-25x on propagate. The engine says so on C++ stderr, which pytest's
file-descriptor capture swallows without ``-s``; these cover the two channels that survive it.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager

import pytest

from monoprop import MajoranaPropagator, placement_report
from monoprop.fermi import MajoranaOperator

_REPORT_FIELDS = {"pinned", "cores_visible", "groups", "partitions", "decisions"}


@contextmanager
def _oversubscribed():
    """Yield a partition count guaranteed to exceed this process's visible physical cores.

    Confines the process to a single CPU where the platform allows, so the oversubscribed build
    costs two threads instead of one per CPU. Where it does not, one more partition than there are
    CPUs still exceeds the physical cores, since a core never has fewer than one. Neither arm
    skips -- both reach the same refusal.
    """
    getter = getattr(os, "sched_getaffinity", None)
    setter = getattr(os, "sched_setaffinity", None)
    if getter is not None and setter is not None:
        saved = getter(0)
        try:
            setter(0, {min(saved)})
        except OSError:
            pass
        else:
            try:
                yield 2
            finally:
                setter(0, saved)
            return
    yield (len(getter(0)) if getter is not None else (os.cpu_count() or 1)) + 1


def _build(serial_comm):
    return MajoranaPropagator(
        MajoranaOperator({(0, 1, 2, 3): 1.0}, 2), [], cutoff=4, comm=serial_comm
    )


def test_placement_report_names_every_field():
    report = placement_report()
    assert set(report) == _REPORT_FIELDS
    assert isinstance(report["pinned"], bool)
    for field in _REPORT_FIELDS - {"pinned"}:
        assert isinstance(report[field], int)


def test_oversubscription_warns_and_reports_unpinned(monkeypatch, serial_comm):
    before = placement_report()["decisions"]
    with _oversubscribed() as partitions:
        monkeypatch.setenv("monoprop_PARTITIONS", str(partitions))
        with pytest.warns(RuntimeWarning, match="threads run unpinned"):
            _build(serial_comm)
        report = placement_report()

    # The refusal itself, not merely that some RuntimeWarning was raised.
    assert report["pinned"] is False
    assert report["partitions"] == partitions
    assert report["cores_visible"] < partitions
    assert report["decisions"] == before + 1


def test_a_build_that_places_nothing_does_not_re_announce(monkeypatch, serial_comm):
    """One placement, one warning: a single-partition build decides nothing and must stay quiet."""
    with _oversubscribed() as partitions:
        monkeypatch.setenv("monoprop_PARTITIONS", str(partitions))
        with pytest.warns(RuntimeWarning, match="threads run unpinned"):
            _build(serial_comm)
        decisions = placement_report()["decisions"]

    monkeypatch.setenv("monoprop_PARTITIONS", "1")
    with warnings.catch_warnings():
        # A re-announcement of the stale report raises here.
        warnings.simplefilter("error")
        _build(serial_comm)
    assert placement_report()["decisions"] == decisions
