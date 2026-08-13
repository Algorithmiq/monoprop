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

"""Unit tests for the benchmark memory primitives (``benches/_memory_cpu.py``).

Two behaviours are pinned. The per-test peak under MPI is the *peak-of-sum* (the largest
footprint that actually coexisted across ranks), not the sum of per-rank lifetime peaks,
which overcounts disjoint transients. And :class:`HighWaterMark` sees transients that the
sampler cannot, which is the whole reason it exists.
"""

from __future__ import annotations

import gc
import resource

import pytest
from _memory_cpu import (
    HighWaterMark,
    PssSampler,
    heap_trim,
    merge_peak_of_sum,
    peak_rss_bytes,
    pss_bytes,
    reset_peak_rss,
    rss_bytes,
)

MIB = 2**20


def test_merge_serial_returns_own_peak() -> None:
    series = [(0.0, 50), (1.0, 150), (2.0, 50)]
    assert merge_peak_of_sum([series]) == 150


def test_merge_overlapping_peaks_sum() -> None:
    # Both ranks peak at the same instant (t=1): the peak-of-sum is their sum.
    rank0 = [(0.0, 10), (1.0, 210), (2.0, 10)]
    rank1 = [(0.0, 10), (1.0, 210), (2.0, 10)]
    assert merge_peak_of_sum([rank0, rank1]) == 420


def test_merge_staggered_peaks_do_not_double_count() -> None:
    # The transients never coexist, so the peak-of-sum is one peak + the other's held baseline.
    rank0 = [(0.0, 10), (1.0, 210), (2.0, 10)]
    rank1 = [(0.0, 10), (3.0, 10), (4.0, 210), (5.0, 10)]
    assert merge_peak_of_sum([rank0, rank1]) == 220
    assert merge_peak_of_sum([rank0, rank1]) != 420


def test_merge_step_holds_last_reading() -> None:
    # rank1 has no sample near t=1; its last reading (100) is held while rank0 peaks.
    rank0 = [(0.0, 0), (1.0, 500), (2.0, 0)]
    rank1 = [(0.0, 100)]
    assert merge_peak_of_sum([rank0, rank1]) == 600


def test_merge_handles_empty_series() -> None:
    assert merge_peak_of_sum([[]]) == 0
    assert merge_peak_of_sum([[(0.0, 100)], []]) == 100


def test_sampler_records_an_ordered_timeline() -> None:
    if pss_bytes() == 0:
        pytest.skip("/proc/self/smaps_rollup Pss unavailable (non-Linux)")
    with PssSampler(interval=0.002) as sampler:
        blob = bytearray(80 * MIB)
        for i in range(0, len(blob), 4096):  # touch pages so they become resident
            blob[i] = 1
        del blob

    samples = sampler.samples
    assert len(samples) >= 2  # baseline on enter, final on exit
    assert all(isinstance(t, float) and isinstance(p, int) for t, p in samples)
    assert all(p > 0 for _t, p in samples)
    assert [t for t, _p in samples] == sorted(t for t, _p in samples)


def test_pss_is_at_most_rss() -> None:
    if pss_bytes() == 0:
        pytest.skip("/proc/self/smaps_rollup Pss unavailable (non-Linux)")
    # PSS splits each shared page across its mappers, so it can only be <= RSS. This is the
    # property that makes summing PSS across ranks on a node meaningful.
    assert 0 < pss_bytes() <= rss_bytes()


def test_high_water_mark_catches_a_freed_transient() -> None:
    if not reset_peak_rss():
        pytest.skip("/proc/self/clear_refs unavailable (non-Linux or kernel < 4.0)")
    with HighWaterMark() as window:
        blob = bytearray(80 * MIB)
        for i in range(0, len(blob), 4096):
            blob[i] = 1
        del blob  # gone before the window closes: only VmHWM still knows it existed

    assert window.exact
    assert window.delta_bytes >= 70 * MIB
    assert window.peak_bytes >= window.baseline_bytes


def test_high_water_mark_window_is_reset_per_block() -> None:
    if not reset_peak_rss():
        pytest.skip("/proc/self/clear_refs unavailable (non-Linux or kernel < 4.0)")
    with HighWaterMark() as first:
        blob = bytearray(80 * MIB)
        for i in range(0, len(blob), 4096):
            blob[i] = 1
        del blob
    with HighWaterMark() as second:
        pass

    # The second window must not inherit the first's spike
    assert first.delta_bytes >= 70 * MIB
    assert second.delta_bytes < 10 * MIB


def test_peak_rss_never_below_current_rss() -> None:
    if peak_rss_bytes() == 0:
        pytest.skip("/proc/self/status VmHWM unavailable (non-Linux)")
    assert peak_rss_bytes() >= rss_bytes()


def test_reset_also_clears_ru_maxrss() -> None:
    """``ru_maxrss`` shares ``mm->hiwater_rss`` with ``VmHWM``, so a window reset drops it.

    Benchmarks that record a whole-run ceiling alongside per-step windows have to take the
    maximum over the windows instead; this pins the behaviour that forces that.
    """
    if not reset_peak_rss():
        pytest.skip("/proc/self/clear_refs unavailable (non-Linux or kernel < 4.0)")

    blob = bytearray(80 * MIB)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    del blob
    gc.collect()
    heap_trim()
    reset_peak_rss()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    assert before >= 70 * MIB
    assert after < before
