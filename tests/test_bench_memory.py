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

"""Unit tests for the benchmark memory primitives (``benches/_memory.py``).

The per-test peak under MPI is the *peak-of-sum* (the largest footprint that actually coexisted
across ranks), not the sum of per-rank lifetime peaks, which overcounts disjoint transients.
"""

from __future__ import annotations

import pytest
from _memory import RssSampler, merge_peak_of_sum, rss_bytes

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


def test_sampler_records_timeline_and_sees_a_transient() -> None:
    if rss_bytes() == 0:
        pytest.skip("/proc/self/status VmRSS unavailable (non-Linux)")
    with RssSampler(interval=0.002) as sampler:
        baseline = sampler.samples[0][1]
        blob = bytearray(80 * MIB)
        for i in range(0, len(blob), 4096):  # touch pages so they become resident
            blob[i] = 1
        # Hold across several sampling intervals so the transient is observed.
        for _ in range(2_000_000):
            pass
        del blob

    samples = sampler.samples
    assert len(samples) >= 2  # baseline on enter, final on exit
    assert all(isinstance(t, float) and isinstance(p, int) for t, p in samples)
    assert max(p for _t, p in samples) > baseline
