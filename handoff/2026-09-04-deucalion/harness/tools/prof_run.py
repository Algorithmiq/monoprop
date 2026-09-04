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

"""Run one fixed model and emit the three lines every collator here reads.

    python prof_run.py <model> <op> [field=value ...]

`op` is `propagate` or `graph` (steps x build_graph). `CALIB_BENCHES` selects which checkout's
`benches/_builders.py` supplies MODELS -- the same convention terms_calib.py uses, and it must point
at the tree under test, because `_builders.py` is versioned alongside the C++ and another tree's
builders would silently profile a different problem.

Deliberately NOT under pytest. pytest's capture is fd-level, and the engine writes LAYERPROF /
COMMPROF straight to fd 2 from a static destructor, so under pytest a live instrument looks exactly
like one that never fired (`-s` is the workaround there; running directly means there is no `-s` to
forget). Only rank 0 prints the banners; every partition of every rank prints its own LAYERPROF
line, so a collator has to sum over partitions.

Three output lines, each with its own consumer:

    PROFRUN   one per run: terms, elapsed, rssmax_mb, rsssum_mb        -> membisect_summary.py
    PROFMEM   the structural ledger, summed over ranks                 -> membisect_summary.py
    LAYERPROF per partition, only under monoprop_LAYER_PROFILE=1       -> layerprof_summary.py
"""

import os
import resource
import sys
import time

sys.path.insert(0, os.environ["CALIB_BENCHES"])

from mpi4py import MPI  # noqa: E402

try:
    # Post-refactor layout: monoprop_bench_tools is a proper (pip-installed) distribution and
    # `_builders` is now its `models` submodule.
    from monoprop_bench_tools.models import MODELS  # noqa: E402
except ImportError:
    try:
        # Pre-refactor layout: a bare `_builders.py` reached via $CALIB_BENCHES above.
        from _builders import MODELS  # noqa: E402
    except ImportError as exc:
        raise SystemExit(
            "prof_run: could not import MODELS.\n"
            "  tried: monoprop_bench_tools.models (post-refactor package; not installed in "
            "this interpreter's site-packages)\n"
            "  tried: _builders on sys.path via $CALIB_BENCHES={calib!r} (pre-refactor "
            "layout)\n"
            "  Neither import succeeded ({exc}). Install monoprop_bench_tools in this venv, "
            "or point CALIB_BENCHES at a checkout containing benches/_builders.py.".format(
                calib=os.environ.get("CALIB_BENCHES"), exc=exc
            )
        ) from exc

model = sys.argv[1]
op = sys.argv[2]
config_cls, build_fn, steps_fn = MODELS[model]

overrides = {}
for arg in sys.argv[3:]:
    key, _, value = arg.partition("=")
    cur = getattr(config_cls(), key)
    overrides[key] = type(cur)(value)

comm = MPI.COMM_WORLD
config = config_cls(**overrides)
propagator, circuit = build_fn(config, comm=comm)
steps = steps_fn(config)

# Barriered on both sides so the reported time is the slowest rank's, which is what a user waits for,
# and so setup cannot leak into it.
comm.Barrier()
t0 = time.perf_counter()
if op == "propagate":
    for _ in range(steps):
        propagator.propagate(circuit)
elif op == "graph":
    for _ in range(steps):
        propagator.build_graph(circuit)
else:
    raise SystemExit(f"unknown op {op!r}")
comm.Barrier()
elapsed = time.perf_counter() - t0

# Term count is summed over ranks: two arms that propagated different totals did different work, and
# the collator refuses the cell rather than reporting a ratio between them.
terms = comm.allreduce(propagator.size())

# Peak resident set, reported two ways because they answer different questions and have been confused
# before: `rssmax` is the largest single rank's high-water mark (what a per-rank memory limit sees),
# `rsssum` is the sum over ranks (what the NODE has to hold). ru_maxrss is KiB on Linux. This is a
# high-water mark for the whole process, so it includes interpreter and setup, not just the operator --
# usable for A/B between arms of the same shape, not as an absolute operator size.
rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_max = comm.allreduce(rss_kib, op=MPI.MAX)
rss_sum = comm.allreduce(rss_kib, op=MPI.SUM)
if comm.Get_rank() == 0:
    print(
        f"PROFRUN model={model} op={op} steps={steps} ranks={comm.Get_size()} "
        f"terms={terms} elapsed={elapsed:.4f} rssmax_mb={rss_max / 1024:.1f} "
        f"rsssum_mb={rss_sum / 1024:.1f} overrides={overrides}",
        flush=True,
    )

# The structural ledger, summed over ranks the way the reports sum it. RSS above answers "what did
# the node have to hold"; this answers "which structure holds it", which is what attributing a memory
# change to a commit needs. Deliberately printed on a separate line so every existing `^PROFRUN ` grep
# keeps working.
#
# TWO KEY SETS. Arms that predate a layout change emit different d_* keys and may carry no
# matched_scratch_bytes and no arena-slack field. Reduce over whatever THIS binary reports -- every rank
# runs the same binary, so the key set is uniform within a run and varies only across arms. The
# collator reconciles them.
#
# The slack fields are `capacity() - size()`: virtual, never faulted in. A collator that sums them into
# a resident total overstates the win -- read the touched column, not total_bytes.
# The binding is on the C++ MonomialPropagator, not on the Python wrapper, so it is reached through
# `._simulator` -- the same way benches/conftest.py and terms_calib.py reach it.
breakdown = getattr(propagator._simulator, "operator_memory_breakdown", None)
if breakdown is None:
    if comm.Get_rank() == 0:
        print("PROFMEM MISSING binding predates operator_memory_breakdown()", flush=True)
else:
    totals = {k: comm.allreduce(int(v), op=MPI.SUM) for k, v in sorted(breakdown().items())}
    if comm.Get_rank() == 0:
        fields = " ".join("{}={}".format(k, v) for k, v in sorted(totals.items()))
        print(f"PROFMEM terms={terms} ranks={comm.Get_size()} {fields}", flush=True)
