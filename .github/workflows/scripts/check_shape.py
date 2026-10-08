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

"""Fail a rung that did not run at the shape it was asked for.

A knob that fails to reach the ranks does not fail the run: it measures a different team at a
plausible wall time. So the recorded rank count and per-rank thread count must be the rung's,
and the removed partition runtime's variable must not be set at all. Placement is the launcher's
and OpenMP's (`--bind-to core`, `OMP_PROC_BIND`/`OMP_PLACES`); its summary is printed for the log.

Usage: check_shape.py <results/<label>.json> <ranks> <threads>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> str | None:
    path, ranks, threads = Path(argv[1]), int(argv[2]), argv[3]
    meta = json.loads(path.read_text())["meta"]
    pin = meta.get("pinning", {})

    print(
        f"  ranks={meta['ranks']} threads={meta['monoprop_threads']} "
        f"partitions_env={meta['partitions_env']} "
        f"single-cpu threads>={pin.get('single_cpu_threads_min', 0)} of {pin.get('threads', 0)} "
        f"mask={pin.get('affinity_cpus_min')}..{pin.get('affinity_cpus_max')}"
    )

    if meta["ranks"] != ranks or meta["monoprop_threads"] != threads:
        return (
            f"::error::{path.stem} ran at ranks={meta['ranks']} "
            f"threads={meta['monoprop_threads']}, asked for {ranks}x{threads}"
        )
    if meta["partitions_env"] != "unset":
        return (
            f"::error::{path.stem} ran with monoprop_PARTITIONS={meta['partitions_env']}; "
            "the sharded runtime ignores it, so a rung must not set it"
        )
    return None


if __name__ == "__main__":
    sys.exit(main(sys.argv))
