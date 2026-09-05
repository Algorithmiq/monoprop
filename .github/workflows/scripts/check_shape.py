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

A knob that fails to reach the ranks does not fail the run: it measures one partition per
rank at a plausible wall time. Neither does `partitions` above a rank's visible cores --
that disables placement entirely, at 13.8x to 24.6x, warning only on C++ stderr, which
pytest's capture eats.

Usage: check_shape.py <results/<label>.json> <ranks> <partitions>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> str | None:
    path, ranks, partitions = Path(argv[1]), int(argv[2]), argv[3]
    meta = json.loads(path.read_text())["meta"]
    pin = meta.get("pinning", {})
    threads = pin.get("threads", 0)
    placed = pin.get("single_cpu_threads_min", 0)

    print(
        f"  ranks={meta['ranks']} partitions_env={meta['partitions_env']} "
        f"threads={meta['monoprop_threads']} pinned>={placed} of {threads} "
        f"mask={pin.get('affinity_cpus_min')}..{pin.get('affinity_cpus_max')}"
    )

    if meta["ranks"] != ranks or meta["partitions_env"] != partitions:
        return (
            f"::error::{path.stem} ran at ranks={meta['ranks']} "
            f"partitions_env={meta['partitions_env']}, asked for {ranks}x{partitions}"
        )
    # An all-zero summary means /proc was unreadable, not that nothing was pinned.
    if int(partitions) > 1 and threads and not placed:
        return (
            f"::error::{path.stem} placed no thread on a CPU of its own. "
            "partitions above the rank's visible cores returns an empty placement order."
        )
    return None


if __name__ == "__main__":
    sys.exit(main(sys.argv))
