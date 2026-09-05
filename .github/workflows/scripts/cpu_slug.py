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

"""Print this run's CPU as a Bencher testbed slug, e.g. `intel-xeon-platinum-8488c`.

Naming a series after the silicon rather than a provider's instance label means the name
changes exactly when the hardware does, which is the property Bencher needs. The brand comes
from the cpuinfo block pytest-benchmark already writes into `time-<label>.json`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NOISE = re.compile(r"\((?:r|tm)\)|\bprocessor\b", re.IGNORECASE)
SEPARATORS = re.compile(r"[^a-z0-9]+")


def slugify(brand: str) -> str:
    return SEPARATORS.sub("-", NOISE.sub(" ", brand).lower()).strip("-") or "unknown-cpu"


def main() -> str | None:
    paths = sorted(Path("benches/results").glob("time-*.json"))
    if not paths:
        return "::error::no timing artifact to read the CPU from"
    brand = json.loads(paths[0].read_text())["machine_info"]["cpu"]["brand_raw"]
    print(slugify(str(brand)))
    return None


if __name__ == "__main__":
    sys.exit(main())
