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

"""Which ISA tiers this install carries, and which of them this CPU can run.

Both answers come from the loaded engine rather than from the filesystem or from Python's own view of
the CPU, because the engine is where the decision is made: a tiered build carries the propagation
kernel once per x86-64 ISA tier in ``monoprop/lib/libmonoprop-tier-<id>.so`` and picks one on first
use, inside the library (``cpp/monoprop/detail/evolution/TierDispatch.cpp``). So nothing here selects
anything -- it forwards, and reports ``()`` for a single-ISA build, where there was nothing to select.
"""

from __future__ import annotations

from . import _core

#: Pin the tier instead of probing the CPU. Read by the engine, not by this module. Refused if the
#: named tier is absent or unrunnable, because the point of pinning one is to know which one ran --
#: benchmarking a tier, reproducing a report, bisecting a codegen difference.
VARIANT_ENV_VAR = "monoprop_VARIANT"


def available_variants() -> tuple[str, ...]:
    """ISA variants this build ships, best first; empty for a single-ISA build."""
    return tuple(_core.shipped_tiers())


def supported_variants() -> tuple[str, ...]:
    """ISA variants the running CPU can execute, best first; empty for a single-ISA build.

    A subset of :func:`available_variants`, and not merely a capability list: the predicate that
    decides a tier is what the engine would dispatch on, so a CPU that *has* 512-bit vectors but is
    better off without them does not report the 512-bit tier as supported.
    """
    return tuple(_core.supported_tiers())
