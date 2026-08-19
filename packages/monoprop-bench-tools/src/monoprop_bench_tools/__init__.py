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

"""Benchmark harness for monoprop.

The reusable half of monoprop's benchmark suite: memory accounting
(:mod:`monoprop_bench_tools.memory`), the builders for the benchmarked problems
(:mod:`monoprop_bench_tools.models`), and the two renderers of a run's artifacts
(:mod:`monoprop_bench_tools.report`, :mod:`monoprop_bench_tools.bmf`).

Submodules are not imported eagerly, so ``import monoprop_bench_tools`` stays
cheap and does not require the optional GPU dependencies.
"""

from __future__ import annotations

from monoprop_bench_tools._version import __version__, __version_tuple__

__all__ = ["__version__", "__version_tuple__"]
