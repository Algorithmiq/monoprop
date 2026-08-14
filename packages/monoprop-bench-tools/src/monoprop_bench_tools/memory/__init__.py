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

"""Peak-memory accounting for benchmarks.

Two backends with a deliberately parallel surface --
:mod:`monoprop_bench_tools.memory.cpu` for the host (kernel ``VmHWM``) and
:mod:`monoprop_bench_tools.memory.gpu` for the device (CUDA memory-pool
counters). Neither is re-exported here: the choice of backend is the caller's,
and importing the GPU one is what pulls in the optional ``gpu`` extra.
"""

from __future__ import annotations
