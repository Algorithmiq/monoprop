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

"""The storage-block rule shared by tools/generate-binders.py and generate-dispatch.py.

These are the invariants that silently break the build if the two generators disagree: the
bindings instantiate `binding_blocks()`, dispatch routes each logical mode count to
`binding_block()`, and every routed block must be one that was instantiated.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "_binding_layout.py"
_spec = importlib.util.spec_from_file_location("_binding_layout", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
binding_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(binding_layout)

binding_block = binding_layout.binding_block
binding_blocks = binding_layout.binding_blocks
BLOCK = binding_layout.MODES_PER_STORAGE_BLOCK


@pytest.mark.parametrize(
    ("logical_num_modes", "expected"),
    [(1, 32), (31, 32), (32, 32), (33, 64), (64, 64), (65, 96), (250, 256)],
)
def test_binding_block_rounds_up_to_a_whole_word(logical_num_modes, expected):
    assert binding_block(logical_num_modes) == expected


@pytest.mark.parametrize("max_logical_num_modes", [1, 32, 33, 64, 250])
def test_blocks_are_ascending_multiples_of_the_word_width(max_logical_num_modes):
    blocks = binding_blocks(max_logical_num_modes)
    assert blocks[0] == BLOCK
    assert blocks == sorted(blocks)
    assert all(block % BLOCK == 0 for block in blocks)


@pytest.mark.parametrize("max_logical_num_modes", [1, 32, 33, 64, 250])
def test_every_dispatched_block_is_instantiated(max_logical_num_modes):
    """The drift guard: dispatch must never route at a template the bindings did not emit."""
    instantiated = set(binding_blocks(max_logical_num_modes))
    dispatched = {binding_block(modes) for modes in range(1, max_logical_num_modes + 1)}
    assert dispatched <= instantiated
    # And nothing is instantiated that no mode count reaches.
    assert instantiated == dispatched
