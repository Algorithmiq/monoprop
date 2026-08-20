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

"""Dict-key wiring of the two memory-breakdown bindings.

Both dicts are built inside nanobind lambdas with no C++ entry point returning the map, so which
struct field each string key was bound to -- and that the ``d_``-prefixed diagnostics stay outside
``total_bytes`` -- is observable only from here.
"""

from __future__ import annotations

import pytest

from monoprop import Circuit, ExpGate, MajoranaPropagator
from monoprop.majorana import MajoranaOperator

_NUM_MODES = 2
# The full width: nothing these one-gate models build can be truncated.
_CUTOFF = 2 * _NUM_MODES

_GRAPH_BYTES = (
    "layer_descriptor_bytes",
    "layer_storage_object_bytes",
    "cos_data_bytes",
    "cross_rank_bytes",
    "exchange_layout_bytes",
)
_GRAPH_DIAGNOSTICS = (
    "d_slot_record_bytes",
    "d_layer_cores",
    "d_slot_records",
    "d_occupied_slots",
    "d_cross_rank_endpoints",
)
_OPERATOR_BYTES = (
    "operator_terms_bytes",
    "op_coeffs_bytes",
    "state_coeffs_bytes",
    "indexing_bytes",
    "init_operator_bytes",
    "initial_state_bytes",
    "inverted_index_bytes",
)
_OPERATOR_DIAGNOSTICS = (
    "d_invidx_dense_bytes",
    "d_invidx_sparse_bytes",
    "d_invidx_dense_columns",
    "d_terms_slack_bytes",
    "d_state_coeffs_nonzero",
)

# CrossRankPartnerRange is two size_t and three TermIndex, the latter 4 B or 8 B under
# monoprop_WIDE_TERM_INDEX, so the padded record is 32 B or 40 B and nothing between.
_SLOT_RECORD_BYTES = (32, 40)


def _observable():
    return MajoranaOperator({(0, 1): 1.0j}, num_modes=_NUM_MODES)


def _idle_gate():
    """A generator that rotates nothing: |G||M| - |G & M| = 4 - 2 is even, so it commutes."""
    return ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=_NUM_MODES))


def _rotating_gate():
    """A generator that rotates the observable's one term to gamma_1: 2 - 1 is odd."""
    return ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=_NUM_MODES))


def _built(serial_comm, monkeypatch, partitions, gate):
    """A propagator over a flat world of exactly ``partitions`` slots, with ``gate`` built in.

    The partition count is pinned because it otherwise resolves to the host's physical-core
    count, which would put a machine-dependent P in every slot-record expectation below.
    """
    monkeypatch.setenv("monoprop_PARTITIONS", str(partitions))
    mp = MajoranaPropagator(_observable(), [0, 1], cutoff=_CUTOFF, comm=serial_comm)
    mp.build_graph(Circuit((gate,), _NUM_MODES))
    return mp


def test_graph_breakdown_keys_are_exactly_the_bound_set(serial_comm, monkeypatch):
    mp = _built(serial_comm, monkeypatch, 1, _rotating_gate())
    assert set(mp._simulator.graph_memory_breakdown()) == {
        *_GRAPH_BYTES,
        *_GRAPH_DIAGNOSTICS,
        "total_bytes",
    }


def test_operator_breakdown_keys_are_exactly_the_bound_set(serial_comm, monkeypatch):
    mp = _built(serial_comm, monkeypatch, 1, _rotating_gate())
    assert set(mp._simulator.operator_memory_breakdown()) == {
        *_OPERATOR_BYTES,
        *_OPERATOR_DIAGNOSTICS,
        "total_bytes",
    }


@pytest.mark.parametrize(
    ("breakdown_name", "byte_keys", "diagnostic_keys"),
    [
        ("graph_memory_breakdown", _GRAPH_BYTES, _GRAPH_DIAGNOSTICS),
        ("operator_memory_breakdown", _OPERATOR_BYTES, _OPERATOR_DIAGNOSTICS),
    ],
)
def test_diagnostics_are_excluded_from_total_bytes(
    serial_comm, monkeypatch, breakdown_name, byte_keys, diagnostic_keys
):
    mp = _built(serial_comm, monkeypatch, 1, _rotating_gate())
    breakdown = getattr(mp._simulator, breakdown_name)()

    assert breakdown["total_bytes"] == sum(breakdown[key] for key in byte_keys)
    assert breakdown["total_bytes"] > 0
    # Vacuous unless a diagnostic is nonzero: each one slices or counts a field already summed.
    assert max(breakdown[key] for key in diagnostic_keys) > 0


@pytest.mark.parametrize(
    ("partitions", "gate", "expected"),
    [
        pytest.param(
            1,
            _idle_gate,
            {
                "d_layer_cores": 1,
                "d_slot_records": 1,
                "d_occupied_slots": 0,
                "d_cross_rank_endpoints": 0,
            },
            id="idle-one-slot",
        ),
        pytest.param(
            2,
            _idle_gate,
            {
                "d_layer_cores": 2,
                "d_slot_records": 4,
                "d_occupied_slots": 0,
                "d_cross_rank_endpoints": 0,
            },
            id="idle-four-slots",
        ),
        pytest.param(
            1,
            _rotating_gate,
            {
                "d_layer_cores": 1,
                "d_slot_records": 1,
                "d_occupied_slots": 1,
                "d_cross_rank_endpoints": 2,
            },
            id="one-rotation-two-endpoints",
        ),
    ],
)
def test_graph_diagnostics_each_carry_their_own_quantity(
    serial_comm, monkeypatch, partitions, gate, expected
):
    mp = _built(serial_comm, monkeypatch, partitions, gate())
    breakdown = mp._simulator.graph_memory_breakdown()

    assert {key: breakdown[key] for key in expected} == expected
    # One LayerCore per gate per partition, each carrying one record per flat-world slot: the two
    # counts differ only by P, so this pair is what separates them.
    assert breakdown["d_layer_cores"] == partitions * mp.graph_layers
    assert breakdown["d_slot_records"] == breakdown["d_layer_cores"] * partitions

    record_bytes, remainder = divmod(
        breakdown["d_slot_record_bytes"], breakdown["d_slot_records"]
    )
    assert remainder == 0
    assert record_bytes in _SLOT_RECORD_BYTES
    assert breakdown["d_slot_record_bytes"] <= breakdown["cross_rank_bytes"]
    assert breakdown["cross_rank_bytes"] <= breakdown["total_bytes"]

    # A slot carrying traffic holds at least one endpoint, so both are ceilings on the occupancy.
    assert breakdown["d_occupied_slots"] <= breakdown["d_slot_records"]
    assert breakdown["d_occupied_slots"] <= breakdown["d_cross_rank_endpoints"]


def test_operator_diagnostics_slice_the_fields_they_name(serial_comm, monkeypatch):
    mp = _built(serial_comm, monkeypatch, 1, _rotating_gate())
    breakdown = mp._simulator.operator_memory_breakdown()

    tiers = breakdown["d_invidx_dense_bytes"] + breakdown["d_invidx_sparse_bytes"]
    # The remainder of inverted_index_bytes is the row-parity words, which neither tier covers.
    assert 0 < tiers <= breakdown["inverted_index_bytes"]
    # A dense column is promoted only with a posting in it, so it owns at least one word.
    assert breakdown["d_invidx_dense_columns"] * 8 <= breakdown["d_invidx_dense_bytes"]
    assert breakdown["d_terms_slack_bytes"] <= breakdown["operator_terms_bytes"]
    # A scored state row costs at least a TermIndex of the state's own byte total.
    assert breakdown["d_state_coeffs_nonzero"] * 4 <= breakdown["state_coeffs_bytes"]
