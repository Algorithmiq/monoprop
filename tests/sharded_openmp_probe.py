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

"""Fresh-process probe for ``tests/test_sharded_openmp.py``.

Run as ``python sharded_openmp_probe.py '<json spec>'`` in a process whose thread configuration was set before
``monoprop`` is imported. It drives only the public API (plus the raw ``_core`` constructor where a test needs an
explicit legacy control) and prints one JSON document on its last stdout line. Floats are emitted as ``float.hex`` so
the parent can compare bits; the parent decides what has to agree.

Not collected by pytest: the file name has no ``test_`` prefix.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS.parent))

import monoprop  # noqa: E402
from monoprop import (  # noqa: E402
    Circuit,
    ExpGate,
    MajoranaPropagator,
    Pauli,
    PauliOperator,
    PauliPropagator,
    _core,
)
from monoprop.circuit import expand_monomials  # noqa: E402
from tests.cases import load_problem  # noqa: E402


def _hex(value: float) -> str:
    return float(value).hex()


def _hexes(values: Any) -> list[str]:  # noqa: ANN401
    return [_hex(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def _raw_map(propagator: Any, params: list[float], atol: float) -> dict[str, list[str]]:  # noqa: ANN401
    raw = propagator._simulator.evolved_operator(params, atol)
    return {
        json.dumps(list(key)): [_hex(complex(v).real), _hex(complex(v).imag)]
        for key, v in raw.items()
    }


def _breakdowns(propagator: Any) -> dict[str, Any]:  # noqa: ANN401
    sim = propagator._simulator
    return {
        "size": sim.size(),
        "graph_size": list(sim.graph_size()),
        "graph_layers": sim.graph_layers(),
        "n_gates": sim.n_gates(),
        "operator": dict(sim.operator_memory_breakdown()),
        "graph": dict(sim.graph_memory_breakdown()),
        "operator_bytes": sim.operator_memory_bytes(),
        "graph_bytes": sim.graph_memory_bytes(),
    }


class _Problem:
    """A fixture: a factory for fresh propagators, a circuit and evaluation parameters."""

    def __init__(self, name: str, picture: str, informed_atol: float | None) -> None:
        self.name = name
        self.schrodinger = picture == "schrodinger"
        self.informed_atol = informed_atol
        if name.startswith("pauli_deep"):
            self._pauli_deep(name)
        else:
            self._majorana(name)

    def _majorana(self, name: str) -> None:
        files = {
            "random_exact": "random_exact",
            "rx_rz": "rx_rz_ry_rz_exact",
            "lih": "lih_fermionic_spin_exact",
            "s0_8e8o": "S0_8e8o_majoranic_c8",
        }
        problem = load_problem(_TESTS / "data" / f"{files[name]}.msgpack")
        self.basis = "majorana"
        self.circuit = problem.monomial_circuit.to_circuit()
        self.params = [float(v) for v in problem.monomial_circuit.parameters]
        self.operator = problem.operator
        self.state = list(problem.monomial_circuit.initial_state)
        self.num_modes = problem.n_modes
        cutoff = {
            "random_exact": 2 * problem.n_modes,
            "rx_rz": 2,
            "lih": 4,
            "s0_8e8o": 6,
        }[name]
        self.kwargs: dict[str, Any] = {"cutoff": cutoff}
        if self.schrodinger:
            self.kwargs["schrodinger_cutoff"] = min(cutoff, 2 * problem.n_modes)

    def _pauli_deep(self, name: str) -> None:
        # The deep-gradient regression family (tests/test_deep_circuit_gradient.py): repeated parameters, records and,
        # at theta = pi/4, a vanishing cosine.
        layers, theta = (2, 1.0) if name == "pauli_deep" else (2, math.pi / 4)
        num_qubits = 6
        observable = PauliOperator(
            {
                Pauli("ZZ", [i, j]): 0.5
                for i, j in itertools.combinations(range(num_qubits), 2)
            },
            num_qubits=num_qubits,
        )
        gates = []
        for k in range(layers):
            gates += [
                ExpGate(PauliOperator({pauli: v}, num_qubits=num_qubits), index=k)
                for pauli, v in observable.terms.items()
            ]
            gates += [
                ExpGate(
                    PauliOperator({Pauli("X", i): -1.0}, num_qubits=num_qubits),
                    index=layers + k,
                )
                for i in range(num_qubits)
            ]
        self.basis = "pauli"
        self.params = [theta] * (2 * layers)
        self.circuit = Circuit(
            gates=gates, system_size=num_qubits, parameters=self.params
        )
        self.operator = observable
        self.state = [0, 2]
        self.num_modes = num_qubits
        self.kwargs = {"cutoff": 3}
        if self.schrodinger:
            self.kwargs["schrodinger_cutoff"] = 3

    def make(self, **overrides: Any) -> Any:  # noqa: ANN401
        kwargs = {**self.kwargs, **overrides}
        cls = PauliPropagator if self.basis == "pauli" else MajoranaPropagator
        return cls(self.operator, self.state, **kwargs)

    def scaled_operator(self, factor: float) -> Any:  # noqa: ANN401
        terms = {k: v * factor for k, v in self.operator.terms.items()}
        if self.basis == "pauli":
            return PauliOperator(terms, num_qubits=self.num_modes)
        return type(self.operator)(terms, self.num_modes)


def _evaluations(propagator: Any, params: list[float]) -> dict[str, Any]:  # noqa: ANN401
    energy = propagator.expectation_value(params)
    value, grad = propagator.expectation_value_and_gradient(params)
    unpared = propagator.expectation_value_and_gradient_functional()
    pared = propagator.expectation_value_and_gradient_functional(1e-3)
    pared_energy = propagator.expectation_value_functional(0.05)
    out: dict[str, Any] = {
        "energy": _hex(energy),
        "value": _hex(value),
        "gradient": _hexes(grad),
    }
    for label, fn in (("unpared", unpared), ("pared", pared)):
        v, g = fn(params)
        out[f"{label}_value"] = _hex(v)
        out[f"{label}_gradient"] = _hexes(g)
        # A second call of the same functional must answer identically.
        v2, g2 = fn(params)
        out[f"{label}_repeat"] = _hex(v2) == _hex(v) and _hexes(g2) == _hexes(g)
    out["pared_energy"] = _hex(pared_energy(params))
    return out


def _scenario_full(spec: dict[str, Any]) -> dict[str, Any]:
    problem = _Problem(spec["fixture"], spec["picture"], spec.get("informed_atol"))
    params = problem.params
    out: dict[str, Any] = {}

    # The public build_graph seeds a first build with the circuit's parameters (the coefficient-informed engine path);
    # the engine's structural path is reached by omitting them at the binding.
    graph = problem.make()
    graph._simulator.build_graph(*_expanded(graph, problem.circuit), None, None)
    graph._n_params = problem.circuit.n_parameters
    out["graph"] = {
        **_evaluations(graph, params),
        "map0": _raw_map(graph, params, 0.0),
        "map": _raw_map(graph, params, 1e-12),
        "contract": _hexes(graph.contract_partially(params, inplace=False)),
        "aggregates": _breakdowns(graph),
        "parameter_mapping": graph.parameter_mapping,
        "python_terms": len(graph.evolved_operator(params, atol=0.0).terms),
    }
    # Non-inplace contraction leaves the graph and its answers unchanged.
    out["graph"]["after_contract_energy"] = _hex(graph.expectation_value(params))

    # Copy, then mutate the copy: the source answers as before; the copy diverges.
    duplicate = copy.deepcopy(graph)
    out["copy"] = {"energy": _hex(duplicate.expectation_value(params))}
    duplicate.update_initial_operator(problem.scaled_operator(2.0))
    out["copy"]["scaled_energy"] = _hex(duplicate.expectation_value(params))
    out["copy"]["source_energy"] = _hex(graph.expectation_value(params))

    # Coefficient-informed construction: the circuit's own parameters seed the lower-atol truncation.
    informed = problem.make(lower_atol=spec.get("informed_atol", 1e-3))
    informed.build_graph(problem.circuit)
    out["informed"] = {
        **_evaluations(informed, params),
        "map0": _raw_map(informed, params, 0.0),
        "aggregates": _breakdowns(informed),
    }

    # Graph-free propagation.
    direct = problem.make()
    direct.propagate(problem.circuit)
    out["propagate"] = {
        "energy": _hex(direct.expectation_value([])),
        "map0": _raw_map(direct, [], 0.0),
        "contract": _hexes(direct.contract_partially([], inplace=False)),
        "aggregates": _breakdowns(direct),
    }

    # Incremental axis: build twice (structural extension), evaluate on the doubled axis, contract in place, then
    # build again (informed by the contracted picture) and evaluate.
    incremental = problem.make()
    incremental.build_graph(problem.circuit)
    incremental.build_graph(problem.circuit)
    doubled = params + params
    out["incremental"] = {"energy": _hex(incremental.expectation_value(doubled))}
    out["incremental"]["contract"] = _hexes(
        incremental.contract_partially(doubled, inplace=True)
    )
    out["incremental"]["after_contract_layers"] = incremental.graph_layers
    out["incremental"]["contracted_energy"] = _hex(incremental.expectation_value([]))
    incremental.build_graph(problem.circuit)
    out["incremental"]["rebuilt_energy"] = _hex(incremental.expectation_value(params))
    out["incremental"]["rebuilt_gradient"] = _hexes(incremental.gradient(params))
    out["incremental"]["map0"] = _raw_map(incremental, params, 0.0)

    # Remapping: every layer driven by parameter 0; then settings updates followed by a further build.
    remap = problem.make()
    remap.build_graph(problem.circuit)
    remap.parameter_mapping = [0] * remap.graph_layers
    out["remap"] = {
        "value": _hex(remap.expectation_value([params[0]])),
        "gradient": _hexes(remap.gradient([params[0]])),
    }
    settings = problem.make(lower_atol=1e-9, upper_atol=1e-1)
    settings.build_graph(problem.circuit)
    settings.cutoff = settings.cutoff + 1
    settings.lower_atol = 1e-6
    settings.build_graph(problem.circuit, seed_parameters=params + params)
    out["settings"] = {
        "energy": _hex(settings.expectation_value(params + params)),
        "map0": _raw_map(settings, params + params, 0.0),
    }

    # Initial-operator updates: an earlier functional is invalidated, a new one sees the new coefficients.
    updates = problem.make()
    updates.build_graph(problem.circuit)
    stale = updates.expectation_value_functional()
    updates.update_initial_operator(problem.scaled_operator(0.5))
    try:
        stale(params)
        out["stale_functional"] = "answered"
    except Exception as exc:  # noqa: BLE001 - the parent checks the type and message
        out["stale_functional"] = f"{type(exc).__name__}: {exc}"
    out["updated"] = {
        "energy": _hex(updates.expectation_value(params)),
        "functional": _hex(updates.expectation_value_functional()(params)),
        "map0": _raw_map(updates, params, 0.0),
    }
    return out


def _expanded(propagator: Any, circuit: Circuit) -> tuple[Any, ...]:  # noqa: ANN401
    gates = propagator._circuit_gates(circuit)
    majoranas, gen_coeffs, mapping, gate_indices = expand_monomials(
        gates,
        circuit.resolved_mapping,
        propagator._system_size,
    )
    return majoranas, mapping, gen_coeffs, gate_indices


def _attempt(fn: Any) -> str:  # noqa: ANN401
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the parent checks the type and message
        return f"{type(exc).__name__}: {exc}"
    return "ok"


def _scenario_controls(spec: dict[str, Any]) -> dict[str, Any]:
    terms = {(0, 1): 1j, (2, 3): 0.5j, (1, 2): 0.25j}

    def construct(**kwargs: Any) -> Any:  # noqa: ANN401
        return _core.MonomialPropagator032(
            initial_operator=terms, cutoff=4, initial_state=[0, 1], **kwargs
        )

    out: dict[str, Any] = {}
    for partitions in spec.get("partitions", []):
        out[f"partitions={partitions}"] = _attempt(
            lambda p=partitions: construct(partitions=p)
        )
    out["default"] = _attempt(construct)
    if spec.get("construction_only"):
        return out
    out["logical_modes_zero"] = _attempt(lambda: construct(logical_num_modes=0))
    out["logical_modes_too_large"] = _attempt(lambda: construct(logical_num_modes=33))
    # A logical width below capacity: indices at or above 2 * logical_num_modes are rejected.
    out["logical_modes_index"] = _attempt(
        lambda: _core.MonomialPropagator032(
            initial_operator={(0, 9): 1j},
            cutoff=4,
            initial_state=[0],
            logical_num_modes=4,
        )
    )
    narrow = _core.MonomialPropagator032(
        initial_operator={(0, 1): 1j, (2, 3): 0.5j},
        cutoff=4,
        initial_state=[0],
        logical_num_modes=4,
    )
    out["logical_generator_index"] = _attempt(
        lambda: narrow.build_graph([[0, 8]], [0], [1.0])
    )
    narrow.build_graph([[1, 2], [0, 3]], [0, 1], [1.0, 1.0])
    out["logical_energy"] = _hex(narrow.expectation_value([0.3, 0.2]))

    # Empty circuits, empty operators and identity generators.
    empty = _core.MonomialPropagator032(initial_operator={}, cutoff=4, initial_state=[])
    empty.build_graph([], [], [])
    out["empty_operator"] = {
        "size": empty.size(),
        "energy": _hex(empty.expectation_value([])),
        "map": _attempt(lambda: empty.evolved_operator([], 0.0)),
    }
    empty.build_graph([[0, 1]], [0], [1.0])
    out["empty_operator"]["energy_after_gate"] = _hex(empty.expectation_value([0.4]))
    identity = construct()
    identity.build_graph([[]], [0], [1.0])
    out["identity_generator"] = {
        "layers": identity.graph_layers(),
        "energy": _hex(identity.expectation_value([0.7])),
        "gradient": _hexes(identity.expectation_value_and_gradient([0.7])[1]),
    }
    core_only = _core.MonomialPropagator032(
        initial_operator={(): 2.5}, cutoff=4, initial_state=[]
    )
    core_only.build_graph([[0, 1]], [0], [1.0])
    out["core_only"] = {
        "energy": _hex(core_only.expectation_value([0.3])),
        "map": {
            json.dumps(list(k)): _hex(complex(v).real)
            for k, v in core_only.evolved_operator([0.3], 0.0).items()
        },
    }

    # Pre-mutation rejection leaves the object usable; a failure after mutation started invalidates it.
    sim = construct()
    sim.build_graph([[0, 2], [1, 3]], [0, 1], [1.0, 1.0])
    out["bad_parameter_count"] = _attempt(lambda: sim.expectation_value([0.1]))
    out["usable_after_rejection"] = _attempt(lambda: sim.expectation_value([0.3, 0.2]))
    out["propagate_on_graph"] = _attempt(
        lambda: sim.propagate([[0, 2]], [0], [1.0], [0.1])
    )
    out["missing_term_update"] = _attempt(
        lambda: sim.update_initial_operator({(4, 5): 1j})
    )
    out["after_missing_term"] = _attempt(lambda: sim.expectation_value([0.3, 0.2]))
    out["copy_of_invalid"] = _attempt(lambda: copy.deepcopy(sim))
    return out


def main() -> None:
    spec = json.loads(sys.argv[1])
    result: dict[str, Any] = {
        "runtime": getattr(_core, "__runtime_identity__", None),
        "has_mpi": monoprop.has_mpi,
        "core_file": _core.__file__,
    }
    scenario = spec["scenario"]
    if scenario == "identity":
        pass
    elif scenario == "full":
        result["full"] = _scenario_full(spec)
    elif scenario == "controls":
        result["controls"] = _scenario_controls(spec)
    else:
        msg = f"unknown scenario {scenario!r}"
        raise ValueError(msg)
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
