# monoprop-cirq

Convert between [Cirq](https://quantumai.google/cirq) Pauli operators/qubit circuits and
[monoprop](https://github.com/Algorithmiq/monoprop)'s native representations
(`Circuit`, `PauliOperator`), so a Cirq-built problem can be propagated with monoprop
and the result handed back to Cirq.

| Function | What it does |
| --- | --- |
| `from_cirq_operator` | Convert a `cirq.PauliSum`, `cirq.PauliString` or single-qubit Pauli operation with real coefficients to a `PauliOperator`. |
| `to_cirq_operator` | Convert a `PauliOperator` to a `cirq.PauliSum`. |
| `from_cirq_circuit` | Convert a Cirq circuit whose every gate is `exp(i t G)` for a commuting Pauli generator `G` to a monoprop `Circuit`. |
| `to_cirq_circuit` | Convert a bound monoprop `Circuit` to a Cirq circuit of `cirq.PauliSumExponential` gates. |

## Install

The package is not on PyPI yet. From a checkout of the monoprop repository,
`uv sync --all-groups` installs it as a workspace member; to add it to another
environment that already has monoprop:

```bash
pip install ./packages/monoprop-cirq
```

It depends on `cirq-core`, not the `cirq` metapackage.

## Use

```python
import cirq
from monoprop_cirq import from_cirq_circuit, from_cirq_operator, to_cirq_circuit

from monoprop import PauliPropagator

q = cirq.LineQubit.range(2)
circuit = cirq.Circuit((cirq.CNOT**0.3)(q[0], q[1]), cirq.rz(0.2)(q[1]), cirq.ry(0.4)(q[0]))
observable = cirq.Z(q[0]) * cirq.Z(q[1]) + 0.5 * cirq.X(q[1])

# Qubit 0 starts in |1>: the initial state is passed separately, not as an X gate.
propagator = PauliPropagator(from_cirq_operator(observable), [0], cutoff=2)
propagator.propagate(from_cirq_circuit(circuit, [0]))
propagator.expectation_value()

# ... and back to Cirq.
to_cirq_circuit(from_cirq_circuit(circuit, [0]))
```

## Conventions

- **Qubits.** `cirq.LineQubit(x)` becomes qubit `x`, so an observable and a circuit converted
  separately line up. Other qubit types (`cirq.GridQubit`, `cirq.NamedQubit`) need an explicit
  `qubit_order=`, either a list of qubits or a `cirq.QubitOrder`.
- **Gates.** Every `cirq.EigenGate` whose generator has commuting Pauli terms converts exactly:
  rotations, X/Y/Z/XX/YY/ZZ powers, `CZ`, `CNOT`, `ISWAP`, `SWAP`, `CCZ`, `CCX` and their powers.
  So do `cirq.PauliStringPhasor`, `cirq.PauliSumExponential` and Pauli-string operations. The
  parameter is the gate's exponent, or its angle in radians for `cirq.rx`/`ry`/`rz`. Other gates
  (`cirq.H`, `cirq.PhasedXZGate`, controlled rotations, ...) raise; run `cirq.decompose` on them first.
- **Measurements** raise; drop terminal ones with `cirq.drop_terminal_measurements`. Sympy
  symbols must be resolved, e.g. with `param_resolver=`.
- **Initial state.** `to_cirq_circuit` leaves it out; prepend `cirq.X` on
  `circuit.initial_state` to run its result in Cirq.

## License

Apache-2.0. See [LICENSE](LICENSE).
