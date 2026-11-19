# monoprop-pennylane

Convert between [PennyLane](https://pennylane.ai) circuits/operators and
[monoprop](https://github.com/Algorithmiq/monoprop)'s native representations
(`Circuit`, `PauliOperator`), so a PennyLane-built problem can be propagated with monoprop
and the result handed back to PennyLane.

| Function | What it does |
| --- | --- |
| `from_pennylane_operator` | Convert a PennyLane operator with a Pauli decomposition (e.g. a `qml.ops.LinearCombination`/`qml.Hamiltonian`) to a `PauliOperator`. |
| `to_pennylane_operator` | Convert a `PauliOperator` to a PennyLane `qml.ops.LinearCombination`. |
| `from_pennylane_circuit` | Convert a PennyLane quantum function or `QNode` built only of single-parameter, Pauli-generator gates to a monoprop `Circuit`. |
| `to_pennylane_circuit` | Convert a monoprop `Circuit` to a reusable PennyLane quantum function of `qml.exp` gates. |

## Install

```bash
pip install monoprop-pennylane
```

## Use

```python
import pennylane as qml

from monoprop_pennylane import (
    from_pennylane_circuit,
    from_pennylane_operator,
    to_pennylane_circuit,
    to_pennylane_operator,
)

operator = from_pennylane_operator(
    qml.ops.LinearCombination([1.0], [qml.PauliX(0) @ qml.PauliZ(1)])
)


def qfunc(theta):
    qml.RX(theta, wires=0)


circuit = from_pennylane_circuit(qfunc, [], 0.5)

# ... propagate `circuit` with monoprop, then convert back.
to_pennylane_operator(operator)
to_pennylane_circuit(circuit)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
