# monoprop-qiskit

Convert between [Qiskit](https://www.ibm.com/quantum/qiskit) circuits/operators and
[monoprop](https://github.com/Algorithmiq/monoprop)'s native representations
(`Circuit`, `PauliOperator`), so a Qiskit-built problem can be propagated with monoprop
and the result handed back to Qiskit.

| Function | What it does |
| --- | --- |
| `from_qiskit_operator` | Convert a Qiskit `SparsePauliOp`, `SparseObservable`, or `Pauli` to a `PauliOperator`. Requires the operator to be Hermitian. |
| `to_qiskit_operator` | Convert a `PauliOperator` to a Qiskit `SparsePauliOp`. |
| `from_qiskit_circuit` | Convert a Qiskit circuit built only of `PauliEvolutionGate`, `PauliProductRotationGate`, or equivalent rotations to a monoprop `Circuit`. |
| `to_qiskit_circuit` | Convert a monoprop `Circuit` to a Qiskit circuit of `PauliEvolutionGate`s. |

## Install

```bash
pip install monoprop-qiskit
```

## Use

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from monoprop_qiskit import (
    from_qiskit_circuit,
    from_qiskit_operator,
    to_qiskit_circuit,
    to_qiskit_operator,
)

operator = from_qiskit_operator(SparsePauliOp("XX"))

qiskit_circuit = QuantumCircuit(2)
qiskit_circuit.append(PauliEvolutionGate(SparsePauliOp("ZZ"), time=0.5), [0, 1])
circuit = from_qiskit_circuit(qiskit_circuit, initial_state=[0, 0])

# ... propagate `circuit` with monoprop, then convert back.
to_qiskit_operator(operator)
to_qiskit_circuit(circuit, num_qubits=2)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
