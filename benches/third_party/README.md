# monoprop benchmarks

Standalone benchmark scripts comparing [monoprop](https://github.com/Algorithmiq/monoprop) against
other Pauli-propagation implementations — [QuEra's `ppvm`](https://github.com/QuEraComputing/ppvm),
Qiskit's `pauli-prop`, NVIDIA's `cuPauliProp` (GPU, via `cuquantum`), and
[`PauliPropagation.jl`](https://github.com/SparqleSim/PauliPropagation.jl) — on the same workload:
Trotterized time evolution of a 2D transverse-field Ising model (TFIM), tracking the runtime,
expectation value, and operator size at each Trotter step.

This directory is a self-contained `uv` project, completely isolated from the main monoprop
development environment. It has its own virtual environment and its own external dependencies.

The end-to-end workflow is:

1. Choose the simulation settings in `settings.json`.
2. Set up the Python environment.
3. (Optional) install Julia and `PauliPropagation.jl`
4. Run the benchmarks to produce `results.json`.
5. Plot the results to produce `runtime.png` and `memory.png`.


For more details, check the documentation at [benchmark guide](https://docs.algorithmiq.fi/monoprop/benchmarks.html)
