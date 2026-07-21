# monoprop benchmarks

Standalone benchmark script comparing [monoprop](https://github.com/Algorithmiq/monoprop) against
MajoranaPropagation.jl on the same workload:
1D Hubbard model, tracking the runtime, expectation value, and operator size at each Trotter step.


The end-to-end workflow is:

1. Set up the Python environment.
2. Install Julia and `MajoranaPropagation.jl`.
3. Run the benchmarks to produce `results.json`.
4. Plot the results to produce `runtime.png` and `memory.png`.

For more details, see the [benchmarks documentation](https://monoprop.readthedocs.io/en/latest/docs/benchmarks.html).
