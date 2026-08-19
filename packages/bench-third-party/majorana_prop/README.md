# monoprop benchmarks

Standalone benchmark script comparing [monoprop](https://github.com/Algorithmiq/monoprop) against
MajoranaPropagation.jl on the same workload:
1D Hubbard model, tracking the runtime, expectation value, and operator size at each Trotter step.


The end-to-end workflow is:

1. Set up the Python environment.
2. Install Julia and `MajoranaPropagation.jl`.
3. Run `run_benchmarks.sh` to append to `monoprop_hubbard1d_benchmark_results.jsonl` and
   `julia_hubbard1d_benchmark_results.jsonl` (one JSON line per case).
4. Run `plot_results.py` to produce `majorana_results.png`.

Both backends are multithreaded, and each is fastest at a different thread count, so
`monoprop_NUM_THREADS` and `JULIA_NUM_THREADS` are read from the environment (default 8 each).
Every result row records `num_threads`, measured `cpu_seconds` and `busy_cores`
(= CPU seconds / wall seconds), so thread effectiveness is visible per case rather than
assumed: `busy_cores` near 1 means the run was serial no matter what the thread count said.
The Julia rows also record `container` — only `VectorMajoranaSum` reaches the
AcceleratedKernels path; the dict-backed `MajoranaSum` is serial.

For more details, see the [benchmarks documentation](https://monoprop.readthedocs.io/en/latest/docs/benchmarks.html).
