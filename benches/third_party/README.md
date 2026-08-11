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

## Layout of the Pauli benchmarks

`pauli_prop/` is split so the model is defined once and the backends stay independent:

| file | role |
| ---- | ---- |
| `model.py` | the TFIM model — lattice, Trotter step, observable. Imports no propagation library. |
| `backends.py` | one function per backend, each importing its own dependencies *inside* the function, so a missing GPU or extension only affects that backend. |
| `run_model.py` | fixed size, all Python backends in one process, per-step curves → `results.json`. |
| `run_model.jl` | adds the `PauliPropagation.jl` column to an existing `results.json`. |
| `plot_results.py` | per-step runtime and memory curves → `pauli_results.png`. |
| `run_scaling.py` | sweeps the lattice size, one subprocess per (size, backend) → a JSONL of totals. |
| `run_one.py` | one backend at one size; the unit of work `run_scaling.py` spawns. |
| `run_scaling.jl` | the `PauliPropagation.jl` counterpart of `run_one.py`, same record schema. |
| `plot_scaling.py` | total runtime and final memory vs lattice size, one figure each → `pauli_scaling_runtime.png`, `pauli_scaling_memory.png`. |
| `plot_speedup.py` | monoprop's speed-up over each other backend, per size → `pauli_speedup.png`. |
| `scaling_results.jsonl` | the committed 36→324-qubit ladder, so both figures can be redrawn without re-measuring. |

Any lattice size works: `--nx/--ny` override `settings.json`, and the observable follows the
lattice (the central horizontally-adjacent bond, which is the committed `[20, 21]` at 6x6)
rather than staying pinned to indices that a resize would invalidate.

### Scaling with lattice size

```bash
# Redraw both committed figures from the committed ladder — seconds, no propagation
uv run python plot_scaling.py scaling_results.jsonl
uv run python plot_speedup.py scaling_results.jsonl

# Or measure your own: square grids 6x6 ... 18x18 (36 -> 324 qubits), 5 minutes per point
uv run python run_scaling.py --output results/scaling.jsonl --timeout 300
uv run python plot_scaling.py results/scaling.jsonl --output-dir results
```

Both plotting scripts also write a markdown sidecar (`pauli_scaling.md`, `pauli_speedup.md`)
holding the same numbers as a table, followed by a provenance table: the host, thread cap and
library version behind each backend's points. That is deliberately *not* in the figure titles —
a title says what was computed, and the machine it ran on qualifies the numbers, so it sits
with them.

Each point runs as its own process, so a backend that exceeds its budget or gets
OOM-killed costs only that point: it is recorded with a non-`ok` status, skipped at every
larger size (`--no-skip-after-fail` to keep trying), and drawn as a curve that simply ends at
the last size it completed — the status itself is in the table and the JSONL, not on the axes.
Select a subset with `--backends`, and sweep a strip
instead of squares with `--ny`.

Two things to know when reading the output:

- **The memory column is one quantity for every backend**: the peak resident set size over
  each step, read from the kernel's `VmHWM` high-water mark and reset per step, so the
  curves may be compared directly. Where a library also accounts for itself, that figure is
  kept separately as `operator_memory_MB` / the `native_memory` series — those are *not*
  commensurable across backends (one counts an operator, another an object graph, another
  device memory), so never plot them against each other. `HOST_MEMORY_METRIC` and
  `OPERATOR_MEMORY_METRICS` in `backends.py` record which is which, and every record also
  carries `peak_rss_MB` for the process lifetime.
- **The GPU backend's own figure is a device high-water mark, not an end-of-step reading**,
  so a transient freed inside a step still counts. `benches/_memory_gpu.py` picks the strongest
  counter the allocator allows and names it in `operator_memory_metric`: CUDA's resettable
  `cudaMemPoolAttrUsedMemHigh` when CuPy runs on `malloc_async` (exact), otherwise the
  CuPy pool's monotone `total_bytes` (an upper bound). Run `python ../_memory_gpu.py` on the
  GPU host to see which one is active and confirm it catches a freed transient.
- **`PauliPropagation.jl` runs in its fastest documented configuration**, which is not its
  default: the `VectorPauliSum` container driven by `Performance.propagate!`, with
  coefficient truncation on. That combination needs the **dev branch (0.8.0)** — earlier
  releases have no `Performance` submodule, and 0.4.1 has no parallel propagation path at all.
  Install it with `Pkg.add(url="https://github.com/SparqleSim/PauliPropagation.jl", rev="dev")`.

  The caveat to carry into any comparison: `Performance.propagate!` is documented to "yield
  slightly different results" and does — it leaves duplicate Pauli strings unmerged, so it
  reports ~38% more terms than the exact `propagate` path and its expectation value differs
  by ~1e-4. Its term count is a storage count, not an operator size.
- **Give this backend its own thread count**, via `--threads juliapp=N`, rather than the node's
  core count: throughput is not monotone in thread count for every backend, so the fair
  comparison runs each at the count it is fastest at — `--threads monoprop=56 juliapp=28` is
  what the committed ladder used on a 112-core node.
- **ppvm and Qiskit `pauli-prop` propagate on one core**, so this is two threaded libraries
  against two serial ones and a wall-clock ratio against them is not purely algorithmic.
  Measured mid-propagation at 144 qubits, both sit at 0.99 busy cores. Neither is
  misconfigured, and no environment variable changes it: ppvm's rayon-parallel `PauliSum`
  lives in its `config::dashmap` variant, but its Python bindings expose only
  `PauliSumIndexMapFxHash*` (`interface.rs` pins them to `config::indexmap`), so
  `RAYON_NUM_THREADS` sizes a pool that never gets work; Qiskit's `_accelerate` extension
  contains no rayon or `par_iter` at all, and its Python layer's only numpy work is an
  elementwise reduction, so the BLAS knobs have nothing to act on. Divide by the thread count
  for a per-core comparison, or say plainly that only two of the four use the node.
- **Qiskit's `pauli-prop` truncates on a term budget**, not a weight, so it needs one.
  `run_model.py` feeds it monoprop's per-step term count; `run_scaling.py` feeds it
  monoprop's *final* term count for that size (recorded as `max_terms_budget`), which is
  more generous in the early steps.

Building monoprop for the large sizes needs a mode tier at least as big as the qubit
count — 324 qubits means `-Dmonoprop_MAX_NUM_MODES=352`, as tiers come in 32-mode blocks.
