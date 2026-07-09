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

1. [Choose the simulation settings](#1-choose-the-simulation-settings) in `settings.json`.
2. [Set up the Python environment](#2-set-up-the-python-environment).
3. [(Optional) install Julia and `PauliPropagation.jl`](#3-optional-install-julia-and-pauliPropagationjl).
4. [Run the benchmarks](#4-run-the-benchmarks) to produce `results.json`.
5. [Plot the results](#5-plot-the-results) to produce `runtime.png` and `memory.png`.

## 1. Choose the simulation settings

All simulation parameters live in [`settings.json`](settings.json) and are shared by every engine
(Python and Julia alike), so a single edit compares them all on an identical problem instance:

| Key          | Meaning                                                                              |
| ------------ | ------------------------------------------------------------------------------------- |
| `nx`, `ny`   | Size of the qubit grid (columns × rows); total qubits = `nx * ny`, row-major indexed. |
| `hx`         | Transverse (X) field strength.                                                        |
| `hz`         | Longitudinal (Z) field strength, i.e. the Hamiltonian's tilt.                        |
| `j`          | ZZ coupling strength between nearest-neighbor qubits.                                 |
| `dt`         | Trotter step size.                                                                    |
| `step_min`   | First Trotter step to simulate (inclusive).                                           |
| `step_max`   | Last Trotter step to simulate (inclusive).                                            |
| `lower_atol` | Coefficient magnitude below which Pauli terms are truncated.                          |
| `cutoff`     | Maximum Pauli weight to keep during propagation; `null` disables weight truncation.  |
| `obs_qubits` | The qubit pair (row-major grid index) whose `ZZ` correlator is measured.              |

Edit the values directly in `settings.json` before running the benchmarks — e.g. increase
`nx`/`ny` for a larger system, widen `step_min`/`step_max` to simulate more Trotter steps, or
relax `lower_atol`/`cutoff` to trade accuracy for speed. All engines read this same file, so no
code changes are needed to reproduce a benchmark under different conditions.

`Qiskit pauli-prop` is the one exception: its `propagate_through_circuit` API has no weight-based
cutoff, only a mandatory positive `max_terms` (which also caps its memory pre-allocation, so it
can't be left unbounded). `run_model.py` sets it to monoprop's own term count at each step, so its
per-step term budget tracks `cutoff`/`lower_atol` indirectly through monoprop rather than directly.

## 2. Set up the Python environment

```bash
# Navigate to this directory
cd benches/third_party

# Create the venv and install all Python dependencies (including monoprop itself, in editable mode)
uv sync
```

This installs `monoprop`, `qiskit`, `ppvm`, `pauli-prop`, and `cuquantum-python-cu12`. Running the
`cuPauliProp (GPU)` engine requires an NVIDIA GPU with a working CUDA setup; the other three Python
engines (`monoprop`, `QuEra ppvm`, `Qiskit pauli-prop`) run on CPU only.

## 3. (Optional) install Julia and `PauliPropagation.jl`

Skip this step if you only want to compare the Python engines — `run_model.py` and
`plot_results.py` work fine without it. Install it if you also want the `PauliPropagation.jl`
comparison:

```bash
# Install juliaup and the latest stable Julia release
curl -fsSL https://install.julialang.org | sh
```

Once Julia has been installed, add some utils:

```bash
julia -e 'using Pkg; Pkg.add(["JSON", "ProgressMeter"])'
```

Finally, install the package [`PauliPropagation.jl`](https://github.com/SparqleSim/PauliPropagation.jl) v0.7.3:

```bash
julia -e 'using Pkg; Pkg.add(Pkg.PackageSpec(name="PauliPropagation", version="0.7.3"))'
```

## 4. Run the benchmarks

```bash
# Run the Python engines (monoprop, QuEra ppvm, Qiskit pauli-prop, cuPauliProp) — (re)writes results.json from scratch
uv run python run_model.py

# (Optional) run the Julia engine — merges its results into the existing results.json
julia run_model.jl
```

`run_model.py` must be run first: it (re)creates `results.json` from `settings.json`.
`run_model.jl` then reads that file and adds the `PauliPropagation.jl` entries to it, so
re-running `run_model.py` afterwards overwrites the file and drops them again — rerun
`run_model.jl` too if you do.

For every engine, `results.json` collects, indexed by Trotter step:

- `runtimes`: wall-clock time per step, in seconds (excluding the first step, see the note above).
- `expvals`: the `ZZ` expectation value on `obs_qubits`, for every step.
- `num_terms`: the number of Pauli/Majorana terms kept in the evolving operator, for every step.
- `memory`: the memory footprint of the evolving operator, in megabytes, for every step. Where an
  engine exposes its own accounting this is exact (monoprop's C++ operator-memory accounting,
  cuPauliProp's cupy device memory pool, `PauliPropagation.jl`'s `Base.summarysize` of the Pauli
  sum); `QuEra ppvm` and `Qiskit pauli-prop` expose no such accounting, so their footprint is
  reconstructed by accumulating this process's host-memory growth across each of their own steps.

## 5. Plot the results

```bash
uv run python plot_results.py
```

This reads `results.json` and produces two log-scale plots for each engine present in the file (so
they reflect whichever engines you actually ran in step 4): `runtime.png` (runtime per Trotter
step) and `memory.png` (operator memory footprint per Trotter step).
