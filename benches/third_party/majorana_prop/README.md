# monoprop benchmarks

Standalone benchmark script comparing [monoprop](https://github.com/Algorithmiq/monoprop) against
MajoranaPropagation.jl on the same workload:
1D Hubbard model, tracking the runtime, expectation value, and operator size at each Trotter step.


The end-to-end workflow is:

1. [Set up the Python environment](#2-set-up-the-python-environment).
2. [Install Julia and `MajoranaPropagation.jl`](#2-install-julia-and-majorana-propagation-jl).
3. [Run the benchmarks](#4-run-the-benchmarks) to produce `results.json`.
4. [Plot the results](#5-plot-the-results) to produce `runtime.png` and `memory.png`.


## SPECS

The benchmark is run on Leonardo (CINECA), in a multi-threaded node with


## 1. Set up the Python environment

```bash
# Navigate to this directory
cd benches/third_party

# Create the venv and install all Python dependencies (including monoprop itself, in editable mode)
uv sync
```

This installs `monoprop`, `qiskit`, `ppvm`, `pauli-prop`, and `cuquantum-python-cu12`. Running the
`cuPauliProp (GPU)` engine requires an NVIDIA GPU with a working CUDA setup; the other three Python
engines (`monoprop`, `QuEra ppvm`, `Qiskit pauli-prop`) run on CPU only.

## 2. Install Julia and `MajoranaPropagation.jl`

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

## 3. Run the benchmarks

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

- `num_terms`: the number of Pauli/Majorana terms kept in the evolving operator, for every step.
- `runtime`: wall-clock time per step, in seconds (excluding the first step, see the note above).
- `memory`: the memory footprint of the evolving operator, in megabytes, for every step. Where an
  engine exposes its own accounting this is exact (monoprop's C++ operator-memory accounting,
  cuPauliProp's cupy device memory pool, `PauliPropagation.jl`'s `Base.summarysize` of the Pauli
  sum); `QuEra ppvm` and `Qiskit pauli-prop` expose no such accounting, so their footprint is
  reconstructed by accumulating this process's host-memory growth across each of their own steps.
- `expvals`: the `ZZ` expectation value on `obs_qubits`, for every step.

## 4. Plot the results

```bash
uv run python plot_results.py
```

This reads `results.json` and produces two log-scale plots for each engine present in the file (so
they reflect whichever engines you actually ran in step 4): `runtime.png` (runtime per Trotter
step) and `memory.png` (operator memory footprint per Trotter step).
