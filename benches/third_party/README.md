# monoprop benchmarks

Standalone benchmark scripts for [monoprop](https://github.com/Algorithmiq/monoprop).

This directory is a self-contained `uv` project, completely isolated from the main monoprop development environment.
It has its own virtual environment and its own external dependencies.


### Setup Python environment
```bash
# Navigate to this directory
cd benches/third_party

# Create venv and install all deps
uv sync
```

### Install Julia package
Some benchmarks compare against a Julia implementation of Pauli Propagation:
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


### Run benchmarks and plot results
```bash
# Run Python script in venv
uv run python trotter_ising_run.py

# Run Julia script
julia trotter_ising_run.jl

# Plot final results
uv run python trotter_ising_plot.py
```
