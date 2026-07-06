# monoprop benchmarks

Standalone benchmark scripts for [monoprop](https://github.com/Algorithmiq/monoprop).

This directory is a self-contained `uv` project, completely isolated from the main monoprop development environment.
It has its own virtual environment and its own external dependencies.


### Setup Python
```bash
# Navigate to this directory
cd benchmarks

# Create venv and install all deps
uv sync
```

### Install Julia
Some benchmarks compare against a Julia implementation of the Pauli Propagation algorithm,
so a Julia toolchain is also required. Install it via [`juliaup`](https://github.com/JuliaLang/juliaup),
the official Julia version manager:

```bash
# Install juliaup and the latest stable Julia release
curl -fsSL https://install.julialang.org | sh

# Reload your shell and verify the install
julia --version
```

Then install [`PauliPropagation.jl`](https://github.com/SparqleSim/PauliPropagation.jl) and some utils
from the official Julia package manager:

```bash
julia -e 'using Pkg; Pkg.add(["PauliPropagation", "JSON", "ProgressMeter"])'
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
