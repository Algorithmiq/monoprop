# monoprop benchmarks

Standalone benchmark scripts for [monoprop](https://github.com/Algorithmiq/monoprop).

This directory is a **self-contained `uv` project**, completely isolated from the main monoprop development environment.
It has its own virtual environment and its own dependencies.


## Setup
```bash
# Navigate to this directory
cd benchmarks

# Create the venv and install all deps
uv sync
```


## Running benchmark scripts
```bash
# Run script in the venv
uv run python my_benchmark_run.py

# Plot final results
uv run python my_benchmark_plot.py
```


## Managing dependencies
```bash
# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Update all deps to latest allowed versions
uv lock --upgrade && uv sync
```
