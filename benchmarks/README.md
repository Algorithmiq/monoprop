# monoprop benchmarks

Standalone benchmark scripts for [monoprop](https://github.com/Algorithmiq/monoprop).

This directory is a **self-contained `uv` project**, completely isolated from the main monoprop development environment. It has its own `uv.lock`, its own virtual environment, and its own dependencies.


## Setup

```bash
# Navigate to this directory
cd benchmarks

# Create the venv and install all dependencies
uv sync
```

## Running benchmark scripts

```bash
# Run a script inside the isolated environment
uv run python my_benchmark.py

# Or activate the venv and run directly
source .venv/bin/activate
python my_benchmark.py
```

## Managing dependencies

```bash
# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Update all dependencies to their latest allowed versions
uv lock --upgrade && uv sync
```
