set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
# Pass recipe arguments through as real argv (preserves quoting, e.g. a
# `--mpiexec-args="--bind-to core"` value with spaces) via "$@" instead of the
# space-splitting `{{ ARGS }}` interpolation.
set positional-arguments

version := `uvx setuptools-scm | tr -d '\n'`
project_source_dir := `pwd | tr -d '\n'`
docs_dir := "build/docs"
html_dir := "build/docs/html"

default: build-docs

test-py:
    uv run python -m pytest -m "not mpi"

# MPI is off by default in source builds, so build an MPI-enabled editable install
# first, then run the suite under mpiexec with --no-sync (avoids a per-rank resync).
test-py-mpi:
    uv sync --all-extras --group test --reinstall-package monoprop --no-cache --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" -v; \
    ranks="${monoprop_MPI_TEST_PROCS:-2}"; \
    for r in ${ranks//;/ }; \
    do echo "Running full Python test suite with ${r} MPI rank(s)"; \
    mpiexec --allow-run-as-root -n "$r" uv run --no-sync python -m pytest tests --with-mpi -v; \
    done

test-py-mpi-matrix:
    uv sync --all-extras --group test --reinstall-package monoprop --no-cache --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" -v; \
    ranks="${monoprop_MPI_TEST_PROCS:-1;2;4}"; for r in ${ranks//;/ }; do echo "Running MPI-marked Python tests with ${r} rank(s)"; mpiexec --allow-run-as-root -n "$r" uv run --no-sync python -m pytest tests --with-mpi -m mpi -v; done

build-docs:
    mkdir -p {{ docs_dir }}
    uv run --no-dev --group docs --all-extras --python 3.12 \
        sphinx-build -b html --define version="{{ version }}" docs {{ html_dir }}

doctest-docs:
    mkdir -p {{ docs_dir }}
    uv run --no-dev --group docs --all-extras --python 3.12 \
        tools/run_sphinx_build.py --reports-dir {{ html_dir }}/reports -- \
        sphinx-build -b doctest -t notebook_test \
            --define version="{{ version }}" docs {{ docs_dir }}/doctest

serve-docs:
    uv run --no-dev --group docs --all-extras --python 3.12 \
        sphinx-autobuild -b html -D version="{{ version }}" docs {{ html_dir }}

# Run the benchmark suite (timing + memory) and write benches/results/REPORT.md.
# Serial by default; pass `--ranks N` for MPI (needs `just bench-build-mpi` first).
# Uses `--no-sync` so a plain run does NOT rebuild monoprop with the default
# (MPI=OFF) settings and clobber an MPI build. Sync deps once first with
# `uv sync --all-groups --all-extras` (or `just bench-build-mpi` for MPI). After
# editing monoprop sources, rebuild explicitly before benching.
# Any other arguments are forwarded to pytest, e.g.
#   just bench --ranks 4 --num-modes 64 --bench-rounds 10
bench *ARGS:
    uv run --no-sync python benches/run.py "$@"

# Rebuild monoprop with MPI enabled (editable). Run once before `just bench --ranks N`.
# A plain `just bench` uses `--no-sync`, so this MPI build survives until the next
# explicit `uv sync` / rebuild.
bench-build-mpi:
    uv sync --all-extras --group bench --reinstall-package monoprop --no-cache \
        --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" -v

# Quick sanity run: tiny sizes, skip the slow static benchmarks.
bench-smoke:
    uv run --no-sync python benches/run.py \
        -m "not slow" --num-generators 8 --num-modes 8 --cutoff 6 --obs-terms 16
