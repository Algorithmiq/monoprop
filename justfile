set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
# Pass recipe arguments through as real argv (preserves quoting, e.g. a
# `--mpiexec-args="--bind-to core"` value with spaces) via "$@" instead of the
# space-splitting `{{ ARGS }}` interpolation.
set positional-arguments

version := `uvx setuptools-scm | tr -d '\n'`
project_source_dir := `pwd | tr -d '\n'`
docs_dir := "build/docs"
html_dir := "build/docs/html"
bench_results := "benches/results"

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

# Each LABEL is one column in results/REPORT.md, so serial / MPI / thread variants
# sit side by side. Set the thread count with the monoprop_NUM_THREADS env var.
# Uses `--no-sync` so a run never rebuilds monoprop with the default (MPI=OFF) and
# clobbers an MPI build; sync deps once first with `uv sync --all-extras --group
# bench` (or `just bench-build-mpi` for MPI). Examples:
#   just bench serial
#   monoprop_NUM_THREADS=10 just bench serial-t10 --num-modes 64 --bench-rounds 10
# Run the suite (timing + memory) for one LABEL; extra args go to pytest.
bench LABEL *ARGS:
    @mkdir -p "{{bench_results}}"
    label="$1"; shift; \
    MONOPROP_BENCH_LABEL="$label" MONOPROP_BENCH_RESULTS="{{bench_results}}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{bench_results}}/time-$label.json" "$@"
    uv run --no-sync python benches/report.py "{{bench_results}}"

# Needs an MPI build (`just bench-build-mpi`) -- a non-MPI build is rejected by the
# preflight. Extra args are passed to mpiexec for pinning, e.g.
#   monoprop_NUM_THREADS=2 just bench-mpi r5t2 5 --map-by slot:PE=2 --bind-to core
# Run under MPI: RANKS ranks recorded as one LABEL column.
bench-mpi LABEL RANKS *MPIARGS:
    uv run --no-sync python benches/_mpi_check.py
    @mkdir -p "{{bench_results}}"
    label="$1"; ranks="$2"; shift 2; \
    MONOPROP_BENCH_LABEL="$label" MONOPROP_BENCH_RESULTS="{{bench_results}}" \
        uv run --no-sync mpiexec --allow-run-as-root -n "$ranks" \
        -x MONOPROP_BENCH_LABEL -x MONOPROP_BENCH_RESULTS "$@" \
        python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{bench_results}}/time-$label.json"
    uv run --no-sync python benches/report.py "{{bench_results}}"

# A plain `just bench` uses `--no-sync`, so this MPI build survives until the next
# explicit `uv sync` / rebuild.
# Rebuild monoprop with MPI enabled (editable). Run once before `just bench-mpi`.
bench-build-mpi:
    uv sync --all-extras --group bench --reinstall-package monoprop --no-cache \
        --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" -v

# Quick sanity run: tiny sizes, skip the slow static benchmarks.
bench-smoke:
    @mkdir -p "{{bench_results}}"
    MONOPROP_BENCH_LABEL=smoke MONOPROP_BENCH_RESULTS="{{bench_results}}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{bench_results}}/time-smoke.json" \
        -m "not slow" --num-generators 8 --num-modes 8 --cutoff 6 --obs-terms 16
    uv run --no-sync python benches/report.py "{{bench_results}}"
