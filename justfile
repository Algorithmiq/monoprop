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

# Fumadocs (Next.js) documentation site lives in `docs/`; the static export
# is written to `docs/out`.
site := "docs"

# Run the Python docs toolchain in the synced docs environment (Python 3.12,
# matching the documentation CI image).
docs_uv := "uv run --no-dev --group docs --all-extras --python 3.12"
# `fumapy` (the fumadocs Python docgen) ships inside the npm package; inject it
# ephemerally and pin griffe to the 1.x line it targets (its newer
# griffe-typingdoc dependency otherwise pulls an incompatible griffe).
fumapy := "--with " + site + "/node_modules/fumadocs-python --with 'griffe<2' --with 'griffe-typingdoc==0.2.8'"

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

# Install the documentation site's JavaScript dependencies.
docs-install:
    cd {{ site }} && npm ci

# Each LABEL is one column in results/REPORT.md, so serial / MPI / thread variants
# sit side by side. Set the thread count with the monoprop_NUM_THREADS env var.
# Uses `--no-sync` so a run never rebuilds monoprop with the default (MPI=OFF) and
# clobbers an MPI build; sync deps once first with `uv sync --all-groups
# --all-extras` (or `just bench-build-mpi` for MPI). Examples:
#   just bench serial
#   monoprop_NUM_THREADS=10 just bench serial-t10 --num-modes 64 --bench-rounds 10
# Run the suite (timing + memory) for one LABEL; extra args go to pytest.
bench LABEL *ARGS:
    @mkdir -p "{{bench_results}}"
    label="$1"; shift; \
    monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="{{bench_results}}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{bench_results}}/time-$label.json" "$@"
    uv run --no-sync python benches/report.py "{{bench_results}}"

# Needs an MPI build (`just bench-build-mpi`) -- a non-MPI build is rejected by the
# preflight. Extra args are passed to mpiexec for pinning (and, as root, add
# `--allow-run-as-root`), e.g.
#   monoprop_NUM_THREADS=2 just bench-mpi r5t2 5 --map-by slot:PE=2 --bind-to core
# Run under MPI: RANKS ranks recorded as one LABEL column.
bench-mpi LABEL RANKS *MPIARGS:
    uv run --no-sync python -c "import monoprop, sys; sys.exit(0 if monoprop.has_mpi else 'monoprop was built without MPI; run just bench-build-mpi first')"
    @mkdir -p "{{bench_results}}"
    label="$1"; ranks="$2"; shift 2; \
    monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="{{bench_results}}" \
        uv run --no-sync mpiexec -n "$ranks" \
        -x monoprop_BENCH_LABEL -x monoprop_BENCH_RESULTS "$@" \
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
    monoprop_BENCH_LABEL=smoke monoprop_BENCH_RESULTS="{{bench_results}}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{bench_results}}/time-smoke.json" \
        -m "not slow" --num-generators 8 --num-modes 8 --cutoff 6 --obs-terms 16
    uv run --no-sync python benches/report.py "{{bench_results}}"

# Execute the tutorial notebooks and convert them to Markdown. Notebook
# execution fails the build on any cell error -- this is the notebook doctest.
gen-notebooks:
    {{ docs_uv }} python tools/notebooks_to_mdx.py

# Generate the Python API reference MDX from docstrings (griffe -> JSON -> MDX).
gen-api:
    {{ docs_uv }} {{ fumapy }} fumapy-generate monoprop -d {{ site }}
    cd {{ site }} && node scripts/generate-api.mjs

# Run the runnable docstring examples (the docstring-level doctest check).
doctest-py:
    {{ docs_uv }} python -m pytest --doctest-modules src/monoprop

# Build the static documentation site into `docs/out`.
build-docs: docs-install gen-api doctest-py gen-notebooks
    cd {{ site }} && npm run build

# Serve the documentation locally with hot reloading.
serve-docs: docs-install gen-api gen-notebooks
    cd {{ site }} && npm run dev
