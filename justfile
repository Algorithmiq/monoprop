set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

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
