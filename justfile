set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Pass recipe arguments through as real argv (preserves quoting, e.g. a
# `--mpiexec-args="--bind-to core"` value with spaces) via "$@" instead of the
# space-splitting `{{ ARGS }}` interpolation.

set positional-arguments := true

version := `uvx setuptools-scm | tr -d '\n'`
project_source_dir := `pwd | tr -d '\n'`
docs_dir := "build/docs"
html_dir := "build/docs/html"
bench_results := "benches/results"

# Fumadocs (Next.js) documentation site lives in `docs/`; the static export
# is written to `docs/out`.

site := "docs"

# Run the Python docs toolchain in the synced docs environment.

docs_uv := "uv run --group docs --group test --no-dev --all-extras"

default: build-docs

test:
    uv run python -m pytest -m "not mpi"
    ctest --test-dir build/editable/Release --output-on-failure

# Output directory for `capture-baseline` / `diff-baseline` (gitignored).
baseline_dir := ".baseline-capture"

# Capture a golden baseline snapshot into `.baseline-capture/LABEL` (default "golden") via
# tools/capture-baseline.py -- the regression instrument for the NumModes-NTTP removal refactor
# (see the plan's Stage 0). Run once on an unmodified tree to seed the golden baseline, then
# `just diff-baseline` after every later-stage change.
capture-baseline LABEL='golden':
    uv run --no-sync python tools/capture-baseline.py --out "{{ baseline_dir }}/{{ LABEL }}"

# Rebuild monoprop and diff a fresh capture against a stored one (default "golden"). Byte-identical
# is the bar: term order is a regression signal, not an implementation detail.
diff-baseline AGAINST='golden':
    uv sync --all-extras --group test --reinstall-package monoprop --no-cache -v
    rm -rf "{{ baseline_dir }}/candidate"
    just capture-baseline candidate
    diff -rq "{{ baseline_dir }}/{{ AGAINST }}" "{{ baseline_dir }}/candidate"

# Capture with the support-form row backend forced (monoprop_ROW_STORE=sparse) and check it against a
# stored capture as term *sets* plus a relative tolerance. Not a byte diff: the sparse rows hash
# differently, so they accumulate in a different order on purpose. This is how the two backends are
# held equivalent on the fixtures, every one of which is below the automatic crossover.
diff-baseline-sparse AGAINST='golden' TOL='1e-10':
    rm -rf "{{ baseline_dir }}/sparse"
    monoprop_ROW_STORE=sparse uv run --no-sync python tools/capture-baseline.py --out "{{ baseline_dir }}/sparse"
    uv run --no-sync python tools/capture-baseline.py --compare "{{ baseline_dir }}/{{ AGAINST }}" "{{ baseline_dir }}/sparse" --tol "{{ TOL }}"

# The Python suite with the support-form row backend forced, the counterpart of ctest's
# `-L sparse-rows` variants.
test-sparse-rows:
    monoprop_ROW_STORE=sparse uv run --no-sync python -m pytest -m "not mpi"

# MPI is off by default in source builds, so build an MPI-enabled editable install
# first, then run the suite under mpiexec with --no-sync (avoids a per-rank resync).
# Pass RANKS as either a single integer or a semicolon-separated list (e.g. "1;2;4").

test-mpi RANKS='':
    uv sync --all-extras --group workspace-test --reinstall-package monoprop --no-cache --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" -v
    export OMPI_MCA_rmaps_base_oversubscribe="1"; \
    ranks="${1:-${monoprop_MPI_TEST_PROCS:-2}}"; \
    for r in ${ranks//;/ }; \
    do echo "Running full Python test suite with ${r} MPI rank(s)"; \
    mpiexec -n "$r" uv run --no-sync python -m pytest tests --with-mpi -v; \
    echo "Running C++ unit tests with ${r} MPI rank(s)"; \
    ctest --test-dir build/editable/Release --output-on-failure; \
    done

# Build the fat binary: the engine compiled once per x86-64 ISA tier, with one selected when
# monoprop is imported. This is what published wheels are, and it is *not* what a source build wants
# -- -march=native beats every tier -- so it is a separate recipe rather than the default. Note
# `uv run --no-sync` for everything afterwards: a plain `uv run` re-syncs without the config setting
# and silently replaces the fat build with a single-ISA one.

build-fat:
    uv sync --all-extras --group test --reinstall-package monoprop --no-cache -v --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_FAT_BINARY=ON"
    uv run --no-sync python -c 'import monoprop; print("loaded", monoprop.__variant__, "of", monoprop.available_variants())'

# The narrow-seam experiment: one tiered translation unit in one module instead of a module per tier.
# Builds, passes everything and is 61% smaller -- and does not deliver a tier, because the linker
# deduplicates the four copies of every template instantiation down to one. `check-tier-symbols` is
# the gate that says so; it fails on purpose. Do not ship this.

build-fat-narrow-seam:
    uv sync --all-extras --group test --reinstall-package monoprop --no-cache -v --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_FAT_BINARY=ON" --config-settings-package="monoprop:cmake.define.monoprop_FAT_BINARY_MODE=narrow-seam"
    uv run --no-sync python -c 'import monoprop; print("loaded", monoprop.__variant__, "of", monoprop.available_variants())'

# Whether a narrow-seam build kept its ISA tiers past the linker. Nothing else notices: the tiers agree
# on every answer, so the whole suite passes while they run identical code.

check-tier-symbols BUILD_DIR='build/editable/Release':
    uv run --no-sync python tools/check-tier-symbols.py "{{ BUILD_DIR }}"

# Run the Python suite once per installed ISA variant, not just the one this CPU selects. Without
# this the lower tiers ship untested on every developer machine and every CI runner, since the
# dispatch always picks the best one available.

test-variants:
    variants=$(uv run --no-sync python -c 'import monoprop; print(" ".join(monoprop.available_variants()))'); \
    if [ -z "$variants" ]; then echo "not a fat binary; run 'just build-fat' first" >&2; exit 1; fi; \
    for v in $variants; do \
    echo "=== monoprop_VARIANT=$v"; \
    monoprop_VARIANT="$v" uv run --no-sync python -m pytest -m "not mpi" -q; \
    done

# Capture a baseline per ISA variant and diff them byte-wise against each other. The bar is
# byte-identical: the tiers exist to change instruction selection, not answers, which is what
# -ffp-contract=off buys and what this recipe is the gate on.

diff-baseline-variants:
    variants=$(uv run --no-sync python -c 'import monoprop; print(" ".join(monoprop.available_variants()))'); \
    if [ -z "$variants" ]; then echo "not a fat binary; run 'just build-fat' first" >&2; exit 1; fi; \
    rm -rf "{{ baseline_dir }}/variants"; \
    for v in $variants; do \
    monoprop_VARIANT="$v" uv run --no-sync python tools/capture-baseline.py --out "{{ baseline_dir }}/variants/$v"; \
    done; \
    reference=$(echo $variants | cut -d' ' -f1); \
    for v in $variants; do \
    echo "=== $v vs $reference"; \
    diff -rq "{{ baseline_dir }}/variants/$reference" "{{ baseline_dir }}/variants/$v"; \
    done

# Build and run the C++ suite with a 64-bit TermIndex (monoprop_WIDE_TERM_INDEX=ON).
# This is the only configuration that compiles the wide `#if defined(monoprop_WIDE_TERM_INDEX)`
# branches (operator_index_tests, large_cosine_storage_tests, graph_encoding_tests), so it
# guards them from bit-rotting. Serial is enough to exercise those branches.

test-wide:
    uv sync --all-extras --group workspace-test --reinstall-package monoprop --no-cache --config-settings-package="monoprop:cmake.define.monoprop_WIDE_TERM_INDEX=ON"
    uv run --no-sync python -m pytest -m "not mpi"
    ctest --test-dir build/editable/Release --output-on-failure

# Report code coverage.

code-coverage:
    uv sync --no-progress --group workspace-test --all-extras --reinstall-package monoprop --no-cache --config-settings-package="monoprop:cmake.build-type=Coverage" -v

    # run python tests
    uv run --no-sync python -m pytest --cov=src/monoprop --cov-report=lcov:python-coverage.info

    # coverage.py emits repo-relative SF: paths while gcovr emits absolute ones;
    # genhtml's common-prefix stripping would abort with "duplicate merge record" without this step
    sed -i 's|^SF:\([^/]\)|SF:{{ project_source_dir }}/\1|' python-coverage.info

    # collect coverage (C++ through Python bindings)
    uvx gcovr \
      --gcov-executable gcov \
      --gcov-ignore-parse-errors \
      --exclude-throw-branches \
      --exclude-unreachable-branches \
      --filter '^(src|include)/' \
      --exclude '^tests/' \
      --merge-lines \
      "build/editable/Coverage" \
      --lcov cpp-coverage-through-python-bindings.info

    # run C++ unit tests
    ctest --test-dir build/editable/Coverage --output-on-failure

    # collect coverage (C++ unit tests)
    uvx gcovr \
      --gcov-executable gcov \
      --gcov-ignore-parse-errors \
      --exclude-throw-branches \
      --exclude-unreachable-branches \
      --filter '^(src|include)/' \
      --exclude '^tests/' \
      --merge-lines \
      "build/editable/Coverage" \
      --lcov cpp-coverage.info

    # merge
    lcov \
      --ignore-errors inconsistent,corrupt \
      -a python-coverage.info \
      -a cpp-coverage-through-python-bindings.info \
      -a cpp-coverage.info \
      -o merged.info

    # output to HTML
    genhtml merged.info -o monoprop-coverage --legend --title "monoprop coverage" \
      --prefix {{ project_source_dir }} \
      --ignore-errors inconsistent \
      --verbose

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
    @mkdir -p "{{ bench_results }}"
    label="$1"; shift; \
    monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="{{ bench_results }}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{ bench_results }}/time-$label.json" "$@"
    uv run --no-sync monoprop-bench-report "{{ bench_results }}"

# Needs an MPI build (`just bench-build-mpi`) -- a non-MPI build is rejected by the
# preflight. Extra args are passed to mpiexec for pinning (and, as root, add
# `--allow-run-as-root`), e.g.
#   monoprop_NUM_THREADS=2 just bench-mpi r5t2 5 --map-by slot:PE=2 --bind-to core

# Run under MPI: RANKS ranks recorded as one LABEL column.
bench-mpi LABEL RANKS *MPIARGS:
    uv run --no-sync python -c "import monoprop, sys; sys.exit(0 if monoprop.has_mpi else 'monoprop was built without MPI; run just bench-build-mpi first')"
    @mkdir -p "{{ bench_results }}"
    label="$1"; ranks="$2"; shift 2; \
    monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="{{ bench_results }}" \
        uv run --no-sync mpiexec -n "$ranks" \
        -x monoprop_BENCH_LABEL -x monoprop_BENCH_RESULTS "$@" \
        python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{ bench_results }}/time-$label.json"
    uv run --no-sync monoprop-bench-report "{{ bench_results }}"

# A plain `just bench` uses `--no-sync`, so this MPI build survives until the next
# explicit `uv sync` / rebuild.

# Rebuild monoprop with MPI enabled (editable). Run once before `just bench-mpi`.
bench-build-mpi:
    uv sync --all-extras --group bench --reinstall-package monoprop --no-cache \
        --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" -v

# Quick sanity run: tiny sizes, skip the slow static benchmarks.
bench-smoke:
    @mkdir -p "{{ bench_results }}"
    monoprop_BENCH_LABEL=smoke monoprop_BENCH_RESULTS="{{ bench_results }}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{ bench_results }}/time-smoke.json" \
        -m "not slow" --num-generators 8 --num-modes 8 --cutoff 6 --obs-terms 16
    uv run --no-sync monoprop-bench-report "{{ bench_results }}"

# Deliberately keeps the `bench` default sizes rather than inventing a second set
# to keep in sync: they run in well under a minute yet leave the heavier
# Schrödinger operations in the 50 ms - 1 s range, where runner noise does not
# swamp the signal. Only the slow fixed models are dropped; they run nightly.
# More rounds than a local run, because CI reports the mean.
#
# The marker expression and round count come from the environment so a caller can
# pass values containing spaces, or an empty marker to select everything:
#   monoprop_BENCH_MARKERS= monoprop_BENCH_ROUNDS=3 just bench-ci ci-bare-metal

# Run the continuous-benchmarking profile tracked by Bencher.
bench-ci LABEL *ARGS:
    @mkdir -p "{{ bench_results }}"
    label="$1"; shift; \
    monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="{{ bench_results }}" \
        uv run --no-sync python -m pytest benches -o filterwarnings=default \
        --benchmark-json="{{ bench_results }}/time-$label.json" \
        -m "${monoprop_BENCH_MARKERS-not slow}" \
        --bench-rounds "${monoprop_BENCH_ROUNDS:-5}" "$@"

# Convert one LABEL's artifacts into Bencher Metric Format JSON on stdout, e.g.
#   just bench-ci ci-linux && just bench-bmf ci-linux > bmf.json

# Emit LABEL's results as Bencher Metric Format JSON.
bench-bmf LABEL:
    uv run --no-sync monoprop-bench-bmf "{{ bench_results }}" "$1"

# Execute the tutorial notebooks and convert them to Markdown. Notebook

# execution fails the build on any cell error -- this is the notebook doctest.
gen-notebooks:
    {{ docs_uv }} python docs/scripts/notebooks_to_mdx.py

# Generate the Python API reference MDX from docstrings (griffe -> JSON -> MDX).
gen-api:
    {{ docs_uv }} python {{ site }}/scripts/gen_api_dump.py monoprop -d {{ site }}
    cd {{ site }} && node scripts/generate-api.mjs

# Run the runnable docstring examples (the docstring-level doctest check).
doctest-py:
    {{ docs_uv }} python -m pytest --doctest-modules src/monoprop

doctest-docs:
    {{ docs_uv }} python -m pytest --markdown-docs docs/content/docs --ignore=docs/content/docs/tutorials --ignore=docs/content/docs/api
    {{ docs_uv }} python -m pytest --markdown-docs README.md

# Build the static documentation site into `docs/out`.
build-docs: docs-install gen-api doctest-py doctest-docs gen-notebooks
    cd {{ site }} && npm run build

# Check exported HTML links (including external URLs).
check-doc-links:
    lychee --config .lychee.postbuild.toml --root-dir "{{ project_source_dir }}/docs/out" --fallback-extensions html --index-files index.html 'docs/out/**/*.html'

# Serve the documentation locally with hot reloading.
serve-docs:
    cd {{ site }} && npm run dev
