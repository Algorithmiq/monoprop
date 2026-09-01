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

docs_uv := "uv run --group docs --group test --no-dev --all-extras --no-extra mpi"
uv_sync := "uv sync --no-progress --all-extras -v"
no_mpi_extra := "--no-extra mpi"
mpi_enabled := lowercase(env('monoprop_ENABLE_MPI', 'off'))
mpi_extra := if mpi_enabled =~ '^(on|true|yes|1)$' { "" } else { no_mpi_extra }

# scikit-build names the editable tree after the build type, so the sanitizer and
# coverage configurations are found through the same environment that built them.

build_dir := "build/editable" / env('SKBUILD_CMAKE_BUILD_TYPE', 'Release')
mpiexec := "mpiexec --map-by :OVERSUBSCRIBE"

# The rank matrix asks for more ranks than a runner has cores. OpenMPI 4 (ORTE) and 5
# (PRRTE) spell oversubscription differently, and neither errors on the other's variable.

export OMPI_MCA_rmaps_base_oversubscribe := "1"
export PRTE_MCA_rmaps_default_mapping_policy := ":oversubscribe"

default: build-docs

build *ARGS:
    {{ uv_sync }} {{ mpi_extra }} "$@"

# Report what the installed extension was actually built as.

info:
    uv run --no-sync python -c 'import pprint, monoprop as mp; pprint.pprint({"version": mp.__version__, "variant": mp.__variant__, "compiler_flags": mp.__compiler_flags__, "MPI": mp.has_mpi})'

test: build test-py test-cpp

# The test legs below run against whatever is installed (--no-sync, so that a run cannot
# silently rebuild a differently configured wheel). LABEL, when given, names the JUnit
# report CI uploads.

test-py LABEL='':
    uv run --no-sync pytest -r aR --durations=50 --durations-min=5.0 \
        {{ if LABEL == '' { '' } else { '--junit-xml=pytest-' + LABEL + '.xml -o junit_family=legacy' } }}

test-cpp LABEL='':
    ctest --test-dir {{ build_dir }} --output-on-failure --no-tests=error --label-exclude mpi \
        {{ if LABEL == '' { '' } else { '--output-junit ctest-serial-' + LABEL + '.xml' } }}

# --no-tests=error: the MPI tests are registered only by an MPI-enabled build, so an
# empty set here means the build was misconfigured, not that there is nothing to run.

test-cpp-mpi LABEL='':
    ctest --test-dir {{ build_dir }} --output-on-failure --no-tests=error --label-regex mpi \
        {{ if LABEL == '' { '' } else { '--output-junit ctest-mpi-' + LABEL + '.xml' } }}

# RANKS is a single integer or a semicolon-separated list (e.g. "1;2;4"); the suite runs
# once per entry. Extra arguments go to pytest, e.g. `just test-py-mpi "1;2;4" -m mpi`.

test-py-mpi RANKS='' *PYTEST_ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    shift 1
    requested_ranks={{ quote(RANKS) }}
    ranks="${requested_ranks:-${monoprop_MPI_TEST_PROCS:-2}}"
    for r in ${ranks//;/ }; do
      echo "Running the Python test suite with ${r} MPI rank(s)"
      {{ mpiexec }} -n "$r" uv run --no-sync pytest tests --with-mpi -v "$@"
    done

# Build MPI-enabled (source builds are serial by default), then run every leg. The C++
# rank matrix is a build-time setting, so it is baked in here rather than passed to ctest.

test-mpi RANKS='': && (test-py-mpi RANKS) test-cpp test-cpp-mpi
    requested_ranks={{ quote(RANKS) }}; \
    ranks="${requested_ranks:-${monoprop_MPI_TEST_PROCS:-2}}"; \
    monoprop_ENABLE_MPI=ON {{ uv_sync }} --group workspace-test \
        --reinstall-package monoprop --no-cache \
        --config-settings-package="monoprop:cmake.define.monoprop_MPI_TEST_PROCS=${ranks}"

# Build and run a consumer project against the installed package, the way a downstream
# user does.

test-find-package BUILD_DIR='build/find-package-smoke':
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir={{ quote(BUILD_DIR) }}
    site_packages="$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    cmake -S cpp/tests/find_package_smoke -B "$build_dir" \
      -Dmonoprop_DIR="$site_packages/monoprop/cmake"
    cmake --build "$build_dir"
    "$build_dir/smoke"

# The sanitizer legs run against a tree built with SKBUILD_CMAKE_BUILD_TYPE=AsanUbsan (or
# Tsan) and the matching monoprop_SANITIZER define; build_dir follows that environment.
#
# Only the C++ binary is fully instrumented, so the option sets differ per leg and cannot
# be hoisted to the environment.

ubsan_options := "halt_on_error=1:print_stacktrace=1"
sanitizer_log := project_source_dir / "sanitizer-log"

# The instrumented binary is the only leg that can check for leaks.

test-cpp-asan:
    ASAN_OPTIONS="detect_leaks=1:leak_check_at_exit=1:detect_stack_use_after_return=1:detect_invalid_pointer_pairs=1:check_initialization_order=1:strict_init_order=1:strict_string_checks=1:halt_on_error=1" \
    LSAN_OPTIONS="suppressions={{ project_source_dir }}/.github/lsan.supp" \
    UBSAN_OPTIONS="{{ ubsan_options }}" \
      ctest --test-dir {{ build_dir }} --output-on-failure

# CPython is uninstrumented, so the leak, pointer-pair and initialization checks are off
# here; the C++ leg covers those. ASan needs libstdc++ preloaded too, or its __cxa_throw
# interceptor does not resolve. pytest replaces stderr, so the reports go to log files.

test-py-asan:
    LD_PRELOAD="$(g++ -print-file-name=libasan.so):$(g++ -print-file-name=libstdc++.so.6)" \
    ASAN_OPTIONS="detect_leaks=0:detect_stack_use_after_return=1:halt_on_error=1:log_path={{ sanitizer_log }}" \
    UBSAN_OPTIONS="{{ ubsan_options }}:log_path={{ sanitizer_log }}" \
      uv run --no-sync pytest -r aR --durations=50 --durations-min=5.0

# Print what test-py-asan sent to the log files. Silent when the tests themselves failed.

sanitizer-reports:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    files=({{ sanitizer_log }}.*)
    if (( ${#files[@]} == 0 )); then
      echo "No sanitizer report was written; the failure came from the tests themselves."
      exit 0
    fi
    for f in "${files[@]}"; do
      echo "::group::$(basename "$f")"
      cat "$f"
      echo "::endgroup::"
    done

# TSan cannot load an instrumented _core into stock CPython, so this leg is C++ only, and
# restricted to the concurrent partition and shared-memory paths.

test-cpp-tsan:
    TSAN_OPTIONS="halt_on_error=1:history_size=4" \
      ctest --test-dir {{ build_dir }} --output-on-failure -R "(partition_|shm_comm_)"

# Collect one instrumented build. MPI must be "on" or "off"; each variant needs its own build
# and output directories because the preprocessor selects different compatibility paths.

code-coverage-collect MPI BUILD_DIR OUTPUT_DIR:
    #!/usr/bin/env bash
    set -euo pipefail

    mpi={{ quote(MPI) }}
    build_dir={{ quote(BUILD_DIR) }}
    output_dir={{ quote(OUTPUT_DIR) }}
    export GCOV_EXIT_AT_ERROR=1
    if [[ "$mpi" != "on" && "$mpi" != "off" ]]; then
      echo "MPI must be 'on' or 'off', got: $mpi" >&2
      exit 2
    fi

    sync_args=({{ uv_sync }})
    if [[ "$mpi" == "off" ]]; then
      sync_args+=({{ no_mpi_extra }})
    fi
    sync_args+=(--group workspace-test --reinstall-package monoprop)
    # The top-level recipe isolates local rebuilds from a wheel cached for the other variant.
    if [[ "${monoprop_COVERAGE_NO_CACHE:-OFF}" == "ON" ]]; then
      sync_args+=(--no-cache)
    fi

    SKBUILD_BUILD_DIR="$build_dir" \
      SKBUILD_CMAKE_BUILD_TYPE=Coverage \
      monoprop_ENABLE_MPI="$mpi" \
      "${sync_args[@]}"

    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    find "$build_dir" -type f -name '*.gcda' -delete
    uv run --no-sync coverage erase

    uv run --no-sync coverage run --parallel-mode -m pytest -m "not mpi"
    if [[ "$mpi" == "on" ]]; then
      # The coverage lane runs in a container as root.
      export OMPI_ALLOW_RUN_AS_ROOT=1
      export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
      {{ mpiexec }} -n 2 \
        uv run --no-sync coverage run --parallel-mode \
          -m pytest tests --with-mpi -m mpi
    fi

    gcovr_args=(
      --gcov-executable gcov
      --gcov-ignore-parse-errors
      --exclude-throw-branches
      --exclude-unreachable-branches
      --filter '^(cpp|src)/'
      --exclude '^(cpp/)?tests/'
      --merge-lines
      "$build_dir"
    )
    GCOV_EXIT_AT_ERROR=1 uvx gcovr "${gcovr_args[@]}" \
      --json "$output_dir/cpp-through-python.json" \
      --json-pretty

    ctest --test-dir "$build_dir" \
      --output-on-failure \
      --no-tests=error \
      --label-exclude mpi
    if [[ "$mpi" == "on" ]]; then
      ctest --test-dir "$build_dir" \
        --output-on-failure \
        --no-tests=error \
        --label-regex mpi
    fi

    GCOV_EXIT_AT_ERROR=1 uvx gcovr "${gcovr_args[@]}" \
      --json "$output_dir/cpp-total.json" \
      --json-pretty

    if [[ "$mpi" == "on" ]]; then
      uv run --no-sync python - "$output_dir/cpp-total.json" <<'PY'
    import json
    import sys
    from pathlib import Path

    report = json.loads(Path(sys.argv[1]).read_text())
    covered = []
    for entry in report["files"]:
        path = entry["file"].replace("\\", "/")
        if "/detail/mpi/" not in f"/{path}":
            continue
        if any(line.get("count", 0) > 0 for line in entry.get("lines", [])):
            covered.append(path)

    if not any(path.endswith("/MPICompat.cpp") for path in covered):
        raise SystemExit("MPICompat.cpp has no covered lines")
    if len(covered) < 2:
        raise SystemExit(f"expected coverage in at least two MPI sources, found: {covered}")
    print("Covered MPI sources:", *covered, sep="\n  ")
    PY
    fi

    shopt -s nullglob
    shards=(.coverage.*)
    if (( ${#shards[@]} == 0 )); then
      echo "No parallel Python coverage files were produced" >&2
      exit 1
    fi
    mv "${shards[@]}" "$output_dir/"

# Combine the raw data from serial and MPI collection runs into the stable report names consumed by
# Codecov, SonarQube, and the local HTML recipe.

code-coverage-aggregate SERIAL_DIR MPI_DIR OUTPUT_DIR='.':
    #!/usr/bin/env bash
    set -euo pipefail

    serial_dir={{ quote(SERIAL_DIR) }}
    mpi_dir={{ quote(MPI_DIR) }}
    output_dir={{ quote(OUTPUT_DIR) }}
    mkdir -p "$output_dir"
    rm -f "$output_dir/.coverage" \
      "$output_dir/python-coverage.xml" \
      "$output_dir/python-coverage.info" \
      "$output_dir/cpp-coverage-through-python-bindings.xml" \
      "$output_dir/cpp-coverage-through-python-bindings-sonar.xml" \
      "$output_dir/cpp-coverage-through-python-bindings.info" \
      "$output_dir/cpp-coverage.xml" \
      "$output_dir/cpp-coverage-sonar.xml" \
      "$output_dir/cpp-coverage.info"

    uvx --from coverage coverage combine \
      --data-file "$output_dir/.coverage" \
      "$serial_dir" "$mpi_dir"
    uvx --from coverage coverage xml \
      --data-file "$output_dir/.coverage" \
      -o "$output_dir/python-coverage.xml"
    uvx --from coverage coverage lcov \
      --data-file "$output_dir/.coverage" \
      -o "$output_dir/python-coverage.info"
    # coverage.py emits repo-relative SF: paths while gcovr emits absolute ones.
    sed -i 's|^SF:\([^/]\)|SF:{{ project_source_dir }}/\1|' \
      "$output_dir/python-coverage.info"

    uvx gcovr \
      --add-tracefile "$serial_dir/cpp-through-python.json" \
      --add-tracefile "$mpi_dir/cpp-through-python.json" \
      --cobertura "$output_dir/cpp-coverage-through-python-bindings.xml" \
      --sonarqube "$output_dir/cpp-coverage-through-python-bindings-sonar.xml" \
      --lcov "$output_dir/cpp-coverage-through-python-bindings.info"

    uvx gcovr \
      --add-tracefile "$serial_dir/cpp-total.json" \
      --add-tracefile "$mpi_dir/cpp-total.json" \
      --cobertura "$output_dir/cpp-coverage.xml" \
      --sonarqube "$output_dir/cpp-coverage-sonar.xml" \
      --lcov "$output_dir/cpp-coverage.info"

# Render combined reports locally. lcov and genhtml are intentionally unnecessary in CI.

code-coverage-html REPORT_DIR='.' OUTPUT_DIR='monoprop-coverage':
    lcov \
      --ignore-errors inconsistent,corrupt \
      -a {{ quote(REPORT_DIR) }}/python-coverage.info \
      -a {{ quote(REPORT_DIR) }}/cpp-coverage-through-python-bindings.info \
      -a {{ quote(REPORT_DIR) }}/cpp-coverage.info \
      -o {{ quote(REPORT_DIR) }}/merged.info
    genhtml {{ quote(REPORT_DIR) }}/merged.info -o {{ quote(OUTPUT_DIR) }} \
      --legend --title "monoprop coverage" \
      --prefix {{ project_source_dir }} \
      --ignore-errors inconsistent \
      --verbose

# Build both compile-time variants, combine their reports, and render the local HTML report.

code-coverage:
    monoprop_COVERAGE_NO_CACHE=ON just code-coverage-collect off build/coverage/mpi-off build/coverage-data/mpi-off
    monoprop_COVERAGE_NO_CACHE=ON just code-coverage-collect on build/coverage/mpi-on build/coverage-data/mpi-on
    just code-coverage-aggregate build/coverage-data/mpi-off build/coverage-data/mpi-on .
    just code-coverage-html . monoprop-coverage

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
    monoprop_ENABLE_MPI=ON {{ uv_sync }} --group bench --reinstall-package monoprop --no-cache

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
