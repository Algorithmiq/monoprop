# monoprop agent guide

A C++23/Python library for Majorana and Pauli propagation. `MonomialPropagator<NumModes>` is the
shared engine; Majorana against Pauli is a runtime `Basis`, and Python exposes `MajoranaPropagator` and
`PauliPropagator`. `monoprop_MAX_NUM_MODES` is the compile-time mode limit (default 1024).

## Rules

- C++23 idioms, trailing return types, follow almost-always-`auto` style, `clang-format` with the
  repository configuration.
- Use snake case for naming variables and functions, camel case for classes. E.g., `kBits` is wrong for a integer variable, use `num_bits` instead.
- Only private member variables are suffixed with an underscore.
- Qt-style Doxygen on every header declaration, member docs after the member. `//` for one-line
  C++ comments, `/* */` for blocks.
- Comment invariants, contracts and non-obvious choices only. Never narrate the code.
- Google-style Python docstrings without type hints, accurate enough for `just gen-api`. Type hints in the code.
- In `docs/content/docs/**/*.mdx` and docstrings, link API symbols as `[Symbol][]` or
  `[Display][fully.qualified.path]` -- never `/api/...` URLs, never backticks around a link.
  See `docs/content/docs/documenting.mdx`.
- **Run `prek run --all-files` and the relevant tests before pushing, and fix every finding.**
  `lint.yml` runs the same hooks over the PR's commit range, so a skipped lint is a red PR.
- Changes to the API or user-facing behavior (including developers, e.g. test workflows) should be reflected in the docs and, if relevant, the `README.md`.

## Git and PRs

- Commits and PR titles: `<type>(<optional scope>): <gitmoji> <description>`. Types: `feat`
  (minor), `fix` (patch), `docs`, `style`, `refactor`, `test`, `chore`; `!` for breaking changes.
- Agent commits carry `Assisted-by: <harness>:<model>` and no `Co-authored-by`.
- Follow `.github/PULL_REQUEST_TEMPLATE.md`; prefix agent-authored descriptions and comments with
  `:robot: _AI text below_ :robot:`.

## Layout

Public headers `cpp/include/monoprop/`, implementation `cpp/monoprop/`, Python API
`src/monoprop/`, binding template `src/monoprop/bindings/binder.h`, generators
`tools/generate-*.py`, sibling distributions `packages/*`. The repository root is the `uv`
workspace's `monoprop` package.

`bindings.cpp` and `_dispatch.py` are generated -- never edit them. To change the C++ API: edit
the header and implementation, then `binder.h` if Python needs it, then rebuild the project and run both C++ and Python tests.

## Commands

We use `uv` for environment management.
We also use [`just`](https://github.com/casey/just) for task automation. The recipes in the `justfile` show how to build and test in the supported configurations.

```bash
uv sync --all-groups --all-extras -v
uv run pytest
prek run --all-files
just build-docs
```

## Notes

- Python cases live in `tests/cases.py` (`load_problem()` reads `tests/data/*.msgpack`); C++ uses
  `test_utils::load_case()` from `cpp/tests/TestData.h`.
- Pytest's fd capture hides C++ stderr such as `COMMPROF`; rerun with `-s`.
- Benchmark evidence: `pytest benches --runtime-shape=... --build-mode=...` refuses contradictory
  runs. `tools/benchmark-rank-local-openmp.py` has exactly three modes, `observe`, `validate` and
  `compare` (see `docs/content/docs/benchmarks.mdx`). Launch MPI-off runs directly: `mpi4py` is
  never imported when `monoprop.has_mpi` is false.
- `uv sync` does not relink `bin/monoprop_unit_tests.x` -- compare mtimes and use the standalone
  recipe in `docs/content/docs/building.mdx`.
- OpenMP is a required build dependency in every configuration (`OpenMP::OpenMP_CXX`, linked PUBLIC on
  both `monoprop-objs` and `monoprop`, `find_dependency` in the package config). The worksharing
  primitive is `detail/parallel/Workshare.h::for_blocks`. Never call `omp_set_*` from the library.
  The library calls the runtime (`ThreadBudget.cpp`), so `import monoprop` loads it: with `OMP_PLACES`
  set and binding on, the importing thread is bound to the first place.
- Transitional one-store prototype: only explicit `partitions=1` on an ordinary comm captures
  `monoprop_NUM_THREADS` (strict parser) and enforces MPI's initializing thread. Every state operation
  separates validation from mutation (`run_operation_`): a failure after mutation starts invalidates the
  object and aborts a multi-rank ordinary communicator (`mpi::operation_failed`). Catch failures inside
  posted `Ticket`/`PendingAlltoallv` lifetimes, never only at the outermost level.
- `cpp/tests/mpi_failure_driver.cpp` has its own `main()`, excluded from the unit-runner glob;
  `ctest -L mpi_failure` runs its two-rank abort scenarios (30 s timeout; a timeout is a hang).
- Slow CTest startup in MPI builds is `MPI_Init` fabric probing; see
  `monoprop_TEST_EXCLUDE_MPI_FABRIC` in `cpp/tests/CMakeLists.txt`.
- Sanitizer rebuild and test:
