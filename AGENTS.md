# monoprop agent guide

A C++23/Python library for Majorana and Pauli propagation. `MonomialPropagator<NumModes>` is the
shared engine; Majorana against Pauli is a runtime `Basis`, and Python exposes `MajoranaPropagator` and
`PauliPropagator`. `monoprop_MAX_NUM_MODES` is the compile-time mode limit (default 250).

## Rules

- C++23 idioms, trailing return types, follow almost-always-`auto` style, `clang-format` with the
  repository configuration.
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
- `uv sync` does not relink `bin/monoprop_unit_tests.x` -- compare mtimes and use the standalone
  recipe in `docs/content/docs/building.mdx`.
- Slow CTest startup in MPI builds is `MPI_Init` fabric probing; see
  `monoprop_TEST_EXCLUDE_MPI_FABRIC` in `cpp/tests/CMakeLists.txt`.
- Sanitizer rebuild and test:

