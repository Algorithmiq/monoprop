# Agent Instructions for monoprop

monoprop is a high-performance C++/Python hybrid library implementing Majorana and Pauli propagation. The project combines modern C++23 with Python bindings via nanobind.


## Repository rules
- **C++23**: All C++ code must use C++23 features and idioms.
- **Auto style**: Use `clang-format` with the provided `.clang-format` file
- Python code must pass lints as defined in pre-commit hooks.
- We use conventional commits with syntax: `<type>(<optional scope>): <gitmoji> <description>`:
  - Commit body and footers are optional.
  - Breaking changes should have `!` in the commit message.
  - `<type>` can only be one of:
    - `feat`: new feature, preserving the API. This corresponds to a *minor version bump*.
    - `fix`: bug fix. This corresponds to a *patch version bump*.
    - `docs`: changes to the documentation. **No** functional changes.
    - `style`: formatting, *i.e.* running `black`. **No** functional changes.
    - `refactor`: refactoring production code, *i.e.* improving class structure. **No** functional changes.
    - `test`: adding/refactoring tests, *i.e.* improving code coverage. **No** functional changes.
    - `chore`: "boring" tasks, *i.e.* updating dependency pins in `pyproject.toml`. **No** functional changes.
- If you make a commit, add a trailer: `Assisted-by: <harness>:<model>`, where `<harness>` is the current agent harness (like ClaudeCode), and `<model>` is the AI model (like claude-opus-4.8). You don't need to add a coauthored-by when you have this.
- PR titles must adhere to the same conventional commits format. PR text should follow the PR template in the `.github` folder.
- Prefix PR descriptions and comments on PRs with the line ":robot: _AI text below_ :robot:" to indicate you are an agent speaking on a user's behalf.
- Python docstrings use Google style, and are rendered into the docs site by `just gen-api` — keep
  them accurate.
- In prose docs (`docs/content/docs/**.mdx`) and Python docstrings, link to API symbols with the mkdocstrings-style `[Symbol][]` reference (or `[Display][fully.qualified.path]`) — never hard-code `/api/...` URLs. Do not backtick the name in the `[Symbol][]` form. See `docs/content/docs/documenting.mdx`.
- C++ comments: bare `//` for one-line comments, `/* */` for block comments.
- Add Doxygen Qt-style documentation on all declarations in header files.
  Put documentation after members in enums, structs, classes.
- Comments state what the code cannot: invariants, ordering and lifetime contracts, sign and bit
  conventions, and why an obvious alternative was rejected. Do not restate the code, narrate history
  (git has it), or repeat a fact that already has a home elsewhere.


## Architecture Overview

- **Core C++ Engine**: Public headers in `cpp/include/monoprop/` and implementation in `cpp/monoprop/`
- **Python Interface**: User-facing API in `src/monoprop/` with C++ bindings in `src/monoprop/bindings/`
- **Template-Based Design**: Heavily templated C++ code with compile-time mode limits (`monoprop_MAX_NUM_MODES`)
- **Generated Code**: Python dispatch and C++ bindings auto-generated via `tools/generate-*.py`
- **uv workspace**: the repository root is the `monoprop` package; `packages/*` holds the sibling
  distributions. See "Workspace layout" below.

### Workspace layout

The root `pyproject.toml` declares a `[tool.uv.workspace]`, so one `uv.lock` covers every member.

- `monoprop` (repository root, `src/monoprop/`) — the library, built by scikit-build-core.
- `packages/monoprop-bench-tools/` (`monoprop_bench_tools`) — the reusable half of the benchmark
  harness, published to PyPI: `memory.cpu` / `memory.gpu` (peak-footprint measurement), `models`
  (the benchmarked problem builders), `report` and `bmf` (the artifact renderers). Pure Python,
  built by hatchling, versioned off the same git tags as `monoprop`. It ships the console scripts
  `monoprop-bench-report` and `monoprop-bench-bmf`. It contains **no benchmarks**.
- `packages/bench-third-party/` — cross-engine comparison scripts. Listed in the workspace
  `exclude`: it pins a narrower `requires-python`, a git dependency and linux-x86_64-only CUDA
  wheels, so it is a standalone uv project with its own `uv.lock`. Run it with
  `cd packages/bench-third-party && uv sync`, never from the root environment.
- `benches/` — monoprop's own benchmark suite (`conftest.py`, `bench_*.py`, `results/`). It stays in
  the repository and imports the tools package. Benchmark names are Bencher's history key, so they
  must not move with a library release; that is why the suite is not in `monoprop-bench-tools`.

Dependency groups follow from that split: `test` is monoprop's own suite only (cibuildwheel installs
it against a built wheel, so it must not reference a workspace member), `workspace-test` adds
`monoprop-bench-tools` and is what `uv run pytest` at the root needs, and `bench` adds
`pytest-benchmark` on top.

Key files:
- `src/monoprop/monomial_propagator.py`: abstract base `MonomialPropagator`; the concrete
  user-facing front-ends are `src/monoprop/majorana_propagator.py` (`MajoranaPropagator`) and
  `src/monoprop/pauli_propagator.py` (`PauliPropagator`).
- `cpp/include/monoprop/MonomialPropagator.h`: the single templated C++ engine `MonomialPropagator<NumModes>`
  (the Majorana/Pauli choice is a runtime `Basis`, not a separate class). Its `only_rotate_len_k`
  arguments use `std::optional<size_t>`; `std::nullopt` means no gate-application length cap.
- `src/monoprop/bindings/binder.h`: hand-written binding template; `tools/generate-*.py` generate the
  per-mode-width `bindings.cpp` and `_dispatch.py` from it (do not hand-edit the generated files).
  Both generators take the 32-mode storage-block rule from `tools/_binding_layout.py` — they must
  agree, or dispatch routes at a template the bindings never instantiated.
- `CMakePresets.json`: the single source of truth for the supported C++ unit-test build/run entry
  points. The presets adopt the scikit-build-core trees generated by `uv sync`; regenerate the tree
  with `uv sync`, then use the matching `skbuild-*` preset to build or run CTest.
- `monoprop_bench_tools.report` and `monoprop_bench_tools.bmf`: the two renderers of a benchmark
  run's artifacts — `REPORT.md` for humans, Bencher Metric Format JSON for the `bench_main.yml`
  continuous-benchmarking workflow. Both read the schema written by `benches/conftest.py`, so a
  change to the recorded sections has to land on both sides of the package boundary. Benchmark
  names are Bencher's history key, so renaming or moving a `bench_*` test orphans its tracked
  series.
- **`rounds > 1` overlaps two rounds' live memory** (`setup=` runs before the prior round's
  teardown) — pin `--bench-rounds=1`; `record_memory` measures that construction transient,
  not per-op cost (use `op_memory`).
- **Peak memory is `HighWaterMark`, not sampled PSS** — exact, unlike `/proc/self/smaps_rollup`.

### Core abstractions (the propagation backbone)

- **`Monomial<N>`** (`cpp/monoprop/core/Monomial.h`) = `Bitset<2*N>`: ONE basis operator, two bits per
  mode/qubit. Basis-agnostic — read as a Majorana product, or as a Pauli string (JW image).
  Collections: `MonomialList<N>` (no coeffs) and `MonomialMap<N>` (monomial → real coeff).
- **`Basis` / the `Algebra` policy** (`cpp/monoprop/algebra/`): the two algebras are sibling models
  (`MajoranaAlgebra`, `PauliAlgebra` in `algebra/Algebra.h`) over shared structural primitives
  (`algebra/AlgebraCommon.h`). The propagation backbone (the scan/fold in `detail/evolution/...`) is
  templated on the algebra policy and bound to a runtime `Basis` once, via `with_algebra`.
- **The partition facade**: `partitions > 1` makes a `MonomialPropagator` a facade over S single-partition
  propagators, one hash partition each. Every method that fans out must use the private partition
  vocabulary declared in `MonomialPropagator.h` (`for_each_partition_`, `map_partitions_`, `concat_partitions_`
  for the mutating/collecting paths, which run on the partitions' own pinned masters; `sum_partitions_`,
  `fold_partitions_`, `first_partition_` for reads off quiescent partitions) rather than hand-rolling a
  `run_on_all` loop — the declarations record which helper is legal where.
- **The distributed graph is indexed by the FLAT world**: a `LayerCore`
  (`cpp/monoprop/detail/graph_encoding/MPGraphEncodingTypes.h`) is held per layer per partition, and
  its partner index space is `P = mpi_ranks × partitions`, not the rank count. Anything stored per
  layer per *possible* partner is `O(P)` on each of `P` participants — `O(P²)` across a job — so the
  rule for this structure is **store per unit of traffic, not per participant**:
  - `PackedCrossRankStorage` keeps one `CrossRankOccupiedSlot` per world slot that carries traffic,
    not per slot that could exist. Walk it with `for_each_occupied_slot`, which is `O(occupied)` for
    the whole sweep and carries each slot's B/D prefix offset with it; `cross_rank_self_slot` is the
    O(1) path for this rank's own slot, resolved once at build by `resolve_self_slot`.
  - The all-to-all counts and displacements are not stored. `derive_exchange_layout` fills a
    `LayerExchangeLayout` in per-thread scratch for the exchange being posted, and the recv layout
    IS the send layout (the count matrix is symmetric), so there is nothing to transpose and nothing
    to resolve over the wire. `derive_layer_exchange` in `Evolution.cpp` is the call site.
  - `mpi::check_exchange_symmetry` (`cpp/monoprop/detail/mpi/Exchange.h`) runs on every exchange.
    Its width check — the layout must hold exactly one entry per rank — is an unconditional
    precondition, because MPI reads one count and one displacement per rank whatever the span holds.
    The symmetry audit behind it is a collective, and is gated at BUILD time by the CMake option
    `monoprop_CHECK_EXCHANGE_SYMMETRY` (default OFF). A per-rank environment variable must never
    decide whether a collective runs: ranks that disagree hang the job instead of misreporting.
  - `cpp/tests/ExchangeLayoutOracle.h` holds the independent reference that derivation is checked
    against. It is outside the library on purpose — an oracle compiled in beside its subject gets
    maintained beside it, and then proves nothing.


### Environment Management
We use `uv` for environment management.
We also use [`just`](https://github.com/casey/just) for task automation.

```bash
uv sync --all-groups --all-extras -v  # Build & install (workspace-wide)
uv run pytest  # Run tests (monoprop's suite + the workspace members' suites)
just build-docs  # Build documentation
```


### Template Metaprogramming

C++ code uses extensive compile-time templates with `NumModes` parameter:
```cpp
template <size_t NumModes>
class MonomialPropagator { /* ... */ };
```

### Mode-Based Dispatching

Python automatically dispatches to the appropriate C++ template based on the operator's mode count.
`MonomialPropagator` is an abstract base; construct a concrete front-end (which reads the mode count
off the operator — there is no `num_modes` argument):
```python
# Routes to MonomialPropagator<4> in C++ (Basis::Majorana here; PauliPropagator uses Basis::Pauli)
mp = MajoranaPropagator(operator, initial_state, cutoff=4)
```

### Testing Structure

- `tests/cases.py`: Parametrized test cases using `pytest-cases`; `load_problem()` loads a `tests/data/*.msgpack` fixture directly into the public API (`MonomialCircuit` + `MonomialOperator`). C++ tests use the equivalent `test_utils::load_case()` in `cpp/tests/TestData.h`
- Fixture msgpack schema is documented in `tests/data/README.md`
- Tests validate against exact solutions for small systems
- Heavy use of `@parametrize_with_cases` decorators
- **pytest's fd-level capture hides C++ stderr** (e.g. `COMMPROF`) — rerun with `-s` to see it.
- **A slow CTest run on an MPI build is `MPI_Init` fabric probing, not slow tests** — see `monoprop_TEST_EXCLUDE_MPI_FABRIC` in `cpp/tests/CMakeLists.txt`.
- **A layout assertion must hold at BOTH `TermIndex` widths, and must not be able to switch itself
  off.** `monoprop_WIDE_TERM_INDEX` is compiled in exactly one place — a single `wide: "on"` cell in
  `.github/workflows/test.yml`'s matrix — and `just test-wide` locally. Every other build, including
  every HPC gate, compiles narrow. So an assertion of the form
  `sizeof(TermIndex) != sizeof(uint32_t) || <the real check>` is not a guard: its first disjunct is
  *true* under the wide build, which is the only configuration that could have failed it. Write the
  layout **rule** instead of a byte count, so there is no arm to disable — see
  `kOccupiedSlotIdField` and the two `static_assert`s beside `CrossRankOccupiedSlot`
  (`cpp/monoprop/detail/graph_encoding/MPGraphEncodingTypes.h`), which are `12` narrow and `24` wide
  from one expression. A `BOOST_CHECK_EQUAL` on the same rule belongs in `cpp/tests/` as well, but it
  is the weaker of the two: it needs the wide build to be *run*, whereas the `static_assert` fails
  the compile.
- **Ledger dictionaries exposed to Python are a measurement contract.** `graph_memory_breakdown()`
  and `operator_memory_breakdown()` (`src/monoprop/bindings/binder.h`) are read from two builds and
  subtracted, so a key that appears, disappears, or moves in or out of `total_bytes()` silently turns
  that subtraction into a comparison of two different quantities — the structs are plain aggregates
  and the bindings are literal `std::map`s, so nothing on the C++ side notices. Pin the key set
  *exactly* (not as a superset) and pin `total_bytes()` to the sum of its members;
  `test_graph_memory_breakdown_keys_and_totals` in `tests/test_monoprop_smoke.py` is the pattern.
  Prefer keeping a key that has become permanently zero, with a comment saying why, over deleting it
  — a zero row is the evidence that the memory left.

## Key Dependencies & Integration

- **nanobind**: Modern Python-C++ binding (prefer over pybind11)
- **scikit-build-core**: Modern build system replacing setuptools
- **uv**: Package management
- **Boost**: Used for various utilities (unordered_map, unit tests)
- **msgpack**: Serialization of the test-data fixtures only (`tests/data/*.msgpack`); consumed by the Python test loaders and the C++ test suite, not by the shipped library
- **hwloc**: CPU topology discovery and thread binding for partition placement (`CpuTopology.cpp`). Required system library (`libhwloc-dev` on Debian/Ubuntu, `hwloc` on Homebrew). Requires `pkg-config` so CMake can locate `hwloc`. Bundled into wheels automatically by auditwheel/delocate.
- **MPI**: For distributed parallelization

## Common Tasks

### Adding New C++ Functionality
1. Add to the appropriate public header in `cpp/include/monoprop/` or internal header in `cpp/monoprop/`.
2. Use C++23 syntax and idioms.
3. Use almost always auto style.
4. Use trailing return type syntax in function declarations.
5. Add a one-line `///` summary if the declaration is in `cpp/include/monoprop/`; elsewhere add a plain
   `//` note only where the code does not already say it.
6. Implement in the corresponding `.cpp` under `cpp/monoprop/`.
7. Add Python bindings in `src/monoprop/bindings/binder.h`
8. Regenerate bindings with `tools/generate-binders.py`
9. Test with both C++ and Python tests

## Documentation Maintenance Policy

When changing behavior, APIs, build/test workflows, paths, or developer conventions:

1. Update `AGENTS.md` in the same change.
2. Update `README.md` in the same change.
3. Update the docs under `docs/` for user-facing or contributor-facing guidance.
4. Keep commands and paths consistent across all three (`AGENTS.md`, `README.md`, and `docs/`).
5. If a section no longer reflects the codebase, either fix it immediately or remove it.

### Debugging Build Issues
- Check `build/*/compile_commands.json` for compilation flags
- Use `rm -rf build` to clear environment-specific builds
- Verify `monoprop_MAX_NUM_MODES` matches your use case (default: 250)
- **`uv sync` does not relink `bin/monoprop_unit_tests.x`** — check its mtime against the
  source's; recipe for a standalone C++ build tree in `docs/content/docs/building.mdx`.

This is a sophisticated scientific computing project requiring careful attention to template instantiation, build system configuration, and the C++/Python boundary.
