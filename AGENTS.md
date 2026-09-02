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
  series. Both bench modules measure the same four operations — `build_graph`, `propagate`,
  `energy`, `gradient`.
- **`rounds > 1` overlaps two rounds' live memory** (`setup=` runs before the prior round's
  teardown) — pin `--bench-rounds=1`; `record_memory` measures that construction transient,
  not per-op cost (use `op_memory`).
- **Peak memory is the kernel's `VmHWM` high-water mark** — exact, with no sampling. Under
  MPI the ranks' peaks are summed, which errs high (disjoint transients, and shared pages
  charged to every rank): an upper bound, good for regressions, not for provisioning.

### Core abstractions (the propagation backbone)

- **`Monomial<N>`** (`cpp/monoprop/core/Monomial.h`) = `Bitset<2*N>`: ONE basis operator, two bits per
  mode/qubit. Basis-agnostic — read as a Majorana product, or as a Pauli string (JW image).
  Collections: `MonomialList<N>` (no coeffs) and `MonomialMap<N>` (monomial → real coeff).
- **Partner lookup is per gate, not per operator** (`cpp/monoprop/detail/evolution/layer_build/BucketJoin.h`):
  a tracked partner `M⊕G` of an anticommuting term anticommutes with `G` too, so it is found inside the
  gate's own anticommuting set. `detail::OperatorIndex<N>` therefore keeps no hash table — rows only —
  and the out-of-gate lookups (`get_operator`'s pending drain, `update_initial_operator`) build a transient
  `detail::TermLookup` (`detail/operator/TermLookup.h`) over the rows they need. The per-gate key is the
  routing fingerprint (`routing::fingerprint_positions`, `detail/mpi/Routing.h`), GF(2)-linear so the
  partner's key is `fp(M) ^ fp(G)`; every key match is confirmed against the positions. Neither side of
  that join is small (with `lower_atol` the records are ~0.85 of Anti(G)), so **neither is the
  random-access side**: the scan stages its anticommuting rows with the key it folds while the row is in
  cache, the records are staged after the exchange, and both are bucketed by the key's top bits and joined
  bucket by bucket in L1. The protocol's per-row state (`rot`, `foll`, `received`, `partner_rot`) is
  `RowMarks` in `GateScratch.h`, bitsets over rows cleared per gate by zeroing the fold's `nz` words.
- **One round per gate** (`cpp/monoprop/detail/evolution/layer_build/Engine.h`, records in `Scan.h`,
  join in `Resolve.h`): every anticommuting term whose partner passes the structural cutoff sends ONE
  record `(key = M⊕G, φ, rot, [value])` to the partner's owner, in the same `alltoallv`. The receiver
  joins each record against its own anticommuting rows: a hit rotates iff the record's `rot` or the row's
  own `rot` is set (the pair's two adds happen on the two owners, with `φ(M⊕G,G) = −φ(M,G)`), a `rot=1`
  miss mints the key at `base+j` in join order, a `rot=0` miss is dropped; afterwards `rot ∧ ¬received`
  on a sent row means the partner is absent everywhere. There is no query→response pass and no
  leader/follower split on the wire; graph mode derives its positional in/out pairing from the same
  join order plus the pivot bit. A silent (`rot=0`) record exists only so its partner can rotate with
  the sender's value and not read the sender as absent; `monoprop_DROP_SILENT_RECORDS=1` (off by
  default, fused path only, `detail/EnvConfig.h`) drops them for a below-`lower_atol` error per pair.
- **Row access** (`cpp/monoprop/detail/operator/RowAccess.h`): the one backend-agnostic vocabulary
  (`materialize_row`, `assign_row`, `row_popcount`, `for_each_row_position`) over the dense
  `MonomialList<N>` and the packed `detail::OperatorIndex<N>`. Any template parameterized on the row
  store must include that header — the `OperatorIndex` overloads live in `monoprop::`, so ADL cannot
  find them from a `monoprop::detail` argument.
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


### Environment Management
We use `uv` for environment management.
We also use [`just`](https://github.com/casey/just) for task automation.

```bash
uv sync --all-groups --all-extras -v  # Build & install (workspace-wide)
monoprop_ENABLE_MPI=ON uv sync --all-extras --reinstall-package monoprop --no-cache -v  # MPI-enabled build
uv run pytest  # Run tests (monoprop's suite + the workspace members' suites)
SKBUILD_CMAKE_BUILD_TYPE=AsanUbsan SKBUILD_CMAKE_DEFINE="monoprop_SANITIZER=asan-ubsan" uv sync --group workspace-test --all-extras --reinstall-package monoprop --no-cache -v  # Rebuild when changing sanitizer settings.
LD_PRELOAD="$(g++ -print-file-name=libasan.so):$(g++ -print-file-name=libstdc++.so.6)" ASAN_OPTIONS=detect_leaks=0 uv run pytest  # Python tests against a sanitizer tree
just build-docs  # Build documentation
```

Nix users get the same toolchain with `nix develop`. The flake (`flake.nix` plus
`nix/`) also exposes `packages.monoprop{,-mpi}`, `overlays.default`, and a `nix run`
app. Downstream flakes should follow their own `nixpkgs` and use the overlay when
composing monoprop into a shared Python package set. The flake and `nix/` are
excluded from Python sdists, so Nix consumers must use a repository flake input.
The packaged build disables the C++ unit tests and arch flags. Its stable version
comes from the tracked root `VERSION` file because setuptools-scm cannot read git
metadata in the build sandbox; update that file when preparing a release. The
release workflow rejects tags that do not match it.


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

- `tests/cases.py`: Parametrized test cases using `pytest-cases`; `load_problem()` loads a `tests/data/*.msgpack` fixture directly into the public API (`Circuit` + `MajoranaOperator`). C++ tests use the equivalent `test_utils::load_case()` in `cpp/tests/TestData.h`
- Fixture msgpack schema is documented in `tests/data/README.md`
- Tests validate against exact solutions for small systems
- Heavy use of `@parametrize_with_cases` decorators
- **pytest's fd-level capture hides C++ stderr** (e.g. `COMMPROF`) — rerun with `-s` to see it.
- QA coverage uses the `just code-coverage-collect` and `just code-coverage-aggregate` recipes for separate serial and MPI `Coverage` builds.
- **A slow CTest run on an MPI build is `MPI_Init` fabric probing, not slow tests** — see `monoprop_TEST_EXCLUDE_MPI_FABRIC` in `cpp/tests/CMakeLists.txt`.

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
