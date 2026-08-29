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
- **Tiered, not multiversioned**: the ISA is chosen per *whole library*, never per function.
  An attribute cannot do this job: GCC will not inline across an `arch` mismatch (so a `target`-attributed
  wrapper around the engine is a `jmp`, `flatten` notwithstanding) and `#pragma GCC target` does not
  capture templates defined outside its region -- header-resident code is widened by its TU's command
  line or not at all. And `flatten` + `target_clones` is unaffordable regardless: four flattened clones
  of `build_layer`'s `with_algebra` x `with_store` x `with_kernel_width` fan-out took one TU from 16.7 s
  to >20 min at ~100 GB of compiler memory. Consequences for the build: every engine
  source goes through the `monoprop_engine_sources(...)` macro rather than
  `target_sources(monoprop-objs ...)`, or it is missing from three of the four tiers; every per-target
  setting goes through `_monoprop_configure_engine_objs` in `cpp/monoprop/CMakeLists.txt`, so the
  tiers cannot drift apart in anything but arch flags. Consequence for numerics: `-ffp-contract=off`
  is project-wide and is a **contract**, not a tuning knob -- without it `-march=x86-64-v3` and up
  contract `a*b+c` into an FMA and the energy moves by 1-2 ULP, which in a fat binary means the same
  wheel answering differently per host CPU. The byte-wise gate on it is a golden-baseline capture per
  variant, which arrives with the baseline tooling.

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
- **Peak memory is the kernel's `VmHWM` high-water mark** — exact, with no sampling. Under
  MPI the ranks' peaks are summed, which errs high (disjoint transients, and shared pages
  charged to every rank): an upper bound, good for regressions, not for provisioning.
- `cmake/compiler_flags/FatBinary.cmake`: the **only** place an ISA tier or a narrow-vector core is
  declared, and it generates the loader's predicate table (`FatVariants.h`) so a tier cannot be built
  without being selectable or selectable without being built. Five tiers, all `-mtune=skylake`:
  `x86-64`, `-v2`, `-v3`, and `-v4 -mavx512vpopcntdq` at each of `-mprefer-vector-width=256` and `512`.
  A published x86-64 wheel compiles the whole engine once per tier and `src/monoprop/_bootstrap.py`
  loads one as `monoprop._core` at import, choosing with the tiny baseline-ISA probe
  `src/monoprop/bindings/isa.cpp`. Off by default in source builds, where `-march=native` beats every
  tier. See `docs/content/docs/fat-binary.mdx`.
- **Two of the tiers are one ISA at two vector widths** (`...-vw256`, `...-vw512`): identical `-march`
  and identical `__builtin_cpu_supports` requirements, differing only in `-mprefer-vector-width`. They
  exist because GCC otherwise takes that from the `-mtune` tables, i.e. from `monoprop_FAT_MTUNE`, and
  because no feature bit reports what the question actually turns on -- how wide the datapath behind the
  registers is and what the core charges in clock for using it. So the discriminator is a core-name
  table, `monoprop_FAT_NARROW_VECTOR_CORES`, read through `__builtin_cpu_is`; `znver4` is on it because
  it is measured (1.1% to the narrow tier on the kicked-Ising model, disjoint ranges, *against* GCC's own
  znver4 tuning). One consequence: each tier now carries **two** predicates -- `runnable` (features only,
  what gates a `monoprop_VARIANT` pin) and `preferred` (plus the table, what the automatic selection and
  `supported_variants()` read) -- and conflating them makes the wide tier unpinnable on exactly the
  machines worth comparing it on. Nothing but a disassembly can tell the pair apart, so
  `tests/test_variants.py` asserts that the width *setting* differs while the feature set does not.

### Core abstractions (the propagation backbone)

- **`Monomial<N>`** (`cpp/monoprop/core/Monomial.h`) = `Bitset<2*N>`: ONE basis operator, two bits per
  mode/qubit. Basis-agnostic — read as a Majorana product, or as a Pauli string (JW image).
  Collections: `MonomialList<N>` (no coeffs) and `MonomialMap<N>` (monomial → real coeff).
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
uv run pytest  # Run tests (monoprop's suite + the workspace members' suites)
SKBUILD_CMAKE_BUILD_TYPE=AsanUbsan SKBUILD_CMAKE_DEFINE="monoprop_SANITIZER=asan-ubsan" uv sync --group workspace-test --all-extras --reinstall-package monoprop --no-cache -v  # Rebuild when changing sanitizer settings.
LD_PRELOAD="$(g++ -print-file-name=libasan.so):$(g++ -print-file-name=libstdc++.so.6)" ASAN_OPTIONS=detect_leaks=0 uv run pytest  # Python tests against a sanitizer tree
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
6. Implement in the corresponding `.cpp` under `cpp/monoprop/`, and register a *new* `.cpp` with
   `monoprop_engine_sources(...)` -- never `target_sources(monoprop-objs ...)`, which reaches only
   the baseline fat-binary tier.
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
