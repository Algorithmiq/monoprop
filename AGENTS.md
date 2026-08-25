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
- **Runtime mode width**: there is no compile-time ceiling on the mode count. A propagator takes its
  logical width as a constructor argument and sizes its monomial storage from it at runtime.
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
- `cpp/include/monoprop/MonomialPropagator.h`: the single C++ engine `MonomialPropagator` (the
  Majorana/Pauli choice is a runtime `Basis`, not a separate class; so is the mode count).
  `storage_modes_for()` is the storage-width rule: a logical width rounds up to a whole 32-mode block,
  never below one block. Pass `storage_num_modes` to override it — the C++ tests do, via
  `cpp/tests/TestPropagator.h`, because the width is part of a monomial's hash and hence of the order
  coefficients accumulate in.
- `src/monoprop/bindings/bindings.cpp`: the binding for that one class, in one translation unit.
  Nothing here is generated -- there is no `binder.h` and no `bindings.cpp.in` any more, so the nanobind
  version the module reports arrives as `monoprop_NANOBIND_VERSION` from CMake rather than through a
  configured template.
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

### Core abstractions (the propagation backbone)

- **A monomial** (`cpp/monoprop/core/Monomial.h`): ONE basis operator, two bits per mode/qubit.
  Basis-agnostic — read as a Majorana product, or as a Pauli string (JW image). Collections:
  `MonomialList` (no coeffs) and `MonomialMap` (monomial → real coeff).
  `Bitset` (`cpp/monoprop/Bitset.h`) *is* a monomial — there is no monomial type and no alias for one,
  because the width is data, not a template parameter: the first 8 words are inline and wider bitsets
  spill the whole word array to the heap. Per-word loops go through `detail::with_nwords`, which
  dispatches a runtime word count to a compile-time-unrolled arm, in place of the `if constexpr`
  branches a compile-time width allowed. A width therefore only exists per *value*: construct with one
  (`Bitset(num_bits)`), never default-construct as a zero monomial (that is width 0 — copy-then-reset
  instead), take a NumModes from `mono.size() / 2`, and recover a width with an instance call
  (`x.size()`), never a qualified `decltype(x)::size()`. Above 8 words (256 modes) the words spill to the
  heap, so memory accounting over a container of monomials must add `Bitset::heap_bytes()` per element;
  `sizeof(Bitset)` per element looks right until someone runs a wide system.
  Two consequences for per-term code, both measured: a `Bitset` is sized for the widest supported
  width rather than its own, and constructing one is a real construction rather than a compile-time
  constant. So in anything on the per-term path, write a word loop (`x.word(w) & m.word(w)`) instead
  of an `a & b` / `^` / `>>` chain that materializes a temporary per step, and take masks from
  `cached_even_bits` (`Utilities.h`) or a per-layer context rather than rebuilding them per call.
  Within-word pair tricks like `(word >> 1) & even_mask` are safe because a mode's two bits are
  `2m, 2m+1` and never straddle a word.
  Only words `[0, num_words())` hold a value: the inline tail above them is deliberately left
  indeterminate so a copy costs the operand's own width instead of the widest supported one. Every
  reader — the word loops, `operator==`, `SplitmixHash`, the MPI readers — must therefore stop at
  `num_words()`, and a new one that walks the whole inline array will read garbage rather than zeros.
- **The row-store seam**: a dense monomial is a transient, not the storage. `TypeAliases.h` declares
  four accessors — `materialize_row`, `assign_row`, `row_popcount`, `for_each_row_position` — and
  three backends answer them: `std::vector<Bitset>`, `detail::OperatorIndex` (packed position lists) and
  `detail::SparseRowStore` (fixed-width mode lanes plus one 2-bit-per-slot `codes` word per row). Reach
  rows through the accessors, never through a backend's own API, and add any fourth backend to
  `cpp/tests/row_accessor_tests.cpp`, which asserts that all of them agree through every accessor.
  A row slot is sized at runtime from the width: `OperatorIndex` keeps one `uint8_t` array and one
  `uint16_t` array with exactly one non-empty, bound per call by its private `with_rows`, because the
  rows are the operator's largest array and one fixed `uint16_t` doubles the whole footprint at or below
  128 modes — where both shipping models sit. Rows are payload, never a hash input and never serialized,
  so a widening there changes no term and no energy: `just diff-baseline` cannot see it, and the gate is
  instead the `operator_terms_bytes` assertions in `cpp/tests/operator_index_tests.cpp` and
  `tests/test_mode_width.py`.
- **Which backend, and where it is bound**: a propagator uses one of the two, chosen once from its
  storage width by `SparseRowStore::preferred_for_modes()` — a build-time constant
  (`monoprop_SPARSE_ROW_MIN_MODES`, defaulted off `monoprop_ENABLE_ARCH_FLAGS`) because what moves the
  crossover is the target ISA. `monoprop_ROW_STORE=dense|sparse` forces it process-wide; an
  unrecognized value throws rather than falling back, since the point of setting it is to know which
  backend ran. `MPOperator` holds one pointer per backend with exactly one non-null and binds the live
  one via `with_store` — **once per layer, inside `build_layer`**, never per term: the scan asks the
  store for a row per anticommuting term, so everything downstream is templated on the store
  (`LayerBuildEngine<Sink, Store>`, `fused_find_and_collect`, `probe_incoming_queries`). Off that path,
  use `MPOperator`'s forwarding accessors; there is no accessor handing out a store, because there is no
  one type to hand out. Every C++ case runs a second time under `monoprop_ROW_STORE=sparse` (the
  `sparse-rows` ctest label) — every fixture is below the crossover, so without that the sparse backend
  would ship untested. The two backends agree on term sets and values but not on term *order*, so
  compare them with `just diff-baseline-sparse` (tolerance), never `just diff-baseline` (byte-wise).
  A benchmark run records the backend it resolved to in its artifact's `meta` (`monoprop_row_store` as
  asked, `row_store_effective` as run) and `REPORT.md` shows both: under the default `auto` the setting
  alone does not identify the backend, and the two differ in footprint and in accumulation order.
  `algebra/CodesAlgebra.h` is the structural algebra on a sparse row, one function per dense
  counterpart, reading the `codes` word instead of looping over storage words, plus `sparse_toggle` --
  the product `M ⊕ G` as one merge over two ascending lane arrays, which is the per-term operation the
  representation exists for. It is exact, not an approximation: `cpp/tests/codes_algebra_tests.cpp` and
  `codes_product_tests.cpp` assert agreement with each dense version over the fixtures and randomized
  rows, and those tests are the gate on making the codes form the default. Change one side and you must
  change the other. A product can occupy more modes than either input, so a scratch row is sized
  `CutoffEvaluator::max_mode_bound()` **plus the generator's locality**; past its capacity
  `sparse_toggle` reports `overflowed` and the caller must fall back to the dense product — never
  truncate, because a truncated mode list still carries a plausible-looking `codes` word.
- **The per-term kernel seam**: everything the scan asks about one anticommuting term — the product,
  the overlap, the rotation sign, the structural cutoff, the owner rank and the query record — goes
  through the per-gate object `TermProductsFor<Store, A, W>` selects (`layer_build/TermProduct.h`), so the
  scan itself names no representation. `SparseTermProducts` answers the first four off the `codes` word
  and falls back to `DenseTermProducts` per term when there is no row to read (a spilled store row, a
  product past the scratch capacity) or no codes form of the cutoff (`CutoffEvaluator` recovered neither
  concrete functor, e.g. under a basis change). `cpp/tests/term_product_tests.cpp` compares the two
  kernels answer for answer: extend it with any new answer, or that answer ships untested.
- **The third thing bound once per layer**: the storage word count, beside the algebra and the backend.
  `with_kernel_width<Store>` (`layer_build/TermProduct.h`, over `detail::with_nwords`) turns
  `gen.num_words()` into a template parameter `W` at the same seam in `build_layer`, and
  `fused_find_and_collect<A, W>` and the `TermProductsFor<Store, A, W>` specialization
  `DenseTermProductsW<A, W>` are templated on it, so every per-term word loop has a compile-time trip
  count and every operand's storage pointer is resolved once per gate — which is what a `Bitset<NumBits>`
  used to give for free. Measured worth ~10% at two and four storage words and nothing at seven or eight,
  so `kNarrowKernelWords` (`TermProduct.h`, 4) caps which widths get an instantiation; the cost is ~11%
  of `.text`. Not a build option, unlike `monoprop_SPARSE_ROW_MIN_MODES`: the cap is a *storage word*
  count, so it names the same width regime on every machine, where the sparse crossover has to follow
  the target ISA. Two conditions on that number, both measured. It is the **Majorana** path:
  the Pauli rotation sign already loops over the generator's non-zero words only, so `W` binds no trip
  count there and the 127-qubit kicked-Ising model gains ~1%. And it scales with how much of a run is in
  the per-term product at all, so a loose `lower_atol` — which rejects a term on its coefficient before
  the product is computed — sees about a third of it. Three consequences for the code.
  `DenseTermProductsW` specializes the cutoff only for a **length** cutoff over the **whole register**; a
  support cutoff or a narrower active window keeps going through `CutoffEvaluator`, and
  `uses_word_cutoff()` is how a test tells those apart. (A support arm was tried and measured: ~1% worse
  everywhere, and no gain even on the Pauli models that use it, because their per-term time is not in the
  cutoff.) `WordKernel<W>` (`Bitset.h`) is the four word ops with `W` fixed that stand in for a `Bitset`
  *method* — standing in for one is the membership rule, which is why the scan's fifth bound-width pass,
  `fully_paired_words<W>`, lives in `algebra/AlgebraCommon.h` beside the `cutoff_sums` it answers for and
  the even-bit literal it shares with `CutoffMasks::make`. Two of the four — the fused XOR and the
  AND-fold behind `parity_and` — are the *same* definitions `Bitset`'s own inline arms use
  (`detail::fused_xor_words` / `and_fold_words`, declared ahead of both), because one of them decides
  emitted term signs and neither may drift from the method it stands in for. `splitmix` is deliberately
  a second implementation instead: that value is `monomial_hash`, so it routes MPI ownership and must
  stay bit-identical, and `cpp/tests/word_kernel_tests.cpp` asserts it equal to `SplitmixHash` at every
  `W` rather than by construction. `term_product_tests.cpp` compares the whole kernel against
  `DenseTermProducts` over the whole inline regime, not just the capped widths — both files sweep the
  regime through the one `test_utils::for_each_inline_width` in `cpp/tests/InlineWidths.h`, so the range
  cannot be narrowed in one of them alone. And the kernel's precondition is that every operand is
  inline, so `W` is never bound above `Bitset::kInlineWords` — which `kNarrowKernelWords` is
  `static_assert`ed against in the same header, so lowering `kInlineWords` fails the compile rather than
  silently specializing a spilled width. `DenseTermProductsW` is non-copyable because its three word
  pointers point into its own bitsets; a copy would read and write the original's storage.
  Two further bindings were tried past this seam and both measured at nothing — under 0.05% of the
  instruction count on either shipping model, pinned single-threaded — because the optimizer already
  hoists them out of the inlined scan loop: resolving the algebra's per-term sign inputs into a
  per-gate struct the kernel holds (`A::SignContext`), and writing the query record with the word count
  bound (`query_push_words<W>`, one capacity check instead of one per word). Measure any third one the
  same way before adding it; wall clock cannot see this range, and neither can an instruction count taken
  with the thread pool live, which spins hard enough to inflate the total ~14x.
- **The query record**: a store is queried in the form it keys its rows by, so a resolve never converts —
  `QueryKeysFor<Store>` (`layer_build/Common.h`) picks the batch, and `query_payload_words_for(store,
  capacity)` the width. A buffer is `[nq][record 0]…[record nq-1][dense escape tail]`: the header,
  because a tail means `size/stride` is no longer the record count; the tail, because a query is `M ⊕ G`
  and a fully paired product escapes the cutoff, so no fixed-stride sparse record can hold every one. An
  escaped record keeps its place and its stride, marks lane 0 with `SparseRowStore::kOverflowLane` and
  carries its tail *index* where the codes word would go — an index into the tail, never an offset into
  the buffer, which is what lets the fused sink widen every record without renumbering anything. Push
  records through `TermProducts::push(QueryOut{records, escapes}, phase)` and finish a stream with
  `append_escape_tail`; never append a record after the tail has started.
  `owner()` is still the dense `monomial_hash` on both sides, because owner routing is that hash
  everywhere including `find_rank` — so a multi-rank run still materializes one monomial per surviving
  term, and moving that means changing `find_rank` too.
- **The anticommutation fold** (`detail::InvertedIndex`): the transpose of the row store, one column per
  *bit position* — not per mode. That keying is settled and measured: a mode-keyed column cannot answer
  a generator slot that names one Majorana of a mode, which is 66% of the Hubbard generators' slots and
  31% of the kicked-Ising ones, so anything mode-shaped is an
  additional derived tier, never a re-keying. Two facts to keep in mind before touching it. The fold is
  `O(rows)` per *gate* regardless of how few terms anticommute, so it is 15% of a many-cheap-gates Pauli
  run and under 3% of a few-expensive-gates Majorana one — measure on the right workload. And its memory
  is the dense tier (>90% on both shipping models) until the operator gets sparse enough that the
  `Column` vector itself takes over, which happens below roughly `terms < 5 × modes`;
  `d_invidx_columns_bytes` in `operator_memory_breakdown()` is that term.
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


### Mode width

There is one compiled `MonomialPropagator`, whatever the mode count. `MonomialPropagator` in Python is
an abstract base; construct a concrete front-end, which reads the mode count off the operator — there
is no `num_modes` argument:
```python
# Basis::Majorana here; PauliPropagator passes Basis::Pauli to the same C++ class
mp = MajoranaPropagator(operator, initial_state, cutoff=4)
```
Above 250 modes (8 words, `Bitset::kInlineWords`) a monomial's words spill to the heap, so wide
systems work but pay an allocation per by-value monomial.
### Testing Structure

- `tests/cases.py`: Parametrized test cases using `pytest-cases`; `load_problem()` loads a `tests/data/*.msgpack` fixture directly into the public API (`MonomialCircuit` + `MonomialOperator`). C++ tests use the equivalent `test_utils::load_case()` in `cpp/tests/TestData.h`
- Fixture msgpack schema is documented in `tests/data/README.md`
- Tests validate against exact solutions for small systems
- Heavy use of `@parametrize_with_cases` decorators
- Nothing on disk is wider than 28 modes, i.e. one 32-mode storage block. A case that needs the
  wide-system regime (260 logical / 288 storage modes: nine words per monomial, so past `Bitset`'s eight
  inline ones *and* above the crossover a wheel selects sparse rows at) is
  *derived*: `ModeEmbedding`/`WIDE_EMBEDDING` in `tests/cases.py` and `test_utils::embed_case` in
  `cpp/tests/TestData.h` relabel a fixture's modes into a wider system. The map is monotone, so the
  fixture's exact energy and gradient still apply and the wide run owes the narrow run's evolved
  operator term for term (`tests/test_wide_system.py`, `cpp/tests/wide_system_tests.cpp`). Add a case
  that way rather than checking in a blob that differs from an existing one by a permutation.
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
6. Implement in the corresponding `.cpp` under `cpp/monoprop/`.
7. Add Python bindings in `src/monoprop/bindings/bindings.cpp`
8. Test with both C++ and Python tests

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
- **`uv sync` does not relink `bin/monoprop_unit_tests.x`** — check its mtime against the
  source's; recipe for a standalone C++ build tree in `docs/content/docs/building.mdx`.

This is a sophisticated scientific computing project requiring careful attention to build system
configuration and the C++/Python boundary.
