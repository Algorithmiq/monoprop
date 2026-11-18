# Changelog

All notable changes to this project are listed here.

## 0.10.0 (2026-11-18)

### Breaking changes

- **wheels:** Publish MPI-enabled wheels across MPI ABI families ([#426](https://github.com/Algorithmiq/monoprop/pull/426))

### Features

- **nix:** Use upstream nanobind 3.1.0 and build for Python 3.12-3.14 ([#391](https://github.com/Algorithmiq/monoprop/pull/391))
- **propagator:** Read every term's expectation value from one adjoint replay ([#393](https://github.com/Algorithmiq/monoprop/pull/393))
- Support Python 3.15 ([#395](https://github.com/Algorithmiq/monoprop/pull/395))
- **build:** Declare named build options with scikit-build-core 1.1 ([#421](https://github.com/Algorithmiq/monoprop/pull/421))
- **container:** Publish an MPI container image for HPC clusters ([#418](https://github.com/Algorithmiq/monoprop/pull/418))
- **propagator:** Add opt-in blockwise truncation to build_graph ([#441](https://github.com/Algorithmiq/monoprop/pull/441))
- **propagator:** Add opt-in max-sins truncation to build_graph ([#444](https://github.com/Algorithmiq/monoprop/pull/444))

### Fixes

- **benchmarks:** Correct step alignment, model name and time axes ([#411](https://github.com/Algorithmiq/monoprop/pull/411))
- **mpi:** Do not fail a completed partition barrier round on a later poison ([#433](https://github.com/Algorithmiq/monoprop/pull/433))
- **propagator:** Forward functional argument instead of moving it ([#453](https://github.com/Algorithmiq/monoprop/pull/453))

### Performance

- **propagate:** Make per-term emit and pass-1 cost independent of N ([#383](https://github.com/Algorithmiq/monoprop/pull/383))

### Documentation

- **mpi:** Ask callers to pass a duplicated communicator ([#430](https://github.com/Algorithmiq/monoprop/pull/430))

## 0.9.2 (2026-10-01)

### Breaking changes

- Bump maximum number of modes to 1024 ([#332](https://github.com/Algorithmiq/monoprop/pull/332))
- **propagator:** Number an incremental Heisenberg axis like the equivalent circuit ([#346](https://github.com/Algorithmiq/monoprop/pull/346))

### Features

- **tools:** Capture and diff a golden baseline ([#302](https://github.com/Algorithmiq/monoprop/pull/302))
- **ci:** Measure propagate at both L2 shapes ([#342](https://github.com/Algorithmiq/monoprop/pull/342))
- **ci:** Run the same six rows at every ladder rung ([#344](https://github.com/Algorithmiq/monoprop/pull/344))
- **ci:** One testbed per shape, and the per-operation memory to compare its rows ([#345](https://github.com/Algorithmiq/monoprop/pull/345))
- **python:** Support `PauliProductRotationGate` in `from_qiskit_circuit` ([#361](https://github.com/Algorithmiq/monoprop/pull/361))
- Support conversion from qiskit `SparseObservable` ([#362](https://github.com/Algorithmiq/monoprop/pull/362))
- **propagator:** Read individual evolved coefficients by term ([#337](https://github.com/Algorithmiq/monoprop/pull/337))

### Fixes

- Compiler warnings ([#334](https://github.com/Algorithmiq/monoprop/pull/334))
- **bench-tools:** Fold the peak-RSS mark into enclosing windows ([#327](https://github.com/Algorithmiq/monoprop/pull/327))
- **ci:** Keep a push to main out of the cancelled group ([#335](https://github.com/Algorithmiq/monoprop/pull/335))
- **cpp:** Build only world-slot-local Schrodinger paired basis ([#328](https://github.com/Algorithmiq/monoprop/pull/328))
- **ci:** Install MPI for the bare-metal benchmark ([#340](https://github.com/Algorithmiq/monoprop/pull/340))
- **bench-tools:** Track every operator's term count, and name series for a legend ([#341](https://github.com/Algorithmiq/monoprop/pull/341))
- **ci:** Wait for 12 samples before a latency alert ([#343](https://github.com/Algorithmiq/monoprop/pull/343))
- **ci:** Fix release-asset upload job ([#364](https://github.com/Algorithmiq/monoprop/pull/364))
- **docs:** Await `llms().index()` for fumadocs-core 16.15.11 ([#368](https://github.com/Algorithmiq/monoprop/pull/368))
- **python:** Add skip_validation opt-out to PauliOperator/MajoranaOperator ([#371](https://github.com/Algorithmiq/monoprop/pull/371))

### Performance

- **mpi:** GF(2)-linear rank routing — messages per rank flat in R ([#296](https://github.com/Algorithmiq/monoprop/pull/296))

### Documentation

- Rewrite the agent guide around what an agent needs ([#322](https://github.com/Algorithmiq/monoprop/pull/322))
- **benches:** Benchmarking ladder for sensitive PRs ([#317](https://github.com/Algorithmiq/monoprop/pull/317))
- Update contribution guidelines in how-to-contribute.mdx ([#349](https://github.com/Algorithmiq/monoprop/pull/349))

## 0.9.1 (2026-09-17)

### Breaking changes

- Fix TermIndex at 32 bits and check the append against the ceiling ([#325](https://github.com/Algorithmiq/monoprop/pull/325))

### Features

- **nix:** Add a flake with package, app and dev shell ([#241](https://github.com/Algorithmiq/monoprop/pull/241))

### Fixes

- Make mpi init and finalize visible in dso ([#313](https://github.com/Algorithmiq/monoprop/pull/313))
- **ci:** Let the C++ MPI test variants oversubscribe under OpenMPI 5 ([#324](https://github.com/Algorithmiq/monoprop/pull/324))
- Cstddef include before boost/test ([#331](https://github.com/Algorithmiq/monoprop/pull/331))

### Performance

- **evolution:** Carry cross-rank queries as a width-adaptive record ([#263](https://github.com/Algorithmiq/monoprop/pull/263))

## 0.9.0 (2026-08-27)

### Breaking changes

- **sonar:** Burn down the SonarQube backlog and unblock the quality gate ([#209](https://github.com/Algorithmiq/monoprop/pull/209))

### Features

- **benches:** Add a lattice-size scaling benchmark for Pauli propagation ([#172](https://github.com/Algorithmiq/monoprop/pull/172))
- **benches:** Track benchmarks with Bencher ([#223](https://github.com/Algorithmiq/monoprop/pull/223))
- **partition:** Report the launcher's placement, and delete the pinning knob ([#269](https://github.com/Algorithmiq/monoprop/pull/269))
- **docs:** Add Figure/Caption component for MDX figures ([#284](https://github.com/Algorithmiq/monoprop/pull/284))
- **docs:** Add Umami analytics tracking script ([#285](https://github.com/Algorithmiq/monoprop/pull/285))
- **partition:** Let derived propagators build correctly-typed partition children ([#265](https://github.com/Algorithmiq/monoprop/pull/265))

### Fixes

- Links to benchmarks in README.md ([#178](https://github.com/Algorithmiq/monoprop/pull/178))
- Leftovers from c++ file restructuring ([#202](https://github.com/Algorithmiq/monoprop/pull/202))
- **benches:** Measure peak memory with the kernel high-water mark ([#205](https://github.com/Algorithmiq/monoprop/pull/205))
- Make number of qubits obligatory  ([#162](https://github.com/Algorithmiq/monoprop/pull/162))
- Missing num_modes argument in hubbard bench ([#212](https://github.com/Algorithmiq/monoprop/pull/212))
- Use `std::optional` for `only_rotate_len_k` argument ([#210](https://github.com/Algorithmiq/monoprop/pull/210))
- Documentation links in README.md ([#221](https://github.com/Algorithmiq/monoprop/pull/221))
- **cmake:** Enable_testing() before add_subdirectory(cpp) for root ctest ([#224](https://github.com/Algorithmiq/monoprop/pull/224))
- **propagator:** Invalidate functionals after `update_initial_operator` ([#225](https://github.com/Algorithmiq/monoprop/pull/225))
- Env-vars in test-mpi recipe in justfile ([#228](https://github.com/Algorithmiq/monoprop/pull/228))
- **bench-tools:** Ship a py.typed marker ([#245](https://github.com/Algorithmiq/monoprop/pull/245))
- **docs:** Make the README quick-start examples runnable ([#222](https://github.com/Algorithmiq/monoprop/pull/222))
- **partition:** Place partitions inside the cgroup the batch system carved ([#249](https://github.com/Algorithmiq/monoprop/pull/249))
- **build:** Anchor uv cache-keys to the source tree ([#272](https://github.com/Algorithmiq/monoprop/pull/272))
- **ci:** Correct the secrets context in the Bencher token
- **evolution:** Bound the epoch-stamp mark by combined_size ([#267](https://github.com/Algorithmiq/monoprop/pull/267))
- Update sdist.include to include license-header file ([#282](https://github.com/Algorithmiq/monoprop/pull/282))
- **partition:** Collapse identical COMMPLACE repeats ([#286](https://github.com/Algorithmiq/monoprop/pull/286))
- **gradient:** A stable gradient at the same time and memory ([#288](https://github.com/Algorithmiq/monoprop/pull/288))
- **build:** Export detail/** free functions reachable across the shared-library boundary ([#290](https://github.com/Algorithmiq/monoprop/pull/290))

### Performance

- **operator:** Release the init_op_map buckets once its terms bind ([#268](https://github.com/Algorithmiq/monoprop/pull/268))
- **evolution:** Halve the follower-marking epoch stamp ([#259](https://github.com/Algorithmiq/monoprop/pull/259))
- **graph,mpi:** Derive the exchange layouts and store only occupied world slots ([#270](https://github.com/Algorithmiq/monoprop/pull/270))

### Documentation

- **ci:** Include README snippets in doctest-docs ([#247](https://github.com/Algorithmiq/monoprop/pull/247))
- Add links to API classes in prose docs ([#256](https://github.com/Algorithmiq/monoprop/pull/256))
- **benchmarks:** Restructure the benchmarks page and refresh every figure ([#283](https://github.com/Algorithmiq/monoprop/pull/283))

## 0.8.0 (2026-07-29)

### Breaking changes

- MajoranaPropagator/PauliPropagator interface ([#40](https://github.com/Algorithmiq/monoprop/pull/40))
- **evolution:** Massive perf update

### Features

- Because your operators deserve to propagate at escape velocity
- Add agent instructions and reference to `AGENTS.md` ([#1](https://github.com/Algorithmiq/monoprop/pull/1))
- Memory and time benchmarks with pytest ([#32](https://github.com/Algorithmiq/monoprop/pull/32))
- __eq__ implemented for operator classes ([#103](https://github.com/Algorithmiq/monoprop/pull/103))
- `to_qiskit_operator` returns now local gates ([#96](https://github.com/Algorithmiq/monoprop/pull/96))
- Truncation of terms in ExpGate.__init__ ([#114](https://github.com/Algorithmiq/monoprop/pull/114))
- Benchmark against external Pauli/Majorana Propagation implementations ([#46](https://github.com/Algorithmiq/monoprop/pull/46))
- Shorter names for expval and grad-oriented methods ([#139](https://github.com/Algorithmiq/monoprop/pull/139))
- Method for checking commutativity for operators ([#150](https://github.com/Algorithmiq/monoprop/pull/150))
- More robust handling of unsorted majorana ([#140](https://github.com/Algorithmiq/monoprop/pull/140))
- Propagators use operators from now ([#117](https://github.com/Algorithmiq/monoprop/pull/117))

### Fixes

- Correct path for c++ tests in cmakelists.txt
- Only print PROJECT_VERSION_FULL when available ([#30](https://github.com/Algorithmiq/monoprop/pull/30))
- Magic eq method corrected for operators ([#102](https://github.com/Algorithmiq/monoprop/pull/102))
- MajoranaOperator are now being checked for consisting of commuting terms ([#116](https://github.com/Algorithmiq/monoprop/pull/116))
- **ci:** Merge duplicate gcovr coverage lines so SonarCloud credits real coverage ([#132](https://github.com/Algorithmiq/monoprop/pull/132))
- **ci:** Exclude gcov throw/unreachable branches from coverage ([#135](https://github.com/Algorithmiq/monoprop/pull/135))
- **docs:** Use static Orama search client for static export ([#138](https://github.com/Algorithmiq/monoprop/pull/138))
- Links in pull request template ([#142](https://github.com/Algorithmiq/monoprop/pull/142))
- LaTeX build ([#145](https://github.com/Algorithmiq/monoprop/pull/145))
- **cmake:** Match four-digit binder batches so >=993 modes link ([#165](https://github.com/Algorithmiq/monoprop/pull/165))

### Performance

- Nitpick performance improvement ([#100](https://github.com/Algorithmiq/monoprop/pull/100))

### Documentation

- Migrate documentation from Sphinx to Fumadocs ([#39](https://github.com/Algorithmiq/monoprop/pull/39))
- **concepts:** Fix notation table rendering ([#52](https://github.com/Algorithmiq/monoprop/pull/52))
- Add pre-release warning ([#57](https://github.com/Algorithmiq/monoprop/pull/57))
- Add MDX code-block doctest checks ([#55](https://github.com/Algorithmiq/monoprop/pull/55))
- **governance:** Add open-source governance boilerplate ([#58](https://github.com/Algorithmiq/monoprop/pull/58))
- Remove duplicate warning callout in index.mdx ([#62](https://github.com/Algorithmiq/monoprop/pull/62))
- Update How to Contribute documentation ([#61](https://github.com/Algorithmiq/monoprop/pull/61))
- Review suggestions on MDX files ([#86](https://github.com/Algorithmiq/monoprop/pull/86))
- Typos and small fixes ([#112](https://github.com/Algorithmiq/monoprop/pull/112))
- Symbol unification ([#120](https://github.com/Algorithmiq/monoprop/pull/120))
- Improved docs build ([#107](https://github.com/Algorithmiq/monoprop/pull/107))
- Improvements for tutorial text and plotting  ([#126](https://github.com/Algorithmiq/monoprop/pull/126))
- Quantum chemistry tutorial improved ([#123](https://github.com/Algorithmiq/monoprop/pull/123))
- Polish documentation wording and examples ([#137](https://github.com/Algorithmiq/monoprop/pull/137))
- Point API GitHub links to Python sources ([#143](https://github.com/Algorithmiq/monoprop/pull/143))
- Mkdocstrings cross-references in docstrings for Python API, drop Sphinx syntax ([#149](https://github.com/Algorithmiq/monoprop/pull/149))
- Simplified technical paragraph ([#159](https://github.com/Algorithmiq/monoprop/pull/159))
- Sync AGENTS and README maintenance policy ([#161](https://github.com/Algorithmiq/monoprop/pull/161))
- Npm installation description ([#164](https://github.com/Algorithmiq/monoprop/pull/164))
- Better docstrings for propagators ([#163](https://github.com/Algorithmiq/monoprop/pull/163))
