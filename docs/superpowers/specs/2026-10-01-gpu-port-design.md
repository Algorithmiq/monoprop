# GPU port of monoprop: design

- **Date:** 2026-10-01
- **Status:** draft for review
- **Kind:** umbrella spec. It fixes the architecture, the decisions, the stages and their gates. Each
  sub-project (Sections 5–9) gets its own detailed spec and implementation plan before any code is
  written.

## 1. Context and goals

monoprop is a C++23/Python library for Majorana and Pauli propagation. Today it runs on CPUs only: it
splits the operator into one partition per physical core inside a process, and across processes with
MPI.

**Goal.** Run monoprop on multiple NVIDIA GPUs on HPC systems.

- **Production targets:** Leonardo Booster (4× A100 64 GB per node), MareNostrum 5 ACC (4× H100 64 GB
  per node), JUPITER Booster (4× GH200 per node). Site details are verified when each system is brought
  up in Stage 3.
- **Development platform:** the Deucalion GPU partition (4× A100 40 GB or 80 GB per node). All
  measurements in this document come from Deucalion (Appendix A).

**Principles.**

1. **Minimise duplication between the CPU and GPU paths.** Code with one correct answer per element is
   written once and compiled for both (Section 7).
2. **Reuse, don't rewrite.** The existing partition, routing and communication machinery carries over
   (Section 6). External libraries are fine where they help.
3. **NVIDIA only.** No vendor portability layer.
4. **One user until the port is done: the developer doing the port.** The GPU API is experimental and
   undocumented until Stage 3, carries no compatibility obligations, and must not spread guards for
   unsupported features through the code base (Section 5).

**Non-goals.** Portability to other GPU vendors; feature parity beyond the scope in Section 2;
streaming graphs from host memory; renumbering terms for locality; memory tiering on GH200; changes to
the CPU algorithms.

## 2. Scope: what "done" means

The GPU path supports what the benchmark ladder (`benches/LADDER.md`) exercises:

- both bases (Majorana, Pauli) and both pictures (Heisenberg, Schrödinger);
- `propagate`, `build_graph`, `expectation_value`, `expectation_value_and_gradient`, their functionals,
  and pared evaluation (`pare_threshold`);
- `lower_atol`, with length and support cutoffs;
- multi-GPU runs on one node, and correct runs on up to 4 Deucalion nodes.

Everything else raises a single, explicit error through the capability check (Section 5): `basis_change`,
`upper_atol`, `only_rotate_len_k`, in-place `contract_partially`, `update_initial_operator`,
`set_parameter_mapping`, extending a graph across several `build_graph` calls, and `evolved_operator`.
Most of these become cheap to add later because their per-element code is shared.

## 3. Background: the engine as a GPU sees it

**State per partition.** Every array is addressed by a 32-bit term index (`TermIndex`):

- **term store** (`OperatorIndex`): fixed-stride packed rows (a count, then the ascending set-bit
  positions) plus an open-addressing hash table. It only grows.
- **inverted index** (`InvertedIndex`): one bit-vector per Majorana position. The terms that
  anticommute with a generator G are the XOR of G's columns, about 0.25–0.5 bytes per term per gate.
- **coefficients**: plain `double` arrays.
- **surrogate graph**: per layer, flat lists of rotation endpoints and packed phases. The set of terms
  a layer scales by the cosine is recomputed from the inverted index, not stored.

**Three execution modes.** `build_graph` and `propagate` run `build_layer` per gate (with `GraphSink`
and `ContractSink` respectively). Replay (`evolve_step`, `state_operator_derivative_local`) runs per
layer and serves expectation values, gradients and contraction.

**Operation patterns.** The code base calls its operator sweep a "scan", but it is a traversal, not a
prefix sum:

| Step | Pattern |
|---|---|
| Fold (anticommutation), leader/follower split | map over 64-term words |
| Partner term, sign, cutoff, atol | map + filter |
| Partner lookup | map with random reads |
| Marking matched followers | scatter (each follower at most once) |
| Numbering new terms | prefix sum on the CPU; on the GPU an atomic counter (Section 9) |
| Row, hash and index inserts | scatter |
| Replay cosine pass | map |
| Replay endpoint snapshot, sine pass | gather, scatter (each target once per layer) |
| Energy, gradient | reductions |
| Layer and gate loops | sequential recurrence |

The layer loop composes linear maps, which is associative in principle, but composing layers roughly
doubles the nonzeros each time, so it stays a sequential loop of parallel kernels.

## 4. Decision log

| # | Decision | Why |
|---|---|---|
| D1 | Success is measured on one Deucalion GPU node against the CPU path on the same node's 128 cores, on the ladder rows | Clean comparison on identical CPUs and problems |
| D2 | The GPU path is opt-in per propagator (`device="cuda"`); unsupported features raise from one place | Honest coverage gaps, no hidden host/device syncing, no scattered guards |
| D3 | Shared-core refactors land on `main` as CPU PRs; GPU code lives on a branch and merges once, at the end of Stage 3 | Shared code can't drift; nobody maintains code they can't test |
| D4 | Toolchain from Deucalion modules: GCC 15.2, CUDA 13.3, NCCL 2.30.4, OpenMPI 5.0.10 | One coherent toolchain generation; nvcc accepts C++23 |
| D5 | Shared code: per-element functions and data views compiled for both; loops and output placement per backend (approach 1) | Removes duplication without forcing the tuned CPU loops through an abstraction |
| D6 | The class keeps its name: `MonomialPropagator<NumModes, Backend = backend::Cpu>`; the CUDA backend is a partial specialization in its own file | Same name everywhere; the CPU class is untouched |
| D7 | `monoprop_HOST_DEVICE` marks shared functions; no `--expt-relaxed-constexpr`; `-Werror cross-execution-space-call` | The relaxed flag let standard-library bit functions silently miscompile (Appendix A.3) |
| D8 | One GPU = one partition = one slot of the existing R×S world; R = processes, S = GPUs per process | Reuses `PartitionGroup`, the router and its two-level routing |
| D9 | Output placement is order-free (warp-aggregated atomics), except graph endpoint lists, which keep ascending source order | Idiomatic on GPUs; source order keeps replay accesses partly coalesced |
| D10 | Remote queries use the CPU's `QueryWire` format unchanged, plus a per-record length array (option A) | Same encoder/decoder on both sides; parallel decode via a scan of lengths |
| D11 | Custom device hash table with the CPU's slot layout and hash functions | Shares more code than cuCollections would |
| D12 | Overflow rows (longer than the inline width) use a spill array indexed from the row itself, on CPU and GPU alike | One layout; replaces the CPU's `unordered_map` |
| D13 | Large device arrays grow through the CUDA virtual memory API | Growth never copies, never needs two copies at once |
| D14 | NCCL is found through its own CMake config: `find_package(NCCL 2.29.7 CONFIG REQUIRED)` | NCCL ships the config since 2.29.7; no find module of our own |
| D15 | Reductions use block partials plus a final kernel, no floating-point atomics | Reproducible for a fixed term numbering |

## 5. Architecture: code structure

Four layers; each depends only on the layers below it.

**1. Python API.**

- `MajoranaPropagator` and `PauliPropagator` take an experimental `device` argument (`"cpu"` by default,
  `"cuda"` for the GPU path). It stays undocumented until Stage 3.
- `_init_simulator` passes it to `dispatch(num_modes, device)`. The generated `_dispatch.py` and
  `bindings.cpp` gain the device dimension through `tools/generate-*.py` and `binder.h`. The generated
  files are never edited by hand.

**2. `MonomialPropagator<NumModes, Backend>`.**

- `Backend` defaults to `backend::Cpu`, so every existing spelling `MonomialPropagator<NumModes>` keeps
  naming today's class.
- `MonomialPropagator<NumModes, backend::Cuda>` is a partial specialization in
  `cpp/monoprop/device/MonomialPropagatorCuda.inl`. Only CUDA translation units and `binder.h` include
  it. Where it supports a feature it uses the CPU class's method names.
- **Single choke point.** One `device_capabilities()` function validates the settings; the constructor
  and the setters call it. Methods not yet ported are bound to one stub that raises
  `DeviceUnsupported("<method>")`, mapped to a Python exception in `monoprop.exceptions`.
- **Stage 1:** the CUDA specialization contains a CPU `MonomialPropagator<NumModes>` for `build_graph`
  and owns the device replay. **Stage 2** replaces the embedded CPU build with the device build. The
  Python surface does not change.
- **Safeguards.** Naming `MonomialPropagator<N, backend::Cuda>` without having included the
  specialization fails to compile (for example through a `static_assert` in the primary template). A
  default template argument may appear only once, so one forward-declaration header declares the
  template; today's forward declaration in `PartitionGroup.h` moves there (a small refactor on `main`).
- **The backend parameter is a name, not a switch.** The CPU class never branches on it
  (`if constexpr` or otherwise). Sharing happens through the shared core only.

**3. Two engines.** The existing CPU engine, unchanged in structure, and a device engine in
`cpp/monoprop/device/` (`.cu` files: device data structures, kernels, NCCL transport).

**4. Shared core** in `cpp/monoprop/shared/` (Section 7).

**Rules.**

- The CPU engine never includes device headers and contains no `#ifdef monoprop_ENABLE_CUDA`. The only
  CUDA-conditional code is CMake, `binder.h` and the generators.
- Device widths: during development only the storage widths under test are instantiated: 128 modes
  (Hubbard with 60 sites; Pauli heavy-hex with 127 qubits) and 160 modes (the random models with 142
  modes). **When the work is done, the CUDA build instantiates all 32 widths like the CPU (32 to
  `monoprop_MAX_NUM_MODES` = 1024).** nvcc compile time at full width is measured, and mitigated if
  needed, before the branch merges.

## 6. Execution model

**Ownership.** One GPU is one partition, which is one flat slot of the existing R×S world. Each GPU
holds, in its own memory, a disjoint hash partition of the terms, that partition's coefficients and
indexes, and that partition's share of every graph layer.

**Processes and threads.**

- R is the number of MPI processes (a power of two, as today); S is the number of GPUs visible to each
  process. The CUDA specialization sets S from the visible devices.
- Each GPU is driven by one `PartitionGroup` master thread (existing machinery: lockstep execution,
  `run_on_all`, error propagation), which selects its device once.
- Each master thread is pinned to the cores of its GPU's NUMA domain, as reported by hwloc (already a
  dependency). Its host-side staging buffers are first-touched there, which `PartitionGroup` already
  does by constructing partitions on the master thread. On Deucalion only 16 cores are close to each
  GPU pair (Appendix A.4).

**The launch shape selects the routing, with no code change:**

| Launch shape | Routing | Use |
|---|---|---|
| 1 process per node, S = 4 | linear between nodes; splitmix across the node's 4 GPUs | default, bandwidth-bound problems |
| 1 process per GPU, S = 1 | linear pairwise everywhere | latency-bound, smaller problems |

The router already composes linear routing between processes with splitmix routing across the S
partitions of a process. With S = 4 GPUs per process, the flat slot `process × S + gpu` addresses a GPU,
and a gate's partners live either on the GPU's own node or on the 4 GPUs of exactly one partner node.
Section 13 gives the measured trade-off.

**Control traffic and data traffic.**

- Control: host-side scalars, agreement checks and barriers keep using the existing communicators (MPI,
  `ShmComm`, `HybridComm`).
- Data: NCCL. One communicator spans all R×S GPUs. It is created at propagator construction: a unique
  id is broadcast over the existing communicator, and each master thread calls `ncclCommInitRank` for
  its slot. Exchanges are grouped `ncclSend`/`ncclRecv` calls to the slots the routing reaches, issued
  on the GPU's compute stream.

**Stage 1: building on the CPU.**

- Start simple: the embedded CPU propagator builds with S partitions, partition g uploading to GPU g.
  Correct, but only S cores build.
- For ladder-sized problems: build with S_cpu partitions (a multiple of S, up to the core count) and
  merge partitions p ≡ g (mod S) onto GPU g at upload. Since S divides S_cpu,
  `(q mod S_cpu) mod S = q mod S`, so every term lands on the GPU a Stage 2 device build would give it.
  The merge renumbers slots and puts cross-partition rotations in an order both sides derive
  independently. It is host code, run once per upload. The CPU and GPU thread groups are never active at
  the same time, so sharing cores between them is harmless.

**Launch commands.**

- Deucalion, n ∈ {1, 2, 4}: `srun -N n --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128` (one
  process per node); `--ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32 --gpu-bind=closest` (one
  process per GPU), checking the `COMMPLACE` placement line.
- Production: the same shapes. On Leonardo (about 8 host cores per GPU) the Stage 1 CPU build is slow,
  so Stage 2 matters most there.

## 7. Shared core

**Purpose.** Code that must behave identically on CPU and GPU: per-element functions, and plain-data
views over the memory layouts both engines use. Everything else stays in its engine.

**Initial contents:**

| Piece | Comes from |
|---|---|
| `monoprop_HOST_DEVICE` macro; bit functions (popcount, parity, count-trailing-zeros, bit width) | new |
| `Bitset`/`Monomial` operations; splitmix hash mixers | `Bitset.h`, `core/Monomial.h` |
| Per-term algebra: interleave-mask parity, Hermitian phase, rotation and emit signs (Majorana, Pauli) | `algebra/*.h` |
| Length and support cutoff checks; atol checks | `AlgebraCommon.h`, `CutoffContext.h` |
| Fold-word masking; per-word XOR of dense columns | `CosineRecompute.h`, `InvertedIndex.h` |
| Partner position merge | `PartnerMerge.h` |
| Term hashing (`fold_hash`, `spread`) | `OperatorIndex.h` |
| Routing arithmetic (`dest`, `dest_from_shift`, `rank_shift`) as a plain-data router; basis table copied to the device | `mpi/Routing.h` |
| `QueryWire` record length, encode, decode | `QueryWire.h` |
| Per-endpoint replay and gradient formulas; partner-index derivation and packed-phase access | `Evolution.cpp`, `MPGraphEncodingStorage.h` |
| Views: term rows, inverted index, layer endpoints, coefficient spans | new, over existing layouts |

**Rules.**

- `monoprop_HOST_DEVICE` is defined in one small header: `__host__ __device__` under nvcc
  (`__CUDACC__`), empty otherwise. It follows the repository's lowercase-prefix convention
  (`monoprop_EXPORT`, `monoprop_ENABLE_MPI`).
- Shared code uses only type-level standard-library facilities (fixed-width integers,
  `<type_traits>`) and monoprop's own helpers. Without `--expt-relaxed-constexpr`, even `std::min` or
  `std::numeric_limits<T>::max()` is a host-only call to nvcc. The helper set stays small and
  hand-written (a bit layer, a minimal span-like view). Adopting libcu++ (`cuda::std::`) would make CCCL
  a dependency of the CPU build; revisit only if the helper list keeps growing.
- Host/device selection happens inside the helpers only: `__CUDA_ARCH__` chooses CUDA intrinsics over
  `std::`, and `if consteval` covers compile-time evaluation.
- Type changes stay layout-compatible. For example, `Bitset` moves from `std::array` storage to a plain
  array; memory layout, MPI safety and `memcpy` safety are unchanged. Host-only extras such as
  `operator<<` stay outside the shared part.
- Shared code moves under `cpp/monoprop/shared/` and existing headers include it, so include paths
  elsewhere barely change.

**Landing on `main`.**

- Each shared-core refactor is an ordinary CPU PR: it passes the existing tests and shows no regression
  beyond noise on the Bencher-tracked benchmarks. In CPU builds the macro expands to nothing, so the
  machine code should be essentially unchanged.
- A CI job on `main` compiles the shared headers with nvcc, with `-Werror cross-execution-space-call`
  and without the relaxed-constexpr flag. It runs on the GitHub-hosted T4 runner `gpu-t4-4-core-custom`,
  whose custom image `cuda-openmpi` (built by `Algorithmiq/runner-images`) moves from CUDA 12.9 to 13.3
  to match the development toolchain. Without the job, a CPU change could break the device build
  unnoticed. This is the only CUDA-related change to `main`'s CI.

**Tests.** Every shared function has a host-vs-device equivalence test on the same input vectors
(Section 11).

## 8. Device data and replay (Stage 1)

**What each GPU holds,** uploaded once from the CPU propagator through the shared views:

- **Coefficients:** `op`; the sparse state (rows and values) in the Heisenberg picture, or the dense
  state in the Schrödinger picture.
- **Inverted index:** dense columns as full-height `uint64` words; sparse columns as row lists
  (unordered on the device, since the fold scatters them with `atomicXor`); row-parity words, built only
  if an odd-weight generator appears.
- **Graph,** all layers concatenated into flat arrays: per-layer records (generator columns,
  `scaled_count`, parameter index, generator coefficient, offsets), occupied-slot records, `sin_send`
  indices, packed phase bits, and the stored cosine masks of pared layers. Pared evaluation runs the
  CPU's `pare_graph` and uploads the result.
- **Memory:** stream-ordered allocation (`cudaMallocAsync`, one pool per GPU). If the problem does not
  fit in GPU memory, construction or upload raises. Streaming the graph from the host is out of scope:
  PCIe (about 25 GB/s) is roughly 60× slower than GPU memory.

**Forward step per layer** (energy and contraction):

1. **Snapshot:** `snap[k] = op[B[k]]` for every endpoint, before the cosine pass.
2. **Exchange:** grouped `ncclSend`/`ncclRecv` of the snapshot slices for remote slots. Sizes come from
   the layer's records; there is no count exchange.
3. **Cosine pass:** the layer's fold is computed into a word buffer up to `scaled_count` (dense columns
   XORed per word; sparse columns scattered with `atomicXor`; parity and last-word masks applied). Then
   one thread per term tests its bit and scales `op[i] *= cos`. Pared layers use their stored masks.
4. **Sine pass:** `op[D[k]] += sin · φ[k] · value[k]`, where `value` comes from the snapshot for the
   GPU's own slot and from the received slices otherwise. Each target is written once: no atomics.

The energy is then the dot product of the sparse state with `op`.

**Gradient.** The forward pass records values for layers whose cosine nearly vanishes, as the CPU
does. The reverse pass, per layer: snapshot the remote endpoints; exchange two values per endpoint; run
the cosine accumulate pass (sum, then scale); apply the GPU's own rotation pairs, one thread per pair,
reading both endpoints before writing either; apply the remote endpoints. Per-layer gradient terms
accumulate into a device array indexed by parameter.

**Reductions** use block partial sums and a final reduction kernel, with no floating-point atomics
(D15). Final scalars and the gradient vector are combined across GPUs with `ncclAllReduce`.

**Launch overhead.** Energy and contraction capture the layer loop, NCCL calls included, as a CUDA Graph
once per functional; each call uploads the angles and relaunches the graph. The gradient starts with
per-layer launches; graph capture follows once the parameter-dependent choice of recorded layers moves
into device-side flags.

**Functionals** keep the CPU semantics: bound to the graph and operator state at creation, invalid
once either changes.

**Memory access patterns.** The kernels have no control-flow divergence beyond single-instruction
predication; indirect indexing affects coalescing, not divergence.

- **Cosine pass:** one thread per term reads its fold word (all lanes of a warp share one or two words)
  and scales a contiguous `op[i]`: fully coalesced. A thread-per-word loop over set bits would diverge,
  which is why the design uses thread-per-term.
- **Endpoint lists:** in each slot, `B` holds the GPU's own queries in ascending source order (partly
  coalesced) and the resolved partners in hash-table order (scattered, except newly inserted terms,
  which are contiguous). Scattered 8-byte accesses use at most a quarter of each 32-byte sector; GPUs
  sustain far more outstanding requests than CPUs, so random-access throughput still exceeds a CPU
  node's.
- Stage 2 keeps ascending source order for stored endpoint lists (D9, Section 9).
- Renumbering terms to bring partners closer is out of scope: it would break the append-only numbering
  that the `scaled_count` recompute relies on.
- Stage 0 measures achieved bandwidth per kernel against peak, and sectors per request and L2 hit rate
  (Nsight Compute), to see whether partner-side gathers dominate.

## 9. Device build and propagate (Stage 2)

**Device-resident operator.** Same layouts as the CPU, read through the shared views:

- **Term rows:** fixed-stride packed rows exactly like `OperatorIndex`: a count, then positions; the
  stride comes from the cutoff bound, with 8-bit positions up to 256 Majorana indices and 16-bit above.
  Fully paired terms are kept whatever their length, so rows longer than the inline width occur. They go
  to a spill array of dense bitsets, with the spill index written into the row itself (D12). The CPU
  adopts the same layout on `main`, replacing its `unordered_map`.
- **Hash index:** a custom open-addressing table copying the CPU's: 8-byte `{TermIndex, h32}` slots,
  linear probing, load factor at most 0.7, the shared `fold_hash`/`spread` and row comparison. Inserts
  are single 64-bit `atomicCAS` operations and never need deduplication (all new terms of a gate are
  pairwise distinct). cuCollections is the fallback.
- **Inverted index:** dense columns plus unordered sparse lists, appended per gate: one thread per new
  64-row word for dense columns, `atomicOr` only on the word shared with pre-existing rows; atomic
  appends for sparse lists. A sparse column is rebuilt as dense once its density reaches 1/64.
- **Growth:** the large arrays (rows, coefficients, index columns) reserve address space up front and
  map physical memory as they grow (D13). Small arrays reallocate normally.
- **Initial operators:** the Heisenberg initial operator and the Schrödinger paired basis are built on
  the host, as today, and uploaded.

**Per-gate pipeline:** the device version of `build_layer`, with the same contract as `Engine.h`. As on
the CPU, a sink policy decides what happens to each resolved rotation.

1. **Host:** generator columns, pivot and routing shift (giving the peer slots), sign mask, cos/sin,
   cutoff and atol parameters, packed into one small per-gate parameter block.
2. **Sweep kernel, one thread per term:** fold bit, follower bit, atol gate; partner positions via the
   shared merge; cutoff, sign and owner slot; query emission.
3. **Emission order by sink:** graph building emits in ascending source order (ordered compaction:
   device-wide scans of per-term record counts and word counts per destination); fused propagate emits
   order-free (warp-aggregated appends).
4. **Resolution:**
   - Local queries: batch lookup in the local table.
   - Remote queries, wire format (D10): the stream is the CPU's `QueryWire` format unchanged
     (word-aligned gap-coded records; the fused form keeps its trailing value word). The device adds a
     side array of one `uint16_t` word count per record, including the value word in the fused form
     (the longest possible record, at 2048 positions, is about 350 words).
   - Order-free sender: each emitting lane computes its record length; a warp-level exclusive scan
     gives offsets within the warp's segment; one 64-bit `atomicAdd` per warp reserves
     `(records << 32) | words` for the destination slot; lanes encode with the shared encoder and write
     their lengths. The ordered sender gets the same offsets from the device-wide scans.
   - Exchange: an NCCL swap of the per-slot record and word counts (host readback to size the receive
     buffers), then of the payload and the lengths.
   - Receiver: a CUB exclusive scan of the lengths gives each record's offset; one thread per record
     decodes with the shared decoder, looks it up, and answers positionally. Answers are fixed-size
     (`TermIndex` for graphs, the partner's value for propagate), so the reply needs no count exchange.
5. **Leader pass, then follower pass.** Leader hits set bits in a matched bitmap; the follower pass
   skips matched followers. Leader-pass inserts always have the pivot bit set and follower queries
   never do, so new terms, including those arriving from the peer, can be inserted immediately.
6. **New terms:** numbered by one atomic counter starting at `base`, so each gate's new terms fill one
   contiguous block `[base, base + n)`; rows written into capacity reserved before the gate; hash
   inserts; inverted index and coefficients extended at the end of the gate.
7. **Finish:** graph building appends the layer's endpoint arrays (its own queries in source order,
   resolved partners in the peer's order); propagate runs the fused apply kernel, where each slot is
   written once.

**Invariants** (they make order-free placement safe):

- each gate's new terms fill one contiguous index block;
- a query's payload, source index and value share one slot;
- both endpoints of a rotation are emitted together;
- the new terms of a gate are pairwise distinct.

**Host synchronisations per gate:** about two (remote counts; the new-term count to size the next
launches). Section 13 sets them against the measured latencies.

**Device-built graphs** are downloaded for pared evaluation (CPU `pare_graph`, then upload) and for
diagnostics. Porting `pare_graph` comes later.

## 10. Stages, gates and measurement

**Measurement protocol (D1).**

- One Deucalion GPU node, `--exclusive`, the same problem on both sides.
- CPU baseline: the node's own 128 cores (2× EPYC 7742) with `monoprop_PARTITIONS=128`, built with the
  same GCC 15.2 stack. The LADDER example numbers were measured on Deucalion x86 nodes, which have the
  same CPUs.
- GPU run: the node's 4 A100s, one process per node, S = 4.
- Problems: the L2a ladder rows (Hubbard and Pauli `propagate`; random Heisenberg `propagate`,
  `build_graph`, energy, gradient; random Schrödinger `build_graph`, energy, gradient) plus the pared
  gradient row from L1. Flags are calibrated so each row fits an a100-80 node and are recorded; the CPU
  runs identical flags.
- Harness: the existing bench harness with a device option, so both sides are timed by the same code.
  Results are stored under labels in `benches/results/`.

| Stage | Scope | Gate | If the gate fails |
|---|---|---|---|
| **0: toolchain and spike** (spike code is throwaway) | CPU build and tests pass with the GCC 15.2 stack, MPI included; nvcc rejects device calls to host-only functions under D7's flags; `graph_data()` bound in Python and forward replay on one GPU (CuPy if a CUDA 13 build installs cleanly, otherwise a standalone `.cu` reading an exported graph); Nsight metrics; the open measurements of Section 15 | **G0:** forward replay on one GPU at least 2× faster than the whole 128-core node on the same problem | Stop and reassess with the data |
| **1: device replay** | Shared-core pieces replay needs; the CUDA specialization and bindings; NCCL through `PartitionGroup`; upload (S partitions first, the S_cpu→S merge later); energy, gradient, functionals, pared evaluation; CUDA Graph for energy; replay correct on 2 and 4 nodes | **G1:** energy and gradient on 4 GPUs at least 5× faster than the 128 cores on the Heisenberg and Schrödinger rows | Optimise first (gathers, exchange, launch shape, graphs), then decide with data |
| **2: device build and propagate** | Section 9; correct on 2 and 4 nodes | **G2:** `propagate` and `build_graph` on 4 GPUs no slower than the 128 cores on every row | Ship Stage 1's hybrid (CPU build, GPU replay); replay is where variational loops spend their time |
| **3: completion** | All 32 widths, compile time measured and mitigated; `sm_90` builds; production toolchains and first runs on a production system; documentation and a public `device=` argument; merge | **Done:** Section 2 complete; the test suite passes on Deucalion; G1 and G2 hold; multi-node runs are correct | n/a |

**Branch policy (D3).** The GPU branch is rebased onto `main` regularly and merges once, at the end of
Stage 3. Shared-core PRs land on `main` as they are ready.

## 11. Testing and verification

1. **Shared core: host-vs-device equivalence.** Each shared function runs on the same input vectors on
   the host and in a kernel. Integer and bit results match exactly; floating-point results agree within
   a few ulp (FMA contraction may differ). The host half runs in the existing Boost.Test suite on
   `main`; the device half is a CUDA test target on the GPU branch.
2. **Device components against their CPU counterparts:** the hash table returns the same hits and
   misses as `OperatorIndex` on the same rows; fold words after appends are bit-identical to the CPU's
   `InvertedIndex`; `QueryWire` records encoded on the CPU decode on the device and vice versa, with
   round trips at every width in use; replaying a merged (S_cpu→S) graph matches CPU replay.
3. **GPU against CPU on the same problem:**
   - Replay: energies, gradients and contracted coefficients agree within a relative and absolute
     tolerance defined once in the test utilities.
   - Build and propagate: compare by monomial, never by index (numbering is order-free). Without
     coefficient-dependent truncation, each layer's term set is exactly equal; rotations compare as sets
     of (source monomial, target monomial, phase) triples and cosine sets as sets of monomials. With
     `lower_atol`, coefficients agree per monomial within the tolerance, and term sets may differ only
     on borderline terms whose |sin · coeff| lies within a small relative epsilon of `lower_atol`.
   - The Python suite runs on the GPU: fixtures are parametrized over `device` for supported features,
     reusing the physics checks in `tests/`. One fixture reads the capability list and skips tests that
     need unsupported features.
4. **Multi-GPU and multi-node:** the MPI tests (`test_monoprop_mpi.py`, the CTest MPI matrix) run in four
   shapes (one node with S = 4; four processes with S = 1; two nodes; four nodes). Results match a
   single-GPU run within the tolerance; without `lower_atol`, term counts match exactly.
5. **Memory and races:** `compute-sanitizer` (memcheck, racecheck, initcheck) over the CUDA test target.
6. **Performance:** the Section 10 rows through the bench harness. CI's only GPU is a single T4, which says
   nothing about A100 or H100 performance, so the GPU rows are not tracked by Bencher.

| Where | What runs |
|---|---|
| `main` CI | CPU tests (unchanged); host half of the equivalence tests; nvcc compile check of the shared headers, on the T4 runner |
| GPU branch, on Deucalion | a `just test-cuda` recipe plus a batch script: build with the Section 12 modules, the CUDA test target, the Python suite with `device="cuda"`, the MPI shapes, the sanitizers; `prek run --all-files` before every push |

`docs/content/docs/testing.mdx` gets the GPU recipe in Stage 3.

## 12. Toolchain, build and environment

**Deucalion (development).**

- Modules: `OpenMPI/5.0.10-GCC-15.2.0` and `NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0`, which loads
  `CUDA/13.3.0`, `UCX-CUDA` and `GDRCopy`. A single GCCcore (15.2.0) loads without conflicts. The NCCL
  module installs NCCL's CMake config (`lib64/cmake/NCCL/`, D14); the admins rebuilt it after the first
  build turned out to lack the config (Appendix A.2).
- Driver: R580 (580.167.08), natively CUDA 13.0. The 13.3 runtime works through CUDA's minor version
  compatibility, so builds embed SASS for every target, never depend on the driver compiling PTX, and
  use no API newer than CUDA 13.0.
- Jobs: nodes are shared by default, so benchmarks use `--exclusive`; 32 CPUs are billed per GPU.

**Compiler flags.** `-std=c++23`; no `--expt-relaxed-constexpr`; `-Werror cross-execution-space-call`;
an explicit `-gencode` per architecture (`sm_80` in development; `sm_80` and `sm_90` for production).

**Production toolchains** are settled per site in Stage 3 under the same constraints: nvcc accepts the
GCC ≥ 14 host compiler, and NCCL (≥ 2.29.7) matches the CUDA major version.

**Build.**

- CMake: a new option `monoprop_ENABLE_CUDA` (OFF by default), `enable_language(CUDA)`, C++23 for CUDA,
  `CMAKE_CUDA_ARCHITECTURES` defaulting to 80, `find_package(NCCL 2.29.7 CONFIG REQUIRED)` and the
  `NCCL::nccl` target (D14). hwloc, already a dependency, also supplies GPU-local core sets.
- Python: `monoprop_ENABLE_CUDA=ON uv sync …`, the same environment-switch pattern as
  `monoprop_ENABLE_MPI`. CPU-only builds are unaffected.
- One translation unit per width, emitted by the generator, so compiles run in parallel; kernels are
  templated on the word count.
- Dependencies: the CUDA toolkit (with its bundled CCCL/CUB) and NCCL. No cuCollections (D11), no RMM
  (`cudaMallocAsync` and the virtual memory API cover memory management).

## 13. Performance model

**Replay, per layer and GPU:**

    t ≈ max(memory bytes ÷ memory bandwidth, exchanged bytes ÷ link bandwidth) + rounds × latency

plus launch overhead unless the loop is captured in a CUDA Graph.

- Memory bytes: the fold reads (the generator's dense columns, n/8 bytes each), 16 bytes per
  anticommuting term, and the endpoint gathers and scatters (up to 4× extra for scattered accesses).
- Exchanged bytes: 8 bytes per remote endpoint for energy, 16 bytes for the gradient.
- Link bandwidth per GPU ≈ link bundles used at once × per-bundle bandwidth × efficiency.

**Measured on Deucalion** (Appendix A):

| Path | Peak per direction | Measured per GPU | Efficiency | Small-message latency |
|---|---|---|---|---|
| Pairwise, one NVLink bundle (4 links) | 100 GB/s | ~78 GB/s | ~78% | ~13–15 µs |
| All-to-all within a node, three bundles | 300 GB/s | ~180 GB/s | ~60% | ~15 µs |
| Pairwise between nodes, all 4 GPUs sending (2 NICs per node) | ~12.5 GB/s | ~11.7 GB/s | ~94% | ~37–40 µs |
| Two-level between nodes: every GPU to all 4 GPUs of the partner node | ~12.5 GB/s | ~11.0 GB/s | ~88% | ~51–64 µs |
| Between nodes, a single GPU pair | 25 GB/s | ~18 GB/s | ~73% | ~23 µs |

GPU memory bandwidth uses NVIDIA's peak figures (1,555 GB/s for A100 40 GB, 2,039 GB/s for A100 80 GB);
it has not been measured. The node's DDR4 peaks at about 410 GB/s (2 sockets × 8 channels of
DDR4-3200).

**Why all-to-all beats pairwise within a node (confirmed).** The four GPUs are fully connected with
4-link NVLink bundles (`NV4` between every pair, 12 links at 25 GB/s per GPU). A pairwise exchange can
use only the bundle to its partner; all-to-all uses all three. NCCL gives every peer the same 8 p2p
channels in both patterns (transport `P2P/CUMEM/read`), so the channel budget per peer is not what
differs. Why each peer reaches only ~60% of its bundle in all-to-all, against ~78% pairwise, is not yet
explained (Section 15).

**Choosing the launch shape.** Pairwise routing sends everything to one partner on 3 gates out of 4 and
nothing on the fourth; splitmix within a node sends 3/4 of the traffic on every gate, split across three
partners. Below about 1 MB exchanged per GPU per layer (about 130k remote endpoints at 8 bytes),
pairwise is as fast or faster; from a few MB upwards, all-to-all wins by up to about 2.3×. Ladder-sized
problems are in the second regime.

Between nodes the two shapes are nearly equal. Both send the same fraction (1 − 1/N on N nodes) of the
gates across the network, and both are bound by the node's two NICs: two-level routing measured about
6% less bandwidth (~11.0 against ~11.7 GB/s per GPU) and 15–25 µs more small-message latency than
pairwise. That is small next to the 2.3× gain within a node, so one process per node with S = 4 stays
the default.

**What the model predicts.** On layers that exchange a lot, the link sets the pace. The node-level
replay advantage lands somewhere between about 3× and the 15–20× memory-bandwidth ratio, depending on the
fraction of rotations that cross GPUs. Stage 0 measures that fraction from CPU graphs: the rotation
count from `graph_size()` against `size()`. Multi-node replay is network-bound: about 11–12 GB/s per GPU
between nodes against ~1.5 TB/s of GPU memory. A Deucalion GPU node still has about 4× the network
bandwidth of an x86 CPU node (2× HDR200 against 1× HDR100).

**Build and propagate.** Each gate whose partners live on another GPU costs about 4–6 NCCL rounds plus
about two host syncs: roughly 0.1–0.2 ms. The CPU node takes about 50 ms per gate at 1B terms, so this is
negligible at ladder size and dominant only for small operators.

## 14. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Exchange bandwidth caps multi-GPU replay (G1) | Two-level routing (S = 4) by default; fewest GPUs that fit; overlap the exchange with the cosine pass; measure the rotation fraction in Stage 0 |
| 2 | Scattered partner-side gathers limit kernel efficiency | Source-ordered endpoint lists; Nsight metrics from Stage 0 |
| 3 | Silent device miscompiles in shared code | No relaxed-constexpr flag; `-Werror cross-execution-space-call`; equivalence tests; nvcc compile check on `main` |
| 4 | GPU memory capacity (40/80 GB on Deucalion, 64 GB on Leonardo and MareNostrum 5) | Rows sized for a100-80; virtual-memory growth avoids 2.5× peaks; production budgets against 64 GB |
| 5 | Per-gate latency for small operators | Accepted; outside the gate rows |
| 6 | Driver/toolkit mismatch (R580 driver, 13.3 toolkit) | SASS-only builds; no newer APIs; recheck when a site updates its driver |
| 7 | Compile time at 32 widths | One translation unit per width; measured before the merge |
| 8 | CPU regressions from shared-core refactors | Bencher tracking; no regression beyond noise per PR |
| 9 | Leonardo's ~8 host cores per GPU make the Stage 1 CPU build slow | Stage 2 is required for Leonardo production |
| 10 | FMA flips borderline atol decisions between CPU and GPU | The borderline-term rule in the tests (Section 11) |

## 15. Open items

All belong to Stage 0 unless stated otherwise.

1. **All-to-all efficiency (optional):** rerun `xchg win 0` with a larger maximum size (4 GiB) to see
   where it levels off, and with `NCCL_P2P_READ_ENABLE=0` (an internal NCCL switch to write mode).
2. **`xor 2` repeat:** a single run measured ~72 GB/s against ~78 GB/s for the other pairings; repeat to
   tell noise from a link-level difference.
3. **Rotation fraction** of the ladder circuits, from CPU graphs.
4. **CPU build and tests with GCC 15.2,** MPI variant included; fixes land on `main`.
5. **Safety-net check:** nvcc must reject device calls to host-only functions under D7's flags. If it
   does not, the equivalence tests are the only safety net and Section 7 says so.
6. **Bug report (optional):** the `std::popcount` miscompile in device code (Appendix A.3), for NVIDIA.
7. **Production toolchains** per site (Stage 3).

## Appendix A: Deucalion data

### A.1 Hardware

GPU partition, from the [Deucalion architecture page](https://docs.deucalion.macc.fccn.pt/deucalion/),
the [GPU jobs page](https://docs.deucalion.macc.fccn.pt/jobs/gpu/) and the
[jobs overview](https://docs.deucalion.macc.fccn.pt/jobs/):

| | a100-40 | a100-80 |
|---|---|---|
| Nodes | 17 | 16 |
| GPUs | 4× NVIDIA A100 40 GB (SXM4, NVLink) | 4× NVIDIA A100 80 GB (NVLink) |
| Host | 2× AMD EPYC 7742 (128 cores), 512 GB DDR4 | same |
| Network | 2× NVIDIA ConnectX-6 200 Gb/s | same |
| Partitions | `dev-a100-40`: 1 node, 4 h; `normal-a100-40`: 4 nodes, 48 h | `dev-a100-80`: 1 node, 4 h; `normal-a100-80`: 4 nodes, 48 h |

For comparison, the x86 partition (where the LADDER CPU numbers were measured) has 500 nodes with the
same 2× EPYC 7742, 256 GB DDR4 and one ConnectX-6 100 Gb/s. GPU nodes are shared by default; 32 CPUs are
billed per GPU.

### A.2 Software environment

```text
$ module list   # after: module load OpenMPI/5.0.10-GCC-15.2.0 NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0
  1) GCCcore/15.2.0                     8) hwloc/2.13.0-GCCcore-15.2.0     15) UCC/1.7.0-GCCcore-15.2.0
  2) zlib/2.3.2-GCCcore-15.2.0          9) OpenSSL/3                       16) OpenMPI/5.0.10-GCC-15.2.0
  3) binutils/2.45-GCCcore-15.2.0      10) libevent/2.1.12-GCCcore-15.2.0  17) CUDA/13.3.0
  4) GCC/15.2.0                        11) UCX/1.20.0-GCCcore-15.2.0       18) GDRCopy/2.5.2-GCCcore-15.2.0
  5) numactl/2.0.19-GCCcore-15.2.0     12) libfabric/2.5.0-GCCcore-15.2.0  19) UCX-CUDA/1.20.0-GCCcore-15.2.0-CUDA-13.3.0
  6) libxml2/2.15.1-GCCcore-15.2.0     13) PMIx/6.1.0-GCCcore-15.2.0       20) NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0
  7) libpciaccess/0.19-GCCcore-15.2.0  14) PRRTE/4.1.0-GCCcore-15.2.0
```

Other modules available at the time: GCC 14.2.0, 14.3.0, 15.2.0; CUDA 12.9.1, 13.2.0, 13.3.0; NCCL for
CUDA 13.3 only with GCCcore 15.2.0 (the GCCcore 14.3.0 build of NCCL targets CUDA 12.9.1); OpenMPI
5.0.8 for GCC 14.3.0 and 5.0.10 for GCC 15.2.0.

Host compiler support, from NVIDIA's installation guides: CUDA 12.9 supports GCC 6–14 and C++ up to
C++20; CUDA 13.2 supports GCC 6–15 and C++20; CUDA 13.3 supports GCC 6–15 and C++23.

The toolkit checked with GCC 14.3.0 loaded:

```text
$ nvcc --version
Cuda compilation tools, release 13.3, V13.3.33
Build cuda_13.3.r13.3/compiler.37862127_0
$ grep -n "unsupported GNU version" .../include/crt/host_config.h
137:#error -- unsupported GNU version! gcc versions later than 15 are not supported! ...
$ nvcc --help | grep -A6 -- '--std'
--std {c++03|c++11|c++14|c++17|c++20|c++23}         (-std)
        Allowed values for this option:  'c++03','c++11','c++14','c++17','c++20','c++23'.
```

Driver on the GPU nodes:

```text
NVIDIA-SMI 580.167.08    Driver Version: 580.167.08    CUDA Version: 13.0
GPU 0: NVIDIA A100-SXM4-40GB
```

NCCL CMake config: the first build of `NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0` lacked it
(`ls $EBROOTNCCL/lib*/cmake/NCCL/` failed with `No such file or directory` for
`/eb/x86_64/software/NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0`). On request, the admins first provided a
`-CMake` variant of the module, then rebuilt `NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0` itself with the
config, which is the module the port uses. The `-CMake` variant's listing:

```text
$ ls $EBROOTNCCL/lib*/cmake/NCCL/
/eb/x86_64/software/NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0-CMake/lib64/cmake/NCCL/:
NCCLConfig.cmake  NCCLConfigVersion.cmake  NCCLTargets.cmake  NCCLTargets-release.cmake

/eb/x86_64/software/NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0-CMake/lib/cmake/NCCL/:
NCCLConfig.cmake  NCCLConfigVersion.cmake  NCCLTargets.cmake  NCCLTargets-release.cmake
```

A minimal CMake project (`cmake_minimum_required(VERSION 3.28)`, `LANGUAGES CXX CUDA`,
`find_package(NCCL 2.29.7 CONFIG REQUIRED)`) configured successfully with the `-CMake` variant loaded.

### A.3 Compiler probes

`probe.cu`:

```cpp
#include <print>              // monoprop uses <print>, so nvcc must at least parse it
#include <cub/cub.cuh>
#include "monoprop/Bitset.h"  // can an existing header be shared as-is?

using B = monoprop::Bitset<256>;

__global__ void count_bits(const B *in, unsigned *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<unsigned>(in[i].count());  // constexpr host code, run on device
}

int main() {
    B h[2]{};
    h[0].set(3).set(200);
    h[1].set(7);
    B *d_in;
    unsigned *d_out, out[2]{};
    cudaMalloc(&d_in, sizeof h);
    cudaMalloc(&d_out, sizeof out);
    cudaMemcpy(d_in, h, sizeof h, cudaMemcpyHostToDevice);
    count_bits<<<1, 32>>>(d_in, d_out, 2);
    cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost);
    std::println("counts {} {} (expect 2 1), zero={}, err={}", out[0], out[1], 0uz,
                 cudaGetErrorString(cudaGetLastError()));
}
```

`probe2.cu`:

```cpp
#include <bit>
#include <cstdint>
#include <print>
#include "monoprop/Bitset.h"

using B = monoprop::Bitset<256>;

#define CHECK(call)                                                                              \
    do {                                                                                         \
        if (const cudaError_t e = (call); e != cudaSuccess) {                                    \
            std::println("line {}: {} -> {}", __LINE__, #call, cudaGetErrorString(e));           \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

__global__ void probe(const B *in, unsigned *out) {
    const B &b = in[threadIdx.x];
    unsigned *o = out + 4 * threadIdx.x;
    o[0] = 42;                                                                        // did it run?
    o[1] = static_cast<unsigned>(b.count());                                          // Bitset -> std::popcount
    o[2] = static_cast<unsigned>(std::popcount(b.word(0)) + std::popcount(b.word(3))); // std::popcount directly
    o[3] = static_cast<unsigned>(__popcll(b.word(0)) + __popcll(b.word(3)));           // CUDA intrinsic
}

int main() {
    int drv = 0, rt = 0;
    CHECK(cudaDriverGetVersion(&drv));
    CHECK(cudaRuntimeGetVersion(&rt));
    std::println("driver {} runtime {}", drv, rt);

    B h[2]{};
    h[0].set(3).set(200); // word 0 and word 3
    h[1].set(7);          // word 0
    B *d_in = nullptr;
    unsigned *d_out = nullptr, out[8]{};
    CHECK(cudaMalloc(&d_in, sizeof h));
    CHECK(cudaMalloc(&d_out, sizeof out));
    CHECK(cudaMemset(d_out, 0xFF, sizeof out)); // untouched slots read 4294967295, not 0
    CHECK(cudaMemcpy(d_in, h, sizeof h, cudaMemcpyHostToDevice));
    probe<<<1, 2>>>(d_in, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost));
    for (int t = 0; t < 2; ++t) {
        std::println("t{}: ran={} Bitset::count={} std::popcount={} __popcll={}  (expect 42 {} {} {})",
                     t, out[4 * t], out[4 * t + 1], out[4 * t + 2], out[4 * t + 3], 2 - t, 2 - t, 2 - t);
    }
    std::println("host Bitset::count: {} {}", h[0].count(), h[1].count());
}
```

Build commands (from the repository root):

```bash
nvcc -std=c++23 -ccbin "$(command -v g++)" -arch=sm_80 --expt-relaxed-constexpr -I cpp -o probe probe.cu
nvcc -std=c++23 -ccbin "$(command -v g++)" -gencode arch=compute_80,code=sm_80 \
     --expt-relaxed-constexpr -I cpp -o probe2 probe2.cu
cuobjdump -sass probe2 | grep -i -c popc
```

Results on an A100-SXM4-40GB, identical with GCC 14.3.0 + CUDA 13.3.0 and with the GCC 15.2.0 module
set. No compiler diagnostics were reported for either probe; the SASS of `probe2` contains 4 `POPC`
instructions.

```text
counts 0 0 (expect 2 1), zero=0, err=no error
driver 13000 runtime 13030
t0: ran=42 Bitset::count=0 std::popcount=4294967295 __popcll=2  (expect 42 2 2 2)
t1: ran=42 Bitset::count=0 std::popcount=4294967295 __popcll=1  (expect 42 1 1 1)
host Bitset::count: 2 1
```

Conclusion: kernels launch and run correctly under minor version compatibility; the CUDA intrinsic is
correct; standard-library bit functions reached through `--expt-relaxed-constexpr` silently produce
wrong values in device code (D7).

### A.4 Topology

`nvidia-smi topo -m` on a GPU node (deduplicated; the command ran once per task):

```text
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV4     NV4     NV4     PXB     SYS     32-47   2               N/A
GPU1    NV4      X      NV4     NV4     PXB     SYS     32-47   2               N/A
GPU2    NV4     NV4      X      NV4     SYS     PXB     96-111  6               N/A
GPU3    NV4     NV4     NV4      X      SYS     PXB     96-111  6               N/A
NIC0    PXB     PXB     SYS     SYS      X      SYS
NIC1    SYS     SYS     PXB     PXB     SYS      X

NIC0: mlx5_0
NIC1: mlx5_1
```

`nvidia-smi nvlink -s -i 0`: GPU 0 (A100-SXM4-40GB) has links 0–11, each at 25 GB/s.

Reading: every GPU pair is joined by a 4-link NVLink bundle; each NIC sits behind the PCIe switch of one
GPU pair; the nodes run with 4 NUMA domains per socket, and only 16 cores are local to each GPU pair.

### A.5 nccl-tests

nccl-tests 2.20.0 (b4d5bee), NCCL 2.30.4, one GPU per rank, `-b 8 -e 1G -f 4 -g 1`. Columns: size in
bytes, element count, then out-of-place time (µs), algbw and busbw (GB/s), and in-place time, algbw and
busbw.

**`sendrecv_perf`, one node** (gnx504, ranks 0–3 on GPUs 0–3; ring: each rank sends to the next):

```text
       size    count     time   algbw   busbw     time   algbw   busbw
          8        2    15.06    0.00    0.00    12.95    0.00    0.00
         32        8    13.09    0.00    0.00    12.93    0.00    0.00
        128       32    13.38    0.01    0.01    13.37    0.01    0.01
        512      128    13.39    0.04    0.04    12.94    0.04    0.04
       2048      512    13.60    0.15    0.15    13.19    0.16    0.16
       8192     2048    16.03    0.51    0.51    15.52    0.53    0.53
      32768     8192    30.09    1.09    1.09    25.06    1.31    1.31
     131072    32768    30.39    4.31    4.31    29.88    4.39    4.39
     524288   131072    33.98   15.43   15.43    33.44   15.68   15.68
    2097152   524288    65.79   31.88   31.88    63.78   32.88   32.88
    8388608  2097152   160.60   52.23   52.23   162.91   51.49   51.49
   33554432  8388608   499.70   67.15   67.15   511.60   65.59   65.59
  134217728 33554432  1792.34   74.88   74.88  1817.85   73.83   73.83
  536870912 134217728 6951.08  77.24   77.24  6934.44   77.42   77.42
# Avg bus bandwidth: 23.1517
```

**`sendrecv_perf`, two nodes, block placement** (ranks 0–3 on gnx516, 4–7 on gnx522; only two ring links
cross between nodes):

```text
       size    count     time   algbw   busbw     time   algbw   busbw
          8        2    45.22    0.00    0.00    25.11    0.00    0.00
         32        8    24.62    0.00    0.00    24.11    0.00    0.00
        128       32    23.36    0.01    0.01    23.47    0.01    0.01
        512      128    23.17    0.02    0.02    24.61    0.02    0.02
       2048      512    23.37    0.09    0.09    23.17    0.09    0.09
       8192     2048    24.23    0.34    0.34    23.38    0.35    0.35
      32768     8192    30.56    1.07    1.07    30.23    1.08    1.08
     131072    32768    42.58    3.08    3.08    41.32    3.17    3.17
     524288   131072    69.27    7.57    7.57    65.89    7.96    7.96
    2097152   524288   139.89   14.99   14.99   141.94   14.77   14.77
    8388608  2097152   471.53   17.79   17.79   470.50   17.83   17.83
   33554432  8388608  1851.68   18.12   18.12  1811.00   18.53   18.53
  134217728 33554432  7340.70   18.28   18.28  7260.82   18.49   18.49
  536870912 134217728 29387.7  18.27   18.27  29288.6   18.33   18.33
# Avg bus bandwidth: 7.15199
```

**`sendrecv_perf`, two nodes, cyclic placement** (`--distribution=cyclic`: ranks alternate between gnx513
and gnx516, so every ring link crosses nodes and both NICs are shared):

```text
       size    count     time   algbw   busbw     time   algbw   busbw
          8        2    41.65    0.00    0.00    39.31    0.00    0.00
         32        8    37.80    0.00    0.00    37.33    0.00    0.00
        128       32    37.59    0.00    0.00    37.58    0.00    0.00
        512      128    37.56    0.01    0.01    37.44    0.01    0.01
       2048      512    38.14    0.05    0.05    38.45    0.05    0.05
       8192     2048    39.23    0.21    0.21    39.16    0.21    0.21
      32768     8192    45.66    0.72    0.72    45.60    0.72    0.72
     131072    32768    64.58    2.03    2.03    64.98    2.02    2.02
     524288   131072    99.16    5.29    5.29    97.35    5.39    5.39
    2097152   524288   238.72    8.79    8.79   233.62    8.98    8.98
    8388608  2097152   766.45   10.94   10.94   791.50   10.60   10.60
   33554432  8388608  2937.78   11.42   11.42  2939.99   11.41   11.41
  134217728 33554432 11584.3    11.59   11.59  11525.7   11.65   11.65
  536870912 134217728 45835.3  11.71   11.71  45678.0   11.75   11.75
# Avg bus bandwidth: 4.48411
```

**`alltoall_perf`, one node** (gnx513, ranks 0–3 on GPUs 0–3; busbw = algbw × 3/4):

```text
       size    count     time   algbw   busbw     time   algbw   busbw
          0        0     0.37    0.00    0.00     0.36    0.00    0.00
          0        0     0.35    0.00    0.00     0.35    0.00    0.00
        128        8    15.96    0.01    0.01    16.54    0.01    0.01
        512       32    14.86    0.03    0.03    14.61    0.04    0.03
       2048      128    15.27    0.13    0.10    14.69    0.14    0.10
       8192      512    15.08    0.54    0.41    15.01    0.55    0.41
      32768     2048    18.17    1.80    1.35    17.68    1.85    1.39
     131072     8192    28.29    4.63    3.47    27.78    4.72    3.54
     524288    32768    30.99   16.92   12.69    30.36   17.27   12.95
    2097152   131072    39.03   53.73   40.30    40.16   52.22   39.17
    8388608   524288    73.60  113.98   85.49    70.63  118.77   89.07
   33554432  2097152   196.10  171.11  128.33   188.53  177.98  133.48
  134217728  8388608   634.61  211.50  158.62   613.51  218.77  164.08
  536870912 33554432  2296.86  233.74  175.31  2202.27  243.78  182.84
# Avg bus bandwidth: 44.0414
```

### A.6 `xchg`: monoprop's exchange patterns

`xchg.cpp` measures per-GPU exchange bandwidth for the two routing patterns: `xor <s>` sends to and
receives from `rank ^ s` (one process per GPU, S = 1); `win <t>` sends equal shares to every other GPU
of node `node ^ t` (one process per node, S = 4). One MPI rank per GPU, block placement.

```cpp
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { if (auto e_ = (x); e_ != cudaSuccess) { std::fprintf(stderr, "%d: %s\n", __LINE__, cudaGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define NK(x) do { if (auto r_ = (x); r_ != ncclSuccess) { std::fprintf(stderr, "%d: %s\n", __LINE__, ncclGetErrorString(r_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size, lrank, lsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    CK(cudaSetDevice(lrank));

    const bool win = argc > 1 && std::strcmp(argv[1], "win") == 0;
    const int arg = argc > 2 ? std::atoi(argv[2]) : (win ? 0 : 1);
    std::vector<int> peers;
    if (win) {
        const int peer_node = (rank / lsize) ^ arg;
        for (int g = 0; g < lsize; ++g)
            if (peer_node * lsize + g != rank) peers.push_back(peer_node * lsize + g);
    } else {
        peers.push_back(rank ^ arg);
    }

    ncclUniqueId id;
    if (rank == 0) NK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, size, id, rank));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    const size_t max_bytes = size_t{1} << 30; // bytes each GPU sends per exchange
    char *sbuf, *rbuf;
    CK(cudaMalloc(&sbuf, max_bytes));
    CK(cudaMalloc(&rbuf, max_bytes));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    auto exchange = [&](size_t share) {
        NK(ncclGroupStart());
        for (size_t k = 0; k < peers.size(); ++k) {
            NK(ncclSend(sbuf + k * share, share, ncclChar, peers[k], comm, st));
            NK(ncclRecv(rbuf + k * share, share, ncclChar, peers[k], comm, st));
        }
        NK(ncclGroupEnd());
    };

    if (rank == 0)
        std::printf("# %s %d, %d ranks, %zu peers\n#   bytes_sent_per_gpu   time_us   GB/s_per_gpu\n",
                    win ? "win" : "xor", arg, size, peers.size());
    for (size_t bytes = 8 * peers.size(); bytes <= max_bytes; bytes *= 4) {
        const size_t share = bytes / peers.size();
        for (int w = 0; w < 5; ++w) exchange(share);
        CK(cudaStreamSynchronize(st));
        MPI_Barrier(MPI_COMM_WORLD);
        const int iters = 20;
        CK(cudaEventRecord(t0, st));
        for (int i = 0; i < iters; ++i) exchange(share);
        CK(cudaEventRecord(t1, st));
        CK(cudaEventSynchronize(t1));
        float ms;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        double us = 1e3 * ms / iters, worst = 0;
        MPI_Reduce(&us, &worst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) std::printf("%20zu %10.2f %12.2f\n", share * peers.size(), worst, share * peers.size() / (worst * 1e3));
    }
    NK(ncclCommDestroy(comm));
    MPI_Finalize();
}
```

Build and run:

```bash
module load OpenMPI/5.0.10-GCC-15.2.0 NCCL/2.30.4-GCCcore-15.2.0-CUDA-13.3.0
mpicxx -std=c++20 -O2 xchg.cpp -o xchg -I"$EBROOTCUDA/include" -I"$EBROOTNCCL/include" \
       -L"$EBROOTCUDA/lib64" -L"$EBROOTNCCL/lib" -lnccl -lcudart
srun -p dev-a100-40 -A <account> -N1 --ntasks-per-node=4 --gpus=4 --cpus-per-task=32 \
     bash -c './xchg xor 1; ./xchg xor 2; ./xchg xor 3; ./xchg win 0'
srun -p normal-a100-40 -A <account> -N2 --ntasks-per-node=4 --gpus=8 --cpus-per-task=32 \
     bash -c './xchg xor 4; ./xchg win 1'
```

Single-node results (one run each):

```text
# xor 1, 4 ranks, 1 peers
#   bytes_sent_per_gpu   time_us   GB/s_per_gpu
                   8      15.51         0.00
                  32      14.49         0.00
                 128      14.95         0.01
                 512      14.49         0.04
                2048      14.44         0.14
                8192      16.13         0.51
               32768      45.62         0.72
              131072      29.13         4.50
              524288      34.05        15.40
             2097152      66.00        31.78
             8388608     161.08        52.08
            33554432     494.59        67.84
           134217728    1776.79        75.54
           536870912    6901.55        77.79
# xor 2, 4 ranks, 1 peers
                   8      14.85         0.00
                  32      12.80         0.00
                 128      12.39         0.01
                 512      12.44         0.04
                2048      12.75         0.16
                8192      29.54         0.28
               32768      24.52         1.34
              131072      30.57         4.29
              524288      35.43        14.80
             2097152      70.30        29.83
             8388608     174.18        48.16
            33554432     537.24        62.46
           134217728    1926.25        69.68
           536870912    7483.80        71.74
# xor 3, 4 ranks, 1 peers
                   8      17.46         0.00
                  32      14.59         0.00
                 128      14.64         0.01
                 512      15.56         0.03
                2048      14.80         0.14
                8192      16.33         0.50
               32768      25.55         1.28
              131072      29.54         4.44
              524288      43.06        12.18
             2097152      66.10        31.73
             8388608     162.36        51.67
            33554432     490.96        68.34
           134217728    1773.52        75.68
           536870912    6894.75        77.87
# win 0, 4 ranks, 3 peers
                  24      16.08         0.00
                  96      14.75         0.01
                 384      15.00         0.03
                1536      14.85         0.10
                6144      14.90         0.41
               24576      17.41         1.41
               98304      27.44         3.58
              393216      30.72        12.80
             1572864      39.07        40.26
             6291456      88.83        70.82
            25165824     197.89       127.17
           100663296     615.42       163.57
           402653184    2217.16       181.61
```

Two-node results (one run each; `xor 4` pairs every GPU with its counterpart on the other node, `win 1`
sends to all 4 GPUs of the other node):

```text
# xor 4, 8 ranks, 1 peers
#   bytes_sent_per_gpu   time_us   GB/s_per_gpu
                   8      40.86         0.00
                  32      57.24         0.00
                 128      38.45         0.00
                 512      38.91         0.01
                2048      39.07         0.05
                8192      39.73         0.21
               32768      45.57         0.72
              131072      65.18         2.01
              524288      97.95         5.35
             2097152     234.70         8.94
             8388608     771.23        10.88
            33554432    2956.34        11.35
           134217728   11553.33        11.62
           536870912   46017.28        11.67
# win 1, 8 ranks, 4 peers
                  32      54.32         0.00
                 128      57.50         0.00
                 512      63.59         0.01
                2048      51.30         0.04
                8192      62.00         0.13
               32768      58.62         0.56
              131072      75.06         1.75
              524288     129.28         4.06
             2097152     271.97         7.71
             8388608     829.34        10.11
            33554432    3164.88        10.60
           134217728   12159.54        11.04
           536870912   48877.72        10.98
```

`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,TUNING` on `xchg xor 1` and `xchg win 0` (node gnx530).
Condensed excerpts: the `host:pid:tid` prefix is replaced by the rank, and channel ranges are
summarised in parentheses (the original grep stopped after 20 lines per run):

```text
== xor1
[0] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 8 p2p channels per peer
[0] NCCL INFO Channel 16/1 : 0[0] -> 1[1] via P2P/CUMEM/read      (channels 16-23 for 0 -> 1)
[1] NCCL INFO Channel 24/1 : 1[1] -> 0[0] via P2P/CUMEM/read      (channels 24-31 for 1 -> 0)
[2] NCCL INFO Channel 16/1 : 2[2] -> 3[3] via P2P/CUMEM/read      (channels 16-19 shown)
== win0
[0] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 8 p2p channels per peer
[0] NCCL INFO Channel 08/1 : 0[0] -> 2[2] via P2P/CUMEM/read      (channels 08-15 for 0 -> 2)
[0] NCCL INFO Channel 16/1 : 0[0] -> 1[1] via P2P/CUMEM/read      (channels 16-23 for 0 -> 1)
[0] NCCL INFO Channel 24/1 : 0[0] -> 3[3] via P2P/CUMEM/read      (channels 24-27 shown)
```

Every rank reports the same channel budget (32 p2p channels, 8 per peer).

## Appendix B: where the repository changes

**On `main`** (ordinary CPU PRs, Stage 0 onwards):

- `cpp/monoprop/shared/`: the shared core and its host-side tests (Section 7).
- Existing headers include the shared core: `Bitset.h`, `core/Monomial.h`, `algebra/*.h`,
  `AlgebraCommon.h`, `CutoffContext.h`, `CosineRecompute.h`, `InvertedIndex.h`, `PartnerMerge.h`,
  `OperatorIndex.h`, `mpi/Routing.h`, `QueryWire.h`, `Evolution.cpp`, `MPGraphEncodingStorage.h`.
- One forward-declaration header for `MonomialPropagator` with the defaulted `Backend` parameter;
  `PartitionGroup.h` uses it.
- `OperatorIndex`: overflow rows move to the in-row spill index (D12).
- CI: the nvcc compile check of the shared headers.
- Fixes needed to build and test with GCC 15.2.

**On the GPU branch:**

- `cpp/monoprop/device/`: the device engine and `MonomialPropagatorCuda.inl`.
- CMake: `monoprop_ENABLE_CUDA`, CUDA language, NCCL through its config.
- `src/monoprop/bindings/binder.h`, `tools/generate-binders.py`, `tools/generate-dispatch.py`: the
  device dimension.
- `src/monoprop/monomial_propagator.py` and the `MajoranaPropagator`/`PauliPropagator` constructors: the
  `device` argument; `monoprop.exceptions`: the unsupported-feature exception.
- Tests: the CUDA test target, `device` parametrization of the Python fixtures, the `just test-cuda`
  recipe and batch script.
- Bench harness: the device option.
- Stage 3: `README.md` and `docs/content/docs/` (building, testing, parallelism, API).
