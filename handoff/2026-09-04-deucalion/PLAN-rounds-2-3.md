# Parallel Opus tracks: memory and time, round 2 (2026-09-03)

## Context

The redesign (PR #296 base → hash-free kernel → 1½-round protocol → join filter/fused sweep → A5 reserves)
is bit-identical everywhere and has landed as `ce500790` on `perf/one-round-exchange`, with the storage
levers rebased on top as `ce16f0f6` on `perf/storage-levers` (L2 coeff growth, L6 ledger fields,
ChunkedArray, chunked dense index, adaptive chunks, gate-buffer HWM, L1a chunked rows/keys). Track C's
agent stopped when plan mode came on: `ce16f0f6` is built, ctest 363/363, pytest 623/8, but golden, prek
and the memory pair are not yet run.

Ladder A/B of `ce500790` vs base-296 (jobs 1872491/1872492; previous bff293d6 in brackets):

| row | time | kernel peak | ledger | peak − ledger |
| --- | ---: | ---: | ---: | ---: |
| L1 Hubbard P=1, 10 M terms | 1.853 (1.968) | 0.861 (0.976) | 0.806 | 80.1 − 40.8 = 39 B/term |
| L1 Pauli P=1 | 1.610 (1.612) | 0.895 (0.946) | 0.860 | |
| L1 random Heisenberg graph | 0.976 | 0.905 | 0.900 | |
| L2a Hubbard 1×128, 1.0 B terms | 1.427 (1.456) | 0.892 (0.906) | 0.721 | 54.0 − 37.7 = 16.3 B/term |
| L2b Hubbard 8×16, 1.0 B terms | 1.481 (1.48) | 0.914 (0.91) | 0.725 | 57.4 − 37.9 = 19.4 B/term |

Ledger at L2a (B/term): rows 12 (stride 1 + 11 `uint8` positions) · inverted index 12.8 (dense 11.9 at
1 bit/row for 76 columns/partition, sparse 0.9) · coefficients 8 · keys 4 · gate scratch 1.3 = 37.7. The
storage tip removes ≈ 0.5 (row slack) + ≈ 2.3 (dense-word slack).

User request (`/plan`, 2026-09-03): spawn Opus subagents on parallelizable ideas to reduce memory and
speed things up. Standing decisions: bit-identical only, no symmetry orbits, stacked branches, one agent per
worktree, frozen `ref-<hash>` arms, ladder A/B with paired medians, build on ln04 at `-j4`, agents stop
and report on a permission block instead of routing around it.

## Where the bytes and the seconds are (from today's exploration)

**Memory.**
1. **Peak − ledger at S > 1 is 16–19 B/term = 30 % of peak, and it is copies.** A record on the wire
   exists in up to five buffers at once: `scan.queries` (one `std::vector` per destination slot, no
   reserve, `Scan.h:259`), `PendingAlltoallv::send_buffer` (a flat copy of all of them,
   `MPICompat.h:280-289`), the peer's `recv_buffer`, `incoming` (per-slot copy again, `wait_into`,
   `MPICompat.h:194-211`) and the decoded `IncomingRecords pr` (8 arrays, `Resolve.h:59-82`). Round 2
   (responses/answers) repeats the chain. Under `HybridComm` (R > 1) the payload is additionally packed
   into permanent, grow-only `stage_send_`/`stage_recv_` (`HybridComm.h:616-621`) after a separate count
   round, 4–6 barriers per round. None of the per-slot wire vectors has a release rule; only
   `BucketJoin` and `pre_cos` do. The standalone verb `mpi::pair_exchange` (`perf/pair-exchange`,
   cfd50db7, `PairExchange.h`/`PairSlots.h`) already does the right thing — in-rank: one barrier, zero
   copy via published spans; cross-rank: one hindexed message each way, no count round, no staging — but
   has no production call site.
2. **Rows (corrected 2026-09-03 by Track C's measurement).** The stride is *already* cutoff-driven:
   `MonomialPropagator.inl:174` builds the store with `packed_inline_width_()` = `CutoffEvaluator::
   max_slot_bound()` (Majorana `cutoff`, Pauli `2·cutoff`, clamped to 32; 11 only for Schrödinger).
   Hubbard c7 runs at stride 8 (verified from `operator_terms_bytes`); L2a is c10 ⇒ stride 11, which is the
   11 B/term above. What is loose is the *bound*: measured widths — Hubbard c7 P100 = 6 (even parity),
   KickedIsing c14 P50/P90/P99/P100 = 16/20/22/26 against a bound of 28. Gains: Majorana −1 B/term at odd
   cutoffs (zero at L2a's c10), Pauli ≈ −5.7 B/term with inline width at P99 plus a fixed-stride wide tier
   (the optimum is P99, not P90: the inline saving is paid by every row, the tier only by the spill).
   Decision: option (a), an upward-only, threshold-triggered restride to the observed width that preserves
   row indices (no index reset), with the wide tier at the static bound replacing `overflow_`'s 80 B/entry
   spill; `update_cutoff` rides the same path. Incidental: `kRowsPerChunk = 2^18` costs 98 B/term at 45 k
   terms — make it size-adaptive like `chunk_words_for_rows` (separate small commit first).
3. Coefficients (flat 1.5×-grown vector) and keys (4 B, `mix64` of the GF(2) fingerprint, read for every
   anti row in `stage_rows`) stay: chunking coefficients buys ≤ 1.3 B/term average slack for a chunk
   indirection in the hottest read, and recomputing keys would read 12 B of row instead of 4 B of key.
4. Dense index columns at density ≥ 1/64 (`kPromoteDensityInv`) at 1 bit/row are within 2× of entropy;
   not worth a decoder in the fold.

**Time.** Residual 1.43–1.85× is flat across layouts ⇒ kernel-bound; Ir/visit 203 vs 160. Per anti row the
kernel makes two passes (coefficient pass B, key+filter pass C) where base makes one; the passes cannot
be fused because the filter needs the incoming queries. Mechanical cuts are near exhaustion (A3/A4
rejected, A5 4.5 %). What is left with room: (a) build-level PGO — nothing in the tree uses
`-fprofile-use`, `EXTRA_CXXFLAGS` is the plumbing (`cmake/compiler_flags/CXXFlags.cmake`), zero memory
cost, both arms benefit; (b) join selectivity — the filter is ≥ 8 bits/query capped at 2²¹ bits
(`BucketJoin.h:204-206`), ≈ 8.6 % false-positive staging at 45 k queries; (c) transport at R ≥ 64, where
base's weak efficiency was 29 % at 100 M terms/node — only measurable on `normal-x86`.

## Tracks

All Opus unless stated. Each track: one worktree, one branch, commits only when every gate is green,
`Assisted-by: ClaudeCode:claude-opus-5` (Sonnet tracks: `claude-sonnet-5`), nothing pushed, parked
work under `refs/recover/*`. Every agent writes one `PROGRESS.txt` line per step under
`scratch/hashfree/<track>-logs/`, runs long steps in a kept-open ssh session on ln04, and reports the
commit hash, `_core.abi3.so` md5, and the gate tallies.

| track | what | worktree / branch | base | success gate |
| --- | --- | --- | --- | --- |
| C (continue) | finish `ce16f0f6` gates; then **R** | `worktrees/storage`, `perf/storage-levers` | ce500790 | as before |
| R | observed-width restride + wide tier (+ adaptive row chunks) | same as C (both touch `OperatorIndex.h`) | storage tip | bit-identical; Pauli rows ≈ −5.7 B/term, Hubbard c7 −1; time ≤ 1.01 |
| W | zero-copy wire: integrate `pair_exchange` for both rounds, drop `incoming`/staging copies | new `worktrees/wire`, `perf/wire-zero-copy` | storage tip (needs `d_gate_buffers_hwm_bytes`) | bit-identical; peak − ledger at L2a/L2b ≤ 8 B/term; time ≤ 1.02 at R ≤ 16 |
| S3 | join selectivity: filter bits/bucket geometry, then pass-B block processing | `worktrees/one-round`, `perf/one-round-exchange` | ce500790 | exclusive-node paired L ≤ 0.97, else park |
| S2 (Sonnet) | PGO arm: `-fprofile-generate` → S/L cells → `-fprofile-use`; bit-identity; timing | new detached `worktrees/pgo-ce500790` + `worktrees/pgo-base` | ce500790, 5ada3da3 | report only; if ≥ 5 % and 0 ULP, follow-up commit adds a `monoprop_PGO_PROFILE` option |
| A/B (Sonnet) | ladder A/B of each landed tip vs frozen refs (`ref-ce500790`, base-296) | frozen `ref-<hash>` trees | — | paired medians, agreement, identical term counts |

**Ladder A/B storage tip `ce16f0f6` vs `ref-ce500790` (jobs 1873587/1873589, 2026-09-03):** time L1 Hubbard 1.010,
Pauli 1.004, graph 0.983, **L2a 0.959**, L2b 0.993; kernel peak 0.966 / 0.953 / 0.908 / 0.963 / 0.895; ledger
0.829 / 0.865 / 0.751 / 0.926 / 0.920 (L2a 37.7 → 34.9 B/term, peak 53.4 → 51.4; L2b peak 57.8 → 51.7). Term
counts identical. Combined vs base-296 at L2a: ledger 0.668, peak 0.859. **S3 first commit `fd795841`** (join
filter with more bits, bucket split skipped): exclusive-node paired vs ref-ce500790 **L 0.9423 (10/10), S 0.9149
(10/10)**, Ir/visit 202.9 → 194.8, BucketJoin.h Ir −25.7 %, golden 0 ULP; residual vs base 1.556 → 1.464 (L),
1.434 → 1.311 (S); remaining gates in progress. **W commits `d15feb7a` (port) + `7de5a4e0` (integration)** on
disk, gates pending. All three Opus agents were cut off by the API spend limit at ~18:10 and resumed after the
reset.

**Stack `a1c122a9` (one-round fd795841 + storage + restride) vs base-296, job 1874208 (L1 3 reps, L2a 2 reps):**

| row | time | kernel peak | ledger |
| --- | ---: | ---: | ---: |
| L1 Hubbard P=1 | 1.649 | **0.793** | **0.668** |
| L1 Pauli P=1 | 1.458 | 0.821 | 0.677 |
| L1 graph | 0.954 | 0.820 | 0.676 |
| L2a 1×128 | **1.315** | 0.846 | 0.668 |

L2b pending (nodes job to submit). W tip `e3462b97` (join row side sized from previous staged count) A/B at L2b vs
ref-ce16f0f6 (job 1874210): time 1.000/1.012, summed peak 0.983 Hubbard / 0.945 Pauli — improved from 1.092 but
not yet at the ≤ 0.95 gate. Process restart at ~22:00 lost the W and A/B agents; both resumed from disk.

### C → R: storage worktree

1. Finish `ce16f0f6`: golden 0 ULP vs `golden/one-round.json` and `one-round-1p5.json`,
   `prek --from-ref ce500790`, the 9.26 M memory pair (VmHWM, `d_terms_slack_bytes`, `row_keys_bytes`).
   Then hand to the A/B agent (storage tip vs `ref-ce500790` at L1/L2a/L2b, one n1 job + one nodes job).
2. **R — row stride from the model.** `OperatorIndex<NumModes>` gets a constructor/`reset(inline_width)`
   taking the inline position count; `MPOperator.h:82` passes the value the propagator knows
   (Majorana: `cutoff`; Pauli: the P90 width from the previous plan's measurement, with the wide tier
   below). If a wider row arrives (cutoff raised between calls) the store rebuilds at the new stride —
   `overflow_` already keeps it lossless in between, so correctness never depends on the guess.
   `RowBlock`/`block_row`/`row_positions` take `stride_` from the object (they already do — confirm no
   compile-time 12 leaks). Pauli: second fixed-stride wide-row store instead of `unordered_map`
   (~80 B/entry) for rows between P90 and `2·cutoff`. Tests: stride ladder in
   `cpp/tests/operator_index_tests.cpp` (rows below/at/over the inline width, rebuild path, bit-identical
   `row_eq_positions`), ledger `operator_terms_bytes` reflects the stride. Gates + exclusive-node timing vs
   `ref-ce500790` (row reads are only 2 % of anti rows, expect ≤ 1.005). Commit
   `perf(operator): ⚡ size the inline row from the model's cutoff`.

**R commit 1 landed: `baf55d2e` `perf(operator): ⚡ size row chunks from the row count`** (parent ce16f0f6, `_core`
md5 827f58f3). Chunk length re-derived on every growth with an index-preserving `rechunk_` (the primitive the
restride reuses); KickedIsing c=8 at 45 k terms 98.0 → 18.4 B/term, 9.26 M reproducer byte-identical. Gates
green (ctest 364/365 — the one failure, `shm_comm_oversubscribed_repeated_collectives`, also fails on both
frozen refs on ln03 today: environmental), timing folded into commit 2's job. Commit 2 decisions: build at a
predicted w (Majorana `2·floor(cutoff/2)` if the initial operator is even-parity, else W; Pauli ≈ 0.79·W)
then upward-only restrides at a 3 % wide-tier trigger; wide marker 254 in `row[0]` with the tier slot packed
in `row[1..4]`; restride/wide-fraction counters for the A/B.

**R commit 2 landed: `16c97a49` `perf(operator): ⚡ restride the row store to the observed row width`** (storage tip,
`_core` md5 e7b72692). Timing job 1874016, 11 reps × 3 cells: vs ref-ce16f0f6 S 0.9937 / L 0.9875 / Pauli c14
0.9854 (11/11 faster each); vs ref-ce500790 1.0048 / 0.9971 / 0.9922 — accepted. Rows: Hubbard c7 8.15 → 7.13
B/term (width 6, 0 wide rows, 0 restrides); kicked Ising c14 29.6 → 24.6 B/term (width 23 of bound 28, 0.128 %
wide, 0 restrides); c8 45 k terms 98 → 18.4 B/term (one restride 13 → 16). Golden 0 ULP both goldens, MPI job
1874009 all suites (ctest 368/368 on the compute node — the ln03 `shm_comm_oversubscribed_repeated_collectives`
failure is confirmed environmental). New ledger counters `d_row_inline_width`, wide rows, restrides. Revisit
later: the Pauli 0.79·W constant (7/8 would avoid the c8/c12 restride at ~2 B/term cost at c14); `raise_bound`
leaves side-map rows in place until the next restride. **Integration order (revised):** one-round `fd795841` ←
storage (rebase in progress, park `refs/recover/storage-levers-pre-rebase-3`) ← wire (rebase after). Ladder
A/B for R: one n1 job (L1 Pauli is the cell that shows it; L2a/L2b are even-cutoff Hubbard, zero gain).

**Storage rebased onto fd795841: tip `a1c122a9`** (`_core` md5 8b3d724a, 11 commits; BucketJoin.h auto-merged
cleanly, verified as exactly the two hoist hunks over S3's stage_rows). Gates: ctest 369/369 on the compute node
(MPI job 1874157, pytest --with-mpi 601), pytest 623/8, golden 0 ULP both, prek. Pre-rebase tip parked
`refs/recover/storage-levers-pre-rebase-3`. Next: freeze `ref-a1c122a9` → ladder vs base-296 (stack headline
without W); W rebases onto a1c122a9 after its own gates.

### W: zero-copy wire (`worktrees/wire`, off the storage tip)

Goal: one copy per hop, nothing retained across gates that is not the live data. Steps, each measured
with `d_gate_buffers_hwm_bytes` at P=128 (L2a shape, 9.26 M reproducer at `monoprop_PARTITIONS=128`)
before and after:

1. **Breakdown first.** Extend the storage branch's stamp with the transport's own buffers
   (`send_buffer`, `recv_buffer`, `stage_send_/stage_recv_`, `incoming`, `pr`) and print the per-buffer
   widest-gate table at P=4 and P=128 and at 2×16 — the S=128 profile has never been measured.
2. **Integrate `mpi::pair_exchange`** (cherry-pick `PairExchange.h`/`PairSlots.h`/tests from cfd50db7,
   port to the current `Comm`/`HybridComm`) into `LayerBuildEngine::exchange_and_join` (`Engine.h:367-
   369` round 1, `:417-452` round 2) for the linear-routing case (`plan.sparse`, one peer rank
   `my_rank ^ shift`). `scan.queries`/`responses` become the published send spans; the receiver decodes
   `PairRecv::from[u]` straight into `pr` (no `incoming`). Lifetime contract: send buffers live until the
   next `pair_exchange` returns ⇒ move `queries`/`responses` into `GateScratch` as a double buffer (with
   the 4× release rule). Dense routing keeps `begin_alltoallv`.
3. **Per-slot vectors → one flat arena per round** with per-slot offsets (`WindowVec<VecZ>` → offsets +
   one `VecZ`), reserved from the previous gate's counts like `self_records_hint`; this also removes the
   `send_buffer` flatten copy in the alltoallv fallback.
4. Release rule for every gate-scoped buffer that survives step 2–3 (`pr`, filter, `marks`, `silent`, `nz`
   are grow-only today — `Engine.h:482-486`), same 4× rule as `BucketJoin`.

Gates: ctest serial + MPI, pytest `--with-mpi` at 2×1/2×16/4×8/16×16, golden 0 ULP, prek; memory pair at
P=1/4/128 and 2×16; ladder A/B L2a/L2b (memory target) and L3 (2 nodes, time ≤ 1.02). Commits, in order:
`feat(mpi): ✨ port pair_exchange onto the current comm layer`, `perf(engine): ⚡ exchange gate records
with pair_exchange and decode them in place`, `perf(engine): ⚡ keep one flat wire buffer per round`.
Transport time at R ≥ 64 is a separate decision (see question below).

**W step 1 result (2026-09-03, `scratch/hashfree/wire-logs/breakdown.md`, job 1873570, 9.26 M reproducer):**
bytes invisible to the ledger's `d_gate_buffers_hwm_bytes` = 9.56 B/term at S=1, 12.3 at S=4, 13.1 at
S=128, **16.7 at 2×16** — matching the 16–19 B/term peak−ledger gap. Widest stamp is round 2 of the same
gate in every multi-slot cell. What `pair_exchange` makes unnecessary (both rounds' send/recv staging,
per-slot `incoming`/`answers`, HybridComm's permanent staging pair): 4.4 B/term at S=4, 6.0 at S=128, 9.55
at 2×16 → clears the ≤ 8 B/term gate by itself. Not in the plan before: (i) BucketJoin minus its filter is
the largest line at every S (50–54 MB world, release rule keyed to `n_anti` while staged ≈ 10 %) → step 4
target; (ii) `FusedContract::halves` is flat at 25.3 MB world at S=1..128 — W re-checked: Engine.h:414 already
reserves from the slot's own tally, the flat world total is the invariant, so no fix (plan note corrected); (iii) the S=1 invisible 88 MB is the
grow-only residue (join, self, halves, pre_cos). Order: port → integration → flat arena → release rules.

**W interim A/B of `7de5a4e0` vs ref-ce16f0f6 (jobs 1873982/1873983, 5 reps, 2026-09-03):** time L1 0.996–1.002,
**L2a 0.944**, L2b 0.995; kernel peak L1 ≈ 1.00, L2a 0.993, **L2b 1.092 Hubbard / 1.043 Pauli (summed ranks)** —
the integration raised the multi-rank peak instead of lowering it. Gates on 7de5a4e0 were green (MPI job 1873980
ctest 371/371, pytest --with-mpi 601, golden 0 ULP arm-to-arm at S=4, prek). `29d225ec` (flat arena) is committed
since; step 4 (BucketJoin release) in progress. Redirect sent: re-stamp the four-cell breakdown at 2×16 with the
pair_exchange buffers included (double-buffered send side, thread_local recv, PairSlots, HybridComm staging still
alive?), fix the growth, gate on peak first (2×16 pair and L2b ≤ 0.95 vs ref-ce16f0f6), then rebase onto a1c122a9.

### S3: join selectivity (`worktrees/one-round`)

1. Instrument (COMMPROF counters already exist): staged rows / anti rows and confirms / staged at S and L.
2. Filter: ≥ 32 bits per query (cap stays 2²¹ = 256 KiB, L2-resident) and bucket geometry re-tuned to the
   smaller staged set; measure Ir/visit under callgrind (`--LL=16MiB`, `--compress-strings=no`) and paired
   exclusive-node time vs `ref-ce500790`.
3. Only if 2 lands: pass-B block processing — process 64-row words with ≥ 50 % overlap as a dense AVX2
   block (load/abs/compare/multiply on 4 doubles, scalar compaction for `pre_cos`), scalar bit-iteration
   below. Must stay bit-identical (same operations, no FMA contraction change — check `-ffp-contract` is
   already `fast` under `-march=native` for both paths).
   Gate ≤ 0.97 L each; parked under `refs/recover/join-selectivity` otherwise.

**S3 complete (2026-09-03).** Counters corrected the premise: |Q| is 15–17 % of |Anti| (not 1.7 %), 4.1 k (S) /
17.5 k (L) queries per gate, staged/anti 22 %, confirms/staged 65–68 %, hits/confirms 99.99 %. `fd795841`
`perf(engine): ⚡ buy the join filter more bits and skip the bucket split it makes pointless`: filter 32 bits/query
(0.962 L alone), no bucket split when the smaller side fits 32 Ki slots (0.974 alone; the bucket count sat on its
16-bucket floor), together **L 0.9423 / S 0.9149 (10/10 each)** vs ref-ce500790; 64 bits/query a tie. Ir/visit
202.9 → 194.8, BucketJoin.h −25.7 %, every other file unchanged to the instruction. Gates: golden 0 ULP, ctest
335/335, pytest 592/8, prek, Slurm 1873963 all suites; installed `_core` md5 97dea1f3. **Step 3 declined with
arithmetic:** pass-B scalar bookkeeping is 7.95 % of Ir (15.5 Ir/row); a dense AVX2 block covers 45 real rows per
64 lanes, ceiling 1.4–2.3 % of Ir vs a 3 % gate, on the load-bound loop where A4 lost. Candidate for later (after
W): fuse the two Anti(G) walks (pass B + `stage_rows`, 224 M Ir at S) — but the filter needs the incoming
queries, so it needs a design, not a patch. Note: `one-round/.venv-dbg` now holds fd795841 RelWithDebInfo.

**Ladder A/B `fd795841` vs base-296 (job 1874182, L1 3 reps + L2a 2 reps):** time L1 Hubbard **1.603** (ce500790:
1.853), L1 Pauli **1.455** (1.610), graph 0.972, **L2a 1.394** (1.427); peak 0.875 / 0.944 / 0.903 / 0.902; ledger
unchanged (0.806 / 0.860 / 0.900 / 0.722). L2b (job 1874191): **1.430** (ce500790: 1.481), peak 0.920, ledger 0.726. Stack a1c122a9 ladder: job 1874208 (L1+L2a) running, nodes job queued.

### S2: PGO arm (Sonnet)

Detached worktrees at ce500790 and 5ada3da3 with `.venv-pgo`: configure with
`EXTRA_CXXFLAGS="-fprofile-generate -fprofile-update=atomic"` (`ir-assert.sh` shows the separate-build-dir
recipe), run the S and L cells + one Pauli cell once, rebuild with `-fprofile-use -fprofile-partial-
training -Wno-missing-profile`, check golden 0 ULP, then one exclusive-node timing job with four arms
(base, base-pgo, a5, a5-pgo; `pf-mkarm.sh` pattern for the venv clones). Report only.

**S2 result (2026-09-03, job 1873677, 10 reps, bit-identical 0 ULP both arms): PGO is not worth a commit.**
a5-pgo/a5 = 0.982 (S, 10/10) / 0.990 (L, 9/10); base-pgo/base = 1.015 (S) / 1.007 (L) — a regression.
Recipe kept under `scratch/hashfree/pgo-logs/` (`pgo-configure.sh`, `pgo-mkarm.sh`, `pgo-train.sh`,
`pgo-golden.sh`, `pgo-tim.sbatch`). Build fact: `EXTRA_CXXFLAGS` reaches only `monoprop-objs` compile
options, not the link line; LTO+PGO needs the same flags in `CMAKE_SHARED_LINKER_FLAGS`/
`CMAKE_MODULE_LINKER_FLAGS`. Worktrees `pgo-ce500790`/`pgo-base` removed.

### A/B agent (Sonnet)

Same recipe as today (`ab-hashfree-n1.sh` + `ab-hashfree-nodes.sh`, `ab-job-id-<tag>.txt`,
`AB-SUMMARY.md`). Runs: storage tip vs `ref-ce500790` (after C step 1); R, W tips vs frozen refs as they
land. Two dev-x86 nodes are shared with the exclusive-node timing gates, so the coordinator serialises
submissions (one queue file `scratch/hashfree/QUEUE.txt`).

### Coordinator (me)

- Dispatch order: C (resume, step 1) and W and S3 and S2 immediately; R when C step 1 is green; A/B jobs
  as tips land. Frozen refs: `ref-ce500790` exists; add `ref-<tip>` for each landed tip before its A/B.
- Docs: `DEUCALION-MONOPROP.md` §7 — `monoprop_PARTITIONS` is a pytest-only cap; exporting it into
  ctest fails 15 unrelated cases (Track C finding). Add settled facts to §3/§6 as they come; record every
  landing/rejection in the plan file and `engine-redesign-decisions.md`.
- Integration at the end: rebase `perf/storage-levers` (C, R) → `perf/wire-zero-copy` (W) →
  `perf/one-round-exchange` (S3) into one stack; final ladder A/B vs base-296 at L1/L2a/L2b/L3.

## Verification

- Every commit: serial ctest (tally read), pytest 623/8, golden 0 ULP over 35 cells against
  `golden/one-round.json` and `one-round-1p5.json`, `prek --from-ref <base>`, Slurm MPI job (ctest mpi,
  pytest `--with-mpi` at 2×1/2×16/4×8/16×16).
- Memory claims: kernel `VmHWM` from `/usr/bin/time` plus the ledger fields, at P=1, 4, 128 and 2×16 on
  the 9.26 M reproducer; then ladder L2a/L2b `peak − ledger` B/term.
- Time claims: exclusive dev-x86 node, arms interleaved and order-rotated, ≥ 10 paired reps for a landing
  decision, ratios vs frozen `ref-ce500790` and base-296.
- Expected end state if R and W land: L2a ledger 37.7 → ≈ 31 B/term, peak 54 → ≈ 39 B/term (0.65× base);
  L2b similar; time unchanged at R ≤ 16. S2/S3 are upside only.

## Decision taken with the user (2026-09-03)

Multi-node evidence for Track W is **capped at 16 nodes on `normal-x86`** (≈ 2 k core-hours per arm): one
A/B of the W tip vs base-296 at 16 × 8 × 16 (Hubbard, sized near 100 M terms/node where base's weak
efficiency was worst), 3 reps, submitted by the A/B agent only after W's dev-x86 gates are green.
`normal-x86` has no outbound internet, so the arms must be fully built frozen venvs; the job script is
`ab-hashfree-nodes.sh` with `NODES=16` and a 2 h walltime. No 64-node run in this round.



---

# Round 3 (revised 2026-09-03 after user feedback): one representation, derived from the algorithm

User: "instead of optimising multiple data structures, consider an optimal data structure / bitset representation …
rethink based on the algorithm and propose fresh ideas that will improve memory and time massively." Decision already
taken: memory levers may cost ≤ 5 % time; bit-identical results stay mandatory.

## 1. What the algorithm actually needs

Per gate `exp(iθG)`, with `G` a monomial over `2N = 256` GF(2) coordinates:

1. **Anti(G)** = { ν : ⟨g, ν⟩ = 1 } for one fixed GF(2)-linear functional `g` (parity of ν on `supp(G)`, 2–4
   coordinates for Hubbard).
2. Anti(G) is a disjoint union of **pairs** {ν, ν ⊕ G}; each pair is one 2×2 rotation of its two coefficients.
   The pair is processed iff at least one side has |c| ≥ atol; a missing partner is created.
3. The rotation's **sign** for Majorana strings is `(−1)^{Σ_{g∈G} |ν ∩ [0,g]|}` (+ |ν| for odd |G|), i.e.
   `⟨s_G, ν⟩` with `s_G = Σ_g 1_{[0,g]}` — **also a GF(2)-linear functional of ν**. (Pauli: the phase depends only on
   ν's bits on `supp(G)`.) Nothing in the hot loop needs the row's positions except creating a new term.
4. Every operation on identities is **XOR with G**: partner, creation, routing. The identity therefore wants a
   representation that is *linear* (so partner/creation are XORs), *injective on rows of weight ≤ cutoff* (so
   equality is proof, no confirm), and *addressable* (so a partner is found by arithmetic, not by a hash or a join).

Today's engine stores the identity three times (position rows 11 B, inverted-index transpose 10.5 B, 32-bit keys
4 B) and rebuilds a per-gate join over records to pair them up. S3's callgrind puts pass B (the coefficient sweep)
at 8 % of instructions and the join at 12 %; the other ~80 % is the record pipeline — emit, encode, wire, decode,
resolve, halves — spent on the ~15 % of anti rows that emit: **≈ 1 200 instructions per emitting row**. That
pipeline exists only because a partner cannot be addressed directly.

## 2. The representation: syndrome-addressed, bit-sliced blocks

One store per partition, blocks of ~4 K rows keyed by an address range; each block holds five parallel arrays.

| component | per term | role | why it is the right encoding |
| --- | ---: | --- | --- |
| **syndrome** `s(ν) = Hν` of a binary code with minimum distance `> 2·cutoff` (BCH, `t = cutoff`; GF(2⁹) shortened to length 256 → `9t` bits: c7 → 8 B, c10 → 12 B; Pauli c14 (weight ≤ 28) is the one case where it exceeds the restrided rows, 32 vs 24.6 B — keep position rows there, see §5) | 8–12 B | the row's identity **and** its key **and** its address | linear (`s(ν⊕G) = s(ν)⊕s(G)`), injective within the cutoff (two rows of weight ≤ t differ by weight ≤ 2t < d), fixed width, sortable; decoding back to positions (Berlekamp–Massey) is only needed at export and at creation |
| **coefficient** | 8 B | value | float64, unchanged (bit-identity) |
| **mode slices** | ≈ 9.5 B | the fold | dense modes bit-sliced per 64-row group, stored in **bundles of 8 consecutive modes = one cache line**, modes ordered so a generator's support lands in ≤ 2 bundles; `Anti(G)` per group = XOR of 2–4 words (as today) |
| **residual positions** | ≈ 2 B | sparse modes | per block, delta-coded (row gap, mode) pairs; folded by scatter when `supp(G)` hits a sparse mode; promotion to a slice at density ≥ 1/16 |
| **checkpoint slices** `P_k(ν) = ⟨1_{[0,8k)}, ν⟩`, k = 1..31 | 4 B (2 B with 16-mode checkpoints) | the sign | `⟨s_G, ν⟩` = XOR over `g ∈ G` of (checkpoint word ⊕ ≤ 7 words of the bundle containing g) — all words the fold already touched. **The sign of every pair in a group comes out of the fold in ~40 word ops; no row is read.** Linear, so maintained by XOR at creation |
| block directory | ~0 | address → block | radix over the top address bits; the address is the syndrome's top bits, the partner's block is `addr ⊕ addr(G)`; partition/rank = higher bits still (today's GF(2) routing, unchanged) |

Per-gate kernel (single partition): fold each group → anti mask + sign mask; for each anti row read `c`; if |c| ≥
atol (or the partner's record arrived) find the partner: `s' = s ⊕ s(G)` → directory → binary search on the
block's sorted syndromes (≤ 3 cache lines) → rotate both coefficients in place, **once per pair** (the side with the
smaller address owns a local pair; a remote pair uses today's 1½-round protocol with records `{s(ν), value}` —
exact by syndrome equality, no positions on the wire); absent partner → decode `s'`… no: `μ = ν ⊕ G` needs ν's
bits → reconstruct ν from its group's bundles (10 lines, creation only) and append μ to the partner block's tail.
Blocks re-sort their tails (and rebuild their slices) at gate end; splits on capacity.

## 3. What this removes and what it costs

Removed outright: position rows (11 B), keys (4 B), the inverted index as a separate structure, `BucketJoin` +
filter + staged rows, `SelfQueryStage`/`sent`/`pr` for local pairs, `pre_cos`, `FusedContract::halves`, the second
coefficient pass, all per-record row reads and confirms. Kept: coefficients, the fold (now also producing signs),
the cross-partition protocol (records shrink to 18–20 B and need no confirm), goldens and the ladder.

Honest byte count at L2a (Hubbard c10, B/term): syndrome 12 + coefficient 8 + slices 9.5 + residual 2 + checkpoints
2–4 = **33.5–35.5 resting** (today 34.9) + block tails ~5 % + wire ~2–3 → **peak ≈ 39 B/term** (today 51, base 60)
→ **≈ 5.7 B terms on one node**. The resting figure does not fall because identity (12) ≈ rows (11) and the
checkpoints replace the keys; what falls is the transient (16 → 3) and the instruction count. With the two
compressions in §4 the peak reaches ≈ 32 → 7 B terms.

Time: the per-emitting-row cost drops from ~1 200 to ~60–80 instructions (lookup + rotation), pass B and the fold
stay; estimated **2–4× faster than base-296 at P=1** (today 1.65× slower), and the same ratio at 1×128 because the
protocol is unchanged in shape but its records are half the size and carry no confirm work.

**The exact floor, stated plainly.** With float64 coefficients, an exact identity and a per-gate anti test that
reads O(|supp G|) bits per row, the resting minimum is ≈ 8 + 7.4 (entropy of a weight-≤10 subset of 256) + ≥ 5
(entropy of the slices) ≈ 21 B/term, so **10 B terms (24 B/term peak) is at the edge of what exactness allows and
is not reachable in this round**. Levers that would get there, each a numerics decision for the user, not for me:
float32 coefficients (−4 B/term; not bit-identical), symmetry orbits (÷ orbit size; user said no), or two nodes.

## 4. Two compressions that fit inside the design (later, measured)

- **Slice entropy coding.** Dense slices at density 1/8–1/2 have 0.54–1 bit of entropy per row; a per-bundle
  run/Elias-Fano coding of the low-density slices with word-level decode in the fold buys ≈ −4 B/term if the fold is
  < 10 % of time (the MP track measures this first).
- **Tighter identity.** 9t bits is 1.3–1.6× the sphere-packing bound; a better length-256 code with d ≥ 2t+1
  (or a two-level code: BCH for the address, residual bits only where needed) buys 1–3 B/term.

## 5. Fresh ideas that change the numerics — listed for the user's decision, not scheduled

- **Layer batching.** All hoppings on disjoint bonds in a Trotter layer commute; applying a layer in one pass
  (each row anticommuting with k gates of the layer expands into 2^k terms) reads each coefficient once per layer
  instead of once per gate — a time lever of order (gates per layer), and one truncation per layer instead of per
  gate. Not bit-identical (rounding order and truncation points change). Could be an opt-in mode.
- **Pauli identity.** For Pauli with 2·cutoff = 28 the syndrome is 32 B; keep restrided position rows there (24.6 B)
  and address them by a 32-bit linear key + exact compare, i.e. today's scheme inside the new block store.
- **float32 coefficients** as an opt-in: −4 B/term, and halves pass-B traffic. Not bit-identical.

## 6. Build plan — prototype first, P=1, measured against the stack

Branch `perf/syndrome-store` off `a1c122a9` (worktree `worktrees/syndrome`), new directory
`cpp/monoprop/detail/syndrome/`. Nothing in the shipping path changes until the prototype wins on the 9.26 M
reproducer. Parallel Opus tracks:

| track | deliverable | gate |
| --- | --- | --- |
| **A. Codec** | `SyndromeCode<NumModes, t>`: parity matrix over GF(2⁹) shortened to 2N, `encode(positions)` (XOR of column syndromes), `decode(s) -> positions` (Berlekamp–Massey + Chien), `add(s, G)`; property tests: linearity, injectivity on 10⁷ random weight-≤t pairs, decode∘encode = id, width table per cutoff | unit tests; encode ≤ 20 ns/row, decode ≤ 2 µs |
| **B. Store + fold** | `SlicedBlockStore`: blocks, bundles, residuals, checkpoint slices, directory, append/split/tail-merge, `fold(G) -> (anti, sign) per group`; test: sign mask equals `Monomial` product phase for 10⁶ random (ν, G); anti mask equals today's fold; reconstruction of a row from bundles | unit tests; fold cost ≤ 1.2× today's `even_parity_scan_pass1` on 9.26 M rows |
| **C. Kernel (after A+B)** | single-partition gate kernel with pair-once processing and today's truncation rule (`rotate iff either side emits`, structural cutoff, atol), reusing A/B; wire it behind a `monoprop_ENGINE=syndrome` knob for `propagate` only (graph mode stays on the current engine) | golden 0 ULP vs `one-round.json` at P=1 on all 35 cells; exclusive-node S/L time vs ref-a1c122a9 and base-296; VmHWM on the 9.26 M reproducer |
| **D. Protocol (after C)** | records `{s(ν), value}` on the existing exchange, responses for silent hits, creation of remote partners; pair-once ownership by address | golden at P=4/128, MPI job, L2a/L2b ladder |
| **MP (Sonnet/Opus)** | memory profiling of the *current* stack (mallinfo2 + smaps_rollup + ledger at the widest gate, 1×128 and 2×16), fold share of time under callgrind → decides §4's slice coding | attribution table in `scratch/hashfree/memprof-logs/` |
| **W (cont.)** | land the retired-half fix (`wire_queries(two_rounds)`) and `pair_recv_` release, rebase onto a1c122a9, ladder | L2b peak ≤ 0.95 vs ref-ce16f0f6 |
| **SZ (Sonnet)** | sizing runs at 2 B and 3.5 B terms with the stack tip (1×128, 8×16), fit GiB vs terms | the terms-per-node number |

Go/no-go for D and for replacing the shipping path: C must be bit-identical and show ≥ 1.5× time vs base-296 with
VmHWM ≤ 0.85× the stack on the 9.26 M reproducer. If it does not, the codec (A) alone still replaces keys + confirm
in the current engine (−4 B/term, fewer instructions) and the checkpoint-sign trick alone removes the emit-side row
reads; both are salvageable as ordinary levers.

## 7. Verification

Codec and store: property tests above plus the existing serial ctest suite untouched. Kernel: golden 0 ULP on 35
cells (P=1, then P=4/128), pytest, prek, MPI job; exclusive-node paired timing (≥ 10 reps) on S/L/Pauli vs
ref-a1c122a9 and base-296; VmHWM + ledger at P=1/4/128 and 2×16 on the 9.26 M reproducer; ladder L1/L2a/L2b of the
frozen tip vs base-296; SZ's fit gives the terms-per-node ceiling before and after.
