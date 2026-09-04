# Handoff: monoprop memory/time campaign on Deucalion (2026-08-11 → 2026-09-04)

Written when the project's node hours ran out on 2026-09-04. Everything below is on disk under
`/projects/EEHPC-DEV-2026D08-260/aaron` and, for the code, pushed to `github.com/Algorithmiq/monoprop`.
The full working plan (rounds 2 and 3, with every measured result inline) is
`scratch/hashfree/PLAN-rounds-2-3.md`. Machine rules are `DEUCALION-MONOPROP.md`; repo rules are
`monoprop/AGENTS.md`. Nothing here has been opened as a PR; nothing is merged.

## 1. State in one paragraph

Starting from PR #296 (`perf/linear-routing-on-wire`, local base commit `5ada3da3`), a stack of 19
commits (`perf/one-round-exchange` → `perf/storage-levers`, tip **`a1c122a9`**) replaces the
hash-table partner lookup with a hash-free kernel joined inside Anti(G), a 1½-round exchange
protocol, a tag-filtered bucket join, and pooled/chunked/restrided term stores. It is
**bit-identical** to base (0 ULP on 35 golden cells, at P=1, 4 and 128 and across MPI layouts),
**0.79–0.85× the peak RSS** (0.67× the ledger) but **1.31–1.65× slower** than base on the PR #317
ladder (the graph row is 0.95, slightly faster). The redesign's target was memory; the time gap is the
"residual" the kernel tracks worked down from 1.97× at the first tip (S3's join filter took it from
1.85× to 1.60× at P=1) and it is the main reason round 3 exists.
A third branch, `perf/wire-zero-copy` (tip **`91806a63`**, off `ce16f0f6`, not yet rebased onto
the stack), removes the wire's copy chain with `mpi::pair_exchange`; its last commit has only
partial gates (see §4). Round 3 — a from-scratch representation designed to cut both memory and
instructions massively — is planned and approved but **no code has been written for it** (§5).

## 2. What is on GitHub (pushed 2026-09-04, all new branch names, no force pushes)

| branch | tip | base | what | gates |
| --- | --- | --- | --- | --- |
| `perf/linear-routing-on-wire-deucalion-base` | `5ada3da3` | — | the local PR #296 base every arm was measured against. **Not** the same commit as origin's `perf/linear-routing-on-wire` (`63cf10fd`); the stack must be rebased onto whatever #296 merges as | n/a |
| `perf/hashfree-kernel` | `1e7a7e98` | 5ada3da3 | first commit of the stack (also contained in the next branch) | green |
| `perf/one-round-exchange` | `fd795841` | 5ada3da3 | 8 commits: hash-free kernel, one-round symmetric exchange, bucket join with 4-byte keys, responses for silent hits, query-tag filter + fused cos sweep, A5 reserves, S3 filter bits | all green (serial+MPI ctest, pytest, golden 0 ULP, prek, Slurm 1873963) |
| `perf/storage-levers` | `a1c122a9` | fd795841 | 11 commits: coefficient growth policy, ledger fields, `ChunkedArray`/`ChunkPool`, chunked dense index and rows/keys, size-adaptive chunks, gate-buffer HWM, restride to observed width | all green (MPI job 1874157, ctest 369/369, pytest 623/8, golden 0 ULP both refs, prek) — **this is the stack tip** |
| `perf/wire-zero-copy` | `91806a63` | ce16f0f6 | 5 commits: `pair_exchange` port, integration, flat wire arena, join row side from previous count, single round-1 buffer for answering sinks | e3462b97: all green (MPI 1874100/1873980). 91806a63: login-node gates green (ctest 371/371, pytest 593/8, golden 0 ULP on all 35 cells vs both references at S=1, prek 0; `scratch/hashfree/wire-logs/s5-gates.log`). **Still needs a compute node:** MPI job, S=4 arm-to-arm golden, L2b A/B |
| `perf/pair-exchange` | `cfd50db7` | older | the standalone `pair_exchange` verb W ported from | green at the time |
| `perf/majorana-sign-from-positions` | `dc85a86f` | older | measured prototype: Majorana sign from positions; input to round 3's sign-from-fold idea | prototype |
| `recover/*` (7 branches) | — | — | parked work: `colo-coeff-key` (A3, rejected, slower), `prefetch-sparse-streams` (A4, rejected), `scan-invariant-hoists` (1 %, rejected), `storage-gate-buffers-hwm`, `storage-levers-pre-rebase{,-2,-3}` (pre-rebase tips) | rejected or superseded; kept for the record |

Frozen local reference worktrees (detached, own venvs, safe A/B arms): `worktrees/base-296`
(5ada3da3, `_core` md5 f991ce3a), `ref-60254b20`, `ref-bff293d6`, `ref-ce500790` (eb365dab),
`ref-ce16f0f6` (043ccf72), `ref-fd795841` (97dea1f3), `ref-a1c122a9` (eccf7a15 in its `.venv`).
Development worktrees: `worktrees/one-round`, `worktrees/storage`, `worktrees/wire`
(`.venv-s5` holds 91806a63, md5 fb347346), `worktrees/bench-317` (ladder tooling, e3588ecb).

## 3. Measured results (PR #317 ladder, paired medians vs base-296, dev-x86, one allocation per job)

Stack `a1c122a9` vs base (jobs 1874208 L1/L2a, 1874505 L2b; 3 reps L1/L2b, 2 reps L2a):

| row | shape | terms | time (stack/base; >1 = slower) | kernel peak RSS (stack/base) | ledger (stack/base) |
| --- | --- | ---: | ---: | ---: | ---: |
| L1 Hubbard | P=1 | ~10 M | 1.649 | 0.793 | 0.668 |
| L1 Pauli | P=1 | | 1.458 | 0.821 | 0.677 |
| L1 random Heisenberg graph | P=1 | | 0.954 | 0.820 | 0.676 |
| L2a Hubbard c10 | 1×128 | 1.0 B | 1.315 | 0.846 | 0.668 |
| L2b Hubbard c10 | 8×16 | 1.0 B | 1.354 | 0.823 (51.6 B/term summed) | 0.668 (34.9 B/term) |

Raw medians for L1 Hubbard: base 25.9 s, stack 42.7 s; L2b: base 130 s, stack 176 s. Every ratio in
this document is arm/base with the arm named first. Intermediate tips (for attribution): ce500790 L1
1.853 / L2a 1.427 / L2b 1.481 slower; fd795841 (S3) L1 1.603 / L2a 1.394 / L2b 1.430 slower with
peak 0.875–0.944; storage `ce16f0f6` vs ce500790 time 0.96–1.01, ledger 0.75–0.93. L3 (2 nodes,
bff293d6) 1.52× slower, 0.89 peak, 0.75 ledger — compute-bound at R=16, so the slowdown is the
kernel (callgrind: 195 vs 160 instructions per visited row), not the transport. Term counts identical in every cell. `AB-SUMMARY.md` for every job is under
`runs/ab-hashfree-<job>/` and `runs/ab-hashfree-nodes-<job>/`.

Where the remaining bytes are at L2b (B/term, stack): rows 10.6 + inverted index 10.1 + coefficients
7.6 + keys 3.8 + gate scratch 1.2 = 34.9 ledger; peak 51.6, so **≈ 17 B/term is transient wire
and join scratch** — that is what W attacks (W's per-buffer stamp: `scratch/hashfree/wire-logs/
breakdown-2.md`, and `worktrees/wire/mp-wire-diag2-1874222.out` for the e3462b97 tip).

Rejected with numbers (do not retry without a new idea): A3 co-located coeff/key record (slower),
A4 software prefetch (slower), pass-B AVX2 blocks (≤ 2.3 % ceiling by Ir arithmetic), PGO (≤ 1.8 %
on the redesign, a regression on base; recipe in `scratch/hashfree/pgo-logs/`), 64 bits/query
filter (tie), dropping the R=1 self/sent reserve (+1.4 %).

## 4. Open work on the existing branches

**W (`perf/wire-zero-copy`), in order:**
1. Login-node gates on 91806a63 are done and green (`scratch/hashfree/wire-logs/s5-gates.log`,
   `pytest-s5.log`, `golden-diff-s5-vs-*.log`, `prek-s5.log`).
2. Needs a compute node: MPI job (`wire-logs/w-diag2.sbatch` pattern; ctest MPI + pytest
   `--with-mpi` at 2×1/2×16/4×8/16×16), golden arm-to-arm at S=4, then the L2b A/B vs
   `ref-ce16f0f6` with `harness/sbatch/ab-hashfree-nodes.sh`. **Gate: summed peak ≤ 0.95**
   (e3462b97 was 0.983 Hubbard / 0.945 Pauli; the retired half of the round-1 pool was 15.5 % of
   the 2×16 per-gate peak, so 91806a63 should clear it).
3. Then a release rule for `HybridComm::pair_recv_` (48.6 % of the widest single-slot stamp at
   2×16, permanent), rebase onto `a1c122a9` (park the pre-rebase tip as
   `refs/recover/wire-pre-rebase`), full gates, ladder vs base.
4. Multi-node evidence was capped by the user at **16 nodes on `normal-x86`** (3 reps, W tip vs
   base, ~2 k core-hours per arm, 2 h walltime, `NODES=16`); `normal-x86` has no internet so both
   arms must be prebuilt venvs.

**Storage/one-round:** complete. Revisit items only: Pauli restride constant 0.79·W (7/8 would
avoid the c8/c12 restride at ~2 B/term cost at c14); `raise_bound` leaves side-map rows until the
next restride; the later kernel candidate is fusing pass B with `stage_rows` (needs a design, the
filter depends on the incoming queries).

**Before any PR:** rebase the stack onto the merged PR #296, then `main`; re-run every gate;
`Assisted-by:` trailers are already on every commit; PR text gets the `:robot: _AI text below_`
prefix per `AGENTS.md`.

## 5. Round 3 — approved, not started: one representation derived from the algorithm

Full text: `scratch/hashfree/PLAN-rounds-2-3.md`, section "Round 3". Rebuild/measure recipes for another machine: `REPRODUCE.md`. The user asked for a single
optimal data structure rethought from the algorithm rather than more per-structure levers, with
the goal of nearer 10 B terms on one node; bit-identical results stay mandatory and memory levers
may cost ≤ 5 % time. Summary:

- **Observation.** Anti(G) and the Majorana rotation sign are both GF(2)-linear functionals of the
  row ν; every identity operation is XOR with G. ~80 % of today's instructions are the record
  pipeline (emit/encode/wire/decode/resolve/halves, ≈ 1 200 Ir per emitting row), which exists
  only because a partner cannot be addressed directly.
- **Representation.** Per partition, blocks of ~4 K rows keyed by address range, five parallel
  arrays: BCH **syndrome** of ν (GF(2⁹) shortened to 256 modes, 9·cutoff bits: 8 B at c7, 12 B at
  c10; linear, injective within the cutoff, sortable — it is identity, key and address at once),
  float64 coefficient, bit-sliced **mode slices** in 8-mode cache-line bundles, delta-coded
  residual positions for sparse modes, and **checkpoint prefix-parity slices** so the sign of
  every pair falls out of the fold with no row read. Partner = `s ⊕ s(G)` → directory → binary
  search; pairs processed once, locally; creation reconstructs ν from the bundles.
- **Honest numbers.** Resting 33.5–35.5 B/term at L2a (≈ today) but transient 16 → ~3, so peak
  ≈ 39 B/term → ≈ 5.7 B terms/node (≈ 32 → 7 B with slice entropy coding and a tighter code);
  time estimated 2–4× faster than base at P=1. Exact floor ≈ 21 B/term, so **10 B terms is not
  reachable while staying exact**; the levers that would get there (float32 coefficients, symmetry
  orbits, layer batching) change numerics and were listed for the user's decision only.
- **Build plan.** Branch `perf/syndrome-store` off `a1c122a9`, new directory
  `cpp/monoprop/detail/syndrome/`, shipping path untouched until the prototype wins on the 9.26 M
  reproducer. Tracks: **A** codec (`SyndromeCode`: encode/decode/add + property tests, encode
  ≤ 20 ns/row); **B** store + fold (`SlicedBlockStore`, sign mask == `Monomial` phase on 10⁶ random
  pairs, fold ≤ 1.2× today's pass 1); **C** single-partition kernel behind `monoprop_ENGINE=syndrome`
  (golden 0 ULP at P=1); **D** protocol with `{s(ν), value}` records; plus MP (memory attribution of
  the current stack, fold share under callgrind), SZ (sizing fit at 2 B and 3.5 B terms). **Go/no-go
  for D and for replacing the shipping path:** C bit-identical, ≥ 1.5× time vs base, VmHWM ≤ 0.85×
  the stack on the 9.26 M reproducer. Fallback if C fails: the codec alone replaces keys + confirms
  (−4 B/term), and the checkpoint-sign trick alone removes emit-side row reads.
- A and B are pure C++ with unit tests and need no compute node; only C's timing and everything
  from D on need Slurm.

## 6. How to resume (recipes that work)

- Build: `ssh ln04`, `export MONOPROP_SRC=<worktree>`, `source harness/env.sh`,
  `bash "$MONOPROP_SRC/hpc/deucalion/build.sh"` (`-j4`, `OMP_NUM_THREADS=4`; wait while
  `pgrep -u $USER | wc -l` > 100; 20 GiB login cgroup). New worktrees need
  `ln -sfn $PWD/harness <tree>/hpc/deucalion`. Fast loop: see the memory note in §3 of
  `DEUCALION-MONOPROP.md` (venv nanobind must be re-applied after every `uv sync`).
- Gates (login node): `scratch/hashfree/wire-logs/w-{ctest,pytest,golden,prek}.sh` are the
  templates (pytest needs `monoprop_PARTITIONS=4`; **unset it for ctest**). Golden: 35-cell dumps
  via `scratch/hashfree/golden.py` + `golden_diff.py`; references `golden/one-round.json` and
  `one-round-1p5.json` (0 ULP required). Assert build: `one-round-logs/ir-assert.sh`.
- Gates (compute node): MPI job scripts `harness/sbatch/` and `wire-logs/w-diag2.sbatch`;
  `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` for MPI ctest.
- A/B: `harness/sbatch/ab-hashfree-n1.sh` (L1 + L2a) and `ab-hashfree-nodes.sh` (L2b/L3/16
  nodes), report via `harness/tools/ab_pairs.py`; arms are frozen `ref-<hash>` venvs identified by
  commit + `_core.abi3.so` md5; ≥ 10 reps for a landing decision, 3 for sizing; submit from inside a
  worktree with `-A eehpc-dev-2026d08-260x`; append to `scratch/hashfree/QUEUE.txt` first.
- Memory: kernel `VmHWM` (not sampled RSS) plus the ledger (`d_gate_buffers_hwm_bytes`,
  `d_row_inline_width`, slack fields); the 9.26 M reproducer is `wire-logs/w_repro.py`.
- Known environmental failure: `shm_comm_oversubscribed_repeated_collectives` fails on ln03 under
  load for every arm including frozen refs; it passes on a compute node.

## 7. Housekeeping

- `scratch/hashfree/golden/*.json` is ≈ 28 GB of 35-cell dumps; only `one-round.json` and
  `one-round-1p5.json` are references. The rest (`s*-p4`, `storage-*`, `pgo-*`, `c1/c2`, …) can be
  deleted.
- Worktrees `ref-*` can be removed once the stack is merged; `worktrees/pair-exchange` is
  superseded by W's port.
- Local-only refs: `git -C monoprop for-each-ref refs/recover` (the seven relevant ones are also
  on GitHub as `recover/*`). `harness/` is its own local-only git repo (job scripts, env, tools) and
  is not on GitHub.
- Agent logs: `scratch/hashfree/{one-round,storage,wire,pgo}-logs/PROGRESS.txt` and `GATES.txt`.
