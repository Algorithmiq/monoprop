# Agent brief: continuing the monoprop memory/time campaign

You are picking up work that a coordinator (Claude Code) and its Opus subagents did on Deucalion between
2026-08-11 and 2026-09-04. Node hours ran out; the code is on GitHub and the evidence is in this directory.
Read `HANDOFF.md`, then `REPRODUCE.md`, then "Round 3" in `PLAN-rounds-2-3.md`, then the repo's
`AGENTS.md`. `NOTES-engine-history.md` lists what was already tried and rejected — check it before
proposing anything.

## Non-negotiable rules (set by the user; do not re-ask)

1. **Bit-identical results** to the base arm by default: 0 ULP on the 35 golden cells at P=1, P=4 and
   across MPI layouts, identical term counts. No float32, no symmetry orbits, no truncation-rule changes.
   Ideas that change numerics are listed for the user's decision, never scheduled.
2. **Memory levers may cost ≤ 5 % time each.** Time levers must not raise peak memory.
3. **One agent per worktree.** Reference arms are detached worktrees nobody edits; identify an arm by
   commit + md5 of the installed `_core.abi3.so` (and `libmonoprop.so`), never by version string.
4. **Measure before claiming.** Interleaved arms in one session, order rotated per repetition, paired
   ratio per repetition, report the paired median with the agreement count; ≥ 10 repetitions for a
   landing decision, 3 for sizing, ≥ 3 sweep points for a trend. Quote kernel VmHWM, not sampled RSS.
   **State the direction of every ratio and give the raw medians** (the handoff once mis-stated a 1.65×
   slowdown as a speedup; the tables in `HANDOFF.md` §3 now carry both).
5. **Commits** follow `AGENTS.md`: C++23, `clang-format`, conventional commit with gitmoji,
   `Assisted-by: ClaudeCode:<model>` trailer, no `Co-authored-by`. Commit only when every gate is green:
   serial ctest (read the tally), pytest, golden 0 ULP, `prek` over the commit range, and the MPI suites
   when a compute node exists. Never create a non-version tag (it breaks `uv sync`); park work under
   `refs/recover/*` or `recover/*` branches. Do not force-push the campaign branches; branch from them.
6. **When asked for "massive" gains,** propose one representation derived from what the algorithm needs,
   with a quantified B/term and instruction budget and an honest statement of the exact floor — not a list
   of per-structure levers. Round 3 in the plan is the template.
7. If a tool or permission block stops you, stop and report; do not route around it.
8. Keep a `PROGRESS.txt` (one line per step, timestamped) and a `GATES.txt` per track, as the logs here do.

## Work, in priority order

**0. Environment check (any machine).** `REPRODUCE.md` §1–4: build `base-296` (5ada3da3) and `stack`
(a1c122a9), run ctest + pytest on both, dump goldens for both and diff them: every maxULP column 0. Do not
proceed until this holds; it validates compiler, MPI and the golden tooling together.

**1. Bring the stack up to date.** Check whether PR #296 (`perf/linear-routing-on-wire`) has merged
(`gh pr view 296`). Rebase `perf/storage-levers` (which contains `perf/one-round-exchange`) onto the merged
commit or onto `main`; park the pre-rebase tip first. Expect conflicts only in files the base moved.
Re-run the full gates, then the ladder (REPRODUCE §6) vs the new base at L1 and, with a large node, L2a/L2b.
Report time and memory ratios with direction.

**2. Finish W (`perf/wire-zero-copy`, tip 91806a63).** Compute-node gates it never got: MPI ctest +
`pytest --with-mpi` at 2×1/2×16/4×8/16×16, arm-to-arm golden at S=4, then L2b A/B vs `ce16f0f6` with the
gate **summed peak ≤ 0.95** (e3462b97 was 0.983 Hubbard / 0.945 Pauli). Then a release rule for
`HybridComm::pair_recv_` (48.6 % of the widest single-slot stamp at 2×16, permanent — `logs/wire/
breakdown-2.md`), rebase onto the stack, full gates, ladder. Success = peak−ledger at L2b ≤ 8 B/term, time
≤ 1.02 at R ≤ 16.

**3. Round 3, tracks A and B in parallel** (pure C++, no cluster needed; plan §6 has the gates):
- A. `SyndromeCode<NumModes, t>` in `cpp/monoprop/detail/syndrome/`: BCH parity matrix over GF(2⁹)
  shortened to 2N coordinates, `encode(positions)` as XOR of column syndromes, `decode(s)` via
  Berlekamp–Massey + Chien, `add(s, G)`; property tests for linearity, injectivity on 10⁷ random
  weight-≤t pairs, decode∘encode = id, and the width table per cutoff. Gate: encode ≤ 20 ns/row,
  decode ≤ 2 µs.
- B. `SlicedBlockStore`: ~4 K-row blocks keyed by address range, bit-sliced modes in 8-mode bundles,
  delta-coded residual positions, checkpoint prefix-parity slices, directory, append/split/tail-merge,
  `fold(G) -> (anti mask, sign mask)` per 64-row group. Tests: sign mask equals the `Monomial` product
  phase on 10⁶ random (ν, G); anti mask equals the current fold; a row reconstructs from its bundles.
  Gate: fold ≤ 1.2× today's `even_parity_scan_pass1` on 9.26 M rows.
Branch `perf/syndrome-store` off a1c122a9 (or off the rebased stack after step 1), one worktree per track,
merge B onto A when both are green.

**4. Round 3, track C (after A+B).** Single-partition gate kernel with pair-once processing and today's
truncation rule (rotate iff either side emits; structural cutoff; atol), behind `monoprop_ENGINE=syndrome`
for `propagate` only. **Go/no-go:** golden 0 ULP at P=1 on all 35 cells, ≥ 1.5× faster than base-296 and
VmHWM ≤ 0.85× the stack on the 9.26 M reproducer (`scripts/w_repro.py 12 7 1e-6 <label>`). If no-go,
salvage: the codec replaces keys + confirms in the current engine (−4 B/term), and the checkpoint-sign
trick removes emit-side row reads. Track D (protocol with `{s(ν), value}` records) only after go.

**5. When nodes exist:** MP (memory attribution of the stack at 1×128 and 2×16: mallinfo2 + smaps_rollup +
ledger at the widest gate; fold share of time under callgrind — decides the slice entropy coding) and SZ
(sizing runs at 2 B and 3.5 B terms, fit GiB vs terms). Multi-node evidence is capped at 16 nodes.

## What "done" looks like for a report

Commit hash, installed `_core.abi3.so` md5, gate tallies (ctest n/n, pytest passed/skipped, golden
maxULP, prek rc, MPI job id if any), and for any performance claim: the cells, repetitions, paired
median ratios with direction, raw medians, agreement counts, and identical term counts across arms.
