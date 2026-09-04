# Deucalion memory/time campaign handoff (2026-09-04)

Read in this order: `AGENT-BRIEF.md` (if you are an agent) → `HANDOFF.md` → `REPRODUCE.md` →
`PLAN-rounds-2-3.md` (section "Round 3") → `NOTES-engine-history.md`. `DEUCALION-MONOPROP.md` is the
machine document for anyone running on Deucalion again.

Path mapping: the documents were written on Deucalion and name paths under
`/projects/EEHPC-DEV-2026D08-260/aaron/`. They map onto this directory as follows.

| Deucalion path | here |
| --- | --- |
| `HANDOFF-2026-09-04.md`, `REPRODUCE.md`, `DEUCALION-MONOPROP.md` | same names (`HANDOFF.md`) |
| `scratch/hashfree/PLAN-rounds-2-3.md` | `PLAN-rounds-2-3.md` |
| `harness/env.sh`, `harness/build.sh`, `harness/sbatch/*`, `harness/tools/ab_pairs.py` | `harness/` |
| `scratch/hashfree/golden.py`, `golden_diff.py`, `wire-logs/w_repro.py` | `scripts/` |
| `scratch/hashfree/{one-round,storage,wire,pgo}-logs/` | `logs/{one-round,storage,wire,pgo}/` |
| `runs/ab-hashfree-<job>/AB-SUMMARY.md` | `ab-summaries/ab-hashfree-<job>.md` |
| `worktrees/<arm>` | your own detached worktrees (REPRODUCE.md §2) |
| `scratch/hashfree/golden/*.json` | not included (28 GB); regenerate with `scripts/golden.py` |

Code branches: `perf/storage-levers` (stack tip a1c122a9), `perf/one-round-exchange` (fd795841),
`perf/wire-zero-copy` (91806a63), `perf/pair-exchange`, `perf/hashfree-kernel`,
`perf/majorana-sign-from-positions`, `perf/linear-routing-on-wire-deucalion-base` (5ada3da3, the
measured base), `recover/*` (parked/rejected work).
