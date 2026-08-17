# Per-commit memory attribution, and the arena-segment fix

Measured on Deucalion x86, `foss/2025b`, 2026-08-16/17. Jobs 1826684, 1826685, 1826752, 1826756,
1826829, 1826830, 1826877, 1826901, 1827644.

This supersedes the headline of [RESULTS-invidx-memory.md](RESULTS-invidx-memory.md), which framed the
`perf/invidx-memory` branch as "1.25× resident memory for ~1.09× time — find the regression and repair
it" against a ≤1.02× acceptance bar. That framing was an artefact of a **missing measurement**: the
time had been bisected per commit for two rounds while the memory never had been, so there was no way
to say which commit was worth its time. The numbers in that file are not wrong; its conclusion is.

Read this file for the **method** as much as the numbers. Three of the rules now in README §10
("Reading a measurement here") were learned here, each after a table was believed and was wrong.

---

## 1. The ledger, attributed per commit

Every commit on the branch touches exactly one structure, so the memory ledger maps onto the time
bisection one-to-one. Hubbard c10 at 96,981,051 terms; Pauli c14 at 91,273,861. `TOUCHED` is
`total_bytes` less the two capacity-only fields, plus the derived u32 stamp where uncounted.

| commit | structure | HUB Δ touched | predicted | PAU Δ touched | predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| a1 smallest-container index | `inverted_index` | **−7.385** | −7.386 | **−7.611** | −7.610 |
| a2 release unearned capacity | epoch stamp u32→u16 | **−2.000** | −1.955 | **−1.955** | −1.955 |
| a3 5-byte dedup slot | `indexing` | **−8.304** | −8.304 | **−4.411** | −4.411 |
| a4 chunked row store | row-store *slack* only | **+0.002** | ~0 | **+0.002** | ~0 |
| branch vs main | | **1.427×** | 1.427× | **1.252×** | 1.252× |

**a4 saves nothing at rest, confirmed to three decimals.** Its win is entirely a growth transient,
which only peak RSS sees. That is the first half of the lesson in §3.

### Slack is virtual — and the residue is not the event

`d_terms_slack_bytes` and `d_invidx_arena_slack_bytes` are `capacity() − size()`: never faulted in.
Charging only touched bytes, the chunked row store — which looked like the single biggest win at
10.9 B/term on Pauli — goes 29.000 → 29.002 B/term.

The converse error was mine, and it is the more interesting one. Having established that the arena
slack (0.453 / 1.032 B/term) was virtual, I concluded there was nothing there to collect. But the
*mechanism producing it* — a monolithic `std::vector` grown geometrically — also produces a **copy
transient that is entirely real** and several times larger than the residue it leaves behind. Reading
`capacity() − size()` and stopping there looked at the residue and not at the event.

---

## 2. Peak RSS says something the ledger cannot see

`rsssum_mb`, min of reps, reps agreeing to ±0.2%; null-control noise floor +0.93 / +0.32 B/term.

| commit | HUB Δ RSS | HUB Δt R=1 | PAU Δ RSS | PAU Δt R=1 |
| --- | ---: | ---: | ---: | ---: |
| a1 smallest-container index | −4.58 | +2.85% | **+7.01 WORSE** | **+9.4%** |
| a2 release unearned capacity | **−4.41** | +0.04% | **−4.53** | **−1.3%** |
| a3 5-byte dedup slot | −11.87 | +5.52% | −4.17 | +2.5% |
| a4 chunked row store | −1.56 | +2.35% | **−7.54** | +1.2% |
| **branch vs main** | **1.216×** | 1.105× | **1.090×** | **1.118×** |

Three readings, each reversing something believed before the job ran:

1. **a1, the inverted-index commit, was the worst commit in the PR** — Pareto-dominated on Pauli, worse
   in time *and* in peak memory than main. Its 7.4 / 7.6 B/term at-rest win did not reach resident
   memory at all.
2. **a2 was the best commit**, by a distance, and its ledger figure (−2.0) *understates* it by 2×.
3. **a4 is a real peak-RSS win the ledger cannot see** (−1.56 / −7.54 B/term) — the growth transient.

**The Pauli R=1 bisection inverted the R=8 reading it replaced** (null control 0.998×, 4/4 every arm):
a1 costs **9.4%**, not the 2.9% R=8 reported. R=8 understated the index's cost by **3×** on this model,
because `exchange_ns` is the wait bucket and absorbs it.

---

## 3. The defect behind a1's regression

`open_segment_` projected the whole arena in one `reserve`, guarded by
`chunk_base_.empty() && chunks_total > 1` with `chunks_total = row_count_ / kChunkRows` evaluated **at
the first seal**.

- `rebuild()` sets `row_count_ = op.size()` — the *final* count — before filling rows, so the
  projection sees every chunk and fires. That is the path the guard was written for, and there it works.
- `append_rows()` sets `row_count_ = base + n` — the count *so far*. On the incremental `propagate`
  path the first seal happens as `row_count_` crosses one chunk, so `chunks_total` is 1 (guard false) or
  2 (reserving two chunks of an eventual 24). Either way `chunk_base_` is non-empty forever after and
  the projection never fires again.

The arena then grew through a 1/8 geometric floor, and `std::vector::reserve` holds the old and new
buffers **simultaneously** — a ~920 MB copy at the top on Pauli. Main has no such transient because its
index is 256 independently growing per-column vectors, so a growth copy doubles one column, never the
whole index. **a1 turned 256 small copies into one large one.**

Corroborated from data as well as code: measured arena slack was 6.5% of the index, the 1/8 geometric
floor. Had the projection fired it would have been ~1/32 ≈ 3%.

### The fix: one exactly-sized segment per sealed chunk

Not "make the projection fire" — on the incremental path there is no final size to project from, which
is why the projection was written for `rebuild()` alone, and any hint the caller could pass is an
underestimate that reduces the copies without removing them. Instead, **remove the thing that copies**:
`std::vector<uint8_t> arena_` becomes `std::vector<std::vector<uint8_t>> segments_`, one per sealed
chunk, each reserved to the exact total that `size_chunk_` already computes before any byte is written.

`arena_reserve_`, `chunk_base_`, `kArenaGrowthDen`, `kArenaProjectionDen` and the projection comment are
**deleted**. The blast radius is one expression: `chunk()` is the only place that resolves the arena, and
`arena_.data() + chunk_base_[k] + e.offset` becomes `segments_[k].data() + e.offset` — the same one
dependent load, one fewer add. The fold's hot loop is untouched.

Three properties it gets for free: arena slack becomes **structurally impossible**; alignment gets
stronger (every segment base comes from `operator new` at ≥16, and `bitmap_words` asserts 8); and a
pointer-invalidation hazard disappears, because growing the outer vector moves inner vector *objects*
but never their heap buffers.

This is the same defect class and the same remedy as a4 ("chunk the row store so growth never copies").
a2's own commit message had already written the sentence — *"that slack needs a row store whose growth
does not copy, not a better-timed shrink"* — applied to the row store rather than to the index arena.

---

## 4. Stage 1 — does the fix rescue a1?

Pre-committed rule: the fix succeeds iff, on **both** cells, the fixed arm's peak RSS beats main and
its time is no worse than a1's.

**A design error, worth recording because the rule did not catch it.** Stage 1 was specified at R=1
because R=1 is this project's primary *time* attribution surface — and that reasoning was carried over
to a *memory* question whose defect was only ever observed at **R=8**. At R=1, a1 does not regress
against main at all (it is 130 / 321 MB *better*), so the R=1 job measured the fix in a regime where the
problem it targets does not occur. It had to be run at both. **Any memory claim states its `P`, and a
defect observed at one `P` is confirmed at that `P`.**

`rsssum_mb`, min of 3 reps, reps agreeing to ±0.1% — the arms are separated by ~60× the rep spread, so
the memory column needs no agree count.

| R=8 | main | a1 | a1+fix | fix vs main | fix vs a1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hubbard | 11239.5 MB | 10765.0 (−4.89 B/t) | **10628.9** | **−6.30 B/t** | −1.40 B/t |
| Pauli | 9329.2 MB | 9962.3 (**+6.94 B/t**) | **8941.7** | **−4.25 B/t** | −11.18 B/t |

a1's Pauli regression reproduced independently at **+6.94** against the +7.01 of the earlier job, and
the fix **inverts** it to −4.25. Time improves too: 1.0084× vs a1's 1.0237× (hub), 1.0262× vs 1.0387×
(pau, 3/3).

Two exact confirmations of the mechanism: **TOUCHED is byte-identical** between a1 and a1+fix
(51.702/51.702 hub, 61.812/61.813 pau), so no resident payload moved and the whole win is transient;
and **arena slack is exactly zero**, with `invidx` falling by 0.452 / 1.032 B/term — precisely the whole
arena-slack line, not a rounding of it.

The number that explains three rounds of this campaign: on Pauli **the transient the fix removes
(11.18 B/term) is larger than a1's entire at-rest win (7.61)**. That is why shrinking the index by 40%
read as a memory regression.

**On the Hubbard agree count of 2/3:** the dissenting rep is a bad *denominator* — main's rep 3 ran
33.19 s against its own 28.57/28.84, while both branch arms sat at 29.1–29.5 s in all three reps. Never
read an agree count without checking whether the shared baseline moved. `membisect_summary.py` now
prints the baseline's per-rep seconds beside the ratios for exactly this reason.

---

## 5. Stage 2 — the candidate, and a rule that splits on `P`

The candidate is a1 + a2 + the fix (`a1+a2` is the only zero-conflict prefix; a3 measured 5.52% on
Hubbard and is excluded, and a4 cannot be cherry-picked past it). Gated first: **212/212 `ctest -L
unit`, 211/211 `-L serial`**, `.so` md5 `ea724dc4515f4138f363d8d0408729cc`. Then one arm against main,
4 reps paired, both cells, both `P`.

| | peak RSS | TOUCHED | time |
| --- | ---: | ---: | ---: |
| **R=1** hub | 1.073× (6503.0 → 6063.2 MB) | **1.200×** (59.149 → 49.271) | 1.0346× (4/4) |
| **R=1** pau | 1.142× (7474.6 → 6543.0) | **1.184×** (70.155 → 59.238) | **1.0748× (4/4)** |
| **R=8** hub | 1.078× (11196.6 → 10389.4) | **1.189×** (59.087 → 49.702) | **1.023×** (3/3) |
| **R=8** pau | 1.049× (9312.1 → 8875.8) | **1.160×** (69.423 → 59.857) | **1.030×** (3/3) |

The byte half passes everywhere (53% hub / 68% pau of the full branch's touched win). The time half is
at the bar at production R=8 and a clear miss at R=1 on Pauli. It is **not** noise: 4/4 with a tight
spread (1.0718 1.0763 1.0733 1.0766). Both numbers are true; the R=8 figure is the wait bucket
absorbing the index's cost, the same mechanism that made R=8 understate a1 by 3×.

The two R=8 outlier reps are unrelated and illustrate the distinction in §4: hub rep 2 was a bad
**denominator** (main 32.84 s against its own 28.56/28.32/28.53), pau rep 2 a bad **arm** (candidate
20.35 s against its own 15.88/15.47/15.59). Different arms, same rep index — a node perturbation.

---

## 6. What the residual cost is *not*

The candidate's Pauli R=1 cost is **unattributed**. Two mechanisms were proposed and both were refuted
on measurement; this is recorded so neither is proposed again.

**Refuted: the U8Delta scalar decode.** `monoprop_INVIDX_BITMAP_PREMIUM` (in sixteenths; 16 is the
shipped argmin, 1000000 the maximum `parse_positive_int` will accept) forces chunks toward bitmaps.
Pauli R=1, 4 reps, knob verified to have moved — delta bytes fell monotonically 225,555,041 → 887,259:

| arm | delta bytes | invidx B/t | peak RSS | time vs main |
| --- | ---: | ---: | ---: | ---: |
| p016 | 225.6M | 8.776 | 6325.8 | 1.1344× 4/4 |
| p064 | 33.6M | 10.175 | 6468.1 | **1.1141×** 4/4 |
| p256 | 5.2M | 12.864 | 6634.5 | 1.1179× 4/4 |
| p1024 | 1.4M | 12.879 | 6719.7 | 1.1183× 4/4 |
| p4096 | 0.97M | 14.478 | 6781.8 | 1.1242× 4/4 |
| pinf | 0.89M | 14.478 | 6824.9 | 1.1182× 4/4 |

Removing **99.6% of the delta bytes recovers 1.8% of an 11.3% regression**, and nothing past p064
helps. The per-chunk cycle reasoning was right; the estimate of how much runtime that path owns was
wrong. It also re-prices deleting the U8Delta container at **+5.702 B/term**, against a pre-committed
ship-only-below +2.000 — so the container stays, and 64% of a1's whole win is not handed back to
accelerate a phase capped at ~2.3% of wall.

**Refuted: index construction.** `index_s` — the inverted-index rebuild/append, never previously
attributed — measures **0.0003 s (main) vs 0.0004 s (branch)**: 0% of the regression.

**Bounded: the index read path, at ~2.4%.** Per-partition phase split at Pauli R=1, total +2.32 s
(1.100× wall):

| phase | main | branch | Δs | share |
| --- | ---: | ---: | ---: | ---: |
| `incoming_s` | 3.390 | 4.283 | +0.893 | 38% |
| `exchange_s` | 4.305 | 5.021 | +0.716 | 31% |
| `fold_s` | 0.335 | 0.893 | +0.558 | **24%** |
| `emit_s` | 12.790 | 13.011 | +0.221 | 10% |
| `index_s` | 0.0003 | 0.0004 | +0.0001 | 0% |
| `scan_s` | 0.680 | 0.589 | −0.091 | −4% |

`fold_s` at 2.4% of wall independently corroborates the λ sweep's 1.8% — two methods, one number.

**But this table cannot be subtracted onto the candidate, and that is a second design error worth
recording.** 69% of the regression sits in `incoming_s` + `exchange_s`, and `resolve_incoming` goes
through the *dedup table* — a3's territory, which the candidate excludes. The instrumented pair was
`main+timers` vs the **full branch**+timers, because that pair already existed, not because it matched
the subject. An exact table that answers the wrong question. `layerprof_summary.py` now carries this
warning in its docstring.

So: ~5% of the Pauli R=1 cost is unexplained on a commit whose only job is to change the index. One
scale clue survives: Pauli carries ~1.9× Hubbard's postings per term (w̄ 16.91 vs 8.90) and shows ~2×
the regression (7.5% vs 3.5%), so the cost is posting-proportional.

---

## 7. Closed by measurement

Recorded with the number that closed each, so none is re-proposed.

| item | why it is closed |
| --- | --- |
| Bit-level / run-coded postings | Mean run length 1.510 / 1.820; only 14.3% / 13.0% of postings sit in runs ≥8. A 4–6 B header eats it at every threshold. |
| The codec search generally | The index is at **4.2–5.8 bits/posting**, not 8 — 79–84% of postings live in bitmap cells costing 3.4–5.0 bits each. Elias-Fano is +2 bits/element flat and loses outright on Pauli; patched packing lands ~4.75; BIC/ANS are 5× our decode on a phase capped at 2.3%. Pricing against the 1/8 crossover instead of the actual density inflated this prize by 1.4×. |
| λ, the bitmap premium | Free at R=1 (0.9881× 4/4) and refuted at production R=8 (1.0047× 3/4). A threshold tuned at one `P` does not transfer. Default stays 16. |
| Per-(generator,row) fold-bit cache | 52.0 / 33.9 B/term — 7.4× and 3.4× the entire index — for a ≤2.3% prize. The 29×/20× recompute redundancy is real; the memory is not affordable. |
| Vectorising the fold's scatter | Impossible at ISA level: no vector read-modify-write to memory in any x86 extension, `VPSCATTERDD` drops overlapping writes, CRoaring ships scalar `bts` asm for exactly this. Zen 2 has no AVX-512 regardless. |
| Deleting U8Delta | +5.702 B/term (§6), against a pre-committed ≤+2.000. Three independent estimates agreed: density histogram +4.87, this repo's own `M_STAR` table +4.94, the λ frontier. |

**Still open, and the largest remaining lever:** the **Pauli row store**. `w̄ = postings/rows` is
already on every `LAYERGAP` line — 8.90 against a permitted 10 on Hubbard (tight, 1.01×, dead) and
**16.91 against 28 on Pauli** (loose by construction, 1.53×). A Length cutoff bounds exactly what the
row stores; a Support cutoff bounds *qubits* while the row stores *slots*, and a weight-`w` Pauli
occupies between `w` and `2w`. Worth 10.2 B/term = **18% of the Pauli operator**, larger than anything
left in the index. Cheapest first experiment is one expression — return a ~99th-percentile `w` from
`packed_inline_width_()` and let the existing lossless `overflow_` spill take the rest — but price the
spill first: it is a node map at `sizeof(Monomial) + ~32 B` per entry, so below ~99% it eats the saving.

---

## 8. Reproducing this

Not reproducible from the library diff — it needs this branch's harness.

```bash
# per-commit memory + paired time, both P
ARMS_FILE=$PROJ/runs/<campaign>/arms RANKS=8 REPS=4 \
    sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" hpc/deucalion/sbatch/membisect.sh
ARMS_FILE=... RANKS=1 REPS=4 sbatch ...   # neither P answers for the other

$VENV/bin/python hpc/deucalion/tools/membisect_summary.py --baseline main \
    --cumulative main,a1-index,a2-release,a3-dedup,a4-rowstore,full $PROJ/runs/membisect-<jobid>

# the phase split (PROFILE=1 needs a build carrying LayerProfile.h)
$VENV/bin/python hpc/deucalion/tools/layerprof_summary.py --baseline main --arm port <dir>
```

Cells are `cells/100m.cells`. Gate an arm with `sbatch/ctest-worktree.sh` before measuring it, and if a
case fails once, establish its rate with `sbatch/ctest-repeat.sh` before calling it a regression — see
README §12.

**On the ≤1.02× acceptance bar** in `RESULTS-invidx-memory.md:390-405`: the candidate measures 1.023×
(hub) and 1.030× (pau) at production R=8, and misses at R=1. That is a miss and should be put as one —
but it is a ~3% miss carrying half the memory win, against a 10.5% miss carrying all of it, and the bar
was written when the alternative to "repair" was "ship everything".
