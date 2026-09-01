# Comprehensive Benchmarking

## Summary

| Group | What it measures |
| --- | --- |
| L1  | Single thread, 20M terms, 5GB |
| L2  | Single-node perf, default user settings without MPI `1xcores`-shape. 0.95B terms Heisenberg, 0.60B Schrödinger, both ≤110GB|
| L3  | Multi-node, same problem as L2|
| L4  | Strong/weak scaling. If L3 looks okay, don't bother running |

## Running one row

```bash
uv sync --all-groups --all-extras          # once
just bench L1-hubbard --hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05 \
    -k "test_model_propagate and hubbard"
```

`LABEL` is one column in `benches/results/REPORT.md`, so name it for the group, the row and the
arm: `L1-hubbard-branch`, `L1-hubbard-base`. Each run writes `results/time-<label>.json`
(timings) and `results/<label>.json` (term counts, peak RSS, operator memory), then rebuilds the
report. Reps are repeated invocations under different labels; leave `--bench-rounds` at 1, since
`pedantic` builds round *k+1*'s arguments before releasing round *k*'s and a second round holds two
propagators.

`build_graph`, `energy` and `gradient` may be selected together in one process: the timed
`build_graph` publishes the graph it built and the two functionals evaluate that one, so the
process builds a single operator. Keep `propagate` out of that selection — it holds its own
operator, so adding it to the other three holds two per rank.

## Declaring the shape

Nodes `N`, ranks per node `R`, partitions per rank `P`, one core
per partition — so `R × P` is the cores per node and `P_total` = `N × R × P` is what a
communication cost is a function of. **`N` and `R` come from the launcher; `P` you must export.**

```bash
export monoprop_PARTITIONS=16 monoprop_NUM_THREADS=16
```

Above one rank this is mandatory: `benches/conftest.py` refuses to start a multi-rank session with
`monoprop_PARTITIONS` unset, because the engine's own default is `ranks == 1 ? cores : 1` and an
unset knob would silently measure one partition per rank at a plausible wall time.

The shape is recorded with every run — `ranks`, `nodes`, `ranks_per_node`, `partitions_env` and
`monoprop_threads` in `results/<label>.json`'s `meta`, surfaced as the Configuration table of
`REPORT.md`. Check it before comparing two rows: a ladder whose points did not all run at the same
`R` and `P` is not a ladder.

## Model size knobs

Every size below is an input, not a constant. The fixed models take `--<model>-<field>` for any
field of their config dataclass (`just bench --help` lists them); the random problem takes
`--num-generators`, `--num-modes`, `--cutoff`, `--obs-terms`, `--gen-length` and `--seed`.

| knob | range | note |
| --- | --- | --- |
| `--num-modes` | ≤ this build's `MAX_NUM_MODES` | rejected above it, rather than failing inside the extension |
| `--hubbard-num-sites` | 2 × sites ≤ `MAX_NUM_MODES` | saturated from 60 at the default `lower_atol`; size Hubbard with `--hubbard-lower-atol` |
| `--hubbard-observable-site` | `< --hubbard-num-sites` | a lattice position, not a constant: the default 46 sits 46/60 along the lattice, and shrinking the lattice without moving it changes the light cone. Out of range now raises |
| `--pauli-num-qubits` | 127 only | `HEAVY_HEX_TOPOLOGY` is the fixed IBM Eagle coupling map, so any other count either indexes past the operator or leaves qubits uncoupled. Raises. Size this model with `--pauli-lower-atol` |
| `--obs-terms` | any | an upper bound on the observable's terms, not an exact count: monomials are drawn independently and duplicates collapse, by about `obs_terms / 2·C(2·num_modes, gen_length)` — 0.06% at L1's 295k, 2.8% at L2's 14.75M. Deterministic for a fixed `--seed`. **Heisenberg only** — see *Sizing the Schrödinger rows* |
| `--num-generators` | any | the circuit's gate count, and the only usable size axis in the Schrödinger picture. It is also the length of the parameter vector, so `gradient` at 5800 generators does nearly six times the per-term work it does at 1000 |
| `--gen-length` | `k % 4` in {0, 1} | a product of `k` Majoranas satisfies `(g_1...g_k)^dag = (-1)^(k(k-1)/2) g_1...g_k`, so it is Hermitian with the real coefficients drawn here only at those lengths. Verified over k=2..11: 4, 5, 8, 9 build and 2, 3, 6, 7, 10, 11 raise. Out of range now raises naming the option, rather than surfacing as `Non-Hermitian coeffs detected` from inside the extension |
| `--pare-threshold` | any, default off | edge-retention cutoff for the graph functionals (`energy`, `gradient`). Unset keeps the exact graph |

### Sizing the Schrödinger rows

The Schrödinger picture evolves the state, so the observable does not drive the operator:
`--obs-terms` is flat there (18.85M / 18.75M / 18.83M across a 74× sweep) and sizes the Heisenberg
rows only. Of the remaining axes, two are cliffs rather than dials — one step of either goes from a
few GiB to not fitting a 242 GiB node, because what grows is a transient inside `build_graph` and
not the retained operator:

| axis | last value that fits | next value tried |
| --- | --- | --- |
| `--cutoff` | 6 — 18,853,861 terms, 3.6 GiB | 7 — SIGKILL |
| `--num-modes` | 160 — 29,082,430 terms, 5.6 GiB | 180 — SIGKILL |

`--num-generators` is the axis that works. At `--cutoff=6 --num-modes=142 --obs-terms=200000`,
one node at 1×128. Term counts are geometry- and operation-independent; the time and memory
columns are not, so each row says what was measured — `build` is `build_graph` alone, `row` is all
three operations in one process, which is what has to fit:

| `--num-generators` | terms | ~s | ~GiB | measured |
| ---: | ---: | ---: | ---: | --- |
| 1000 | 18,853,861 | 1.3 | 3.6 | build |
| 2000 | 24,707,186 | 3.2 | 4.2 | build |
| 3000 | 42,566,127 | 6.1 | 7.2 | build |
| 4000 | 97,631,757 | 13.3 | 16.2 | build |
| 5000 | 261,929,069 | 30.6 | 32.2 | build |
| 5600 | 485,959,982 | 97.9 | 78.0 | row |
| **5800** | **597,445,055** | **115.9** | **99.7** | **row** |
| 6000 | 737,159,992 | 142.8 | 115.1 | row |
| 6200 | 907,357,991 | 172.1 | 143.1 | row |
| 6400 | 1,115,757,089 | 117.0 | 138.5 | build |
| 8000 | — | — | > 242 (SIGKILL) | build |

L2 and L3 take the 5800-generator row: it is the largest whose whole-row peak fits the ~110 GiB
one-node budget the other L2 rows are held to. 6000 (115.1 GiB) and 6200 (143.1 GiB) clear 1B terms
at 6400 but not the budget.

**Size this rung against `gradient`, not `build_graph`.** At 6200 generators `build_graph` peaks at
102.7 GiB and looks like it fits, while `gradient` in the same process peaks at 143.1 GiB. Two
things make the last operation the expensive one, and only the second is about the gradient:

- The three operations share a process, and peak RSS follows capacity rather than live bytes, so an
  operation starts at the high-water mark its predecessors left. Measured at 6200: `gradient`'s
  floor was 114.31 GiB against `energy`'s peak of 114.32 — the same bytes, not new ones. Of that
  143.1 GiB row peak only 28.8 GiB is `gradient`'s own transient (`opmemdelta`).
- That transient still tracks the *circuit*, not the operator: 28.8 GiB at 6200 generators against
  7.9 GiB at 1000, on an operator that is slightly smaller (907M terms against 949M). Sizing this
  picture by gate count therefore buys terms and gradient memory together.

The axis accelerates — ×1.31, ×1.72, ×2.29, ×2.68 per extra 1000 gates — so read a size off this
table rather than interpolating, and do not size the rung from two points: 1000→2000 alone
predicts ~73,000 generators for 100M terms, where 4000 already reaches it. The counts are
deterministic at a fixed `--seed`; 4000 reproduced to the term across two jobs.

**A gate count is not a free parameter here.** `--num-generators` is also the length of the
parameter vector, so the 5800-generator Schrödinger rows return a 5800-long gradient where the
1000-generator Heisenberg rows return a 1000-long one. The two pictures' `gradient` cells are
matched in terms and in memory, not in the work the functional does; do not read one against the
other as a per-term ratio.

## L1 — one thread

One node, one rank, one partition: `export monoprop_PARTITIONS=1 monoprop_NUM_THREADS=1`.
Removes every threading and MPI effect, so what moves is the kernel.

The two `propagate` rows sit at ~10M terms and the two `gradient` rows at ~20M. The gradient rows
are the same problem twice, once against the exact graph and once against a plan pared at
`1e-10`, so a change to the functionals or to the masked plan shows on the pair rather than on
either alone.

| row | flags | `-k` | terms | ~s | ~GiB |
| --- | --- | --- | ---: | ---: | ---: |
| hubbard `propagate` | `--hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05` | `test_model_propagate and hubbard` | 9,953,109 | 30 | 1 |
| pauli `propagate` | `--pauli-cutoff=12 --pauli-lower-atol=1.22e-04` | `test_model_propagate and pauli` | 10,069,308 | 20 | 1 |
| random `gradient` | `--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000` | `test_random_gradient and heisenberg` | 19,902,244 | 11 | 2.5 |
| random `gradient`, pared | *(same)* `--pare-threshold=1e-10` | `test_random_gradient and heisenberg` | 19,902,244 | 0.8 | 2.8 |

## L2 — one node, ~1B terms

One node, one rank, every core: `export monoprop_PARTITIONS=128 monoprop_NUM_THREADS=128`
(substitute your core count). The random rows are the only ones carrying all four operations, so
they are where the operations separate.

| row | flags | `-k` | terms | ~s | ~GiB |
| --- | --- | --- | ---: | ---: | ---: |
| hubbard `propagate` | `--hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06` | `test_model_propagate and hubbard` | 1,001,661,534 | 140 | 60 |
| pauli `propagate` | `--pauli-cutoff=14 --pauli-lower-atol=8.9e-06` | `test_model_propagate and pauli` | 985,970,588 | 90 | 80 |
| random `propagate` | `--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=14750000` | `test_random_propagate and heisenberg` | 948,937,993 | 50 | 95 |
| random `build_graph` | *(same)* | `test_random_build_graph and heisenberg` | 948,937,993 | 50 | 109 |
| random `energy` | *(same)* | `test_random_energy and heisenberg` | 948,937,993 | 9.9 | 102 |
| random `gradient` | *(same)* | `test_random_gradient and heisenberg` | 948,937,993 | 34 | 110 |
| Schrödinger `build_graph` | `--num-generators=5800 --num-modes=142 --cutoff=6 --obs-terms=200000` | `test_random_build_graph and schrodinger` | 597,445,055 | 66 | 73 |
| Schrödinger `energy` | *(same)* | `test_random_energy and schrodinger` | 597,445,055 | 11 | 79 |
| Schrödinger `gradient` | *(same)* | `test_random_gradient and schrodinger` | 597,445,055 | 39 | 100 |

Medians of three reps, arm order flipped between reps. `energy` and `gradient` are separate rows
because they are not one cost: `gradient` is 3.4x `energy` in the Heisenberg picture and 3.4x in
the Schrödinger one.

`build_graph`, `energy` and `gradient` share one process and one build — the graph the timed
`build_graph` produces is what the other two evaluate — so run them with one selector, not three.
`propagate` holds its own operator and stays out of that group.

**The Schrödinger rows carry their own flags**, differing only in the gate count. `--obs-terms`
cannot size that picture, so the rung is set by `--num-generators=5800` (see *Sizing the Schrödinger
rows*), which is the largest gate count whose whole-row peak stays inside the ~110 GiB budget the
Heisenberg rows sit at. Two consequences worth reading before comparing the pictures:

- The gradient's parameter vector is one entry per generator, so the Schrödinger `gradient`
  differentiates 5800 parameters against the Heisenberg row's 1000. Unequal work, not just unequal
  size.
- Memory per term is higher in the Schrödinger picture — 0.167 GiB/Mterm at the `gradient` peak
  against the Heisenberg rows' 0.115 — which is why matching the two pictures on memory leaves them
  at 597M against 949M terms rather than at equal counts.

## L3 — several nodes, the same problems

The L2 problems again, at your own `N`, `R` and `P`. Choose `R × P` equal to L2's core count and
the pair isolates what is paid per process rather than per core; choose anything else and it is
still a valid multi-node run, just not that comparison.

Same flags and same `-k` as L2, so only the measured cells differ. Term counts are
geometry-independent (`cpp/tests/partition_equivalence_tests.cpp`), which is why one calibration
serves every shape and a term-count disagreement means the *problem* changed, not the topology.

| row | terms | ~s | ~GiB/node | ~GiB worst rank |
| --- | ---: | ---: | ---: | ---: |
| hubbard `propagate` | 1,001,661,534 | 35 | 18 | |
| pauli `propagate` | 985,970,588 | 23 | 25 | |
| random `propagate` | 948,937,993 | 13 | 100 | |
| random `build_graph` | 948,937,993 | 14 | 105 | 13.1 |
| random `energy` | 948,937,993 | 2.3 | 61 | 7.7 |
| random `gradient` | 948,937,993 | 7.6 | 63 | 7.9 |
| Schrödinger `build_graph` | 597,445,055 | 20 | 26 | 3.3 |
| Schrödinger `energy` | 597,445,055 | 1.9 | 26 | 3.3 |
| Schrödinger `gradient` | 597,445,055 | 8.9 | 36 | 4.5 |

Measured at `N`=4, `R`=8, `P`=16 (medians of two reps) — a ballpark for sizing a job at that
shape, not a target. Term counts are identical to L2's, which is the geometry-independence check.

Memory per node does not fall the way time does, but there is **no fixed per-rank floor**: the
Heisenberg rows pay 13.1 GiB on their worst rank and the Schrödinger rows 3.3 GiB, on the same
shape. An earlier revision of this table quoted a ~9.5 GiB per-rank floor "paid whatever the
problem", which the Schrödinger rows here refute by a factor of four. Size a job from the row
you are running.

The Schrödinger rows are also where the flag split pays off: 26 GiB/node here against the
80 GiB/node an earlier revision recorded at 18.8M terms, at 32x the terms. That row inherited the
Heisenberg arm's 14.75M-term observable and every rank on the node held a copy of it, while it
contributed nothing to the evolved state.

The single-node budget, not this shape, is what sizes the Schrödinger rung: at `N`=4 its worst rank
holds 4.5 GiB and the row would fit many times over. L2 and L3 share one flag set so the term
counts stay comparable, so L2 is the binding constraint.

## L4 — strong and weak scaling

Hubbard `propagate` only: a communication cost is a function of `P_total`, so varying the model as
well would confound it. Hold `R` and `P` fixed across every point of a ladder and vary only `N`;
`P_total` is then `N × R × P`. The tables below were built at `R`=8, `P`=16.

Size is set by `--hubbard-lower-atol` at a fixed `--hubbard-cutoff=10`.

| k | `--hubbard-lower-atol` | terms |
| ---: | --- | ---: |
| 0 | `1.25e-05` | 96,981,051 |
| 1 | `8.8e-06` | 184,124,520 |
| 2 | `5.9e-06` | 377,482,074 |
| 3 | `3.9e-06` | 781,669,404 |
| 4 | `2.6e-06` | 1,569,152,761 |
| 5 | `1.73e-06` | 3,104,527,573 |
| 6 | `1.14e-06` | 6,125,805,627 |
| 7 | `7.35e-07` | 12,255,330,837 |
| 8 | `4.68e-07` | 24,419,998,198 |
| 9 | `2.94e-07` | 48,317,129,677 |
| 10 | `1.82e-07` | 94,684,031,363 |

**Weak** — hold terms per node constant: **step one row down the table each time you double the
node count.** Three ladders, each seven points from N=1 to N=64:

| ladder | starts at | Mterms/node | ~s at N=1 → N=64 |
| --- | --- | ---: | --- |
| ~97M/node | k=0 at N=1 | 97 | 11 → 37 |
| ~385M/node | k=2 at N=1 | 385 | 48 → 80 |
| ~1529M/node | k=4 at N=1 | 1529 | 216 → 257 |

That spread **is** the weak-scaling result; ideal scaling would be flat. Efficiency is
`t(N=1)/t(N)`.

**Strong** — hold one k and vary N. Three ladders:

| ladder | k | terms | N | ~s at smallest N → N=64 |
| --- | ---: | ---: | --- | --- |
| 1.57e9 fixed | 4 | 1,569,152,761 | 1…64 | 216 → 24 |
| 6.13e9 fixed | 6 | 6,125,805,627 | 2…64 | 450 → 37 |
| 2.44e10 fixed | 8 | 24,419,998,198 | 8…64 | 471 → 80 |

Efficiency against the ladder's smallest node count `N0` is `N0 · t(N0) / (N · t(N))`. The larger
two ladders start above one node because the problem does not fit below it, not because the point
was skipped.

## Shapes

One rank over a whole node — L1 (with `P`=1) and L2:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --exclusive --mem=0 --time=0:30:00
set -euo pipefail
export monoprop_PARTITIONS=128 monoprop_NUM_THREADS=128
just bench L2-hubbard-branch --hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06 \
    -k "test_model_propagate and hubbard"
```

`N` nodes at `R` ranks × `P` partitions — L3 and L4. This is the same command with `srun` inserted
and the report moved out of it; substitute your own `N`, `R` and `P` in all four places:

```bash
#!/bin/bash
#SBATCH --nodes=4 --ntasks-per-node=8 --cpus-per-task=16 --exclusive --mem=0 --time=1:00:00
set -euo pipefail
export monoprop_PARTITIONS=16 monoprop_NUM_THREADS=16
export monoprop_BENCH_LABEL=L3-hubbard-branch monoprop_BENCH_RESULTS=benches/results
srun --ntasks-per-node=8 --cpus-per-task=16 --cpu-bind=cores \
    uv run --no-sync python -m pytest benches -o filterwarnings=default \
    --benchmark-json="benches/results/time-$monoprop_BENCH_LABEL.json" \
    -k "test_model_propagate and hubbard" \
    --hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06
uv run --no-sync monoprop-bench-report benches/results
```

The report runs **outside** `srun`: under it, one process per rank would race to write the same
file. The two JSON artifacts are safe because only rank 0 writes them (`conftest.py`
`pytest_configure`). A ladder is a loop over sizes inside one allocation sized for its largest
point.

## Traps

Each of these produces a plausible wrong number rather than an error.

- **`srun --cpu-bind=cores` with no `--cpus-per-task` on the `srun` confines each task to one
  core.** Measured in one allocation: no flags → 128 CPUs, `--cpus-per-task=128` → 128,
  `--cpu-bind=cores` alone → 1. Repeat both flags on the `srun`, not only on the `#SBATCH`.
- **`--cpu-bind=none` cost 1.45x** against Slurm's own cgroup pinning at 8 ranks per node. With one
  rank holding a whole node there is nothing to confine and it stops mattering.
- **`nproc` lies inside a job** — it honours `OMP_NUM_THREADS`, which the BLAS pinning sets to 1,
  so a job holding 128 CPUs can log one core. Read `os.sched_getaffinity(0)`.
- **Peak RSS follows capacity, not live bytes**, so a `shrink_to_fit` can raise it: `realloc` holds
  the old and the new buffer at once.
- **A two-operation row's peak is the MAX over its operations, never the sum.** `HighWaterMark`
  resets `VmHWM` per benchmark, so both windows contain the same resident operator.
- Unset `monoprop_PARTITIONS` and out-of-range model sizes used to be traps of this kind. Both now
  raise; see *Declaring the shape* and *Model size knobs*.
