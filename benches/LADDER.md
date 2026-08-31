# Comprehensive Benchmarking

## Which group

| a change to | run |
| --- | --- |
| a kernel, encoding or data structure | L1, then L2. Hubbard makes 29 `propagate` calls and pauli 1, so a per-call cost shows on one and hides on the other |
| the graph build or contraction | L2's graph and eval rows, both pictures |
| MPI, routing, exchange or placement | L4. A communication cost is a function of `P` = ranks × partitions, not of node count |
| memory layout or allocation | L1 **and** L2. The per-term and the per-process terms of the memory cost only separate across sizes |
| ranks per node, or anything paid per process | L2 against L3 — same cores per node, 1 rank against 8 |

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

Run one group of operations per process at scale: `build_graph`/`propagate` and `energy`/`gradient`
each hold their own operator, and selecting all four holds two per rank.


## L1 — one thread, ~10M terms

One node, one rank, one partition: `export monoprop_PARTITIONS=1 monoprop_NUM_THREADS=1`.
Removes every threading and MPI effect, so what moves is the kernel.

| row | flags | `-k` | terms | ~s | ~GiB |
| --- | --- | --- | ---: | ---: | ---: |
| hubbard `propagate` | `--hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05` | `test_model_propagate and hubbard` | 9,953,109 | 30 | 1 |
| pauli `propagate` | `--pauli-cutoff=12 --pauli-lower-atol=1.22e-04` | `test_model_propagate and pauli` | 10,069,308 | 20 | 1 |

## L2 — one node, ~1B terms

One node, one rank, every core: `export monoprop_PARTITIONS=128 monoprop_NUM_THREADS=128`
(substitute your core count). The random rows are the only ones carrying all four operations, so
they are where the operations separate.

| row | flags | `-k` | terms | ~s | ~GiB |
| --- | --- | --- | ---: | ---: | ---: |
| hubbard `propagate` | `--hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06` | `test_model_propagate and hubbard` | 1,001,661,534 | 140 | 60 |
| pauli `propagate` | `--pauli-cutoff=14 --pauli-lower-atol=8.9e-06` | `test_model_propagate and pauli` | 985,970,588 | 90 | 80 |
| random `propagate` | `--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=14750000` | `test_random_propagate and heisenberg` | 948,937,993 | 50 | 95 |
| random `build_graph` | *(same)* | `test_random_build_graph and heisenberg` | 948,937,993 | 50 | 110 |
| random `energy`+`gradient` | *(same)* | `(test_random_energy or test_random_gradient) and heisenberg` | 948,937,993 | 40 | 110 |
| Schrödinger `build_graph` | *(same)* | `test_random_build_graph and schrodinger` | 18,825,440 | 1.5 | 17 |
| Schrödinger `energy`+`gradient` | *(same)* | `(test_random_energy or test_random_gradient) and schrodinger` | 18,825,440 | 0.3 | 11 |

## L3 — four nodes, the same problems

Same flags, same term counts, same cores per node as L2; 8 ranks per node × 16 partitions instead
of 1 × 128, so `R` = 32 and `P` = 512. Directly comparable to L2 row for row, which is the point of
having both.

| row | ~s | ~GiB/node | against L2 |
| --- | ---: | ---: | --- |
| hubbard `propagate` | 35 | 18 | 4.1x faster, 3.2x less memory |
| pauli `propagate` | 23 | 25 | 4.1x faster, 3.3x less memory |
| random `propagate` | 13 | 100 | 3.8x faster, **0.95x — slightly worse** |
| random `build_graph` | 13 | 100 | 3.9x faster, **1.05x — flat** |
| random `energy`+`gradient` | 9 | 60 | 4.3x faster, 1.8x less memory |
| Schrödinger `build_graph` | 1.1 | 80 | 1.3x faster, **4.8x more memory** |
| Schrödinger `energy`+`gradient` | 0.15 | 40 | 2.0x faster, **3.5x more memory** |


## L4 — strong and weak scaling

Hubbard `propagate` only: a communication cost is a function of `P`, so varying the model as well
would confound it. 8 ranks per node × 16 partitions throughout, so `R` = 8N and `P` = 128N.

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

Three numbers: nodes `N`, ranks per node `R`, partitions per rank `P`, one core per partition —
so `R × P` is the cores per node.

One rank over a whole node — L1 (with `P`=1) and L2:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --exclusive --mem=0 --time=0:30:00
set -euo pipefail
export monoprop_PARTITIONS=128 monoprop_NUM_THREADS=128
just bench L2-hubbard-branch --hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06 \
    -k "test_model_propagate and hubbard"
```

`N` nodes at 8 ranks × 16 partitions — L3 and L4. This is the same command with `srun` inserted
and the report moved out of it:

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
- **Export `monoprop_PARTITIONS`.** The engine's own default is `ranks == 1 ? cores : 1`, so an
  unset knob on a multi-rank run measures one partition per rank.
- **`--cpu-bind=none` cost 1.45x** against Slurm's own cgroup pinning at 8 ranks per node. With one
  rank holding a whole node there is nothing to confine and it stops mattering.
- **`nproc` lies inside a job** — it honours `OMP_NUM_THREADS`, which the BLAS pinning sets to 1,
  so a job holding 128 CPUs can log one core. Read `os.sched_getaffinity(0)`.
- **Peak RSS follows capacity, not live bytes**, so a `shrink_to_fit` can raise it: `realloc` holds
  the old and the new buffer at once.
- **`--hubbard-observable-site` (46) must stay below `--hubbard-num-sites` (60).** Shrinking the
  model without moving the observable fails.
