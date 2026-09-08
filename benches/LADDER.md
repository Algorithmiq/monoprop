# Comprehensive Benchmarking

This benchmark separates kernel, shared-memory, process, network, and scaling effects. Define:

- `N`: nodes
- `C`: usable cores per node
- `R`: MPI ranks per node
- `P`: in-process partitions per rank
- `P_total = N × R × P`: total partitions

Choose `R` for the platform, normally one rank per NUMA domain, then set `P = C / R`. Keep `R`
and `P` fixed wherever the ladder requires a direct comparison.

> **Example platform.** All flags, term counts, timings, and memory figures below were measured on
> code based on `main` commit `97f95f762dfc7174243ccd59dd4ecbbb9775b610`, on Deucalion x86
> nodes with `C=128`, using `R=8` and `P=16` for MPI runs. They are calibration and job-sizing
> examples, not portable targets but memory should be relatively similar.

## Ladder

| rung | shape | purpose |
| --- | --- | --- |
| L1 | `N=1, R=1, P=1` | isolate single-thread kernel performance |
| L2a | `N=1, R=1, P=C` | measure full-node shared-memory performance |
| L2b | `N=1, R=R_platform, P=C/R` | isolate process and MPI overhead without a network hop |
| L3 | `N>1`, same `R` and `P` as L2b | isolate the network |
| L4 | vary `N`, hold `R` and `P` fixed | measure strong and weak scaling |

L2a, L2b, and L3 use the same model flags. Only the execution shape changes. Term counts are
topology-independent, so a mismatch means the problem definition changed.

## Running a row

```bash
uv sync --all-groups --all-extras          # once
just bench L1-hubbard --hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05 \
    -k "test_model_propagate and hubbard"
```

Name `LABEL` by rung, row, and arm, for example `L1-hubbard-branch`. A run writes
`results/time-<label>.json`, `results/<label>.json`, and rebuilds `benches/results/REPORT.md`.
Use separate labels for repetitions and leave `--bench-rounds=1`; later rounds can overlap the
previous round's live propagator and inflate memory.

Run `build_graph`, `energy`, and `gradient` together when all three are needed. They share one
graph. Run `propagate` separately because it owns another operator.

## Declaring the shape

The launcher sets `N` and `R`; export `P` explicitly:

```bash
export monoprop_PARTITIONS=<partitions-per-rank>
export monoprop_NUM_THREADS=<cores-per-rank>
```

For multi-rank runs this is mandatory. The engine otherwise defaults to one partition per rank.
Before comparing results, check `ranks`, `nodes`, `ranks_per_node`, `partitions_env`, and
`monoprop_threads` in the report's Configuration table.

## Calibrating model size

| model | preferred size control | constraint |
| --- | --- | --- |
| Hubbard | `--hubbard-lower-atol` | `2 × num_sites <= MAX_NUM_MODES` |
| Pauli | `--pauli-lower-atol` | the supplied heavy-hex model requires 127 qubits |
| random, Heisenberg | `--obs-terms` | duplicates make this an upper bound; fix `--seed` |
| random, Schrödinger | `--num-generators` | also sets gradient length and work |


Calibrate each row on the target platform before running the ladder:

1. Fix the model structure and seed.
2. Sweep the preferred size control until the full row fits the desired memory budget.
3. Size graph rows against the largest peak of `build_graph`, `energy`, and `gradient`, not the
   graph build alone.
4. Reuse the chosen flags unchanged in L2a, L2b, and L3.

In the Schrödinger picture, sweep number of qubits or `--num-generators` instead.

## Example problem set

Examples 128-core platform.

### L1 example

Set `monoprop_PARTITIONS=1` and `monoprop_NUM_THREADS=1`.

| row | flags | selector | terms | time | peak RSS |
| --- | --- | --- | ---: | ---: | ---: |
| Hubbard `propagate` | `--hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05` | `test_model_propagate and hubbard` | 9.95M | 30 s | 1 GiB |
| Pauli `propagate` | `--pauli-cutoff=12 --pauli-lower-atol=1.22e-04` | `test_model_propagate and pauli` | 10.07M | 20 s | 1 GiB |
| random `gradient` | `--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000` | `test_random_gradient and heisenberg` | 19.90M | 11 s | 2.5 GiB |
| random `gradient`, pared | same, plus `--pare-threshold=1e-10` | `test_random_gradient and heisenberg` | 19.90M | 0.8 s | 2.8 GiB |

### L2a example

Set `monoprop_PARTITIONS=C` and `monoprop_NUM_THREADS=C`. On the example platform, `C=128`. These are medians of three repetitions.


| row | flags | selector | terms | time | peak RSS |
| --- | --- | --- | ---: | ---: | ---: |
| Hubbard `propagate` | `--hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06` | `test_model_propagate and hubbard` | 1.002B | 140 s | 60 GiB |
| Pauli `propagate` | `--pauli-cutoff=14 --pauli-lower-atol=8.9e-06` | `test_model_propagate and pauli` | 0.986B | 90 s | 80 GiB |
| random Heisenberg `propagate` | `--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=14750000` | `test_random_propagate and heisenberg` | 0.949B | 50 s | 95 GiB |
| random Heisenberg `build_graph` | same | `test_random_build_graph and heisenberg` | 0.949B | 50 s | 109 GiB |
| random Heisenberg `energy` | same | `test_random_energy and heisenberg` | 0.949B | 9.9 s | 102 GiB |
| random Heisenberg `gradient` | same | `test_random_gradient and heisenberg` | 0.949B | 34 s | 110 GiB |
| random Schrödinger `build_graph` | `--num-generators=5800 --num-modes=142 --cutoff=6 --obs-terms=200000` | `test_random_build_graph and schrodinger` | 0.597B | 66 s | 73 GiB |
| random Schrödinger `energy` | same | `test_random_energy and schrodinger` | 0.597B | 11 s | 79 GiB |
| random Schrödinger `gradient` | same | `test_random_gradient and schrodinger` | 0.597B | 39 s | 100 GiB |


## L2b: process overhead

Run the L2a problems on one node with `R × P = C`. Use the same `R` and `P` in L3. One rank per
NUMA domain is a portable starting point, not a guaranteed optimum.

Example results at `N=1, R=8, P=16`:

| row | time | RSS sum per node | worst-rank RSS |
| --- | ---: | ---: | ---: |
| Hubbard `propagate` | 169 s | 59.5 GiB | 7.5 GiB |
| Pauli `propagate` | 102 s | 85.6 GiB | 10.7 GiB |
| random Heisenberg `propagate` | 48.4 s | 160 GiB | 20.2 GiB |
| random Heisenberg `build_graph` | 53.2 s | 174 GiB | 21.8 GiB |
| random Heisenberg `energy` | 10.4 s | 131 GiB | 16.4 GiB |
| random Heisenberg `gradient` | 30.9 s | 139 GiB | 17.4 GiB |
| random Schrödinger `build_graph` | 67.3 s | 77.5 GiB | 9.8 GiB |
| random Schrödinger `energy` | 11.7 s | 81.3 GiB | 10.2 GiB |
| random Schrödinger `gradient` | 39.5 s | 102 GiB | 12.8 GiB |

The node sum is an upper bound because shared pages are charged to every rank. Use it for
provisioning, but do not compare it directly with a single process's peak. Per-rank overhead is
model-dependent, so size L2b from its own measurements.

## L3: network overhead

Run the L2b shape on multiple nodes without changing `R`, `P`, model flags, or selectors. The
L2b-to-L3 difference then isolates the network.

Example results at `N=4, R=8, P=16`:

| row | time | RSS sum per node | worst-rank RSS |
| --- | ---: | ---: | ---: |
| Hubbard `propagate` | 35 s | 18 GiB | — |
| Pauli `propagate` | 23 s | 25 GiB | — |
| random Heisenberg `propagate` | 13 s | 100 GiB | — |
| random Heisenberg `build_graph` | 14 s | 105 GiB | 13.1 GiB |
| random Heisenberg `energy` | 2.3 s | 61 GiB | 7.7 GiB |
| random Heisenberg `gradient` | 7.6 s | 63 GiB | 7.9 GiB |
| random Schrödinger `build_graph` | 20 s | 26 GiB | 3.3 GiB |
| random Schrödinger `energy` | 1.9 s | 26 GiB | 3.3 GiB |
| random Schrödinger `gradient` | 8.9 s | 36 GiB | 4.5 GiB |

## L4: strong and weak scaling

Use one representative operation and hold `R` and `P` fixed.

- **Strong scaling:** hold the problem fixed and vary `N`.
- **Weak scaling:** hold work per node approximately fixed while varying `N`.

For strong scaling relative to the smallest run `(N0, t0)`, efficiency is
`N0 × t0 / (N × t(N))`. For weak scaling relative to one node, it is `t(1) / t(N)`.

The example uses Hubbard `propagate` with `--hubbard-cutoff=10`. This platform-specific sweep
approximately doubles the term count at each step:

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

For a weak-scaling series that doubles `N`, advance one `k` per point. For strong scaling, choose
one `k` that fits the smallest node count and keep it fixed. On another platform, build a fresh
size sweep rather than treating this table as a target.

## Launch templates

For L1 and L2a, request one node and one rank. Set the core and partition counts from the granted
CPU affinity:

```bash
export monoprop_PARTITIONS="$C"
export monoprop_NUM_THREADS="$C"
just bench L2a-hubbard-branch --hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06 \
    -k "test_model_propagate and hubbard"
```

For L2b, L3, and L4, request `R` ranks per node and `P` cores per rank. This Slurm example assumes
the scheduler variables have already been set consistently:

```bash
export monoprop_PARTITIONS="$P"
export monoprop_NUM_THREADS="$P"
export monoprop_BENCH_LABEL=L3-hubbard-branch
export monoprop_BENCH_RESULTS=benches/results

srun --ntasks-per-node="$R" --cpus-per-task="$P" --cpu-bind=cores \
    .venv/bin/python -m pytest benches -o filterwarnings=default \
    --benchmark-json="benches/results/time-$monoprop_BENCH_LABEL.json" \
    -k "test_model_propagate and hubbard" \
    --hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06

uv run --no-sync monoprop-bench-report benches/results
```

Run the report outside the MPI launcher. Invoke the virtual-environment interpreter directly on
each rank to avoid repeated dependency resolution on shared filesystems.

## Validation and common errors

- Repeat the rank count, cores per rank, and binding options on the launcher command; scheduler
  allocation directives may not propagate them.
- Keep `P` no larger than the CPUs visible to each rank. Otherwise thread placement is disabled.
- Use the scheduler affinity mask, such as `os.sched_getaffinity(0)`, to verify visible CPUs;
  `nproc` may honor unrelated thread limits.
- Keep CPU binding enabled unless the target platform has been measured otherwise.
- Interpret peak RSS as a process high-water mark. Allocator capacity and temporary reallocation
  can dominate live object size.
- For grouped graph operations, compare the maximum peak across operations, not their sum.
- Verify term counts and the recorded shape before interpreting timing differences.
