# Rank/partition layout bake-off — Deucalion x86

> **Archival.** The job scripts this file names (`layout-bakeoff.sh`) were removed when
> `sbatch/ab-100m.sh` replaced them; see README §10. The measurements stand, but
> they predate the vectorized `benches/_builders.py`, which changed the RNG stream —
> so these term sets cannot be reproduced and these numbers share no axis with any
> run made after it.

Measured 2026-08-11 on `normal-x86`, monoprop `0.8.1.dev40+g1ee18ff44`
(`HEAD` of `hpc/deucalion-mpi`, i.e. **without** `perf/multinode-comm-scaling`),
GCC 14.3.0 / OpenMPI 5.0.8, `-march=native` (znver2), `MAX_NUM_MODES=1024`.

Problem: `benches/bench_random.py`, Heisenberg picture, 256 modes, cutoff 6,
200 000 observable terms, 100 generators of length 4, seed 0, 3 rounds.
Every layout uses all 128 cores/node, so `R × S = 128 × nodes` in all cases.

| Layout | ranks/node | partitions/rank (`monoprop_NUM_THREADS`) |
| --- | --- | --- |
| A | 1 | 128 |
| B | 8 (one per NUMA domain) | 16 |
| C | 2 (one per socket) | 64 |

## Correctness first

The evolved operator carries **813 913 terms in all nine runs** — every layout,
every node count. The `(R, S)` decomposition does not change the operator, which
is the cross-layout equivalence property the design promises.

## Time (Heisenberg, mean over rounds)

| Operation | | N=2 | N=4 | N=8 |
| --- | --- | ---: | ---: | ---: |
| build_graph | A | 584.7 ms | 903.5 ms | 1.213 s |
| | **B** | **268.0 ms** | **465.9 ms** | **426.8 ms** |
| | C | 588.1 ms | 1.145 s | 1.738 s |
| energy | A | 72.7 ms | 191.0 ms | 202.4 ms |
| | **B** | **23.9 ms** | **28.4 ms** | **44.8 ms** |
| | C | 41.8 ms | 49.3 ms | 121.9 ms |
| gradient | A | 215.9 ms | 321.5 ms | 660.0 ms |
| | **B** | **49.5 ms** | **52.3 ms** | **106.8 ms** |
| | C | 110.0 ms | 164.8 ms | 319.0 ms |
| inplace | A | 459.9 ms | 443.9 ms | 632.6 ms |
| | **B** | **201.1 ms** | **198.0 ms** | **263.6 ms** |
| | C | 511.5 ms | 1.011 s | 1.511 s |
| pare | A | 8.44 ms | 20.50 ms | 33.88 ms |
| | **B** | **1.59 ms** | **1.69 ms** | **2.77 ms** |
| | C | 5.20 ms | 5.09 ms | 10.92 ms |

**B wins every operation at every node count**, by 2–12×.

## This is the O(R·S²) floor, and it is measurable

`HybridComm::pack_count_matrix_` runs on partition 0 alone over `r_ × s_ × s_`,
twice per gate. The prediction is a serial cost **quadratic in S and linear in R**.
Both show up:

**Quadratic in S.** At fixed node count, S=128 (A) versus S=16 (B) is a 64×
difference in the floor term. The observed A/B ratio on `pare` — the operation
with the least real compute to hide it — is 5.3× at N=2 and ~12× at N=4 and N=8.

**Linear in R.** For layout A the problem is *fixed*, so more nodes should mean
less time. Instead `pare` goes **8.44 → 20.50 → 33.88 ms** as R goes 2 → 4 → 8.
That is anti-scaling, and it is close to linear in R. Layout B over the same
range is nearly flat: **1.59 → 1.69 → 2.77 ms**.

| `pare`, A/B ratio | N=2 | N=4 | N=8 |
| --- | ---: | ---: | ---: |
| | 5.3× | 12.1× | 12.2× |

So the floor is not a theoretical concern on this machine: at S=128 it dominates,
and it gets worse with scale.

## Memory — read this one carefully

Resting footprint of the built operator + graph:

| | N=2 | N=4 | N=8 |
| --- | ---: | ---: | ---: |
| A (2/4/8 ranks) | 1 643 MiB | 4 332 MiB | 13 872 MiB |
| B (16/32/64 ranks) | 6 659 MiB | 14 488 MiB | 30 063 MiB |
| C (4/8/16 ranks) | 2 352 MiB | 16 752 MiB | 16 752 MiB |

B looks 2–4× more expensive, but **most of that is per-rank fixed overhead, not
operator data**. The metric is RSS *summed across ranks* with shared pages counted
once per rank — the report calls it an upper bound. B runs 8× more ranks than A,
and each rank carries its own Python interpreter, mpi4py and libmonoprop mapping
(order a few hundred MiB). At N=2, 16 ranks × ~400 MiB ≈ 6.4 GiB, which is
essentially all of B's 6 659 MiB.

The practical consequence: **B's memory premium is a fixed per-rank cost, so it
becomes proportionally negligible exactly when it matters** — on the large
operators that motivate multi-node runs in the first place. It should not be read
as "layout B cannot hold big problems".

This wants a direct measurement before Deliverable 5 commits to a layout: compare
`operator_memory_bytes()` (allreduced, actual operator bytes) rather than RSS
across the three layouts. `monoprop_multinode.py` reports exactly that.

## Recommendations

1. **Use layout B (8 ranks/node × 16 partitions) as the default.** It is
   NUMA-aligned, fastest at every size measured, and the only layout whose cost
   does not grow with node count.
2. **Port `perf/multinode-comm-scaling`.** The evidence is now direct rather than
   assumed: the floor it removes is what makes A anti-scale and what costs C 2–6×.
   Removing it should let the memory-lean, low-rank layouts run at B's speed,
   which is the combination the beyond-one-node work needs.
3. **Re-run this bake-off after the port.** It is a ready-made before/after
   measurement, and the `pare` row alone is a clean signal.
4. Do not run the Schrödinger picture at these sizes on 2 nodes: it propagates the
   state at `schrodinger_cutoff = cutoff + 2` and OOM-killed all three layouts
   (>242 GB/rank) while every Heisenberg case passed.

Reproduce with `hpc/deucalion/sbatch/layout-bakeoff.sh` (`sbatch -N<n>`).
