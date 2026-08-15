# Multi-node scaling on Deucalion x86 — strong, weak, and where it stops

> **Archival.** The job scripts this file names (`strong-scaling.sh, weak-scaling.sh, memory-probe.sh`) were removed when
> `sbatch/ab-100m.sh` replaced them; see README §10. The measurements stand, but
> they predate the vectorized `benches/_builders.py`, which changed the RNG stream —
> so these term sets cannot be reproduced and these numbers share no axis with any
> run made after it.

Measured 2026-08-11 on `normal-x86`, monoprop `0.8.1.dev40+g1ee18ff44` (`HEAD`,
i.e. **without** `perf/multinode-comm-scaling`), GCC 14.3.0 / OpenMPI 5.0.8,
`-march=native` (znver2), `MAX_NUM_MODES=1024`.

Layout **B** throughout — 8 ranks/node × 16 partitions, one rank per NUMA domain,
the winner of [the bake-off](RESULTS-layout-bakeoff.md). Every node contributes
all 128 cores, so `R × S = 128 × nodes`.

> **Measurement provenance.** Every number here was taken *before* `env.sh` began
> pinning the BLAS thread pools (`OMP_NUM_THREADS=1` and friends), which it now
> does — nothing had been setting them, so each rank's numpy sized its pool by its
> own heuristic. The direct evidence says the effect on these numbers is small:
> the probe found `affinity=16 cores` and `live_threads=1`, i.e. the cgroup was
> respected and no pool had been spawned at that point, and monoprop's hot path is
> its own C++ kernels rather than BLAS. But it is **unquantified**, so treat this
> file as a baseline taken under the old environment: a before/after against the
> port must either re-run these rungs with the pinning in place or disable it, not
> compare across the change.

**Headline: monoprop on `HEAD` does not scale across nodes. Past roughly 8 nodes,
adding hardware makes a fixed problem slower in wall clock, and a weak-scaled
problem slower per unit of work.** The cause is measured, not inferred, and it is
the serial floor that `perf/multinode-comm-scaling` exists to remove.

> **The observation stands; the attribution does not.** Later profiling at 29M terms
> retired the serial `O(R·S²)` partition-0 prefix as a target (§3 carries the
> details and the numbers). The failure to scale is still measured here, but "the
> serial floor" is no longer a sufficient explanation of it, and the largest cost at
> production scale — the `MPI_THREAD_SERIALIZED` funnel — is not measured anywhere in
> this file. Do not carry the causal claim forward without re-measuring it.

---

## 1. Strong scaling — fixed problem, more nodes

`benches/bench_random.py`, Heisenberg, 256 modes, cutoff 6, 200 000 observable
terms, 100 generators, seed 0, 3 rounds. **813 913 terms in every run**, so the
work is genuinely identical across the ladder.

| Operation | N=1 (R=8) | N=2 (R=16) | N=4 (R=32) | N=8 (R=64) | N=16 (R=128) | N=32 (R=256) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| build_graph | 203.5 ms | 268.0 ms | 465.9 ms | 426.8 ms | 1.114 s | 2.219 s |
| energy | 22.0 ms | 23.9 ms | 28.4 ms | 44.8 ms | 73.8 ms | 122.6 ms |
| gradient | 38.7 ms | 49.5 ms | 52.3 ms | 106.8 ms | 251.3 ms | 459.7 ms |
| inplace | 165.1 ms | 201.1 ms | 198.0 ms | 263.6 ms | 913.3 ms | 1.788 s |
| pare | 2.62 ms | 1.59 ms | 1.69 ms | 2.77 ms | 7.93 ms | 16.69 ms |

(N=2/4/8 are the layout-B rows from the bake-off, same problem and same build,
which is why those rungs cost no extra node-hours.)

**Parallel efficiency against the 1-node baseline is catastrophic**, because the
times do not merely fail to fall — they rise:

| | N=2 | N=4 | N=8 | N=16 | N=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| build_graph speedup | 0.76× | 0.44× | 0.48× | 0.18× | 0.09× |
| efficiency | 38 % | 11 % | 6 % | 1.1 % | 0.29 % |

The single best case anywhere in the table is `pare` at N=2 (1.65× on 2 nodes);
everything else is at or below 1× from the first rung.

**The ladder stops at 32 nodes deliberately.** A 64-node rung was queued and then
cancelled: six points cannot say anything five points have not already said, and
by the time it would have run, `env.sh` had begun pinning the BLAS thread pools
(see the provenance note above) — so it would have been one point from a different
environment rather than the top of this ladder, and an anomalous result could not
have been attributed. N=64 comes free with the re-baseline described in §5.

### The time is linear in R

Divide `build_graph` by the rank count: **25.4, 16.8, 14.6, 6.7, 8.7, 8.7 ms per
unit R**. From N=8 onward that is flat, i.e. `T ∝ R` on a problem whose total work
is constant. A cost that grows linearly with the number of ranks while the work is
fixed is a serial term, and `HybridComm` has exactly one of the right shape.

---

## 2. Weak scaling — problem grows with the nodes

`--obs-terms` scaled proportionally (200 000 per node), everything else fixed.
Ideal weak scaling is a **flat row**.

| Operation | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| terms | 813 913 | 1 627 766 | 3 254 800 | 6 511 662 | 13 019 552 | 26 030 763 |
| build_graph | 197.6 ms | 423.8 ms | 717.0 ms | 1.360 s | 1.969 s | 5.129 s |
| energy | 21.5 ms | 22.5 ms | 38.8 ms | 58.2 ms | 93.8 ms | 236.5 ms |
| gradient | 29.3 ms | 55.7 ms | 58.6 ms | 137.0 ms | 518.9 ms | 3.179 s |
| inplace | 167.5 ms | 286.0 ms | 281.8 ms | 520.9 ms | 1.785 s | 3.275 s |
| pare | 3.36 ms | 4.37 ms | 5.78 ms | 5.29 ms | 29.69 ms | 12.60 ms |

The term count doubles exactly with the node count, so the scaling is set up
correctly. Weak efficiency at N=32:

| | build_graph | energy | gradient | inplace |
| --- | ---: | ---: | ---: | ---: |
| efficiency vs N=1 | 3.9 % | 9.1 % | 0.9 % | 5.1 % |

`gradient` is the worst: 108× the time for the same work per rank.

Again the shape is `T ∝ R`. `build_graph` per unit R is **24.7, 26.5, 22.4, 21.3,
15.4, 20.0 ms** — flat across a 32× range in R, on a problem that grew 32×.

---

## 3. The floor, isolated

The cleanest evidence needs no payload at all. Running the multi-node driver on a
**1 520-term** operator — small enough that the real work is nil:

| R | S | world | terms | operator | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 128 | 1 520 | 0.5 MiB | 0.150 |
| 16 | 16 | 256 | 1 520 | 0.7 MiB | 0.224 |
| 32 | 16 | 512 | 1 520 | 1.1 MiB | 0.608 |
| 64 | 16 | 1024 | 1 520 | 1.9 MiB | 0.997 |

and at **fixed R = 64**, sweeping S:

| R | S | world | seconds |
| ---: | ---: | ---: | ---: |
| 64 | 2 | 128 | 0.769 |
| 64 | 4 | 256 | 0.563 |
| 64 | 8 | 512 | 0.674 |
| 64 | 16 | 1024 | 0.997 |

A half-megabyte operator takes **a full second** to propagate on 64 ranks. This is
pure `O(R·S²)` overhead with nothing to hide behind, and it is the single most
sensitive before/after target for the port.

`HybridComm` on `HEAD` runs **three** serial `O(R·S²)` loops on partition 0, not
one: `pack_count_matrix_`, the count sums in `size_staging_impl_`, and the two
offset prefix scans in the same function.

> **Superseded — do not cite the 0.997 s as a live target.** The table above is a
> valid measurement of the code as it stood, and is kept for that reason, but the
> conclusion drawn from it no longer holds and `pack_count_matrix_` no longer
> exists. `size_staging_parallel_` reduced the prefix to `O(R·S)`, and profiling at
> **29M terms** (`monoprop_COMM_PROFILE=1`, 2 nodes) puts the residual serial
> partition-0 work at `table_p0_s / table_par_s = 0.12` in *both* production
> layouts — about **5 ms of a 2.8 s wall**. See `RESULTS-threading-baseline.md`.
>
> Two things make the number above misleading rather than merely old. It was taken
> on a **1 520-term** operator, where the serial term has, by construction, nothing
> to hide behind — that is what made it a clean isolation of the floor, and also
> what makes it useless for sizing the floor's share of a real workload. And the
> dominant cost at production scale turned out to be somewhere this document does
> not look at all: the `MPI_THREAD_SERIALIZED` funnel, where the S−1 non-funnel
> partitions idle for as long as partition 0 spends inside MPI
> (`barrier_peers_s ≈ mpi_s`, the largest single item in the profile).

---

## 4. Memory — a second, independent defect

This one is **not** fixed by `perf/multinode-comm-scaling`, and it is easy to
misattribute after the port lands.

On the weak ladder, **terms per rank is constant** — 101 739 at R=8, 101 682 at
R=256 — yet resident memory per rank climbs steadily:

| R | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiB / rank | 343 | 504 | 685 | 981 | 1711 | 3176 |
| total | 2.7 GiB | 7.9 GiB | 21 GiB | 61 GiB | 214 GiB | 794 GiB |

That is **9.3× the memory per rank for identical per-rank work**, tracking at
roughly **11.4 MiB per unit R** at the top end — which makes the job total
quadratic in R.

Two controls rule out the obvious explanations:

- **Not the MPI transport.** `tools/mpi_memory_probe.py` measures per-rank `VmHWM`
  with and without an all-to-all that forces endpoint setup to every peer. It is
  flat in R (190 → 199 → 199 → 199 MiB at R=1…32) and the connection cost is a
  constant ~10 MiB. Forcing UCX's dynamically-connected transport
  (`UCX_TLS=dc_x,sm,self`) saves a flat ~45 MiB per rank and **does not change the
  slope**, which is what proves it is not per-peer endpoint state.
- **Not fixed per-rank bookkeeping.** On a near-empty problem, per-rank RSS is flat
  in both R (265, 267, 275, 213 MiB at R=8…64) and S (201→213 MiB at S=2…16). The
  baseline does not scale; only loaded structures do.

So the growth is **traffic-coupled per-(partition, destination) storage**, which at
`S` local partitions × `R·S` destinations is another `O(R·S²)` structure —
this time in memory rather than time. At R=256, S=16 that is 65 536 destination
buffers per rank, and the measured ~11.4 MiB per unit R works out to roughly 45 KB
per buffer, far more than the ~1.5 terms each actually carries.

**Consequence for the beyond-one-node goal: past some rank count, adding nodes
reduces the largest problem the job can hold.** Spreading wider costs more memory
than it adds. This wants fixing before any run that is trying to exceed 242 GB.

`monoprop_multinode.py` now reports `memory_bytes` (operator only, allreduced),
`rss_bytes`, and `overhead_per_rank` side by side, which is what separates the two.

---

## 5. What follows

1. **Land `perf/multinode-comm-scaling`.** Not optional: it is the difference
   between a machine that scales and one that does not. §3 is a ready-made
   before/after that needs 8 nodes and about a minute.
2. **Then re-run §1 and §2 as a matched pair** — once on the pre-port build and
   once on the post-port build, both under the current (BLAS-pinned) environment,
   and take the ladder to N=64 this time. They are scripted, and the whole pair
   costs well under 1 % of the allocation. Re-running the pre-port half is not
   waste: it is what makes the comparison a controlled one rather than a
   cross-environment guess.
3. **Treat the memory growth in §4 as a separate defect** with its own fix. The
   branch's new arrays are all `O(R·S)`, so it will not address this, and a
   post-port memory regression should not be blamed on the port.
4. **Do not size production runs off these numbers.** They characterise a known
   bottleneck; the useful capacity numbers come after (1).

Reproduce: `hpc/deucalion/sbatch/strong-scaling.sh`, `weak-scaling.sh`,
`memory-scaling.sh`, `memory-probe.sh` (each `sbatch -N<n>`).
Cost of everything above: **798 of 384 000 core-hours (0.21 %).**
