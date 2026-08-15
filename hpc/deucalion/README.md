# Deucalion — cluster reference

Operations notes for the Portuguese EuroHPC system, written for the x86 partition.

Sources: the official user guide (<https://docs.deucalion.acnca.pt/>, mirrored at
<https://docs.macc.fccn.pt/>) **plus** live probing of the system on 2026-08-11.
Where the two disagree the measured value is used and the discrepancy is called
out — see [Where the published docs are wrong](#where-the-published-docs-are-wrong),
which is the most useful section here.

This is an operations runbook, not part of the published documentation site.

Companion files:

| File | What it is for |
| --- | --- |
| [§10 Running monoprop here](#10-running-monoprop-here) | Building and launching monoprop: toolchain choice, rank/partition layout, the A/B procedure |
| [RESULTS-layout-bakeoff.md](RESULTS-layout-bakeoff.md) | Which `R × S` layout to use, measured. Short answer: 8 ranks/node × 16 partitions |
| [RESULTS-scaling.md](RESULTS-scaling.md) | Strong and weak scaling, and the defect that currently stops monoprop scaling across nodes. **Read before planning any multi-node run** |
| [RESULTS-threading-baseline.md](RESULTS-threading-baseline.md) | The placement/barrier fixes, measured before and after, at 0.8M and 29M terms |
| [RESULTS-ab-100m.md](RESULTS-ab-100m.md) | Branch vs `main` at **100M terms**, time and memory, 1 and 2 nodes. **The most current numbers here** — and the size at which the placement fix stops paying for itself |
| `env.sh` | Module loads and exports; sourced by every build and job script. A **runtime** requirement, not just build-time |
| `sbatch/` | `build-worktree.sh` (build one checkout), `mpi-tests-worktree.sh` (correctness), `ab-100m.sh` (the time+memory A/B) |
| `tools/` | `mpi_sanity.py` (smoke-test a fresh venv), `ab_summary.py` (render and gate an A/B run) |

---

## 1. Hardware

| Partition family | Nodes | CPU | Memory | Network |
| --- | --- | --- | --- | --- |
| `arm` | 1632 | Fujitsu A64FX 48-core @ 2.0 GHz | 32 GB HBM2 | ConnectX-6 100 Gb/s |
| `x86` | 500 | 2× AMD EPYC 7742 64-core @ 2.25 GHz | 256 GB DDR4 | ConnectX-6 100 Gb/s |
| `a100-40` | 17 | 2× EPYC 7742 | 512 GB + 4× A100 40 GB | 2× ConnectX-6 200 Gb/s |
| `a100-80` | 16 | 2× EPYC 7742 | 512 GB + 4× A100 80 GB | 2× ConnectX-6 200 Gb/s |

### The x86 compute node, measured

```
CPU(s):              128        Thread(s) per core:  1      <-- SMT is OFF
Core(s) per socket:  64         Socket(s):           2
NUMA node(s):        8                                      <-- NPS4
NUMA node0 CPU(s):   0-15       NUMA node4 CPU(s):   64-79
NUMA node1 CPU(s):   16-31      NUMA node5 CPU(s):   80-95
NUMA node2 CPU(s):   32-47      NUMA node6 CPU(s):   96-111
NUMA node3 CPU(s):   48-63      NUMA node7 CPU(s):   112-127
Mem:                 251 GB total, 240 GB available
```

**8 NUMA domains of 16 cores each.** This is the single most important number for
placing MPI ranks: 8 ranks/node × 16 threads is the NUMA-aligned layout, 2×64 is
socket-aligned, 1×128 puts the whole node on one shared-memory group.

The **login node is not the same shape**: it is the same Zen 2 silicon (so
`-march=native` → `znver2` is correct on both) but runs **SMT on** and **NPS1**, so
it reports 256 threads across 2 NUMA nodes. Never derive core counts or affinity
from the login node.

Interconnect on x86 is a **non-blocking** fat-tree (27 QM8790 switches, 17 leaf ×
20 down/20 up, 10 spine). ARM is 1:6 blocking. Verified on the login node:
ConnectX-6 (`MT4123`), port `Active`, rate `100`, `active_mtu 4096`, plus six
ConnectX-4 Lx Ethernet NICs bonded for management/storage.

---

## 2. Partitions

| Partition | Arch | Max nodes | Max walltime | Default walltime |
| --- | --- | --- | --- | --- |
| `dev-arm` | aarch64 | 2 | 4 h | 2 h |
| `normal-arm` | aarch64 | 128 | 2 d | 2 h |
| `large-arm` | aarch64 | 512 | 3 d | 2 h |
| `dev-x86` | x86_64 | 2 | 4 h | 2 h |
| `normal-x86` | x86_64 | 64 | 2 d | 2 h |
| `large-x86` | x86_64 | 128 | 3 d | 2 h |
| `dev-a100-40` / `dev-a100-80` | x86_64 | 1 | 4 h | 2 h |
| `normal-a100-40` / `normal-a100-80` | x86_64 | 4 | 2 d | 2 h |
| `ooda` | aarch64 | 1 | 8 h | 1 h |

- **All CPU partitions are `OverSubscribe=EXCLUSIVE`.** You always get whole nodes
  and are billed for all 128 cores (x86) or 48 (ARM) whether you use them or not.
  Never request partial nodes on a CPU partition.
- **The default walltime is 2 h and `OverTimeLimit=0`** — there is no grace period.
  A job one second over its limit is killed (`KillWait=60`).
- `AvailableFeatures` is `(null)` on every node, so `--constraint` does nothing.
  Architecture is selected purely by partition.
- The 40 GB vs 80 GB A100 distinction exists **only in the partition name** — GRES
  reads `gpu:a100:4` on both.

### Hidden partitions

`sinfo -a` also shows `all-arm`, `all-x86` and `all-a100`: the full machine at
**6 d 6 h** walltime, gated on unix groups `allarm` / `allx86` / `alla100`.
`scontrol show partition all-x86` returns "not found" unless you are in the group.
Worth a support request if you need more than 3 days or more than 128 x86 nodes.

---

## 3. Accounts and billing

Three architecture-suffixed accounts. **The suffix must match the partition** or
the job is rejected at submit (`AccountingStorageEnforce=associations,limits,safe`,
`EnforcePartLimits=ALL`).

| `#SBATCH -A` | Unlocks |
| --- | --- |
| `<project>a` | `dev-arm`, `normal-arm`, `large-arm`, `ooda` |
| `<project>x` | `dev-x86`, `normal-x86`, `large-x86` |
| `<project>g` | the `a100` partitions |

> **The default account is the ARM one.** Always pass `-A` explicitly, or an x86
> job silently fails to submit.

`billing` prints usage and limits per account. The number is **core-hours**, so
divide to get node-hours: x86 by 128, ARM by 48. GPU accounts are quoted in
GPU-hours (divide by 4 for full-node hours).

There is exactly one QOS, `normal`, with no limits and priority 0. Don't bother
with `--qos`.

### Scheduling — the part that actually matters

```
PriorityType            = priority/multifactor
PriorityWeightAge       = 0    PriorityWeightFairShare = 0
PriorityWeightAssoc     = 0    PriorityWeightJobSize   = 0
PriorityWeightPartition = 0    PriorityWeightQOS       = 0
SchedulerType           = sched/backfill
SchedulerParameters     = ...bf_interval=60,bf_window=1440,bf_max_job_start=200...
```

**Every priority weight is zero.** Scheduling is therefore FIFO by job ID plus very
aggressive backfill. Consequences:

- **A tight, accurate `--time` is the single biggest lever on queue wait.** A job
  that fits a backfill hole starts immediately regardless of queue position.
- Fairshare is irrelevant; heavy use now carries no penalty later.
- Age confers no bonus, so resubmitting costs nothing.
- Prefer several short jobs to one long one.

A busy-looking queue is usually not competition. On 2026-08-11, 430 of 492
`normal-x86` nodes were allocated with ~400 jobs pending, yet **exactly one**
pending job cluster-wide was blocked by `Resources`; the rest were blocked by
other projects' exhausted budgets (`AssocGrpCPUMinutesLimit`) or their own
association caps. Check with:

```bash
squeue -t PENDING -h -o "%P|%r" | sort | uniq -c | sort -rn
```

Other caps: `MaxArraySize=10000`, `MaxJobCount=100000`, `MaxTasksPerNode=512`,
`MaxStepCount=40000`.

> ARM only: 633 nodes sit in `idle~` (powered down), with `SuspendTime=3600` and
> `ResumeTimeout=600`. A large ARM allocation can take up to 10 minutes to boot
> after being scheduled. That is not a hang.

---

## 4. Filesystems

| Path | Type | Quota | Notes |
| --- | --- | --- | --- |
| `/home/<user>` | NFSv4 | **25 GB and 25,000 inodes** | `quotahome` |
| `/projects/<PROJECT>` | Lustre (`deucfs01`, 9.8 PB) | per-project, e.g. 1.46 TB, no inode limit | `quotaprojects` |
| `/dev/shm` | tmpfs | 126 GB on x86 compute | RAM |
| `/` on compute | ext4, 220 GB | ~204 GB free | see caveat below |

**The inode limit on `$HOME` is the binding constraint, not the byte quota.** A uv
venv plus a managed CPython plus a scikit-build tree runs to tens of thousands of
files. Put source, environments, caches and output on `/projects`. The published
guide says the same.

There is **no `/scratch` and no `/work`**. `/projects` is the only parallel
filesystem available to you.

### Lustre striping

The project directory default is `stripe_count=1, stripe_size=1 MiB` — a single
OST. Widen it *only* for directories that will hold multi-GB files:

```bash
lfs setstripe -c 16 -S 4M /projects/<PROJECT>/runs/large
```

Do **not** stripe wide by default. Wide striping many small files (JSONL result
records, per-rank logs) costs more than it gains. `deucfs01` has 16 MDTs, so
`lfs mkdir -i <n>` can spread directories across metadata targets if you generate
very many files.

---

## 5. Modules

Lmod 8.5.22, OpenHPC base with an EasyBuild overlay at `/eb/$(uname -i)/modules/all`.

### Two traps that will cost you an hour each

1. **Never `module purge`.** `MODULEPATH` is assembled by
   `/etc/profile.d/zz-hpcnow-arch.sh`. `purge` discards it and every subsequent
   load fails with "The following module(s) are unknown". There is no in-place
   repair — start a new shell.
2. **Never re-source `/etc/profile.d/lmod.sh` or `zz-hpcnow-arch.sh` by hand.** The
   login profile already ran them. Re-running resets Lmod's state, after which
   `module load` silently succeeds while loading nothing.

Corollary for batch scripts: the shebang must be **`#!/bin/bash -l`**. Without
`-l` the `module` function is never defined and every `module load` is a no-op —
your job then runs against system GCC 8.5 and fails in a confusing way.

### x86/ARM split

`/eb/x86_64` → `.../rocky/8.5/amd/zen2` is mounted on the login nodes.
`/eb/aarch64` **does not exist there** — it appears only on `cna*` nodes, where
`/share/env/module_select.sh` rewrites `MODULEPATH`. You cannot inspect or build
against the ARM stack from an x86 login node. ARM login nodes exist at
`ln0{1..4}-arm`; the Fujitsu cross-compilers (`fccpx`, `mpiFCCpx`, …) from
`FJSVstclanga` do run on x86 login nodes.

### Available on x86 (abbreviated)

- **GCC** 8.3.0 → **15.2.0**; **LLVM/Clang** → 21.1.8; Intel oneAPI → 2025.2.0; NVHPC → 25.9
- **OpenMPI** 3.1.4 → **5.0.10**; Intel MPI → 2021.16.1; MVAPICH2 2.3.7; HPC-X ×3
- **Toolchains**: `foss`/`gompi` 2019b → **2026.1**, `intel`/`iimpi`, `lompi` (LLVM+OpenMPI)
- **CMake** → 4.2.1; **Ninja** → 1.13.2; **Python** → 3.14.2; **uv** 0.7.13
- **Boost** → **1.88.0** (GCC 14.2, 14.3, Intel, LLVM — **none for GCC 15.2**)
- UCX → 1.20.0, libfabric → 2.5.0, PMIx → 6.1.0, hwloc → 2.13.0
- OpenBLAS → 0.3.32, FlexiBLAS → 3.5.0, AOCL-BLAS 5.0–5.2, imkl → 2025.2.0, FFTW 3.3.10
- `ccache` is **not** packaged.

---

## 6. MPI

`MpiDefault=pmix`, and `srun --mpi=list` offers only `none`, `cray_shasta`, `pmi2`,
`pmix` (`pmix_v4`). So:

```bash
srun --mpi=pmix ...            # OpenMPI
srun --mpi=pmi2 ...            # Intel MPI — its module sets SLURM_MPI_TYPE=pmi2
```

Match the PMIx generations. OpenMPI 5.0.8 (`foss/2025b`) carries PMIx 5.0.8, which
pairs cleanly with this Slurm's `pmix_v4`. OpenMPI 5.0.10 (`foss/2026.1`) pulls
PMIx 6.1.0, an untested pairing here. If PMIx misbehaves, `mpirun` inside an
allocation works — PRRTE bypasses Slurm's PMIx server.

Two site build choices worth knowing:

- **Every OpenMPI module force-sets `OMPI_MCA_io=^ompio`**, pushing MPI-IO onto
  ROMIO instead of OMPIO — a deliberate Lustre choice. Tune with `romio_cb_*`
  hints against your stripe count, not OMPIO ones.
- The EasyBuild OpenMPI is configured **`--without-xpmem`**, so there is no
  single-copy intra-node shared-memory transport; large intra-node messages go via
  CMA (`btl/sm`) or `uct`. On 128-core nodes this is worth measuring. `hcoll` and
  SHARP in-network collectives are **not** in the EasyBuild OpenMPI — they are only
  in the HPC-X modules.

Available components: `pml: cm, ob1, ucx, v` · `btl: self, ofi, sm, tcp, uct` ·
`mtl: ofi` · `coll: ..., han, tuned, ucc` · `io: ompio, romio341`.

---

## 7. Where the published docs are wrong

The guide is materially out of date on the x86 software stack. Everything below
was checked directly.

| The guide says | Actually |
| --- | --- |
| x86 compilers are "GCC 12.3.0, Intel oneAPI 2023.1.0"; `ml GCCcore/11.3.0` | GCC up to **15.2.0**, LLVM up to 21.1.8, Intel up to 2025.2.0 |
| MPI examples use `OpenMPI/4.1.5-GCC-12.3.0` | OpenMPI up to **5.0.10**; `foss` toolchains up to **2026.1** |
| Singularity "3.11.3" (Compilation page) / "4.3.2" (overview) | Both stale relative to each other; check `singularity --version` on the node |
| Slurm 23.11.4 | 23.11.**8** |
| Home quota "25 GB and 20000 files" | 25 GB and **25,000** inodes (soft 20 GB / 20,000) |
| Jobs "must run in `/projects/$project` only" | Enforced by convention and quota, not by policy |

This matters concretely: **monoprop requires GCC ≥ 14**, so on the documented
stack it would not compile at all. It builds fine on `foss/2025b`.

Two more measured corrections to widely-repeated assumptions:

- **Compute nodes have full outbound internet.** From `dev-x86`, GitHub, PyPI and
  the internal `sn02` mirror all return 200. Build-time dependency fetching
  (`uv sync`, CPM/FetchContent git clones) works directly on a compute node; no
  login-node cache warming is needed.
- **Compute nodes do have local disk**, despite Slurm advertising `TmpDisk=0`:
  `/` is a 220 GB ext4 with ~204 GB free, and `/dev/shm` is 126 GB. Slurm's
  `TmpDisk=0` is a configuration value, not a measurement. (Confirm per-job
  writability and cleanup policy before relying on it.)
- **The site pip proxy is unreliable on cache misses for source distributions.**
  `/etc/profile.d/proxpi.sh` points `PIP_INDEX_URL` at a proxpi cache on
  `sn02:9191`. Fetching an *uncached* sdist (`mpi4py-4.1.2.tar.gz`) from a compute
  node failed three retries and died with `Invalid gzip header`, while the same
  URL served a byte-identical valid archive on every attempt from a login node
  once warm. Wheels were unaffected. Since compute nodes reach `pypi.org`
  directly, the robust configuration is to leave `uv` on its default index rather
  than mirroring `PIP_INDEX_URL` into `UV_DEFAULT_INDEX`. Warming the proxy from a
  login node first also works.

---

## 8. Site tools

| Command | What it does |
| --- | --- |
| `billing` | core-hours used vs limit, per account (`billing -a <acct>` for per-user) |
| `quotahome` | `$HOME` bytes and inodes |
| `quotaprojects` | per-project Lustre usage |
| `cdp` | jump to a project directory |
| `module load chat` | the site documentation chatbot |

Containers: `singularity` and `enroot` in `/usr/bin`; no docker/podman/apptainer.
Prebuilt images under `/share/apps-{x86,arm}/containers/`. Singularity images must
be **built on the matching architecture**, and `--fakeroot` builds will not work
inside `/projects` — build them in `$HOME` and move them.

---

## 9. Checklist for a new job script

```bash
#!/bin/bash -l                      # -l or `module` is a silent no-op
#SBATCH -A <project>x               # explicit: the default account is ARM
#SBATCH -p normal-x86
#SBATCH -N 8
#SBATCH --ntasks-per-node=8         # 8 NUMA domains
#SBATCH --cpus-per-task=16          # 16 cores each
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 1:30:00                  # tight: backfill rewards accuracy
#SBATCH -o %x-%j.out -e %x-%j.err

source hpc/deucalion/env.sh
srun --mpi=pmix --cpu-bind=cores --distribution=block:block ./your_binary
```

- [ ] `#!/bin/bash -l`
- [ ] `-A` matches the partition's architecture suffix
- [ ] `--exclusive --mem=0` (CPU partitions are exclusive anyway; `--mem=0` avoids a default cap)
- [ ] `--time` is tight and honest
- [ ] output goes to `/projects`, not `$HOME`
- [ ] no `module purge` anywhere

---

## 10. Running monoprop here

### Build

Both A/B arms are worktrees under `$PROJ`, built the same way. Keep them on the same
filesystem: a venv in `$HOME` differs from one on Lustre in shared-library load path, and
`$HOME` is inode-capped at 25k, which a second build tree can exhaust.

```bash
git worktree add --detach "$PROJ/src/mp-main" origin/main   # baseline arm
git worktree add --detach "$PROJ/src/mp-port" HEAD          # branch arm

for wt in mp-main mp-port; do
    cp -r hpc "$PROJ/src/$wt/hpc"                           # hpc/ is gitignored, so a
    (cd "$PROJ/src/$wt" \                                   # worktree has none of it
        && sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" \
                  hpc/deucalion/sbatch/build-worktree.sh)
done
```

`build-worktree.sh` exports `MONOPROP_SRC="$PWD"` and runs `build.sh`, so each checkout
builds against its own sources into its own `.venv`. That one line is the whole mechanism
that lets two builds coexist. Defaults: MPI on, `monoprop_MAX_NUM_MODES=1024`, `-j32`
(`MONOPROP_MAX_NUM_MODES` / `BUILD_JOBS` override) — and `MAX_NUM_MODES` must match across
arms or `ab_summary.py` refuses the run.

**Submit from inside the worktree, not with `--chdir` alone.** The script starts with
`cd "${SLURM_SUBMIT_DIR:-$PWD}"`, and Slurm sets `SLURM_SUBMIT_DIR` to wherever `sbatch`
ran, which overrides `--chdir`. Submitting from the primary checkout therefore rebuilds the
primary checkout twice and leaves the worktree unbuilt — with a job log that names the right
tree in its `=== src ===` line only if you also passed `--chdir`, so it looks correct.

Why `foss/2025b` and not the newest stack:

| Constraint | Consequence |
| --- | --- |
| `GNU.CXX.cmake` raises `FATAL_ERROR` below **GCC 14** | GCC 12.3.0, the version the site docs advertise for x86, cannot build this project at all |
| `find_package(Boost 1.85 CONFIG REQUIRED)` | needs a Boost module — and **there is none built against GCC 15.2.0**. `foss/2026.1` would force a source build via `tools/install-deps.sh` |
| Slurm offers only `pmix_v4` | OpenMPI 5.0.8 (PMIx 5.0.8) pairs cleanly; `foss/2026.1`'s OpenMPI 5.0.10 pulls PMIx 6.1.0, untested here |

Build knobs that matter:

- **`monoprop_ENABLE_MPI=ON`.** Off by default everywhere, including the PyPI wheels.
  Without it the communicator argument is ignored and *every rank holds the full operator*.
- **`--reinstall-package monoprop --no-cache` is mandatory.** `[tool.uv] cache-keys` does
  not include `SKBUILD_*` or config-settings, so uv otherwise hands back a cached
  MPI-**off** build and everything downstream silently runs single-rank.
- **`build.tool-args=-j32`** overrides `pyproject.toml`'s `["-j2"]`, which also beats
  `CMAKE_BUILD_PARALLEL_LEVEL` because it is an explicit ninja argument.
- **Leave `monoprop_WIDE_TERM_INDEX` off.** It only matters above 2³² terms in one
  partition, an operator needing ~430 GB. Memory binds first.

Verify: `"$VENV/bin/python" -c "import monoprop as m; print(m.has_mpi, m.MAX_NUM_MODES)"`.

### Rank and partition layout

monoprop composes two axes into one flat world of `R × S`: `R` MPI ranks, `S` in-process
pinned partitions per rank. An x86 node is 8 NUMA domains × 16 cores:

| Layout | `--ntasks-per-node` | `--cpus-per-task` | `monoprop_NUM_THREADS` | Shape |
| --- | --- | --- | --- | --- |
| A | 1 | 128 | 128 | whole node in one shared-memory group |
| B | 8 | 16 | 16 | one rank per NUMA domain — **the default choice** |
| C | 2 | 64 | 64 | one rank per socket |

Three hard rules:

1. **A multi-rank run defaults to `S = 1`.** `resolve_partition_count_` uses
   `ranks == 1 ? cores : 1` when `monoprop_NUM_THREADS` is unset, so an MPI job that
   forgets it leaves 15/16 of every node idle. **Always set it explicitly.**
2. **Every rank must resolve the same `S`,** or the job deadlocks. The code aborts on
   mismatch. Do not let `monoprop_NUM_THREADS` vary between ranks.
3. **If `S` exceeds the cores the rank was granted, pinning turns off entirely** — and
   with it the barrier's locality domains. Under `--cpu-bind=cores` the grant is the rank's
   own mask, so the test is `S > cpus-per-task`, not `ranks_per_node × S > 128`. That
   distinction was a real bug: a rank divided its already-confined mask a second time, so
   layouts B and C ran unpinned with `barrier_groups=0` — 437 vs 15.5 µs/sync at layout B.
   Fixed on `perf/multinode-comm-scaling` and **not on `main`**, which is why an A/B across
   that boundary must probe placement rather than assume it.

Runtime environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `monoprop_NUM_THREADS` | one partition per physical core | caps partition count |
| `monoprop_PARTITIONS` | `auto` | `auto` \| integer `N` \| `off` (one partition, whole operator) |
| `monoprop_PARTITION_PINNING` | `on` | `0`/`false`/`no` disables per-core pinning |
| `monoprop_COMM_PROFILE` | off | per-partition collective profile — **branch-only**, absent from `main` |
| `monoprop_SPIN_BUDGET_US` | `30` | on-core barrier spin before yielding — **branch-only** |
| `monoprop_BARRIER_GROUPING` | `on` | `0` forces the flat barrier with pinning left on — **branch-only** |

### Launching

```bash
srun --mpi=pmix --cpu-bind=cores --distribution=block:block \
     "$VENV/bin/python" your_script.py
```

- **`--mpi=pmix`**, not the default. Intel MPI would need `--mpi=pmi2`.
- **Launch `$VENV/bin/python` directly, not `uv run`** — `uv run` re-resolves the
  environment in every rank, which is a Lustre metadata storm at scale.
- **`just bench-mpi` does not work here.** It hardcodes `mpiexec --map-by slot:PE=N`. Run
  the same pytest command under `srun`; `monoprop_BENCH_LABEL` / `monoprop_BENCH_RESULTS`
  is the entire contract `benches/` needs.
- **`comm=None` silently means `MPI_COMM_WORLD`**, not serial, in an MPI build. Pass
  `MPI.COMM_SELF` for independent per-rank replicas.
- **`-s` is load-bearing under pytest.** Capture is fd-level, and the `COMMPROF` line is
  written straight to fd 2 from a transport destructor, so without it a live instrument
  looks like one that never fired.

### The 100M-term A/B

```bash
sbatch -N1 -A "$MONOPROP_SLURM_ACCOUNT" -t 3:30:00 --chdir="$PWD" hpc/deucalion/sbatch/ab-100m.sh
sbatch -N2 -A "$MONOPROP_SLURM_ACCOUNT" -t 3:00:00 --chdir="$PWD" hpc/deucalion/sbatch/ab-100m.sh
```

Runs both builds in **one** allocation, order flipped every `(rep, cell)`, and reports
time and memory per operation for `build_graph`, `propagate`, `energy`, `gradient`.
Knobs: `MAIN_VENV` (`$PROJ/src/mp-main/.venv`), `PORT_VENV` (`$PROJ/src/mp-port/.venv`),
`REPS` (4), `RESULTS_TAG`, `OBS_TERMS` (24000000), `NUM_MODES` (250),
`RANKS_PER_NODE` (8), `PARTITIONS` (16). `benches/` comes from `$PWD` for both arms, so
only the compiled extension differs.

Order of operations, each step cheap enough to catch the previous one's mistakes:

1. `mpi-tests-worktree.sh`, submitted **from inside** each worktree (it too honours
   `SLURM_SUBMIT_DIR` over `--chdir`). Gate on `-L serial` plus the Python suite;
   `ctest -L mpi` is known to fail at 2 ranks on `main` and is not a signal.
2. `OBS_TERMS=200000 REPS=1 sbatch -N1 -p dev-x86 ...` — proves the labels, both cells and
   every provenance gate in minutes.
3. A second rung at `OBS_TERMS=6000000`, also on `dev-x86`. **Two rungs, not one full-size
   pilot.** Memory here is affine in propagated terms, so two cheap points give both the
   intercept and the slope, and predict full size better than one blind 120× extrapolation
   — for a fraction of the node-hours. Measured, layout B, per *node* at 8 ranks/node:

   | cell | fixed | slope | at 99.4M terms |
   | --- | --- | --- | --- |
   | `fresh` (`build_graph`, `propagate`) | 1.8 GiB | 1234 B/term | **116 GiB** |
   | `graph` (`energy`, `gradient`) | 1.8 GiB | 483 B/term | **47 GiB** |

   Against 233 GiB of usable node memory, so full size fits at 8 ranks/node with room. The
   `fresh` figure is the binding one: it is a whole-test high-water that includes the
   `make_random_problem` transient, which is replicated on every rank.
4. The real runs, N=1 before N=2 — N=1 is the memory-tightest.

**`--num-modes` must be 250, not 256.** A `MAX_NUM_MODES=250` build rejects 256 outright,
and 250 is accepted by both a 250- and a 1024-cap build.

**Sizing `--time`.** Setup is replicated on every rank, so wall time is dominated by
`make_random_problem` and barely falls with node count. Measured ~53 s per invocation at
6M obs-terms; full size is 16 invocations (2 cells × 2 sides × 4 reps) at roughly 4× that,
so **`-t 2:00:00` is right for both N=1 and N=2** — not the 3:30:00 a linear guess suggests.
Tight `--time` is the single biggest lever on backfill queue wait here. A timeout is
recoverable: per-label JSON is written as each cell finishes, so
`ab_summary.py <results-dir>` can be re-run by hand over a partial directory.

`ab_summary.py` **exits non-zero and prints a `## REFUSED` section** rather than letting a
void run read as a null result. It refuses when the arms propagated different term counts,
when neither arm placed a thread (everything ran unpinned), when both did (the baseline
cannot place at layout B, so a venv is mislabelled — override with `--allow-both-placed`),
when `MAX_NUM_MODES`/`MALLOC_ARENA_MAX`/thread counts differ across cells, or when
`--bench-rounds` was not 1. A non-zero exit means the tables are a diagnostic, not a result.
