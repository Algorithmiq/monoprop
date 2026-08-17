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
| [RESULTS-invidx-arena-segments.md](RESULTS-invidx-arena-segments.md) | The per-commit **memory** attribution and the arena-segment fix. Read it for the method as much as the numbers: it is where "state your `P`" was learned |
| `env.sh` | Module loads and exports; sourced by every build and job script. A **runtime** requirement, not just build-time |
| `sbatch/` | `build-worktree.sh` (build one checkout), `mpi-tests-worktree.sh` (MPI layouts + Python suite), `ctest-worktree.sh` (gate any checkout in a standalone build tree), `ctest-repeat.sh` (is that failure a rate?), `ab-100m.sh` (the time+memory A/B), `membisect.sh` (per-commit memory + paired time) |
| `tools/` | `mpi_sanity.py` (smoke-test a fresh venv), `ab_summary.py` (render and gate an A/B run), `terms_calib.py` (how many terms a config actually reaches), `prof_run.py` (the one-model driver), `membisect_summary.py`, `layerprof_summary.py` |
| `cells/` | Cell definitions for `membisect.sh`: model, expected term count, and the overrides that reach it |

**Sections 11 and 12 are the ones that will save you a day.** They are the traps found in campaign
use, after the probing that produced sections 1–10 — every item cost a job round-trip or a voided
measurement, and most of them fail by producing a *plausible wrong number* rather than an error.

---

## 1. Hardware

| Partition family | Nodes | CPU | Memory | Network |
| --- | --- | --- | --- | --- |
| `arm` | 1632 | Fujitsu A64FX 48-core @ 2.0 GHz | 32 GB HBM2 | ConnectX-6 100 Gb/s |
| `x86` | 500 | 2× AMD EPYC 7742 64-core @ 2.25 GHz | 251 GB (234 GiB) DDR4 | ConnectX-6 100 Gb/s |
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

**Quote memory in one unit.** 251 GB total = **234 GiB**, of which ~240 GB = **223 GiB** is available
to jobs; the vendor's "256 GB" is the nameplate. Campaign sizing in the RESULTS files uses GiB, so
convert before comparing a `1234 B/term` slope against a node budget — the same number in GB and GiB
differs by 7%, which is wider than several effects measured here.

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
srun --mpi=pmix --cpu-bind=none --distribution=block:block ./your_binary
```

- [ ] `--cpu-bind=none` (see §10 rule 3 — `=cores` silently unpins every rank)
- [ ] `#!/bin/bash -l`
- [ ] `-A` matches the partition's architecture suffix
- [ ] `--exclusive --mem=0` (CPU partitions are exclusive anyway; `--mem=0` avoids a default cap)
- [ ] `--time` is tight and honest
- [ ] output goes to `/projects`, not `$HOME`
- [ ] no `module purge` anywhere
- [ ] nothing the job reads lives in `/tmp` — see below
- [ ] in-job analysis uses a venv python, never `python3`

### Six ways a job script here fails *quietly*

Each of these produces a plausible wrong result rather than an error, which is what makes them
expensive. They are ordered by how long each one cost.

1. **`/tmp` is node-local, and `sbatch` stages only the batch script — not what it references.** A job
   whose driver lives in `/tmp` starts perfectly and then fails on every `srun`, in about 14 s each,
   with `can't open file`. A whole 4-cell job died this way. Copy drivers to `$PROJ/runs/<label>/` and
   reference them there. Same for anything else a job resolves by path: `nanobind_DIR` under `/tmp`
   vanishes when a session moves login nodes. And a compute node's `/tmp` is not readable from the
   login node afterwards, so output there is simply lost.
2. **`/usr/bin/python3` on the compute nodes is 3.6.8.** No `statistics.fmean`, no `dict |` merge, and
   `from __future__ import annotations` will not save a script that uses newer syntax. An in-job
   heredoc using them raises to **stderr**, so a script redirecting only stdout prints its table
   header and then nothing — which reads exactly like "no data matched". Use `$TREE/.venv/bin/python`.
3. **`nproc` honours `OMP_NUM_THREADS`**, which `env.sh` pins to 1 for the BLAS pools — so every job
   logs `cores: 1` while holding a full 128-CPU allocation, and any `-j$(nproc)` builds serially. Use
   `getconf _NPROCESSORS_ONLN` or Slurm's `AllocCPUS`. C++ `hardware_concurrency()` is unaffected.
4. **`squeue` / `sacct` can fail to fork under login-node pressure and return *empty* output**, which
   naively reads as "the job finished". Require a terminal state string from
   `sacct -j <id> --format=State -X -n` (`COMPLETED`/`FAILED`/`CANCELLED`/`TIMEOUT`/`NODE_FAIL`/`OUT_OF_MEMORY`)
   and treat empty as *retry*. This has corrupted a run's bookkeeping already.
5. **`monoprop_BENCH_RESULTS` must exist** or the bench raises `FileNotFoundError` *after* the
   measurement completes — a good run that reports as a failure. Always `mkdir -p` it first.
6. **`uv sync --all-extras` prunes pytest out of the venv.** pytest is in a dependency *group*, and
   `--all-extras` covers extras, so uv prunes to the declared set and the next `python -m pytest`
   fails with `No module named pytest`, looking like a venv that was never created. Always add
   `--group test` (or `--all-groups`); recovery costs a full `--no-cache` rebuild.

### `$HOME` and the login node

- **At the inode quota, every write fails with `EDQUOT`** — including an editor's temp file, which
  surfaces as a confusing tool error rather than "disk full". Recover by deleting regenerable
  bytecode (`find . -xdev -name '*.pyc' -delete`), never someone's hook cache. The durable fix is to
  **relocate** caches rather than clear them: `PREK_HOME=$PROJ/caches/prek` is set in `~/.bashrc`, and
  `.vscode-server` (~15k files, a full copy per client version) is the biggest consumer — check which
  one is *running* before deleting any.
- **The login node caps threads at 150** via the cgroup-v1 `pids` controller
  (`/sys/fs/cgroup/pids/user.slice/user-<uid>.slice/pids.max`), while `ulimit -u` reports 2,061,974
  and hides it. monoprop's default is one partition per physical core = 128 threads, so a login-node
  `pytest -m "not mpi"` fails **291 of 609** cases with `Resource temporarily unavailable`, surfacing
  as a repr that itself raises `RecursionError` — nothing about it points at threads. Run
  `monoprop_PARTITIONS=4 python -m pytest -m "not mpi"` instead: 609 passed, and 3.5× faster.
- **There is no `gh` module and no root.** A static release lives at `$PROJ/tools/gh/bin/gh`. It needs
  its own one-time `gh auth login` (device flow works headless) — `git push` works via VS Code's
  `GIT_ASKPASS` helper, which `gh` cannot reuse, and there is no credential helper or `GH_TOKEN`.

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
    cp -r hpc "$PROJ/src/$wt/hpc"
    (cd "$PROJ/src/$wt" && sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" \
         hpc/deucalion/sbatch/build-worktree.sh)
done
```

The `cp -r hpc` is needed because **`hpc/` lives only on the harness branch**, so a worktree checked
out from a library branch has none of it. (It is *not* gitignored — an earlier version of this note
said so, and `git check-ignore hpc/deucalion/README.md` clears it.)

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

   **On any branch without that fix, use `--cpu-bind=none`.** It is a launcher-side
   workaround and needs no engine change. Measured at 8 ranks × 16 partitions:

   | srun binding | `affinity_cpus` | threads pinned per rank | verdict |
   | --- | ---: | ---: | --- |
   | `--cpu-bind=cores` | 16 | 0 | unplaced |
   | `--cpu-bind=threads` | 16 | 0 | unplaced |
   | `--cpu-bind=none` | 128 | 16 | **placed** |

   Leaving all 128 CPUs visible is what lets the engine divide the node itself, and the
   ranks then take **disjoint** shares — verified by gathering every rank's actual pinned-CPU
   set, since `distinct_pinned_cpus` is per process and cannot tell "eight disjoint
   sixteenths" from "eight ranks on the same sixteen". The engine's own split is striped
   (rank 0 takes 0–3, 32–35, 64–67, 96–99), so it is not what `--cpu-bind=cores` would have
   granted; it is pinned and disjoint, which is what a measurement needs.

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
srun --mpi=pmix --cpu-bind=none --distribution=block:block \
     "$VENV/bin/python" your_script.py
```

- **`--mpi=pmix`**, not the default. Intel MPI would need `--mpi=pmi2`.
- **`--cpu-bind=none`, unless you are on a branch that has the per-rank-slice collapse.**
  See rule 3 above: `=cores` confines each rank to a slice the engine then refuses to divide
  again, and every rank runs unpinned on both arms of an A/B — which reads as a clean null.
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

### One command per PR

```bash
DRY_RUN=1 hpc/deucalion/pr-ab.sh profiling perf/layer-profile   # print the chain, submit nothing
hpc/deucalion/pr-ab.sh profiling perf/layer-profile
```

`pr-ab.sh` is a login-node submitter, not a job. It creates `$PROJ/src/ab-<pr>-port` at the
ref, copies `hpc/` and `benches/` in from this checkout, then submits build → `ctest -L serial`
→ `pytest --with-mpi` → the A/B cells → `pr_report.py`, chained with `--dependency=afterok`
so a red gate stops everything before a measurement is taken. Output is one
`$PROJ/runs/pr-<pr>/PR-AB-<pr>.md` in a format identical across every PR.

**Always dry-run first.** Running this file to read its own help once submitted a real
four-job chain and left a worktree behind.

The **common grid** is the same in every PR: `hubbard` c10/1.25e-5 (~97M terms, `propagate`
only — its graph path retains 29 layer-sets and exceeds a 242 GiB node) and `pauli` c14/5e-5
(~91M terms, all four operations), layout B 8×16, N=1 and N=2, 6 interleaved reps.
Highlight cells are added per PR as `label|ENV=v,ENV=v|-N1 -N2`:

```bash
EXTRA_CELLS='bind-cores|CPU_BIND=cores,ALLOW_BOTH_PLACED=0|-N1' \
    hpc/deucalion/pr-ab.sh placement pr/partition-cgroup-placement
```

`RANKS_PER_NODE`, `PARTITIONS`, `CPU_BIND`, `ALLOW_BOTH_PLACED` and `WORKLOAD` are pulled out
of that list and passed as `ab.sh`'s own variables, because each changes the label or a
refusal; anything else is exported to **both** arms.

**The baseline is built once and run every time.** `BASELINE_TREE` defaults to
`$PROJ/src/mp-main` and is pinned by its `_core.so` md5 in `hpc/deucalion/baseline.md5`;
`pr-ab.sh` refuses to submit if that binary moved, because a rebuilt baseline silently
re-bases every ratio in the campaign and `__version__` cannot catch it. The baseline is still
**re-run, interleaved, inside every allocation** — sharing the measurement rather than the
build would put per-allocation drift entirely on one side.

### The A/B driver itself

```bash
WORKLOAD=hubbard PORT_VENV=... sbatch -N1 -A "$MONOPROP_SLURM_ACCOUNT" -t 3:30:00 \
    --chdir="$PWD" hpc/deucalion/sbatch/ab.sh
```

`ab.sh` replaces `ab-100m.sh` and `models-ab.sh`, which were one protocol driving two bench
files; keeping two copies is how their summaries drifted into two formats. It runs both builds
in **one** allocation, order flipped every `(rep, cell)`, and reports time and memory per
operation for `build_graph`, `propagate`, `energy`, `gradient`.
Knobs: `WORKLOAD` (`hubbard`|`pauli`|`random`), `MAIN_VENV`, `PORT_VENV` (required),
`REPS` (6), `RESULTS_ROOT`, `RESULTS_TAG`, `CELL_SPEC`, `EXTRA_ENV`, `CPU_BIND` (`none`),
`ALLOW_BOTH_PLACED` (1), `RANKS_PER_NODE` (8), `PARTITIONS` (16). `benches/` comes from `$PWD`
for both arms, so only the compiled extension differs, and `monoprop_PARTITIONS` is exported
explicitly rather than left to the engine's auto path — that path reads the visible core count,
which is exactly what `--cpu-bind` and the placement fix perturb.

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

### Attributing memory per commit

`sbatch/membisect.sh` is the memory counterpart to the time A/B: it runs one fixed model per arm and
records the structural ledger (`operator_memory_breakdown()`), peak RSS both ways, and elapsed — so
one job yields per-commit **bytes** and **time** on the same runs.

```bash
ARMS_FILE=$PROJ/runs/mycampaign/arms RANKS=8 REPS=4 \
    sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" hpc/deucalion/sbatch/membisect.sh
$VENV/bin/python hpc/deucalion/tools/membisect_summary.py --baseline main $PROJ/runs/membisect-<jobid>
```

`ARMS_FILE` lines are `<arm> <worktree> [KEY=VAL ...]`; trailing pairs are exported for that arm only,
which is how one tree supplies several arms differing by an env knob. Cells come from
`cells/100m.cells` (`CELLS_FILE` to override, `CELLS` to select a subset). `PROFILE=1` adds
`monoprop_LAYER_PROFILE=1` for the per-phase split, read with `tools/layerprof_summary.py`.

### Reading a measurement here

Five rules that each exist because a table was believed and was wrong.

1. **State the `P`, and confirm a defect at the `P` where it was observed.** `R × S` is one flat world,
   and results do not transfer across it. R=8 understated one commit's time cost by **3×** on one
   model, and a peak-RSS regression measured at +7 B/term at R=8 **did not exist at R=1 at all** — so a
   job run at the convenient `P` measured a fix in a regime without the bug. R=1 is the primary *time*
   attribution surface (8.6× the probe volume, ~2× rather than ~20× barrier shadow, ±0.5% reps); R=8 is
   the shipping configuration. Neither substitutes for the other. Note also that phase *shares* are a
   function of `P` rather than of the code, so an optimisation must not be judged on a share measured
   at one `P`.
2. **Divide per rep, then take the median — never the ratio of two medians.** The arms of one rep run
   back to back on the same node, so a node-state swing cancels in the quotient. Ratio-of-medians once
   printed 6× noise as a clean `0.95x (flat)`. Report an `agree` count as the sign test: 3/3 is
   p=0.25, 4/4 is **p=0.125** (the best four reps can do), 6/6 is p=0.031 — and don't run a 6-rep A/B
   for a sub-2% expected effect. When reps disagree, check whether the odd one is a bad **arm** or a
   bad **denominator**: a baseline rep that moved makes every arm disagree at once.
3. **Collate from an explicit directory list, never a glob.** `$PROJ/runs` is shared by every
   concurrent session on the account. A `models-*` glob swept another branch's A/B into a campaign
   table and printed a **1.17× 6/6 regression that was not ours**; the tell was a ledger column of
   exactly 1.00×. Check the point count against the cells you actually submitted.
4. **The ledger's slack fields are virtual.** `d_terms_slack_bytes` and `d_invidx_arena_slack_bytes`
   are `capacity() − size()` — never faulted in. `total_bytes` counts them, resident memory does not,
   so read the TOUCHED column. Reading `total_bytes` made a commit that saves *nothing* at rest look
   like a campaign's biggest win. The converse trap is real too: a monolithic `std::vector` grown
   geometrically leaves only a small slack residue but produces a **copy transient** several times
   larger, which only peak RSS sees. Read the event, not the residue.
5. **Prove a knob moved a counter before believing the arm.** `parse_positive_int`
   (`detail/EnvConfig.h`) rejects values above `1'000'000` and then does `.value_or(<default>)`, so an
   out-of-range knob silently runs the *shipped* behaviour and two "different" arms come back
   byte-identical. `membisect.sh` refuses such an arm at preflight; there is no diagnostic in the
   engine. More generally: `two points are not a trend` — report points, hypothesise, and commit to a
   mechanism only after a falsifying point survives.

**Sizing a cell: `lower_atol` is the size knob, not `cutoff`.** At each model's default
`lower_atol=1e-4` both nominal size axes are *saturated* — hubbard reads 1,887,255 terms at both c10
and c11, and `num_sites` 60 and 90 are identical — so a sweep over them is a flat line that looks like
a null result. Terms scale as ~`atol^-1.9`. The 100M rungs are **hubbard c10 / 1.25e-5 = 96,981,051**
and **pauli c14 / 5e-5 = 91,273,861**; see `cells/100m.cells` and `tools/terms_calib.py`. Size a
`build_graph` cell from `graph_vmhwm_mib`, never from the propagate figure: `propagate` releases each
layer as it contracts it while `steps × build_graph` retains all of them, and hubbard's 29 Trotter
steps propagate 23.9M terms in 1.7 GiB but build the graph in more than 229 GiB.

---

## 11. Build identity and provenance

**The single most expensive class of mistake here.** Every item below is a way of measuring a binary
that is not the one you think, and each fails silently.

### An arm is its installed `_core.so` md5, and nothing else

Two identifiers look like they would work. Both are stale, and they fail **together**:

- **`monoprop.__version__` reports HEAD for every arm.** monoprop installs *editable*: every venv's
  `_editable_skbc_monoprop.pth` resolves the Python layer to the same live source tree, so
  `__version__` reads whatever HEAD is. A control venv built two commits earlier advertised the very
  change it was controlling for.
- **The dist-info stamp is written at install time and never rewritten by a later `cmake` rebuild.** An
  arm installed at one commit, rebuilt through three more, still advertised the original string — the
  same string as the baseline.

This cost a whole measurement wave: `ab_summary.py`'s provenance guard keyed on the version, saw one
string on both sides, and refused five cells that were provably distinct binaries. Hash instead:

```bash
# what the interpreter will actually load -- site-packages holds a COPY, not a view
python -c 'from monoprop import _core; print(_core.__file__)'
```

Two rules on the hash itself:

- **Assert exactly one match, never `find … | head -1`.** scipy ships
  `optimize/_highspy/_core.cpython-*.so`, and both arms install the same scipy — a loose glob has made
  two genuinely different arms compare byte-identical, a false negative in exactly the direction the
  check exists to catch. `build.sh`, `membisect.sh` and `ctest-worktree.sh` all do this.
- **A build script must not be able to fail in the checking.** `build.sh` hashes the *file* rather than
  importing it: an import needs `foss/2025b` loaded or it dies on `CXXABI_1.3.15 not found` against the
  system libstdc++, and under `set -e` that would make a successful build report failure.

### A `ninja` rebuild does not reach the venv

`cmake --build build/editable/Release` relinks the build tree's `_core.so` and `libmonoprop.so`.
Python loads **neither**. `site-packages/monoprop/` holds *copies* of both, placed there at `uv sync`
time, and they do not update when ninja runs.

This cost a job: a probe was added, rebuilt, and the sbatch gated on
`strings <build-dir .so> | grep <symbol>` — which passed. Every stage then reported "no probe files",
reading exactly like "no cross-rank traffic happened". The job had run the previous day's binary
throughout. A gate on the build dir proves the *compiler* ran, not that the measured thing contains
the change.

- After `cmake --build`, copy **both** artifacts into site-packages or re-run
  `uv sync --reinstall-package monoprop`. Copying only `_core.so` leaves engine changes behind — most
  of the C++ is in `libmonoprop.so`.
- Back up the site-packages pair before overwriting if a measured arm depends on it, or the campaign's
  binary is gone.
- **`strings … | grep -q` under `set -o pipefail` reports FAILURE on a match** — `-q` exits early,
  `strings` takes SIGPIPE, and the pipeline status is the failure. Use `grep -c`.

### One build dir per worktree, or concurrent jobs corrupt each other

`pyproject.toml` sets `build-dir = "build/{state}/{build_type}"`, and **neither placeholder encodes the
cmake defines or the venv** — so every configuration in a worktree shares one directory. Two jobs
submitted together raced on it: 3 of 4 arms died with `configure_file: No such file or directory`, and
ctest then ran against the half-written tree and reported a meaningless **262/262 passed**.

The venv is not the isolation boundary it looks like: `UV_PROJECT_ENVIRONMENT` separates where the
wheel is *installed*; the compile happens in the shared tree either way. So pass a per-arm build dir —
`build.sh` now does this via `MONOPROP_BUILD_TAG`, and `ctest-worktree.sh` via `TAG`. Never edit the
shipped default. Then three defences, because one is not enough: per-arm build dir; a guard that
aborts if another job is already building this worktree; and **check the build rc and abort** — never
fall through to ctest on a failed build.

### Rebuilding the editable tree in place

`cmake --build build/editable/Release` **cannot reconfigure** that tree: its cache pins the
build-isolation interpreter to a scikit-build-core temp dir that no longer exists, so ninja fails
regenerating `build.ninja`, and the `skbuild-*` presets inherit the failure because they only adopt the
tree and set no cache variables. `CMakeLists.txt` globs with `CONFIGURE_DEPENDS`, so the tree cannot
skip regeneration either.

Use `sbatch/ctest-worktree.sh`, which configures its own tree with the three defines that a plain
`cmake -S . -B` does not supply (`Python_EXECUTABLE`, `SKBUILD_PROJECT_VERSION_FULL`, `nanobind_DIR` —
see the script's comments for why each fails late without it). If you must repair the editable tree
itself, `uv pip install nanobind` into `.venv` first and reconfigure **after** any `uv sync`, which
re-poisons `Python_EXECUTABLE` with a fresh ephemeral path — then *assert* `monoprop_ENABLE_MPI` and
`MAX_NUM_MODES` afterwards, because a test binary silently built without MPI passes while testing
something else.

Do not assume in either direction whether `uv sync` rebuilt the C++ test binary — it sometimes does.
The binary is at `build/editable/Release/bin/`, **not** `cpp/tests/`; checking the wrong path produced
a bogus "no test binary" abort. Verify with evidence: `ninja -n <target>` reporting `no work to do`,
plus `strings <bin> | grep -c <new_test_name>`.

---

## 12. Testing on this cluster

### A slow CTest run is `MPI_Init`, not the tests

CTest runs **each Boost case as its own process**, so an MPI-enabled build pays a full `MPI_Init` per
case — and `MPI_Init` initialises **every fabric device present**, whether or not the process will ever
send a message. There are 8 `mlx5_*` HCAs here, which is **8.8 s wall against 0.17 s user + 0.92 s
sys** per case: 224 cases = 34 minutes of blocked process.

**The tell is wall time with no CPU behind it** — `/usr/bin/time -f "user %U sys %S wall %e"` settles it
in one run. Ruled out by measurement: dynamic linking (28 ms), cold page cache, hostname resolution,
and the tests themselves (`--list_content`, which runs nothing, cost the same 9.24 s).

| exclusions | wall |
| --- | --- |
| none | 8.8 s |
| `OMPI_MCA_btl=^openib,ofi,uct` | 6.4 s |
| `OMPI_MCA_pml=^ucx` | 4.2 s |
| both | **1.9 s** |

**`pml=^ucx` carries most of the win and it HANGS multi-rank runs** — a 2-rank case that passes in
29 ms hangs indefinitely under it, on pre-branch commits too, so it is component selection rather than
engine code. Every UCX-restricting variant that reaches ~2 s hangs likewise. So the remedy is
**scoping, not tuning**: apply the exclusions to the per-case `serial` variants *only*, via a
`SERIAL_ENVIRONMENT` argument to `discover_tests` in `cpp/tests/boost-test.cmake`. Measured that way
the suite went 34 min → **6.8 min, 224/224**. Verify the split with `ctest --show-only=json-v1`; the
number that matters is *mpi cases with exclusions == 0*.

> **Not on `main`, and not on this branch.** `SERIAL_ENVIRONMENT` exists only on
> `perf/multinode-comm-scaling`. So a CTest run on any other branch still pays the full 8.8 s per case
> — budget `--time` accordingly (`ctest-worktree.sh` allows 1:30:00 for build + both gates), and do not
> read a slow suite as a hang.

- **Never name a positive component list.** `vader` was renamed `sm` in Open MPI 5, so
  `OMPI_MCA_btl=self,vader` there silently reduces to `self` alone — and it looks fine, because
  single-process cases need no transport. `ompi_info --param btl vader` returning **0 lines** is the
  check. Unknown `OMPI_MCA_*` variables are ignored by other MPIs, so this is inert under MPICH.
- **CMake trap when attaching env:** the `ENVIRONMENT` test property takes ONE value holding every
  `VAR=value`, and separators must be **escaped** semicolons. A plain `"${list}"` flattens (a CMake
  list *is* a semicolon-joined string), so entries after the first are read as property names — it
  configures cleanly and silently drops them.
- `mpirun` binds each rank to **one core** by default, so a whole-suite multi-rank ctest entry runs the
  engine on one core per rank.

### One failure is not a rate

At least one case asserts on **allocator behaviour** rather than on library behaviour:
`lazy_fold_survives_operator_growth` requires a reallocated buffer to land at a *new* address, and
malloc is entitled to hand the freed block straight back.

In one job the **same binary** failed it under `ctest -L serial` and passed it under `ctest -L unit`.
A controlled rerun — 60 reps × 3 trees, plus 8 on the login node — came back **180/180**. So the rate
is under ~1/190, the case is genuinely flaky, and the failure was **not attributable** to the patch
under test. Establish the rate before letting a single failure void a build:

```bash
CASES=<case> BUILDS="<build-dir> [more]" REPS=60 \
    sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" hpc/deucalion/sbatch/ctest-repeat.sh
```

A zero-failure sweep is an **upper bound** on the rate, not proof of determinism — quote it as "0 of
N". And note what such a sweep cannot do alone: a pre-patch baseline needs a build tree, and a
worktree holding only `build/editable` has no `CTestTestfile.cmake` for ctest to read, so "the baseline
was not covered" must be said rather than implied.

### Which gates count

- **`ctest -L serial`** is the gate; `-L unit` is its superset and worth running too so numbers stay
  comparable across a campaign. `sbatch/ctest-worktree.sh` runs both and reports each rc.
- **`ctest -L mpi` fails on `main` as well** (`shm_comm_oversubscribed` aborts at 2 ranks) and is
  **not** a signal. Use `sbatch/mpi-tests-worktree.sh`, which drives the MPI cases and the Python
  suite across four rank/partition layouts, or a targeted `mpirun`. That script reads
  `build/editable/Release-<tag>/cpp/tests` (build.sh tags the build dir per arm; the tag defaults to
  the worktree basename) — note the registry is rooted one level down from the build root
  (`enable_testing()` is in `cpp/CMakeLists.txt`), and pointing ctest at the root reports "No tests
  were found!!!" and exits **0**, a silent pass.
- The MPI variants shell out to `mpiexec` themselves, so they need the batch context and OpenMPI
  **5**'s spelling of the oversubscribe knob: `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`.
  The justfile's `OMPI_MCA_rmaps_base_oversubscribe` is the v4 name and does nothing here.
- **`-s` is load-bearing under pytest.** Capture is fd-level and the engine writes `COMMPROF` /
  `LAYERPROF` straight to fd 2 from a static destructor, so without it a live instrument looks exactly
  like one that never fired. `tools/prof_run.py` sidesteps this by not being a pytest test at all.
- **Deleting a test can delete coverage silently.** "It no longer compiles" is not "it is no longer
  needed", and a surviving assertion can be vacuous (`-0.0 == 0.0` is true). Mutation-test a ported
  test: break the thing it covers and confirm it goes red.
