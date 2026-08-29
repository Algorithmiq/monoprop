# Running the rungs

`rungs.toml` is monoprop's benchmark set. This page is how to run it: what each model's
parameters mean, how the geometry maps onto a machine, and example Slurm scripts.

Nothing here runs automatically. No CI workflow gates a pull request on these. Running the
rungs your change could plausibly move, and putting the numbers in the pull request, is your
job — and `monoprop-bench-ladder` prints the block to paste.

## The three commands

```bash
# What the set contains, and what one rep of each last cost.
monoprop-bench-rung benches/rungs.toml list

# The exact pytest line, geometry and environment, without spending anything.
monoprop-bench-rung benches/rungs.toml n1-100m-hubbard-propagate --dry-run

# One rep. Repeat with --rep 2, 3, ... for the rest.
monoprop-bench-rung benches/rungs.toml n1-100m-hubbard-propagate --rep 1 --results runs

# Collate every gate-clean rep into the block to paste into the pull request.
monoprop-bench-ladder benches/rungs.toml runs
```

`list` marks an uncalibrated row with `*` and shows `?` where nobody has timed it:

```
  n1-100m-hubbard-propagate    size   N=1   P=128      96,981,051  ?
  weak-97m-n4                  weak   N=4   P=512     377,482,074  13.1s 11.5GiB/node
* st-10m-random-propagate      size   N=1   P=1                 0  ?
```

## What a row declares

| field | meaning |
| --- | --- |
| `id` | unique; names the artifacts, so it is stable once a rung has been run |
| `family` | `size`, `strong` or `weak` |
| `picture` | `heisenberg` or `schrodinger`; only the random model has this axis |
| `model` | `random`, `hubbard` or `pauli` |
| `ops` | any of `build_graph`, `propagate`, `energy`, `gradient` |
| `nodes`, `ranks_per_node`, `partitions` | the geometry; see below |
| `args` | `--<option>=<value>` handed to pytest verbatim — the size knobs live here |
| `expect_terms` | **the gate**: the exact term count this configuration produces |
| `reps` | how many separate process launches to run |
| `cost_seconds`, `cost_gib_per_node` | what one rep last cost — documentation, never a gate |
| `walltime` | a Slurm allowance for one rung, reps included |

Two of these behave in opposite directions, and it matters which is which.

**`expect_terms` is a gate.** A result missing it by more than 0.1% is refused and the artifact
renamed `.refused.json`. The term count is reproducible to the digit at a fixed seed and
tolerance, so this catches a mistyped knob before it becomes a number in a pull request. The
geometry and the timing-round count are checked the same way.

**`cost_seconds` and `cost_gib_per_node` are documentation.** They say what one rep took the
last time anyone measured it, so you can size an allocation before you spend it. A timing is
noisy and your machine is not the machine they were taken on; nothing gates on them.

A row nobody has calibrated says so twice — `expect_terms = 0` and `TBD` in place of every
unmeasured size knob — and refuses to run. `TBD` rather than `0` because `0` is not neutral:
`lower_atol = 0` prunes nothing and is the *largest* problem the model can pose.

## The models and their parameters

Override any field with `--<model>-<field>`, underscores becoming hyphens:
`--hubbard-trotter-steps=2`, `--pauli-lower-atol=5e-05`.

The resolved values — not just your overrides — are recorded in each run's artifact and printed
by `monoprop-bench-ladder` under `## Problems measured`, so a reviewer never has to resolve your
flags against these defaults by hand.

### `hubbard` — 1D Fermi-Hubbard under first-order Trotter

Heisenberg picture only. Calls `propagate` once per Trotter step, so a cost paid per call shows
up here and is nearly invisible on `pauli`.

| `--hubbard-…` | default | what it does |
| --- | --- | --- |
| `num-sites` | 60 | lattice sites; 2 spin orbitals each, so 120 modes. At most 125 |
| `hopping` | 1.0 | nearest-neighbour hopping amplitude `t` |
| `interaction` | -2.0 | on-site interaction `U` |
| `chemical-potential` | 0.0 | `mu` |
| `trotter-dt` | 0.2 | time step |
| `trotter-steps` | 29 | Trotter steps, and therefore `propagate` calls |
| `observable-site` | 46 | site the measured number operator sits on |
| `observable-spin` | `up` | spin of that operator |
| `neel-start-spin` | `down` | spin on site 0 of the Néel initial state |
| `cutoff` | 6 | maximum Majorana-string length kept |
| `lower-atol` | 1e-4 | **the size knob** — prune coefficients below this |

### `pauli` — kicked Ising on an IBM Eagle heavy-hex lattice

Heisenberg picture only. One circuit layer per `num-layers`, so all four operations fit.

| `--pauli-…` | default | what it does |
| --- | --- | --- |
| `num-qubits` | 127 | heavy-hex lattice size — **effectively fixed**, see below |
| `num-layers` | 20 | kicked-Ising layers |
| `observable-qubit` | 62 | qubit the measured `Z` sits on |
| `theta` | π/4 | single-qubit X-rotation angle |
| `coupling` | π/4 | two-qubit ZZ coupling angle |
| `cutoff` | 8 | maximum Pauli-string weight kept |
| `lower-atol` | 1e-4 | **the size knob** |

### `random` — random Majorana generators

The only model with both pictures. Its options carry no model prefix.

| option | default | what it does |
| --- | --- | --- |
| `--num-generators` | 100 | random generators, i.e. circuit gates |
| `--num-modes` | 128 | fermionic modes — **at most 250**, see below |
| `--gen-length` | 4 | Majorana operators per generator |
| `--obs-terms` | 10000 | **the size knob** — terms in the observable |
| `--cutoff` | 6 | truncation cutoff |
| `--seed` | 0 | random seed; the term count is only reproducible at a fixed seed |

The Schrödinger rungs set `schrodinger_cutoff = cutoff + 2`. Its cap is lower than the
Heisenberg one at the same knobs, so calibrate the two separately rather than assuming they match.

### How large can the system be

The system-size axes are not all free, and two of them have hard ceilings that are not obvious
from the option list.

**Modes are capped at 250 by the build.** `monoprop_MAX_NUM_MODES` is a CMake cache variable
defaulting to `250`, and the extension ships propagator variants in strides of 32 up to 256. So
`--num-modes` accepts at most 250 and `--hubbard-num-sites` at most 125 (two spin orbitals per
site) unless you rebuild with a larger `monoprop_MAX_NUM_MODES`. The random rungs sit at 250,
which is the ceiling.

**`--pauli-num-qubits` is effectively fixed at 127.** The kicked-Ising circuit is built over
`HEAVY_HEX_TOPOLOGY`, a hard-coded IBM Eagle coupling map: 144 pairs whose highest index is 126.
Passing anything below 127 fails with `If a term acts on a qubit index >= num_qubits`, and
anything above 127 just adds qubits no gate touches. Size the Pauli model with `--pauli-cutoff`,
`--pauli-num-layers` and `--pauli-lower-atol` instead. Benchmarking a different lattice means
adding a second topology to `models.py`, not changing a flag.

**Hubbard's lattice is free**, up to the 125-site ceiling above, and `--hubbard-trotter-steps`
scales the `propagate` call count independently of the term count.

### Why `lower_atol` is the size knob

At the default `lower_atol = 1e-4` both fixed models are saturated in their other axes:
`--hubbard-cutoff` is flat from 10 and `--hubbard-num-sites` flat from 60, so sweeping either
measures nothing. Tightening `lower_atol` is the only axis that reaches 100M terms and beyond.

Shrinking a model needs its observable index too — `--hubbard-observable-site` defaults to 46 and
`--pauli-observable-qubit` to 62, so a smaller lattice must move the observable or the build
fails with an index past the end.

## Geometry

Three numbers, and one derived:

- **`nodes`** — compute nodes in the allocation.
- **`ranks_per_node`** — MPI ranks per node. `ranks = nodes × ranks_per_node`.
- **`partitions`** — engine partitions per rank, one thread each.
- **`P = ranks × partitions`** is the flat world the engine sees, and the number that matters:
  phase shares are a function of `P`, not of the code. The same change can look like a win at
  `P=128` and a loss at `P=4096`, so a result quoted without its geometry says little.

`ranks_per_node × partitions` should equal the node's physical core count. The two layouts the
scaling rungs were taken at are `1 × 128` (one rank spanning the node) and `8 × 16` (one rank per
NUMA domain on a dual EPYC 7742).

The runner exports `monoprop_PARTITIONS` and `monoprop_NUM_THREADS` from the row; you supply the
ranks through `srun`. The gate checks that what actually ran matches the row, so a mismatch fails
the cell rather than producing a mislabelled number.

## Running one rung locally

Single-rank rungs (`ranks_per_node = 1`) need no MPI:

```bash
monoprop-bench-rung benches/rungs.toml st-10m-hubbard-propagate --rep 1 --results runs
```

On a login or shared node, cap the threads first. `nproc` reports the whole machine, not your
share, and an uncapped construction phase will fail with `Resource temporarily unavailable`:

```bash
export monoprop_NUM_THREADS=4 OMP_NUM_THREADS=4
```

Anything past the smallest rungs wants a compute node.

## Slurm

The launcher is not in this repository — these are examples to adapt, not a supported script.
Substitute your own account, partition and module names.

### One rung on one node

```bash
#!/bin/bash
#SBATCH --job-name=monoprop-rung
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8         # = ranks_per_node
#SBATCH --cpus-per-task=16          # = partitions
#SBATCH --exclusive
#SBATCH --time=0:20:00

set -euo pipefail

RUNG=n1-100m-hubbard-propagate
REPS=10
RESULTS="$SLURM_SUBMIT_DIR/runs/$RUNG"
mkdir -p "$RESULTS"

module load foss/2025b                  # your MPI + GCC >= 14 toolchain
source .venv/bin/activate

for rep in $(seq 1 "$REPS"); do
  srun --cpu-bind=cores --distribution=block:block \
    monoprop-bench-rung benches/rungs.toml "$RUNG" --rep "$rep" --results "$RESULTS" || true
done

monoprop-bench-ladder benches/rungs.toml "$RESULTS"
```

Each rep is a separate `srun`, and that is deliberate: `pytest-benchmark`'s `pedantic` builds
round *k+1*'s arguments before releasing round *k*'s, so a second timing round inside one process
holds two propagators and doubles peak RSS. The runner forces `--bench-rounds=1` for that reason.
Repetition has to come from repeated process launches.

The `|| true` is what keeps a refused rep from taking the rest of the job with it under `set -e`;
one refusal is not a reason to lose the other nine. `monoprop-bench-ladder` skips refused
artifacts and reports how many reps it actually found, so a partial run is still readable. Drop
the `|| true` if you would rather the job stop on the first refusal — but read the message either
way, because it names exactly which check failed.

### One rung across many nodes

```bash
#!/bin/bash
#SBATCH --job-name=monoprop-strong
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=1:15:00
```

The body is identical — `srun` inherits the node count, and the runner reads `nodes` from the row
only to check that it matches. Confirm with `--dry-run` before submitting: it prints the derived
`R` and `P` alongside the declared cost, which is the cheapest place to catch a wrong geometry.

### A ladder in one allocation

Rungs sharing a node count can share a job. Keep one results directory per rung so the
artifacts stay separable:

```bash
for rung in strong-1569m-n8 strong-6126m-n8 strong-24420m-n8; do
  for rep in $(seq 1 5); do
    srun --cpu-bind=cores --distribution=block:block \
      monoprop-bench-rung benches/rungs.toml "$rung" --rep "$rep" --results "runs/$rung" || true
  done
done
monoprop-bench-ladder benches/rungs.toml runs --family strong
```

### Running everything

A whole family needs one allocation per node count, so the job list is the table grouped by
`nodes`. Generate it rather than typing it — the table is the only place that knows:

```bash
# jobs.txt: one line per allocation, "<nodes> <rung> <rung> ..."
python - <<'EOF' > jobs.txt
import collections, pathlib
from monoprop_bench_tools import rungs

FAMILY = "strong"          # or "weak", or "size", or None for the lot
table = rungs.load_rungs(pathlib.Path("benches/rungs.toml"))
jobs = collections.defaultdict(list)
for r in table.values():
    if not r.calibrated:
        continue           # uncalibrated rows refuse to run; calibrate them first
    if FAMILY and r.family != FAMILY:
        continue
    jobs[r.nodes].append(r.id)
for nodes, ids in sorted(jobs.items()):
    print(nodes, *ids)
EOF
cat jobs.txt
```

```
1 strong-1569m-n1
2 strong-1569m-n2 strong-6126m-n2
4 strong-1569m-n4 strong-6126m-n4
8 strong-1569m-n8 strong-6126m-n8 strong-24420m-n8
16 strong-1569m-n16 strong-6126m-n16 strong-24420m-n16
32 strong-1569m-n32 strong-6126m-n32 strong-24420m-n32
64 strong-1569m-n64 strong-6126m-n64 strong-24420m-n64
```

Submit one job per line. `run_rungs.sh` takes the node count and the rung ids as its arguments:

```bash
#!/bin/bash
#SBATCH --job-name=monoprop-ladder
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

set -uo pipefail                  # not -e: a refused rep must not kill the job

cd "$SLURM_SUBMIT_DIR"
module load foss/2025b
source .venv/bin/activate

for rung in "$@"; do
  reps=$(python -c "
import pathlib, sys
from monoprop_bench_tools import rungs
print(rungs.load_rungs(pathlib.Path('benches/rungs.toml'))['$rung'].reps)")
  for rep in $(seq 1 "$reps"); do
    srun --cpu-bind=cores --distribution=block:block \
      monoprop-bench-rung benches/rungs.toml "$rung" --rep "$rep" --results runs
  done
done
```

```bash
while read -r nodes rungs_on_this_node; do
  sbatch --nodes="$nodes" --time=2:00:00 run_rungs.sh $rungs_on_this_node
done < jobs.txt
```

Then collate once, over every job's artifacts:

```bash
monoprop-bench-ladder benches/rungs.toml runs --family strong
```

### What you get out

One table per family plus the resolved parameters. The `strong` table from the campaign whose
numbers this repository ships looks like this — 8 ranks/node × 16 partitions on 242 GiB EPYC 7742
nodes, five reps per rung:

```
## strong

| rung | nodes | R | P | terms | Mterms/node | reps | median s | min s | Mterms/s/node | GiB/node | declared s | vs declared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong-1569m-n1 | 1 | 8 | 128 | 1,569,152,761 | 1569 | 5 | 215.60 | ... | 7.28 | 99.9 | 215.60 | 1.000x |
| strong-1569m-n2 | 2 | 16 | 256 | 1,569,152,761 | 785 | 5 | 105.90 | ... | 7.41 | 51.5 | 105.90 | 1.000x |
| strong-1569m-n4 | 4 | 32 | 512 | 1,569,152,761 | 392 | 5 | 53.30 | ... | 7.35 | 28.0 | 53.30 | 1.000x |
| strong-1569m-n8 | 8 | 64 | 1024 | 1,569,152,761 | 196 | 5 | 28.60 | ... | 6.86 | 15.9 | 28.60 | 1.000x |
| strong-1569m-n16 | 16 | 128 | 2048 | 1,569,152,761 | 98 | 5 | 18.10 | ... | 5.41 | 11.0 | 18.10 | 1.000x |
| strong-1569m-n32 | 32 | 256 | 4096 | 1,569,152,761 | 49 | 5 | 16.10 | ... | 3.04 | 6.6 | 16.10 | 1.000x |
| strong-1569m-n64 | 64 | 512 | 8192 | 1,569,152,761 | 25 | 5 | 24.00 | ... | 1.02 | 4.1 | 24.00 | 1.000x |

## Problems measured

- **strong-1569m-n8** — hubbard / heisenberg, propagate, 8 x 8 x 16 (nodes x ranks/node x partitions)
  - chemical_potential=0.0, cutoff=10, hopping=1.0, interaction=-2.0, lower_atol=1.25e-05,
    neel_start_spin=down, num_sites=60, observable_site=46, observable_spin=up,
    trotter_dt=0.2, trotter_steps=29
```

Those medians are the campaign's, replayed through the collator, so `vs declared` is `1.000x` by
construction and `min s` is elided — a real run fills both. The **shape** is what to expect, and
the shape is the answer: this ladder halves cleanly to 8 nodes, gains almost nothing from 16 to
32, and is *slower* at 64 than at 32. `Mterms/s/node` is where you read that — 7.28 at one node,
6.86 at eight, 1.02 at sixty-four. A strong-scaling curve that turns over is the result, not a
failed run.

### Things that quietly cost you

Measured on Deucalion (AMD EPYC 7742, 128 physical cores per node, SMT off). The numbers are
that machine's; the failure modes are not.

- **Let Slurm do the pinning.** `--cpu-bind=none` measured a 1.45x penalty against
  `--cpu-bind=cores`. A Slurm cgroup with no pinning at all still beat the engine placing threads
  over an unconfined mask, so the engine's own placement is not a substitute.
- **Keep `ranks × partitions` an exact divisor of the cores.** Above the core count, thread
  placement falls back to unpinned silently — no warning, just a slower run. Involuntary context
  switches are the tell: they jumped by four orders of magnitude when pinning was lost.
- **Leave `MALLOC_ARENA_MAX` alone.** Setting it to the partition count cost ~16% of wall, because
  the rank runs more threads than partitions and they then contend for too few arenas.
- **`--exclusive`, always.** A shared node makes the timing a measurement of your neighbour.
- **Add `-s` if you are debugging a crash.** pytest's fd capture swallows the C++ layer's stderr,
  so a run can look like it produced nothing when it in fact printed the reason.

### Sizing an allocation

Peak resident memory over the 38 scaling rungs fits

```
GiB/node = 3.39 + 0.0634 × Mterms/node
```

within −2.9/+4.5 GiB/node of the line. Use it to pick a node count, not to predict a rung: it is
a fit over one campaign on one machine, and `cost_gib_per_node` on the row itself is the better
number where a row has one.

`build_graph` does not obey it. It *extends* the graph rather than replacing it, retaining one
layer-set per gate, while `propagate` releases each layer as it contracts. Hubbard reaches 97M
terms through `propagate` in well under 2 GiB, yet four successive `build_graph` calls on the same
model exceed a 242 GiB node. That is why the graph-holding rungs stop at a single node.

## Reporting the result

```bash
monoprop-bench-ladder benches/rungs.toml runs
```

prints two things: a table per family, and a `## Problems measured` section listing each rung's
resolved parameters. Paste both. Read the median and the min — these distributions are skewed, so
a mean tracks stragglers rather than the cost of the work.

`vs declared` compares your median to the row's `cost_seconds`. Treat it as context, not a verdict:
it may have been taken on other hardware, at another commit, under another Slurm configuration.
If your run *is* the new reference, update `cost_seconds` and `cost_gib_per_node` in the row and
say in the pull request which machine and commit they came from.

## Calibrating an uncalibrated row

A `TBD` row cannot run, so measure it before you fill it in:

1. Copy the row into a scratch table, replace `TBD` with your candidate knob value and set
   `expect_terms` to something deliberately wrong.
2. Run one rep. The gate refuses it and prints the term count it actually measured.
3. Put that number in `expect_terms`, and the knob value in `args`. Run again; it should pass.
4. Fill in `cost_seconds` and `cost_gib_per_node` from the ladder, and open a pull request with
   the row and the machine it was measured on.

Step 2 is not a workaround. Reading the count off a refusal is how the gate is meant to be used,
and it means a calibrated row has been seen to pass at least once.
