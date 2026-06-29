# monoprop benchmarks

A pytest-based benchmark suite measuring the **time** and **peak memory** of
monoprop's core operations. Timing uses
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io); memory uses
[`pytest-memray`](https://pytest-memray.readthedocs.io), whose allocator-level
tracking captures the C++-side graph and coefficient allocations that dominate
this library (Python's `tracemalloc` would miss them).

This directory is **not** collected by the normal `pytest` run (`testpaths` is
`tests/`). Run the benchmarks explicitly with `just bench`.

## Running

There is one command. `just bench` runs the timing pass, the memory pass, and
regenerates `benches/results/REPORT.md`:

```bash
just bench                       # serial (label np1)
just bench-smoke                 # quick sanity check (tiny sizes, no memory pass)
```

Anything after `just bench` is handled by the driver (`benches/run.py`):
driver options (`--ranks`, `--mpiexec-args`, `--no-mem`) are consumed, and
everything else is forwarded to pytest — so you size the random benchmarks,
choose the round count, or select tests the same way:

```bash
just bench --num-generators 200 --num-modes 64 --cutoff 10
just bench --bench-rounds 20 benches/bench_random_evolve.py::test_energy
just bench --no-mem              # timing only
```

## Configuring a run (MPI, threads, pinning)

Configuration is external and explicit — you set it on the command line and the
driver records it in the report. Three knobs:

| Knob | How | Example |
|---|---|---|
| MPI ranks | `--ranks N` | `--ranks 4` |
| mpiexec args (incl. rank pinning) | `--mpiexec-args` | `--mpiexec-args="--bind-to core"` |
| threads & thread pinning | `--env KEY=VAL` (repeatable) | `--env monoprop_NUM_THREADS=2` |
| run name / report column | `--label` | `--label r4t2-pinned` |

`monoprop` reads its thread count from `monoprop_NUM_THREADS` at import; add
`OMP_NUM_THREADS` / `OMP_PROC_BIND` / `OMP_PLACES` for OpenMP-side pinning. Give a
custom `--label` when you vary threads/pinning at a fixed rank count so runs do
not overwrite each other.

```bash
# threads only (serial), named column
just bench --env monoprop_NUM_THREADS=4 --label t4

# 4 ranks × 2 threads, cores pinned
just bench-build-mpi                                       # one-time MPI rebuild
just bench --ranks 4 --mpiexec-args="--bind-to core" \
    --env monoprop_NUM_THREADS=2 --env OMP_PROC_BIND=close --label r4t2
```

The benchmarks are communicator-aware: each operation is barrier-wrapped so the
timed cost is the **makespan** across ranks, only rank 0 writes the timing JSON,
and memory is captured per rank (the report shows the worst-rank peak). MPI uses
fixed-round `pedantic` timing because pytest-benchmark's auto-calibration would
let ranks diverge and deadlock at the barriers; set the count with
`--bench-rounds` (default 5).

**MPI build must be loaded, and stay loaded.** `monoprop` builds with
`monoprop_ENABLE_MPI=OFF` by default; a non-MPI extension silently ignores the
communicator, so every rank holds the **full** operator (no distribution, and
N× the memory — this OOMs at large sizes). `just bench-build-mpi` installs the
MPI build, and `just bench` runs with `uv run --no-sync` so a later run does not
re-sync a default non-MPI build over it. Because of `--no-sync`, sync deps once
up front (`uv sync --all-groups --all-extras`, or `just bench-build-mpi`), and
rebuild explicitly after editing monoprop sources.

Before the heavy passes, an `--ranks > 1` run runs a fast **distribution
preflight** (`benches/_mpi_check.py`): it builds a tiny problem and checks that
each rank holds only a *fraction* of the operator. If the loaded build is not
distributing it aborts immediately with remediation instead of OOMing hours
later. Pass `--skip-mpi-check` to bypass it. To run the check on its own:

```bash
mpiexec --allow-run-as-root -n 4 uv run --no-sync python benches/_mpi_check.py
```

## Reporting

Each run regenerates `REPORT.md`, which has:

- **Configuration** — one row per run label: ranks, thread counts, launcher (incl.
  pinning args), CPU count, host.
- **Hyperparameters** — the resolved random-problem sizes and run knobs each run
  actually used (defaults included, not just CLI overrides), one column per label.
- **Graph size** — the number of terms in the evolved operator reached per
  picture (Heisenberg / Schrödinger), one column per label.
- **Heisenberg** and **Schrödinger** — one section each, every section holding a
  **Time** and a **Peak memory** table (one row per operation, one column per
  label). The Schrödinger section is omitted when no Schrödinger ops were run.

Every run is one column, so serial / MPI / thread variants sit side by side. To
rebuild the report from existing artifacts without re-running anything:

```bash
uv run --group bench python benches/report.py
```

## Benchmarks

All benchmarks live in a single file, `bench_monoprop.py`.

### Random (configurable, both pictures)

`make_random_problem` builds `--num-generators` random fixed-length Majorana
generators and a random Hermitian observable with a configurable number of
terms. Each random operation is run in **both** the Heisenberg and Schrödinger
pictures (the latter with `schrodinger_cutoff = cutoff + 2`), shown as
`[heisenberg]` / `[schrodinger]` variants. All sizes are CLI options (defaults
in parentheses):

| Option | Meaning | Default |
|---|---|---|
| `--gen-length` | Majorana operators per generator | 4 |
| `--obs-terms` | number of observable terms | 10000 |
| `--num-generators` | number of generators (circuit gates) | 100 |
| `--num-modes` | fermionic modes (Majorana indices = `2·modes`) | 128 |
| `--cutoff` | truncation cutoff | 6 |
| `--seed` | RNG seed | 0 |

Operations: `test_random_build_graph`, `test_random_pare` (masked execution
plan), `test_random_energy`, `test_random_gradient` (graph-based path), and
`test_random_inplace` (in-place coefficient truncation, never materialising the
graph).

### Static (fixed, in-place, Heisenberg only)

- `test_static[hubbard]` — 120-qubit Fermi-Hubbard model (60 sites), sandbox
  default input, run as a 29-step in-place Trotter trajectory.
- `test_static[pauli]` — 127-qubit Pauli-basis kicked-Ising simulation (IBM Eagle
  heavy-hex, 20 layers, ⟨Z₆₂⟩).

Both are marked `slow`. Pass `--lower-atol VALUE` to override their coefficient
truncation tolerance (defaults: Hubbard 1e-5, Pauli 1e-4).

## Notes

- The driver passes `-o filterwarnings=default` to pytest, overriding the
  project-wide `filterwarnings=["error"]` so benchmark-plugin warnings do not
  fail the run.
- Static benchmarks use `benchmark.pedantic(rounds=1)` so the multi-second
  simulations are not re-run hundreds of times.
- Files are organised as: `run.py` (the driver), `report.py` (Markdown report),
  `conftest.py` (fixtures + CLI options), `_builders.py` (model construction),
  and `bench_monoprop.py` (the unified benchmark suite).
