# monoprop benchmarks

A pytest suite measuring the **time** ([`pytest-benchmark`](https://pytest-benchmark.readthedocs.io))
and **peak physical memory (PSS)** of monoprop's core operations. Memory is read
from the kernel (peak RSS corrected to PSS), capturing the C++/TBB allocations
`tracemalloc` misses at no measurable cost, so timing and memory share one sweep.
It is not part of the normal `pytest` run; invoke it with `just bench`.

## Running

`just bench` runs a single timing + memory sweep and regenerates
`benches/results/REPORT.md`. Driver options (`--ranks`, `--mpiexec-args`,
`--label`) are consumed by `run.py`; everything else is forwarded to pytest.

```bash
just bench                                   # serial (label np1)
just bench-smoke                             # quick sanity check (tiny sizes)
just bench --num-modes 64 --bench-rounds 10  # forwarded to pytest
```

Each run adds a column to the same `REPORT.md`, so an MPI run and serial runs at
different thread counts land side by side — each with timing and memory:

```bash
just bench-build-mpi  # build once (MPI on; also runs serially)
# 5 ranks, each pinned to 2 cores (2 threads/rank, fills a 10-core host)
just bench --ranks 5 --mpiexec-args="--map-by slot:PE=2 --bind-to core" \
    --env monoprop_NUM_THREADS=2 --label mpi-r5t2
# serial, single-threaded
just bench --env monoprop_NUM_THREADS=1 --label serial
# serial, 10 threads
just bench --env monoprop_NUM_THREADS=10 --label serial-t10
```

## Configuring a run (MPI, threads, pinning)

You set configuration on the command line; the driver records it in the report.

| Knob | How | Example |
|---|---|---|
| MPI ranks | `--ranks N` | `--ranks 4` |
| mpiexec args (incl. pinning) | `--mpiexec-args` | `--mpiexec-args="--bind-to core"` |
| monoprop threads | `--env monoprop_NUM_THREADS=K` | `--env monoprop_NUM_THREADS=2` |
| per-process pinning | `--mpiexec-args` (`--bind-to`) | `--mpiexec-args="--bind-to core"` |
| run name / report column | `--label` | `--label r4t2` |

`monoprop` is parallelised with oneTBB and reads only `monoprop_NUM_THREADS` (at
import) — that is the single knob for its thread count. Per-process pinning is
done by `mpiexec` (`--bind-to core`). Use a custom `--label` when varying
threads/pinning at a fixed rank count so runs don't overwrite each other.

`--ranks N` launches `mpiexec` with these defaults; `--mpiexec-args` is appended:

- `--allow-run-as-root` — needed in the container/CI images, which run as root.
- `-n N` — the rank count from `--ranks`.
- `-x <var>` for each `--env` override plus `MONOPROP_BENCH_LABEL` /
  `MONOPROP_BENCH_RESULTS` — forwards them to every rank (OpenMPI).
- no binding/mapping by default — add pinning yourself via `--mpiexec-args`
  (e.g. `--bind-to core --map-by socket`).

Benchmarks are communicator-aware: each operation is barrier-wrapped so the
timed cost is the **makespan** across ranks, rank 0 writes the timing JSON, and
memory is the per-rank peak PSS **summed across ranks** (true physical RAM, with
shared library pages counted once — not a per-node figure). MPI uses fixed-round
`pedantic` timing (`--bench-rounds`, default 1) so ranks don't diverge and
deadlock at the barriers.

**The MPI build must be loaded and stay loaded.** monoprop builds with
`monoprop_ENABLE_MPI=OFF` by default; a non-MPI extension silently ignores the
communicator, so every rank holds the **full** operator (N× memory, OOMs at
scale). `just bench-build-mpi` installs the MPI build and `just bench` runs with
`--no-sync` so it survives — rebuild explicitly after editing monoprop sources.
An `--ranks > 1` run first runs a fast **build preflight** (`_mpi_check.py`,
which just checks `monoprop.has_mpi`) that aborts with remediation if the loaded
extension was not built with MPI; bypass it with `--skip-mpi-check`.

## Reporting

Each run regenerates `REPORT.md`; rebuild it from existing artifacts (no re-run)
with `uv run --group bench python benches/report.py`. Every run is one column, so
serial / MPI / thread variants sit side by side. Sections:

- **Configuration** — ranks, thread counts, launcher, CPU count, host.
- **Hyperparameters** — the resolved random-problem sizes each run used.
- **Operator size** — terms in the evolved operator per picture.
- **Static model configuration** — the resolved config of each static model.
- **Heisenberg** / **Schrödinger** — a **Time** and **Memory (PSS)** table each
  (Schrödinger omitted when unused).

## Benchmarks

All live in `bench_monoprop.py`.

**Random** (configurable, both pictures) — random fixed-length Majorana
generators and a random Hermitian observable, run in the Heisenberg and
Schrödinger pictures (`schrodinger_cutoff = cutoff + 2`), shown as `[heisenberg]`
/ `[schrodinger]`. CLI options (defaults):

| Option | Meaning | Default |
|---|---|---|
| `--gen-length` | Majorana operators per generator | 4 |
| `--obs-terms` | observable terms | 10000 |
| `--num-generators` | generators (circuit gates) | 100 |
| `--num-modes` | fermionic modes (Majorana indices = `2·modes`) | 128 |
| `--cutoff` | truncation cutoff | 6 |
| `--seed` | RNG seed | 0 |

Operations: `build_graph`, `pare`, `energy`, `gradient` (graph-based), and
`inplace` (in-place truncation, no graph stored).

**Static** (fixed, in-place, Heisenberg only, marked `slow`):

- `test_static[hubbard]` — 120-qubit Fermi-Hubbard (60 sites), 29-step Trotter.
- `test_static[pauli]` — 127-qubit Pauli-basis kicked-Ising (IBM Eagle heavy-hex,
  20 layers, ⟨Z₆₂⟩).

`--hubbard-lower-atol VALUE` / `--pauli-lower-atol VALUE` override each model's
truncation tolerance (both default to 1e-4).

## Notes

- The driver passes `-o filterwarnings=default` so benchmark-plugin warnings
  don't fail the run (the project default is `error`).
- Files: `run.py` (driver), `report.py` (report), `conftest.py` (fixtures + CLI
  options), `_builders.py` (model construction), `bench_monoprop.py` (suite).
