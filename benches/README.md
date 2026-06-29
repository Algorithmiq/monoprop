# monoprop benchmarks

A pytest suite measuring the **time** ([`pytest-benchmark`](https://pytest-benchmark.readthedocs.io))
and **peak memory** ([`pytest-memray`](https://pytest-memray.readthedocs.io),
which tracks the C++-side allocations `tracemalloc` misses) of monoprop's core
operations. It is not part of the normal `pytest` run; invoke it with `just bench`.

## Running

`just bench` runs the timing pass, the memory pass, and regenerates
`benches/results/REPORT.md`. Driver options (`--ranks`, `--mpiexec-args`,
`--no-mem`) are consumed by `run.py`; everything else is forwarded to pytest.

```bash
just bench                                   # serial (label np1)
just bench-smoke                             # quick check: tiny sizes, no memory pass
just bench --num-modes 64 --bench-rounds 10  # forwarded to pytest
just bench --no-mem                          # timing only
```

## Configuring a run (MPI, threads, pinning)

You set configuration on the command line; the driver records it in the report.

| Knob | How | Example |
|---|---|---|
| MPI ranks | `--ranks N` | `--ranks 4` |
| mpiexec args (incl. pinning) | `--mpiexec-args` | `--mpiexec-args="--bind-to core"` |
| threads & pinning | `--env KEY=VAL` (repeatable) | `--env monoprop_NUM_THREADS=2` |
| run name / report column | `--label` | `--label r4t2` |

`monoprop` reads `monoprop_NUM_THREADS` at import; add `OMP_NUM_THREADS` /
`OMP_PROC_BIND` / `OMP_PLACES` for OpenMP pinning. Use a custom `--label` when
varying threads/pinning at a fixed rank count so runs don't overwrite each other.

```bash
just bench-build-mpi                                       # one-time MPI rebuild
just bench --ranks 4 --mpiexec-args="--bind-to core" \
    --env monoprop_NUM_THREADS=2 --env OMP_PROC_BIND=close --label r4t2
```

Benchmarks are communicator-aware: each operation is barrier-wrapped so the
timed cost is the **makespan** across ranks, rank 0 writes the timing JSON, and
memory is the worst-rank peak. MPI uses fixed-round `pedantic` timing
(`--bench-rounds`, default 5) so ranks don't diverge and deadlock at the barriers.

**The MPI build must be loaded and stay loaded.** monoprop builds with
`monoprop_ENABLE_MPI=OFF` by default; a non-MPI extension silently ignores the
communicator, so every rank holds the **full** operator (N× memory, OOMs at
scale). `just bench-build-mpi` installs the MPI build and `just bench` runs with
`--no-sync` so it survives — rebuild explicitly after editing monoprop sources.
An `--ranks > 1` run first runs a fast **distribution preflight**
(`_mpi_check.py`) that aborts with remediation if the build is not distributing;
bypass it with `--skip-mpi-check`.

## Reporting

Each run regenerates `REPORT.md`; rebuild it from existing artifacts (no re-run)
with `uv run --group bench python benches/report.py`. Every run is one column, so
serial / MPI / thread variants sit side by side. Sections:

- **Configuration** — ranks, thread counts, launcher, CPU count, host.
- **Hyperparameters** — the resolved random-problem sizes each run used.
- **Graph size** — terms in the evolved operator per picture.
- **Static model configuration** — the resolved config of each static model.
- **Heisenberg** / **Schrödinger** — a **Time** and **Peak memory** table each
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

`--lower-atol VALUE` overrides their truncation tolerance (defaults: Hubbard
1e-5, Pauli 1e-4).

## Notes

- The driver passes `-o filterwarnings=default` so benchmark-plugin warnings
  don't fail the run (the project default is `error`).
- Files: `run.py` (driver), `report.py` (report), `conftest.py` (fixtures + CLI
  options), `_builders.py` (model construction), `bench_monoprop.py` (suite).
