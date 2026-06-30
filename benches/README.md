# monoprop benchmarks

A pytest suite measuring the **time** ([`pytest-benchmark`](https://pytest-benchmark.readthedocs.io))
and **peak physical memory (PSS)** of monoprop's core operations. Memory is read
from the kernel (peak RSS corrected to PSS), capturing the C++/TBB allocations
`tracemalloc` misses at no measurable cost, so timing and memory share one sweep.
It is not part of the normal `pytest` run; invoke it with `just bench`.

## Running

Each run takes a **label** and becomes one column in `results/REPORT.md`, so
serial, MPI, and thread variants sit side by side. Extra arguments are forwarded
to pytest.

```bash
just bench serial                            # serial run, column "serial"
just bench-smoke                             # quick sanity check (tiny sizes)
just bench serial --num-modes 64 --bench-rounds 10
```

Set the monoprop thread count with the `monoprop_NUM_THREADS` env var (oneTBB
worker count — monoprop's only thread knob, read at import):

```bash
monoprop_NUM_THREADS=1  just bench serial      # single-threaded
monoprop_NUM_THREADS=10 just bench serial-t10  # 10 threads
```

Rebuild the report from existing artifacts without re-running:

```bash
uv run --group bench python benches/report.py
```

## MPI

The benchmarks are communicator-aware: each operation is barrier-wrapped so the
timed cost is the **makespan** across ranks, rank 0 writes the results, and
memory is the per-rank peak PSS **summed across ranks** (true physical RAM, with
shared library pages counted once).

```bash
just bench-build-mpi                         # build once (MPI on; also runs serially)
# 5 ranks, 2 cores each, 2 threads/rank (fills a 10-core host)
monoprop_NUM_THREADS=2 just bench-mpi r5t2 5 --map-by slot:PE=2 --bind-to core
```

**The MPI build must be loaded and stay loaded.** monoprop builds with
`monoprop_ENABLE_MPI=OFF` by default; a non-MPI extension silently ignores the
communicator, so every rank holds the **full** operator (N× memory, OOMs at
scale). `just bench-build-mpi` installs the MPI build, and the bench recipes use
`--no-sync` so it survives — rebuild explicitly after editing monoprop sources.
`just bench-mpi` runs a fast preflight (`_mpi_check.py`, checking
`monoprop.has_mpi`) that aborts with remediation if the loaded extension lacks MPI.

## Benchmarks

All live in `bench_monoprop.py`.

**Random** (configurable, both pictures) — random fixed-length Majorana
generators and a random Hermitian observable, run in the Heisenberg and
Schrödinger pictures (`schrodinger_cutoff = cutoff + 2`). Operations:
`build_graph`, `pare`, `energy`, `gradient` (graph-based), and `inplace`
(in-place truncation, no graph stored). CLI options (defaults):

| Option | Meaning | Default |
|---|---|---|
| `--gen-length` | Majorana operators per generator | 4 |
| `--obs-terms` | observable terms | 10000 |
| `--num-generators` | generators (circuit gates) | 100 |
| `--num-modes` | fermionic modes (Majorana indices = `2·modes`) | 128 |
| `--cutoff` | truncation cutoff | 6 |
| `--seed` | RNG seed | 0 |
| `--bench-rounds` | fixed timing rounds (MPI-safe) | 1 |

**Static** (fixed, in-place, Heisenberg only, marked `slow`):

- `test_static[hubbard]` — 120-qubit Fermi-Hubbard (60 sites), 29-step Trotter.
- `test_static[pauli]` — 127-qubit Pauli-basis kicked-Ising (IBM Eagle heavy-hex,
  20 layers, ⟨Z₆₂⟩).

Every field of `HubbardConfig` / `KickedIsingConfig` is overridable via a
`--<model>-<field>` option that defaults to the dataclass's own default — e.g.
`--pauli-num-layers 30`, `--hubbard-trotter-steps 10`. Run `just bench --help`
for the full list.

## The report

`results/REPORT.md` is regenerated after each run with one column per label:
Configuration, Hyperparameters, Operator size, resting footprint and
operator-vs-graph storage breakdown, static-model configs, and a **Time** and
**Memory (PSS)** table per picture. Each run writes two artifacts to `results/`:
`time-<label>.json` (pytest-benchmark) and `<label>.json` (everything else);
`report.py` merges all labels found there.

## Files

`report.py` (report), `conftest.py` (fixtures, CLI options, memory + result
recording), `_builders.py` (model construction), `bench_monoprop.py` (suite),
`_mpi_check.py` (MPI preflight).
