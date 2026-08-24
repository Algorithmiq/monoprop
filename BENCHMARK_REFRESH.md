# Benchmark refresh checklist (Leonardo)

Working notes for the `docs/benchmarks-refresh` branch. **Delete this file before merging.**

The prose in `docs/content/docs/benchmarks.mdx` is written against the *currently committed*
data. Every numeral below must be re-checked once the Leonardo run lands; the sentences are
phrased so that only the numbers change, not the structure.

## Why a re-run is required

1. **The committed scaling data predates the current schema.** `scaling_results.jsonl` was
   carried into `1b7533f` from an earlier run: its `memory_metric` fields still say
   `"operator memory (reported by the library)"`, `"Base.summarysize of the Pauli sum"`, etc.,
   while the current `backends.HOST_MEMORY_METRIC` is
   `"peak process RSS over the step (kernel VmHWM)"`. `final_memory_MB` in that file is
   therefore each library's *own* accounting, but `plot_scaling.py` labels that axis
   "final-step peak host RSS". **The published `pauli_scaling_memory.png` is mislabelled.**
   `check_memory_metric()` (added on this branch) now catches exactly this and goes silent once
   the data is regenerated.

2. **Memory-relevant commits landed after the data.** `638ee6f` (release `init_op_map` buckets)
   and `cb9a033` (halve the follower-marking epoch stamp) both change the footprint, and the
   PNGs are older still (`bb3d28f`, 2026-08-13).

3. **The per-step `results.json` has no provenance**, so its hardware claim was unverifiable.
   `run_model.py` now stamps a `provenance` block (host, settings, memory metric, per-engine
   thread counts); the re-run is what populates it.

## Commands

```bash
cd packages/bench-third-party
uv sync

# Pauli, fixed lattice (12x12 per settings.json)
cd pauli_prop
uv run python run_model.py
julia run_model.jl                     # needs PauliPropagation.jl v0.7.3
uv run python plot_results.py

# Pauli, lattice ladder 6x6 -> 18x18
uv run python run_scaling.py --output scaling_results.jsonl \
    --timeout 300 --threads monoprop=56 juliapp=28
uv run python plot_scaling.py scaling_results.jsonl     # must print NO warning now
uv run python plot_speedup.py scaling_results.jsonl

# Majorana, 1D Hubbard
cd ../majorana_prop
monoprop_NUM_THREADS=56 JULIA_NUM_THREADS=28 bash run_benchmarks.sh
```

Build/env gotchas:

- 324 qubits needs `-Dmonoprop_MAX_NUM_MODES=352` (tiers come in 32-mode blocks).
- The ladder needs `PauliPropagation.jl` **dev (0.8.0+)** for `Performance`; the fixed-lattice
  `run_model.jl` wants the pinned **v0.7.3**. Two Julia environments.
- Confirm `plot_scaling.py` prints no `WARNING:` line. If it still does, the run did not pick
  up the current `backends.py`.

## Copy the figures over

```bash
cp packages/bench-third-party/pauli_prop/pauli_results.png \
   packages/bench-third-party/pauli_prop/pauli_scaling_runtime.png \
   packages/bench-third-party/pauli_prop/pauli_scaling_memory.png \
   packages/bench-third-party/pauli_prop/pauli_speedup.png \
   packages/bench-third-party/majorana_prop/majorana_results.png \
   packages/bench-third-party/majorana_prop/majorana_runtime.png \
   docs/public/benchmarks/
```

## Numbers in benchmarks.mdx to re-derive

Section "Increasing lattice size" and its table — all from `peak_rss_MB` and
`total_runtime_s` in the new `scaling_results.jsonl`:

| claim | current value (committed data) |
| --- | --- |
| term saturation | ~5.84M terms past ~100 qubits |
| monoprop RSS, 36 -> 324 qubits | 871 MB -> 1134 MB (1.3x) |
| PauliPropagation.jl RSS, 36 -> 324 | 1199 MB -> 4185 MB (3.5x) |
| RSS ratio range vs monoprop | 1.4x -> 3.7x |
| speed-up row, 36/100/196/324 | 16/33/55/105x, 107/190/184/-x, 165/287/-/-x, 4.4/8.5/11/15x |
| RSS-ratio row, 36/100/196/324 | 1.4/2.0/2.6/3.7x, 0.6/0.7/0.8/-x, 2.6/5.5/-/-x |
| ppvm leaner than monoprop | 0.6-0.8x RSS, 107-209x slower |
| cuPauliProp host RSS (Callout) | flat near 534 MB |

Section "Fixed lattice, increasing depth": confirm the lattice is still **12x12 (144 qubits),
28 Trotter points, dt=0.05, atol=1e-6, no weight cutoff** — read it from the new `provenance`
block rather than from `settings.json`, in case settings are edited before the run.

Section "Majorana propagation" — all from `majorana_prop/results.json`:

| claim | current value (committed data) |
| --- | --- |
| system / depth / threads | 60 spinful sites, 20 layers, **24 threads both engines** |
| expectation-value agreement | within 2.0e-6 at every layer |
| relative term count | monoprop keeps 1.38x more |
| total runtime | 92.5 s vs 2908.5 s (31x) |
| final-layer runtime | 37.6 s vs 1082.0 s (29x) |
| peak RSS | 10.6 GB vs 15.7 GB (1.48x) |

**Note:** the committed Majorana run used 24 threads for both engines. If the re-run uses
56/28 as the Pauli sweeps do, the thread sentence in that section must change too, and the
speed-up will move.

## Also worth knowing

The old page documented a Majorana harness that no longer exists: it described two JSONL files
(`julia_hubbard1d_benchmark_results.jsonl`, `monoprop_hubbard1d_benchmark_results.jsonl`), a
3-size grid (n=20/40/60), and fields (`busy_cores`, `cpu_seconds`, `mu_gates`,
`affinity_cores`, `final_overlap`) that the current scripts do not emit. The scripts write a
single `results.json` for one system size. Those claims — 22-142x, 4.2 s vs 592 s,
2834 MB vs 3390 MB, 1.44-1.50x more terms — were not reproducible from anything in the repo
and have been replaced with values derived from the committed data.
