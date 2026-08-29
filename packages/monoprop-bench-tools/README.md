# monoprop-bench-tools

The benchmark harness behind [monoprop](https://github.com/Algorithmiq/monoprop):
the pieces that are reusable outside a single benchmark run, packaged separately
so scripts, notebooks and third-party comparisons can depend on them without
vendoring the repository's `benches/` directory.

| Module | What it gives you |
| --- | --- |
| `monoprop_bench_tools.memory.cpu` | Exact peak-RSS measurement from the kernel's `VmHWM`. |
| `monoprop_bench_tools.memory.gpu` | The device-side counterpart, over CUDA memory-pool counters or a CuPy allocation hook. |
| `monoprop_bench_tools.models` | Builders for the benchmarked problems: a configurable random problem, a 120-mode Fermi-Hubbard model, and a 127-qubit kicked-Ising model. |
| `monoprop_bench_tools.report` | Renders a run's artifacts into a side-by-side Markdown report. |
| `monoprop_bench_tools.bmf` | Renders a run's artifacts into [Bencher](https://bencher.dev/) Metric Format JSON. |
| `monoprop_bench_tools.rungs` | Reads the rung table, runs one rung, gates the result on its term count, and collates a ladder. |

## Install

```bash
pip install monoprop-bench-tools
```

### GPU support

`monoprop_bench_tools.memory.gpu` needs [CuPy](https://cupy.dev/), and only
CuPy. Pick the extra matching your CUDA toolkit:

```bash
pip install "monoprop-bench-tools[gpu-cuda12]"   # CUDA 12.x
pip install "monoprop-bench-tools[gpu-cuda13]"   # CUDA 13.x
```

Pick one, not both: the two CuPy builds ship the same top-level `cupy` package.
On a host with no system CUDA toolkit, add CuPy's bundled one with
`pip install "cupy-cuda12x[ctk]"`.

Without CuPy the module still imports, and every reading reports
`Method.UNAVAILABLE` instead of a figure you might mistake for a real one.

## Use

Measure the peak resident footprint of a block, exactly (no sampling):

```python
from monoprop_bench_tools.memory.cpu import HighWaterMark
from monoprop_bench_tools.models import build_hubbard_problem

with HighWaterMark() as hwm:
    propagator, circuit = build_hubbard_problem()
    propagator.build_graph(circuit)

print(f"{hwm.peak_mb:.0f} MB peak, exact={hwm.exact}")
```

Render the artifacts of a run:

```bash
monoprop-bench-report benches/results             # writes REPORT.md
monoprop-bench-bmf benches/results ci-linux       # BMF JSON on stdout
monoprop-bench-rung benches/rungs.toml list      # the benchmark set
monoprop-bench-ladder benches/rungs.toml runs/   # scaling tables from rung artifacts
```

Both read the two files a run leaves in the results directory:
`time-<label>.json` (pytest-benchmark timings) and `<label>.json` (memory,
operator sizes, configuration). The schema is documented in the repository's
`benches/results/README.md`, and written by the suite in `benches/conftest.py`.

## Scope

This package deliberately holds no benchmarks, and no rung table. monoprop's own
suite and its `rungs.toml` live in the repository's `benches/` directory, because
benchmark names are the key Bencher's history is stored under and a rung id names
the artifacts a campaign has already written -- neither may move with a library
release. What lives here is the machinery that reads them.

## License

Apache-2.0. See [LICENSE](LICENSE).
