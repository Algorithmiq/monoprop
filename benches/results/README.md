# Benchmark results

This directory holds **locally generated** benchmark artifacts: `REPORT.md`, the
pytest-benchmark timings (`time-<label>.json`), and everything else per run
(`<label>.json` — metadata, hyperparameters, peak memory, operator sizes and
footprints, model configs). See [`../README.md`](../README.md) for how they are
produced and combined.

These files should **NOT** be added to the repo — only this `README.md` is
tracked (see the `benches/results/**` rule in the top-level `.gitignore`).
