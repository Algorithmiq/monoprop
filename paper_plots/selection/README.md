# Paper figure — cost of one gate selection

Self-contained figure package for the gate-selection cost study: what it costs the engine to
**choose** one gate, as a function of the mode count `N`.

Selection time is the three phases timed inside the engine — pair enumeration (state support ×
candidate), building the occupation-flip table over the diagonal monomials, and scoring. Building
and evolving the observable happen *before* the clock starts, so none of the propagation cost is
in these curves.

## Result

> Every number below is printed by `make_selection_figure.py` from `data/selection_scaling.jsonl`,
> and repeated in `figures/captions.txt`, which that script computes from the plotted rows. If
> this section and that file ever disagree, that file is right.

**Selecting a gate costs what enumerating the candidate pairs costs, and nothing on top of it.**
For a generator of length `ell` applied at order `p` the pair count is bounded by `N^(ell+p)`, and
the measured time lands on that bound in all three cases:

| case | cutoff | N range | rungs | time fit | pair-count fit | bound | ns per pair |
|------|--------|---------|-------|----------|----------------|-------|-------------|
| `ell=2, p=1` | 4 | 8–192 | 11 | `N^3.070` | `N^3.015` | `N^3` | 9.1–16.2 |
| `ell=2, p=2` | 4 | 8–160 | 10 | `N^4.217` | `N^4.055` | `N^4` | 11.9–16.3 |
| `ell=3, p=2` | 6 | 8–80  | 8  | `N^5.186` | `N^5.187` | `N^5` | 10.8–15.5 |

Fits are least-squares log-log slopes over the **4 widest rungs** of each case (`FIT_POINTS`).
Time and pair count are fitted over the same window, so the gap between the two columns — at most
0.16, and zero for `ell=3, p=2` — is per-pair cost drifting, not extra work: the cost of
enumerating one pair stays inside 9–16 ns across the whole sweep, seven orders of magnitude of
pair count.

The phase split says where the time is not. The flip table never exceeds 21 µs anywhere in the
sweep (it is built over the *diagonal* monomials, of which there are 41–199, not over `K`), and
scoring falls from as much as 45% of the time at the narrow end to under 2% at the wide
end (0.1% for `ell=3, p=2`). Everything
that grows is enumeration.

## Method

- **Basis:** propagated, single partition, **single thread** — a serial, like-for-like sweep.
- **Cases:** generator length `ell` ∈ {2, 3} at order `p` ∈ {1, 2}, at weight cutoff 4 (`ell=2`)
  and 6 (`ell=3`).
- **Ladder:** `N = 8 … 192`, each case stopping where a rung stops being affordable — the
  `ell=3, p=2` case ends at N=80, where one selection already takes 197 s over 1.6e10 pairs.
- **Recorded per rung:** tracked monomials `K`, enumerated pairs, hits, pool size, diagonal
  monomials, table entries, the three phase times, total and wall seconds, and the observable
  build time (excluded from the plotted quantity).

## Layout

```
make_selection_figure.py   the figure and figures/captions.txt, from the JSONL and nothing else
build.sh                   `PY=<python> ./build.sh` — matplotlib, no monoprop build, no engine
data/selection_scaling.jsonl   one record per rung, as emitted by the sweep
data/UPSTREAM-REPORT.md    the original write-up, with the full per-rung tables
figures/                   fig-selection-scaling.{pdf,png} + captions.txt
```

Output is byte-reproducible (the PDF creation stamp is suppressed), so a regenerated figure that
differs from the shipped one means the *data* changed, not the clock.

## Provenance limits of the shipped data

Worth knowing before citing a number from `data/selection_scaling.jsonl`:

- **No provenance fields.** The records carry no host, no monoprop version and no build hash, so
  they cannot be audited the way `node_scaling/` gates every rep on the built `_core.so` md5. They
  were produced on **2026-08-25** and lived in `monoprop`'s gitignored `benches/results/`, whose
  sweep driver is not in either repository — the data here is the only surviving artefact, and a
  new ladder would have to be re-derived rather than re-run.
- **Repeats are not uniform.** `repeats = 3` at `N ≤ 24` and `repeats = 1` above it, so every rung
  the fits are actually taken over is a **single-shot timing**. The fitted exponents are stable to
  the third digit against the report computed independently in August, but there is no spread to
  quote per point.
- **Unequal cutoffs.** The `ell=3` case is measured at cutoff 6, the `ell=2` cases at 4, so the
  three curves' vertical offsets mix generator order with cutoff. The claim is about each curve's
  *slope* against its own bound, which is unaffected; do not read the gaps between curves as an
  `ell` effect.
