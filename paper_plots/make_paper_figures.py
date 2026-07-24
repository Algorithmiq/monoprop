# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the monoprop repository for the full licence text.

"""Publication-quality figures for the single-layer Pauli-propagation scaling study.

Renders three standalone, print-ready figures (vector PDF + PNG twin) from the
merged benchmark JSONL of monoprop's ``PauliPropagator`` and the reference
``PauliPropagation.jl`` on the *same operator* (kicked-Ising, extensive ``Σᵢ Zᵢ``,
``lower_atol = 0`` → identical term set), single-thread and single-shard:

* ``fig1_absolute_scaling`` — time-vs-N and total-memory-vs-N (log-log). The raw
  power laws, with faint N^1/N^2/N^3 slope guides and the fitted exponent over the
  clean pre-cliff window (N=128–512) printed per panel. The two engines' curves
  fan apart: that gap is the divergence, made explicit in Fig 2.
* ``fig2_divergence_scaling`` — THE headline. Julia ÷ monoprop for time and memory
  vs N (log-log), monoprop = 1× baseline. The ratio grows as a power law whose
  exponent is the *divergence exponent* (= difference of the two absolute
  exponents); the region past N≈512, where Julia's packed key outgrows its fast
  BitInteger width, is shaded as the super-linear "key-width cliff".
* ``fig3_per_term_memory`` — the mechanism. bytes/term vs N: monoprop bounded
  (O(cutoff) symplectic support), Julia a staircase as its 2-bits/qubit key widens
  across word boundaries (shaded).

Encoding (accessible by construction): colour = cutoff (Okabe–Ito, CVD-safe, fixed
order), line style/marker = engine (monoprop = solid/filled, Julia = dashed/open).
Identity is therefore never colour-alone. A two-part legend (colour→cutoff,
style→engine) keeps the two dimensions separable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Print styling (no LaTeX dependency; mathtext only, always available)
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # embed TrueType so text stays selectable/editable
    "ps.fonttype": 42,
})

# Okabe–Ito, CVD-safe, assigned to cutoff in fixed order (never cycled).
CUTOFF_COLORS = {2: "#0072B2", 4: "#E69F00", 6: "#009E73", 8: "#CC79A7", 10: "#D55E00"}
ENGINE_STYLE = {"monoprop": ("-", "o"), "julia": ("--", "o")}
ENGINE_LABEL = {"monoprop": "monoprop", "julia": "PauliPropagation.jl"}

# Shared mark spec — a package is drawn identically in EVERY figure (monoprop =
# solid line / filled circle, PauliPropagation.jl = dashed line / open square), so
# one legend key reads the same across all figures. Thin lines, small clean markers.
LINE_WIDTH = 1.4
MARKER_SIZE = 4.0
MARKER_EDGE = 0.9
LEGEND_LW = 1.6
LEGEND_MS = 5.0


def _curve_style(fam, color):
    """The one place a package's line+marker style is defined; used everywhere."""
    ls, marker = ENGINE_STYLE[fam]
    return dict(
        ls=ls, marker=marker, color=color,
        lw=LINE_WIDTH, ms=MARKER_SIZE, markeredgewidth=MARKER_EDGE,
        markeredgecolor=color,
        markerfacecolor=(color if fam == "monoprop" else "white"),
        solid_capstyle="round", dash_capstyle="round", zorder=3,
    )


FIT_NMIN, FIT_NMAX = 128, 512   # clean, pre-cliff power-law window
GUIDE = "#b0b0b0"
GRID = "#d5d5d5"
INK = "#222222"


# --------------------------------------------------------------------------- #
# Data + fits (logic shared with plot_scaling_clear.py)
# --------------------------------------------------------------------------- #
def load(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["engine_family"] = "monoprop" if r["engine"] == "monoprop" else "julia"
            r["memory_mb"] = r["memory_bytes"] / 1024**2
            records.append(r)
    return records


def _series(records, fam, cutoff, metric):
    pts = sorted(
        (r["num_qubits"], r[metric])
        for r in records
        if r["engine_family"] == fam and r["cutoff"] == cutoff
    )
    return [x for x, _ in pts], [y for _, y in pts]


def _fit_exponent(xs, ys, nmin=FIT_NMIN, nmax=FIT_NMAX):
    """Least-squares power-law exponent over nmin <= N <= nmax.

    The window stops at N=512: PauliPropagation.jl's packed key exceeds its fast
    BitInteger width beyond ~512 qubits and falls onto a slow wide-integer path
    (a ~10x time jump at N~600, c6), so a fit spanning the cliff would report an
    implementation artefact rather than the algorithmic exponent.
    """
    lx, ly = [], []
    for x, y in zip(xs, ys):
        if nmin <= x <= nmax and y > 0:
            lx.append(math.log(x))
            ly.append(math.log(y))
    if len(lx) < 2:
        return None
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else None


def _slope_guides(ax, xs_all, exponents, anchor_frac=0.60):
    """Faint reference lines of the given integer slopes on a log-log axis.

    The guides never rescale the axis: the data y-limits are captured up front and
    restored afterwards, so a steep N^3 guide can't blow the view up and squash the
    data. Each label is pinned where its guide exits the top of the frame."""
    lo, hi = min(xs_all), max(xs_all)
    y0, y1 = ax.get_ylim()
    lly0, lly1 = math.log(y0), math.log(y1)
    ya = math.exp(lly0 + anchor_frac * (lly1 - lly0))
    for p in exponents:
        y_end = ya * (hi / lo) ** p
        ax.plot([lo, hi], [ya, y_end], color=GUIDE, lw=0.9, ls=(0, (4, 3)), zorder=0)
        if y_end <= y1:                       # guide stays in frame → label at its end
            lx, ly, dy, va = hi, y_end, 2, "bottom"
        else:                                 # guide leaves the top → label at the exit
            frac = (lly1 - math.log(ya)) / (math.log(y_end) - math.log(ya))
            lx, ly, dy, va = lo * (hi / lo) ** frac, y1, -2, "top"
        ax.annotate(f"$N^{{{p}}}$", xy=(lx, ly), xytext=(-2, dy),
                    textcoords="offset points", fontsize=8, color="#8a8a8a",
                    ha="right", va=va)
    ax.set_ylim(y0, y1)


def _two_part_legend(fig, cutoffs, *, engines=("monoprop", "julia"),
                     y_cut=0.075, y_eng=0.015):
    """Two stacked, centred legend rows (width-independent, no collision):
    colour→cutoff above (coloured entries), style→engine below (grey entries).
    Titles are omitted — the entries are self-explanatory and titles would crowd
    the band; the mapping is stated once in each figure's caption/README."""
    cut_handles = [
        Line2D([0], [0], color=CUTOFF_COLORS.get(c, "#666"), lw=LEGEND_LW,
               marker="o", ms=LEGEND_MS, markeredgewidth=MARKER_EDGE,
               markeredgecolor=CUTOFF_COLORS.get(c, "#666"), label=f"cutoff {c}")
        for c in cutoffs
    ]
    eng_handles = [
        Line2D([0], [0], color="#555555", lw=LEGEND_LW, ls=ENGINE_STYLE[e][0],
               marker=ENGINE_STYLE[e][1], ms=LEGEND_MS, markeredgewidth=MARKER_EDGE,
               markeredgecolor="#555555",
               markerfacecolor="#555555" if e == "monoprop" else "white",
               label=ENGINE_LABEL[e])
        for e in engines
    ]
    fig.legend(handles=cut_handles, loc="lower center",
               bbox_to_anchor=(0.5, y_cut), ncol=len(cutoffs), frameon=False,
               handlelength=2.2, columnspacing=2.2)
    fig.legend(handles=eng_handles, loc="lower center",
               bbox_to_anchor=(0.5, y_eng), ncol=len(engines), frameon=False,
               handlelength=2.6, columnspacing=2.2)


def _plot_curves(ax, records, cutoffs, metric):
    """Plot both engines × all cutoffs for a metric; return fitted exponents."""
    fits = {}
    for cutoff in cutoffs:
        color = CUTOFF_COLORS.get(cutoff, "#666666")
        for fam in ("monoprop", "julia"):
            xs, ys = _series(records, fam, cutoff, metric)
            if not xs:
                continue
            ax.plot(xs, ys, **_curve_style(fam, color))
            p = _fit_exponent(xs, ys)
            if p is not None:
                fits[(cutoff, fam)] = p
    return fits


def _plot_ratio(ax, records, cutoffs, metric):
    """Julia / monoprop at each shared N, per cutoff, plus the =1x baseline.

    A ratio is a derived quantity (neither package), so it borrows the "monoprop"
    solid/filled key for a clean single curve per cutoff.
    """
    for cutoff in cutoffs:
        color = CUTOFF_COLORS.get(cutoff, "#666666")
        mono = {r["num_qubits"]: r[metric] for r in records
                if r["engine_family"] == "monoprop" and r["cutoff"] == cutoff}
        jul = {r["num_qubits"]: r[metric] for r in records
               if r["engine_family"] == "julia" and r["cutoff"] == cutoff}
        xs = sorted(set(mono) & set(jul))
        ys = [jul[x] / mono[x] for x in xs if mono[x]]
        if not xs:
            continue
        ax.plot(xs, ys, **_curve_style("monoprop", color))
    ax.axhline(1.0, color="#555555", lw=1.4, zorder=2)
    ax.annotate("monoprop $=1\\times$", xy=(0.985, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8.3, color="#555555", fontweight="bold")


def _finish_axis(ax, xlabel, ylabel, title, logy=True, base2=True):
    if base2:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, which="both", ls=":", lw=0.5, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("pdf", "png"):
        out = outdir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        outs.append(out)
    plt.close(fig)
    return outs


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig1_absolute_scaling(records, layers, outdir):
    cutoffs = sorted({r["cutoff"] for r in records})
    xs_all = sorted({r["num_qubits"] for r in records})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    panels = [
        ("seconds", f"time  (s, {layers} layers)"),
        ("memory_mb", "memory  (MB)"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        _plot_curves(ax, records, cutoffs, metric)
        _finish_axis(ax, "number of qubits  $N$", ylabel, "")
        _slope_guides(ax, xs_all, [1, 2, 3])
    fig.tight_layout(rect=(0, 0.17, 1, 0.99))
    _two_part_legend(fig, cutoffs, y_cut=0.085, y_eng=0.02)
    return _save(fig, outdir, "fig1_absolute_scaling")


def fig2_divergence_scaling(records, layers, outdir):
    """Julia ÷ monoprop: the divergence, and how it scales."""
    cutoffs = sorted({r["cutoff"] for r in records})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    panels = [("seconds", "time overhead  ($\\times$ monoprop)"),
              ("memory_mb", "memory overhead  ($\\times$ monoprop)")]
    for ax, (metric, ylabel) in zip(axes, panels):
        _plot_ratio(ax, records, cutoffs, metric)
        _finish_axis(ax, "number of qubits  $N$", ylabel, "")
    fig.tight_layout(rect=(0, 0.10, 1, 0.99))
    # cutoff-only legend (all curves are ratios → one style), centred below.
    cut_handles = [Line2D([0], [0], color=CUTOFF_COLORS.get(c, "#666"), lw=LEGEND_LW,
                          marker="o", ms=LEGEND_MS, markeredgewidth=MARKER_EDGE,
                          markeredgecolor=CUTOFF_COLORS.get(c, "#666"),
                          label=f"cutoff {c}") for c in cutoffs]
    fig.legend(handles=cut_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(cutoffs), frameon=False,
               handlelength=2.2, columnspacing=2.2)
    return _save(fig, outdir, "fig2_divergence_scaling")


def fig3_per_term_memory(records, outdir):
    cutoffs = sorted({r["cutoff"] for r in records})
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    xlo, xhi = 28, 1120
    # Julia packed-key width: 2 bits/qubit → word widens at N = 32,64,128,256,512.
    regimes = [(28, 32, "64-bit"), (32, 64, "128-bit"), (64, 128, "256-bit"),
               (128, 256, "512-bit"), (256, 512, "1024-bit"), (512, xhi, "2048-bit")]
    for i, (a, b, lbl) in enumerate(regimes):
        if i % 2 == 0:
            ax.axvspan(a, b, color="#eef0f2", zorder=0)
        ax.annotate(lbl, xy=(math.sqrt(a * b), 0.015),
                    xycoords=("data", "axes fraction"), fontsize=7.2,
                    color="#999999", ha="center", va="bottom")
    for cutoff in cutoffs:
        color = CUTOFF_COLORS.get(cutoff, "#666666")
        for fam in ("monoprop", "julia"):
            xs, ys = _series(records, fam, cutoff, "bytes_per_term")
            if not xs:
                continue
            ax.plot(xs, ys, **_curve_style(fam, color))
    ax.set_xscale("log", base=2)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("number of qubits  $N$")
    ax.set_ylabel("memory per term  (bytes)")
    ax.grid(True, which="both", ls=":", lw=0.5, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.17, 1, 0.98))
    _two_part_legend(fig, cutoffs, y_cut=0.085, y_eng=0.02)
    return _save(fig, outdir, "fig3_per_term_memory")


def fig4_scaling_and_divergence(records, layers, outdir):
    """2×2 merge of Fig 1 + Fig 2: absolute scaling (top row) sitting directly above
    its divergence ratio (bottom row); columns are time (left) and memory (right),
    so each column reads top-to-bottom as "absolute cost, then its overhead", on a
    shared N axis."""
    cutoffs = sorted({r["cutoff"] for r in records})
    xs_all = sorted({r["num_qubits"] for r in records})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.8), sharex=True)

    abs_panels = [("seconds", f"time  (s, {layers} layers)"),
                  ("memory_mb", "memory  (MB)")]
    for ax, (metric, ylabel) in zip(axes[0], abs_panels):
        _plot_curves(ax, records, cutoffs, metric)
        _finish_axis(ax, "", ylabel, "")

    rat_panels = [("seconds", "time overhead  ($\\times$ monoprop)"),
                  ("memory_mb", "memory overhead  ($\\times$ monoprop)")]
    for ax, (metric, ylabel) in zip(axes[1], rat_panels):
        _plot_ratio(ax, records, cutoffs, metric)
        _finish_axis(ax, "number of qubits  $N$", ylabel, "")
        _slope_guides(ax, xs_all, [1, 2, 3])

    fig.tight_layout(rect=(0, 0.09, 1, 0.99))
    _two_part_legend(fig, cutoffs, y_cut=0.055, y_eng=0.012)
    return _save(fig, outdir, "fig4_scaling_and_divergence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="JSONL result file(s)")
    parser.add_argument("--outdir", default="figures", type=Path)
    args = parser.parse_args()
    records = load([Path(p) for p in args.results])
    if not records:
        raise SystemExit("no records loaded")
    layers = records[0].get("layers", 1)

    outputs = []
    outputs += fig1_absolute_scaling(records, layers, args.outdir)
    outputs += fig2_divergence_scaling(records, layers, args.outdir)
    outputs += fig3_per_term_memory(records, args.outdir)
    outputs += fig4_scaling_and_divergence(records, layers, args.outdir)

    print("wrote:")
    for o in outputs:
        print(f"  {o}")

    # Verification readout: fitted exponents to stdout.
    cutoffs = sorted({r["cutoff"] for r in records})
    print("\nfitted exponents (N=128–512):")
    for metric, name in (("seconds", "time"), ("memory_mb", "memory")):
        for c in cutoffs:
            mx, my = _series(records, "monoprop", c, metric)
            jx, jy = _series(records, "julia", c, metric)
            pm, pj = _fit_exponent(mx, my), _fit_exponent(jx, jy)
            mono = {x: y for x, y in zip(mx, my)}
            jul = {x: y for x, y in zip(jx, jy)}
            xs = sorted(set(mono) & set(jul))
            pr = _fit_exponent(xs, [jul[x] / mono[x] for x in xs])
            if pm and pj:
                print(f"  {name} c{c}: mono N^{pm:.2f}  julia N^{pj:.2f}  "
                      f"ratio N^{pr:.2f}  (check: {pj - pm:.2f})")


if __name__ == "__main__":
    main()
