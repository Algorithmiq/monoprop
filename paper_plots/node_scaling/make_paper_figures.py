#!/usr/bin/env python3
"""Four minimal single-panel scaling figures, conventional HPC style.

Reads data/SCALE-CELLS.tsv and nothing else -- no cluster access, no run directories. That is the
check that the plots and the data cannot drift apart. Writes the four figures and, from
captions.py, figures/captions.txt, so a caption cannot be edited apart from the figure it
describes.

No panel titles, no (a)/(b) lettering, no secondary axes, no in-plot annotations. One quantity
per figure, drawn at single-column width so the type is at true size in the paper.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,  # noqa: E402
                              NullLocator)

import captions  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data" / "SCALE-CELLS.tsv"
FIGS = ROOT / "figures"

INK, INK2, GRID, GUIDE = "#0b0b0b", "#52514e", "#dedcd6", "#9c9a94"

# One-hue light->dark ramp: the three curves differ in problem SIZE or LOAD, which is a magnitude,
# so colour encodes order rather than identity. Validated with validate_palette.js --ordinal
# against a white surface (monotone L, adjacent dL >= 0.06, light end 2.11:1, hue spread 4 deg).
RAMP3 = ["#86b6ef", "#2a78d6", "#0d366b"]
# A distinct marker per curve as well, so identity survives greyscale print and does not rest on
# colour alone.
MARKS = ["o", "s", "^"]

W, H = 3.4, 2.95                  # single column, drawn at final size
# The 2x2 composite is a page-wide top-of-page float: 7.0 in is the usual two-column textwidth,
# and 4.9 in keeps it to roughly the top half of the page so body text still follows it.
W2, H2 = 7.0, 5.2
CORES_PER_NODE = 128

# STIXGeneral is Times-metric, so the figure text matches the body text of a Times-set paper
# instead of announcing itself as a default matplotlib plot in DejaVu Sans. mathtext follows it.
# ONE tick size for both major and minor: two sizes down a single axis reads as a mistake, and
# the log minor labels (500, 200, 50, 20) are ordinary axis numbers, not annotations.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": "#c9c7c0",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

WEAK = ["weak_97m", "weak_377m", "weak_1569m"]
STRONG = ["strong_s4", "strong_s6", "strong_s8"]


def load(path=DATA):
    """One rung per (ladder, nodes): median seconds over the gate-clean reps. A rung with no
    clean rep is absent, not zero."""
    rungs: dict[tuple[str, int], dict] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["gate"] != "ok":
                continue
            key = (r["ladder"], int(r["nodes"]))
            e = rungs.setdefault(key, {"secs": [], "terms": int(r["terms"]),
                                       "nodes": int(r["nodes"])})
            e["secs"].append(float(r["seconds"]))
    for e in rungs.values():
        e["median"] = statistics.median(e["secs"])
        e["reps"] = len(e["secs"])
        e["cores"] = e["nodes"] * CORES_PER_NODE
    return rungs


def curve(rungs, ladder):
    return [rungs[k] for k in sorted(rungs, key=lambda k: k[1]) if k[0] == ladder]


def fmt_terms(n):
    return f"{n / 1e9:.2f}e9" if n >= 1e9 else f"{n / 1e6:.0f}M"


def plain_log(axis):
    """Plain numbers on both tick levels. The MAJOR formatter defaults to scientific notation, so
    setting only the minor one prints "500 200 10^2 50 20 10^1" down a single axis."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axis.set_major_formatter(fmt)
    axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=20))
    axis.set_minor_formatter(fmt)


def nodes_axis(ax, cores, label=True):
    """A second x-axis on top reading in NODES. Cores are what the machine is booked in, nodes are
    what the reader reasons about, and the two differ by exactly CORES_PER_NODE -- so this is a
    relabelling of the same axis, never a second scale carrying a second quantity."""
    top = ax.secondary_xaxis("top", functions=(lambda c: c / CORES_PER_NODE,
                                              lambda n: n * CORES_PER_NODE))
    top.xaxis.set_major_locator(FixedLocator([c / CORES_PER_NODE for c in cores]))
    top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    top.xaxis.set_minor_locator(NullLocator())
    if label:
        top.set_xlabel("Nodes", labelpad=3)
    top.tick_params(colors=INK2, labelsize=plt.rcParams["xtick.labelsize"], pad=2)
    top.spines["top"].set_color(plt.rcParams["axes.edgecolor"])
    top.spines["top"].set_linewidth(plt.rcParams["axes.linewidth"])
    return top


def dress(ax, cores, logy=True, xlabel=True):
    """Cores on a log2 x-axis, light major grid, no title and no right/top furniture of its own
    (the nodes axis supplies the top spine)."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(cores))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlim(min(cores) / 1.3, max(cores) * 1.3)
    if xlabel:
        ax.set_xlabel("Cores")
    if logy:
        ax.set_yscale("log")
        plain_log(ax.yaxis)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def panel(cores, logy=True):
    """One standalone single-column figure, with its own nodes axis on top."""
    fig, ax = plt.subplots(figsize=(W, H))
    dress(ax, cores, logy=logy)
    nodes_axis(ax, cores)
    return fig, ax


def clip_y(ax, curves, key, lo=0.72, hi=1.4):
    """Limit the y-axis to the DATA. The ideal 1/N guides run an order of magnitude below the
    measured points at the wide end, and letting them set the limit wastes half the panel; they
    simply leave the frame instead."""
    vals = [key(p) for _, pts in curves for p in pts]
    ax.set_ylim(min(vals) * lo, max(vals) * hi)


def line(ax, xs, ys, i, label):
    ax.plot(xs, ys, color=RAMP3[i], lw=1.5, marker=MARKS[i], ms=4.0, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6, label=label)


def save(fig, outdir, stem):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        # CreationDate=None drops the wall-clock stamp the PDF backend writes by default; without
        # it two runs of the same data differ in exactly those bytes, which destroys the check
        # that a figure was regenerated from the shipped data rather than edited.
        fig.savefig(outdir / f"{stem}.{ext}", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  {outdir / stem}.pdf + .png")


def strong_curves(rungs):
    out = []
    for lad in STRONG:
        pts = curve(rungs, lad)
        if pts:
            out.append((f"{fmt_terms(pts[0]['terms'])} terms", pts))
    return out


def weak_curves(rungs):
    out = []
    for lad in WEAK:
        pts = curve(rungs, lad)
        if not pts:
            continue
        # The size sequence steps by x1.90-2.07, not exactly x2, so a "constant load" curve
        # drifts. Quote the measured mean and range, never the nominal target.
        pn = [p["terms"] / p["nodes"] / 1e6 for p in pts]
        out.append((f"{sum(pn) / len(pn):.0f}M/node", pts))
    return out


def all_cores(curves):
    return sorted({p["cores"] for _, pts in curves for p in pts})


# --- the four drawing bodies. Each takes an axes so the standalone figure and the 2x2 composite
# --- draw from the SAME code: a composite rebuilt with its own copy of the plotting would drift
# --- from the singles, and the two would then disagree in a paper that shows both.

def draw_strong_time(ax, curves, legend=True):
    ax.set_ylabel("Time (s)")
    for i, (lab, pts) in enumerate(curves):
        xs = [p["cores"] for p in pts]
        line(ax, xs, [p["median"] for p in pts], i, lab)
        c0, t0 = pts[0]["cores"], pts[0]["median"]
        ax.plot(xs, [t0 * c0 / x for x in xs], color=RAMP3[i], lw=0.7, ls=":",
                alpha=0.65, zorder=2)
    clip_y(ax, curves, lambda p: p["median"])
    if legend:
        ax.legend(loc="lower left")


def draw_strong_efficiency(ax, curves, legend=True):
    """Strong-scaling parallel efficiency, t(N0)*N0 / (t(N)*N), each curve from its OWN smallest
    run (1, 2 and 8 nodes).

    Speedup against a single common baseline put that baseline mid-plot and made every point below
    it read under 1. An efficiency axis is the conventional fix: each curve starts at 100% where
    it starts, and the one 100% line is valid for all three. The price, stated in the caption, is
    that the three baselines differ -- so the curves say how well each SIZE scales from the
    narrowest run that fits it, not how fast one is against another.
    """
    ax.set_ylabel("Parallel efficiency (%)")
    ax.axhline(100.0, color=GUIDE, lw=0.8, ls="--", zorder=2)
    for i, (lab, pts) in enumerate(curves):
        n0, t0 = pts[0]["nodes"], pts[0]["median"]
        line(ax, [p["cores"] for p in pts],
             [100.0 * t0 * n0 / (p["median"] * p["nodes"]) for p in pts], i, lab)
    # A rung can measure just over 100%; the limit leaves room for it rather than clipping a
    # measurement to make the ideal line look like a ceiling.
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    if legend:
        ax.legend(loc="lower left")


def draw_weak_time(ax, curves, legend=True):
    ax.set_ylabel("Time (s)")
    for i, (lab, pts) in enumerate(curves):
        xs = [p["cores"] for p in pts]
        line(ax, xs, [p["median"] for p in pts], i, lab)
        ax.plot(xs, [pts[0]["median"]] * len(xs), color=RAMP3[i], lw=0.7, ls=":",
                alpha=0.65, zorder=2)
    # Headroom above the top curve is the one region empty at every x, so the legend goes there.
    clip_y(ax, curves, lambda p: p["median"], hi=3.4)
    if legend:
        ax.legend(loc="upper left", ncol=3, columnspacing=1.1, handlelength=1.6,
                  handletextpad=0.4)


def draw_weak_efficiency(ax, curves, legend=True):
    ax.set_ylabel("Parallel efficiency (%)")
    ax.axhline(100.0, color=GUIDE, lw=0.8, ls="--", zorder=2)
    for i, (lab, pts) in enumerate(curves):
        t0 = pts[0]["median"]
        line(ax, [p["cores"] for p in pts], [100.0 * t0 / p["median"] for p in pts], i, lab)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    if legend:
        ax.legend(loc="lower left")


# (kind, stem, which curves, drawing body, log y)
SINGLES = [
    ("strong", "fig-strong-time", draw_strong_time, True),
    ("strong", "fig-strong-efficiency", draw_strong_efficiency, False),
    ("weak", "fig-weak-time", draw_weak_time, True),
    ("weak", "fig-weak-efficiency", draw_weak_efficiency, False),
]


def fig_single(rungs, outdir, kind, stem, draw, logy):
    curves = strong_curves(rungs) if kind == "strong" else weak_curves(rungs)
    if not curves:
        return None
    fig, ax = panel(all_cores(curves), logy=logy)
    draw(ax, curves)
    fig.tight_layout()
    save(fig, outdir, stem)
    return curves


def fig_combined(rungs, outdir):
    """The 2x2 page-wide composite: rows are the scaling family, columns are the quantity.

    Both families span the same 128-8192 cores, so the four panels share one x axis exactly and
    only the bottom row needs the cores labels and only the top row the nodes axis -- which buys
    back the vertical space the second axis costs. The row legend lives in the left panel and
    serves the right one too, since colour and marker mean the same thing across a row.
    """
    strong, weak = strong_curves(rungs), weak_curves(rungs)
    if not (strong and weak):
        return None
    cores = sorted(set(all_cores(strong)) | set(all_cores(weak)))
    fig, axes = plt.subplots(2, 2, figsize=(W2, H2), sharex=True)
    rows = [(strong, draw_strong_time, draw_strong_efficiency, (True, False)),
            (weak, draw_weak_time, draw_weak_efficiency, (True, False))]
    for r, (curves, draw_t, draw_e, logys) in enumerate(rows):
        for c, (draw, logy) in enumerate(zip((draw_t, draw_e), logys)):
            ax = axes[r][c]
            dress(ax, cores, logy=logy, xlabel=(r == 1))
            # legend in the left panel of each row only; the right panel shares it
            draw(ax, curves, legend=(c == 0))
            nodes_axis(ax, cores)
            letter(ax, "abcd"[2 * r + c], above=True)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save(fig, outdir, "fig-scaling-2x2")
    return {"strong": strong, "weak": weak}


def letter(ax, ch, above=False):
    """Panel label outside the axes' top-left. `above` clears the nodes axis on the top row, so
    each letter sits just above its own panel's topmost furniture rather than at a shared height
    that would collide in one row and float in the other."""
    ax.annotate(f"({ch})", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(-2, 30 if above else 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=INK)


TITLE = "Figure captions -- strong and weak scaling of the Hubbard propagate operator"

PREAMBLE = """LaTeX-ready captions for the figures in this directory, converted from the single
caption definition in ../captions.py -- edit that file, not this one, and re-run
make_paper_figures.py. Ranges and percentages are computed from the plotted rows, never typed.

Figs. 1-4 are the standalone single-column figures. Fig. 5 is the page-wide 2x2 composite of the
same four panels, lettered (a)-(d), for use as a single top-of-page float; it draws from the same
code, so it cannot disagree with them. Use either the four or the one, not both.

Encoding shared by every figure: colour and marker = problem size (strong) or load per node
(weak), on a light-to-dark single-hue ordinal ramp, since the three curves differ in a MAGNITUDE
rather than in identity. Each panel carries a cores axis below and a nodes axis above -- the same
axis in the two units the machine and the reader respectively think in, not a second quantity.
The figures carry no titles, so the caption is the only place the reference lines and the machine
configuration are stated."""

FIGURES = [
    ("fig-strong-time", "Strong scaling, wall time", captions.strong_time),
    ("fig-strong-efficiency", "Strong scaling, parallel efficiency", captions.strong_efficiency),
    ("fig-weak-time", "Weak scaling, wall time", captions.weak_time),
    ("fig-weak-efficiency", "Weak scaling, parallel efficiency", captions.weak_efficiency),
    ("fig-scaling-2x2", "All four panels, page-wide composite", captions.combined),
]


def wrap(text, width=88):
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        out.append(line)
    return "\n".join(out)


def write_captions(drawn, outdir):
    """figures/captions.txt, in the same shape as the sibling monoprop paper_plots package."""
    parts = [TITLE, "=" * len(TITLE), "", PREAMBLE, ""]
    for i, (stem, heading, fn) in enumerate(FIGURES, start=1):
        curves = drawn.get(stem)
        if not curves:
            continue
        head = f"Fig. {i}  ({stem})  --  {heading}"
        parts += ["", head, "-" * len(head), wrap(captions.to_latex(fn(curves))), ""]
    path = pathlib.Path(outdir) / "captions.txt"
    path.write_text("\n".join(parts).rstrip() + "\n")
    print(f"  {path}")


def main():
    tsv = sys.argv[1] if len(sys.argv) > 1 else DATA
    outdir = sys.argv[2] if len(sys.argv) > 2 else FIGS
    rungs = load(tsv)
    have = sorted({k[0] for k in rungs})
    print(f"{tsv}: {len(rungs)} rungs over {len(have)} ladders")
    missing = [l for l in STRONG + WEAK if l not in have]
    if missing:
        print(f"  NOT MEASURED (curve omitted, not drawn short): {', '.join(missing)}")
    drawn = {stem: fig_single(rungs, outdir, kind, stem, draw, logy)
             for kind, stem, draw, logy in SINGLES}
    drawn["fig-scaling-2x2"] = fig_combined(rungs, outdir)
    write_captions(drawn, outdir)


if __name__ == "__main__":
    main()
