#!/usr/bin/env python3
# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What one gate selection costs, as the mode space widens.

Selecting a gate means enumerating every (state support, candidate) pair, building the
occupation-flip table over the diagonal monomials, and scoring -- the three phases the engine
times internally. Building and evolving the observable happen before the clock starts, so this is
the cost of the *choice*, not of the propagation it is choosing for.

The claim is that the choice costs what enumerating the pairs costs and nothing more. For a
generator of length `ell` applied at order `p` the pair count is bounded by `N^(ell+p)`, so each
curve is drawn against its own dotted `N^(ell+p)` reference: if selection carried hidden work on
top of the enumeration, the measured slope would exceed the bound rather than sit on it. It does
not -- the fitted time exponents land within 0.16 of the fitted pair-count exponents (printed by
this script), which is per-pair cost drifting, not extra work.

Reads data/selection_scaling.jsonl and nothing else -- no monoprop build, no engine. Style
follows the sibling node_scaling package exactly (single-column, DejaVu Serif, one-hue ordinal
ramp with a marker per curve), since both appear in the same paper.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data" / "selection_scaling.jsonl"
FIGS = ROOT / "figures"

INK, INK2, GRID, GUIDE = "#0b0b0b", "#52514e", "#dedcd6", "#9c9a94"

# The same light->dark single-hue ordinal ramp and marker set as node_scaling: the three curves
# differ in the ORDER of the cost bound (N^3, N^4, N^5), which is a magnitude, so colour encodes
# rank and the marker carries identity into greyscale print.
RAMP3 = ["#86b6ef", "#2a78d6", "#0d366b"]
MARKS = ["o", "s", "^"]

W, H = 3.4, 2.95  # single column, drawn at final size
FIT_POINTS = 4  # the widest rungs, where the asymptotic slope is the one being claimed

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#c9c7c0",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(path=DATA):
    """One record per rung, grouped by (ell, p) and ascending in N.

    Ordered by the cost bound ell+p, so the ramp's light-to-dark reads as cheap-to-expensive
    rather than as file order.
    """
    groups: dict[tuple[int, int], list[dict]] = {}
    with pathlib.Path(path).open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                groups.setdefault((r["ell"], r["p"]), []).append(r)
    for rs in groups.values():
        rs.sort(key=lambda r: r["num_modes"])
    return {k: groups[k] for k in sorted(groups, key=lambda k: (k[0] + k[1], k))}


def fit(rs, field, npoints=FIT_POINTS):
    """Least-squares log-log slope of `field` against N over the widest `npoints` rungs."""
    w = [r for r in rs[-npoints:] if r[field] > 0 and r["num_modes"] > 0]
    if len(w) < 3:
        return None
    lx = [math.log(r["num_modes"]) for r in w]
    ly = [math.log(r[field]) for r in w]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    den = sum((x - mx) ** 2 for x in lx)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly, strict=True)) / den


def plain_log(axis):
    """Plain numbers on both tick levels -- the major formatter defaults to scientific notation,
    so setting only the minor one prints "500 200 10^2" down a single axis."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axis.set_major_formatter(fmt)
    axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=20))
    axis.set_minor_formatter(fmt)


def dress(ax):
    """Log-log, light major grid, no title and no right/top furniture."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax.xaxis)
    ax.set_xlabel("Modes $N$")
    ax.set_ylabel("Selection time [s]")
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def guide(ax, rs, order):
    """The dotted `N^order` bound, anchored on the widest measured rung of its own curve.

    Anchoring at the wide end rather than fitting an intercept is what makes the line a BOUND and
    not a second fit: the eye compares slopes over the decade the claim is about, and any
    divergence between measurement and bound opens up to the left where it is visible.
    """
    xs = [r["num_modes"] for r in rs]
    x0, y0 = xs[-1], rs[-1]["seconds"]
    ax.plot(
        xs,
        [y0 * (x / x0) ** order for x in xs],
        ls=":",
        lw=0.9,
        color=GUIDE,
        zorder=1,
    )


def draw(groups, outdir=FIGS):
    fig, ax = plt.subplots(figsize=(W, H))
    dress(ax)
    rows = []
    for i, ((ell, p), rs) in enumerate(groups.items()):
        slope_t, slope_pairs = fit(rs, "seconds"), fit(rs, "pairs")
        exp = f"$N^{{{slope_t:.2f}}}$" if slope_t else "measured"
        ax.plot(
            [r["num_modes"] for r in rs],
            [r["seconds"] for r in rs],
            color=RAMP3[i % len(RAMP3)],
            lw=1.5,
            marker=MARKS[i % len(MARKS)],
            ms=4.0,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=0.6,
            # The paper states the case as (cutoff d, generator degree g) = (2 ell, 2 p), so the
            # legend does too; the records' half-degree `ell`/`p` never reach the page.
            label=rf"$d={2 * ell}$, $g={2 * p}$: {exp}",
        )
        guide(ax, rs, ell + p)
        rows.append(
            {
                "ell": ell,
                "p": p,
                "cutoff": rs[-1]["cutoff"],
                "n_min": rs[0]["num_modes"],
                "n_max": rs[-1]["num_modes"],
                "rungs": len(rs),
                "fit_window": [r["num_modes"] for r in rs[-FIT_POINTS:]],
                "slope_time": slope_t,
                "slope_pairs": slope_pairs,
                "bound": ell + p,
                "seconds_max": rs[-1]["seconds"],
                "ns_per_pair": [1e9 * r["pairs_seconds"] / r["pairs"] for r in rs],
            }
        )
    # The keys are stacked in the upper left, the one corner every curve leaves empty: all three
    # rise to the right, so a key there costs no plotted area and needs no reserved headroom.
    ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.4, borderaxespad=0.4)
    # The bound is the same statement for all three curves, so it is annotated once, in the
    # opposite corner, instead of three times in the legend.
    ax.text(
        0.98,
        0.03,
        r"dotted: $N^{(d+g)/2}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=INK2,
    )
    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig-selection-scaling")
    return rows


def save(fig, outdir, stem):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        # CreationDate=None drops the wall-clock stamp, so two runs over the same data are
        # byte-identical and a changed figure means changed DATA, not a changed clock.
        fig.savefig(outdir / f"{stem}.{ext}", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  {outdir / stem}.pdf + .png")


TITLE = "Figure caption -- what one gate selection costs as the mode space widens"

PREAMBLE = """LaTeX-ready caption, computed from the plotted rows by make_selection_figure.py --
edit that script, not this file. Every exponent and range below is measured, never typed.

Encoding: colour and marker = the (ell, p) case, on a light-to-dark single-hue ordinal ramp,
since the three curves differ in the ORDER of the cost bound rather than in identity. Both axes
are logarithmic; the dotted line beside each curve is that curve's own N^(ell+p) bound, anchored
on its widest measured rung. The figure carries no title, so this caption is the only place the
reference lines and the measurement configuration are stated."""


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


def caption(rows):
    cases = "; ".join(
        rf"$d={2 * r['ell']}$, $g={2 * r['p']}$: $N^{{{r['slope_time']:.2f}}}$ against the "
        rf"$N^{{{r['bound']}}}$ bound, over $N={r['fit_window'][0]}$--{r['fit_window'][-1]}"
        for r in rows
        if r["slope_time"]
    )
    ns = [v for r in rows for v in r["ns_per_pair"]]
    return (
        r"\textbf{Cost of one gate selection.} Time to select a single gate -- pair enumeration, "
        r"the occupation-flip table and scoring, the three phases timed inside the engine -- "
        rf"against the mode count $N$, for {len(rows)} generator/order cases in the propagated "
        r"basis, single partition, single thread. Building and evolving the observable happen "
        r"before the clock starts. Fitted log-log slopes over the "
        rf"{FIT_POINTS} widest rungs of each case: {cases}. Each dotted "
        r"line is the corresponding $N^{(d+g)/2}$ pair-count bound, anchored on the widest "
        r"measured rung. The measured time exponents track the fitted pair-count exponents "
        + ", ".join(f"({r['slope_pairs']:.2f})" for r in rows if r["slope_pairs"])
        + r" to within "
        + f"{max(abs(r['slope_time'] - r['slope_pairs']) for r in rows if r['slope_pairs']):.2f}"
        + r", and the cost of enumerating one pair stays flat at "
        + f"{min(ns):.0f}--{max(ns):.0f}"
        + r"~ns across the whole sweep, so selection costs what enumerating the pairs costs and "
        r"carries no hidden work on top of it."
    )


def write_caption(rows, outdir=FIGS):
    path = pathlib.Path(outdir) / "captions.txt"
    head = "Fig. 1  (fig-selection-scaling)  --  Cost of one gate selection"
    parts = [TITLE, "=" * len(TITLE), "", PREAMBLE, "", "", head, "-" * len(head), ""]
    parts.append(wrap(caption(rows)))
    path.write_text("\n".join(parts).rstrip() + "\n")
    print(f"  {path}")


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else DATA
    outdir = sys.argv[2] if len(sys.argv) > 2 else FIGS
    groups = load(data)
    print(
        f"{data}: {sum(len(v) for v in groups.values())} rungs over {len(groups)} cases"
    )
    rows = draw(groups, outdir)
    for r in rows:
        print(
            f"  ell={r['ell']} p={r['p']} cutoff={r['cutoff']}: "
            f"N={r['n_min']}..{r['n_max']} ({r['rungs']} rungs), "
            f"time N^{r['slope_time']:.3f} vs pairs N^{r['slope_pairs']:.3f} "
            f"(bound N^{r['bound']}), "
            f"{min(r['ns_per_pair']):.1f}-{max(r['ns_per_pair']):.1f} ns/pair"
        )
    write_caption(rows, outdir)


if __name__ == "__main__":
    main()
