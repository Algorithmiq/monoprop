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

"""Paper Fig. 3 -- what a gate costs per term against N, at three circuit depths.

One page-wide row of three panels:

* (a), (b) the cost: nanoseconds for one gate to act on one term, against N, for the
  kicked-Ising chain against PauliPropagation.jl and ppvm, and the 2D Hubbard lattice against
  MajoranaPropagation.jl. Colour is the engine, each in its own colour, every series the same
  solid line; the marker is the depth (small, medium, large), the same for every engine and
  keyed with the engines in each panel's lower left. The two panels share the y axis. A dense
  engine tests every one of the K terms against every gate; monoprop visits only the terms that
  anticommute with it, so on this axis monoprop falls as 1/N and the references do not. Each
  engine's exponent per depth, over the widest N every engine finished there, goes to the caption.
* (c) the work behind it, from counts rather than timings: the share of the K terms a gate
  reads and rotates. A dense code tests all of them; monoprop reads the index lists of the
  gate's anticommutation support and rotates only the terms that anticommute with the gate
  (``scripts/count_anticommuting.py``); both shares fall about as 1/N.

Every engine runs the identical circuit on the identical operator: ``check_same_workload``
refuses to plot unless term counts, gate counts and expectation values agree at every point,
and the counts must have seen the same operator. Reads data/*.jsonl (written by
merge_octave_runs.py) and nothing else.

A point whose engine was killed at the time limit, whose one round alone ran past it, or that
was not run because a shallower circuit already exceeded it, has no cost to plot: it is left
out of the figure and the fits, and the caption lists it.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.transforms import ScaledTranslation
from merge_octave_runs import CEILING_S, COUNTS, DEPTHS, ROWS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STEM = "fig3_inverse_scaling"

# The Deucalion x86 compute nodes (cnxNNN) are one hardware type. The rows record only the
# hostname, so the CPU model is stated here, not measured.
CPU = "AMD EPYC 7742 (Zen 2), 128 cores, SMT off"
MACHINE = f"Deucalion x86, {CPU}"
HOST = re.compile(r"cnx\d+(\.deucalion\.macc\.fccn\.pt)?")

# Print styling: mathtext only, TrueType embedded so the text stays selectable. Text is ink, never
# a series colour; a coloured mark beside it carries the identity.
INK = "#1a1a1a"
INK_2 = "#4d4d4d"  # secondary text, axes and ticks
INK_3 = "#7a7a7a"  # muted text
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5,
        "text.color": INK,
        "axes.titlesize": 7.6,
        "axes.labelsize": 7.4,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "ytick.minor.size": 1.5,
        "legend.fontsize": 6.8,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
# Colour is the engine, one hue each. The two Julia references never share a panel but are
# different codes, so they get different colours: Okabe-Ito for three, and Tol's wine for
# MajoranaPropagation.jl, which clears the colour-blind check against the other three.
MONOPROP = "#0072B2"
ENGINE_COLOR = {
    "monoprop": MONOPROP,
    "monoprop-majorana": MONOPROP,
    "julia": "#D55E00",
    "julia-majorana": "#882255",
    "ppvm": "#009E73",
}
LABEL = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "PauliPropagation.jl",
    "julia-majorana": "MajoranaPropagation.jl",
    "ppvm": "ppvm",
}
# Depth (small, medium, large) is the marker alone, the same for every engine; every series is
# the same solid line at the full engine colour. Triangles read smaller than circles and
# squares at one size, hence the larger one.
DEPTH_STYLE = (
    {"marker": "o", "ls": "-", "ms": 3.4},
    {"marker": "s", "ls": "-", "ms": 3.1},
    {"marker": "^", "ls": "-", "ms": 3.9},
)
LINE_WIDTH = 1.15
# A white ring keeps a filled marker legible where it crosses another line.
RING = 0.5
# The 1/N guide and the 100% line are near-black or grey so that they survive a printer, and
# dashed, unlike every series.
GUIDE = "#222222"
GUIDE_WIDTH = 1.0
GUIDE_DASH = (0, (6, 1.6, 1.2, 1.6))
DENSE_DASH = (0, (5, 2))
GRID = "#e6e6e6"
PNG_DPI = 150  # preview only; the PDF keeps savefig.dpi

# Page width, one row: a cost panel per model, then the work per gate of both; each panel keys
# its own engines and depths in its lower left.
FIGSIZE = (7.0, 2.3)
MODELS = (
    ("pauli", "Kicked Ising chain", "qubits $N$", "layers"),
    ("majorana", "2D Hubbard lattice", "modes $N$", "layers"),
)
BAND = "#ececec"  # what the index skips
# The cost axis reaches this factor below the cheapest point, which leaves the lower left of
# each cost panel clear for its key.
COST_FLOOR = 6.0


def load(path):
    return [
        json.loads(raw) for raw in Path(path).read_text().splitlines() if raw.strip()
    ]


def _engine_key(row):
    """The campaign's engine name for a row (the monoprop field is shared by both bases)."""
    if row["engine"] == "monoprop":
        return "monoprop-majorana" if row.get("basis") == "majorana" else "monoprop"
    return {"julia_pauli": "julia", "julia_majorana": "julia-majorana"}.get(
        row["engine"], row["engine"]
    )


def _style(color, depth_index=1):
    """One series: ``color`` at full strength, a solid line with the depth's marker."""
    depth = DEPTH_STYLE[depth_index]
    return {
        **depth,
        "color": color,
        "lw": LINE_WIDTH,
        "mfc": color,
        "mec": "white",
        "mew": RING,
    }


def _per_gate_per_term(row):
    """Nanoseconds for one gate to act on one term: the panel's quantity."""
    return row["seconds"] / row["gates"] / row["num_terms"] * 1e9


def _exponent(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def by_depth(rows):
    """{layers: rows} in the order of the row's depths."""
    out = {}
    for r in rows:
        out.setdefault(r["layers"], []).append(r)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# Fairness preconditions -- the figure refuses to build if any of them fails
# --------------------------------------------------------------------------- #
def check_same_workload(datasets, counts, expectation_tol=1e-9, *, partial=False):
    """Every engine propagated the same operator through the same circuit, at every point.

    Returns the problems and the worst expectation spread, so the caption quotes a measured
    agreement.
    """
    problems, worst = [], 0.0
    for kind, rows in datasets.items():
        done = [r for r in rows if not r.get("timeout")]
        # The cutoff alone bounds every operator: a tolerance would make term counts depend on
        # rounding, and the comparison is defined without one.
        problems += [
            f"{kind} {_engine_key(r)} N={r['num_qubits']}: lower_atol {r['lower_atol']}"
            for r in done
            if r["lower_atol"] != 0
        ]
        for point in sorted({(r["layers"], r["num_qubits"]) for r in done}):
            at = {
                _engine_key(r): r
                for r in done
                if (r["layers"], r["num_qubits"]) == point
            }
            if len(at) < 2:
                continue
            for key in ("num_terms", "gates", "lower_atol", "cutoff"):
                if len({json.dumps(r.get(key)) for r in at.values()}) != 1:
                    problems.append(
                        f"{kind} depth {point[0]} N={point[1]}: {key} differs"
                    )
            exps = [r["expectation"] for r in at.values()]
            spread = max(exps) - min(exps)
            worst = max(worst, spread)
            if spread > expectation_tol:
                problems.append(
                    f"{kind} depth {point[0]} N={point[1]}: spread {spread:.2e}"
                )
        terms = {(r["layers"], r["num_qubits"]): r["num_terms"] for r in done}
        for c in counts:
            if c["basis"] != kind:
                continue
            want = terms.get((c["layers"], c["num_qubits"]))
            if want != c["num_terms"] and not (partial and want is None):
                problems.append(
                    f"{kind} depth {c['layers']} N={c['num_qubits']}: count saw "
                    f"{c['num_terms']} terms, timed {want}"
                )
    return problems, worst


def check_single_thread(datasets):
    """One node type, one thread, and the evidence for it."""
    rows = [r for rs in datasets.values() for r in rs]
    hosts = sorted({r["host"] for r in rows})
    problems = [
        f"host outside the Deucalion x86 fleet: {h}"
        for h in hosts
        if not HOST.fullmatch(h)
    ]
    for r in rows:
        if r.get("timeout"):
            continue
        if str(r["num_threads"]) != "1":
            problems.append(
                f"{_engine_key(r)} N={r['num_qubits']}: num_threads {r['num_threads']}"
            )
        busy = r.get("busy_cores")
        if busy is not None and r["seconds"] >= 0.01 and busy > 1.05:
            problems.append(
                f"{_engine_key(r)} N={r['num_qubits']}: busy_cores {busy:.2f}"
            )
    if len(hosts) > 1:
        problems.append(f"more than one node: {hosts}")
    return problems, hosts


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _frame(ax):
    """Hairline solid grid at the major ticks only, no top or right spine."""
    ax.grid(visible=True, which="major", color=GRID, lw=0.5)
    ax.grid(visible=False, which="minor")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _limit(row):
    """The time limit if the point is censored by it, else None.

    A killed point carries ``ceiling_seconds``; a point whose single round finished past the
    limit (allowed for, since the kill comes only after the setup allowance) is no better than a
    bound either. Rows measured under a shorter limit enter only if ``settled`` under this one.
    """
    if row.get("timeout"):
        return row["ceiling_seconds"]
    return CEILING_S if row["seconds"] > CEILING_S else None


def series(rows, engines):
    """{engine: (completed [(N, ns)], censored [N])} for one depth."""
    out = {}
    for e in engines:
        mine = [r for r in rows if _engine_key(r) == e]
        pts = sorted(
            (r["num_qubits"], _per_gate_per_term(r)) for r in mine if _limit(r) is None
        )
        out[e] = (pts, sorted(r["num_qubits"] for r in mine if _limit(r) is not None))
    return out


def fit_window(ser):
    """The widest N every engine completed, so every exponent is like for like."""
    return max(
        set.intersection(*({n for n, _ in pts} for pts, _ in ser.values() if pts))
    )


def _log2_axis(ax, ns):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in ns]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(ns[0] / 1.2, ns[-1] * 1.2)


def _depth_name(layers, depth_word):
    """'1 layer', '2 layers'."""
    return f"{layers} {depth_word if layers > 1 else depth_word[:-1]}"


def depth_stats(rows, engines):
    """One depth's series, exponents over the common fit window, term counts and implied rows."""
    ser = series(rows, engines)
    nmax = fit_window(ser)
    exps = {}
    for e in engines:
        fitted = [(n, y) for n, y in ser[e][0] if n <= nmax]
        exps[e] = _exponent(*zip(*fitted, strict=True)) if len(fitted) > 1 else None
    done = [r for r in rows if _engine_key(r) == engines[0] and not r.get("timeout")]
    return {
        "series": ser,
        "exponents": exps,
        "fit_nmax": nmax,
        "terms": sorted((r["num_qubits"], r["num_terms"]) for r in done),
        "implied": sorted(
            (_engine_key(r), r["num_qubits"], r["implied"])
            for r in rows
            if r.get("implied")
        ),
    }


def cost_panel(ax, rows, kind, ns):
    """ns per gate per term against N: every engine at every depth, and the 1/N guide.

    Colour is the engine; the marker is the depth, the same for every engine, keyed in the
    panel's lower left with this panel's layer counts. A point over the time limit has no cost
    to plot, only a bound, and is left out (the caption lists it).
    """
    engines = ROWS[kind][1]
    stats = {}
    for i, layers in enumerate(DEPTHS[kind]):
        st = stats[layers] = depth_stats(
            [r for r in rows if r["layers"] == layers], engines
        )
        for e in engines:
            pts, _ = st["series"][e]
            # monoprop on top: its bundle is the one the figure is about.
            z = 3 if e == engines[0] else 2
            if pts:
                ax.plot(*zip(*pts, strict=True), **_style(ENGINE_COLOR[e], i), zorder=z)

    # The 1/N guide, through monoprop's first point at the medium depth.
    mono = stats[DEPTHS[kind][1]]["series"][engines[0]][0]
    (x0, y0), xs = mono[0], [n for n, _ in mono]
    ax.plot(
        xs,
        [y0 * x0 / x for x in xs],
        ls=GUIDE_DASH,
        color=GUIDE,
        lw=GUIDE_WIDTH,
        zorder=1,
    )
    # The key: this panel's engines and the guide, one per row, over one row of its depths.
    depths = DEPTHS[kind]
    _panel_legend(
        ax,
        [Line2D([], [], **{**_style(INK_2, i), "ls": ""}) for i in range(len(depths))],
        [str(n) for n in depths[:-1]] + [f"{depths[-1]} layers"],
        ncol=len(depths),
        handlelength=0.9,
        handletextpad=0.25,
    )
    _panel_legend(
        ax,
        [Line2D([], [], color=ENGINE_COLOR[e], lw=LINE_WIDTH + 0.3) for e in engines]
        + [Line2D([], [], ls=GUIDE_DASH, color=GUIDE, lw=GUIDE_WIDTH)],
        [LABEL[e] for e in engines] + ["$\\propto 1/N$"],
        rows_below=1,
    )
    ax.set_yscale("log")
    _log2_axis(ax, ns)
    _frame(ax)
    return stats


KEY_SIZE = 6.3  # points
KEY_SPACING = 0.28  # of the font size, between rows


def _panel_legend(ax, handles, labels, *, ncol=1, rows_below=0, **kw):
    """A frameless key in the panel's lower left, the same in every panel.

    ``rows_below`` lifts it by that many key rows, over a key already drawn there; a panel can
    hold several, each kept as its own artist.
    """
    lift = rows_below * KEY_SIZE * (1 + KEY_SPACING) / 72
    key = ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 0),
        bbox_transform=ax.transAxes
        + ScaledTranslation(0, lift, ax.figure.dpi_scale_trans),
        ncol=ncol,
        frameon=False,
        fontsize=KEY_SIZE,
        labelcolor=INK,
        labelspacing=KEY_SPACING,
        borderaxespad=0.35,
        **{"handlelength": 1.9, "handletextpad": 0.45, "columnspacing": 0.7} | kw,
    )
    # Kept as a plain artist, so that the next key does not replace it (and it is drawn once).
    ax.legend_ = None
    ax.add_artist(key)


# Panel (c): every line is monoprop at the medium depth, so the model is the marker's fill and
# the quantity the line style: the index entries a gate reads dotted, the anticommuting set solid.
WORK_FILL = {"pauli": True, "majorana": False}
WORK_NAME = {"pauli": "kicked Ising", "majorana": "2D Hubbard"}
READ_DASH = (0, (1.3, 1.3))


def _log_interp(x, xs, ys):
    """``ys`` at ``x``, linear in log-log between the points ``xs``; None outside them."""
    if not xs[0] <= x <= xs[-1]:
        return None
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def work_panel(ax, counts, ns):
    """The share of the K terms a gate reads and rotates, against N, at the medium depth.

    A dense code tests every term (the 100% line). monoprop reads the index lists of the gate's
    anticommutation support (``index_reads_per_gate``: the entries of every list its scans fold)
    and rotates only the terms in their symmetric difference, the anticommuting set
    (``anticommuting_per_gate``); both from ``scripts/count_anticommuting.py`` and both falling
    about as 1/N. The band, above the higher read line, is what the index never reads.
    """
    out, reads, handles, labels = {}, {}, [], []
    for kind, _, _, _ in MODELS:
        cs = sorted(
            (c for c in counts if c["basis"] == kind), key=lambda c: c["num_qubits"]
        )
        out[kind] = {
            layers: {
                c["num_qubits"]: (
                    c["num_terms"],
                    c["anticommuting_per_gate"],
                    c.get("index_reads_per_gate"),
                )
                for c in cs
                if c["layers"] == layers
            }
            for layers in sorted({c["layers"] for c in cs})
        }
        mid = sorted(out[kind][DEPTHS[kind][1]].items())
        if any(r is None for _, (_, _, r) in mid):
            raise SystemExit(
                f"{kind}: a medium-depth count has no index_reads_per_gate"
            )
        n = np.array([x for x, _ in mid], float)
        anti = np.array([100 * a / k for _, (k, a, _) in mid])
        read = np.array([100 * r / k for _, (k, _, r) in mid])
        reads[kind] = (n, read)
        style = _style(MONOPROP, 1)
        if not WORK_FILL[kind]:
            style |= {"mfc": "white", "mec": MONOPROP, "mew": 0.9}
        ax.plot(n, read, **style | {"ls": READ_DASH}, zorder=3)
        ax.plot(n, anti, **style, zorder=3)
        handles.append(Line2D([], [], **{**style, "ls": ""}))
        layers = DEPTHS[kind][1]
        labels.append(f"{WORK_NAME[kind]}, {layers} layer{'s' if layers > 1 else ''}")
    handles += [
        Line2D([], [], color=MONOPROP, lw=LINE_WIDTH, ls=READ_DASH),
        Line2D([], [], color=MONOPROP, lw=LINE_WIDTH),
    ]
    labels += ["index entries read", "anticommuting set"]
    ax.axhline(100, color=INK_2, lw=LINE_WIDTH, ls=DENSE_DASH, zorder=2)
    # The band runs from the higher of the read lines at each N up to the 100% a dense code tests.
    grid = sorted({x for n, _ in reads.values() for x in n})
    edge = [
        max(
            v
            for n, r in reads.values()
            if (v := _log_interp(x, list(n), list(r))) is not None
        )
        for x in grid
    ]
    ax.fill_between(grid, edge, 100, color=BAND, lw=0, zorder=0)
    handles.append(Patch(facecolor=BAND, edgecolor="none"))
    labels.append("never read")
    _panel_legend(ax, handles, labels)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 250)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:g}%")
    _log2_axis(ax, ns)
    _frame(ax)
    return out


def draw(datasets, counts, outdir: Path):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)
    grid = fig.add_gridspec(1, 3)
    results = {}
    # (a) and (b) share one cost axis, with room at the bottom for the keys.
    ys = [
        _per_gate_per_term(x)
        for rows in datasets.values()
        for x in rows
        if _limit(x) is None
    ]
    first = None
    for j, (kind, title, axis, _) in enumerate(MODELS):
        rows = datasets[kind]
        ns = sorted({x["num_qubits"] for x in rows})
        ax = fig.add_subplot(grid[0, j], sharey=first)
        first = first or ax
        cost = cost_panel(ax, rows, kind, ns)
        ax.set_title(f"({'ab'[j]}) {title}", loc="left", fontsize=7.8, color=INK)
        ax.set_xlabel(axis, labelpad=1.5)
        if j == 0:
            ax.set_ylabel("ns per gate, per term", labelpad=2)
        else:
            ax.tick_params(labelleft=False)
        results[kind] = {"cost": cost}
    first.set_ylim(min(ys) / COST_FLOOR, max(ys) * 1.6)
    wax = fig.add_subplot(grid[0, 2])
    # Panel (c) holds both models; its ticks are the chain's powers of two, and the square
    # Hubbard lattices' N fall between them.
    work = work_panel(wax, counts, sorted({x["num_qubits"] for x in datasets["pauli"]}))
    wax.set_title("(c) Terms a gate reads", loc="left", fontsize=7.8, color=INK)
    wax.set_xlabel("$N$", labelpad=1.5)
    wax.set_ylabel("share of the $K$ terms, per gate", labelpad=2)
    for kind in work:
        results[kind]["work"] = work[kind]
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext, dpi in (("pdf", None), ("png", PNG_DPI)):
        out = outdir / f"{STEM}.{ext}"
        # CreationDate=None keeps a rebuild from the same rows byte-identical.
        fig.savefig(out, dpi=dpi, metadata={"CreationDate": None})
        outs.append(out)
    plt.close(fig)
    return outs, results


# --------------------------------------------------------------------------- #
# Caption -- every number is taken from the rows
# --------------------------------------------------------------------------- #
def _para(text):
    return textwrap.fill(
        " ".join(text.split()), width=94, break_on_hyphens=False, break_long_words=False
    )


def _ceiling_minutes(rows):
    """The time limit, checked against every timeout row that was run."""
    ceilings = {
        r["ceiling_seconds"] for r in rows if r.get("timeout") and not r.get("implied")
    }
    if ceilings - {CEILING_S}:
        raise SystemExit(
            f"timeout rows at another limit than {CEILING_S} s: {sorted(ceilings)}"
        )
    return f"{CEILING_S / 60:.0f}"


def _max_rounds(datasets):
    rounds = {r["max_rounds"] for rs in datasets.values() for r in rs}
    if len(rounds) > 1:
        raise SystemExit(f"rows disagree on the round cap: {sorted(rounds)}")
    return rounds.pop()


def _depth_text(engines, layers, depth_word, v):
    ser, mono = v["series"], dict(v["series"][engines[0]][0])
    exps = ", ".join(f"{LABEL[e]} N^{v['exponents'][e]:+.2f}" for e in engines)
    leads = []
    for e in engines[1:]:
        pts = dict(ser[e][0])
        common = sorted(set(pts) & set(mono))
        r0, r1 = pts[common[0]] / mono[common[0]], pts[common[-1]] / mono[common[-1]]
        leads.append(
            f"{LABEL[e]}/monoprop {r0:.2g}x at N={common[0]}, {r1:.2g}x at N={common[-1]}"
        )
        timed_out = ser[e][1]
        implied = [n for x, n, _ in v["implied"] if x == e]
        run = [n for n in timed_out if n not in implied]
        if run:
            leads.append(f"{LABEL[e]} over the limit at N={', '.join(map(str, run))}")
        if implied:
            leads.append(
                f"{LABEL[e]} not run at N={', '.join(map(str, implied))} (over the limit "
                "at a shallower depth)"
            )
    (n0, k0), (n1, k1) = v["terms"][0], v["terms"][-1]
    return (
        f"{_depth_name(layers, depth_word)}: K={k0:,} at N={n0} to {k1:,} at N={n1}; fitted over "
        f"N <= {v['fit_nmax']}: {exps}; " + "; ".join(leads)
    )


def write_caption(outdir: Path, datasets, results, *, spread, hosts):
    parts = []
    for kind, title, _, depth_word in MODELS:
        engines = ROWS[kind][1]
        cost = results[kind]["cost"]
        texts = [
            _depth_text(engines, layers, depth_word, v) for layers, v in cost.items()
        ]
        parts.append(_para(f"{title}, cost per term. " + ". ".join(texts) + "."))
        work = results[kind]["work"]
        scaled = [n * a / k for pts in work.values() for n, (k, a, _) in pts.items()]
        mid = work[DEPTHS[kind][1]]
        read_scaled = [n * r / k for n, (k, _, r) in mid.items()]
        read_over = [r / a for k, a, r in mid.values()]
        missing = [
            f"{_depth_name(layers, depth_word)} N={n}"
            for layers, v in cost.items()
            for n, _ in v["terms"]
            if n not in work.get(layers, {})
        ]
        parts.append(
            _para(
                f"{title}, work per gate: the anticommuting terms per gate are "
                f"{min(scaled):.2f}K/N to {max(scaled):.2f}K/N over every counted point"
                + (
                    f" (not counted: {', '.join(missing)}, too large for the counter)"
                    if missing
                    else ""
                )
                + f"; at {_depth_name(DEPTHS[kind][1], depth_word)} a gate reads "
                f"{min(read_scaled):.2f}K/N to {max(read_scaled):.2f}K/N index entries, "
                f"{min(read_over):.2f} to {max(read_over):.2f} times its anticommuting set."
            )
        )
    parts.append(
        _para(
            f"Agreement: identical term and gate counts at every point, expectation values within "
            f"{spread:.1e}. One thread, one pinned core, one exclusive node ({', '.join(hosts)}; "
            f"{MACHINE}). Every engine runs under the same "
            f"{_ceiling_minutes([r for rs in datasets.values() for r in rs])} min limit of timed "
            f"propagation per point, and the same rule for repetitions: the minimum of up to "
            f"{_max_rounds(datasets)} fresh rounds, at least two wherever two fit in the limit."
        )
    )
    out = outdir / "captions.txt"
    out.write_text("\n\n".join(parts) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", type=Path, default=DATA, help="directory of the merged JSONL files"
    )
    ap.add_argument("--outdir", type=Path, default=HERE / "figures")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="draw a campaign in flight (missing depths allowed, no caption); never into figures/",
    )
    args = ap.parse_args()
    if args.allow_partial and args.outdir.resolve() == HERE / "figures":
        raise SystemExit(
            "--allow-partial output is a preview: pass --outdir outside figures/"
        )

    datasets = {kind: load(args.data / name) for kind, (name, _) in ROWS.items()}
    counts = load(args.data / COUNTS)
    for kind, rows in datasets.items():
        depths = sorted({r["layers"] for r in rows})
        if depths != list(DEPTHS[kind]) and not args.allow_partial:
            raise SystemExit(f"{kind}: depths {depths}, want {list(DEPTHS[kind])}")
    problems, spread = check_same_workload(datasets, counts, partial=args.allow_partial)
    thread_problems, hosts = check_single_thread(datasets)
    if problems + thread_problems:
        raise SystemExit(
            "\n".join(["not measured like for like:", *problems, *thread_problems])
        )
    outs, results = draw(datasets, counts, args.outdir)
    if args.allow_partial:
        print("preview:", *outs, sep="\n  ")
        return
    cap = write_caption(args.outdir, datasets, results, spread=spread, hosts=hosts)
    print(cap.read_text())
    print("wrote:", *outs, cap, sep="\n  ")


if __name__ == "__main__":
    main()
