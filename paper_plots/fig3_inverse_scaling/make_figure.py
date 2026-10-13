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
  MajoranaPropagation.jl. Colour is the engine and shade the depth (small, medium, large); the
  three depths of an engine lie close together, and only the medium one carries markers. A
  dense engine tests every one of the K terms against every gate; monoprop visits only the
  terms that anticommute with it, so on this axis monoprop falls as 1/N and the references do
  not. Each engine's exponent per depth, over the widest N every engine finished there, goes
  to the caption.
* (c) the work per gate behind it, from counts rather than timings: the K terms a dense code
  tests per term that anticommutes with the gate (``scripts/count_anticommuting.py``), which
  grows as N for both models and every depth.

Every engine runs the identical circuit on the identical operator: ``check_same_workload``
refuses to plot unless term counts, gate counts and expectation values agree at every point,
and the counts must have seen the same operator. Reads data/*.jsonl (written by
merge_octave_runs.py) and nothing else.

A point whose engine was killed at the time limit, or whose one round alone ran past it, enters
no fit and is drawn as an open marker on the rail at the top of its panel.
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
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.transforms import blended_transform_factory
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
INK_3 = "#7a7a7a"  # muted: subtitles, the rail label
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
# Colour carries the engine's role (Okabe-Ito; passes the CVD and normal-vision checks over all
# three pairs): monoprop, the Julia reference of the row, and ppvm. Line style and marker repeat
# the role for print. In the work panel, shade carries the depth.
ROLE = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "julia",
    "julia-majorana": "julia",
    "ppvm": "ppvm",
}
ENGINE_STYLE = {
    "monoprop": ("-", "o"),
    "julia": ((0, (4, 2)), "o"),
    "ppvm": ((0, (1, 1.5)), "^"),
}
ENGINE_COLOR = {"monoprop": "#0072B2", "julia": "#D55E00", "ppvm": "#009E73"}
LABEL = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "PauliPropagation.jl",
    "julia-majorana": "MajoranaPropagation.jl",
    "ppvm": "ppvm",
}
# Small depths mixed towards white, large ones towards black; medium is the engine colour. Each
# three-step ramp passes the ordinal checks (monotone lightness, light end >= 2:1 on white).
SHADE = ((1.0, 0.42), (None, 0.0), (0.0, 0.42))
LINE_WIDTH = 1.25
MARKER_SIZE = 3.6
MARKER_EDGE = 0.75
# A white ring keeps a filled marker legible where it crosses another line.
RING = 0.5
# The 1/N guide and the gap bracket are near-black so that they survive a printer.
GUIDE = "#222222"
GUIDE_WIDTH = 1.1
GUIDE_DASH = (0, (5, 2.5))
GRID = "#e6e6e6"
RAIL = 0.94  # axes fraction of the timeout rail
PNG_DPI = 150  # preview only; the PDF keeps savefig.dpi

# Page width, one row: a cost panel per model, then the work per gate of both.
FIGSIZE = (7.0, 2.7)
MODELS = (
    ("pauli", "Kicked Ising chain", "qubits $N$", "layers"),
    ("majorana", "2D Hubbard lattice", "modes $N$", "Trotter steps"),
)
# The work panel tells the models apart without the engine colours.
MODEL_STYLE = {"pauli": (INK, "o"), "majorana": (INK_3, "s")}


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


def shade(color, depth_index):
    target, t = SHADE[depth_index]
    if target is None:
        return color
    rgb = np.array(to_rgb(color))
    return to_hex(rgb + (target - rgb) * t)


def _style(engine, depth_index=1):
    role = ROLE[engine]
    ls, marker = ENGINE_STYLE[role]
    color = shade(ENGINE_COLOR[role], depth_index)
    filled = role == "monoprop"
    return {
        "ls": ls,
        "marker": marker,
        "color": color,
        "lw": LINE_WIDTH,
        "ms": MARKER_SIZE + (0.5 if filled else 0.0),
        "mew": RING if filled else MARKER_EDGE,
        "mfc": color if filled else "white",
        "mec": "white" if filled else color,
        "zorder": 3 if filled else 2,
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
    """'1 Trotter step', '2 Trotter steps'."""
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


def cost_panel(ax, rows, kind, ns, depth_word):
    """ns per gate per term against N: every engine at every depth, the 1/N guide, the rail.

    Depth is the shade of the engine's colour; only the medium depth carries markers, so the
    three depths of one engine read as one bundle.
    """
    engines = ROWS[kind][1]
    stats, rail = {}, {}
    for i, layers in enumerate(DEPTHS[kind]):
        st = stats[layers] = depth_stats(
            [r for r in rows if r["layers"] == layers], engines
        )
        for e in engines:
            pts, censored = st["series"][e]
            style = _style(e, i)
            if i != 1:
                style.update(marker="none", lw=LINE_WIDTH - 0.25)
            if pts:
                ax.plot(*zip(*pts, strict=True), **style, solid_capstyle="round")
            for n in censored:
                rail.setdefault(n, []).append((e, i))

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

    # The rail: a point censored by the time limit has no cost to plot, only a lower bound
    # that can sit under the last finished point, so it is marked above the data instead,
    # the marks at one N stepping left from it.
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [0, 1],
        [RAIL + 0.045] * 2,
        transform=ax.transAxes,
        color="#c8c8c8",
        lw=0.5,
        zorder=0,
    )
    for n, marks in rail.items():
        for k, (e, i) in enumerate(sorted(marks, key=lambda m: (ROLE[m[0]], m[1]))):
            st = _style(e, i)
            ax.plot(
                [n * 2 ** (-0.3 * (len(marks) - 1 - k))],
                [RAIL],
                transform=tr,
                ls="none",
                marker=st["marker"],
                ms=MARKER_SIZE - 0.4,
                mew=MARKER_EDGE,
                mfc="white",
                mec=st["color"],
                clip_on=False,
                zorder=4,
            )
    ax.text(
        0.03,
        RAIL,
        f"> {CEILING_S / 60:.0f} min",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.0,
        color=INK_3,
    )

    # Depth key: the shade ramp in grey, so it applies to every engine.
    grey = "#555555"
    ax.legend(
        [Line2D([], [], color=shade(grey, i), lw=LINE_WIDTH + 0.4) for i in range(3)],
        [str(layers) for layers in DEPTHS[kind]],
        title=depth_word,
        title_fontsize=6.2,
        loc="lower left",
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.3,
        alignment="left",
    )
    ax.set_yscale("log")
    _log2_axis(ax, ns)
    _frame(ax)
    return stats


def work_panel(ax, counts, ns):
    """K / anticommuting per gate for both models: the terms a dense code tests per term the
    index visits, against N. Every depth is drawn; they lie on one line."""
    out = {}
    ends = {}
    for kind, title, _, _ in MODELS:
        color, marker = MODEL_STYLE[kind]
        mine = by_depth([c for c in counts if c["basis"] == kind])
        out[kind] = {}
        for layers, unsorted in mine.items():
            mid = layers == DEPTHS[kind][1]
            cs = sorted(unsorted, key=lambda c: c["num_qubits"])
            n = [c["num_qubits"] for c in cs]
            ratio = [c["num_terms"] / c["anticommuting_per_gate"] for c in cs]
            ax.plot(
                n,
                ratio,
                color=color,
                lw=LINE_WIDTH if mid else LINE_WIDTH - 0.5,
                alpha=1.0 if mid else 0.45,
                marker=marker if mid else "none",
                ms=MARKER_SIZE - 0.6,
                mfc=color,
                mec="white",
                mew=RING,
                zorder=3 if mid else 2,
            )
            out[kind][layers] = {
                c["num_qubits"]: (c["num_terms"], c["anticommuting_per_gate"])
                for c in cs
            }
            if mid:
                ends[kind] = (n[-1], ratio[-1], n[0], ratio[0])
    # Direct labels in ink: the model beside its line, the ratio at its last point. Both
    # lines are the N guide themselves, so none is drawn.
    for kind, title, _, _ in MODELS:
        x1, y1, x0, y0 = ends[kind]
        above = kind == "pauli"
        # Beside the steep line: the upper one labelled to its left, the lower to its right.
        xm = 2 ** np.round(np.log2(np.sqrt(x0 * x1)) + (0.5 if above else -0.5))
        ax.annotate(
            "kicked Ising" if above else "2D Hubbard",
            xy=(xm, y0 * xm / x0),
            xytext=(-7, 0) if above else (7, 0),
            textcoords="offset points",
            ha="right" if above else "left",
            va="center",
            fontsize=6.6,
            color=INK,
        )
        ax.annotate(
            f"{y1:.0f}×",
            xy=(x1, y1),
            xytext=(-5, 4) if above else (4, -5),
            textcoords="offset points",
            ha="right" if above else "left",
            va="bottom" if above else "top",
            fontsize=6.6,
            color=INK,
        )
    ax.set_yscale("log")
    _log2_axis(ax, ns)
    _frame(ax)
    return out


def _legend(fig):
    """One key for the figure: the engines and the 1/N guide."""
    handles = [
        Line2D([], [], **{**_style(e), "zorder": None})
        for e in ("monoprop", "julia", "julia-majorana", "ppvm")
    ]
    handles.append(Line2D([], [], ls=GUIDE_DASH, color=GUIDE, lw=GUIDE_WIDTH))
    labels = [LABEL[e] for e in ("monoprop", "julia", "julia-majorana", "ppvm")]
    fig.legend(
        handles,
        [*labels, "$\\propto 1/N$"],
        loc="outside upper center",
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.5,
    )


def draw(datasets, counts, outdir: Path):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.15, 1))
    results = {}
    for j, (kind, title, axis, depth_word) in enumerate(MODELS):
        rows = datasets[kind]
        ns = sorted({x["num_qubits"] for x in rows})
        ax = fig.add_subplot(grid[0, j])
        cost = cost_panel(ax, rows, kind, ns, depth_word)
        ax.set_title(f"({'ab'[j]}) {title}", loc="left", fontsize=7.8, color=INK)
        ax.set_xlabel(axis, labelpad=1.5)
        if j == 0:
            ax.set_ylabel("ns per gate, per term", labelpad=2)
        # Room above the highest point for the rail, below the lowest for the depth key.
        ys = [_per_gate_per_term(x) for x in rows if _limit(x) is None]
        ax.set_ylim(min(ys) / 4.0, max(ys) * 8.0)
        results[kind] = {"cost": cost}
    wax = fig.add_subplot(grid[0, 2])
    work = work_panel(wax, counts, ns)
    wax.set_title("(c) Work per gate", loc="left", fontsize=7.8, color=INK)
    wax.set_xlabel("$N$", labelpad=1.5)
    wax.set_ylabel("terms tested / terms visited", labelpad=2)
    for kind in work:
        results[kind]["work"] = work[kind]
    _legend(fig)
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
        scaled = [n * a / k for pts in work.values() for n, (k, a) in pts.items()]
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
                + "."
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
