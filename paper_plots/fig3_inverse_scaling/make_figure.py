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

One column per model: the kicked-Ising chain against PauliPropagation.jl and ppvm, and the 2D
Hubbard lattice against MajoranaPropagation.jl. Each column is a tall panel over a strip,
sharing the N axis:

* the panel: nanoseconds for one gate to act on one term, against N, at a small, a medium and a
  large depth (colour is the engine, shade is the depth). A dense engine tests every one of the
  K terms against every gate; monoprop visits only the terms that anticommute with it, so on
  this axis monoprop falls as 1/N and the references do not.
* the strip: that work per gate, from counts rather than timings -- the K terms a dense code
  tests, and the terms that anticommute with the gate (``scripts/count_anticommuting.py``),
  whose ratio grows as N.

Every engine runs the identical circuit on the identical operator: ``check_same_workload``
refuses to plot unless term counts, gate counts and expectation values agree at every point,
and the counts must have seen the same operator. Reads data/*.jsonl (written by
merge_octave_runs.py) and nothing else.

A point whose engine was killed at the time limit, or whose one round alone ran past it, enters
no fit and is drawn as an open marker on the "> 15 min" rail at the top of its panel.
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
from merge_octave_runs import COUNTS, DEPTHS, ROWS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STEM = "fig3_inverse_scaling"

# The Deucalion x86 compute nodes (cnxNNN) are one hardware type. The rows record only the
# hostname, so the CPU model is stated here, not measured.
CPU = "AMD EPYC 7742 (Zen 2), 128 cores, SMT off"
MACHINE = f"Deucalion x86, {CPU}"
HOST = re.compile(r"cnx\d+(\.deucalion\.macc\.fccn\.pt)?")
# octave_point.sbatch's CEILING_S default, quoted when no point timed out.
CEILING_DEFAULT_S = 900

# Print styling: mathtext only, TrueType embedded so the text stays selectable.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
# Colour carries the engine's role (Okabe-Ito, CVD-safe): monoprop, the Julia reference of the
# column, and ppvm. Shade carries the depth.
ROLE = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "julia",
    "julia-majorana": "julia",
    "ppvm": "ppvm",
}
ENGINE_STYLE = {
    "monoprop": ("-", "o"),
    "julia": ("--", "o"),
    "ppvm": ((0, (1, 1.6)), "^"),
}
ENGINE_COLOR = {"monoprop": "#0072B2", "julia": "#D55E00", "ppvm": "#009E73"}
LABEL = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "PauliPropagation.jl",
    "julia-majorana": "MajoranaPropagation.jl",
    "ppvm": "ppvm",
}
# Small depths are mixed towards white, large ones towards black; medium is the engine colour.
SHADE = ((1.0, 0.5), (None, 0.0), (0.0, 0.42))
LINE_WIDTH = 1.3
MARKER_SIZE = 3.2
MARKER_EDGE = 0.8
# The 1/N guide and the strip's bracket are near-black so that they survive a printer.
GUIDE = "#222222"
GUIDE_WIDTH = 1.2
GRID = "#d5d5d5"
RAIL = 0.94  # axes fraction of the timeout rail
PNG_DPI = 150  # preview only; the PDF keeps savefig.dpi

# Page width; one column per model, each a panel over a strip.
FIGSIZE = (7.0, 4.8)
COLUMNS = (
    ("pauli", "Kicked Ising", "qubits $N$", "layers"),
    ("majorana", "2D Hubbard", "modes $N$", "Trotter steps"),
)


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
    return {
        "ls": ls,
        "marker": marker,
        "color": color,
        "lw": LINE_WIDTH,
        "ms": MARKER_SIZE,
        "mew": MARKER_EDGE,
        "mfc": color if role == "monoprop" else "white",
        "mec": color,
        "zorder": 3 if role == "monoprop" else 2,
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
    ax.grid(visible=True, which="major", color=GRID, lw=0.4, alpha=0.9)
    ax.grid(visible=True, which="minor", axis="y", color=GRID, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _limit(row):
    """The row's time limit if the point is censored by it, else None.

    A killed point carries ``ceiling_seconds``; a point whose single round finished past its
    ``time_limit`` (allowed for, since the kill comes only after the setup allowance) is no
    better than a bound either.
    """
    if row.get("timeout"):
        return row["ceiling_seconds"]
    limit = row.get("time_limit")
    return limit if limit is not None and row["seconds"] > limit else None


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
    ax.set_xlim(ns[0] / 1.15, ns[-1] * 1.15)


def cost_panel(ax, rows, engines, depth_word, all_depths):
    """The ns-per-gate-per-term panel: every engine at every depth, timeouts on the rail."""
    depths = by_depth(rows)
    results, rail = {}, {}
    # A depth's shade is fixed by its place among the row's depths, even with one missing.
    shades = [all_depths.index(layers) for layers in depths]
    for i, (layers, drows) in zip(shades, depths.items(), strict=True):
        ser = series(drows, engines)
        nmax = fit_window(ser)
        exps = {}
        for e in engines:
            pts, censored = ser[e]
            if pts:
                ax.plot(*zip(*pts, strict=True), **_style(e, i))
            fitted = [(n, y) for n, y in pts if n <= nmax]
            exps[e] = _exponent(*zip(*fitted, strict=True)) if len(fitted) > 1 else None
            for n in censored:
                rail.setdefault(n, []).append((e, i))
        mono = [
            r for r in drows if _engine_key(r) == engines[0] and not r.get("timeout")
        ]
        results[layers] = {
            "series": ser,
            "exponents": exps,
            "fit_nmax": nmax,
            "terms": sorted((r["num_qubits"], r["num_terms"]) for r in mono),
            "implied": sorted(
                (_engine_key(r), r["num_qubits"], r["implied"])
                for r in drows
                if r.get("implied")
            ),
        }

    ns = sorted({r["num_qubits"] for r in rows})
    lo = min(
        y for v in results.values() for pts, _ in v["series"].values() for _, y in pts
    )
    hi = max(
        y for v in results.values() for pts, _ in v["series"].values() for _, y in pts
    )
    ax.set_yscale("log")
    # Room above the highest point for the rail, below the lowest for the guide's label.
    ax.set_ylim(lo / 3.0, hi * 9.0)
    _log2_axis(ax, ns)

    # The 1/N guide, through monoprop's first point at the medium depth.
    mid = results.get(all_depths[1], next(iter(results.values())))["series"][
        engines[0]
    ][0]
    (x0, y0), xs = mid[0], [n for n, _ in mid]
    ax.plot(
        xs,
        [y0 * x0 / x for x in xs],
        ls=(0, (5, 3)),
        color=GUIDE,
        lw=GUIDE_WIDTH,
        zorder=1,
    )
    lx = xs[0] * (xs[-1] / xs[0]) ** 0.55
    ax.annotate(
        "$\\propto 1/N$",
        xy=(lx, y0 * x0 / lx),
        xytext=(-4, -4),
        textcoords="offset points",
        va="top",
        ha="right",
        color=GUIDE,
        fontsize=7.0,
    )

    # The rail: a point censored by the time limit has no cost to plot, only a lower bound
    # that can sit under the last finished point, so it is marked above the data instead.
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [0, 1],
        [RAIL, RAIL],
        transform=ax.transAxes,
        color="#999999",
        lw=0.5,
        ls=(0, (1, 2)),
        zorder=0,
    )
    for n, marks in rail.items():
        for k, (e, i) in enumerate(marks):
            dx = 2 ** ((k - (len(marks) - 1) / 2) * 0.075)
            st = _style(e, i)
            ax.plot(
                [n * dx],
                [RAIL],
                transform=tr,
                ls="none",
                marker=st["marker"],
                ms=MARKER_SIZE + 0.4,
                mew=MARKER_EDGE,
                mfc="white",
                mec=st["color"],
                clip_on=False,
                zorder=4,
            )
    ax.text(
        0.01,
        RAIL,
        f"> {_ceiling_minutes(rows)} min",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.4,
        color="#555555",
        bbox={"fc": "white", "ec": "none", "pad": 0.6},
        zorder=5,
    )
    ax.set_ylabel("ns per gate, per term")
    _frame(ax)

    engine_handles = [
        Line2D([], [], **{**_style(e), "lw": LINE_WIDTH + 0.2}, label=LABEL[e])
        for e in engines
    ]
    depth_handles = [
        Line2D(
            [],
            [],
            color=shade("#555555", i),
            lw=2.4,
            label=_depth_name(layers, depth_word)
            + (
                f": $K$ = {_sci(res['terms'][0][1])}–{_sci(res['terms'][-1][1])}"
                if res["terms"]
                else ""
            ),
        )
        for i, (layers, res) in zip(shades, results.items(), strict=True)
    ]
    leg = ax.legend(
        handles=engine_handles,
        frameon=False,
        loc="lower left",
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.3,
    )
    ax.add_artist(leg)
    ax.legend(
        handles=depth_handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, RAIL - 0.04),
        handlelength=1.4,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.3,
    )
    return results


def _depth_name(layers, depth_word):
    """'1 Trotter step', '2 Trotter steps'."""
    return f"{layers} {depth_word if layers > 1 else depth_word[:-1]}"


def _sci(x):
    """3.2×10⁵ in mathtext."""
    e = int(np.floor(np.log10(x)))
    m = x / 10**e
    return f"${m:.1f}\\times10^{{{e}}}$"


def work_strip(ax, counts, kind, xlabel, ns):
    """Terms per gate: the K a dense code tests, and the anticommuting terms monoprop visits."""
    mine = by_depth([c for c in counts if c["basis"] == kind])
    out = {}
    for layers, unsorted in mine.items():
        i = DEPTHS[kind].index(layers)
        cs = sorted(unsorted, key=lambda c: c["num_qubits"])
        n = [c["num_qubits"] for c in cs]
        k = [c["num_terms"] for c in cs]
        a = [c["anticommuting_per_gate"] for c in cs]
        common = {
            "lw": LINE_WIDTH,
            "ms": MARKER_SIZE - 0.6,
            "mew": MARKER_EDGE,
            "marker": "o",
        }
        dense = shade(ENGINE_COLOR["julia"], i)
        mono = shade(ENGINE_COLOR["monoprop"], i)
        ax.plot(n, k, ls="--", color=dense, mfc="white", mec=dense, zorder=2, **common)
        ax.plot(n, a, ls="-", color=mono, mfc=mono, mec=mono, zorder=3, **common)
        out[layers] = {
            c["num_qubits"]: (c["num_terms"], c["anticommuting_per_gate"]) for c in cs
        }

    # The bracket: the gap between the two at the medium depth's largest counted N.
    mid = out.get(DEPTHS[kind][1], next(iter(out.values())))
    nb = max(mid)
    kb, ab = mid[nb]
    xb = nb * 1.09
    ax.annotate(
        "",
        xy=(xb, kb),
        xytext=(xb, ab),
        arrowprops={
            "arrowstyle": "<->",
            "color": GUIDE,
            "lw": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    ax.text(
        xb * 1.04,
        np.sqrt(kb * ab),
        f"{kb / ab:.0f}×",
        ha="left",
        va="center",
        fontsize=6.6,
        color=GUIDE,
    )
    ax.set_yscale("log")
    _log2_axis(ax, ns)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("terms per gate")
    _frame(ax)
    leg = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="--",
                marker="o",
                ms=MARKER_SIZE - 0.6,
                color=ENGINE_COLOR["julia"],
                mfc="white",
                lw=LINE_WIDTH,
                label="all $K$ terms (dense codes)",
            ),
            Line2D(
                [],
                [],
                ls="-",
                marker="o",
                ms=MARKER_SIZE - 0.6,
                color=ENGINE_COLOR["monoprop"],
                lw=LINE_WIDTH,
                label="anticommuting (monoprop)",
            ),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.13, 0.97),
        ncol=2,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    # Above the strip, in the gap under the panel's tick row: out of the layout, which would
    # otherwise shrink the axes to make room for it.
    leg.set_in_layout(False)
    return out


def draw(datasets, counts, outdir: Path):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.04, hspace=0.07)
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1.2))
    results = {}
    for j, (kind, title, xlabel, depth_word) in enumerate(COLUMNS):
        top = fig.add_subplot(grid[0, j])
        strip = fig.add_subplot(grid[1, j], sharex=top)
        engines = ROWS[kind][1]
        res = cost_panel(top, datasets[kind], engines, depth_word, DEPTHS[kind])
        ns = sorted({r["num_qubits"] for r in datasets[kind]})
        work = work_strip(strip, counts, kind, xlabel, ns)
        top.tick_params(labelbottom=False)
        top.set_title(f"({'ab'[j]})  {title}", loc="left")
        results[kind] = {"cost": res, "work": work}
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
    """The time limit, from the timeout rows; the campaign default when nothing timed out."""
    ceilings = {r["ceiling_seconds"] for r in rows if r.get("timeout")}
    if len(ceilings) > 1:
        raise SystemExit(f"timeout rows disagree on the time limit: {sorted(ceilings)}")
    return f"{(ceilings.pop() if ceilings else CEILING_DEFAULT_S) / 60:.0f}"


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
    for kind, title, _, depth_word in COLUMNS:
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
