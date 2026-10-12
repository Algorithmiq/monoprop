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

"""Paper Fig. 3 -- what a gate costs, per term against N and per gate against K.

Two rows, one per basis: the kicked-Ising chain against PauliPropagation.jl and ppvm, and the
2D Hubbard lattice against MajoranaPropagation.jl. Two columns:

* left, the N sweep: nanoseconds for one gate to act on one term, against N. monoprop's per-gate
  cost is ~K/N while a dense engine pays ~K, so on this axis monoprop falls as 1/N and the
  references do not.
* right, the per-step sweep at fixed N: microseconds per gate against the term count K, both
  engines drawn, one point per propagate call (a layer; for Hubbard, one of a layer's nine
  commuting gate groups, which resolves the thousandfold rise of K within a layer). A dense
  engine's per-term cost grows with N (its words widen); monoprop's falls as 1/N above a
  per-gate floor. Where the two lines cross, the crossover K* is marked.

Every engine runs the identical circuit on the identical operator: ``check_same_workload``
refuses to plot unless term counts (per layer, in the per-layer sweep), gate counts and
expectation values agree. Reads data/*.jsonl (written by merge_octave_runs.py) and nothing else.

A point whose engine was killed at the time limit, or whose one round alone ran past it, is
not plotted but labelled "> 15 min" at its N, and enters no fit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from merge_octave_runs import DATASETS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STEM = "fig3_inverse_scaling"

MILLION = 1.0e6
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
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
# Colour carries the engine's role (Okabe-Ito, CVD-safe, separable in grey): monoprop, the Julia
# reference of the row, and ppvm. In the crossover panels the marker carries N.
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
N_MARKER = {32: "o", 128: "s", 256: "v", 512: "^"}
LINE_WIDTH = 1.4
MARKER_SIZE = 4.0
MARKER_EDGE = 0.9
# The 1/N guide and the reference marks are near-black so that they survive a printer.
GUIDE = "#333333"
GUIDE_WIDTH = 1.1
GRID = "#d5d5d5"
SHADE = "#D55E00"
PNG_DPI = 150  # preview only; the PDF keeps savefig.dpi

# Page width, two rows of two panels.
FIGSIZE = (7.0, 5.3)
ROWS = (
    ("pauli", "Kicked Ising", "qubits $N$"),
    ("majorana", "2D Hubbard", "modes $N$"),
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


def _style(engine, *, marker=None):
    role = ROLE[engine]
    ls, default_marker = ENGINE_STYLE[role]
    color = ENGINE_COLOR[role]
    return {
        "ls": ls,
        "marker": marker or default_marker,
        "color": color,
        "lw": LINE_WIDTH,
        "ms": MARKER_SIZE - 1.2,
        "mew": MARKER_EDGE - 0.2,
        "mfc": color if role == "monoprop" else "white",
        "mec": color,
        "zorder": 3 if role == "monoprop" else 2,
    }


def _per_gate_per_term(row):
    """Nanoseconds for one gate to act on one term: the N-sweep quantity."""
    return row["seconds"] / row["gates"] / row["num_terms"] * 1e9


def _exponent(xs, ys):
    lx, ly = np.log(xs), np.log(ys)
    return float(np.polyfit(lx, ly, 1)[0])


# --------------------------------------------------------------------------- #
# Fairness preconditions -- the figure refuses to build if any of them fails
# --------------------------------------------------------------------------- #
def check_same_workload(datasets, expectation_tol=1e-9):
    """Every engine propagated the same operator through the same circuit, at every point.

    Returns the problems and the worst expectation spread, so the caption quotes a measured
    agreement.
    """
    problems, worst = [], 0.0
    for (kind, sweep), rows in datasets.items():
        done = [r for r in rows if not r.get("timeout")]
        # The cutoff alone bounds every operator: a tolerance would make term counts depend on
        # rounding, and the comparison is defined without one.
        problems += [
            f"{kind}/{sweep} {_engine_key(r)} N={r['num_qubits']}: lower_atol {r['lower_atol']}"
            for r in done
            if r["lower_atol"] != 0
        ]
        for n in sorted({r["num_qubits"] for r in done}):
            at = {_engine_key(r): r for r in done if r["num_qubits"] == n}
            if len(at) < 2:
                continue
            keys = ["num_terms", "gates", "layers", "lower_atol", "cutoff"]
            if sweep == "perlayer":
                keys += ["layer_terms", "gates_per_layer", "step_terms", "step_gates"]
            for key in keys:
                vals = {e: json.dumps(r.get(key)) for e, r in at.items()}
                if len(set(vals.values())) != 1:
                    problems.append(f"{kind}/{sweep} N={n}: {key} differs")
            exps = [r["expectation"] for r in at.values()]
            spread = max(exps) - min(exps)
            worst = max(worst, spread)
            if spread > expectation_tol:
                problems.append(
                    f"{kind}/{sweep} N={n}: expectation spread {spread:.2e}"
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


def nsweep_series(rows, engines):
    """{engine: (completed [(N, ns)], censored [(N, lower-bound ns, limit s)])}.

    A timed-out point has no row of its own, so its bound uses the gate and term counts of an
    engine that finished at that N (every engine runs the same workload, checked above).
    """
    done = [r for r in rows if not r.get("timeout")]
    ref = {r["num_qubits"]: r for r in done}
    out = {}
    for e in engines:
        mine = [r for r in rows if _engine_key(r) == e]
        pts = sorted(
            (r["num_qubits"], _per_gate_per_term(r)) for r in mine if _limit(r) is None
        )
        bounds = sorted(
            (
                r["num_qubits"],
                _limit(r)
                / ref[r["num_qubits"]]["gates"]
                / ref[r["num_qubits"]]["num_terms"]
                * 1e9,
                _limit(r),
            )
            for r in mine
            if _limit(r) is not None and r["num_qubits"] in ref
        )
        out[e] = (pts, bounds)
    return out


def fit_window(series):
    """The widest N <= nmax every engine completed, so every exponent is like for like."""
    complete = set.intersection(*({n for n, _ in pts} for pts, _ in series.values()))
    return max(complete)


def scaling_panel(ax, rows, engines, xlabel):
    series = nsweep_series(rows, engines)
    nmax = fit_window(series)
    exps = {}
    for e in engines:
        pts, bounds = series[e]
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, **_style(e))
        fitted = [(n, y) for n, y in pts if n <= nmax]
        exps[e] = _exponent(*zip(*fitted, strict=True))
        st = _style(e)
        # A point killed at the limit bounds the cost per term only from below, and that bound
        # (limit / gates / terms) can sit under the last finished point, so drawing it as a
        # marker would read as a speed-up. It is labelled instead, in the engine's colour, at
        # the height of the engine's last finished point.
        for n, _, ceiling in bounds:
            ax.annotate(
                f"> {ceiling / 60:.0f} min",
                xy=(n, ys[-1]),
                ha="center",
                va="center",
                fontsize=6.0,
                color=st["color"],
            )
    mono = series[engines[0]][0]
    xs = [n for n, _ in mono]
    x0, y0 = mono[0]
    ax.plot(
        xs,
        [y0 * x0 / x for x in xs],
        ls=(0, (5, 3)),
        color=GUIDE,
        lw=GUIDE_WIDTH,
        zorder=1,
    )
    lx = xs[0] * (xs[-1] / xs[0]) ** 0.62
    ax.annotate(
        "$\\propto 1/N$",
        xy=(lx, y0 * x0 / lx),
        xytext=(0, -9),
        textcoords="offset points",
        va="top",
        ha="center",
        color=GUIDE,
        fontsize=7.0,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xs]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(xs[0] / 1.12, xs[-1] * 1.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ns per gate, per term")
    _frame(ax)
    lo = min(y for pts, _ in series.values() for _, y in pts)
    hi = max(y for pts, _ in series.values() for _, y in pts)
    ax.set_ylim(lo / 3.4, hi * 2.0)
    handles = [
        Line2D(
            [],
            [],
            **{**_style(e), "ms": MARKER_SIZE, "lw": LINE_WIDTH + 0.2},
            label=f"{LABEL[e]}  $\\propto N^{{{exps[e]:+.2f}}}$",
        )
        for e in engines
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower left",
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0.0,
        labelspacing=0.35,
    )
    return {"series": series, "exponents": exps, "fit_nmax": nmax}


def layer_points(row):
    """[(K, microseconds per gate)] for each step (propagate call) of a per-layer row.

    K is the geometric mean of the terms entering and leaving the step: its gates act on an
    operator growing between the two.
    """
    k_in = [row["initial_terms"], *row["step_terms"][:-1]]
    return [
        (math.sqrt(a * b), t / g * MILLION)
        for a, b, t, g in zip(
            k_in, row["step_terms"], row["step_seconds"], row["step_gates"], strict=True
        )
    ]


def crossing(mono, ref):
    """The K at which monoprop's per-gate time first drops below the reference's, or None.

    Both engines' layers share K exactly, so the ratio is taken layer by layer and the crossing
    interpolated in log-log between the two layers that bracket it. None if monoprop is never
    slower (no crossing in range) or never faster.
    """
    ratio = [(k, tm / tr) for (k, tm), (_, tr) in zip(mono, ref, strict=True)]
    for (k0, r0), (k1, r1) in pairwise(ratio):
        if r0 >= 1.0 > r1:
            f = math.log(r0) / (math.log(r0) - math.log(r1))
            return math.exp(math.log(k0) + f * (math.log(k1) - math.log(k0)))
    return None


def crossover_panel(ax, rows, engines):
    mono_e, ref_e = engines[0], engines[1]
    by = {(_engine_key(r), r["num_qubits"]): r for r in rows}
    ns = sorted({r["num_qubits"] for r in rows})
    stars, out = [], {}
    for n in ns:
        mono, ref = layer_points(by[(mono_e, n)]), layer_points(by[(ref_e, n)])
        for e, pts in ((mono_e, mono), (ref_e, ref)):
            ax.plot(*zip(*pts, strict=True), **_style(e, marker=N_MARKER.get(n, "D")))
        kstar = crossing(mono, ref)
        out[n] = {
            "k_star": kstar,
            "ratio_last": ref[-1][1] / mono[-1][1],
            "ratio_first": ref[0][1] / mono[0][1],
            "k_last": mono[-1][0],
            "k_first": mono[0][0],
        }
        if kstar is not None:
            t = math.exp(
                np.interp(
                    math.log(kstar),
                    np.log([k for k, _ in mono]),
                    np.log([t for _, t in mono]),
                )
            )
            stars.append((kstar, t))
    for k, t in stars:
        ax.plot(
            [k],
            [t],
            marker="D",
            ms=4.2,
            mfc=GUIDE,
            mec="white",
            mew=0.6,
            ls="none",
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("terms $K$")
    ax.set_ylabel("µs per gate")
    _frame(ax)
    xlo, xhi = ax.get_xlim()
    # The regions are labelled only when every N crosses: otherwise "faster" left of a K* is
    # not true of every line, and the caption says which engine is faster at every K.
    if stars and len(stars) == len(ns):
        kmin = min(k for k, _ in stars)
        ax.axvspan(xlo, kmin, color=SHADE, alpha=0.07, lw=0, zorder=0)
        ax.text(
            0.03,
            0.97,
            f"{LABEL[ref_e]}\nfaster",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            color="#8a3a00",
        )
        ax.text(
            0.97,
            0.03,
            "monoprop faster",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="#004f7c",
        )
    ax.set_xlim(xlo, xhi)
    handles = (
        [
            Line2D(
                [],
                [],
                **{**_style(e), "marker": None, "lw": LINE_WIDTH + 0.2},
                label=LABEL[e],
            )
            for e in (mono_e, ref_e)
        ]
        + [
            Line2D(
                [],
                [],
                ls="none",
                marker=N_MARKER.get(n, "D"),
                ms=MARKER_SIZE,
                mfc="#777777",
                mec="#777777",
                label=f"$N={n}$",
            )
            for n in ns
        ]
        + [
            Line2D(
                [],
                [],
                ls="none",
                marker="D",
                ms=4.2,
                mfc=GUIDE,
                mec="white",
                label="crossover $K^*$",
            )
        ][: len(stars) > 0]
    )
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.86),
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.0,
        labelspacing=0.3,
    )
    return out


def draw(datasets, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    results = {}
    for i, (kind, title, xlabel) in enumerate(ROWS):
        engines = DATASETS[(kind, "nsweep")][1]
        results[(kind, "nsweep")] = scaling_panel(
            axes[i, 0], datasets[(kind, "nsweep")], engines, xlabel
        )
        results[(kind, "perlayer")] = crossover_panel(
            axes[i, 1], datasets[(kind, "perlayer")], DATASETS[(kind, "perlayer")][1]
        )
        axes[i, 0].set_title(f"({'ac'[i]})  {title}: cost per term", loc="left")
        axes[i, 1].set_title(f"({'bd'[i]})  {title}: cost per gate", loc="left")
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


def _crossover_text(v, ref_label):
    if v["k_star"] is not None:
        return f"K*={v['k_star']:.2g}"
    if v["ratio_first"] < 1 and v["ratio_last"] < 1:
        return f"no crossover, {ref_label} faster at every K"
    if v["ratio_first"] > 1 and v["ratio_last"] > 1:
        return "no crossover, monoprop faster at every K"
    return "no single crossover"


def _ceiling_minutes(datasets):
    """The time limit, from the timeout rows; the campaign default when nothing timed out."""
    ceilings = {
        r["ceiling_seconds"] for rs in datasets.values() for r in rs if r.get("timeout")
    }
    if len(ceilings) > 1:
        raise SystemExit(f"timeout rows disagree on the time limit: {sorted(ceilings)}")
    return f"{(ceilings.pop() if ceilings else CEILING_DEFAULT_S) / 60:.0f}"


def _max_rounds(datasets):
    rounds = {r["max_rounds"] for rs in datasets.values() for r in rs}
    if len(rounds) > 1:
        raise SystemExit(f"rows disagree on the round cap: {sorted(rounds)}")
    return rounds.pop()


def write_caption(outdir: Path, datasets, results, *, spread, hosts):
    parts = []
    for kind, title, _ in ROWS:
        rows = datasets[(kind, "nsweep")]
        engines = DATASETS[(kind, "nsweep")][1]
        res = results[(kind, "nsweep")]
        mono_pts = res["series"][engines[0]][0]
        lo = next(
            r
            for r in rows
            if _engine_key(r) == engines[0] and r["num_qubits"] == mono_pts[0][0]
        )
        hi = next(
            r
            for r in rows
            if _engine_key(r) == engines[0] and r["num_qubits"] == mono_pts[-1][0]
        )
        exps = ", ".join(f"{LABEL[e]} N^{res['exponents'][e]:+.2f}" for e in engines)
        leads = []
        for e in engines[1:]:
            pts = dict(res["series"][e][0])
            mono = dict(mono_pts)
            common = sorted(set(pts) & set(mono))
            leads.append(
                f"{LABEL[e]} pays {pts[common[0]] / mono[common[0]]:.2g}x monoprop's per-term "
                f"cost at N={common[0]} and {pts[common[-1]] / mono[common[-1]]:.2g}x at "
                f"N={common[-1]}"
            )
            timeouts = [n for n, _, _ in res["series"][e][1]]
            if timeouts:
                leads.append(
                    f"{LABEL[e]} exceeded the time limit at N={', '.join(map(str, timeouts))} "
                    "(labelled in the figure)"
                )
        parts.append(
            _para(
                f"{title}, N sweep ({lo['layers']} layers, cutoff {lo['cutoff']} and no coefficient "
                f"truncation): {lo['num_terms']:,} terms at N={lo['num_qubits']} to "
                f"{hi['num_terms']:,} at N={hi['num_qubits']}. Fitted over N <= {res['fit_nmax']} "
                f"(the widest range every engine finished): {exps}. "
                + "; ".join(leads)
                + "."
            )
        )
        cross = results[(kind, "perlayer")]
        ref_e = DATASETS[(kind, "perlayer")][1][1]
        per_n = "; ".join(
            f"N={n}: {_crossover_text(v, LABEL[ref_e])}; {LABEL[ref_e]} takes "
            f"{v['ratio_first']:.2g}x monoprop's time per gate at K~{v['k_first']:.2g} and "
            f"{v['ratio_last']:.2g}x at K~{v['k_last']:.2g}"
            for n, v in cross.items()
        )
        parts.append(_para(f"{title}, step by step: {per_n}."))
    parts.append(
        _para(
            f"Agreement: identical term and gate counts at every point (per step in the per-step "
            f"sweeps), expectation values within {spread:.1e}. One thread, one pinned core, one "
            f"exclusive node ({', '.join(hosts)}; {MACHINE}). Every engine runs under the same "
            f"{_ceiling_minutes(datasets)} min limit of timed propagation per point, and the same "
            f"rule for repetitions: the minimum of up to {_max_rounds(datasets)} fresh rounds, "
            f"at least two wherever two fit in the limit."
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
    args = ap.parse_args()

    datasets = {key: load(args.data / name) for key, (name, _) in DATASETS.items()}
    problems, spread = check_same_workload(datasets)
    thread_problems, hosts = check_single_thread(datasets)
    if problems + thread_problems:
        raise SystemExit(
            "\n".join(["not measured like for like:", *problems, *thread_problems])
        )
    outs, results = draw(datasets, args.outdir)
    cap = write_caption(args.outdir, datasets, results, spread=spread, hosts=hosts)
    print(cap.read_text())
    print("wrote:", *outs, cap, sep="\n  ")


if __name__ == "__main__":
    main()
