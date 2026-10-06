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

"""Paper Fig. 3 -- what one more gate costs, per term, against N.

Per gate, monoprop's cost is ~K/N while the reference engines pay ~K, so per layer of 2N-1
gates monoprop pays K and they pay K*N. The figure plots that and nothing else; the prose is
in figures/captions.txt, computed from the plotted rows.

The model is identical for all three engines: an open kicked-Ising chain of N qubits, seven
layers of ``Rzz(pi/4)`` on every bond then ``Rx(pi/4)`` on every site (2N-1 gates a layer),
observable ``sum_i Z_i`` propagated in the Heisenberg picture, Pauli-weight cutoff 6 with
``lower_atol = 0`` so that no coefficient is pruned and the term set is fixed by gate supports.
``check_same_workload`` refuses to plot unless all engines report the same term count, gate
count and expectation value at every point, so the two divisors of the y axis (gates and terms)
are shared and cannot manufacture a difference between the curves.

Reads data/*_octave_l7_deucalion.jsonl (written by merge_octave_runs.py) and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from merge_octave_runs import DEUCALION_FILES

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STEM = "fig3_inverse_scaling"

CUTOFF = 6
MILLION = 1.0e6
# The Deucalion x86 compute nodes (cnxNNN) are one hardware type. The rows record only the
# hostname, so the CPU model is stated here, not measured.
CPU = "AMD EPYC 7742 (Zen 2), 128 cores, SMT off"
MACHINE = f"Deucalion x86, {CPU}"
HOST = re.compile(r"cnx\d+(\.deucalion\.macc\.fccn\.pt)?")
# The ceiling octave_point.sbatch gives PauliPropagation.jl at N=1024; it is not recorded in
# the rows because a timed-out point has none.
JULIA_TIMEOUT_S = 3600

# Every engine's exponent is fitted over N <= FIT_NMAX, the widest window all three engines
# completed, so the three numbers stay like-for-like. Points above it are drawn, not fitted.
FIT_NMAX = 512

# The y axis tops out Y_HEADROOM above the highest point, capped at Y_TOP_NS (ns per gate per
# term) so that one runaway point cannot squash the decade the 1/N lives in.
Y_TOP_NS = 20.0
Y_HEADROOM = 1.3

# Print styling: mathtext only, TrueType embedded so the text stays selectable.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
# One mark per engine; colour carries the engine (Okabe-Ito, CVD-safe, separable in grey).
ENGINE_STYLE = {
    "monoprop": ("-", "o"),
    "julia": ("--", "o"),
    "ppvm": ((0, (1, 1.6)), "^"),
}
ENGINE_COLOR = {"monoprop": "#0072B2", "julia": "#D55E00", "ppvm": "#009E73"}
ENGINE_FAMILY = {"monoprop": "monoprop", "julia_pauli": "julia", "ppvm": "ppvm"}
LABEL = {"monoprop": "monoprop", "julia": "PauliPropagation.jl", "ppvm": "ppvm"}
ORDER = ("monoprop", "ppvm", "julia")
LINE_WIDTH = 1.4
MARKER_SIZE = 4.0
MARKER_EDGE = 0.9
GUIDE = "#b0b0b0"
GRID = "#d5d5d5"
PNG_DPI = 150  # preview only; the PDF keeps savefig.dpi

# A single journal column, with the legend inside the empty lower-left corner.
FIGSIZE = (3.4, 3.0)
COLUMN_RC = {
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
}
LEGEND = {"loc": "lower left", "ncol": 1, "labelspacing": 0.35}
# The floor drops far enough (per legend entry) to keep the legend clear of monoprop's tail.
BOTTOM_PAD = 3.4

# Why monoprop sits above the 1/N guide: per-term instruction counts are flat in N (full-size
# callgrind, PR #383), so the gap is accrued below N=256 as one-time steps rather than a slope.
RESIDUAL_CAUSE = (
    "per-term instruction counts are flat in N, so the gap is accrued below N=256 as "
    "one-time steps rather than a slope: there the operator outgrows a 16 MiB L3 slice "
    "(simulated last-level misses per anticommuting term rise 0.63 -> 2.33 from N=32 to "
    "N=256) and, above N=128, stored term positions widen from 8 to 16 bits"
)


def load(paths):
    """The JSONL rows, tagged with the engine family they belong to."""
    records = []
    for p in paths:
        for raw in Path(p).read_text().splitlines():
            if raw.strip():
                r = json.loads(raw)
                r["engine_family"] = ENGINE_FAMILY[r["engine"]]
                records.append(r)
    return records


def _style(fam):
    ls, marker = ENGINE_STYLE[fam]
    return {
        "ls": ls,
        "marker": marker,
        "color": ENGINE_COLOR[fam],
        "lw": LINE_WIDTH,
        "ms": MARKER_SIZE - 1.2,
        "mew": MARKER_EDGE - 0.2,
        "mfc": ENGINE_COLOR[fam] if fam == "monoprop" else "white",
        "mec": ENGINE_COLOR[fam],
        "zorder": 3 if fam == "monoprop" else 2,
    }


def _arm(records, fam, cutoff=CUTOFF):
    """The rows for one engine at one cutoff, ordered by N."""
    rows = [r for r in records if r["engine_family"] == fam and r["cutoff"] == cutoff]
    return sorted(rows, key=lambda r: r["num_qubits"])


def _per_gate_per_term(row):
    """Nanoseconds for one gate to act on one term. The only quantity this figure plots."""
    return row["seconds"] / row["gates"] / row["num_terms"] * 1e9


def _exponent(xs, ys):
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True))
    return num / sum((a - mx) ** 2 for a in lx)


def _fits(records, nmax=FIT_NMAX):
    """{engine: (all xs, all ys, exponent over N <= nmax)}."""
    out = {}
    for fam in ORDER:
        rows = _arm(records, fam)
        fitted = [r for r in rows if r["num_qubits"] <= nmax]
        out[fam] = (
            [r["num_qubits"] for r in rows],
            [_per_gate_per_term(r) for r in rows],
            _exponent(
                [r["num_qubits"] for r in fitted],
                [_per_gate_per_term(r) for r in fitted],
            ),
        )
    return out


# --------------------------------------------------------------------------- #
# Fairness preconditions -- the figure refuses to build if any of them fails
# --------------------------------------------------------------------------- #
def check_unique(records):
    """One row per (series, N): a duplicate would be drawn as a zigzag and fitted."""
    seen, problems = {}, []
    for r in records:
        key = (r["engine_family"], r["cutoff"], r["num_qubits"])
        seen[key] = seen.get(key, 0) + 1
    for (fam, cutoff, n), count in sorted(seen.items()):
        if count > 1:
            problems.append(f"{LABEL[fam]}: {count} rows at c{cutoff} N={n}")
    return problems


def check_grid(records):
    """Every engine must cover the identical set of N inside the fit window.

    A curve fitted over a different N range than its neighbour is not comparable to it. Above
    FIT_NMAX an engine may stop early: those points are drawn but never fitted. Returns the
    problems, the points per engine inside the window, and notes about the unfitted gaps.
    """
    grids = {
        fam: {
            (r["cutoff"], r["num_qubits"]) for r in records if r["engine_family"] == fam
        }
        for fam in ORDER
    }
    reference = grids["monoprop"]
    problems, notes = [], []
    for fam in ORDER[1:]:
        for label, missing in (
            (f"{LABEL[fam]} is missing", reference - grids[fam]),
            (f"{LABEL[fam]} has extra", grids[fam] - reference),
        ):
            inside = sorted(p for p in missing if p[1] <= FIT_NMAX)
            outside = sorted(p for p in missing if p[1] > FIT_NMAX)
            if inside:
                points = ", ".join(f"c{c} N={n}" for c, n in inside)
                problems.append(f"{label} {len(inside)} point(s): {points}")
            if outside:
                points = ", ".join(f"c{c} N={n}" for c, n in outside)
                notes.append(
                    f"{label} {len(outside)} unfitted point(s) above N={FIT_NMAX}: {points}"
                )
    return problems, sum(1 for _, n in reference if n <= FIT_NMAX), notes


def check_same_workload(records, expectation_tol=1e-9):
    """Every engine must have propagated the same operator through the same circuit.

    With ``lower_atol = 0`` the term set is fixed by gate supports alone, so the term counts
    agree exactly; the expectation value is the independent check that the circuits were
    identical and not merely the same size. Returns the problems and the worst expectation
    spread, so the caption quotes a measured agreement.
    """
    problems = []
    worst_spread = 0.0
    for num_qubits in sorted({r["num_qubits"] for r in records}):
        rows = {
            r["engine_family"]: r
            for r in records
            if r["cutoff"] == CUTOFF and r["num_qubits"] == num_qubits
        }
        if len(rows) < 2:
            continue
        for key in ("num_terms", "gates", "layers", "lower_atol"):
            vals = {fam: row[key] for fam, row in rows.items()}
            if len(set(vals.values())) != 1:
                detail = ", ".join(f"{LABEL[f]}={v}" for f, v in vals.items())
                problems.append(f"N={num_qubits}: {key} differs -- {detail}")
        exps = [row["expectation"] for row in rows.values()]
        spread = max(exps) - min(exps)
        worst_spread = max(worst_spread, spread)
        if spread > expectation_tol:
            problems.append(
                f"N={num_qubits}: expectation spread {spread:.3e} exceeds "
                f"{expectation_tol:.0e} -- the engines evolved different circuits"
            )
    return problems, worst_spread


def check_single_thread(records):
    """One node type, one thread -- and say which evidence each engine's claim rests on.

    monoprop and ppvm record ``cpu_seconds``, so theirs is a measured cpu/wall ratio. The
    Julia driver records the requested thread count instead (Base exposes no accurate
    per-interval process CPU clock), and PauliPropagation.jl's propagate has no threaded path
    under these settings.
    """
    problems = []
    hosts = sorted({r["host"] for r in records})
    bad = [h for h in hosts if not HOST.fullmatch(h)]
    if bad:
        problems.append(f"hosts outside the declared Deucalion x86 fleet: {bad}")
    lines = [
        f"{len(hosts)} host(s) of one node type: {MACHINE} (recorded as {', '.join(hosts)})"
    ]
    for fam in ORDER:
        rows = [r for r in records if r["engine_family"] == fam]
        threads = sorted({str(r["num_threads"]) for r in rows})
        if threads != ["1"]:
            problems.append(f"{LABEL[fam]}: num_threads {threads}, expected 1")
        busy = [
            r["busy_cores"]
            for r in rows
            if r.get("busy_cores") is not None and r["seconds"] >= 0.01
        ]
        if busy:
            if max(busy) > 1.05:
                problems.append(
                    f"{LABEL[fam]}: busy_cores up to {max(busy):.2f}, expected <= 1.05"
                )
            evidence = f"busy_cores <= {max(busy):.2f} (measured cpu/wall)"
        else:
            evidence = "declared thread count only (no cpu/wall ratio recorded)"
        lines.append(f"{LABEL[fam]:<20} num_threads=1, {evidence}")
    return problems, lines


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def _log_axis(ax, xs, xlabel, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [x for x in (32, 64, 128, 256, 512, 1024) if xs[0] <= x <= xs[-1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which="major", color=GRID, lw=0.4, alpha=0.9)
    ax.grid(visible=True, which="minor", axis="y", color=GRID, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _guide(ax, xs, exponent, anchor_xy, label, *, label_frac=0.55):
    """A reference power law through anchor_xy, annotated along its own slope."""
    x0, y0 = anchor_xy
    ys = [y0 * (x / x0) ** exponent for x in xs]
    ax.plot(xs, ys, ls=(0, (5, 3)), color=GUIDE, lw=0.9, zorder=1)
    # label_frac is a fraction of the log-x span: indexing the six octave points would
    # quantise the label to a whole octave.
    lx = xs[0] * (xs[-1] / xs[0]) ** label_frac
    ax.annotate(
        label,
        xy=(lx, y0 * (lx / x0) ** exponent),
        xytext=(0, -11),
        textcoords="offset points",
        va="top",
        color="#8a8a8a",
        fontsize=6.8,
        ha="center",
    )


def draw(records, outdir: Path):
    """ns per gate per term against N, cutoff 6, three engines, in one journal column.

    The panel is not a decade-square, so the rendered angle of a true -1 slope is a property
    of the frame. That is what the 1/N guide is for: the eye reads monoprop against the guide,
    never against the frame.
    """
    with plt.rc_context(COLUMN_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fits = _fits(records)
        for fam in ORDER:
            ax.plot(fits[fam][0], fits[fam][1], **_style(fam))

        xs = fits["monoprop"][0]
        # Anchored through monoprop's first point, so its drift above the guide IS the
        # fitted exponent the legend reports. The other engines' rulers are not drawn: their
        # exponents are in the legend.
        _guide(
            ax,
            xs,
            -1.0,
            (xs[0], fits["monoprop"][1][0]),
            "$\\propto 1/N$",
            label_frac=0.62,
        )
        _log_axis(ax, xs, "qubits $N$", "ns per gate, per term")
        ax.set_xlim(xs[0] / 1.12, xs[-1] * 1.12)
        lo = min(min(ys) for _, ys, _ in fits.values())
        hi = max(max(ys) for _, ys, _ in fits.values())
        ax.set_ylim(lo / BOTTOM_PAD, min(hi * Y_HEADROOM, Y_TOP_NS))

        handles = [
            Line2D(
                [],
                [],
                **{**_style(fam), "ms": MARKER_SIZE, "lw": LINE_WIDTH + 0.2},
                label=f"{LABEL[fam]}  $\\propto N^{{{fits[fam][2]:+.2f}}}$",
            )
            for fam in ORDER
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.5,
            borderaxespad=0.0,
            **LEGEND,
        )
        outdir.mkdir(parents=True, exist_ok=True)
        outs = []
        for ext, dpi in (("pdf", None), ("png", PNG_DPI)):
            out = outdir / f"{STEM}.{ext}"
            # CreationDate=None keeps a rebuild from the same rows byte-identical.
            fig.savefig(
                out, bbox_inches="tight", dpi=dpi, metadata={"CreationDate": None}
            )
            outs.append(out)
        plt.close(fig)
    return outs, fits


# --------------------------------------------------------------------------- #
# Caption -- every number is taken from the rows
# --------------------------------------------------------------------------- #
def _para(text, indent=""):
    """One caption paragraph, wrapped the way the hand-written ones are."""
    return textwrap.fill(
        " ".join(text.split()),
        width=94,
        initial_indent=indent,
        subsequent_indent=indent + ("  " if indent else ""),
        break_on_hyphens=False,
        break_long_words=False,
    )


def _lead_at(fits, fam, n):
    """How many times more than monoprop engine ``fam`` pays per gate per term at the same N."""
    xs, ys, _ = fits[fam]
    mx, my, _ = fits["monoprop"]
    return ys[xs.index(n)] / my[mx.index(n)]


def _gap_above_guide(xs, ys):
    """How many times the last point sits above the 1/N guide drawn through the first."""
    return ys[-1] / (ys[0] * xs[0] / xs[-1])


def _largest_step(rows):
    """The biggest single-N jump in the plotted quantity, and where it is."""
    ys = [_per_gate_per_term(r) for r in rows]
    return max(
        (
            (ys[i] / ys[i - 1], rows[i - 1]["num_qubits"], rows[i]["num_qubits"])
            for i in range(1, len(ys))
        ),
        key=lambda t: t[0],
    )


def _rounds_note(records):
    """How many timed repetitions each engine's minimum was taken over."""
    parts = []
    for fam in ORDER:
        vals = sorted(
            {
                r.get("rounds", "unrecorded")
                for r in records
                if r["engine_family"] == fam
            }
        )
        parts.append(f"{LABEL[fam]} {'/'.join(str(v) for v in vals)}")
    return ", ".join(parts)


def write_caption(outdir: Path, fits, records, *, spread):
    """Write the caption, with every number taken from the data."""
    m, pv, j = (fits[f] for f in ORDER)
    lat = _arm(records, "monoprop")
    lo, hi = lat[0], lat[-1]
    layers = lo["layers"]
    n_lo, n_hi = lo["num_qubits"], hi["num_qubits"]
    k_lo, k_hi = lo["num_terms"], hi["num_terms"]
    m_lo, m_hi = m[1][0], m[1][-1]
    fall, widen = m_lo / m_hi, n_hi // n_lo
    n_common = min(max(fits[f][0]) for f in ORDER)
    g_lo, g_hi = lo["gates"] // layers, hi["gates"] // layers
    pl_lo = lo["seconds"] / layers / k_lo * MILLION * 1e3
    pl_hi = hi["seconds"] / layers / k_hi * MILLION * 1e3
    counts = ", ".join(f"{LABEL[f]} {len(fits[f][0])}" for f in ORDER)
    hosts = sorted({r["host"] for r in records})
    all_exp = _fits(records, nmax=n_hi)
    pv_step = _largest_step(_arm(records, "ppvm"))
    gap = _gap_above_guide(m[0], m[1])
    # Successive slopes over the last two octaves, to show the curve itself is on 1/N.
    slopes = [
        math.log(m[1][i] / m[1][i - 1]) / math.log(m[0][i] / m[0][i - 1])
        for i in (-2, -1)
    ]
    stopped = [f for f in ORDER if max(fits[f][0]) < n_hi]
    stop_note = ""
    if stopped:
        stop_note = (
            f" {' and '.join(LABEL[f] for f in stopped)} stops at N={n_common}: its N={n_hi} "
            f"point exceeded the {JULIA_TIMEOUT_S} s time limit and is absent."
        )
    times = [r["seconds"] for r in lat]
    longest = max(r["seconds"] for f in ORDER[1:] for r in _arm(records, f))

    model = _para(
        f"Model: kicked-Ising chain of N qubits, open boundary; {layers} layers, each "
        "Rzz(pi/4) on all N-1 nearest-neighbour bonds then Rx(pi/4) on all N sites, so a "
        "layer holds 2N-1 gates; observable sum_i Z_i propagated in the Heisenberg picture "
        f"(the layer order therefore reverses); truncation a Pauli-weight cutoff of {CUTOFF} "
        "with lower_atol = 0, so no coefficient pruning at all and the surviving term set is "
        f"fixed by gate supports alone. N = {n_lo}..{n_hi} in powers of two ({counts} "
        f"points), one thread, every series on one exclusive node ({', '.join(hosts)})."
    )
    plotted = _para(
        "Plotted: wall-clock nanoseconds for ONE gate to act on ONE term, against N. Both "
        "divisors are checked, not assumed: at every point all engines report the same term "
        f"count ({k_lo:,} at N={n_lo} rising to {k_hi:,} at N={n_hi}), the same gate count "
        f"({lo['gates']} to {hi['gates']}) and an expectation value agreeing to {spread:.1e} "
        "absolute, so the divisors are identical across the curves and the figure script "
        "refuses to build if any of that moves."
    )
    near = "close to" if abs(m[2] + 1) <= 0.25 else "against"
    result = _para(
        f"Result. monoprop falls as N^{m[2]:+.2f}, {near} the ideal 1/N drawn beside it: "
        f"{m_lo:.3f} ns at N={n_lo} down to {m_hi:.4f} ns at N={n_hi}, a {fall:.1f}x "
        f"reduction across a {widen}x wider system. Neither reference engine falls: ppvm "
        f"N^{pv[2]:+.2f}, PauliPropagation.jl N^{j[2]:+.2f}. At N={n_lo} monoprop leads ppvm "
        f"by only {_lead_at(fits, 'ppvm', n_lo):.1f}x and PauliPropagation.jl by "
        f"{_lead_at(fits, 'julia', n_lo):.1f}x, against {_lead_at(fits, 'ppvm', n_common):.0f}x "
        f"and {_lead_at(fits, 'julia', n_common):.0f}x by N={n_common}, so the gap is opened "
        f"over the sweep and not assumed at the origin. At N={n_hi} monoprop leads ppvm by "
        f"{_lead_at(fits, 'ppvm', n_hi):.0f}x."
    )
    window = _para(
        f"The fit window. Every exponent above, and in the legend, is fitted over "
        f"N <= {FIT_NMAX}, the widest window all three engines completed; the points beyond it "
        f"are drawn but enter no fit.{stop_note} Fitted over every point each engine reached "
        f"instead, the same rows give monoprop N^{all_exp['monoprop'][2]:+.2f}, ppvm "
        f"N^{all_exp['ppvm'][2]:+.2f} and PauliPropagation.jl N^{all_exp['julia'][2]:+.2f}. "
        f"ppvm's largest single-N step in these rows is {pv_step[0]:.2f}x, at "
        f"N={pv_step[1]}->{pv_step[2]}."
    )
    divisor = _para(
        "The divisor is not the effect. A falling curve invites the objection that the gate "
        "count is itself proportional to N, so state the same rows with no gate division at "
        f"all: a monoprop layer of {g_hi} gates costs {pl_hi / pl_lo:.2f}x what a layer of "
        f"{g_lo} gates costs ({pl_lo:.1f} -> {pl_hi:.1f} ms per 10^6 terms). "
        f"{g_hi / g_lo:.1f}x the gates for {pl_hi / pl_lo:.2f}x the time. The identical "
        "divisor applied to the other two engines leaves them flat or rising, so it cannot "
        "be the source of the fall."
    )
    mechanism = _para(
        "Mechanism: selectivity, not batching. The operator is held transposed -- one column "
        "per bit position, bit r set iff term r touches that position "
        "(cpp/monoprop/detail/operator/InvertedIndex.h). Below a density of 1/64 a column is "
        "an ascending set-row list rather than a full-height bit-vector. A gate on qubit i "
        "therefore costs work proportional to the terms that touch qubit i, which at a fixed "
        "weight cutoff is a ~w/N fraction of the operator, rather than a scan over all K "
        "terms. The reference engines hold a packed key per term and revisit the whole sum "
        "for every gate, so nothing in them can fall."
    )
    where = _para(
        f"* WHERE. Measured on Deucalion's x86 partition ({CPU}), not on Leonardo. Every "
        "series was measured in one Slurm job on one node, held exclusively, with the timed "
        "process pinned to one core, so every curve -- and every fitted exponent -- comes "
        "from the same node. The dataset is cited for a SHAPE (an exponent in N), never for "
        "an absolute time.",
        "  ",
    )
    reps = _para(
        f"* Timed repetitions differ by engine, and the rows record it: {_rounds_note(records)}. "
        f"monoprop's points run {min(times):.2f} to {max(times):.0f} s, short enough at the "
        "low end that timer granularity and process noise matter, so its minimum is taken "
        f"over many more rounds; the reference engines' points run for up to "
        f"{longest / 60:.0f} minutes each, where one round already resolves them.",
        "  ",
    )
    final_terms = _para(
        f"* num_terms is the FINAL term count, while K grows through the {layers} layers, so "
        "the absolute ns per term understates the true per-term cost. It understates it "
        "identically for all three engines, so the comparison between the curves is "
        "unaffected.",
        "  ",
    )
    ceiling = _para(
        f"* N={n_hi} is the ceiling, not a choice: monoprop_MAX_NUM_MODES defaults to {n_hi} "
        "with no headroom above it, so a wider sweep needs a rebuild.",
        "  ",
    )
    residual = _para(
        f"* monoprop's residual against the ideal N^-1 is measured, not assumed: at N={n_hi} "
        f"the curve sits {gap:.2f}x above the 1/N guide through its N={n_lo} point, while its "
        f"last two octaves fall as N^{slopes[0]:+.2f} and N^{slopes[1]:+.2f}; "
        f"{RESIDUAL_CAUSE}. The 1/N is approached, not attained.",
        "  ",
    )
    text = f"""Fig. 3 -- Inverse time scaling with qubit number.

{model}

{plotted}

{result}

{window}

{divisor}

{mechanism}

Caveats, stated because the figure would otherwise overclaim:

{where}
{reps}
{final_terms}
{ceiling}
{residual}
"""
    out = outdir / "captions.txt"
    out.write_text(text)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _preconditions(records):
    """Run the like-for-like checks and refuse to build on any failure."""
    grid_problems, points, grid_notes = check_grid(records)
    workload_problems, spread = check_same_workload(records)
    thread_problems, thread_lines = check_single_thread(records)
    problems = (
        check_unique(records) + grid_problems + workload_problems + thread_problems
    )
    if problems:
        raise SystemExit(
            "\n".join(["the engines were not measured like for like:", *problems])
        )
    print("preconditions:")
    print(f"  identical N grid over N <= {FIT_NMAX}, {points} points per engine")
    for note in grid_notes:
        print(f"  note: {note}")
    print(f"  identical term and gate counts; expectation agrees to {spread:.1e}")
    for line in thread_lines:
        print(f"  {line}")
    return spread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lattice",
        nargs="+",
        type=Path,
        default=[
            DATA / f
            for f in (DEUCALION_FILES[s] for s in ("monoprop", "ppvm", "julia"))
        ],
        help="JSONL rows, one file per engine. Default: the merged Deucalion sweep in data/.",
    )
    ap.add_argument("--outdir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    records = load(args.lattice)
    print("inputs:")
    for p in args.lattice:
        print(f"  {p}")
    spread = _preconditions(records)
    outs, fits = draw(records, args.outdir)
    cap = write_caption(args.outdir, fits, records, spread=spread)

    print(
        f"\nfitted exponents over N <= {FIT_NMAX}, ns per gate per term [ideal: -1 (K/N) vs 0 (K)]"
    )
    for fam in ORDER:
        print(f"  {LABEL[fam]:<22} N^{fits[fam][2]:+.2f}")
    xs, ys, _ = fits["monoprop"]
    print(
        f"  monoprop N={xs[-1]} sits {_gap_above_guide(xs, ys):.2f}x above the 1/N guide "
        f"through its N={xs[0]} point"
    )
    print("\nwrote:")
    for o in (*outs, cap):
        print(f"  {o}")


if __name__ == "__main__":
    main()
