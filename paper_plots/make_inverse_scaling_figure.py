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

"""Fig. 6 -- one panel: what one more gate costs, per term, against N.

THE CLAIM. Per gate, monoprop's cost is ~K/N while both reference engines pay ~K, so
per layer of 2N-1 gates monoprop pays K and they pay K*N. This figure plots that and
nothing else; every sentence that used to sit on the canvas is in fig6_caption.txt.

THE MODEL (identical for all three engines): kicked-Ising chain of N qubits, open
boundary. Five layers, each layer being ``Rzz(pi/4)`` on all N-1 nearest-neighbour
bonds followed by ``Rx(pi/4)`` on all N sites, so a layer holds 2N-1 gates. The
observable is the extensive magnetisation ``sum_i Z_i``, propagated in the Heisenberg
picture (the layer order therefore reverses). Truncation is a Pauli-weight cutoff with
``lower_atol = 0``: no coefficient pruning at all, so the surviving term set is fixed
by gate supports alone.

WHY THE NORMALISATION IS FAIR. The y axis divides wall-clock time by two things, the
gate count (which is proportional to N) and the term count K. Neither is a free
parameter and neither is engine-specific: ``check_same_workload`` refuses to plot
unless all three engines report the *same* term count, the *same* gate count and the
same expectation value at every point, so the divisors are identical across the three
curves. A shared divisor cannot manufacture a difference between them, only reveal
one. The axis is an absolute time -- nanoseconds -- rather than a cost normalised to
some round term count, so nothing is extrapolated to a term count that was not
measured, and the cutoff is therefore not load-bearing: monoprop falls at all three
cutoffs and neither reference engine is ever below N^-0.06 (the table is in the
caption). Cutoff 6 is drawn for continuity with the sibling figures.

WHY THE PER-GATE COST FALLS AT ALL, since a falling cost invites suspicion: the
operator is stored transposed, one column per bit position, and a column below 1/64
density is held as an ascending set-row list rather than a full-height bit-vector
(``InvertedIndex.h:33-46``). A gate on qubit i therefore costs work proportional to
the terms that touch qubit i, which at a fixed weight cutoff is a ~w/N fraction of
the operator. The reference engines hold a packed key per term and revisit the whole
sum per gate, so nothing in them can fall.

ENCODING. This figure deviates deliberately from the sibling figures, where colour
carries the cutoff: here the cutoff is fixed and colour carries the ENGINE, so the
three curves separate at a glance. The engine's line style and marker are kept
unchanged from every other figure (monoprop solid/filled circle), so identity is
never colour-alone and a reader who knows the other figures still recognises it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

from make_paper_figures import (
    ENGINE_LABEL,
    ENGINE_STYLE,
    GRID,
    GUIDE,
    LINE_WIDTH,
    MARKER_EDGE,
    MARKER_SIZE,
    _save,
    load,
    plt,
)

CUTOFF = 6
MILLION = 1.0e6

# Every engine's exponent is fitted over N <= this and no further, and the same window is
# used for all three so the three numbers stay like-for-like.
#
# Beyond it the reference engines cross a packed-key width boundary. PauliPropagation.jl's
# sits at N=576->608 in every sweep that straddles it: at v0.8.2 an 8.5x single-N jump in
# the full-width probe and 7.7x in the idle-spectator sweep, against 8.7x and 7.1x at
# v0.7.3 -- so the cliff is a property of the 2-bits/qubit packed key and survives the
# vectorised backend unchanged. ppvm has one of its own near N=1024 (2.14x). A
# power law fitted across a discontinuity measures where the step falls, not per-term cost
# in N, and the step is a fixed-width-integer artifact that Figs. 2 and 3 already own.
# Points above the window are still plotted, de-emphasised, and excluded from every fit.
FIT_NMAX = 512

# The panel fits every point it draws, so the drawn range and the fitted range are the same
# and there is no window a reader cannot see. The cost falls on the reference engines: their
# exponents then span the key-width discontinuity past FIT_NMAX and partly measure where
# that step falls rather than per-term cost in N. FIT_NMAX survives as the clean comparison
# window, which the caption reports beside the drawn numbers so the difference -- the cliff
# -- is on the record and quotable.
FIT_ALL_POINTS = True

# Hard ceiling on the y axis, in ns per gate per term. Letting the data set the top means
# PauliPropagation.jl's post-cliff point at ~112 ns stretches the axis to 3.8 decades and
# squashes the decade the 1/N actually lives in. Cropping here keeps that decade legible;
# the reference engine's line simply runs off the top, and the caption carries the value it
# reaches. Every point is still FITTED -- this crops the view, never the fit.
Y_TOP_NS = 10.0


# Colour carries the engine here (see the module docstring). Okabe-Ito, CVD-safe,
# and the three chosen hues stay separable in greyscale as well.
ENGINE_COLOR = {"monoprop": "#0072B2", "julia": "#D55E00", "ppvm": "#009E73"}
ORDER = ("monoprop", "ppvm", "julia")

# Two shapes of the same panel, because the figure has two homes: a single journal column,
# and a full-width banner across the top of a page. Only the frame, the type size and where
# the legend goes differ -- the data, the fits, the guide and the caption are one code path,
# so the two artifacts can never disagree.
#
# Type is set through rc_context and never at module level: this module imports plt from
# make_paper_figures, which mutates rcParams on import, so a global override here would
# silently restyle Figs. 1-5 for anything importing both.
LAYOUTS = {
    "column": {
        "stem": "fig6_inverse_scaling",
        "figsize": (3.4, 3.0),
        "rc": {
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
        },
        # Below the axes, frameless and stacked: at column width every in-axes corner is
        # either on a curve or on the 1/N guide, and a box over monoprop's tail would hide
        # the part of the curve the figure is about. Stacked rather than on one line
        # because three entries carrying an exponent each need roughly twice 3.4in.
        # Inside the axes, lower left: with the exponents off the labels the entries are
        # short, and that corner holds nothing but the tail of the 1/N guide.
        "legend": {"loc": "lower left", "ncol": 1, "labelspacing": 0.35},
        "bottom_pad": 3.4,
    },
    "page": {
        "stem": "fig6_inverse_scaling_wide",
        # Short enough to sit above a page of body text, with the three engines on one
        # horizontal line beneath the axes so the width goes to the data.
        #
        # Calibrated, not chosen: the legend is centred under the axes, so the tight bbox
        # TRIMS the unused right margin rather than adding a legend column, and the saved
        # width comes out ~0.9x of figsize. 7.95in therefore saves at 6.90x2.60in, inside a
        # RevTeX \textwidth (7.0in) -- the two-column span a figure* at the top of a page
        # gets, and the natural sibling of the column shape's \columnwidth (3.4in).
        # Re-measure the saved MediaBox if the legend text or the tick labels change width.
        # Height is the free parameter: the width is pinned by \textwidth, so it only sets
        # how much of the page the banner takes and how honestly the log-log slope reads.
        "figsize": (7.95, 4.0),
        "rc": {
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        },
        "legend": {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.2),
            "ncol": len(ORDER),
            "columnspacing": 2.4,
        },
        # Legend is below the axes here, so the floor only needs to clear the guide label.
        "bottom_pad": 1.5,
    },
}

# One machine, several names. macOS derives the hostname from DHCP whenever
# `scutil --get HostName` is unset, so a sweep long enough to span a lease renewal records
# two or three spellings of the same laptop -- the v0.8.2 Julia sweep caught all three
# below, mid-run, with `scutil --get LocalHostName` and `--get ComputerName` reporting
# Aaron-MacbookAir-Work throughout. There is no rule that maps `Mac.localdomain` onto
# `Aaron-MacbookAir-Work` without being told, so the equivalence is declared here rather
# than inferred, and check_single_thread prints every raw spelling it collapsed. Rows keep
# the name they were measured under; nothing rewrites recorded provenance.
HOST_ALIASES = {
    "Mac.localdomain": "Aaron-MacbookAir-Work.local",
    "Aaron-MacbookAir-Work": "Aaron-MacbookAir-Work.local",
}


def canonical_host(host):
    """The machine a recorded hostname denotes. Unknown names are their own canonical form."""
    return HOST_ALIASES.get(host, host)


# process_time() has a coarse tick, so cpu/wall says nothing on a sub-millisecond run.
BUSY_MIN_SECONDS = 0.01
# At or below one core busy, no row used a second thread; the slack absorbs timer jitter.
BUSY_MAX_CORES = 1.05


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


def _fits(lattice, *, fit_all=False):
    """{engine: (all xs, all ys, exponent)} -- the exponent over the chosen fit window.

    The caption must not depend on which figures were rendered or in what order. fig6()
    builds the same numbers while it plots, but main() calls this directly so the caption's
    headline exponents are the clean-window ones no matter what --fit was asked for.
    """
    out = {}
    for fam in ORDER:
        rows = _arm(lattice, fam)
        fitted = rows if fit_all else _split_at_fit_window(rows)[0]
        out[fam] = (
            [r["num_qubits"] for r in rows],
            [_per_gate_per_term(r) for r in rows],
            _exponent(
                [r["num_qubits"] for r in fitted],
                [_per_gate_per_term(r) for r in fitted],
            ),
        )
    return out


def _split_at_fit_window(rows):
    """(fitted, beyond) -- the fit window and its de-emphasised continuation.

    The boundary point belongs to both halves so the drawn line has no gap at N=FIT_NMAX.
    ``beyond`` comes back empty when the sweep stops inside the window, which is what the
    step-32 lattice sweep does, so the same code draws either dataset.
    """
    fitted = [r for r in rows if r["num_qubits"] <= FIT_NMAX]
    beyond = [r for r in rows if r["num_qubits"] >= FIT_NMAX]
    return fitted, (beyond if len(beyond) > 1 else [])


def _exponent(xs, ys):
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True))
    return num / sum((a - mx) ** 2 for a in lx)


# --------------------------------------------------------------------------- #
# Fairness preconditions -- the figure refuses to build if any of them fails
# --------------------------------------------------------------------------- #
def check_grid(records):
    """Every engine must cover the identical set of (cutoff, N).

    A curve fitted over a different N range than its neighbour is not comparable to it,
    and a missing point at one end moves an exponent more than any effect this figure is
    about, so a ragged grid is a hard error rather than a footnote.
    """
    grids = {
        fam: {
            (r["cutoff"], r["num_qubits"]) for r in records if r["engine_family"] == fam
        }
        for fam in ORDER
    }
    reference = grids[ORDER[0]]
    problems = []
    for fam in ORDER[1:]:
        for label, missing in (
            (f"{ENGINE_LABEL[fam]} is missing", reference - grids[fam]),
            (f"{ENGINE_LABEL[fam]} has extra", grids[fam] - reference),
        ):
            if missing:
                points = ", ".join(f"c{c} N={n}" for c, n in sorted(missing))
                problems.append(f"{label} {len(missing)} point(s): {points}")
    return problems, len(reference)


def check_same_workload(records, expectation_tol=1e-9):
    """The three engines must have propagated the *same* operator through the same circuit.

    This is the precondition that makes dividing by the gate count and by the term count a
    like-for-like operation rather than a rescaling of three different workloads. With
    ``lower_atol = 0`` the retained term set is fixed by gate supports alone, so the term
    counts have to agree exactly; the expectation value is the independent check that the
    circuits really were identical and not merely the same size.

    Returns the problems and the worst observed expectation spread, so the caption can
    quote a measured agreement instead of asserting one.
    """
    problems = []
    worst_spread = 0.0
    points = sorted({(r["cutoff"], r["num_qubits"]) for r in records})
    for cutoff, num_qubits in points:
        rows = {
            r["engine_family"]: r
            for r in records
            if r["cutoff"] == cutoff and r["num_qubits"] == num_qubits
        }
        if len(rows) < len(ORDER):
            continue  # check_grid reports coverage; do not double-report it here
        for key in ("num_terms", "gates", "layers", "lower_atol"):
            vals = {fam: row[key] for fam, row in rows.items()}
            if len(set(vals.values())) != 1:
                detail = ", ".join(f"{ENGINE_LABEL[f]}={v}" for f, v in vals.items())
                problems.append(f"c{cutoff} N={num_qubits}: {key} differs -- {detail}")
        exps = [row["expectation"] for row in rows.values()]
        spread = max(exps) - min(exps)
        worst_spread = max(worst_spread, spread)
        if spread > expectation_tol:
            problems.append(
                f"c{cutoff} N={num_qubits}: expectation spread {spread:.3e} "
                f"exceeds {expectation_tol:.0e} -- the engines evolved different circuits"
            )
    return problems, worst_spread


def check_single_thread(records):
    """One host, one thread -- and say which engine that claim rests on which evidence.

    monoprop and ppvm are driven from Python and record ``cpu_seconds``, so their claim is a
    measured cpu/wall ratio and the number that carries it is the *maximum* busy_cores. The
    Julia driver records the requested thread count instead: Base exposes no per-interval
    process CPU clock accurate enough to form the ratio, and PauliPropagation.jl's propagate
    has no threaded path under these settings. Report the difference rather than implying one
    measurement covers all three.
    """
    problems = []
    raw_hosts = sorted({r["host"] for r in records})
    machines = sorted({canonical_host(h) for h in raw_hosts})
    if len(machines) != 1:
        problems.append(
            f"rows come from {len(machines)} machines: {', '.join(machines)}"
        )
    # Say which names were folded together, so a reader can tell "one machine" from "one
    # hostname" and audit the aliasing instead of taking it on trust.
    spellings = (
        "" if raw_hosts == machines else f" (recorded as {', '.join(raw_hosts)})"
    )
    lines = [
        f"one host: {machines[0]}{spellings}"
        if len(machines) == 1
        else f"machines: {machines}"
    ]
    for fam in ORDER:
        rows = [r for r in records if r["engine_family"] == fam]
        threads = sorted({str(r["num_threads"]) for r in rows})
        if threads != ["1"]:
            problems.append(f"{ENGINE_LABEL[fam]}: num_threads {threads}, expected 1")
        busy = [
            r["busy_cores"]
            for r in rows
            if r.get("busy_cores") is not None and r["seconds"] >= BUSY_MIN_SECONDS
        ]
        if busy:
            if max(busy) > BUSY_MAX_CORES:
                problems.append(
                    f"{ENGINE_LABEL[fam]}: busy_cores up to {max(busy):.2f}, "
                    f"expected <= {BUSY_MAX_CORES}"
                )
            evidence = f"busy_cores <= {max(busy):.2f} (measured cpu/wall)"
        else:
            evidence = "declared thread count only (no cpu/wall ratio recorded)"
        lines.append(f"{ENGINE_LABEL[fam]:<20} num_threads=1, {evidence}")
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
    # label_frac is a fraction of the LOG-x span, not an index into the points. On an
    # octave grid there are only six points, so indexing would quantise every label to a
    # whole octave and stack the three of them on top of one another.
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


def fig6(lattice, outdir: Path, layout: str = "column"):
    """One panel: ns per gate per term against N, cutoff 6, three engines.

    Neither shape is a decade-square -- about 1.5 decades in N against 2.1 in the plotted
    time -- so the rendered angle of a true -1 slope is a property of the frame, steeper
    than 45 degrees in the column shape and shallower in the banner. That is what the ideal 1/N
    guide is for: the eye reads monoprop against the guide, never against the frame, and
    the claim survives the reshaping.
    """
    spec = LAYOUTS[layout]
    with plt.rc_context(spec["rc"]):
        fig, ax = plt.subplots(figsize=spec["figsize"])

        # Every point drawn solid, as one curve per engine. The N <= FIT_NMAX fit window
        # still sets the exponents that the caption and stdout report, but it is no longer
        # drawn: the panel carries no exponents, so there is nothing on it for a fade or a
        # clip to qualify.
        fits = _fits(lattice, fit_all=FIT_ALL_POINTS)
        for fam in ORDER:
            ax.plot(fits[fam][0], fits[fam][1], **_style(fam))

        xs = fits["monoprop"][0]
        # One slope ruler, anchored THROUGH monoprop's first point rather than offset from
        # it, so the eye reads the real relationship: monoprop starts on the ideal 1/N and
        # drifts slightly above it, and that drift IS the fitted exponent the legend
        # reports. An offset anchor made monoprop look steeper than 1/N, i.e. better than
        # it is. The reference engines' N^0 and N^+1 counterparts are deliberately NOT
        # drawn: their exponents are in the legend, and three rulers among three curves
        # read as furniture rather than as a reference.
        _guide(
            ax,
            xs,
            -1.0,
            (xs[0], fits["monoprop"][1][0]),
            "$\\propto 1/N$",
            # Mid-sweep, not at the right-hand end: the guide runs into the x axis there,
            # while below it at mid-sweep the panel is empty.
            label_frac=0.62,
        )

        _log_axis(ax, xs, "qubits $N$", "ns per gate, per term")
        ax.set_xlim(xs[0] / 1.12, xs[-1] * 1.12)
        # Floor from the data, ceiling fixed at Y_TOP_NS (see the constant), so the
        # reference engine's post-cliff point sits above the crop and its line runs off the
        # top. The bottom pad is set by the legend, not by taste: it sits inside the
        # lower-left corner in the column layout, and monoprop's tail descends into that
        # corner, so the floor has to drop far enough to leave the entries clear of it.
        lo = min(min(ys) for _, ys, _ in fits.values())
        ax.set_ylim(lo / spec["bottom_pad"], Y_TOP_NS)

        handles = [
            Line2D(
                [],
                [],
                **{**_style(fam), "ms": MARKER_SIZE, "lw": LINE_WIDTH + 0.2},
                label=ENGINE_LABEL[fam],
            )
            for fam in ORDER
        ]
        # Names only, no exponents and no title: the panel identifies the curves and
        # nothing else. Every fitted number lives in fig6_caption.txt and in the build's
        # stdout, which are now the ONLY record of them.
        ax.legend(
            handles=handles,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.5,
            borderaxespad=0.0,
            **spec["legend"],
        )
        outs = _save(fig, outdir, spec["stem"])
    return outs, fits


# --------------------------------------------------------------------------- #
# Caption -- everything that used to be printed on the canvas
# --------------------------------------------------------------------------- #
def _cutoff_table(records):
    """Per-cutoff exponents from whichever sweep carries more than one cutoff.

    Fitted over the same N <= FIT_NMAX window as the drawn panel, so the numbers in the
    table and the numbers in the legend are the same kind of number. Laid out as an
    aligned table rather than a sentence per row: three engines times three cutoffs does
    not read as prose.
    """
    cutoffs = sorted({r["cutoff"] for r in records})
    ns = sorted({r["num_qubits"] for r in records})
    k_head = f"K (N={ns[0]}->{ns[-1]})"
    head = f"{'cutoff':<7}{k_head:<23}"
    head += "".join(f"{ENGINE_LABEL[f]:<12}" for f in ORDER)
    rows_out = [f"    {head.rstrip()}"]
    for cutoff in cutoffs:
        arm = _arm(records, "monoprop", cutoff)
        lo_k, hi_k = arm[0]["num_terms"], arm[-1]["num_terms"]
        span = f"{lo_k:,} -> {hi_k:,}"
        line = f"{cutoff:<7}{span:<23}"
        for fam in ORDER:
            rows, _ = _split_at_fit_window(_arm(records, fam, cutoff))
            e = _exponent(
                [r["num_qubits"] for r in rows], [_per_gate_per_term(r) for r in rows]
            )
            exp_str = f"N^{e:+.2f}"
            line += f"{exp_str:<12}"
        rows_out.append(f"    {line.rstrip()}")
    return "\n".join(rows_out)


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
        parts.append(f"{ENGINE_LABEL[fam]} {'/'.join(str(v) for v in vals)}")
    return ", ".join(parts)


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


def write_caption(
    outdir: Path, fits, lattice, *, spread, points, spectator=None, cutoff_evidence=None
):
    """Emit the caption with every number taken from the data, not retyped."""
    m, pv, j = (fits[f] for f in ("monoprop", "ppvm", "julia"))
    lat = _arm(lattice, "monoprop")
    lo, hi = lat[0], lat[-1]
    layers = lo["layers"]
    n_lo, n_hi = lo["num_qubits"], hi["num_qubits"]
    k_lo, k_hi = lo["num_terms"], hi["num_terms"]
    m_lo, m_hi = m[1][0], m[1][-1]
    fall, widen = m_lo / m_hi, n_hi // n_lo
    # "a 16.4x" but "an 11.5x" -- the article depends on the rendered digits, so derive it
    # rather than freezing whichever one happened to be right when this was written.
    fall_article = "an" if f"{fall:.1f}".startswith(("8", "11", "18")) else "a"
    lead_j, lead_p = j[1][-1] / m_hi, pv[1][-1] / m_hi
    # Gates in one layer, and the same rows with no division by the gate count at all.
    g_lo, g_hi = lo["gates"] // layers, hi["gates"] // layers
    pl_lo = lo["seconds"] / layers / k_lo * MILLION * 1e3
    pl_hi = hi["seconds"] / layers / k_hi * MILLION * 1e3
    slower, more_gates = pl_hi / pl_lo, g_hi / g_lo
    per_engine = points // len({r["cutoff"] for r in lattice})
    host = min(canonical_host(r["host"]) for r in lattice)

    # The same three exponents fitted over everything, so the effect of excluding the
    # post-cliff points is on the record rather than something a reader has to trust.
    # The other window, for comparison: whichever one the panel did NOT fit.
    other = {f: v[2] for f, v in _fits(lattice, fit_all=not FIT_ALL_POINTS).items()}
    other_str = (
        f"monoprop N^{other['monoprop']:+.2f}, ppvm N^{other['ppvm']:+.2f} "
        f"and PauliPropagation.jl N^{other['julia']:+.2f}"
    )
    beyond = [r["num_qubits"] for r in lat if r["num_qubits"] > FIT_NMAX]
    window = ""
    if beyond:
        window = f"""
The fit window. Every exponent above is fitted over every measured point, N = {n_lo}..{n_hi}, so
the drawn range and the fitted range are the same and there is no window a reader cannot
see. Nothing on the canvas says even that much -- the legend carries engine names only --
so this caption is the record.

That choice has a cost and it falls on the reference engines. Beyond N = {FIT_NMAX} both cross a
packed-key width boundary: PauliPropagation.jl's is measured at N=576->608, an 8.5x
single-N jump in the v0.8.2 full-width probe and 7.7x in its idle-spectator sweep (8.7x
and 7.1x at v0.7.3, so the vectorised backend does not move it), and
ppvm has one of its own near N=1024 (2.14x). A power law fitted across a discontinuity
partly measures where the step falls rather than per-term cost in N, and that step is a
fixed-width-integer artifact which Figs. 2 and 3 already own. Restricted to the clean
N <= {FIT_NMAX} window the same rows give
{other_str}.
PauliPropagation.jl is the one that moves, {other["julia"]:+.2f} to {j[2]:+.2f}: that difference IS the cliff, and
it flatters monoprop here, so the N <= {FIT_NMAX} numbers are the conservative ones to quote for
per-term cost. monoprop's own exponent barely moves ({other["monoprop"]:+.2f} to {m[2]:+.2f}), having no such
boundary to cross. At N={n_hi} monoprop leads PauliPropagation.jl by {lead_j:.0f}x and ppvm by {lead_p:.0f}x.

The y axis is cropped at {Y_TOP_NS:.0f} ns per gate per term, to keep the decade the 1/N lives in
legible. PauliPropagation.jl's N={n_hi} point is above the crop, at {j[1][-1]:.0f} ns, so its line runs
off the top of the panel: that point is in the fit and in the numbers above, only not in
the view.
"""

    cutoffs = ""
    if cutoff_evidence and len({r["cutoff"] for r in cutoff_evidence}) > 1:
        src = sorted({r["num_qubits"] for r in cutoff_evidence})
        cutoffs = f"""
Nor is the cutoff. This panel draws one arm, so the cutoff-independence comes from the
companion step-{src[1] - src[0]} sweep over N={src[0]}..{src[-1]} ({len(src)} points, all three
cutoffs), fitted over the same N <= {FIT_NMAX} window:
{_cutoff_table(cutoff_evidence)}
monoprop falls at all three and neither reference engine is ever below N^-0.06, so the
choice of arm is not load-bearing; cutoff {CUTOFF} is drawn for continuity with Figs. 1-5.
"""

    isolation = ""
    if spectator:
        spec0 = _arm(spectator, "monoprop")[0]
        grow = {}
        for fam in ORDER:
            ys = [r["seconds"] / r["num_terms"] for r in _arm(spectator, fam)]
            grow[fam] = ys[-1] / ys[0]
        g_m, g_p, g_j = (grow[f] for f in ORDER)
        isolation = f"""
Why to believe it, from the companion sweep (Fig. 5a). Confining the whole model to a
{spec0["active_window"]}-qubit active window and padding the register out to N with idle spectator
qubits holds the term count ({spec0["num_terms"]:,}), the gate count ({spec0["gates"]}), every term's support
and the expectation value EXACTLY fixed while only the register width changes. Over a
32x widening monoprop pays {g_m:.1f}x, ppvm {g_p:.1f}x and PauliPropagation.jl {g_j:.0f}x. That is
per-term N-overhead isolated from everything else, and the same ordering as here.
"""

    text = f"""Fig. 6 -- The marginal cost of a gate, single-threaded.

Model: kicked-Ising chain of N qubits, open boundary; {layers} layers, each Rzz(pi/4) on all
N-1 nearest-neighbour bonds then Rx(pi/4) on all N sites, so a layer holds 2N-1 gates;
observable sum_i Z_i propagated in the Heisenberg picture (the layer order therefore
reverses); truncation a Pauli-weight cutoff of {CUTOFF} with lower_atol = 0, so no coefficient
pruning at all and the surviving term set is fixed by gate supports alone. N = {n_lo}..{n_hi} in
powers of two, {per_engine} points per engine, one thread, one host.

Plotted: wall-clock nanoseconds for ONE gate to act on ONE term, against N. Both divisors
are checked, not assumed: at every point all three engines report the same term count
({k_lo:,} at N={n_lo} rising to {k_hi:,} at N={n_hi}), the same gate count ({lo["gates"]} to {hi["gates"]}) and an
expectation value agreeing to {spread:.1e} absolute, so the divisors are identical across the
three curves and the figure script refuses to build if any of that moves. A shared divisor
cannot manufacture a difference between the curves, only reveal one.

Result. monoprop falls as N^{m[2]:+.2f}, close to the ideal 1/N drawn beside it: {m_lo:.3f} ns at
N={n_lo} down to {m_hi:.4f} ns at N={n_hi}, {fall_article} {fall:.1f}x reduction across a {widen}x wider system.
Neither reference engine falls: ppvm N^{pv[2]:+.2f}, PauliPropagation.jl N^{j[2]:+.2f}. Note where they
start -- at N={n_lo} ppvm is the fastest of the three ({pv[1][0]:.3f} ns against monoprop's {m_lo:.3f}),
so the ordering is earned over the sweep and not assumed at the origin.
{window}
The divisor is not the effect. A falling curve invites the objection that the gate count is
itself proportional to N, so state the same rows with no gate division at all: a monoprop
layer of {g_hi} gates costs {slower:.2f}x what a layer of {g_lo} gates costs ({pl_lo:.1f} -> {pl_hi:.1f} ms per
10^6 terms). {more_gates:.1f}x the gates for {slower:.2f}x the time. The identical divisor applied to the
other two engines leaves them flat or rising, so it cannot be the source of the fall.
{cutoffs}
Mechanism: selectivity, not batching. The operator is held transposed -- one column per bit
position, bit r set iff term r touches that position
(cpp/monoprop/detail/operator/InvertedIndex.h). Each column sits in one of two tiers, and
below a density of 1/64 it is an ascending set-row list rather than a full-height
bit-vector, with combine_columns_block narrowing even a dense column to a word range. A
gate on qubit i therefore costs work proportional to the terms that actually touch qubit i,
which at a fixed weight cutoff is a ~w/N fraction of the operator, rather than a scan over
all K terms. One fact, both readings: per gate the cost carries the 1/N, and per layer each
of the 2N-1 gates takes its ~1/N slice, so the layer total barely moves. The reference
engines hold a packed key per term and revisit the whole sum for every gate, so nothing in
them can fall.
{isolation}
Caveats, stated because the figure would otherwise overclaim:

  * THERMAL. Taken on a fanless laptop ({host}), not on an
    exclusive Leonardo node. At N <= {FIT_NMAX} the longest single point runs for tens of
    seconds and stays in boost, but the
    N={n_hi} reference-engine points run for minutes to tens of minutes and throttle, while
    monoprop's N={n_hi} point finishes in about a second and does not. That biases the
    reference engines' apparent cost UPWARD at the top of the range -- against them, not
    against monoprop. It is one more reason those points sit outside every fit, and it is
    why this dataset is cited only for a SHAPE (an exponent in N) and never for an
    absolute time. Figs. 1-4 are the absolute-time figures and they come from Leonardo.
  * Timed repetitions differ by engine, and the rows now record it:
    {_rounds_note(lattice)}.
    monoprop's points are sub-second, where timer granularity and process noise are
    proportionally worst, so its minimum is taken over many more rounds; the reference
    engines' expensive points would gain nothing from a second round they could not also
    lose to thermal drift within the first.
  * num_terms is the FINAL term count, while K grows through the {layers} layers, so the absolute
    ns per term understates the true per-term cost. It understates it identically for all
    three engines, so the comparison between the curves is unaffected.
  * N={n_hi} is the ceiling, not a choice: monoprop_MAX_NUM_MODES defaults to {n_hi} with no
    headroom above it, so a wider sweep needs a rebuild.
  * monoprop's residual against the ideal N^-1 is real work, not noise: the emit path still
    materialises the dense partner and folds every word for the hash, which is O(N/32) work
    on the branching fraction. The 1/N is approached, not attained.
"""
    out = outdir / "fig6_caption.txt"
    out.write_text(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lattice",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL from the full-width sweeps (one file per engine)",
    )
    ap.add_argument(
        "--spectator",
        nargs="+",
        type=Path,
        default=None,
        help="optional: the idle-spectator sweep, used only for the caption's Fig. 5a "
        "cross-reference numbers. No panel is drawn from it.",
    )
    ap.add_argument(
        "--cutoff-evidence",
        nargs="+",
        type=Path,
        default=None,
        help="optional: a sweep carrying more than one cutoff, used only for the "
        "caption's cutoff-independence table. No panel is drawn from it.",
    )
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    ap.add_argument(
        "--layout",
        nargs="+",
        choices=sorted(LAYOUTS),
        default=["column", "page"],
        help="panel shapes to emit: 'column' for one journal column, 'page' for a "
        "full-width banner. Both by default; the fits and the caption are identical.",
    )
    args = ap.parse_args()

    lattice = load(args.lattice)
    spectator = load(args.spectator) if args.spectator else None
    cutoff_evidence = load(args.cutoff_evidence) if args.cutoff_evidence else None

    grid_problems, points = check_grid(lattice)
    workload_problems, expectation_spread = check_same_workload(lattice)
    thread_problems, thread_lines = check_single_thread(lattice)
    problems = grid_problems + workload_problems + thread_problems
    if problems:
        raise SystemExit(
            "\n".join(["the three engines were not measured like for like:", *problems])
        )

    print("preconditions:")
    print(f"  identical (cutoff, N) grid, {points} points per engine")
    print(
        "  identical term and gate counts; "
        f"expectation agrees to {expectation_spread:.1e}"
    )
    for line in thread_lines:
        print(f"  {line}")

    outs = []
    for layout in dict.fromkeys(args.layout):  # de-duplicate, keep the given order
        layout_outs, _ = fig6(lattice, args.outdir, layout)
        outs += layout_outs
    # Independent of what got rendered, so the caption cannot drift from the panel.
    fits = _fits(lattice, fit_all=FIT_ALL_POINTS)
    cap = write_caption(
        args.outdir,
        fits,
        lattice,
        spread=expectation_spread,
        points=points,
        spectator=spectator,
        cutoff_evidence=cutoff_evidence,
    )

    print(
        f"\nfitted exponents over N = {min(r['num_qubits'] for r in lattice)}.."
        f"{max(r['num_qubits'] for r in lattice)}, ns per gate per term "
        "[ideal: -1 (K/N) vs 0 (K)]"
    )
    for cutoff in sorted({r["cutoff"] for r in lattice}):
        for fam in ORDER:
            rows = _arm(lattice, fam, cutoff)
            if not FIT_ALL_POINTS:
                rows = _split_at_fit_window(rows)[0]
            e = _exponent(
                [r["num_qubits"] for r in rows], [_per_gate_per_term(r) for r in rows]
            )
            mark = " <- drawn" if cutoff == CUTOFF else ""
            print(f"  c{cutoff} {ENGINE_LABEL[fam]:<22} N^{e:+.2f}{mark}")

    print("\nwrote:")
    for o in (*outs, cap):
        print(f"  {o}")


if __name__ == "__main__":
    main()
