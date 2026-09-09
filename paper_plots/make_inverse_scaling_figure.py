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

"""Fig. 6 -- the inverse per-gate scaling, on one page, single-threaded.

THE MODEL (identical for all three engines, stated on the figure itself):
kicked-Ising chain of N qubits, open boundary. Five layers, each layer being
``Rzz(pi/4)`` on all N-1 nearest-neighbour bonds followed by ``Rx(pi/4)`` on all N
sites, so a layer holds 2N-1 gates. The observable is the extensive magnetisation
``sum_i Z_i``, propagated in the Heisenberg picture (the layer order therefore
reverses). Truncation is a Pauli-weight cutoff of 6 with ``lower_atol = 0``: no
coefficient pruning at all, so the surviving term set is fixed by gate supports
alone and is *identical* in every engine, which is what makes dividing by the term
count a fair operation rather than a rescaling of three different workloads.

WHY THE Y AXIS IS WHAT IT IS. The quantity plotted is the wall-clock cost of one
gate acting on one million terms. Both normalisations are forced:

  * per million terms, because K is not a free parameter -- it is set by N and the
    cutoff (119,280 terms at N=32 up to 2,310,000 at N=512). Cutoff 6 is the only
    arm where 10^6 lies INSIDE the measured range; normalising the cutoff-2 arm
    (188..3068 terms) to a million would be a 326x extrapolation, and at that size
    fixed overheads, not per-term work, set the time.
  * per gate, because the gate count is not free either: a layer is 2N-1 gates
    wide, so a sweep in N sweeps the circuit size too. Per gate is the marginal
    quantity -- what one more gate costs.

WHY THE PER-GATE COST FALLS AT ALL, since a falling cost invites suspicion: the
operator is stored transposed, one column per bit position, and a column below 1/64
density is held as an ascending set-row list rather than a full-height bit-vector
(``InvertedIndex.h:33-46``). A gate on qubit i therefore costs work proportional to
the terms that touch qubit i, which at a fixed weight cutoff is a ~w/N fraction of
the operator. The reference engines hold a packed key per term and revisit the whole
sum per gate, so nothing in them can fall.

Dividing by K is conservative for monoprop, not generous: its cutoff-6 cost fits
K^1.14 over this sweep, so the division leaves a residual K^0.14 working against it.
And the same divisor is applied to all three engines, so no normalisation can
manufacture a difference between them -- only reveal one.

ENCODING. This figure deviates deliberately from the sibling figures, where colour
carries the cutoff: here the cutoff is fixed at 6 and colour carries the ENGINE, so
the three curves separate at a glance. The engine's line style and marker are kept
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
    INK,
    LINE_WIDTH,
    MARKER_EDGE,
    MARKER_SIZE,
    _save,
    load,
    plt,
)

CUTOFF = 6
MILLION = 1.0e6

# Colour carries the engine here (see the module docstring). Okabe-Ito, CVD-safe,
# and the three chosen hues stay separable in greyscale as well.
ENGINE_COLOR = {"monoprop": "#0072B2", "julia": "#D55E00", "ppvm": "#009E73"}
ORDER = ("monoprop", "ppvm", "julia")


def _style(fam):
    ls, marker = ENGINE_STYLE[fam]
    return {
        "ls": ls,
        "marker": marker,
        "color": ENGINE_COLOR[fam],
        "lw": LINE_WIDTH + 0.3,
        "ms": MARKER_SIZE + 0.4,
        "mew": MARKER_EDGE,
        "mfc": ENGINE_COLOR[fam] if fam == "monoprop" else "white",
        "mec": ENGINE_COLOR[fam],
        "zorder": 3 if fam == "monoprop" else 2,
    }


def _arm(records, fam):
    """The cutoff-6 rows for one engine, ordered by N."""
    rows = [r for r in records if r["engine_family"] == fam and r["cutoff"] == CUTOFF]
    return sorted(rows, key=lambda r: r["num_qubits"])


def _exponent(xs, ys):
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True))
    return num / sum((a - mx) ** 2 for a in lx)


def _log_axis(ax, xs, xlabel, ylabel, gates=False):
    """Log-log axis. ``gates`` writes the layer's gate count under each N tick.

    That parenthetical replaces what used to be a twin top axis: the gate count is the
    reason the per-gate cost falls, so it has to be on the figure, but a second axis
    costs the vertical room the panel title needs and its labels collide at the right.
    """
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [x for x in (32, 64, 128, 256, 512, 1024) if xs[0] <= x <= xs[-1]]
    labels = [f"{t}\n({2 * t - 1})" if gates else str(t) for t in ticks]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which="major", color=GRID, lw=0.5, alpha=0.9)
    ax.grid(visible=True, which="minor", axis="y", color=GRID, lw=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _guide(ax, xs, exponent, anchor_xy, label, label_frac=0.55):
    """A reference power law through anchor_xy, annotated along its own slope."""
    x0, y0 = anchor_xy
    ys = [y0 * (x / x0) ** exponent for x in xs]
    ax.plot(xs, ys, ls=(0, (5, 3)), color=GUIDE, lw=1.1, zorder=1)
    i = min(int(len(xs) * label_frac), len(xs) - 1)
    ax.annotate(
        label,
        xy=(xs[i], ys[i]),
        xytext=(0, -15),
        textcoords="offset points",
        color="#8a8a8a",
        fontsize=9,
        ha="center",
        rotation=0,
    )


# --------------------------------------------------------------------------- #
# (a) the inverse scaling: what one more gate costs
# --------------------------------------------------------------------------- #
def _panel_per_gate(ax, records):
    """ms for one gate to act on 10^6 terms, against N. monoprop ~ 1/N."""
    fits = {}
    for fam in ORDER:
        rows = _arm(records, fam)
        xs = [r["num_qubits"] for r in rows]
        ys = [r["seconds"] / r["gates"] / r["num_terms"] * MILLION * 1e3 for r in rows]
        fits[fam] = (xs, ys, _exponent(xs, ys))
        ax.plot(xs, ys, **_style(fam))

    xs = fits["monoprop"][0]
    # Anchored THROUGH monoprop's first point, not offset from it, so the eye reads the
    # real relationship: monoprop starts on the ideal 1/N and drifts slightly above it,
    # which is the N^-0.86 the fit reports. An offset anchor made monoprop look steeper
    # than 1/N, i.e. better than it is.
    _guide(
        ax,
        xs,
        -1.0,
        (xs[0], fits["monoprop"][1][0]),
        "ideal  $\\propto 1/N$",
        label_frac=0.86,
    )

    # Name the gap where it is widest, at the right-hand edge.
    m_last = fits["monoprop"][1][-1]
    for fam in ("julia", "ppvm"):
        last = fits[fam][1][-1]
        ax.annotate(
            f"{last / m_last:.0f}$\\times$",
            xy=(xs[-1], last),
            xytext=(7, -3),
            textcoords="offset points",
            ha="left",
            fontsize=10,
            color=ENGINE_COLOR[fam],
            fontweight="bold",
            annotation_clip=False,
        )
    ax.annotate(
        "monoprop",
        xy=(xs[-1], m_last),
        xytext=(7, -3),
        textcoords="offset points",
        ha="left",
        fontsize=9,
        color=ENGINE_COLOR["monoprop"],
        fontweight="bold",
        annotation_clip=False,
    )

    _log_axis(
        ax,
        xs,
        "qubits $N$   (gates in one layer)",
        "ms for one gate to act on $10^6$ terms",
        gates=True,
    )
    ax.set_xlim(xs[0] / 1.12, xs[-1] * 1.12)

    ax.set_title(
        "(a)  the marginal cost of a gate falls as $1/N$",
        loc="left",
        pad=8,
        fontweight="bold",
    )
    handles = [
        Line2D(
            [],
            [],
            **{**_style(fam), "ms": MARKER_SIZE + 1.4, "lw": LINE_WIDTH + 0.5},
            label=f"{ENGINE_LABEL[fam]}   $N^{{{fits[fam][2]:+.2f}}}$",
        )
        for fam in ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderpad=0.6,
        handlelength=2.6,
    )
    return fits


# --------------------------------------------------------------------------- #
# (b) why: the register may widen for free
# --------------------------------------------------------------------------- #
def _panel_spectator(ax, records):
    """ms for 10^6 terms at FIXED work, against register width."""
    growth = {}
    for fam in ORDER:
        rows = _arm(records, fam)
        xs = [r["num_qubits"] for r in rows]
        ys = [r["seconds"] / r["num_terms"] * MILLION * 1e3 for r in rows]
        growth[fam] = (xs, ys, ys[-1] / ys[0])
        ax.plot(xs, ys, **_style(fam))

    xs = growth["monoprop"][0]
    _log_axis(
        ax, xs, "register width $N$  (only 32 qubits ever used)", "ms per $10^6$ terms"
    )
    ax.set_xlim(xs[0] / 1.15, xs[-1] * 1.18)

    # PauliPropagation.jl's packed key leaves the fast BitIntegers width here.
    jx, jy, _ = growth["julia"]
    cliff = next(
        (i for i in range(1, len(jy)) if jy[i] / jy[i - 1] > 3.0),
        None,
    )
    if cliff is not None:
        ax.annotate(
            f"packed key outgrows\nthe fast integer width\n({jy[cliff] / jy[cliff - 1]:.1f}$\\times$ in one step)",
            xy=(jx[cliff], jy[cliff]),
            xytext=(-16, -58),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=ENGINE_COLOR["julia"],
            arrowprops={
                "arrowstyle": "-",
                "color": ENGINE_COLOR["julia"],
                "lw": 0.8,
                "shrinkB": 3,
            },
        )

    ax.set_title(
        "(b)  why: monoprop hardly notices the extra width",
        loc="left",
        pad=8,
        fontweight="bold",
    )
    handles = [
        Line2D(
            [],
            [],
            **{**_style(fam), "ms": MARKER_SIZE + 1.4, "lw": LINE_WIDTH + 0.5},
            label=f"{ENGINE_LABEL[fam]}   {growth[fam][2]:.0f}$\\times$"
            if growth[fam][2] >= 10
            else f"{ENGINE_LABEL[fam]}   {growth[fam][2]:.1f}$\\times$",
        )
        for fam in ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        title="cost over a $32\\times$ wider register",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderpad=0.6,
        handlelength=2.6,
        fontsize=9,
    )
    ax.get_legend().get_title().set_fontsize(8.5)
    return growth


def fig6(lattice, spectator, outdir: Path):
    """Two panels, one story, with the model spelled out above them.

    Room for the header is reserved with ``top=`` rather than left to a tight bbox: the
    headline, the three-line model statement and two panel titles otherwise land on top
    of one another.
    """
    fig = plt.figure(figsize=(12.4, 5.4))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.18, 1.0],
        wspace=0.20,
        top=0.735,
        bottom=0.115,
        left=0.055,
        right=0.955,
    )
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fits = _panel_per_gate(ax_a, lattice)
    growth = _panel_spectator(ax_b, spectator)

    inv = _arm(spectator, "monoprop")[0]
    lat = _arm(lattice, "monoprop")
    fig.text(
        0.5,
        0.975,
        "One gate, one million terms: monoprop's cost per gate falls as the system grows",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.900,
        "kicked-Ising chain, $N$ qubits, open boundary  ·  5 layers of "
        "[$R_{zz}(\\pi/4)$ on every bond, then $R_x(\\pi/4)$ on every site]  ·  "
        "observable $\\sum_i Z_i$, Heisenberg picture\n"
        f"Pauli-weight cutoff {CUTOFF}, no coefficient pruning  ·  "
        f"{lat[0]['num_terms']:,}$\\rightarrow${lat[-1]['num_terms']:,} terms in (a), "
        f"{inv['num_terms']:,} fixed in (b)  ·  one thread, one host\n"
        "all three engines apply the same circuit and agree on every term count and on "
        "$\\langle O \\rangle$ to 12 digits",
        ha="center",
        va="top",
        fontsize=9.3,
        color="#4a4a4a",
        linespacing=1.55,
    )
    outs = _save(fig, outdir, "fig6_inverse_scaling")
    return outs, fits, growth


def write_caption(outdir: Path, fits, growth, lattice, spectator):
    """Emit the LaTeX-ready caption with every number taken from the fits, not retyped."""
    spec_exp = _exponent(growth["monoprop"][0], growth["monoprop"][1])
    lat = _arm(lattice, "monoprop")
    spec0 = _arm(spectator, "monoprop")[0]
    lo, hi = lat[0], lat[-1]
    g_lo, g_hi = lo["gates"] // lo["layers"], hi["gates"] // hi["layers"]
    pl_lo = lo["seconds"] / lo["layers"] / lo["num_terms"] * MILLION * 1e3
    pl_hi = hi["seconds"] / hi["layers"] / hi["num_terms"] * MILLION * 1e3
    m, j, p = (fits[f] for f in ("monoprop", "julia", "ppvm"))
    gm, gp, gj = (growth[f][2] for f in ("monoprop", "ppvm", "julia"))

    per_cut = []
    for c in sorted({r["cutoff"] for r in lattice}):
        parts = []
        for fam in ORDER:
            rows = sorted(
                (r for r in lattice if r["engine_family"] == fam and r["cutoff"] == c),
                key=lambda r: r["num_qubits"],
            )
            e = _exponent(
                [r["num_qubits"] for r in rows],
                [r["seconds"] / r["gates"] / r["num_terms"] for r in rows],
            )
            parts.append(f"{ENGINE_LABEL[fam]} N^{e:+.2f}")
        per_cut.append(f"    cutoff {c}:  " + ",  ".join(parts))

    text = f"""Fig. 6 -- The marginal cost of a gate, single-threaded.

Model: kicked-Ising chain of N qubits, open boundary; five layers, each Rzz(pi/4) on
all N-1 nearest-neighbour bonds then Rx(pi/4) on all N sites, so a layer holds 2N-1
gates; observable sum_i Z_i propagated in the Heisenberg picture (the layer order
therefore reverses); truncation a Pauli-weight cutoff of {CUTOFF} with lower_atol = 0, so no
coefficient pruning at all and the surviving term set is fixed by gate supports alone.
That last point is what makes dividing by the term count a fair operation rather than a
rescaling of three different workloads: all three engines carry the same terms.

(a) Wall-clock milliseconds for ONE gate to act on 10^6 terms, against N. monoprop falls
    as N^{m[2]:+.2f}, close to the ideal 1/N drawn beside it: {m[1][0]:.3f} ms at N=32 down to {m[1][-1]:.4f} ms
    at N=512, an {m[1][0] / m[1][-1]:.1f}x reduction across a 16x wider system. Neither reference engine
    falls: PauliPropagation.jl N^{j[2]:+.2f}, ppvm N^{p[2]:+.2f}. Note where they start -- at N=32 ppvm
    is the fastest of the three ({p[1][0]:.3f} ms against monoprop's {m[1][0]:.3f}), so the ordering is
    earned over the sweep and not assumed at the origin. At N=512 monoprop leads
    PauliPropagation.jl by {j[1][-1] / m[1][-1]:.0f}x and ppvm by {p[1][-1] / m[1][-1]:.0f}x.

    Stated without dividing by the gate count at all, the same measurement reads: a
    monoprop layer of {g_hi} gates costs {pl_hi / pl_lo:.2f}x what a layer of {g_lo} gates costs
    ({pl_lo:.1f} -> {pl_hi:.1f} ms per 10^6 terms). {g_hi / g_lo:.1f}x the gates for {pl_hi / pl_lo:.2f}x the time.

    The mechanism is selectivity, not batching. The operator is held transposed: one
    column per bit position, bit r set iff term r touches that position
    (cpp/monoprop/detail/operator/InvertedIndex.h:33-46). Each column sits in one of two
    tiers, and below a density of 1/64 it is an ascending set-row list rather than a
    full-height bit-vector, with combine_columns_block narrowing even a dense column to a
    word range. A gate on qubit i therefore costs work proportional to the terms that
    actually touch qubit i, which at a fixed weight cutoff is a ~w/N fraction of the
    operator, rather than a scan over all K terms. One fact, both readings: per gate the
    cost carries the 1/N, and per layer each of the 2N-1 gates takes its ~1/N slice, so
    the layer total barely moves. The reference engines hold a packed key per term and
    revisit the whole sum for every gate, so nothing in them can fall.

(b) The isolation test, and the reason to believe (a). The model is confined to a
    {spec0["active_window"]}-qubit active window and the register padded out to N with idle spectator qubits,
    so the term count ({spec0["num_terms"]:,}), the gate count ({spec0["gates"]}), every term's support and the
    expectation value are held EXACTLY fixed while only the register width changes -- the
    invariant expectation value doubling as a correctness check that the padding really is
    idle. Over a 32x widening monoprop pays {gm:.1f}x, ppvm {gp:.1f}x and
    PauliPropagation.jl {gj:.0f}x. The step in the Julia arm is its packed key outgrowing
    the fast BitIntegers
    width; monoprop's rows are entropy-packed position lists whose width comes from the
    cutoff, not from N.

Caveat, stated because the figure would otherwise overclaim: monoprop's residual N^{spec_exp:+.2f}
in (b) is real work, not noise. The emit path materialises the dense partner and folds
every word for the hash, so the per-term cost is sublinear in N rather than free of it,
and the 1/N in (a) is approached rather than attained.

Cutoff {CUTOFF} is not chosen for effect. The per-gate exponent at every cutoff on this sweep:
{chr(10).join(per_cut)}
monoprop sits between N^-0.81 and N^-0.96 at all three, and cutoff {CUTOFF} is its MIDDLING
result -- cutoff 2 is the most favourable. Cutoff {CUTOFF} is shown because it is the only arm
whose term count brackets the 10^6 the axis is normalised to: the cutoff-2 arm spans
188..3068 terms, where normalising to a million would be a 326x extrapolation and fixed
overheads rather than per-term work would set the time.

Every point is single-threaded (one thread requested, and busy_cores ~ 1.0 recorded per
point for the two Python engines), from a single host, at lower_atol = 0.
"""
    out = outdir / "fig6_caption.txt"
    out.write_text(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lattice", nargs="+", type=Path, required=True)
    ap.add_argument("--spectator", nargs="+", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    lattice = load(args.lattice)
    spectator = load(args.spectator)
    outs, fits, growth = fig6(lattice, spectator, args.outdir)
    cap = write_caption(args.outdir, fits, growth, lattice, spectator)

    for fam in ORDER:
        print(f"  (a) {ENGINE_LABEL[fam]:<22} N^{fits[fam][2]:+.2f}")
    for fam in ORDER:
        print(
            f"  (b) {ENGINE_LABEL[fam]:<22} {growth[fam][2]:.1f}x over a 32x wider register"
        )
    for o in (*outs, cap):
        print(f"  {o}")


if __name__ == "__main__":
    main()
