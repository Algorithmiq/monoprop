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

"""Fig. 5 -- where the cost of a term goes as the mode space widens, on one thread.

Figs. 1-4 measure the full-width kicked-Ising chain, where the term count K, the gate count
and the mode space N all grow together. That is the realistic workload, but it is the wrong
instrument for a claim about *per-term* cost: three N-dependencies are superposed in every
exponent and none of them can be read off alone.

This figure separates them with two sweeps that share an engine, a host and a thread.

**Panel (a), the idle-spectator sweep.** The kicked-Ising model is confined to an active
window of M qubits and the register is padded to N with qubits no gate and no observable
term ever touches. K, the gate count, the term supports and the expectation value are then
*identical at every N* -- verified, not assumed -- so the only variable left is the width of
the mode space, and the curve is per-term cost in N and nothing else. monoprop stores a term
as an entropy-packed position list whose width comes from the cutoff, and settles the
commute/anticommute question for 64 terms per word op against a transposed index, so its
cost should barely move; PauliPropagation.jl and ppvm carry a 2-bits-per-qubit packed key
per term, so theirs should grow with N. Guides: $N^0$ (total cost $\\propto K$) and $N^1$
($\\propto K N$).

**Panel (b), the full-width sweep, per gate.** Back on the real model, where the operator is
spread over all N modes and a layer is $\\Theta(N)$ gates. A gate on two modes reaches only
the terms that touch them -- about $K/N$ of the operator through the inverted index -- so
monoprop's per-gate cost should *fall* as $1/N$ while an engine that rescans the whole sum
pays $K$ per gate and $K N$ per layer. Guides: $N^{-1}$ and $N^{+1}$. This is the same claim
as panel (a), seen per gate instead of per term.

Encoding follows make_paper_figures.py exactly: colour = cutoff, line style and marker =
engine, so an engine looks the same here as in Figs. 1-4.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Importing this applies the shared rcParams and the Agg backend, and hands over the one
# definition of each engine's mark; the two figure scripts cannot drift apart on style.
from make_paper_figures import (
    CUTOFF_COLORS,
    ENGINE_LABEL,
    ENGINE_STYLE,
    GUIDE,
    PNG_DPI,
    _curve_style,
    load,
    plt,
)

ENGINE_ORDER = ("monoprop", "julia", "ppvm")


def per_term(records, fam, cutoff):
    """(N, seconds per term) for one engine and cutoff, ascending in N."""
    pts = sorted(
        (r["num_qubits"], r["seconds"] / r["num_terms"])
        for r in records
        if r["engine_family"] == fam and r["cutoff"] == cutoff and r["num_terms"]
    )
    return [x for x, _ in pts], [y for _, y in pts]


def per_term_relative(records, fam, cutoff):
    """Per-term cost divided by its own value at the narrowest register.

    Nine absolute curves at three cutoffs sit at nine different heights, because a bigger
    cutoff amortises the fixed per-layer work over more terms -- so the panel ends up
    showing that spread rather than the thing it is about. Dividing each curve by its own
    first point removes the cutoff offset and leaves only the quantity being claimed: how
    much more a term costs purely because the register got wider. Every curve then starts
    at 1, so the reference slopes are exact rather than eyeballed, and the engines separate
    into bands. The absolute per-term cost is Fig. 3's subject.
    """
    xs, ys = per_term(records, fam, cutoff)
    if not ys or ys[0] <= 0:
        return [], []
    return xs, [y / ys[0] for y in ys]


def per_gate_term(records, fam, cutoff):
    """(N, seconds per gate per term). Divides out both quantities that grow with N."""
    pts = sorted(
        (r["num_qubits"], r["seconds"] / r["num_terms"] / r["gates"])
        for r in records
        if r["engine_family"] == fam and r["cutoff"] == cutoff and r["num_terms"]
    )
    return [x for x, _ in pts], [y for _, y in pts]


def fit(xs, ys, nmin=0, nmax=1 << 30):
    """Least-squares log-log slope over nmin <= N <= nmax; None if under three points."""
    pts = [(x, y) for x, y in zip(xs, ys, strict=True) if nmin <= x <= nmax and y > 0]
    if len(pts) < 3:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    den = sum((x - mx) ** 2 for x in lx)
    return (
        None
        if den == 0
        else sum((x - mx) * (y - my) for x, y in zip(lx, ly, strict=True)) / den
    )


def check_invariants(records, label):
    """The spectator sweep is only meaningful if the padding really is idle.

    K, the gate count and the expectation value must each be one value across the whole
    sweep. If any of them moves, a gate or an observable term reached a spectator qubit and
    the panel is measuring the wrong thing, so say so loudly rather than plot it.
    """
    problems = []
    for fam in ENGINE_ORDER:
        for cutoff in sorted({r["cutoff"] for r in records}):
            rows = [
                r
                for r in records
                if r["engine_family"] == fam and r["cutoff"] == cutoff
            ]
            if not rows:
                continue
            for key, tol in (("num_terms", 0), ("gates", 0), ("expectation", 1e-9)):
                vals = [r[key] for r in rows]
                spread = max(vals) - min(vals)
                if abs(spread) > tol:
                    problems.append(
                        f"{label}: {fam} c{cutoff} {key} varies across N "
                        f"({min(vals)} .. {max(vals)})"
                    )
    return problems


# process_time() has a coarse tick, so cpu/wall is meaningless on a sub-millisecond run;
# below this the ratio says nothing about how many threads ran.
BUSY_MIN_SECONDS = 0.01


def report_threads(records, label):
    """A single-thread figure has to show it was one thread, not assert it.

    The number that carries the claim is the *maximum* busy_cores: at or below 1, no row
    used a second core. The minimum is only informative on runs long enough to time --
    process_time() has too coarse a tick to say anything about a sub-millisecond run.
    """
    busy = [r["busy_cores"] for r in records if r.get("busy_cores")]
    timed = [
        r["busy_cores"]
        for r in records
        if r.get("busy_cores") and r["seconds"] >= BUSY_MIN_SECONDS
    ]
    if busy:
        head = f"  {label}: busy_cores max {max(busy):.2f} over {len(busy)} rows"
        if timed:
            print(
                f"{head}; {min(timed):.2f}..{max(timed):.2f} over the {len(timed)} rows "
                f"longer than {BUSY_MIN_SECONDS * 1e3:.0f} ms"
            )
        else:
            print(f"{head} (all too short to time CPU reliably)")
    hosts = sorted({r.get("host", "?") for r in records})
    versions = sorted({str(r.get("library_version", "?")) for r in records})
    print(f"  {label}: hosts {hosts}, library versions {versions}")


def _guides(ax, xs, exponents, anchor_y=None, anchor_frac=0.35):
    """Faint reference slopes.

    With `anchor_y` the guides start from that exact ordinate at the leftmost N -- used on
    the relative panel, where every curve starts at 1 by construction, so a guide through
    that point is a true reference and not a decoration placed by eye. Without it they are
    floated a fraction of the way up the axis to frame the data instead of crossing it.
    """
    if not xs:
        return
    lo, hi = min(xs), max(xs)
    y0, y1 = ax.get_ylim()
    floated = math.exp(math.log(y0) + anchor_frac * (math.log(y1) - math.log(y0)))
    ya = floated if anchor_y is None else anchor_y
    for e in exponents:
        yb = ya * (hi / lo) ** e
        ax.plot([lo, hi], [ya, yb], color=GUIDE, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.annotate(
            f"$N^{{{e:+g}}}$".replace("+", ""),
            xy=(hi, yb),
            xytext=(-2, 3 if e >= 0 else -11),
            textcoords="offset points",
            color=GUIDE,
            fontsize=8.5,
            ha="right",
        )
    ax.set_ylim(y0, y1)


def _panel(ax, records, series, title, ylabel, guides, anchor_y=None):
    """One log-log panel: every engine x every cutoff, plus the reference slopes."""
    cutoffs = sorted({r["cutoff"] for r in records})
    fits, seen_x = {}, []
    for cutoff in cutoffs:
        color = CUTOFF_COLORS.get(cutoff, "#666666")
        for fam in ENGINE_ORDER:
            xs, ys = series(records, fam, cutoff)
            if not xs:
                continue
            ax.plot(xs, ys, **_curve_style(fam, color))
            seen_x += xs
            p = fit(xs, ys)
            if p is not None:
                fits[(cutoff, fam)] = p
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _guides(ax, seen_x, guides, anchor_y=anchor_y)
    ax.set_title(title, pad=8)
    ax.set_xlabel("number of qubits  $N$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.5, color="#d5d5d5", alpha=0.7)
    ax.set_axisbelow(True)
    return fits


def _legend(fig, records):
    """Two rows: colour = cutoff, then style = engine. Identity is never colour-alone."""
    cutoffs = sorted({r["cutoff"] for r in records})
    colour_keys = [
        plt.Line2D(
            [], [], color=CUTOFF_COLORS.get(c, "#666666"), lw=2.0, label=f"cutoff {c}"
        )
        for c in cutoffs
    ]
    engines = [f for f in ENGINE_ORDER if any(r["engine_family"] == f for r in records)]
    style_keys = [
        plt.Line2D(
            [],
            [],
            color="#555555",
            ls=ENGINE_STYLE[f][0],
            marker=ENGINE_STYLE[f][1],
            ms=4.0,
            markerfacecolor=("#555555" if f == "monoprop" else "white"),
            markeredgecolor="#555555",
            label=ENGINE_LABEL[f],
        )
        for f in engines
    ]
    fig.legend(
        handles=colour_keys,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=len(colour_keys),
        frameon=False,
        handlelength=2.2,
    )
    fig.legend(
        handles=style_keys,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(style_keys),
        frameon=False,
        handlelength=2.6,
    )


def fig5(spectator, lattice, outdir: Path, active_window):
    """The two panels, side by side, sharing the legend."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    a = _panel(
        axes[0],
        spectator,
        per_term_relative,
        f"(a) fixed operator, widening register ($M={active_window}$ active)",
        "time per term,  relative to $N=32$",
        [0, 1],
        anchor_y=1.0,
    )
    b = _panel(
        axes[1],
        lattice,
        per_gate_term,
        "(b) full-width chain, per gate",
        "time per gate per term  (s)",
        [-1, 1],
    )
    _legend(fig, spectator + lattice)
    fig.tight_layout(rect=(0, 0.15, 1, 0.98))
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext, dpi in (("pdf", None), ("png", PNG_DPI)):
        out = outdir / f"fig5_single_thread_per_term.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=dpi, metadata={"CreationDate": None})
        outs.append(out)
    plt.close(fig)
    return outs, a, b


def _span(fits, fam):
    """The fitted exponents for one engine across cutoffs, as a range string."""
    vals = sorted(v for (_, f), v in fits.items() if f == fam)
    if not vals:
        return "not measured"
    if len(vals) == 1:
        return f"$N^{{{vals[0]:+.2f}}}$"
    return f"$N^{{{vals[0]:+.2f}}}$ to $N^{{{vals[-1]:+.2f}}}$"


def _growth(records, fam):
    """Largest per-term slowdown from the narrowest to the widest register."""
    best = None
    for cutoff in sorted({r["cutoff"] for r in records}):
        _, ys = per_term(records, fam, cutoff)
        if len(ys) >= 2 and ys[0] > 0:
            g = ys[-1] / ys[0]
            best = g if best is None else max(best, g)
    return best


def write_caption(spectator, lattice, outdir: Path, active_window, fits_a, fits_b):
    """Write the LaTeX caption with every number computed from the plotted rows.

    Generated, not typed, for the same reason node_scaling/captions.py is: a hand-written
    percentage outlives the measurement it described. Edit this function, never the .txt.
    """
    cutoffs = sorted({r["cutoff"] for r in spectator})
    ks = {}
    for c in cutoffs:
        rows = [r for r in spectator if r["cutoff"] == c]
        ks[c] = rows[0]["num_terms"] if rows else 0
    ns = sorted({r["num_qubits"] for r in spectator})
    nl = sorted({r["num_qubits"] for r in lattice})
    engines = [
        f for f in ENGINE_ORDER if any(r["engine_family"] == f for r in spectator)
    ]
    hosts = sorted({r.get("host", "?") for r in spectator + lattice})
    max_busy = max(r["busy_cores"] for r in spectator + lattice if r.get("busy_cores"))
    k_list = ", ".join(f"{ks[c]:,}".replace(",", "\\,") for c in cutoffs)
    jl_growth = _growth(spectator, "julia")

    text = f"""Figure caption -- Fig. 5, single-thread per-term scaling
=======================================================

Generated by make_single_thread_figure.py from the rows the figure is drawn from. Every
number here is computed, never typed; edit write_caption() in that script, not this file.


Fig. 5  (fig5_single_thread_per_term)
-------------------------------------
Where the cost of a Pauli term goes as the mode space widens, on **one thread**
(measured, not assumed: `busy_cores` peaks at {max_busy:.2f} over every row).
**(a)** The operator is held fixed and the register is widened. The kicked-Ising model is
confined to an active window of $M={active_window}$ qubits and the register padded to $N$
with qubits no gate and no observable term touches, so the term count, the gate count, the
term supports and the expectation value are identical at every $N$ -- and identical across
all {len(engines)} engines ({k_list} terms at cutoff
{", ".join(str(c) for c in cutoffs)}). The only variable left is the width of the mode
space, so each curve is per-term cost in $N$ and nothing else; each is divided by its own
value at $N={ns[0]}$, which removes the per-cutoff offset and makes the $N^0$ and $N^1$
guides exact. monoprop is flat to {_span(fits_a, "monoprop")}: a term is a position list
whose width comes from the cutoff, and one word operation settles the
commute/anticommute question for 64 terms against a transposed index.
PauliPropagation.jl climbs {_span(fits_a, "julia")} (up to
{jl_growth:.0f}$\\times$ across the sweep, with the step where its packed key outgrows
its fast BitInteger width) and ppvm {_span(fits_a, "ppvm")}; both carry a
2-bits-per-qubit key per term and pay for the width they declare.
**(b)** The same engines on the full-width chain, $N={nl[0]}$ to ${nl[-1]}$, where the
operator is spread over all $N$ modes and a layer is $\\Theta(N)$ gates; the range stops
below the key-width cliff, so no engine is measured across a discontinuity. Cost is
divided by both the term count and the gate count. A gate on two modes reaches only the terms
touching them, about $K/N$ of the operator through the inverted index, so monoprop's
per-gate cost *falls*, at {_span(fits_b, "monoprop")} against the $N^{{-1}}$ guide, while
PauliPropagation.jl at {_span(fits_b, "julia")} and ppvm at {_span(fits_b, "ppvm")} stay
near $N^0$ -- $K$ per gate, so $K N$ per layer. The cutoff-2 arm is the least reliable in
this panel: it retains only {ks[cutoffs[0]]:,} terms at $N={nl[0]}$, few enough that fixed
per-layer cost is a visible share of a millisecond-scale measurement. Colour encodes the
weight cutoff; the line style and marker encode the engine, as in Figs. 1--4.

Measured on {", ".join(hosts)}, not on the Leonardo nodes Figs. 1--4 use: this is a shape
measurement (an exponent in $N$), and the two datasets are never drawn in one panel.
"""
    out = outdir / "fig5_caption.txt"
    out.write_text(text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectator",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL from the --active-window sweeps (one file per engine)",
    )
    parser.add_argument(
        "--lattice",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL from the full-width sweeps (one file per engine)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    spectator, lattice = load(args.spectator), load(args.lattice)
    windows = {r.get("active_window") for r in spectator}
    if len(windows) != 1 or None in windows:
        msg = f"--spectator must be one active window, got {windows}"
        raise SystemExit(msg)
    active_window = windows.pop()

    print("provenance:")
    report_threads(spectator, "spectator")
    report_threads(lattice, "lattice")

    problems = check_invariants(spectator, "spectator")
    if problems:
        raise SystemExit("\n".join(["spectator sweep is not invariant:", *problems]))
    print(f"  spectator: K, gates and expectation invariant in N (M={active_window})")

    outs, a, b = fig5(spectator, lattice, args.outdir, active_window)
    outs.append(write_caption(spectator, lattice, args.outdir, active_window, a, b))
    print("wrote:")
    for o in outs:
        print(f"  {o}")

    for label, fits, ideal in (
        ("(a) time/term", a, "0 (cost ∝ K) vs 1 (∝ K·N)"),
        ("(b) time/gate/term", b, "-1 (K/N per gate) vs +1 (K·N per layer)"),
    ):
        print(f"\nfitted exponents, {label}  [ideal: {ideal}]")
        for (cutoff, fam), p in sorted(fits.items()):
            print(f"  c{cutoff:<2} {ENGINE_LABEL[fam]:<20} N^{p:+.2f}")


if __name__ == "__main__":
    main()
