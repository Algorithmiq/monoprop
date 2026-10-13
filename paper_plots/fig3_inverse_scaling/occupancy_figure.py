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

"""Appendix figure: how the Fig. 3 operators spread over the anticommutation index.

A gate reads the index lists of the positions it acts on, so its cost is the length of those
lists. The longest list over the mean, c, bounds the worst single gate (paper, Definition
``def:even-occupancy``). What a layer pays is the total over its gates, and since every
position is read by a bounded number of gates per layer, that total is a bounded multiple of the
whole index: the average gate reads lists of about the mean length, whatever c is. This draws
both from the list lengths measured after every layer (``scripts/occupancy.py``):

* (a)-(c) the list length of every site of the Hubbard lattice, row by row, over the mean,
  after two layers, on the three largest lattices measured; shade is the distance from the
  open edge;
* (d) against N, at the medium depth of each model: c, and the average list a gate reads over
  one layer, both over the mean list.

    python occupancy_figure.py --merge <occupancy run dir> ...   # writes data/occupancy.jsonl
    python occupancy_figure.py                                    # draws from it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import make_figure as mf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
from merge_octave_runs import CUTOFF, DEPTHS, NS, cutoff_of

OCCUPANCY = "occupancy.jsonl"
STEM = "fig3_occupancy"
# The Hubbard lattice of each mode count, as the campaign builds it.
LATTICE = {
    32: (4, 4),
    64: (8, 4),
    128: (8, 8),
    256: (16, 8),
    512: (16, 16),
    1024: (32, 16),
}
FIGSIZE = (7.0, 2.4)
# The lattices of panels (a)-(c), at the medium Hubbard depth.
SITE_PANELS = (128, 256, 512)
# Sites from the open edge, in bands of two (the profile steps every two sites): light at the
# edge, dark in the interior.
EDGE_BANDS = ((0, 1), (2, 3), (4, 5), (6, 7))
EDGE_MIX = (0.62, 0.42, 0.2, 0.0)
MODEL_NAME = {"pauli": "kicked Ising", "majorana": "2D Hubbard"}
LONGEST = "#8a8a8a"


def _edge_distance(lx, ly):
    x, y = np.arange(lx * ly) % lx, np.arange(lx * ly) // lx
    return x, y, np.minimum.reduce([x, lx - 1 - x, y, ly - 1 - y])


def gate_read(row):
    """The average list a gate of one layer reads, over the mean list.

    Kicked Ising: X_q reads index 2q+1, Z_q Z_{q+1} reads 2q and 2q+2. Hubbard: the hops of
    a bond read the four indices of each of its two sites over the two spins, and the
    interaction the four of its site. Each read is weighted by the gates that make it.
    """
    lengths = np.asarray(row["list_lengths"], float)
    n = row["num_qubits"]
    if row["basis"] == "pauli":
        degree = np.full(n, 2.0)
        degree[[0, -1]] = 1
        reads = lengths[1::2].sum() + (degree * lengths[0::2]).sum()
        positions = n + 2 * (n - 1)
    else:
        lx, ly = LATTICE[n]
        x, y, _ = _edge_distance(lx, ly)
        degree = (x > 0).astype(float) + (x < lx - 1) + (y > 0) + (y < ly - 1)
        per_site = lengths.reshape(lx * ly, 4).sum(axis=1)
        reads = ((degree + 1) * per_site).sum()
        positions = 4 * (degree + 1).sum()
    return float(reads / positions / lengths.mean())


def merge(run_dirs, out: Path) -> None:
    """Rows at the Fig. 3 cutoffs and depths, one per (model, N, layers); refuse duplicates.

    Every row gains ``gate_read_over_mean``; the per-position list lengths are kept for the
    Hubbard rows only, which panels (a)-(c) draw.
    """
    rows = {}
    for run_dir in run_dirs:
        for f in sorted((run_dir / "occupancy").glob("*.jsonl")):
            for raw in f.read_text().splitlines():
                if not raw.strip():
                    continue
                r = json.loads(raw)
                if (
                    cutoff_of(r) != CUTOFF[r["basis"]]
                    or r["layers"] not in DEPTHS[r["basis"]]
                ):
                    continue
                key = (r["basis"], r["num_qubits"], r["layers"])
                if key in rows:
                    raise SystemExit(f"{key}: two rows ({rows[key]['source']}, {f})")
                r["gate_read_over_mean"] = gate_read(r)
                if r["basis"] == "pauli":
                    r.pop("list_lengths")
                rows[key] = {**r, "source": str(f)}
    out.write_text("".join(json.dumps(rows[k]) + "\n" for k in sorted(rows)))
    print(f"wrote {len(rows)} rows to {out}")


def _band_colors():
    blue = np.array(to_rgb(mf.ENGINE_COLOR["monoprop"]))
    return [to_hex(blue + (1 - blue) * t) for t in EDGE_MIX]


def site_panel(ax, row, letter):
    """One bar per site, row by row, over the mean; the mean as the dashed line."""
    lx, ly = LATTICE[row["num_qubits"]]
    per_site = np.asarray(row["list_lengths"], float).reshape(lx * ly, 4).sum(axis=1)
    ratio = per_site / per_site.mean()
    _, _, dist = _edge_distance(lx, ly)
    colors = _band_colors()
    band = np.minimum(dist // 2, len(colors) - 1)
    ax.bar(
        np.arange(lx * ly),
        ratio,
        width=1.0,
        color=[colors[b] for b in band],
        lw=0,
        zorder=2,
    )
    ax.axhline(1, color=mf.GUIDE, lw=mf.GUIDE_WIDTH, ls=mf.GUIDE_DASH, zorder=3)
    top = int(ratio.argmax())
    ax.annotate(
        f"$c={ratio.max():.2f}$",
        xy=(top, ratio.max()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=mf.INK,
    )
    # A tick at the start of every row of the lattice.
    ax.set_xticks(np.arange(0, lx * ly, lx), labels=[])
    ax.tick_params(axis="x", length=2)
    ax.set_xlim(-0.5, lx * ly - 0.5)
    ax.set_ylim(0, 2.15)
    ax.grid(visible=True, axis="y", color=mf.GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"({letter}) {lx}×{ly}, $N={row['num_qubits']}$",
        loc="left",
        fontsize=7.8,
        color=mf.INK,
    )


def read_panel(ax, rows):
    """c and the average list a gate reads, over the mean, at each model's medium depth."""
    blue = mf.ENGINE_COLOR["monoprop"]
    for kind, marker in mf.MODEL_STYLE.items():
        layers = DEPTHS[kind][1]
        mine = sorted(
            (r for r in rows if r["basis"] == kind and r["layers"] == layers),
            key=lambda r: r["num_qubits"],
        )
        n = [r["num_qubits"] for r in mine]
        filled = kind == "pauli"
        for key, color, ls, z in (
            ("c_max_over_mean", LONGEST, (0, (3, 1.6)), 2),
            ("gate_read_over_mean", blue, "-", 3),
        ):
            ax.plot(
                n,
                [r[key] for r in mine],
                color=color,
                ls=ls,
                lw=mf.LINE_WIDTH,
                marker=marker,
                ms=mf.MARKER_SIZE - 0.3,
                mfc=color if filled else "white",
                mec="white" if filled else color,
                mew=mf.RING if filled else mf.MARKER_EDGE,
                zorder=z,
            )
        # Each longest-list line is named at its far end from the other.
        first, last = (mine[0], mine[-1])
        anchor, ha, dx = (first, "left", 4) if filled else (last, "right", -6)
        ax.annotate(
            f"{MODEL_NAME[kind]}, longest ($c$)",
            xy=(anchor["num_qubits"], anchor["c_max_over_mean"]),
            xytext=(dx, 4 if filled else 0),
            textcoords="offset points",
            ha=ha,
            va="bottom" if filled else "center",
            fontsize=6.4,
            color=mf.INK,
        )
    ax.axhline(1, color=mf.GUIDE, lw=mf.GUIDE_WIDTH, ls=mf.GUIDE_DASH, zorder=1)
    ax.annotate(
        "both: the list a gate reads,\naveraged over a layer",
        xy=(min(NS), 0.99),
        xytext=(-2, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.4,
        color=mf.INK,
        linespacing=1.1,
    )
    ax.set_ylim(0.82, 2.15)
    mf._log2_axis(ax, list(NS))
    mf._frame(ax)
    ax.set_title("(d) Longest and read", loc="left", fontsize=7.8, color=mf.INK)
    ax.set_xlabel("$N$", labelpad=1.5)
    ax.set_ylabel("list length / mean", labelpad=2)


def draw(rows, outdir: Path):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.05)
    grid = fig.add_gridspec(1, 4, width_ratios=(0.85, 1.0, 1.35, 1.35))
    layers = DEPTHS["majorana"][1]
    by_n = {
        r["num_qubits"]: r
        for r in rows
        if r["basis"] == "majorana" and r["layers"] == layers
    }
    axes = []
    for j, n in enumerate(SITE_PANELS):
        ax = fig.add_subplot(grid[0, j], sharey=axes[0] if axes else None)
        site_panel(ax, by_n[n], "abc"[j])
        axes.append(ax)
    axes[0].set_ylabel(f"site's lists / mean, {layers} layers", labelpad=2)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[1].set_xlabel("sites in row order (a tick starts each row)", labelpad=1.5)
    axes[0].legend(
        [Patch(color=c, lw=0) for c in _band_colors()],
        [f"{lo}–{hi}" for lo, hi in EDGE_BANDS],
        title="sites from edge",
        title_fontsize=6.0,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=6.0,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.6,
        borderaxespad=0.2,
        alignment="left",
    )
    read_panel(fig.add_subplot(grid[0, 3]), rows)
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext, dpi in (("pdf", None), ("png", mf.PNG_DPI)):
        out = outdir / f"{STEM}.{ext}"
        fig.savefig(out, dpi=dpi, metadata={"CreationDate": None})
        outs.append(out)
    plt.close(fig)
    return outs


def summary(rows) -> str:
    """The numbers the appendix quotes: c and the average read, per model, depth and N."""
    lines = []
    for kind in ("pauli", "majorana"):
        mine = [r for r in rows if r["basis"] == kind]
        for key, name in (("c_max_over_mean", "c"), ("gate_read_over_mean", "read")):
            values = [r[key] for r in mine]
            lines.append(f"{kind} {name}: {min(values):.3f} to {max(values):.3f}")
        for layers in DEPTHS[kind]:
            pts = sorted(
                (r["num_qubits"], r["c_max_over_mean"], r["gate_read_over_mean"])
                for r in mine
                if r["layers"] == layers
            )
            lines.append(
                f"  {layers} layers: "
                + ", ".join(f"N={n} c={c:.3f} read={g:.3f}" for n, c, g in pts)
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge", nargs="+", type=Path, metavar="RUN_DIR")
    ap.add_argument("--outdir", type=Path, default=mf.HERE / "figures")
    args = ap.parse_args()
    path = mf.DATA / OCCUPANCY
    if args.merge:
        merge(args.merge, path)
    rows = mf.load(path)
    print(*draw(rows, args.outdir), sep="\n")
    print(summary(rows))


if __name__ == "__main__":
    main()
