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

"""Appendix figure: how evenly the Fig. 3 operators are spread over the anticommutation index.

The O(K/N) bound holds for an evenly supported operator, whose longest index list is within a
constant factor c of the mean (paper, Definition ``def:even-occupancy``). This measures c on
the operators Fig. 3 times, after every layer (``scripts/occupancy.py``):

* (a), (b) c = longest list / mean list against N, at the three Fig. 3 depths of each model;
* (c) where the long lists are on the Hubbard lattice: the mean list length at each distance
  from the open edge, relative to the mean over the lattice, at the medium depth.

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
from matplotlib.lines import Line2D
from merge_octave_runs import CUTOFF, DEPTHS, MAX_N, NS, cutoff_of

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
FIGSIZE = (7.0, 2.3)
TITLE = {"pauli": "(a) Kicked Ising chain", "majorana": "(b) 2D Hubbard lattice"}
AXIS = {"pauli": "qubits $N$", "majorana": "modes $N$"}


def merge(run_dirs, out: Path) -> None:
    """Rows at the Fig. 3 cutoffs and depths, one per (model, N, layers); refuse duplicates.

    The per-position list lengths are kept for the Hubbard rows only, which panel (c) maps onto
    the lattice; a kicked-Ising row keeps its summary.
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
                if r["basis"] == "pauli":
                    r.pop("list_lengths")
                rows[key] = {**r, "source": str(f)}
    out.write_text("".join(json.dumps(rows[k]) + "\n" for k in sorted(rows)))
    print(f"wrote {len(rows)} rows to {out}")


def edge_profile(row):
    """{distance from the open edge: mean list length of a site there / mean over sites}."""
    lx, ly = LATTICE[row["num_qubits"]]
    # Four Majorana positions per site: two spin modes, interleaved, two Majoranas each.
    per_site = np.asarray(row["list_lengths"], float).reshape(lx * ly, 4).sum(axis=1)
    x, y = np.arange(lx * ly) % lx, np.arange(lx * ly) // lx
    dist = np.minimum.reduce([x, lx - 1 - x, y, ly - 1 - y])
    return {
        int(d): float(per_site[dist == d].mean() / per_site.mean())
        for d in np.unique(dist)
    }


def c_panel(ax, rows, kind):
    for i, layers in enumerate(DEPTHS[kind]):
        pts = sorted(
            (r["num_qubits"], r["c_max_over_mean"])
            for r in rows
            if r["basis"] == kind and r["layers"] == layers
        )
        style = mf._style("monoprop", i)
        if i != 1:
            style.update(marker="none", lw=mf.LINE_WIDTH - 0.25)
        ax.plot(*zip(*pts, strict=True), **style, solid_capstyle="round")
    ax.axhline(1, color=mf.GUIDE, lw=mf.GUIDE_WIDTH, ls=mf.GUIDE_DASH, zorder=1)
    grey = "#555555"
    ax.legend(
        [
            Line2D([], [], color=mf.shade(grey, i), lw=mf.LINE_WIDTH + 0.4)
            for i in range(3)
        ],
        [str(layers) for layers in DEPTHS[kind]],
        title="layers",
        title_fontsize=6.2,
        loc="upper right" if kind == "pauli" else "upper left",
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.3,
        alignment="left",
    )
    ax.set_ylim(0.9, 2.0)
    mf._log2_axis(ax, list(NS))
    mf._frame(ax)
    ax.set_title(TITLE[kind], loc="left", fontsize=7.8, color=mf.INK)
    ax.set_xlabel(AXIS[kind], labelpad=1.5)


def profile_panel(ax, rows):
    layers = DEPTHS["majorana"][1]
    mine = sorted(
        (r for r in rows if r["basis"] == "majorana" and r["layers"] == layers),
        key=lambda r: r["num_qubits"],
    )
    mine = [r for r in mine if r["num_qubits"] >= 128]
    handles, labels = [], []
    for i, r in enumerate(mine):
        prof = edge_profile(r)
        # Lattices from small to large take the depth ramp from light to dark.
        tone = mf.shade(
            mf.ENGINE_COLOR["monoprop"], round(2 * i / max(len(mine) - 1, 1))
        )
        (line,) = ax.plot(
            list(prof),
            list(prof.values()),
            color=tone,
            lw=mf.LINE_WIDTH,
            marker="o",
            ms=mf.MARKER_SIZE - 0.6,
            mfc=tone,
            mec="white",
            mew=mf.RING,
        )
        lx, ly = LATTICE[r["num_qubits"]]
        handles.append(line)
        labels.append(f"{lx}×{ly}")
    ax.axhline(1, color=mf.GUIDE, lw=mf.GUIDE_WIDTH, ls=mf.GUIDE_DASH, zorder=1)
    ax.legend(
        handles,
        labels,
        title="lattice",
        title_fontsize=6.2,
        loc="lower right",
        frameon=False,
        fontsize=6.2,
        handlelength=1.6,
        labelspacing=0.25,
        borderaxespad=0.3,
        alignment="left",
    )
    ax.set_xlim(-0.5, 9)
    ax.set_xticks(range(0, 9, 2))
    mf._frame(ax)
    ax.set_title(
        f"(c) Hubbard, {layers} layers: by site", loc="left", fontsize=7.8, color=mf.INK
    )
    ax.set_xlabel("sites from the open edge", labelpad=1.5)
    ax.set_ylabel("site's lists / mean", labelpad=2)


def draw(rows, outdir: Path):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.15, 1))
    for j, kind in enumerate(("pauli", "majorana")):
        ax = fig.add_subplot(grid[0, j])
        c_panel(ax, rows, kind)
        if j == 0:
            ax.set_ylabel("longest list / mean, $c$", labelpad=2)
    profile_panel(fig.add_subplot(grid[0, 2]), rows)
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext, dpi in (("pdf", None), ("png", mf.PNG_DPI)):
        out = outdir / f"{STEM}.{ext}"
        fig.savefig(out, dpi=dpi, metadata={"CreationDate": None})
        outs.append(out)
    plt.close(fig)
    return outs


def summary(rows) -> str:
    """The numbers the appendix quotes: the largest c per model, and c at the largest N."""
    lines = []
    for kind in ("pauli", "majorana"):
        mine = [r for r in rows if r["basis"] == kind]
        worst = max(mine, key=lambda r: r["c_max_over_mean"])
        lines.append(
            f"{kind}: largest c = {worst['c_max_over_mean']:.3f} "
            f"(N={worst['num_qubits']}, {worst['layers']} layers)"
        )
        for layers in DEPTHS[kind]:
            pts = sorted(
                (r["num_qubits"], r["c_max_over_mean"])
                for r in mine
                if r["layers"] == layers
            )
            cap = MAX_N.get((kind, layers))
            lines.append(
                f"  {layers} layers: "
                + ", ".join(f"N={n} {c:.3f}" for n, c in pts)
                + (f" (timed to N={cap})" if cap else "")
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
