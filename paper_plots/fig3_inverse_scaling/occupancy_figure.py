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

monoprop applies a gate one generator monomial at a time, and each scan reads whole index
lists, so its cost is the length of the lists it reads. The longest list over the mean, c,
bounds the worst single scan (paper, Definition ``def:even-occupancy``). What a layer pays is
the total over its scans, and every list is read a fixed number of times per layer
(``scripts/read_weights.py``), so the average scan reads lists of about the mean length.

At fixed depth a list depends only on the circuit within the light cone of its position. On the
Hubbard torus every site is alike and every list is the same length. On the open kicked-Ising
chain only a strip at each end is shorter; every list beyond it is the same bulk list, for
every N. This draws, from the list lengths measured after every layer
(``scripts/occupancy.py``):

* (a) kicked Ising: each qubit's lists against its distance from the chain's end, over the bulk
  list, at each depth on the longest chain measured (every shorter chain falls on the same
  curve, which the build checks);
* (b) against N, at the medium depth of each model: c, and the average list a scan reads over
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
from merge_octave_runs import CUTOFF, DEPTHS, NS, cutoff_of

OCCUPANCY = "occupancy.jsonl"
WEIGHTS = "read_weights.json"
STEM = "fig3_occupancy"
FIGSIZE = (7.0, 2.35)
# Panel (a) shows the first qubits of the chain, past the widest strip (2 * 13 - 2 = 24).
PROFILE_QUBITS = 30
LONGEST = "#6b6b6b"
MODEL_NAME = {"pauli": "kicked Ising, open chain", "majorana": "2D Hubbard, torus"}


def qubit_lists(row):
    """Each qubit's two lists added (monoprop stores qubit q as slots 2q and 2q+1)."""
    return np.asarray(row["list_lengths"], float).reshape(-1, 2).sum(axis=1)


def strip(row):
    """(bulk list per qubit, strip width): the longest per-qubit total, and the first qubit
    from either end that holds it."""
    per = qubit_lists(row)
    n = len(per)
    dist = np.minimum(np.arange(n), n - 1 - np.arange(n))
    return per.max(), int(dist[per == per.max()].min())


def check_profiles(rows):
    """Every chain longer than twice its strip has the same per-qubit lists from either end.

    Returns {layers: (bulk, strip width, the N checked)}.
    """
    out = {}
    for layers in DEPTHS["pauli"]:
        mine = sorted(
            (r for r in rows if r["basis"] == "pauli" and r["layers"] == layers),
            key=lambda r: r["num_qubits"],
        )
        longest = qubit_lists(mine[-1])
        bulk, width = strip(mine[-1])
        checked = []
        for r in mine:
            per = qubit_lists(r)
            if len(per) < 2 * width + 2:
                continue
            half = len(per) // 2
            if not (
                np.array_equal(per[:half], longest[:half])
                and np.array_equal(per[::-1][:half], longest[::-1][:half])
            ):
                raise SystemExit(
                    f"kicked Ising {layers} layers: N={r['num_qubits']} profile differs"
                )
            checked.append(r["num_qubits"])
        out[layers] = (bulk, width, checked)
    return out


def gate_read(row, weights):
    """The average list a scan of one layer reads, over the mean list.

    Each list is weighted by the number of scans of one layer that fold it in full.
    """
    lengths = np.asarray(row["list_lengths"], float)
    fold = np.asarray(weights[f"{row['basis']}:{row['num_qubits']}"]["fold"], float)
    return float((fold * lengths).sum() / fold.sum() / lengths.mean())


def merge(run_dirs, out: Path) -> None:
    """Rows at the Fig. 3 workload and depths, one per (model, N, layers).

    A second row for the same key must be the identical measurement (a rerun); anything else
    is refused. Every row gains ``gate_read_over_mean``.
    """
    weights = json.loads((mf.DATA / WEIGHTS).read_text())
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
                    held = rows[key]
                    if (held["num_terms"], held["list_lengths"]) != (
                        r["num_terms"],
                        r["list_lengths"],
                    ):
                        raise SystemExit(f"{key}: two rows ({held['source']}, {f})")
                    continue
                r["gate_read_over_mean"] = gate_read(r, weights)
                rows[key] = {**r, "source": str(f)}
    out.write_text("".join(json.dumps(rows[k]) + "\n" for k in sorted(rows)))
    print(f"wrote {len(rows)} rows to {out}")


def profile_panel(ax, rows, profiles):
    """Each qubit's lists over the bulk list, from the chain's end, one line per depth."""
    color = mf.ENGINE_COLOR["monoprop"]
    for j, layers in enumerate(DEPTHS["pauli"]):
        bulk, width, _ = profiles[layers]
        row = max(
            (r for r in rows if r["basis"] == "pauli" and r["layers"] == layers),
            key=lambda r: r["num_qubits"],
        )
        per = qubit_lists(row)[: PROFILE_QUBITS + 1] / bulk
        ax.plot(
            np.arange(len(per)),
            per,
            **mf._style(color, j),
            label=f"{layers} layers: flat from qubit {width}",
            zorder=3,
        )
    ax.axhline(1, color=mf.GUIDE, lw=mf.GUIDE_WIDTH, ls=mf.DENSE_DASH, zorder=1)
    ax.annotate(
        "bulk list, the same at every $N$",
        xy=(PROFILE_QUBITS, 1),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=mf.INK,
    )
    ax.set_xlim(-0.5, PROFILE_QUBITS + 0.5)
    ax.set_ylim(0, 1.12)
    mf._frame(ax)
    ax.legend(
        loc="lower right",
        title="one curve per depth for every $N$",
        title_fontsize=6.4,
        alignment="left",
        frameon=False,
        fontsize=6.4,
        handlelength=2.6,
        borderaxespad=0.3,
    )
    ax.set_title(
        "(a) Kicked Ising chain: lists near the open end",
        loc="left",
        fontsize=7.8,
        color=mf.INK,
    )
    ax.set_xlabel("qubit, counted from the end of the chain", labelpad=1.5)
    ax.set_ylabel("qubit's lists / bulk", labelpad=2)


def read_panel(ax, rows):
    """c and the average list a scan reads, over the mean, at each model's medium depth."""
    blue = mf.ENGINE_COLOR["monoprop"]
    for kind in ("pauli", "majorana"):
        layers = DEPTHS[kind][1]
        mine = sorted(
            (r for r in rows if r["basis"] == kind and r["layers"] == layers),
            key=lambda r: r["num_qubits"],
        )
        n = [r["num_qubits"] for r in mine]
        for key, color, z in (
            ("c_max_over_mean", LONGEST, 2),
            ("gate_read_over_mean", blue, 3),
        ):
            style = mf._style(color, 1)
            if kind == "majorana":
                style |= {"mfc": "white", "mec": color, "mew": 0.9}
            ax.plot(n, [r[key] for r in mine], **style, zorder=z)
        if kind == "pauli":
            ax.annotate(
                f"{MODEL_NAME[kind]}: longest ($c$)",
                xy=(n[0], mine[0]["c_max_over_mean"]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=6.4,
                color=mf.INK,
            )
    ax.axhline(1, color=mf.GUIDE, lw=mf.GUIDE_WIDTH, ls=mf.DENSE_DASH, zorder=1)
    ax.annotate(
        "both models: the list a scan reads;\n"
        f"{MODEL_NAME['majorana']} (open markers): also the longest",
        xy=(min(NS), 0.985),
        xytext=(-2, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.4,
        color=mf.INK,
        linespacing=1.15,
    )
    ax.set_ylim(0.82, 1.62)
    mf._log2_axis(ax, list(NS))
    mf._frame(ax)
    ax.set_title(
        "(b) Longest list and list read", loc="left", fontsize=7.8, color=mf.INK
    )
    ax.set_xlabel("$N$", labelpad=1.5)
    ax.set_ylabel("list length / mean", labelpad=2)


def draw(rows, profiles, outdir: Path):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    profile_panel(fig.add_subplot(grid[0, 0]), rows, profiles)
    read_panel(fig.add_subplot(grid[0, 1]), rows)
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext, dpi in (("pdf", None), ("png", mf.PNG_DPI)):
        out = outdir / f"{STEM}.{ext}"
        fig.savefig(out, dpi=dpi, metadata={"CreationDate": None})
        outs.append(out)
    plt.close(fig)
    return outs


def summary(rows, profiles) -> str:
    """The numbers the appendix quotes."""
    lines = [
        f"kicked Ising {layers} layers: bulk {bulk:.0f} per qubit, flat from qubit {width}, "
        f"same profile at N={checked}"
        for layers, (bulk, width, checked) in profiles.items()
    ]
    for kind in ("pauli", "majorana"):
        mine = [r for r in rows if r["basis"] == kind]
        if not mine:
            continue
        for key, name in (("c_max_over_mean", "c"), ("gate_read_over_mean", "read")):
            values = [r[key] for r in mine]
            lines.append(f"{kind} {name}: {min(values):.4f} to {max(values):.4f}")
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
    profiles = check_profiles(rows)
    print(*draw(rows, profiles, args.outdir), sep="\n")
    print(summary(rows, profiles))


if __name__ == "__main__":
    main()
