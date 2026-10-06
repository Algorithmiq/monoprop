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

"""One definition of each figure's caption, shared by the results document, the artifact page and
figures/captions.txt, so the three cannot drift. Captions are authored in plain text and
converted for LaTeX by `to_latex`; embedding LaTeX in the source would leak markup into HTML.

The figures carry no titles and no panel letters, so the caption is the only place the reader is
told what the reference lines mean and what the machine configuration was. Every range is
computed from the rows the figure is drawn from -- a caption in an earlier campaign read "88-98M"
against rungs that landed at 92-98M.
"""

from __future__ import annotations

import re

CONFIG = (
    "Deucalion x86, 128 cores per node, 8 MPI ranks per node with 16 partitions per rank, "
    "so R = 8N ranks at N nodes. Each point is the median of five reps."
)


def _cores(curves):
    cs = sorted({p["cores"] for _, pts in curves for p in pts})
    return cs[0], cs[-1]


# Plain-text atom -> LaTeX math. Substituted in ONE regex pass, longest alternative first: a
# sequential replace() chain rescans its own output, so the short "t(N)" atom fires again inside
# the already-wrapped "$t(N_0)N_0 / t(N)N$" and yields "$t(N_0)N_0 / $t(N)$N$". The captions are
# authored in plain text because the artifact page renders them as prose; LaTeX is derived.
MATH = {
    "t(N0)N0/t(N)N": r"$t(N_0)N_0 / t(N)N$",
    "t(128 cores) / t(N)": r"$t(128\,\mathrm{cores}) / t(N)$",  # \, not "\ ": wrap() splits on
    # whitespace, and a lone trailing backslash at a line break is not safe LaTeX
    "1/N": r"$1/N$",
    "R = 8N": r"$R = 8N$",
    "N nodes": r"$N$ nodes",
}
_MATH_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(MATH, key=len, reverse=True))
)


def to_latex(text):
    """Make a caption safe to drop verbatim into \\caption{}.

    `%` is a LaTeX COMMENT character: an unescaped "100%" silently swallows the rest of the line,
    which is the one failure this conversion exists to prevent. Escaping it is not cosmetic.
    """
    out = (
        text.replace("\u00d7", r"$\times$")
        .replace("\u2013", "--")
        .replace("\u2014", "---")
        .replace("\u2019", "'")
    )
    out = _MATH_RE.sub(lambda m: MATH[m.group(0)], out)
    return out.replace("%", r"\%")


def combined(drawn):
    """The 2x2 composite's caption, every range and percentage computed from the plotted rows."""
    strong, weak = drawn["strong"], drawn["weak"]
    lo, hi = _cores(strong + weak)
    sizes = ", ".join(lab.replace(" monomials", "") for lab, _ in strong)
    loads = ", ".join(lab.replace(" monomials/node", "") for lab, _ in weak)
    seff, weff = [], []
    for lab, pts in strong:
        p0, pn = min(pts, key=lambda p: p["cores"]), max(pts, key=lambda p: p["cores"])
        seff.append(
            f"{100 * p0['median'] * p0['nodes'] / (pn['median'] * pn['nodes']):.0f}%"
        )
    for lab, pts in weak:
        p0, pn = min(pts, key=lambda p: p["cores"]), max(pts, key=lambda p: p["cores"])
        weff.append(f"{100 * p0['median'] / pn['median']:.0f}%")
    return (
        f"Strong and weak scaling of monoprop. Distributed propagation of the linear Hubbard "
        f"model, from {lo} cores on {lo // 128} node to {hi} cores on {hi // 128} nodes (the "
        f"four panels share one x axis, labelled in cores under the bottom row and in nodes "
        f"over the top row). "
        f"Top row, strong scaling at three fixed numbers of tracked monomials ({sizes}): "
        f"(a) wall time, with each curve’s own ideal 1/N dotted beside it, and "
        f"(b) parallel efficiency t(N0)N0/t(N)N, each curve normalised to the narrowest run "
        f"its problem fits in, so all three begin at 100% and the single ideal line applies "
        f"to every one — reaching {', '.join(seff)} at {hi} cores. "
        f"Bottom row, weak scaling at three loads per node ({loads} monomials/node): "
        f"(c) wall time, with each curve’s own flat ideal dotted beside it, and "
        f"(d) parallel efficiency t({lo} cores) / t(N), reaching {', '.join(weff)} at {hi} "
        f"cores. Colour and marker encode problem size (top) or load per node (bottom); the "
        f"legend in each row’s efficiency panel serves both panels of that row. Together the "
        f"rows "
        f"show one thing: the departure from ideal is set by the load a node carries, not by "
        f"a core count, and it moves to higher core counts as the problem grows. {CONFIG}"
    )
