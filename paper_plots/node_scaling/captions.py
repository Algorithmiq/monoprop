"""One definition of each figure's caption, shared by the results document, the artifact page and
figures/captions.txt, so the three cannot drift. Captions are authored in plain text and
converted for LaTeX by `to_latex`; embedding LaTeX in the source would leak markup into HTML.

The figures carry no titles and no panel letters, so the caption is the only place the reader is
told what the reference lines mean and what the machine configuration was. Every range is
computed from the rows the figure is drawn from -- a caption in an earlier campaign read "88-98M"
against rungs that landed at 92-98M.
"""

import re

CONFIG = ("Deucalion x86, 128 cores per node, 8 MPI ranks per node with 16 partitions per rank, "
          "so R = 8N ranks at N nodes. Each point is the median of five reps.")


def _cores(curves):
    cs = sorted({p["cores"] for _, pts in curves for p in pts})
    return cs[0], cs[-1]


# Plain-text atom -> LaTeX math. Substituted in ONE regex pass, longest alternative first: a
# sequential replace() chain rescans its own output, so the short "t(N)" atom fires again inside
# the already-wrapped "$t(N_0)N_0 / t(N)N$" and yields "$t(N_0)N_0 / $t(N)$N$". The captions are
# authored in plain text because the artifact page renders them as prose; LaTeX is derived.
MATH = {
    "t(N0)N0/t(N)N": r"$t(N_0)N_0 / t(N)N$",
    "t(128 cores) / t(N)": r"$t(128\ \mathrm{cores}) / t(N)$",
    "1/N": r"$1/N$",
    "R = 8N": r"$R = 8N$",
    "N nodes": r"$N$ nodes",
}
_MATH_RE = re.compile("|".join(re.escape(k) for k in sorted(MATH, key=len, reverse=True)))


def to_latex(text):
    """Make a caption safe to drop verbatim into \\caption{}.

    `%` is a LaTeX COMMENT character: an unescaped "100%" silently swallows the rest of the line,
    which is the one failure this conversion exists to prevent. Escaping it is not cosmetic.
    """
    out = (text.replace("\u00d7", r"$\times$").replace("\u2013", "--")
               .replace("\u2014", "---").replace("\u2019", "'"))
    out = _MATH_RE.sub(lambda m: MATH[m.group(0)], out)
    return out.replace("%", r"\%")


def strong_time(curves):
    lo, hi = _cores(curves)
    sizes = ", ".join(lab.replace(" terms", "") for lab, _ in curves)
    return (f"Strong scaling of the Hubbard propagate operator: wall time against core count, "
            f"{lo} to {hi} cores, at three fixed problem sizes ({sizes} terms). The dotted line "
            f"beside each curve is that curve’s own ideal 1/N, anchored at its first point. "
            f"{CONFIG}")


def strong_efficiency(curves):
    lo, hi = _cores(curves)
    bases, got = [], []
    for lab, pts in curves:
        p0 = min(pts, key=lambda p: p["cores"])
        pn = max(pts, key=lambda p: p["cores"])
        bases.append(f"{lab.replace(' terms', '')} from {p0['cores']}")
        got.append(f"{lab.replace(' terms', '')}: "
                   f"{100 * p0['median'] * p0['nodes'] / (pn['median'] * pn['nodes']):.0f}%")
    return (f"Strong-scaling parallel efficiency, t(N0)N0/t(N)N, for the same runs. Each curve is "
            f"normalised to the narrowest run its problem fits in ({'; '.join(bases)} cores), so "
            f"every curve begins at 100% and the single ideal line applies to all three; the "
            f"curves therefore describe how well each size scales from that point, and are not "
            f"absolute comparisons between sizes. The departure from ideal moves to higher core "
            f"counts as the problem grows, reaching {', '.join(got)} at {hi} cores. {CONFIG}")


def weak_time(curves):
    lo, hi = _cores(curves)
    loads = []
    for lab, pts in curves:
        pn = [p["terms"] / p["nodes"] / 1e6 for p in pts]
        loads.append(f"{sum(pn) / len(pn):.0f}M ({min(pn):.0f}–{max(pn):.0f}M measured)")
    return (f"Weak scaling of the same operator: wall time against core count, {lo} to {hi} "
            f"cores, holding the load per node constant at {'; '.join(loads)} terms. The problem "
            f"sizes form a sequence stepping by ×1.90–2.07 rather than exactly ×2, "
            f"so the load along a curve drifts by a few per cent and every quantity is normalised "
            f"on the measured term count. The dotted line beside each curve is that "
            f"curve’s own ideal, flat from its first point. {CONFIG}")


def weak_efficiency(curves):
    lo, hi = _cores(curves)
    eff = []
    for lab, pts in curves:
        t0 = min(pts, key=lambda p: p["cores"])["median"]
        tn = max(pts, key=lambda p: p["cores"])["median"]
        eff.append(f"{lab}: {100 * t0 / tn:.0f}%")
    return (f"Weak-scaling parallel efficiency, t({lo} cores) / t(N), for the same runs. At {hi} "
            f"cores the efficiencies are {', '.join(eff)}. The larger the load a node carries, "
            f"the closer weak scaling stays to ideal. {CONFIG}")
