#!/usr/bin/env python3
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
"""Execute the tutorial notebooks and convert them to Markdown for Fumadocs.

Each notebook in ``docs/notebooks/<name>/<name>.ipynb`` is executed top to
bottom and rendered to ``docs/content/docs/tutorials/<name>.md``.  Execution
fails the build on the first cell error (``allow_errors=False``); this is the
notebook-level "doctest".

Output images (matplotlib figures, …) and images referenced from markdown cells
are inlined as base64 data URIs so the static export is fully self-contained.
Notebooks are emitted as ``.md`` (not ``.mdx``) so that ``{`` / ``<`` in code
and prose are treated literally and ``$…$`` math is left for ``remark-math``.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor

DOCS_DIR = Path(__file__).resolve().parent.parent
NB_DIR = DOCS_DIR / "notebooks"
OUT_DIR = DOCS_DIR / "content" / "docs" / "tutorials"
TIMEOUT = 600  # seconds per cell

_IMG_REF = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_H1 = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def discover_notebooks() -> list[str]:
    """Return the stems of `docs/notebooks/<stem>/<stem>.ipynb`, alphabetically."""
    return sorted(
        p.name
        for p in NB_DIR.iterdir()
        if p.is_dir() and (p / f"{p.name}.ipynb").is_file()
    )


def _notebook_title(nb: nbformat.NotebookNode, stem: str) -> str:
    """Extract the page title from the notebook's leading H1 heading."""
    text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    match = _H1.search(text)
    return match.group(1).strip() if match else stem.replace("_", " ").title()


def title_for(stem: str) -> str:
    """Read a notebook's title without executing it."""
    nb = nbformat.read(NB_DIR / stem / f"{stem}.ipynb", as_version=4)
    return _notebook_title(nb, stem)


def _data_uri(data: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _inline_images(body: str, resources: dict, nb_dir: Path) -> str:
    """Replace image references with self-contained base64 data URIs."""
    outputs: dict[str, bytes] = resources.get("outputs", {}) or {}

    def as_img(alt: str, uri: str) -> str:
        # Emit raw HTML rather than markdown image syntax: a `data:` URI looks
        # like a relative path to fumadocs' remark-image plugin, which then
        # fails trying to read it as a file. Raw <img> bypasses that plugin.
        alt = alt.replace('"', "&quot;")
        return f'<img alt="{alt}" src="{uri}" />'

    def repl(match: re.Match[str]) -> str:
        alt, target = match.group(1), match.group(2)
        if target.startswith("data:"):
            return as_img(alt, target)
        name = target.split("/")[-1]
        if name in outputs:  # image extracted from a cell output
            return as_img(alt, _data_uri(outputs[name], name))
        local = (nb_dir / target).resolve()
        if local.is_file():  # image committed next to the notebook
            return as_img(alt, _data_uri(local.read_bytes(), local.name))
        return match.group(0)

    return _IMG_REF.sub(repl, body)


def convert(stem: str) -> Path:
    """Execute one notebook and write its Markdown page; return the path."""
    nb_path = NB_DIR / stem / f"{stem}.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    title = _notebook_title(nb, stem)

    # Execute with the notebook directory as cwd so relative data loads work.
    ExecutePreprocessor(timeout=TIMEOUT, allow_errors=False).preprocess(
        nb, {"metadata": {"path": str(nb_path.parent)}}
    )

    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(nb)
    body = _inline_images(body, resources, nb_path.parent)

    # The frontmatter title becomes the page H1; drop the notebook's own H1.
    body = _H1.sub("", body, count=1).lstrip()

    fm = f"---\ntitle: {title}\n---\n\n"
    out_path = OUT_DIR / f"{stem}.md"
    out_path.write_text(fm + body)
    return out_path


def write_index(stems: list[str]) -> Path:
    """Write the tutorials landing page (index.mdx) listing every notebook."""
    bullets = "\n".join(f"- [{title_for(stem)}](/tutorials/{stem})" for stem in stems)
    body = (
        "---\n"
        "title: Tutorials\n"
        "description: Step-by-step notebooks applying monoprop to concrete "
        "problems and workflows.\n"
        "---\n\n"
        "These tutorials work through complete problems end to end, from setting "
        "up the operator and circuit to evaluating the propagated result. Each "
        "one is a runnable notebook — for the ideas behind the method see "
        "[Concepts](/concepts), and for the full constructors and arguments "
        "see [Python API](/api).\n\n"
        f"{bullets}\n"
    )
    out_path = OUT_DIR / "index.mdx"
    out_path.write_text(body)
    return out_path


def main() -> int:
    """Convert the selected tutorial notebooks and write the sidebar meta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Notebook stems to convert (default: all). E.g. fermi_hubbard",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = discover_notebooks()
    selected = (
        [s for s in stems if s in set(args.notebooks)] if args.notebooks else stems
    )

    for stem in selected:
        print(f"Executing and converting {stem} …", flush=True)  # noqa: T201
        out = convert(stem)
        print(f"  -> {out.relative_to(DOCS_DIR)}", flush=True)  # noqa: T201

    # Landing page for `/tutorials`. fumadocs picks up `index.mdx` as the
    # folder's own index automatically, so the "Tutorials" sidebar/nav title
    # links here. Always regenerated from the full discovered list (independent
    # of the `selected` subset) so every tutorial stays listed.
    write_index(stems)

    # Sidebar ordering + section title for the tutorials folder. "index" is
    # deliberately *not* listed: fumadocs already treats `index.mdx` as the
    # folder index, so the folder title links to `/tutorials`. Listing it
    # would add a redundant child entry duplicating the section title.
    meta = OUT_DIR / "meta.json"
    pages = ", ".join(f'"{s}"' for s in stems)
    meta.write_text(f'{{\n  "title": "Tutorials",\n  "pages": [{pages}]\n}}\n')
    return 0


if __name__ == "__main__":
    sys.exit(main())
