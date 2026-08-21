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
#
# ---------------------------------------------------------------------------
# Portions of this file are derived from `fumapy`, the Python half of the
# fumadocs-python package (https://github.com/fuma-nama/fumadocs):
#
#   MIT License. Copyright (c) 2023 Fuma
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------
"""Dump a Python package's documented API surface to JSON, via griffe.

The JSON is the input to two consumers, which is why its schema is a contract
rather than an implementation detail:

- `scripts/generate-api.mjs`, which renders it to the MDX under
  `content/docs/api/`, and
- `scripts/remark-xref.mjs`, which builds the `[Symbol][]` cross-reference index
  for prose pages and docstrings from it.

Run it through the `gen-api` recipe rather than directly; it needs `monoprop`
importable, which means the compiled extension has to be built first.
"""

from __future__ import annotations

import argparse
import json
import typing as t
from importlib.metadata import version
from pathlib import Path

import griffe

# ---------------------------------------------------------------------------
# The JSON schema
#
# These describe the *serialised* shape, which is what the two JS consumers
# read. In flight, the same fields hold griffe model objects (a
# `DocstringParameter` where a `Parameter` is declared, an `Expr` where a `str`
# is); `Encoder` flattens them at dump time. The `t.cast` calls in the `parse_*`
# functions are where that gap is crossed deliberately.
# ---------------------------------------------------------------------------


class Module(t.TypedDict):
    """A documented module or package."""

    name: str
    path: str
    filepath: str
    description: str | None
    docstring: Docstring
    attributes: list[Attribute]
    modules: dict[str, Module]
    classes: dict[str, Class]
    functions: dict[str, Function]
    version: str | None


class Class(t.TypedDict):
    """A documented class."""

    name: str
    path: str
    description: str | None
    parameters: list[Parameter]
    attributes: list[Attribute]
    docstring: Docstring
    functions: dict[str, Function]
    source: str
    inherited_members: dict[str, list[dict[str, str]]]


class Function(t.TypedDict):
    """A documented function or method."""

    name: str
    path: str
    signature: str
    description: str | None
    parameters: list[Parameter]
    returns: dict[str, str | None]
    docstring: Docstring
    source: str


class DocstringSection(t.TypedDict):
    """A docstring section that `simplify_docstring` did not consume."""

    kind: str
    value: str | list[Parameter]


Docstring = list[DocstringSection]


class Parameter(t.TypedDict):
    """A signature parameter, merged with its docstring entry if it has one."""

    name: str
    annotation: str
    description: str
    value: str


class Attribute(t.TypedDict):
    """A module- or class-level attribute."""

    name: str
    annotation: str
    description: str
    value: str


# ---------------------------------------------------------------------------
# Docstring simplification
# ---------------------------------------------------------------------------


class SimplifiedDocstring(t.NamedTuple):
    """A docstring split into the parts the renderer treats specially.

    Entries are either plain dicts built from the signature or the griffe
    docstring elements they were parsed from; `Encoder` serialises both, so the
    two are interchangeable downstream. A field is `None` when the parent cannot
    have it at all (a module has no parameters), as opposed to an empty list for
    a parent that has none.
    """

    description: str | None
    parameters: list[object] | None
    returns: object | None
    attributes: list[object] | None
    remainder: list[griffe.DocstringSection]


def _parameters_from_signature(
    parent: griffe.Class | griffe.Function,
) -> list[object]:
    return [
        {
            "name": p.name,
            "annotation": p.annotation,
            "description": None,
            "value": p.default,
        }
        for p in parent.parameters
    ]


def _returns_from_signature(parent: griffe.Function) -> object:
    return {
        "name": "",
        "annotation": (
            parent.returns
            if isinstance(parent.returns, (str, type(None)))
            else "".join(
                elem if isinstance(elem, str) else elem.canonical_path
                for elem in parent.returns.iterate(flat=True)
            )
        ),
        "description": None,
    }


def _attributes_from_signature(
    parent: griffe.Module | griffe.Class,
) -> list[object]:
    return [
        {
            "name": attr.name,
            "annotation": attr.annotation,
            "description": attr.docstring.parsed if attr.docstring else None,
            "value": attr.value,
        }
        for attr in parent.attributes.values()
        if (not attr.is_alias and not attr.is_private)
    ]


def _parse_nested(description: str) -> object:
    """Re-parse a docstring entry's description as a Google-style docstring.

    Parameter and attribute descriptions can themselves carry sections; the
    renderer expects the parsed form. A description that is already parsed is
    returned unchanged -- `AttributeError` is how that shows up.
    """
    try:
        return griffe.parse_google(griffe.Docstring(description))
    except AttributeError:
        return description


def _merge_parameters(
    documented: list[griffe.DocstringParameter], parent: griffe.Class | griffe.Function
) -> list[object]:
    """Order a docstring's parameter entries by the real signature.

    Yields one entry per signature parameter, documented or not, so the renderer
    can lay out a full parameter list from a partial docstring.
    """
    by_name = {entry.name: entry for entry in documented}
    merged: list[object] = []
    for param in parent.parameters:
        entry = by_name.get(param.name)
        if entry is not None:
            # Replaces the raw text with its parsed sections in place, which the
            # renderer reads back -- griffe declares `description` as `str`, so
            # this widening is deliberate.
            entry.description = t.cast("str", _parse_nested(entry.description))
            merged.append(entry)
        else:
            merged.append(
                {
                    "name": param.name,
                    "annotation": param.annotation,
                    "description": None,
                    "value": param.default,
                }
            )
    return merged


def _merge_attributes(
    documented: list[griffe.DocstringAttribute], parent: griffe.Module | griffe.Class
) -> list[object]:
    """Order a docstring's attribute entries by the real attributes.

    Yields one entry per non-aliased attribute, documented or not.
    """
    by_name = {entry.name: entry for entry in documented}
    merged: list[object] = []
    for attr in parent.attributes.values():
        if attr.is_alias:
            continue
        entry = by_name.get(attr.name)
        if entry is not None:
            merged.append(
                {
                    "name": entry.name,
                    "description": _parse_nested(entry.description),
                    # The declared annotation wins over the inferred one, but
                    # the value only exists on the real attribute.
                    "annotation": entry.annotation,
                    "value": attr.value,
                }
            )
        else:
            merged.append(
                {
                    "name": attr.name,
                    "annotation": attr.annotation,
                    "description": (attr.docstring.parsed if attr.docstring else None),
                    "value": attr.value,
                }
            )
    return merged


def simplify_docstring(
    doc: griffe.Docstring | None,
    parent: griffe.Module | griffe.Class | griffe.Function | None = None,
) -> SimplifiedDocstring:
    """Split a docstring into description, parameters, returns and the rest.

    The `parameters` and `attributes` sections are re-ordered to follow the real
    signature, and entries the docstring omits are filled in from it, so the
    renderer can lay out every parameter whether or not it is documented.

    Args:
        doc: The docstring to simplify; may be `None`.
        parent: The object the docstring belongs to, used for the signature.

    Returns:
        The docstring's parts, with anything not recognised in `remainder`.
    """
    description = None
    parameters = (
        _parameters_from_signature(parent)
        if isinstance(parent, (griffe.Class, griffe.Function))
        else None
    )
    attributes = (
        _attributes_from_signature(parent)
        if isinstance(parent, (griffe.Module, griffe.Class))
        else None
    )
    returns = (
        _returns_from_signature(parent) if isinstance(parent, griffe.Function) else None
    )
    remainder: list[griffe.DocstringSection] = []

    if not doc:
        return SimplifiedDocstring(
            description, parameters, returns, attributes, remainder
        )

    for i, sec in enumerate(doc.parsed):
        # The leading prose is the summary; later text sections stay in the
        # remainder so the renderer can place them after the signature.
        if sec.kind == "text" and i == 0:
            description = sec.value
            continue

        # Each branch is gated on the parent kind that can actually have the
        # section, so a misplaced section (`Args:` on a module) lands in the
        # remainder instead of failing on a missing signature.
        if sec.kind == "parameters" and isinstance(
            parent, (griffe.Class, griffe.Function)
        ):
            parameters = _merge_parameters(sec.value, parent)
            continue

        if sec.kind == "returns" and isinstance(parent, griffe.Function):
            returns = sec.value[0]
            returns.annotation = (
                returns.annotation.canonical_path
                if isinstance(returns.annotation, griffe.Expr)
                else returns.annotation
            )
            continue

        if sec.kind == "attributes" and isinstance(
            parent, (griffe.Module, griffe.Class)
        ):
            attributes = _merge_attributes(sec.value, parent)
            continue

        remainder.append(sec)

    return SimplifiedDocstring(description, parameters, returns, attributes, remainder)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def _render_parameter(p: griffe.Parameter) -> str:
    """Render one parameter, with its `*`/`**` prefix and default if it has one."""
    if p.kind == griffe.ParameterKind.var_keyword:
        return f"**{p.name}"
    if p.kind == griffe.ParameterKind.var_positional:
        return f"*{p.name}"
    if p.default is not None:
        return f"{p.name}={p.default}"
    return p.name


def build_signature(func: griffe.Function) -> str:
    """Render a function's signature, including the `/` and `*` separators.

    griffe stores parameter kinds rather than the separators, so the markers are
    reconstructed: `/` closes the positional-only run at the first parameter
    that is not positional-only, and `*` opens the keyword-only run.

    Args:
        func: The function to render.

    Returns:
        The signature, from the opening parenthesis to the return annotation.
    """
    tokens: list[str] = []
    positional_only = True
    keyword_only = False
    for i, p in enumerate(func.parameters):
        if p.kind in (
            griffe.ParameterKind.positional_or_keyword,
            griffe.ParameterKind.keyword_only,
        ):
            # The first parameter that is not positional-only closes the run,
            # unless it is also the first parameter overall (nothing to close).
            if positional_only and i != 0:
                tokens.append("/")
            positional_only = False

            if p.kind == griffe.ParameterKind.keyword_only and not keyword_only:
                tokens.append("*")
                keyword_only = True

        tokens.append(_render_parameter(p))

    signature = f"({', '.join(tokens)})"
    if func.returns:
        signature += f" -> {func.returns}"

    return signature


# ---------------------------------------------------------------------------
# Walking the package
# ---------------------------------------------------------------------------


def parse_module(m: griffe.Object | griffe.Alias) -> Module:
    """Walk a module recursively, dropping re-exported (aliased) members.

    Aliases are skipped so a symbol is documented once, on the page for the
    module that defines it -- which is also what makes the cross-reference index
    single-valued.

    Args:
        m: The module to walk.

    Returns:
        The module's documented surface.

    Raises:
        TypeError: If `m` is not a module.
    """
    if not isinstance(m, griffe.Module):
        raise TypeError("Module must be a module")

    out = simplify_docstring(m.docstring, m)
    res: dict[str, object] = {
        "name": m.name,
        "path": m.path,
        "filepath": m.filepath,
        "description": out.description,
        "docstring": out.remainder,
        "attributes": out.attributes,
        "modules": {
            name: parse_module(value)
            for name, value in m.modules.items()
            if not value.is_alias
        },
        "classes": {
            name: parse_class(value)
            for name, value in m.classes.items()
            if not value.is_alias
        },
        "functions": {
            name: parse_function(value)
            for name, value in m.functions.items()
            if not value.is_alias
        },
    }
    if m.is_package:
        try:
            res["version"] = version(m.name)
        except AttributeError:
            res["version"] = "unknown"

    return t.cast("Module", res)


def parse_class(c: griffe.Class) -> Class:
    """Walk a class.

    Inherited members are recorded by canonical path and grouped by the class
    that defines them, so the renderer can point at the base class rather than
    duplicating the documentation.

    Args:
        c: The class to walk.

    Returns:
        The class's documented surface.
    """
    out = simplify_docstring(c.docstring, c)
    inherited: dict[str, list[dict[str, str]]] = {}
    res: dict[str, object] = {
        "name": c.name,
        "path": c.path,
        "description": out.description,
        "parameters": out.parameters,
        "attributes": out.attributes,
        "docstring": out.remainder,
        "functions": {
            name: parse_function(value)
            for name, value in c.functions.items()
            if not value.is_alias
        },
        "source": c.source,
        "inherited_members": inherited,
    }
    for member in c.inherited_members.values():
        parent_path = ".".join(member.canonical_path.split(".")[:-1])
        entry = {"kind": member.kind, "path": member.canonical_path}
        inherited.setdefault(parent_path, []).append(entry)
    return t.cast("Class", res)


def parse_function(f: griffe.Function) -> Function:
    """Walk a function or method."""
    out = simplify_docstring(f.docstring, f)
    return t.cast(
        "Function",
        {
            "name": f.name,
            "path": f.path,
            "signature": build_signature(f),
            "description": out.description,
            "parameters": out.parameters,
            "returns": out.returns,
            "docstring": out.remainder,
            "source": f.source,
        },
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class Encoder(griffe.JSONEncoder):
    """Serialise the griffe objects `simplify_docstring` leaves in the tree.

    The base class already handles every griffe model (via `as_dict`) plus the
    `Path`/`set` values they carry; the one thing it gets wrong for us is
    `Expr`, which *has* an `as_dict` and so would serialise as a nested tree
    where the consumers want the annotation as written.
    """

    def default(self, obj: object) -> object:
        """Return a serialisable representation of `obj`."""
        if isinstance(obj, griffe.Expr):
            return str(obj)
        return super().default(obj)


def main() -> None:
    """Dump the API surface of the module named on the command line."""
    parser = argparse.ArgumentParser(description="Dump a Python API surface to JSON")
    parser.add_argument("module", type=str, help="The module to document")
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=Path(),
        help="The directory to write `<module>.json` into",
    )
    args = parser.parse_args()

    pkg = parse_module(
        griffe.load(
            args.module,
            docstring_parser="auto",
            # The renderer shows each function's source in a collapsible block.
            store_source=True,
        )
    )

    out_path = args.dir / f"{args.module}.json"
    with out_path.open("w") as file:
        json.dump(pkg, file, cls=Encoder, indent=2, full=True)


if __name__ == "__main__":
    main()
