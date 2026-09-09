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

"""Keep one pinned Bencher plot carrying every `terms` series in the project.

`terms` is deterministic for a fixed seed and problem size, so its interest is entirely in
whether any series moved at all -- which reads better as one plot of every operator in the
project than as a line buried in each rung's own view. A pinned plot enumerates branch,
testbed, benchmark and measure UUIDs rather than matching them by pattern, so a new rung or
row would not appear in it on its own; re-running this after each upload is what keeps it
whole.

Idempotent, and keyed on the plot title: it patches the plot of that title if there is one
and creates it otherwise. Nothing else about the plot is managed, so a maintainer can move
or rename it in the console and only its membership will be rewritten.

Usage::

    pin_terms_plot.py <project> [--title TITLE] [--measure terms] [--branch main]

Needs a *user*-scoped credential, which is not the one `bencher run` uploads with: the plots
endpoint resolves a bearer to a user, accepting a `bencher_user_*` API key or a JWT, while a
project key (`BENCHER_API_KEY`) authenticates report submission down another path and is
rejected here. So this reads `BENCHER_USER_KEY` or `BENCHER_API_TOKEN`, and skips when neither
is set. `BENCHER_HOST` overrides the API host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_HOST = "https://api.bencher.dev"

# Bencher tells a user key from a project key by this prefix, and parses anything else as a JWT.
USER_KEY_PREFIX = "bencher_user_"

# One data point per commit on one branch, so a year of main is a readable width.
WINDOW_SECONDS = 365 * 24 * 60 * 60

# `terms` uploads a bare value against an exact-match threshold, so neither the one-sigma
# interval nor the boundary lines carry anything here.
PLOT_STYLE = {
    "lower_value": False,
    "upper_value": False,
    "lower_boundary": False,
    "upper_boundary": False,
    "x_axis": "date_time",
    "window": WINDOW_SECONDS,
}


class ApiError(RuntimeError):
    """A Bencher API call that did not return success."""


def warn(message: str) -> None:
    """Annotate the run without failing the step.

    The plot sits beside an upload that has already succeeded, so everything short of a
    missing API key is worth saying and not worth failing over.
    """
    print(f"::warning::{message}", file=sys.stderr)


def request(
    method: str, url: str, token: str, body: dict[str, Any] | None = None
) -> Any:
    """Return the parsed `data` of one Bencher API response."""
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, method=method)  # noqa: S310
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            return json.loads(response.read() or "null")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        msg = f"{method} {url} returned {exc.code}: {detail}"
        raise ApiError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"{method} {url} failed: {exc.reason}"
        raise ApiError(msg) from exc


def collection(
    base: str, project: str, name: str, token: str, **query: str
) -> list[Any]:
    """Return every page of one project collection, oldest first."""
    items: list[Any] = []
    page = 1
    while True:
        params = {"page": str(page), "per_page": "255", **query}
        url = f"{base}/v0/projects/{project}/{name}?{urllib.parse.urlencode(params)}"
        batch = request("GET", url, token) or []
        items.extend(batch)
        # A short page is the last one; the API caps per_page, so never trust a count.
        if len(batch) < 255:
            return items
        page += 1


def uuid_of(items: list[Any], wanted: str) -> str | None:
    """Return the UUID of the item whose name or slug is ``wanted``."""
    for item in items:
        if wanted in (item.get("name"), item.get("slug")):
            return str(item["uuid"])
    return None


def main(argv: list[str] | None = None) -> str | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="Bencher project slug or UUID.")
    parser.add_argument("--title", default="Operator terms, every rung")
    parser.add_argument("--measure", default="terms")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args(argv)

    token = os.environ.get("BENCHER_USER_KEY") or os.environ.get("BENCHER_API_TOKEN")
    if not token:
        warn(
            "no user-scoped credential, so the pinned plot is left alone. Set "
            "BENCHER_USER_KEY to a `bencher_user_*` key, or BENCHER_API_TOKEN to a JWT: "
            "the project key that uploads reports cannot write a plot."
        )
        return None
    if not token.startswith(USER_KEY_PREFIX) and token.count(".") != 2:
        # Neither prefix nor JWT shape: the server would parse it as a JWT and reject it, which
        # is what a project key pasted into BENCHER_USER_KEY looks like.
        warn(
            f"the credential is neither a `{USER_KEY_PREFIX}*` key nor a JWT, so the plots "
            "endpoint would reject it; leaving the pinned plot alone"
        )
        return None
    base = os.environ.get("BENCHER_HOST", DEFAULT_HOST).rstrip("/")

    measures = collection(base, args.project, "measures", token)
    measure = uuid_of(measures, args.measure)
    branch = uuid_of(collection(base, args.project, "branches", token), args.branch)
    # Before the first upload of a measure there is nothing to plot, and a project that
    # never tracks `terms` is a valid one. Neither is worth failing a benchmark run over.
    if measure is None or branch is None:
        missing = "measure" if measure is None else "branch"
        warn(
            f"no {missing} to plot yet ({args.measure!r} / {args.branch!r}); "
            "leaving the pinned plot alone"
        )
        return None

    testbeds = [
        str(t["uuid"]) for t in collection(base, args.project, "testbeds", token)
    ]
    # Every benchmark, not only the ones that carry `terms` today: a benchmark without the
    # measure contributes no line, and listing them all means a new row joins the plot the
    # first time this runs after it.
    benchmarks = [
        str(b["uuid"]) for b in collection(base, args.project, "benchmarks", token)
    ]
    if not testbeds or not benchmarks:
        warn("the project has no testbeds or benchmarks yet")
        return None

    membership = {
        "branches": [branch],
        "testbeds": testbeds,
        "benchmarks": benchmarks,
        "measures": [measure],
    }
    existing = collection(base, args.project, "plots", token, title=args.title)
    # `title` is a search rather than a lookup, so match it exactly before patching.
    match = next((p for p in existing if p.get("title") == args.title), None)

    if match is None:
        request(
            "POST",
            f"{base}/v0/projects/{args.project}/plots",
            token,
            {"title": args.title, **PLOT_STYLE, **membership},
        )
        action = "created"
    else:
        request(
            "PATCH",
            f"{base}/v0/projects/{args.project}/plots/{match['uuid']}",
            token,
            membership,
        )
        action = "updated"

    print(
        f"{action} pinned plot {args.title!r}: {len(benchmarks)} benchmarks "
        f"x {len(testbeds)} testbeds, measure {args.measure}"
    )
    return None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ApiError as error:
        # A plot is a convenience beside the upload that just succeeded, so say what broke
        # and let the run stand.
        warn(f"pinning the terms plot failed: {error}")
