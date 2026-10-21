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

"""Delete stale versions of monoprop's container images from GHCR.

``container.yml`` pushes ``ghcr.io/algorithmiq/monoprop`` and
``ghcr.io/algorithmiq/monoprop-base`` for pull requests, ``main`` and releases. This
keeps what someone may still pull and deletes the rest. A version is kept when it is

- tagged with a release (``1.2.3``, ``v1.2.3``, pre-releases), ``latest`` or ``main``;
- tagged ``pr-N`` while pull request ``N`` is open;
- a platform or attestation manifest of a version kept above. GHCR lists those as
  separate untagged versions, and deleting one breaks the image that references it;
- younger than ``--min-age-hours``, so a push in progress is never cut short.

Only the standard library is used, so the workflow runs it with a bare ``python3``.
The token is read from ``GITHUB_TOKEN``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Collection, Mapping, Sequence

RELEASE_TAG = re.compile(r"^v?\d+\.\d+\.\d+([.+-]?[0-9A-Za-z.]+)?$")
PR_TAG = re.compile(r"^pr-(\d+)$")
BRANCH_TAGS = frozenset({"latest", "main"})


@dataclass(frozen=True)
class Version:
    """One container package version as GHCR lists it."""

    version_id: int
    """GitHub's package version id, used to delete the version."""
    digest: str
    """Manifest digest, ``sha256:...``."""
    tags: tuple[str, ...]
    """Tags pointing at the manifest; empty for untagged versions."""
    created_at: dt.datetime
    """Creation time, timezone-aware."""


def is_pinned(tags: Collection[str], open_prs: Collection[int]) -> bool:
    """Return whether a version must be kept because of its own tags.

    Args:
        tags: The version's tags.
        open_prs: Numbers of the repository's open pull requests.

    Returns:
        True for a release, ``latest`` or ``main`` tag, or the ``pr-N`` tag of an open
        pull request.
    """
    for tag in tags:
        if tag in BRANCH_TAGS or RELEASE_TAG.match(tag):
            return True
        pr = PR_TAG.match(tag)
        if pr and int(pr.group(1)) in open_prs:
            return True
    return False


def select_deletions(
    versions: Sequence[Version],
    open_prs: Collection[int],
    children_of: Mapping[str, Sequence[str]],
    *,
    now: dt.datetime,
    min_age: dt.timedelta,
) -> list[Version]:
    """Return the versions of one package that the policy deletes.

    Args:
        versions: Every version of the package.
        open_prs: Numbers of the repository's open pull requests.
        children_of: For each pinned image index, the digests of the platform and
            attestation manifests it references.
        now: The current time.
        min_age: Versions younger than this are always kept.

    Returns:
        The versions to delete, in input order.
    """
    kept: set[str] = set()
    for version in versions:
        if is_pinned(version.tags, open_prs):
            kept.add(version.digest)
            kept.update(children_of.get(version.digest, ()))
    return [
        version
        for version in versions
        if version.digest not in kept and now - version.created_at >= min_age
    ]
