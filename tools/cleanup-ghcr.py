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
- younger than ``--min-age-hours``, so a push in progress is never cut short;
- a platform or attestation manifest of a version kept above. GHCR lists those as
  separate untagged versions, and deleting one breaks the image that references it.

Only the standard library is used, so the workflow runs it with a bare ``python3``.
The token is read from ``GITHUB_TOKEN``.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

logger = logging.getLogger("cleanup-ghcr")

OWNER = "algorithmiq"
REPOSITORY = "Algorithmiq/monoprop"
PACKAGES = ("monoprop", "monoprop-base")

API_URL = "https://api.github.com"
REGISTRY_URL = "https://ghcr.io"
INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
MANIFEST_ACCEPT = ", ".join(
    [
        *sorted(INDEX_MEDIA_TYPES),
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')

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


def is_retained(
    version: Version,
    open_prs: Collection[int],
    *,
    now: dt.datetime,
    min_age: dt.timedelta,
) -> bool:
    """Return whether a version is kept for its own sake, by its tags or its age.

    Args:
        version: The version to classify.
        open_prs: Numbers of the repository's open pull requests.
        now: The current time.
        min_age: Versions younger than this are always kept.

    Returns:
        True if the version is pinned or younger than ``min_age``.
    """
    return is_pinned(version.tags, open_prs) or now - version.created_at < min_age


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
        children_of: For each retained image index, the digests of the platform and
            attestation manifests it references.
        now: The current time.
        min_age: Versions younger than this are always kept.

    Returns:
        The versions to delete, in input order.
    """
    kept: set[str] = set()
    for version in versions:
        if is_retained(version, open_prs, now=now, min_age=min_age):
            kept.add(version.digest)
            # A child can be older than its index, e.g. after a registry-side retag.
            kept.update(children_of.get(version.digest, ()))
    return [version for version in versions if version.digest not in kept]


def _request(
    url: str, *, headers: Mapping[str, str], method: str = "GET"
) -> tuple[bytes, str]:
    """Send one request and return its body and ``Link`` header."""
    # The URLs are built from the fixed https bases above.
    request = urllib.request.Request(url, headers=dict(headers), method=method)  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read(), response.headers.get("Link", "")


def _api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _paginate(url: str, token: str) -> list[dict]:
    items: list[dict] = []
    next_url: str | None = url
    while next_url:
        body, link = _request(next_url, headers=_api_headers(token))
        items.extend(json.loads(body))
        match = _NEXT_LINK.search(link)
        next_url = match.group(1) if match else None
    return items


def list_versions(package: str, token: str) -> list[Version]:
    """List every version of one of the organisation's container packages.

    Args:
        package: Package name, e.g. ``monoprop``.
        token: GitHub token with ``packages: read``.

    Returns:
        The package's versions.
    """
    url = f"{API_URL}/orgs/{OWNER}/packages/container/{package}/versions?per_page=100"
    return [
        Version(
            version_id=item["id"],
            digest=item["name"],
            tags=tuple(item["metadata"]["container"]["tags"]),
            created_at=dt.datetime.fromisoformat(item["created_at"]),
        )
        for item in _paginate(url, token)
    ]


def open_pull_requests(token: str) -> set[int]:
    """Return the numbers of the repository's open pull requests.

    Args:
        token: GitHub token with ``pull-requests: read``.

    Returns:
        The open pull request numbers.
    """
    url = f"{API_URL}/repos/{REPOSITORY}/pulls?state=open&per_page=100"
    return {item["number"] for item in _paginate(url, token)}


def registry_token(package: str, token: str) -> str:
    """Exchange a GitHub token for a GHCR pull token for one package.

    Args:
        package: Package name.
        token: GitHub token; authenticating also covers packages that are still private.

    Returns:
        A bearer token for the registry API.
    """
    basic = base64.b64encode(f"token:{token}".encode()).decode()
    url = (
        f"{REGISTRY_URL}/token?service=ghcr.io&scope=repository:{OWNER}/{package}:pull"
    )
    body, _ = _request(url, headers={"Authorization": f"Basic {basic}"})
    return json.loads(body)["token"]


def child_digests(package: str, digest: str, pull_token: str) -> list[str]:
    """Return the manifests an image index references, or nothing for a plain manifest.

    Args:
        package: Package name.
        digest: Digest of the version to inspect.
        pull_token: Registry token from ``registry_token``.

    Returns:
        The referenced platform and attestation manifest digests.
    """
    url = f"{REGISTRY_URL}/v2/{OWNER}/{package}/manifests/{digest}"
    body, _ = _request(
        url,
        headers={"Accept": MANIFEST_ACCEPT, "Authorization": f"Bearer {pull_token}"},
    )
    manifest = json.loads(body)
    # mediaType is optional in OCI, so a body with a manifests list is an index too.
    if (
        manifest.get("mediaType") not in INDEX_MEDIA_TYPES
        and "manifests" not in manifest
    ):
        return []
    return [entry["digest"] for entry in manifest.get("manifests", [])]


def delete_version(package: str, version: Version, token: str) -> None:
    """Delete one package version.

    Args:
        package: Package name.
        version: The version to delete.
        token: GitHub token with ``packages: write`` and admin access to the package.
    """
    url = f"{API_URL}/orgs/{OWNER}/packages/container/{package}/versions/{version.version_id}"
    _request(url, headers=_api_headers(token), method="DELETE")


def clean_package(
    package: str,
    token: str,
    open_prs: Collection[int],
    *,
    now: dt.datetime,
    min_age: dt.timedelta,
    dry_run: bool,
) -> int:
    """Apply the retention policy to one package.

    Args:
        package: Package name.
        token: GitHub token.
        open_prs: Numbers of the repository's open pull requests.
        now: The current time.
        min_age: Versions younger than this are always kept.
        dry_run: Only log what would be deleted.

    Returns:
        The number of deletions that failed.
    """
    try:
        versions = list_versions(package, token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        # An unclosed HTTPError leaks its response file (ResourceWarning on 3.14).
        error.close()
        logger.info("%s: no such package yet, skipping", package)
        return 0
    pull_token = registry_token(package, token)
    children_of = {
        version.digest: child_digests(package, version.digest, pull_token)
        for version in versions
        if is_retained(version, open_prs, now=now, min_age=min_age)
    }
    deletions = select_deletions(
        versions, open_prs, children_of, now=now, min_age=min_age
    )
    logger.info("%s: %d versions, %d to delete", package, len(versions), len(deletions))
    failures = 0
    for version in deletions:
        tags = ", ".join(version.tags) or "untagged"
        if dry_run:
            logger.info("  would delete %s (%s)", version.digest, tags)
            continue
        try:
            delete_version(package, version, token)
        except urllib.error.HTTPError as error:
            error.close()
            failures += 1
            logger.warning(
                "  failed to delete %s (%s): HTTP %d", version.digest, tags, error.code
            )
        else:
            logger.info("  deleted %s (%s)", version.digest, tags)
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cleanup from the command line."""
    parser = argparse.ArgumentParser(
        description="Delete stale monoprop container images from GHCR."
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only log what would be deleted (default: on)",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=24.0,
        help="always keep versions younger than this (default: 24)",
    )
    parser.add_argument(
        "--package",
        action="append",
        choices=PACKAGES,
        help="package to clean; repeatable (default: all)",
    )
    args = parser.parse_args(argv)
    if args.min_age_hours < 0:
        parser.error("--min-age-hours must not be negative")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GITHUB_TOKEN is not set")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    open_prs = open_pull_requests(token)
    logger.info(
        "open pull requests: %s", ", ".join(map(str, sorted(open_prs))) or "none"
    )
    now = dt.datetime.now(dt.UTC)
    min_age = dt.timedelta(hours=args.min_age_hours)
    failures = sum(
        clean_package(
            package, token, open_prs, now=now, min_age=min_age, dry_run=args.dry_run
        )
        for package in args.package or PACKAGES
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
