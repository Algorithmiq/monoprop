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

"""Tests for the GHCR cleanup policy in tools/cleanup-ghcr.py."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "tools" / "cleanup-ghcr.py"
_spec = importlib.util.spec_from_file_location("cleanup_ghcr", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
# dataclasses resolves the defining module through sys.modules.
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

Version = _module.Version
is_pinned = _module.is_pinned
select_deletions = _module.select_deletions

NOW = dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.UTC)
MIN_AGE = dt.timedelta(hours=24)


def _version(version_id, digest, *tags, age=dt.timedelta(days=3)):
    return Version(
        version_id=version_id, digest=digest, tags=tags, created_at=NOW - age
    )


def _deleted(versions, *, open_prs=(), children_of=None):
    selected = select_deletions(
        versions, set(open_prs), children_of or {}, now=NOW, min_age=MIN_AGE
    )
    return {version.digest for version in selected}


@pytest.mark.parametrize(
    "tag", ["1.2.3", "v1.2.3", "0.5.0rc1", "v0.5.0-rc.1", "latest", "main"]
)
def test_release_and_branch_tags_are_pinned(tag):
    assert is_pinned((tag,), set())


@pytest.mark.parametrize(
    "tag", ["sha-1a2b3c4", "3f9c0a1b2d4e", "pr-7", "feature-x", "1.2"]
)
def test_other_tags_are_not_pinned(tag):
    assert not is_pinned((tag,), set())


def test_open_pr_tag_is_pinned():
    assert is_pinned(("pr-7",), {7})


@pytest.mark.parametrize(
    "tag",
    [
        "1.2.3-sapphirerapids",
        "v0.5.0-rc.1-sapphirerapids",
        "latest-sapphirerapids",
        "main-sapphirerapids",
    ],
)
def test_cpu_target_suffix_keeps_release_and_branch_tags_pinned(tag):
    assert is_pinned((tag,), set())


def test_cpu_target_suffix_keeps_open_pr_tags_pinned():
    assert is_pinned(("pr-7-sapphirerapids",), {7})
    assert not is_pinned(("pr-7-sapphirerapids",), set())


@pytest.mark.parametrize(
    "tag",
    [
        "sha-1a2b3c4-sapphirerapids",
        # a dispatch on a branch named main-feature
        "main-feature-sapphirerapids",
        # an unknown suffix is not a CPU target
        "main-feature",
    ],
)
def test_other_suffixed_tags_are_not_pinned(tag):
    assert not is_pinned((tag,), set())


def test_release_keeps_its_platform_and_attestation_manifests():
    versions = [
        _version(1, "sha256:index", "1.2.3"),
        _version(2, "sha256:amd64"),
        _version(3, "sha256:attestation"),
    ]
    children = {"sha256:index": ["sha256:amd64", "sha256:attestation"]}
    assert _deleted(versions, children_of=children) == set()


def test_superseded_main_is_deleted_with_its_children():
    versions = [
        _version(1, "sha256:new", "main", "sha-bbbbbbb"),
        _version(2, "sha256:new-amd64"),
        _version(3, "sha256:old", "sha-aaaaaaa"),
        _version(4, "sha256:old-amd64"),
    ]
    children = {"sha256:new": ["sha256:new-amd64"]}
    assert _deleted(versions, children_of=children) == {
        "sha256:old",
        "sha256:old-amd64",
    }


def test_closed_pr_is_deleted_and_open_pr_is_kept():
    versions = [
        _version(1, "sha256:open", "pr-9"),
        _version(2, "sha256:closed", "pr-7"),
    ]
    assert _deleted(versions, open_prs={9}) == {"sha256:closed"}


def test_young_versions_are_kept():
    versions = [
        _version(1, "sha256:pushing", age=dt.timedelta(hours=1)),
        _version(2, "sha256:stale", age=dt.timedelta(hours=25)),
    ]
    assert _deleted(versions) == {"sha256:stale"}


def test_young_index_keeps_its_older_children():
    # A retag can point a new index at platform manifests pushed long before it.
    versions = [
        _version(1, "sha256:young-index", age=dt.timedelta(hours=1)),
        _version(2, "sha256:old-amd64"),
    ]
    children = {"sha256:young-index": ["sha256:old-amd64"]}
    assert _deleted(versions, children_of=children) == set()


def test_children_are_fetched_for_young_indexes(monkeypatch):
    versions = [_version(1, "sha256:young-index", age=dt.timedelta(hours=1))]
    _fake_registry(monkeypatch, versions)
    fetched = []
    monkeypatch.setattr(
        _module,
        "child_digests",
        lambda _package, digest, _token: fetched.append(digest) or [],
    )
    _module.clean_package(
        "monoprop", "token", set(), now=NOW, min_age=MIN_AGE, dry_run=True
    )
    assert fetched == ["sha256:young-index"]


def test_base_shared_by_two_prs_is_kept_while_either_is_open():
    versions = [_version(1, "sha256:base", "3f9c0a1b2d4e", "pr-7", "pr-9")]
    assert _deleted(versions, open_prs={9}) == set()
    assert _deleted(versions, open_prs=set()) == {"sha256:base"}


def test_sha_only_versions_are_deleted():
    versions = [_version(1, "sha256:commit", "sha-1a2b3c4")]
    assert _deleted(versions) == {"sha256:commit"}


def _fake_registry(monkeypatch, versions, *, fail_on=()):
    deleted = []

    def delete_version(_package, version, _token):
        if version.digest in fail_on:
            raise _module.urllib.error.HTTPError(
                "https://api.github.com", 403, "Forbidden", {}, None
            )
        deleted.append(version.digest)

    monkeypatch.setattr(_module, "list_versions", lambda *_args: versions)
    monkeypatch.setattr(_module, "registry_token", lambda *_args: "pull-token")
    monkeypatch.setattr(_module, "child_digests", lambda *_args: [])
    monkeypatch.setattr(_module, "delete_version", delete_version)
    return deleted


def test_dry_run_deletes_nothing(monkeypatch):
    deleted = _fake_registry(monkeypatch, [_version(1, "sha256:stale", "sha-1a2b3c4")])
    failures = _module.clean_package(
        "monoprop", "token", set(), now=NOW, min_age=MIN_AGE, dry_run=True
    )
    assert failures == 0
    assert deleted == []


def test_real_run_deletes_selected_and_counts_failures(monkeypatch):
    versions = [
        _version(1, "sha256:stale", "sha-1a2b3c4"),
        _version(2, "sha256:protected", "sha-5d6e7f8"),
        _version(3, "sha256:release", "1.2.3"),
    ]
    deleted = _fake_registry(monkeypatch, versions, fail_on={"sha256:protected"})
    failures = _module.clean_package(
        "monoprop", "token", set(), now=NOW, min_age=MIN_AGE, dry_run=False
    )
    assert failures == 1
    assert deleted == ["sha256:stale"]


def test_missing_package_is_skipped(monkeypatch):
    def list_versions(*_args):
        raise _module.urllib.error.HTTPError(
            "https://api.github.com", 404, "Not Found", {}, None
        )

    monkeypatch.setattr(_module, "list_versions", list_versions)
    failures = _module.clean_package(
        "monoprop-base", "token", set(), now=NOW, min_age=MIN_AGE, dry_run=False
    )
    assert failures == 0


def test_main_requires_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _module.main(["--dry-run"])
    assert excinfo.value.code == 2


def test_main_rejects_a_negative_minimum_age(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        _module,
        "open_pull_requests",
        lambda _token: pytest.fail("must not reach the API"),
    )
    with pytest.raises(SystemExit) as excinfo:
        _module.main(["--dry-run", "--min-age-hours", "-1"])
    assert excinfo.value.code == 2


def test_paginate_follows_link_headers(monkeypatch):
    pages = {
        "https://api.github.com/first": (
            b'[{"number": 7}]',
            '<https://api.github.com/second>; rel="next", <https://api.github.com/first>; rel="last"',
        ),
        "https://api.github.com/second": (b'[{"number": 9}]', ""),
    }
    requested = []

    def request(url, **_options):
        requested.append(url)
        return pages[url]

    monkeypatch.setattr(_module, "_request", request)
    assert _module._paginate("https://api.github.com/first", "token") == [
        {"number": 7},
        {"number": 9},
    ]
    assert requested == [
        "https://api.github.com/first",
        "https://api.github.com/second",
    ]


def test_child_digests_lists_index_members_only(monkeypatch):
    bodies = {
        "sha256:index": (
            b'{"mediaType": "application/vnd.oci.image.index.v1+json", '
            b'"manifests": [{"digest": "sha256:amd64"}, {"digest": "sha256:attestation"}]}'
        ),
        "sha256:plain": (
            b'{"mediaType": "application/vnd.oci.image.manifest.v1+json", "layers": []}'
        ),
    }

    def request(url, **_options):
        return bodies[url.rsplit("/", 1)[1]], ""

    monkeypatch.setattr(_module, "_request", request)
    assert _module.child_digests("monoprop", "sha256:index", "pull-token") == [
        "sha256:amd64",
        "sha256:attestation",
    ]
    assert _module.child_digests("monoprop", "sha256:plain", "pull-token") == []


def test_child_digests_recognises_an_index_without_a_media_type(monkeypatch):
    body = (
        b'{"manifests": [{"digest": "sha256:amd64"}, {"digest": "sha256:attestation"}]}'
    )
    monkeypatch.setattr(_module, "_request", lambda *_args, **_options: (body, ""))
    assert _module.child_digests("monoprop", "sha256:index", "pull-token") == [
        "sha256:amd64",
        "sha256:attestation",
    ]
