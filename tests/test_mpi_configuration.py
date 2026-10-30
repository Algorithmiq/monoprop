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

"""Regression tests for the suite's MPI selection and reporting hooks."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from tests import conftest


@pytest.fixture
def mpi_config(monkeypatch):
    monkeypatch.setattr(conftest, "MPI", None)
    config = Mock()
    config.option.with_mpi = False
    config.invocation_params.args = ()
    config.pluginmanager.hasplugin.return_value = False
    config.getoption.side_effect = lambda _: config.option.with_mpi
    return config


@pytest.fixture
def mpi_item(mpi_config):
    item = Mock(config=mpi_config)
    item.iter_markers.return_value = [pytest.mark.mpi.mark]
    return item


def test_mpi_tests_require_opt_in(mpi_item):
    with pytest.raises(pytest.skip.Exception, match="need --with-mpi"):
        conftest.pytest_runtest_setup(mpi_item)


@pytest.mark.parametrize("with_mpi", [False, True])
def test_unmarked_tests_do_not_require_mpi(mpi_item, with_mpi):
    mpi_item.config.option.with_mpi = with_mpi
    mpi_item.iter_markers.return_value = []
    conftest.pytest_runtest_setup(mpi_item)


def test_mpi_opt_in_requires_mpi4py(mpi_item):
    mpi_item.config.option.with_mpi = True
    with pytest.raises(pytest.fail.Exception, match="MPI tests require mpi4py"):
        conftest.pytest_runtest_setup(mpi_item)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_mpi_minimum_rank_count(mpi_item, monkeypatch, size):
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(size=size)))
    mpi_item.config.option.with_mpi = True
    mpi_item.iter_markers.return_value = [pytest.mark.mpi(min_size=2).mark]
    if size < 2:
        with pytest.raises(pytest.skip.Exception, match="requires 2 MPI processes"):
            conftest.pytest_runtest_setup(mpi_item)
    else:
        conftest.pytest_runtest_setup(mpi_item)


def test_mpi_marker_defaults_to_one_rank(mpi_item, monkeypatch):
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(size=1)))
    mpi_item.config.option.with_mpi = True
    conftest.pytest_runtest_setup(mpi_item)


@pytest.mark.parametrize("adapter_plugin", [False, True])
def test_vscode_enables_mpi(mpi_config, adapter_plugin):
    mpi_config.pluginmanager.hasplugin.return_value = adapter_plugin
    mpi_config.invocation_params.args = (
        () if adapter_plugin else ("-p", "vscode_pytest")
    )
    conftest.pytest_configure(mpi_config)
    assert mpi_config.option.with_mpi


def test_serial_configuration_does_not_require_mpi(mpi_config):
    conftest.pytest_configure(mpi_config)
    assert not mpi_config.option.with_mpi
    mpi_config.pluginmanager.unregister.assert_not_called()


@pytest.mark.parametrize(
    ("with_mpi", "rank", "muted"),
    [(False, 1, False), (True, 0, False), (True, 1, True)],
)
def test_only_rank_zero_reports_under_mpi(
    mpi_config, monkeypatch, with_mpi, rank, muted
):
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(rank=rank)))
    mpi_config.option.with_mpi = with_mpi
    conftest.pytest_configure(mpi_config)
    if muted:
        mpi_config.pluginmanager.unregister.assert_called_once_with(
            mpi_config.pluginmanager.getplugin.return_value
        )
    else:
        mpi_config.pluginmanager.unregister.assert_not_called()


def test_mpi_configuration_without_terminal_reporter(mpi_config, monkeypatch):
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(rank=1)))
    mpi_config.option.with_mpi = True
    mpi_config.pluginmanager.getplugin.return_value = None
    conftest.pytest_configure(mpi_config)
    mpi_config.pluginmanager.unregister.assert_not_called()


@pytest.mark.parametrize(
    ("size", "given", "expected"),
    [
        pytest.param(1, None, None, id="serial"),
        pytest.param(3, None, "mpi-rank-2", id="ranks"),
        pytest.param(3, "/explicit", "/explicit", id="explicit"),
    ],
)
def test_each_rank_gets_its_own_basetemp(monkeypatch, tmp_path, size, given, expected):
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(size=size, rank=2)))
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(tmp_path))
    config = Mock()
    config.option.basetemp = given
    assert conftest.pytest_cmdline_main(config) is None
    if expected is None:
        assert config.option.basetemp is None
    elif expected == "/explicit":
        assert config.option.basetemp == "/explicit"
    else:
        basetemp = Path(config.option.basetemp)
        assert basetemp.name == expected
        assert basetemp.parent.parent == tmp_path
        assert basetemp.parent.name.startswith("pytest-of-")


def test_basetemp_needs_mpi4py(monkeypatch):
    monkeypatch.setattr(conftest, "MPI", None)
    config = Mock()
    config.option.basetemp = None
    conftest.pytest_cmdline_main(config)
    assert config.option.basetemp is None


def test_basetemp_parent_is_created(monkeypatch, tmp_path):
    temproot = tmp_path / "fresh-temproot"
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(size=3, rank=2)))
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(temproot))
    config = Mock()
    config.option.basetemp = None
    conftest.pytest_cmdline_main(config)
    assert Path(config.option.basetemp).parent.is_dir()


def _basetemp_for_rank(monkeypatch, temproot: Path) -> Path:
    monkeypatch.setattr(conftest, "MPI", Mock(COMM_WORLD=Mock(size=3, rank=2)))
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(temproot))
    config = Mock()
    config.option.basetemp = None
    conftest.pytest_cmdline_main(config)
    return Path(config.option.basetemp)


def test_basetemp_parent_owned_by_another_user_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(conftest.os, "getuid", lambda: tmp_path.stat().st_uid + 1)
    with pytest.raises(OSError, match="not owned by the current user"):
        _basetemp_for_rank(monkeypatch, tmp_path)


def test_basetemp_parent_symlink_is_refused(monkeypatch, tmp_path):
    temproot = tmp_path / "temproot"
    temproot.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    monkeypatch.setattr(conftest.getpass, "getuser", lambda: "someone")
    (temproot / "pytest-of-someone").symlink_to(elsewhere)
    with pytest.raises(OSError, match="symbolic link"):
        _basetemp_for_rank(monkeypatch, temproot)


def test_basetemp_parent_is_made_private(monkeypatch, tmp_path):
    monkeypatch.setattr(conftest.getpass, "getuser", lambda: "someone")
    parent = tmp_path / "pytest-of-someone"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    assert _basetemp_for_rank(monkeypatch, tmp_path).parent == parent
    assert parent.stat().st_mode & 0o777 == 0o700
