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
