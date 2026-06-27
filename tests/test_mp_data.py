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

from pathlib import Path

import numpy as np
import pytest

from monoprop import MPData


def test_from_msgpack_loads_data(lazy_shared_datadir):
    """Test that from_msgpack correctly loads data from a msgpack file."""
    filepath = lazy_shared_datadir / "random_exact.msgpack"

    monoprop_data = MPData.from_msgpack(filepath=filepath)

    # Verify the object is an instance of MPData
    assert isinstance(monoprop_data, MPData)


def test_loaded_data_has_expected_attributes(lazy_shared_datadir):
    """Test that the loaded data has all expected attributes."""
    filepath = lazy_shared_datadir / "random_exact.msgpack"

    monoprop_data = MPData.from_msgpack(filepath=filepath)

    # Check that all expected attributes exist
    expected_attrs = [
        "majoranas",
        "gen_coeffs",
        "param_inds",
        "parameters",
        "fermionic_hamiltonian",
        "actual_energy",
        "actual_gradient",
        "num_modes",
        "hartree_fock",
    ]

    for attr in expected_attrs:
        assert hasattr(monoprop_data, attr), f"Missing attribute: {attr}"


@pytest.mark.parametrize(
    ("attr_name", "expected_type", "extra_check"),
    [
        ("majoranas", list, lambda x: all(isinstance(m, tuple) for m in x)),
        ("gen_coeffs", np.ndarray, None),
        ("param_inds", np.ndarray, None),
        ("parameters", np.ndarray, None),
        (
            "fermionic_hamiltonian",
            dict,
            lambda x: all(isinstance(k, tuple) for k in x),
        ),
        ("actual_energy", (float, np.number), None),
        ("actual_gradient", np.ndarray, None),
        ("num_modes", int, None),
        ("hartree_fock", list, None),
    ],
)
def test_attribute_type(lazy_shared_datadir, attr_name, expected_type, extra_check):
    """Test that loaded data attributes have correct types."""
    monoprop_data = MPData.from_msgpack(lazy_shared_datadir / "random_exact.msgpack")
    attr_value = getattr(monoprop_data, attr_name)
    assert isinstance(attr_value, expected_type), f"Wrong type for {attr_name}"

    if extra_check:
        assert extra_check(attr_value), f"Extra check failed for {attr_name}"


def test_file_not_found():
    """Test that the appropriate exception is raised when the file is not found."""
    with pytest.raises(FileNotFoundError):
        MPData.from_msgpack(filepath=Path("nonexistent_file.msgpack"))


def test_consistency_across_files(lazy_shared_datadir):
    """Test that data can be loaded from different msgpack files."""
    files = [
        "random_exact.msgpack",
        "rx_rz_ry_rz_exact.msgpack",
        "lih_fermionic_spin_exact.msgpack",
    ]

    for file in files:
        filepath = lazy_shared_datadir / file
        if filepath.exists():
            monoprop_data = MPData.from_msgpack(filepath=filepath)
            assert isinstance(monoprop_data, MPData)

            # Basic validation that the object has data
            assert len(monoprop_data.majoranas) > 0
            assert monoprop_data.fermionic_hamiltonian


def test_internal_consistency(lazy_shared_datadir):
    """Test the internal consistency of the data."""
    filepath = lazy_shared_datadir / "random_exact.msgpack"

    monoprop_data = MPData.from_msgpack(filepath=filepath)

    # Check that param_inds shape is consistent with parameters
    if len(monoprop_data.param_inds) > 0:
        assert monoprop_data.param_inds.max() < len(monoprop_data.parameters)


# Check gradient length matches parameters length
def test_to_msgpack_saves_data(tmp_path, lazy_shared_datadir):
    """Test that to_msgpack correctly saves data to a msgpack file."""
    # First load example data
    source_path = lazy_shared_datadir / "random_exact.msgpack"
    monoprop_data = MPData.from_msgpack(filepath=source_path)

    # Save to a temporary file
    output_path = tmp_path / "test_output.msgpack"
    monoprop_data.to_msgpack(filepath=output_path)

    # Verify file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_roundtrip_serialization(tmp_path, lazy_shared_datadir):
    """Test that data can be serialized and then deserialized without loss."""
    # Load example data
    source_path = lazy_shared_datadir / "random_exact.msgpack"
    original_data = MPData.from_msgpack(filepath=source_path)

    # Save to a temporary file
    output_path = tmp_path / "roundtrip_test.msgpack"
    original_data.to_msgpack(filepath=output_path)

    # Load it back
    reloaded_data = MPData.from_msgpack(filepath=output_path)

    # Verify key attributes are the same
    # Test equality of attributes after roundtrip
    equality_tests = [
        ("majoranas", lambda o, r: len(o.majoranas) == len(r.majoranas)),
        ("gen_coeffs", lambda o, r: np.array_equal(o.gen_coeffs, r.gen_coeffs)),
        ("param_inds", lambda o, r: np.array_equal(o.param_inds, r.param_inds)),
        ("parameters", lambda o, r: np.array_equal(o.parameters, r.parameters)),
        (
            "fermionic_hamiltonian",
            lambda o, r: o.fermionic_hamiltonian == r.fermionic_hamiltonian,
        ),
    ]

    for attr_name, equality_func in equality_tests:
        assert equality_func(original_data, reloaded_data), (
            f"Roundtrip failed for {attr_name}"
        )
    assert original_data.evolved_hamiltonian == reloaded_data.evolved_hamiltonian


def test_to_dict_from_dict_roundtrip(lazy_shared_datadir):
    """Test that to_dict and from_dict methods work correctly."""
    # Load example data
    source_path = lazy_shared_datadir / "random_exact.msgpack"
    original_data = MPData.from_msgpack(filepath=source_path)

    # Convert to dict
    data_dict = original_data.to_dict()

    # Convert back to MPData
    recreated_data = MPData.from_dict(data_dict)

    # Verify key attributes are preserved
    assert len(original_data.majoranas) == len(recreated_data.majoranas)
    assert np.array_equal(original_data.gen_coeffs, recreated_data.gen_coeffs)
    assert np.array_equal(original_data.param_inds, recreated_data.param_inds)
    assert np.array_equal(original_data.parameters, recreated_data.parameters)
    assert original_data.fermionic_hamiltonian == recreated_data.fermionic_hamiltonian


def test_from_dict_accepts_missing_optional_metadata(lazy_shared_datadir):
    """Legacy dictionaries without version/tag metadata should still load."""
    source_path = lazy_shared_datadir / "random_exact.msgpack"
    original_data = MPData.from_msgpack(filepath=source_path)

    data_dict = original_data.to_dict()
    data_dict.pop("monoprop_version")
    data_dict.pop("tag")

    recreated_data = MPData.from_dict(data_dict)

    assert recreated_data.tag == ""
    assert np.array_equal(original_data.gen_coeffs, recreated_data.gen_coeffs)
    assert np.array_equal(original_data.param_inds, recreated_data.param_inds)
    assert np.array_equal(original_data.parameters, recreated_data.parameters)


def test_tag_attribute(lazy_shared_datadir, tmp_path):
    """Test the tag attribute."""
    # Load example data
    source_path = lazy_shared_datadir / "random_exact.msgpack"
    monoprop_data = MPData.from_msgpack(filepath=source_path)

    # Set a custom tag
    monoprop_data.tag = "test_tag_value"

    # Save and reload to verify tag persists
    output_path = tmp_path / "tag_test.msgpack"
    monoprop_data.to_msgpack(filepath=output_path)
    reloaded_data = MPData.from_msgpack(filepath=output_path)

    assert reloaded_data.tag == "test_tag_value"
