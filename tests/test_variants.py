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

"""The fat binary: which ISA variant gets loaded, and whether they all agree.

Skipped wholesale on a single-ISA build (any source build without
``-Dmonoprop_ENABLE_FAT_BINARY=ON``), where there is nothing to select.

Every variant has to be exercised in a subprocess: the selection happens once, when ``monoprop`` is
first imported, and cannot be redone in-process.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import pytest

import monoprop
from monoprop._bootstrap import VARIANT_ENV_VAR

INSTALLED = monoprop.available_variants()

pytestmark = pytest.mark.skipif(
    not INSTALLED, reason="not a fat binary: this build ships one ISA variant"
)


def _has_isa_probe() -> bool:
    """Whether this build ships the standalone CPU probe, i.e. whether it is whole-library.

    A seam-mode build has no probe: the tier list and the predicates that read it are inside the
    engine, so there is only one copy of both and nothing for a second copy to disagree with.
    """
    try:
        from monoprop import _isa  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


HAS_ISA_PROBE = _has_isa_probe()

# One rotation on a weight-2 observable inside a wider register, evaluated as bits rather than as a
# float, so "the tiers agree" means bit-for-bit and not "to some tolerance". The point is not the
# value: it is that -ffp-contract=off holds the value fixed across ISA levels that would otherwise
# contract a*b+c differently.
_PROBE = """
import json, monoprop
from monoprop import Circuit, ExpGate, MajoranaOperator, MajoranaPropagator

num_modes = 40
observable = MajoranaOperator({(0, 1): 1j, (2, 3): 0.5j, (0, 1, 2, 3): 0.25}, num_modes)
gates = [
    ExpGate(MajoranaOperator({(2, 3): 1j}, num_modes)),
    ExpGate(MajoranaOperator({(1, 2): 1j}, num_modes)),
    ExpGate(MajoranaOperator({(0, 3): 1j}, num_modes)),
]
circuit = Circuit(gates=gates, system_size=num_modes, initial_state=[0, 1])

mp = MajoranaPropagator(observable, [0, 1], cutoff=8)
mp.build_graph(circuit)
angles = [0.37, -1.21, 0.05]
energy, gradient = mp.expectation_value_and_gradient(angles)

print(json.dumps({
    "variant": monoprop.__variant__,
    "machine_flags": monoprop.__compiler_flags__["machine-flags"],
    "energy": float(energy).hex(),
    "gradient": [float(g).hex() for g in gradient],
}))
"""


def _run_variant(variant: str) -> dict:
    env = dict(os.environ, **{VARIANT_ENV_VAR: variant})
    completed = subprocess.run(  # noqa: S603 - the interpreter running this test, not user input
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def per_variant() -> dict[str, dict]:
    return {variant: _run_variant(variant) for variant in INSTALLED}


def test_the_loaded_variant_is_installed_and_runnable_here():
    assert monoprop.__variant__ in INSTALLED
    assert monoprop.__variant__ in monoprop.supported_variants()


@pytest.mark.skipif(
    os.environ.get(VARIANT_ENV_VAR) is not None,
    reason=f"{VARIANT_ENV_VAR} pins the variant, so there is no automatic choice to check",
)
def test_selection_takes_the_best_supported_variant():
    # supported_variants() is ordered best first, so the first entry that is also installed is the
    # only correct answer -- anything else means the dispatch is leaving performance on the table.
    best = next(v for v in monoprop.supported_variants() if v in INSTALLED)
    assert monoprop.__variant__ == best


@pytest.mark.skipif(
    not HAS_ISA_PROBE,
    reason="seam-mode builds have one tier list, inside the engine; there is no second copy to check",
)
def test_the_probe_and_the_install_agree_on_the_tier_list():
    # A tier known to the probe but never installed is silently unreachable; one installed but unknown
    # to the probe can never be selected. Either way the wheel quietly loses a tier.
    # not a top-level import: absent on a single-ISA build
    from monoprop import _isa  # noqa: PLC0415

    assert set(_isa.known_variants()) == set(INSTALLED)


def test_every_variant_reports_its_own_identity(per_variant):
    # Guards the provenance: without a per-tier Variants.h every variant would claim the build's
    # default ISA, and there would be no way to tell which one had actually loaded.
    for variant, result in per_variant.items():
        assert result["variant"] == variant


def test_every_variant_returns_bit_identical_numbers(per_variant):
    reference_name, reference = next(iter(per_variant.items()))
    for variant, result in per_variant.items():
        assert result["energy"] == reference["energy"], (
            f"{variant} disagrees with {reference_name} on the energy"
        )
        assert result["gradient"] == reference["gradient"], (
            f"{variant} disagrees with {reference_name} on the gradient"
        )


def test_reported_machine_flags_widen_with_the_tier(per_variant):
    def features(variant: str) -> set[str]:
        return {
            token
            for token in per_variant[variant]["machine_flags"].split()
            # bare booleans only: "-mfoo=value" is a setting, "-mno-foo" is its own negation
            if "=" not in token and not token.startswith("-mno-")
        }

    # INSTALLED is best first, so walking it backwards walks the tiers upwards.
    for lower, higher in itertools.pairwise(reversed(INSTALLED)):
        assert features(lower) < features(higher), (
            f"{higher} does not strictly widen {lower}"
        )


def test_pinning_an_uninstalled_variant_is_refused():
    env = dict(os.environ, **{VARIANT_ENV_VAR: "x86-64-v99"})
    completed = subprocess.run(
        [sys.executable, "-c", "import monoprop"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    # Both build shapes refuse it, in their own words: the Python loader has a directory listing to
    # compare against, the engine has its generated tier table.
    assert (
        "is not installed" in completed.stderr
        or "is not an ISA tier this build ships" in completed.stderr
    )


@pytest.mark.parametrize("variant", INSTALLED)
def test_pinning_an_installed_variant_selects_exactly_it(variant):
    assert _run_variant(variant)["variant"] == variant
