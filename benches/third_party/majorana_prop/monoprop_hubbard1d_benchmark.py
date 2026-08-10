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

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

import monoprop
import numpy as np
from monoprop import Circuit, ExpGate, MajoranaPropagator
from monoprop.fermi import FermiOperator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import RssPeakSampler  # noqa: E402


def mode(site, spin):
    """Return the interleaved mode index for a given site and spin (even index for spin-up, odd for spin-down)."""
    return 2 * site if spin == "up" else 2 * site + 1


def bricklayer_topology(num_sites):
    """Return 0-indexed nearest-neighbour bonds in bricklayer order (even bonds, then odd bonds), matching PauliPropagation.jl's ``bricklayertopology`` for an open 1D chain."""
    topology = [(i, i + 1) for i in range(0, num_sites - 1, 2)]
    topology += [(i, i + 1) for i in range(1, num_sites - 1, 2)]
    return topology


def hubbard_fermion_terms(
    num_sites, hopping, interaction, chemical_potential, mu_gates=False
):
    """Return the ordered list of local FermionOperator terms for the first-order Trotter decomposition of the 1D Hubbard model.

    Args:
        num_sites: Number of spinful sites in the chain.
        hopping: Hopping amplitude t.
        interaction: On-site interaction U.
        chemical_potential: Chemical potential mu.
        mu_gates: Emit the ``2 * num_sites`` on-site number terms even when
            ``chemical_potential`` is zero. Zero-angle rotations are inert but still cost a
            pass over the operator, and MajoranaPropagation.jl's circuit has no such gates,
            so the fair-comparison default omits them.
    """
    terms = []
    topology = bricklayer_topology(num_sites)

    for spin in ("up", "down"):
        for left_site, right_site in topology:
            left, right = mode(left_site, spin), mode(right_site, spin)
            op_terms = [((left, "+"), (right, "-")), ((right, "+"), (left, "-"))]
            terms.append(
                FermiOperator(terms=op_terms, coefficients=[-hopping, -hopping])
            )

    for site in range(num_sites):
        up, down = mode(site, "up"), mode(site, "down")
        terms.append(
            FermiOperator(
                terms=[((up, "+"), (up, "-"), (down, "+"), (down, "-"))],
                coefficients=[interaction],
            )
        )

    if chemical_potential != 0 or mu_gates:
        for site in range(num_sites):
            for spin in ("up", "down"):
                m = mode(site, spin)
                terms.append(
                    FermiOperator(
                        terms=[((m, "+"), (m, "-"))],
                        coefficients=[-chemical_potential],
                    )
                )

    return terms


def build_trotter_gates(
    num_sites, hopping, interaction, chemical_potential, mu_gates=False
):
    """Convert each local Hubbard term into a Majorana generator gate for the MP simulator."""
    ferm_ops = hubbard_fermion_terms(
        num_sites, hopping, interaction, chemical_potential, mu_gates=mu_gates
    )
    return [ExpGate(term) for term in ferm_ops]


def neel_occupied_modes(num_sites, start_spin="up"):
    """Return the list of occupied mode indices for the half-filled Néel state, alternating spin between even and odd sites."""
    other = "down" if start_spin == "up" else "up"
    return [
        mode(site, start_spin if site % 2 == 0 else other) for site in range(num_sites)
    ]


def number_operator_majorana(site, spin, num_qubits):
    """Return the Majorana-basis representation of the number operator n_{site, spin}."""
    m = mode(site, spin)
    return FermiOperator(
        terms=[((m, "+"), (m, "-"))], coefficients=[1.0], num_modes=num_qubits
    )


SOURCE_LABEL = "monoprop"


def process_cpu_seconds():
    """Return CPU seconds (user + system) consumed by this process, summed over all threads."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def save_result(
    output_path,
    n_spinful_sites,
    n_layers,
    values,
    term_counts,
    cumulative_runtimes,
    memory_size,
    native_memory_size,
    provenance,
):
    """Merge this run's per-step data into the shared results JSON file, keyed by source label."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open() as f:
            data = json.load(f)
    else:
        data = {
            "n_spinful_sites": n_spinful_sites,
            "n_layers": n_layers,
            "step_range": list(range(n_layers + 1)),
            "num_threads": {},
            "runtime_seconds": {},
            "expectation_value": {},
            "num_terms": {},
            "memory_MB": {},
            "native_memory_MB": {},
        }
    requested_threads = os.environ.get("monoprop_NUM_THREADS")
    data.setdefault("num_threads", {})[SOURCE_LABEL] = (
        int(requested_threads) if requested_threads else None
    )
    data["runtime_seconds"][SOURCE_LABEL] = cumulative_runtimes
    data["expectation_value"][SOURCE_LABEL] = values
    data["num_terms"][SOURCE_LABEL] = term_counts
    data["memory_MB"][SOURCE_LABEL] = memory_size
    data.setdefault("native_memory_MB", {})[SOURCE_LABEL] = native_memory_size
    data.setdefault("provenance", {})[SOURCE_LABEL] = provenance
    with output_path.open("w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Benchmark for 1D Hubbard model")
    parser.add_argument(
        "--n-spins", "-n", help="Number of spinful sites.", type=int, default=60
    )
    parser.add_argument(
        "--max-layers", "-l", help="Number of Trotter layers.", type=int, default=20
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to the shared JSON file results are merged into.",
        default=Path(__file__).with_name("results.json"),
    )
    parser.add_argument(
        "--mu-gates",
        action="store_true",
        help="Keep the 2N inert chemical-potential gates (mu=0) that the Julia circuit lacks.",
    )

    args = parser.parse_args()

    n_spinful_sites, n_layers = args.n_spins, args.max_layers
    trotter_steps = n_layers
    t = 1.0
    u = 1.5
    dt = 0.07
    min_abs = 1.0e-8
    max_cutoff = 10
    chemical_potential = 0
    intial_state = neel_occupied_modes(n_spinful_sites, start_spin="up")
    num_qubits = 2 * n_spinful_sites

    # 0-indexed equivalent of Julia's 1-indexed N // 2 observable site.
    obs_site = n_spinful_sites // 2 - 1
    obs_spin = "up"
    observable = number_operator_majorana(obs_site, obs_spin, num_qubits)

    trotter_gates = build_trotter_gates(
        n_spinful_sites, t, u, chemical_potential, mu_gates=args.mu_gates
    )
    trotter_parameters = [dt for _ in trotter_gates]

    fermi_circuit = Circuit(
        gates=trotter_gates,
        parameters=trotter_parameters,
        initial_state=intial_state,
        system_size=num_qubits,
    )

    simulator = MajoranaPropagator(
        observable,
        fermi_circuit.initial_state,
        cutoff=max_cutoff,  # matches Julia's max_unpaired
        cutoff_type="support",
        lower_atol=min_abs,  # matches Julia's min_abs_coeff
    )

    values = np.empty(trotter_steps + 1)
    term_counts = np.empty(trotter_steps + 1, dtype=int)
    cumulative_runtimes = np.empty(trotter_steps + 1)
    memory_size = np.empty(trotter_steps + 1)
    native_memory_size = np.empty(trotter_steps + 1)

    with RssPeakSampler() as sampler:
        sampler.reset()
        cpu_start = process_cpu_seconds()
        t_start = perf_counter()
        values[0] = simulator.expectation_value()
        cumulative_runtimes[0] = perf_counter() - t_start
        term_counts[0] = simulator.size()
        memory_size[0] = sampler.peak_mb()
        native_memory_size[0] = simulator._simulator.operator_memory_bytes() / 1024**2
        for step in range(trotter_steps):
            sampler.reset()
            step_start = perf_counter()
            simulator.propagate(fermi_circuit)
            step_runtime = perf_counter() - step_start
            values[step + 1] = simulator.expectation_value()
            term_counts[step + 1] = simulator.size()
            cumulative_runtimes[step + 1] = cumulative_runtimes[step] + step_runtime
            memory_size[step + 1] = sampler.peak_mb()
            native_memory_size[step + 1] = (
                simulator._simulator.operator_memory_bytes() / 1024**2
            )
        cpu_total = process_cpu_seconds() - cpu_start

    # The engine picks one partition per core when the env var is unset, so an unset value is
    # not "1 thread"; busy_cores is the only measurement of what the threads actually did.
    t_total = cumulative_runtimes[-1]
    busy_cores = cpu_total / t_total if t_total > 0 else float("nan")
    print(
        f"{n_spinful_sites} n_spin {n_layers} layers {term_counts[-1]} num_terms {values[-1]} final overlap  runtime {t_total:.3f} seconds"
        f"  cpu {cpu_total:.1f} s  busy_cores {busy_cores:.2f}"
    )
    save_result(
        args.output,
        n_spinful_sites,
        n_layers,
        values.tolist(),
        term_counts.tolist(),
        cumulative_runtimes.tolist(),
        memory_size.tolist(),
        native_memory_size.tolist(),
        {
            "affinity_cores": len(os.sched_getaffinity(0)),
            "mu_gates": bool(args.mu_gates),
            "cpu_seconds": cpu_total,
            "busy_cores": busy_cores,
            "library_version": monoprop.__version__,
            "host": platform.node(),
        },
    )


if __name__ == "__main__":
    main()
