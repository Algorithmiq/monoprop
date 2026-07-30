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
from pathlib import Path
from time import perf_counter

import monoprop
import numpy as np
from monoprop import Circuit, ExpGate, MajoranaPropagator
from monoprop.fermi import FermiOperator

os.environ["YAQS_LOG_LEVEL"] = "INFO"


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


def process_cpu_seconds():
    """Return CPU seconds (user + system) consumed by this process, summed over all threads."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def save_result(output_path, record):
    """Append one benchmark result as a JSON line, creating the parent directory if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark for 1D Hubbard model")
    parser.add_argument("--case", "-c", help="Case pair to run.", type=int, default=0)
    parser.add_argument(
        "--output",
        "-o",
        help="Path to the JSONL file results are appended to.",
        default=Path(__file__).with_name("monoprop_hubbard1d_benchmark_results.jsonl"),
    )
    parser.add_argument(
        "--mu-gates",
        action="store_true",
        help="Keep the 2N inert chemical-potential gates (mu=0) that the Julia circuit lacks.",
    )

    args = parser.parse_args()

    spin_layer_cases = []
    for i in [20, 40, 60]:
        for j in range(10, 19, 2):
            spin_layer_cases.append((i, j))

    case_pair = args.case
    n_spinful_sites, n_layers = spin_layer_cases[case_pair]
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

    values[0] = simulator.expectation_value()
    term_counts[0] = simulator.size()

    cpu_start = process_cpu_seconds()
    t_start = perf_counter()
    for step in range(trotter_steps):
        simulator.propagate(fermi_circuit)
        values[step + 1] = simulator.expectation_value()
        term_counts[step + 1] = simulator.size()
    t_total = perf_counter() - t_start
    cpu_total = process_cpu_seconds() - cpu_start
    memory_size = simulator._simulator.operator_memory_bytes() / 1024**2
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # The engine picks one partition per core when the env var is unset, so an unset value is
    # not "1 thread"; busy_cores is the only measurement of what the threads actually did.
    requested_threads = os.environ.get("monoprop_NUM_THREADS")
    busy_cores = cpu_total / t_total if t_total > 0 else float("nan")
    print(
        f"{n_spinful_sites} n_spin {n_layers} layers {term_counts[-1]} num_terms {values[-1]} final overlap  runtime {t_total:.3f} seconds"
        f"  cpu {cpu_total:.1f} s  busy_cores {busy_cores:.2f}"
    )
    save_result(
        args.output,
        {
            "n_spinful_sites": n_spinful_sites,
            "n_layers": n_layers,
            "num_threads": int(requested_threads) if requested_threads else None,
            "affinity_cores": len(os.sched_getaffinity(0)),
            "mu_gates": bool(args.mu_gates),
            "runtime_seconds": t_total,
            "cpu_seconds": cpu_total,
            "busy_cores": busy_cores,
            "final_overlap": float(values[-1]),
            # np.int64 is not a subclass of int, so json.dumps rejects it as-is.
            "num_terms": int(term_counts[-1]),
            "memory_MB": memory_size,
            "peak_rss_MB": peak_rss_mb,
            "library_version": monoprop.__version__,
            "host": platform.node(),
        },
    )


if __name__ == "__main__":
    main()
