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

"""Benchmark expectation_value_and_gradient_functional on a large random problem.

Default workload:
- 50 qubits (50 fermionic modes)
- 10,000-term random observable
- 100 random gates
- multiple repeated evaluations

Reports both runtime and memory (PSS-based peak + resting footprint).
"""

from __future__ import annotations

import argparse
import statistics
import time

try:
    from _builders import build_random_propagator, make_random_problem
    from _memory import PssSampler, merge_peak_of_sum, resting_pss_bytes
except ImportError:
    from benches._builders import build_random_propagator, make_random_problem
    from benches._memory import PssSampler, merge_peak_of_sum, resting_pss_bytes


def _bytes_to_mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubits", type=int, default=50, help="Number of qubits/modes."
    )
    parser.add_argument(
        "--terms", type=int, default=10_000, help="Number of observable terms."
    )
    parser.add_argument(
        "--gates", type=int, default=100, help="Number of random gates."
    )
    parser.add_argument("--cutoff", type=int, default=8, help="Propagator cutoff.")
    parser.add_argument(
        "--repeats", type=int, default=5, help="Number of timed evaluations."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("Building random benchmark problem...")
    print(
        f"qubits={args.qubits}, terms={args.terms}, gates={args.gates}, "
        f"cutoff={args.cutoff}, repeats={args.repeats}, seed={args.seed}"
    )

    problem = make_random_problem(
        num_modes=args.qubits,
        obs_terms=args.terms,
        num_generators=args.gates,
        cutoff=args.cutoff,
        seed=args.seed,
    )
    propagator, circuit = build_random_propagator(problem)

    # Build graph once; benchmark the functional evaluation itself.
    propagator.build_graph(circuit)
    functional = propagator.expectation_value_and_gradient_functional()
    params = problem.parameters

    print("Warming up functional call...")
    value, grad = functional(params)
    print(f"warmup: value={value:.12g}, gradient_size={len(grad)}")

    resting_before = resting_pss_bytes()

    times_s: list[float] = []
    peak_pss_bytes: list[int] = []

    print("Running timed repetitions...")
    for i in range(args.repeats):
        with PssSampler() as sampler:
            t0 = time.perf_counter()
            value_i, grad_i = functional(params)
            dt = time.perf_counter() - t0

        times_s.append(dt)
        peak = merge_peak_of_sum([sampler.samples])
        peak_pss_bytes.append(peak)

        print(
            f"run={i + 1:02d}  time={dt:.6f}s  "
            f"peak_pss={_bytes_to_mib(peak):.2f} MiB  "
            f"value={value_i:.12g}  grad_norm={(sum(g * g for g in grad_i) ** 0.5):.6g}"
        )

    resting_after = resting_pss_bytes()

    print("\nSummary")
    print(f"time_mean={statistics.mean(times_s):.6f}s")
    print(f"time_median={statistics.median(times_s):.6f}s")
    print(f"time_min={min(times_s):.6f}s")
    print(f"time_max={max(times_s):.6f}s")
    print(
        f"peak_pss_mean={_bytes_to_mib(int(statistics.mean(peak_pss_bytes))):.2f} MiB"
    )
    print(f"peak_pss_max={_bytes_to_mib(max(peak_pss_bytes)):.2f} MiB")
    print(f"resting_pss_before={_bytes_to_mib(resting_before):.2f} MiB")
    print(f"resting_pss_after={_bytes_to_mib(resting_after):.2f} MiB")


if __name__ == "__main__":
    main()
