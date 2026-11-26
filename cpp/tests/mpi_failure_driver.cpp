// Copyright 2026 Algorithmiq
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Launcher-supervised distributed scenarios. Its own main(), outside the unit runner: every failure scenario ends the
// whole job. run_mpi_failure_scenario.cmake launches it on two ranks with a 30 s timeout and requires either a prompt
// nonzero exit carrying the expected diagnostic, or (the thread-support acceptance scenario) a clean exit carrying the
// expected completion line.
//
// Rank 0 fails while its peer follows the normal phase. A failure the library does not handle escapes to main(), which
// models a host that catches it and shuts down: rank 0 finalizes while its peer is still inside the operation, so the
// job hangs until the timeout. Every failure scenario therefore passes only through the library's own abort or
// fail-fast path:
//
//   build-failure                    a root at two threads per rank: rank 0's cutoff predicate throws in a build that
//                                    has started mutating, before any physical round
//   active-ticket                    the exported evolve_step on the sole shard of a root at one thread per rank: the
//                                    cosine pass throws on rank 0 while the replay Ticket is posted
//
// Thread support. The library requests MPI_THREAD_FUNNELED and requires at least FUNNELED from whoever initialized
// MPI; only the initializing thread, as the primary of the library's team, calls MPI. A launch whose provided level
// does not exercise the route ends with "NOT EXERCISED" (exit 77, reported as skipped):
//
//   host-level funneled              the host initializes MPI at exactly FUNNELED; real operations at two threads per
//                                    rank complete on the world and on a reordered split communicator
//   insufficient-thread-level single host-initialized MPI below FUNNELED is refused at construction
//   wrong-thread-entry funneled      a guarded entry called from a non-initializing std::thread fails fast locally

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "PropagatorTestAccess.h"
#include "monoprop/Evolution.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/OperationFailure.h"

namespace {

using namespace monoprop;

constexpr size_t num_modes = 6;
using Propagator = MonomialPropagator<num_modes>;
using Access = detail::PropagatorTestAccess<num_modes>;

// Exit code CTest reads as "skipped": the MPI library's provided level does not exercise the scenario's route.
constexpr int skip_exit_code = 77;

auto say(int rank, std::string_view text) -> void {
    std::fputs(std::format("[driver] rank {}: {}\n", rank, text).c_str(), stderr);
    std::fflush(stderr);
}

// Twenty terms anticommuting with the {0, 1} generator, so both ranks own work under any hash split.
auto hamiltonian() -> OperatorDict {
    OperatorDict ham;
    for (size_t j = 2; j < 2 * num_modes; ++j) {
        ham[VecZ{0, j}] = std::complex<double>{0.0, 0.1 * static_cast<double>(j)};
        ham[VecZ{1, j}] = std::complex<double>{0.0, 0.05 * static_cast<double>(j)};
    }
    return ham;
}

const std::vector<VecZ> circuit_gates{{0, 1}, {2, 3}, {1, 4}};
const VecZ circuit_mapping{0, 1, 2};
const VecD circuit_gen_coeffs{1.0, 1.0, 1.0};
const VecD circuit_params{0.3, 0.2, 0.1};

// A root at `threads` threads per rank: the budget is captured at construction, from monoprop_NUM_THREADS.
auto make_root(int threads, MPI_Comm comm = MPI_COMM_WORLD) -> std::unique_ptr<Propagator> {
    ::setenv("monoprop_NUM_THREADS", std::to_string(threads).c_str(), 1);
    return std::make_unique<Propagator>(hamiltonian(),
                                        static_cast<unsigned int>(2 * num_modes),
                                        VecZ{0, 1},
                                        std::nullopt,
                                        mpi::Comm(comm));
}

// --- failures after an operation has started ------------------------------------------------------------------------

// Rank 0's cutoff predicate throws in the first gate's traversal, after the build has started mutating and before any
// physical round; rank 1 goes on into the round.
auto build_failure(int rank) -> void {
    auto sim = make_root(2);
    if (rank == 0) {
        Access::set_cutoff_fn(*sim, [](const Monomial<num_modes> &) -> bool {
            throw std::runtime_error("injected traversal failure on rank 0");
        });
    }
    sim->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
}

// One evolve_step over the first layer of a built T = 1 root's sole shard, with `scale` as its cosine pass.
auto post_replay(int rank, const detail::LayerCosScale &scale) -> void {
    auto sim = make_root(1);
    sim->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    VecD coeffs = sim->mp_op().get_operator();
    const auto view = sim->graph().replay_view();
    say(rank, "posting the replay exchange");
    evolve_step(coeffs, view, 0.3, 0, MPI_COMM_WORLD, scale);
}

auto active_ticket(int rank) -> void {
    // Splitmix replays through MPI_Ialltoallv, so the Ticket always holds a posted request here.
    post_replay(rank, [rank](size_t, double *, double) {
        if (rank == 0) {
            throw std::runtime_error("injected failure while the replay Ticket is in flight");
        }
    });
}

// --- thread support ------------------------------------------------------------------------------------------------

auto close_to(double a, double b) -> bool {
    return std::abs(a - b) <= 1e-9 + (1e-7 * std::max(std::abs(a), std::abs(b)));
}

// Real distributed work at two threads per rank on `comm`: construction, graph build, energies, gradients, retained
// functionals, contraction, export, a copy, an initial-operator update and graph-free propagation; every result is
// checked against a single-process root of the same budget, so the physical rounds really carried data. Every MPI call
// inside the library comes from this (the initializing) thread as its team's primary. Returns the energy.
auto run_operations(int rank, MPI_Comm comm, std::string_view label) -> double {
    auto reference = make_root(2, MPI_COMM_SELF);
    reference->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    const auto [ref_energy, ref_gradient] = reference->expectation_value_and_gradient(circuit_params);

    auto sim = make_root(2, comm);
    sim->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    const double energy = sim->expectation_value(circuit_params);
    const auto [energy_g, gradient] = sim->expectation_value_and_gradient(circuit_params);
    const auto functional = sim->expectation_value_and_gradient_functional();
    const auto [energy_f, gradient_f] = functional(circuit_params);
    const auto pared = sim->expectation_value_functional(1e-10);
    const double energy_p = pared(circuit_params);
    const auto contracted = sim->contract_partially(circuit_params, false);
    const auto terms = sim->evolved_operator_terms(circuit_params, 0.0);
    const Propagator copy(*sim);
    const double energy_c = Propagator(copy).expectation_value(circuit_params);

    auto fresh = make_root(2, comm);
    fresh->propagate(circuit_gates, circuit_mapping, circuit_gen_coeffs, circuit_params);
    auto fresh_reference = make_root(2, MPI_COMM_SELF);
    fresh_reference->propagate(circuit_gates, circuit_mapping, circuit_gen_coeffs, circuit_params);
    const double propagated = fresh->expectation_value(VecD{});
    const double propagated_ref = fresh_reference->expectation_value(VecD{});

    auto updated = make_root(2, comm);
    updated->update_initial_operator(hamiltonian());
    updated->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    const double energy_u = updated->expectation_value(circuit_params);

    bool ok = close_to(energy, ref_energy) && close_to(energy_g, ref_energy) && close_to(energy_f, ref_energy)
              && close_to(energy_p, ref_energy) && close_to(energy_c, ref_energy) && close_to(energy_u, ref_energy)
              && close_to(propagated, propagated_ref) && gradient.size() == ref_gradient.size()
              && gradient_f.size() == ref_gradient.size();
    for (size_t i = 0; ok && i < ref_gradient.size(); ++i) {
        ok = close_to(gradient[i], ref_gradient[i]) && close_to(gradient_f[i], ref_gradient[i]);
    }
    const size_t global_terms = mpi::allreduce_sum<size_t>(terms.size(), mpi::Comm(comm));
    const size_t global_size = mpi::allreduce_sum<size_t>(sim->size(), mpi::Comm(comm));
    ok =
        ok && global_terms == reference->size() && global_size == reference->size() && contracted.size() == sim->size();
    if (!ok) {
        throw std::runtime_error(std::format("{}: results differ from the single-process reference (energy {} vs {})",
                                             label,
                                             energy,
                                             ref_energy));
    }
    say(rank,
        std::format("{}: energy {:.17g}, {} local of {} global terms, team {}",
                    label,
                    energy,
                    sim->size(),
                    global_size,
                    Access::options(*sim).threads));
    return energy;
}

auto thread_support_operations(int rank) -> void {
    const double world = run_operations(rank, MPI_COMM_WORLD, "world");
    // A supplied communicator whose rank order is the reverse of the world's: ownership moves, results do not.
    const int size = mpi::size(MPI_COMM_WORLD);
    MPI_Comm reordered = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, 0, size - 1 - rank, &reordered);
    const double split = run_operations(rank, reordered, "reordered split communicator");
    MPI_Comm_free(&reordered);
    if (!close_to(world, split)) {
        throw std::runtime_error("the reordered communicator changed the energy");
    }
}

auto insufficient_level(int rank) -> void {
    // The constructor validates the provided level before it communicates.
    const auto sim = make_root(2);
    say(rank, std::format("constructed with {} terms despite the insufficient level", sim->size()));
}

auto wrong_thread(int rank) -> void {
    auto sim = make_root(2);
    sim->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    if (rank == 0) {
        std::thread host([&sim] { (void)sim->expectation_value(circuit_params); });
        host.join();
    }
    else {
        (void)sim->expectation_value(circuit_params);
    }
}

auto parse_level(std::string_view name) -> std::optional<int> {
    if (name == "single") {
        return MPI_THREAD_SINGLE;
    }
    if (name == "funneled") {
        return MPI_THREAD_FUNNELED;
    }
    return std::nullopt;
}

} // namespace

auto main(int argc, char **argv) -> int {
    if (argc < 2) {
        std::fputs("usage: monoprop_mpi_failure_driver.x <scenario> [option]\n", stderr);
        return 2;
    }
    const std::string_view scenario = argv[1];
    const std::string_view option = argc > 2 ? argv[2] : "";
    // Dense collectives on every exchange, so each peer is committed to the round rank 0 abandons.
    ::setenv("monoprop_ROUTING", "splitmix", 1);

    const bool host_owned =
        scenario == "insufficient-thread-level" || scenario == "wrong-thread-entry" || scenario == "host-level";
    std::optional<int> requested;
    if (host_owned) {
        requested = parse_level(option);
        if (!requested) {
            std::fputs("a host-initialized scenario needs a thread level\n", stderr);
            return 2;
        }
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, *requested, &provided);
    }
    else {
        mpi::init(&argc, &argv);
    }
    const int rank = mpi::rank(MPI_COMM_WORLD);
    const int size = mpi::size(MPI_COMM_WORLD);
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    say(rank,
        std::format("scenario={} {} size={} MPI {} requested={} provided={}",
                    scenario,
                    option,
                    size,
                    host_owned ? "host-initialized" : "library-initialized",
                    requested ? mpi::thread_level_name(*requested) : mpi::thread_level_name(mpi::required_thread_level),
                    mpi::thread_level_name(provided)));

    const auto not_exercised = [&](std::string_view why) {
        say(rank,
            std::format("NOT EXERCISED: requested {} but the MPI library provided {}; {}",
                        mpi::thread_level_name(requested.value_or(mpi::required_thread_level)),
                        mpi::thread_level_name(provided),
                        why));
        MPI_Finalize();
        return skip_exit_code;
    };
    if (scenario == "insufficient-thread-level" && mpi::thread_level_satisfies(provided, mpi::required_thread_level)) {
        return not_exercised("the refusal needs a level below the required one");
    }
    if (scenario == "host-level" && provided != *requested) {
        return not_exercised("the acceptance run needs exactly the requested level");
    }

    try {
        if (scenario == "insufficient-thread-level") {
            insufficient_level(rank);
        }
        else if (scenario == "wrong-thread-entry") {
            wrong_thread(rank);
        }
        else if (scenario == "host-level") {
            thread_support_operations(rank);
            say(rank,
                std::format("thread-support operations completed with provided={}", mpi::thread_level_name(provided)));
        }
        else if (scenario == "build-failure") {
            build_failure(rank);
        }
        else if (scenario == "active-ticket") {
            active_ticket(rank);
        }
        else {
            say(rank, std::format("unknown scenario '{}' '{}'", scenario, option));
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    catch (const std::exception &e) {
        say(rank, std::format("UNGUARDED: an exception escaped the library: {}", e.what()));
        // A host that handled the error and shut down; peers still inside the operation hang.
        MPI_Finalize();
        return 3;
    }

    // The next collective of the peers' normal phase.
    MPI_Barrier(MPI_COMM_WORLD);
    say(rank, "scenario completed without a failure");
    MPI_Finalize();
    return 0;
}
