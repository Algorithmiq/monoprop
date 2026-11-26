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
// nonzero exit carrying the expected diagnostic, or (the thread-support acceptance scenarios) a clean exit carrying the
// expected completion line.
//
// Rank 0 fails while its peers follow the normal phase. A failure the library does not handle escapes to main(), which
// models a host that catches it and shuts down: rank 0 finalizes while its peers are still inside the operation, so the
// job hangs until the timeout. Every failure scenario therefore passes only through the library's own abort or
// fail-fast path. The sharded root runs at two OpenMP threads per rank and is injected through its test-only
// RootObserver on rank 0 (rank 1 for the malformed receive):
//
//   worker-throw                     worker 1 throws in a replay finish of an evaluation
//   before-exchange                  the primary throws in the first gate's traversal, before any physical round
//   active-ticket round              the primary throws in a cosine pass while the replay round is posted
//   active-ticket counts             the primary throws while packing queries, with the count round posted
//   active-ticket local-round        as `round`, through the exported evaluation seam with a caller-local round
//                                    (no persistent rounds): the live failure must reach the abort before that
//                                    round's draining destructor runs
//   malformed-receive                rank 1 rewrites its outgoing query blocks; rank 0's owners must refuse them
//
// The exported replay functions, on the sole shard of a root launched with one thread per rank (raw access is
// available at T = 1, also over several ranks), with explicit kernel budgets:
//
//   active-ticket ticket|cosine-worker   a failure (on the caller, or on worker 1 of the threaded cosine kernel) while
//                                        evolve_step's replay Ticket is posted
//
// Thread support. The library requests MPI_THREAD_FUNNELED and requires at least FUNNELED from whoever initialized
// MPI; only the initializing thread, as the primary of the library's team, calls MPI. Each line below records the
// requested and the provided level; a launch whose provided level does not exercise the route ends with
// "NOT EXERCISED" (exit 77, reported as skipped):
//
//   library-level                    the library initializes MPI; real operations complete (success expected)
//   host-level funneled|serialized|multiple
//                                    the host initializes MPI at that level; real operations complete on the world and
//                                    on a reordered split communicator (success expected; exactly FUNNELED for
//                                    funneled)
//   insufficient-thread-level single host-initialized MPI below FUNNELED is refused at construction
//   wrong-thread-entry funneled|serialized|multiple
//                                    a guarded entry called from a non-initializing std::thread fails fast locally

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
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
#include "monoprop/MPFunctions.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/sharded/Exchange.h"
#include "monoprop/detail/sharded/RootObserver.h"

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

// --- the sharded root, injected through its observer ---------------------------------------------------------------

// Throws once, on `rank` only, at the armed visit of the sharded root's seams; or rewrites outgoing queries.
class Injector final : public detail::sharded::RootObserver {
public:
    struct Arm {
        bool construction = false;
        int work = 0;
        size_t shard = 0;
        const char *what = "";
    };
    Injector(int rank, std::optional<Arm> arm, bool corrupt) : rank_(rank), arm_(arm), corrupt_(corrupt) {}

    auto visit(detail::sharded::EvaluationWork work, size_t /*step*/, size_t shard) const -> void override {
        fire(false, static_cast<int>(work), shard);
    }
    auto construction(detail::sharded::ConstructionWork work, size_t /*step*/, size_t shard) const -> void override {
        fire(true, static_cast<int>(work), shard);
    }
    auto queries_packed(size_t /*step*/, size_t shard, detail::sharded::PhysicalExchange &round) const
        -> void override {
        if (!corrupt_) {
            return;
        }
        // Every word all ones: a record whose phase field is not ternary.
        for (size_t k = 0; k < round.peers().size(); ++k) {
            for (size_t t = 0; t < round.threads(); ++t) {
                for (auto &word : round.send_block<size_t>(shard, k, t)) {
                    word = ~size_t{0};
                }
            }
        }
    }

private:
    auto fire(bool construction, int work, size_t shard) const -> void {
        if (arm_ && arm_->construction == construction && arm_->work == work && arm_->shard == shard) {
            throw std::runtime_error(
                std::format("injected {} on OpenMP worker {} of rank {}", arm_->what, omp_get_thread_num(), rank_));
        }
    }

    int rank_;
    std::optional<Arm> arm_;
    bool corrupt_;
};

// A root at `threads` threads per rank: the budget is captured at construction, from monoprop_NUM_THREADS.
auto make_root(int threads, const detail::sharded::RootObserver *observer, MPI_Comm comm = MPI_COMM_WORLD)
    -> std::unique_ptr<Propagator> {
    ::setenv("monoprop_NUM_THREADS", std::to_string(threads).c_str(), 1);
    return Access::construct_observed(observer,
                                      hamiltonian(),
                                      static_cast<unsigned int>(2 * num_modes),
                                      VecZ{0, 1},
                                      std::optional<unsigned int>{},
                                      mpi::Comm(comm),
                                      std::optional<double>{},
                                      std::optional<double>{},
                                      CutoffType::Length,
                                      std::optional<std::vector<VecZ>>{},
                                      num_modes,
                                      Basis::Majorana);
}

auto sharded_scenario(int rank, std::string_view scenario, std::string_view option) -> void {
    using CW = detail::sharded::ConstructionWork;
    using EW = detail::sharded::EvaluationWork;
    std::optional<Injector::Arm> arm;
    bool corrupt = false;
    if (scenario == "worker-throw") {
        arm = Injector::Arm{false, static_cast<int>(EW::finish), 1, "replay-finish failure"};
    }
    else if (scenario == "before-exchange") {
        arm = Injector::Arm{true, static_cast<int>(CW::traverse), 0, "traversal failure before any physical round"};
    }
    else if (scenario == "active-ticket" && option == "round") {
        arm = Injector::Arm{false, static_cast<int>(EW::cosine), 0, "cosine failure while the replay round is posted"};
    }
    else if (scenario == "active-ticket" && option == "counts") {
        arm = Injector::Arm{true, static_cast<int>(CW::pack), 0, "pack failure while the count round is posted"};
    }
    else if (scenario == "active-ticket" && option == "local-round") {
        arm = Injector::Arm{false,
                            static_cast<int>(EW::cosine),
                            0,
                            "cosine failure while a caller-local replay round is posted"};
    }
    else if (scenario == "malformed-receive") {
        corrupt = rank == 1;
    }
    const Injector injector(rank, rank == 0 ? arm : std::nullopt, corrupt);
    auto sim = make_root(2, &injector);
    say(rank, std::format("{} local terms, team {}", sim->size(), Access::options(*sim).threads));
    sim->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    if (option == "local-round") {
        const auto outcome = Access::evaluate_with_local_round(*sim, circuit_params, &injector);
        if (outcome.error) {
            std::rethrow_exception(outcome.error); // a caller that got the failure back; its peer is still in a round
        }
        return;
    }
    (void)sim->expectation_value(circuit_params);
}

// --- the exported replay functions on the sole shard of a T = 1 root -----------------------------------------------

// One evolve_step over the first layer of a built T = 1 root's sole shard, with `scale` as its cosine pass.
auto post_replay(int rank, const detail::LayerCosScale &scale) -> void {
    auto sim = make_root(1, nullptr);
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

// Throws from one logical range of a real kernel, on whichever worker owns it.
struct ThrowingRangeObserver {
    auto prepare(detail::KernelRange, size_t) const noexcept -> void {}
    auto visit(detail::KernelRange, size_t range) const -> void {
        if (range == 2) {
            throw std::runtime_error(std::format("injected failure in cosine range {} on OpenMP worker {} while the "
                                                 "replay Ticket is in flight",
                                                 range,
                                                 omp_get_thread_num()));
        }
    }
};

auto active_ticket_cosine_worker(int rank) -> void {
    // A stored mask of four kernel ranges, so the kernel really opens a team; the real layer is tiny.
    CosMask mask;
    for (size_t w = 0; w < 4 * detail::cos_mask_range_blocks; ++w) {
        mask.blocks.emplace_back(w * 64, ~uint64_t{0});
        mask.total_count += 64;
    }
    auto scratch = std::make_shared<VecD>(mask.blocks.size() * 64, 1.0);
    post_replay(rank, [rank, mask, scratch](size_t, double *, double v) {
        if (rank == 0) {
            detail::scale_cos_mask(scratch->data(), mask, v, {.threads = 2}, ThrowingRangeObserver{});
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
    auto reference = make_root(2, nullptr, MPI_COMM_SELF);
    reference->build_graph(circuit_gates, circuit_mapping, circuit_gen_coeffs);
    const auto [ref_energy, ref_gradient] = reference->expectation_value_and_gradient(circuit_params);

    auto sim = make_root(2, nullptr, comm);
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

    auto fresh = make_root(2, nullptr, comm);
    fresh->propagate(circuit_gates, circuit_mapping, circuit_gen_coeffs, circuit_params);
    auto fresh_reference = make_root(2, nullptr, MPI_COMM_SELF);
    fresh_reference->propagate(circuit_gates, circuit_mapping, circuit_gen_coeffs, circuit_params);
    const double propagated = fresh->expectation_value(VecD{});
    const double propagated_ref = fresh_reference->expectation_value(VecD{});

    auto updated = make_root(2, nullptr, comm);
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
    const auto sim = make_root(2, nullptr);
    say(rank, std::format("constructed with {} terms despite the insufficient level", sim->size()));
}

auto wrong_thread(int rank) -> void {
    auto sim = make_root(2, nullptr);
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
    if (name == "serialized") {
        return MPI_THREAD_SERIALIZED;
    }
    if (name == "multiple") {
        return MPI_THREAD_MULTIPLE;
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
        else if (scenario == "host-level" || scenario == "library-level") {
            thread_support_operations(rank);
            say(rank,
                std::format("thread-support operations completed with provided={}", mpi::thread_level_name(provided)));
        }
        else if (scenario == "active-ticket" && option == "local-round") {
            sharded_scenario(rank, scenario, option);
        }
        else if (scenario == "active-ticket" && option == "ticket") {
            active_ticket(rank);
        }
        else if (scenario == "active-ticket" && option == "cosine-worker") {
            active_ticket_cosine_worker(rank);
        }
        else if (scenario == "worker-throw" || scenario == "before-exchange" || scenario == "active-ticket"
                 || scenario == "malformed-receive") {
            sharded_scenario(rank, scenario, option);
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
