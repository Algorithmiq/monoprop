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

// Launcher-supervised distributed failure scenarios. Its own main(), outside the unit runner: every
// scenario ends the whole job. run_mpi_failure_scenario.cmake launches it on two ranks with a 30 s
// timeout and requires a prompt nonzero exit carrying the expected diagnostic.
//
// Rank 0 fails while its peers follow the normal phase. A failure the library does not handle escapes to
// main(), which models a host that catches it and shuts down: rank 0 finalizes while its peers are still
// inside the operation, so the job hangs until the timeout. Every scenario therefore passes only through
// the library's own abort or fail-fast path.
//
//   worker-throw                     worksharing worker throws inside a retained-functional evaluation
//   before-exchange                  a build fails in the scan, before the first query exchange
//   before-exchange scan-worker      a worker of the threaded scan throws in one of its word ranges, before
//                                    the first query exchange the peers enter
//   resolve-worker self-probe        a worker of the threaded self probe throws, before the first query
//                                    exchange the peers enter
//   resolve-worker decode|incoming-probe|scatter
//                                    a worker of the threaded incoming decode / frozen probe / scatter throws
//                                    after the query round, while the peers enter the response round
//   active-ticket ticket|pending     a failure while a replay Ticket / construction PendingAlltoallv is posted
//   active-ticket cosine-worker      a worker of the threaded cosine kernel throws while the replay Ticket is
//                                    posted
//   insufficient-thread-level single|funneled   host-initialized MPI below the required level
//   wrong-thread-entry serialized|multiple      a guarded entry called from a non-initializing std::thread

#include <mpi.h>
#include <omp.h>

#include <algorithm>
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
#include "monoprop/detail/parallel/Workshare.h"

namespace {

using namespace monoprop;

constexpr size_t kModes = 6;
using Propagator = MonomialPropagator<kModes>;
using Access = detail::PropagatorTestAccess<kModes>;

// Exit code CTest reads as "skipped": the MPI library gave more support than the scenario needs.
constexpr int kNotExercised = 77;

auto say(int rank, std::string_view text) -> void {
    std::fputs(std::format("[driver] rank {}: {}\n", rank, text).c_str(), stderr);
    std::fflush(stderr);
}

auto world_rank() -> int {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

// Twenty terms anticommuting with the {0, 1} generator, so both ranks own scan work under any hash split.
auto hamiltonian() -> OperatorDict {
    OperatorDict ham;
    for (size_t j = 2; j < 2 * kModes; ++j) {
        ham[VecZ{0, j}] = std::complex<double>{0.0, 0.1 * static_cast<double>(j)};
        ham[VecZ{1, j}] = std::complex<double>{0.0, 0.05 * static_cast<double>(j)};
    }
    return ham;
}

auto make_prototype() -> Propagator {
    return Propagator(hamiltonian(),
                      2 * kModes,
                      VecZ{0, 1},
                      std::nullopt,
                      MPI_COMM_WORLD,
                      std::nullopt,
                      std::nullopt,
                      CutoffType::Length,
                      std::nullopt,
                      kModes,
                      Basis::Majorana,
                      /*partitions=*/1);
}

const std::vector<VecZ> kGates{{0, 1}, {2, 3}, {1, 4}};
const VecZ kMapping{0, 1, 2};
const VecD kGenCoeffs{1.0, 1.0, 1.0};
const VecD kParams{0.3, 0.2, 0.1};

auto build(Propagator &sim) -> void {
    sim.build_graph(kGates, kMapping, kGenCoeffs);
}

auto worker_throw(int rank) -> void {
    auto sim = make_prototype();
    build(sim);
    auto functional = Access::make_functional(
        sim,
        [rank](const EvalRequest &request, mpi::Comm comm, const detail::CosCallbacks &cos) -> double {
            if (rank == 0) {
                detail::parallel::for_blocks(4, {.threads = 2}, [](size_t block) {
                    if (block == 3) {
                        throw std::runtime_error(
                            std::format("injected worker failure on OpenMP worker {}", omp_get_thread_num()));
                    }
                });
            }
            return ev(request, comm, cos);
        });
    (void)functional(kParams);
}

auto before_exchange(int rank) -> void {
    auto sim = make_prototype();
    if (rank == 0) {
        Access::set_cutoff_fn(sim, [](const Monomial<kModes> &) -> bool {
            throw std::runtime_error("injected failure in the scan, before the query exchange");
        });
    }
    build(sim);
    if (rank == 0) {
        say(rank, "INJECTION NOT REACHED: the scan never evaluated the cutoff");
    }
}

// Throws from one word range of the real threaded scan, on whichever worker owns it.
struct ThrowingScanObserver {
    int rank;
    auto prepare(detail::KernelRange, size_t) const noexcept -> void {}
    auto visit(detail::KernelRange kind, size_t range) const -> void {
        if (rank == 0 && kind == detail::KernelRange::scan && range == 1) {
            throw std::runtime_error(
                std::format("injected failure in scan range {} on OpenMP worker {}", range, omp_get_thread_num()));
        }
    }
};

// Enough terms that each rank's store spans two fold blocks, so the scan really opens a two-worker team.
constexpr size_t kWideModes = 16;

auto scan_worker(int rank) -> void {
    OperatorDict ham;
    uint64_t state = 0x5CA3ULL;
    while (ham.size() < 200000) {
        VecZ idx;
        while (idx.size() < 6) { // weight 6: C(32, 6) ~ 9e5 distinct terms, and an imaginary Hermitian coefficient
            state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
            const size_t m = (state >> 33) % (2 * kWideModes);
            if (std::ranges::find(idx, m) == idx.end()) {
                idx.push_back(m);
            }
        }
        std::ranges::sort(idx);
        ham[idx] = std::complex<double>{0.0, 1e-3};
    }
    ::setenv("monoprop_NUM_THREADS", "2", 1);
    MonomialPropagator<kWideModes> sim(ham,
                                       2 * kWideModes,
                                       VecZ{0, 1},
                                       std::nullopt,
                                       MPI_COMM_WORLD,
                                       std::nullopt,
                                       std::nullopt,
                                       CutoffType::Length,
                                       std::nullopt,
                                       kWideModes,
                                       Basis::Majorana,
                                       /*partitions=*/1);
    say(rank,
        std::format("{} local terms, {} words, budget {}",
                    sim.size(),
                    sim.mp_op().inverted_index().words(),
                    detail::PropagatorTestAccess<kWideModes>::options(sim).threads));
    (void)detail::PropagatorTestAccess<kWideModes>::build_layer_observed(sim, VecZ{0, 5}, ThrowingScanObserver{rank});
    if (rank == 0) {
        say(rank, "INJECTION NOT REACHED: the scan ran in fewer than two ranges");
    }
}

auto active_ticket(int rank) -> void {
    auto sim = make_prototype();
    build(sim);
    VecD coeffs = sim.mp_op().get_operator();
    const auto view = sim.graph().replay_view();
    // Splitmix replays through MPI_Ialltoallv, so the Ticket always holds a posted request here.
    const detail::LayerCosScale scale = [rank](size_t, double *, double) {
        if (rank == 0) {
            throw std::runtime_error("injected failure while the replay Ticket is in flight");
        }
    };
    say(rank, "posting the replay exchange");
    evolve_step(coeffs, view, 0.3, 0, MPI_COMM_WORLD, scale);
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
    auto sim = make_prototype();
    build(sim);
    VecD coeffs = sim.mp_op().get_operator();
    const auto view = sim.graph().replay_view();
    // A stored mask of four kernel ranges, so the kernel really opens a team; the real layer is tiny.
    CosMask mask;
    for (size_t w = 0; w < 4 * detail::kCosMaskRangeBlocks; ++w) {
        mask.blocks.emplace_back(w * 64, ~uint64_t{0});
        mask.total_count += 64;
    }
    auto scratch = std::make_shared<VecD>(mask.blocks.size() * 64, 1.0);
    const detail::LayerCosScale scale = [rank, mask, scratch](size_t, double *, double v) {
        if (rank == 0) {
            detail::scale_cos_mask(scratch->data(), mask, v, {.threads = 2}, ThrowingRangeObserver{});
        }
    };
    say(rank, "posting the replay exchange");
    evolve_step(coeffs, view, 0.3, 0, MPI_COMM_WORLD, scale);
}

auto active_pending(int rank, int size) -> void {
    const mpi::SlotWindow window{.base = 0, .count = static_cast<size_t>(size)};
    mpi::WindowVec<std::vector<double>> send(window);
    for (const auto wi : window.indices()) {
        send[wi].assign(4, static_cast<double>(rank));
    }
    auto pending = mpi::begin_alltoallv(send, MPI_COMM_WORLD);
    say(rank, std::format("PendingAlltoallv posted {} request(s)", pending.posted));
    if (rank == 0) {
        mpi::guard_distributed(MPI_COMM_WORLD, [] {
            throw std::runtime_error("injected failure while a PendingAlltoallv is in flight");
        });
    }
    mpi::WindowVec<std::vector<double>> recv;
    pending.wait_into(recv);
}

auto insufficient_level(int rank) -> void {
    // The constructor validates the provided level before it communicates.
    const auto sim = make_prototype();
    say(rank, std::format("constructed with {} terms despite the insufficient level", sim.size()));
}

auto wrong_thread(int rank) -> void {
    auto sim = make_prototype();
    build(sim);
    if (rank == 0) {
        std::thread host([&sim] { (void)sim.expectation_value(kParams); });
        host.join();
    }
    else {
        (void)sim.expectation_value(kParams);
    }
}

// Throws from the last logical range of one resolve phase on rank 0, so with a two-worker team the failure
// starts on worker 1, off the calling thread. Only the first call of that phase with two or more ranges throws.
struct ThrowingResolveObserver {
    int rank;
    detail::KernelRange kind;
    const char *label;
    std::shared_ptr<size_t> ranges = std::make_shared<size_t>(0);

    auto prepare(detail::KernelRange k, size_t n) const noexcept -> void {
        if (k == kind && *ranges < 2) {
            *ranges = n;
        }
    }
    auto visit(detail::KernelRange k, size_t range) const -> void {
        if (rank == 0 && k == kind && *ranges >= 2 && range + 1 == *ranges) {
            throw std::runtime_error(
                std::format("injected failure in {} range {} on OpenMP worker {}", label, range, omp_get_thread_num()));
        }
    }
};

auto resolve_worker(int rank, std::string_view phase) -> void {
    OperatorDict ham;
    uint64_t state = 0x5CA3ULL;
    while (ham.size() < 200000) {
        VecZ idx;
        while (idx.size() < 6) {
            state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
            const size_t m = (state >> 33) % (2 * kWideModes);
            if (std::ranges::find(idx, m) == idx.end()) {
                idx.push_back(m);
            }
        }
        std::ranges::sort(idx);
        ham[idx] = std::complex<double>{0.0, 1e-3};
    }
    ::setenv("monoprop_NUM_THREADS", "2", 1);
    MonomialPropagator<kWideModes> sim(ham,
                                       2 * kWideModes,
                                       VecZ{0, 1},
                                       std::nullopt,
                                       MPI_COMM_WORLD,
                                       std::nullopt,
                                       std::nullopt,
                                       CutoffType::Length,
                                       std::nullopt,
                                       kWideModes,
                                       Basis::Majorana,
                                       /*partitions=*/1);
    const auto kind = phase == "self-probe" ? detail::KernelRange::self_probe
                      : phase == "decode"   ? detail::KernelRange::decode
                      : phase == "scatter"  ? detail::KernelRange::scatter
                                            : detail::KernelRange::incoming_probe;
    say(rank,
        std::format("{} local terms, budget {}",
                    sim.size(),
                    detail::PropagatorTestAccess<kWideModes>::options(sim).threads));
    (void)detail::PropagatorTestAccess<kWideModes>::build_layer_observed(
        sim,
        VecZ{0, 5},
        ThrowingResolveObserver{rank, kind, phase.data()});
    if (rank == 0) {
        say(rank, std::format("INJECTION NOT REACHED: the {} phase ran in fewer than two ranges", phase));
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

    const bool host_owned = scenario == "insufficient-thread-level" || scenario == "wrong-thread-entry";
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
    const int rank = world_rank();
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    say(rank,
        std::format("scenario={} {} size={} MPI {} requested={} provided={}",
                    scenario,
                    option,
                    size,
                    host_owned ? "host-initialized" : "library-initialized",
                    requested ? mpi::thread_level_name(*requested) : mpi::thread_level_name(mpi::kRequiredThreadLevel),
                    mpi::thread_level_name(provided)));

    if (scenario == "insufficient-thread-level" && mpi::thread_level_satisfies(provided, mpi::kRequiredThreadLevel)) {
        say(rank,
            std::format("NOT EXERCISED: requested {} but the MPI library provided {}",
                        mpi::thread_level_name(*requested),
                        mpi::thread_level_name(provided)));
        MPI_Finalize();
        return kNotExercised;
    }

    try {
        if (scenario == "worker-throw") {
            worker_throw(rank);
        }
        else if (scenario == "before-exchange" && option == "scan-worker") {
            scan_worker(rank);
        }
        else if (scenario == "resolve-worker"
                 && (option == "self-probe" || option == "decode" || option == "incoming-probe"
                     || option == "scatter")) {
            resolve_worker(rank, option);
        }
        else if (scenario == "before-exchange") {
            before_exchange(rank);
        }
        else if (scenario == "active-ticket" && option == "ticket") {
            active_ticket(rank);
        }
        else if (scenario == "active-ticket" && option == "cosine-worker") {
            active_ticket_cosine_worker(rank);
        }
        else if (scenario == "active-ticket" && option == "pending") {
            active_pending(rank, size);
        }
        else if (scenario == "insufficient-thread-level") {
            insufficient_level(rank);
        }
        else if (scenario == "wrong-thread-entry") {
            wrong_thread(rank);
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
