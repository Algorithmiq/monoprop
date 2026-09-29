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

// Link-time export-visibility probe. Unlike monoprop_unit_tests.x (linked against monoprop-objs, the
// plain OBJECT library), this target links against the installed "monoprop" SHARED target, exactly as
// an external find_package(monoprop CONFIG) consumer would -- so it crosses the same hidden-visibility
// boundary. cpp/tests/CMakeLists.txt is compiled with "-Wl,--no-undefined" / "-Wl,-undefined,error" so
// the link step itself fails, reporting every symbol MonomialPropagator<NumModes>'s public template
// chain references but that the shared library does not export.
//
// Explicit class-template instantiation alone is not sufficient: it emits every member function's
// object code (including their calls into detail/** free functions), which is what the linker checks,
// but it does not run any of it. main() below actually drives both chains implicated by the bug report:
//  (a) the graph-building / Schrodinger path (detail/graph_encoding/MPGraphEncodingStorage.h), via
//      build_graph(), graph_memory_usage(), and expectation_value_and_gradient().
//  (b) the partition path (detail/partition/CpuTopology.h), via a partitions > 1 construction, which is
//      the only way to make PartitionGroup actually place and pin partition-worker threads.
// It also checks the installed usage requirements a consumer inherits:
//  (c) the OpenMP worksharing template (detail/parallel/Workshare.h), instantiated in this translation
//      unit, so OpenMP compile and link flags must reach the consumer through the imported target alone;
//  (d) the Pauli basis, alongside the Majorana graph chain in (a);
//  (e) the one-store prototype (partitions = 1), whose constructor template calls the out-of-line
//      capture_thread_budget(), require_thread_support() and require_initializing_thread(), and whose
//      operations reach the exported evolve/derivative entry points with their trailing Options.
//  (f) the fixed-team phase primitive (detail/sharded/Team.h), instantiated here like (c): its region,
//      barriers and masked construct must compile and link through the imported target's OpenMP flags.

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/parallel/Workshare.h"
#include "monoprop/detail/sharded/Team.h"

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <optional>
#include <print>
#include <stdexcept>
#include <vector>

// Forces every member function of MonomialPropagator<NumModes> to be compiled for these two
// representative widths, regardless of which ones main() below happens to call.
template class monoprop::MonomialPropagator<2>;
template class monoprop::MonomialPropagator<6>;

namespace {

using namespace monoprop;

// Drives Engine::finish() -> LayerBuildSink::finalize() -> build_layer_storage_unified(), plus the
// graph-memory and gradient accessors that read the resulting PackedCrossRankStorage / exchange layout.
auto run_graph_build_chain() -> void {
    constexpr size_t kModes = 2;
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    const VecZ initial_state{0, 1};

    MonomialPropagator<kModes> sim(ham,
                                   2 * kModes,
                                   initial_state,
                                   /*schrodinger_cutoff=*/std::optional<unsigned int>{4U},
                                   MPI_COMM_SELF,
                                   /*lower_atol=*/std::nullopt,
                                   /*upper_atol=*/std::nullopt,
                                   CutoffType::Length,
                                   /*basis_change=*/std::nullopt);

    const std::vector<VecZ> monos{{0}, {1}, {2}};
    sim.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0});

    const auto mem = sim.graph_memory_usage();
    const auto [value, grad] = sim.expectation_value_and_gradient(VecD{0.1, 0.2, 0.3});

    std::println(stderr,
                 "[link_export_probe] graph chain: layers={} cross_rank_bytes={} value={} grad_size={}",
                 sim.graph_layers(),
                 mem.cross_rank_bytes,
                 value,
                 grad.size());
}

// Without the imported target's OpenMP compile flags the pragmas in Workshare.h would silently compile
// out, so check for them explicitly rather than rely on a link failure.
#ifndef _OPENMP
#error "monoprop::monoprop did not propagate the OpenMP compile flags to this consumer"
#endif

// Instantiates for_blocks here: its region, and the omp_* calls it makes, are compiled and linked in
// the consumer. The body only writes its own element; library calls stay outside the region.
auto run_openmp_workshare_chain() -> void {
    constexpr size_t kBlocks = 257;
    std::vector<size_t> squares(kBlocks, 0);
    monoprop::detail::parallel::for_blocks(kBlocks, {.threads = 2}, [&](size_t b) { squares[b] = b * b; });
    size_t sum = 0;
    for (const auto s : squares) {
        sum += s;
    }
    const size_t expected = (kBlocks - 1) * kBlocks * (2 * kBlocks - 1) / 6;
    std::println(stderr, "[link_export_probe] openmp chain: blocks={} sum={} expected={}", kBlocks, sum, expected);
    if (sum != expected) {
        std::println(stderr, "[link_export_probe] openmp chain: FAILED");
        std::exit(1);
    }
}

// Teams at the captured budget: in the first, every owner publishes in one phase and reads its neighbour's value in
// the next; in the second, the last owner fails and its exception must come back as the returned error.
auto run_sharded_team_chain() -> void {
    namespace sharded = monoprop::detail::sharded;
    const auto options = monoprop::detail::parallel::capture_thread_budget();
    const auto threads = static_cast<size_t>(options.threads);
    std::vector<size_t> published(threads, 0);
    std::vector<size_t> seen(threads, 0);
    const auto ok = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        if (sharded::phase(failure, shard, [&] { published[shard] = shard + 1; })) {
            sharded::phase(failure, shard, [&] { seen[shard] = published[(shard + 1) % threads]; });
        }
    });
    const auto failed = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        sharded::phase(failure, shard, [&] {
            if (shard + 1 == threads) {
                throw std::runtime_error("expected");
            }
        });
    });
    auto correct = !ok && static_cast<bool>(failed);
    for (size_t shard = 0; shard < threads; ++shard) {
        correct = correct && seen[shard] == (shard + 1) % threads + 1;
    }
    std::println(stderr, "[link_export_probe] sharded team chain: team={} correct={}", threads, correct);
    if (!correct) {
        std::println(stderr, "[link_export_probe] sharded team chain: FAILED");
        std::exit(1);
    }
}

// The Pauli basis: slots 2q/2q+1 are qubit q's x/z planes, so {0, 1} is Z on qubit 0 and {0} is X.
auto run_pauli_chain() -> void {
    constexpr size_t kQubits = 3;
    OperatorDict observable;
    observable[VecZ{0, 1}] = std::complex<double>{1.0, 0.0};

    MonomialPropagator<kQubits> sim(observable,
                                    /*cutoff=*/2 * kQubits,
                                    VecZ{},
                                    /*schrodinger_cutoff=*/std::nullopt,
                                    MPI_COMM_SELF,
                                    /*lower_atol=*/std::nullopt,
                                    /*upper_atol=*/std::nullopt,
                                    CutoffType::Support,
                                    /*basis_change=*/std::nullopt,
                                    /*logical_num_modes=*/kQubits,
                                    Basis::Pauli);
    sim.propagate(std::vector<VecZ>{{0}}, VecZ{0}, VecD{1.0}, VecD{0.3});

    std::println(stderr, "[link_export_probe] pauli chain: size={}", sim.size());
}

// Drives PartitionGroup's constructor: partitions > 1 is required for the facade to actually exist, so
// enumerate_physical_cores / affinity_mask_words / summarize_masks / format_place_line / partition_cpusets /
// pin_this_thread all run for real (not merely compiled) on the master threads it spawns.
auto run_partition_chain() -> void {
    constexpr size_t kModes = 6;
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};

    MonomialPropagator<kModes> sim(ham,
                                   2 * kModes,
                                   VecZ{0, 1},
                                   /*schrodinger_cutoff=*/std::nullopt,
                                   MPI_COMM_SELF,
                                   /*lower_atol=*/std::nullopt,
                                   /*upper_atol=*/std::nullopt,
                                   CutoffType::Length,
                                   /*basis_change=*/std::nullopt,
                                   /*logical_num_modes=*/kModes,
                                   Basis::Majorana,
                                   /*partitions=*/2);

    std::println(stderr, "[link_export_probe] partition chain: size={}", sim.size());
}

auto run_one_store_prototype_chain() -> void {
    constexpr size_t kModes = 2;
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    MonomialPropagator<kModes> sim(ham,
                                   2 * kModes,
                                   VecZ{0, 1},
                                   /*schrodinger_cutoff=*/std::nullopt,
                                   MPI_COMM_SELF,
                                   /*lower_atol=*/std::nullopt,
                                   /*upper_atol=*/std::nullopt,
                                   CutoffType::Length,
                                   /*basis_change=*/std::nullopt,
                                   /*logical_num_modes=*/kModes,
                                   Basis::Majorana,
                                   /*partitions=*/1);
    sim.build_graph(std::vector<VecZ>{{0, 2}, {1, 3}}, VecZ{0, 1}, VecD{1.0, 1.0});
    const auto [value, grad] = sim.expectation_value_and_gradient(VecD{0.1, 0.2});
    const auto budget = monoprop::detail::parallel::capture_thread_budget();
    std::println(stderr,
                 "[link_export_probe] one-store prototype chain: budget={} value={} grad_size={}",
                 budget.threads,
                 value,
                 grad.size());
}

} // namespace

auto main() -> int {
    monoprop::mpi::init();
    run_graph_build_chain();
    run_partition_chain();
    run_openmp_workshare_chain();
    run_sharded_team_chain();
    run_pauli_chain();
    run_one_store_prototype_chain();
    monoprop::mpi::finalize();
    std::println(stderr, "[link_export_probe] OK");
    return 0;
}
