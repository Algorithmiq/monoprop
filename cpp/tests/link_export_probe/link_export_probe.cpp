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

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

#include <complex>
#include <cstdio>
#include <optional>
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

    std::fprintf(stderr,
                 "[link_export_probe] graph chain: layers=%zu cross_rank_bytes=%zu value=%f grad_size=%zu\n",
                 sim.graph_layers(),
                 mem.cross_rank_bytes,
                 value,
                 grad.size());
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

    std::fprintf(stderr, "[link_export_probe] partition chain: size=%zu\n", sim.size());
}

} // namespace

auto main() -> int {
    run_graph_build_chain();
    run_partition_chain();
    std::fprintf(stderr, "[link_export_probe] OK\n");
    return 0;
}
