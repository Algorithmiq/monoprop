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
// but it does not run any of it. main() below actually drives the chain implicated by the bug report:
//  (a) the graph-building / Schrodinger path (detail/graph_encoding/MPGraphEncodingStorage.h), via
//      build_graph(), graph_memory_usage(), and expectation_value_and_gradient().
// It also checks the installed usage requirements a consumer inherits:
//  (b) the OpenMP compile and link flags, which must reach the consumer through the imported target alone: the
//      instantiations below compile the library's OpenMP regions in this translation unit;
//  (c) the MPI setting: the consumer sees monoprop_ENABLE_MPI exactly when it linked an MPI-enabled libmonoprop
//      (monoprop_EXPECT_MPI, when set).

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

#include <complex>
#include <optional>
#include <print>
#include <vector>

#if defined(monoprop_EXPECT_MPI)
#if monoprop_EXPECT_MPI && !defined(monoprop_ENABLE_MPI)
#error "expected an MPI-enabled libmonoprop, but the imported target did not carry monoprop_ENABLE_MPI"
#elif !monoprop_EXPECT_MPI && defined(monoprop_ENABLE_MPI)
#error "expected the serial libmonoprop, but the imported target carries monoprop_ENABLE_MPI"
#endif
#endif

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

// Without the imported target's OpenMP compile flags the library's pragmas would silently compile out, so check for
// them explicitly rather than rely on a link failure.
#ifndef _OPENMP
#error "monoprop::monoprop did not propagate the OpenMP compile flags to this consumer"
#endif

} // namespace

auto main() -> int {
    monoprop::mpi::init();
    run_graph_build_chain();
    monoprop::mpi::finalize();
    std::println(stderr, "[link_export_probe] OK");
    return 0;
}
