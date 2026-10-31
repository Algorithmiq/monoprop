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

// Chain (h) of link_export_probe.cpp, in its own translation unit so that detail/sharded/Construction.h is the first
// and only monoprop header: an installed consumer compiles it self-contained, with OpenMP flags from the imported
// target.

#include "monoprop/detail/sharded/Construction.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <print>
#include <vector>

#include "monoprop/detail/parallel/ThreadBudget.h"

namespace {

constexpr size_t kModes = 4;
using Mono = monoprop::Monomial<kModes>;

// Slot indices map to bits as for every generator the library receives.
auto mono(const monoprop::VecZ &slots) -> Mono {
    return monoprop::indices_to_bitset<kModes>(slots);
}

} // namespace

// Seeds a Heisenberg operator on the captured budget's shards, propagates one gate through the direct-buffer team
// and checks the rotated pair against its closed form, then builds a two-gate graph on fresh shards.
auto run_sharded_construction_chain() -> bool {
    namespace sharded = monoprop::detail::sharded;
    const auto options = monoprop::detail::parallel::capture_thread_budget();
    const auto threads = static_cast<size_t>(options.threads);
    const auto router = monoprop::routing::make_router<kModes>(1, threads);

    // H = 1.0 * i g0 g1; the gate G = g1 g2 rotates it into the partner g0 g2.
    monoprop::OperatorDict op;
    op[monoprop::VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    const monoprop::VecZ state{0, 1};
    const auto seed = sharded::OperatorSeed<kModes>{.initial_operator = op,
                                                    .initial_state = state,
                                                    .router = router,
                                                    .basis = monoprop::Basis::Majorana,
                                                    .logical_num_modes = kModes,
                                                    .paired = std::nullopt,
                                                    .inline_width = 4};
    const monoprop::CutoffFn<kModes> cutoff = monoprop::detail::LengthCutoff<kModes>{2 * kModes, kModes};
    const auto ctx = sharded::ConstructionContext<kModes>{.cutoff_fn = cutoff, .router = router};

    auto shards = sharded::seed_shards(options, seed, 0);
    const std::vector<Mono> gates{mono({1, 2})};
    const std::vector<double> angles{0.3};
    const auto propagated =
        sharded::propagate<kModes>(options,
                                   shards,
                                   ctx,
                                   {.generators = gates, .mapped_params = angles, .only_rotate_len_k = {}});
    double norm = 0.0;
    size_t rows = 0;
    for (const auto &s : shards) {
        for (const double c : s->op.op_coeffs) {
            norm += c * c;
        }
        rows += s->size();
    }

    auto graph_shards = sharded::seed_shards(options, seed, 0);
    const std::vector<Mono> circuit{mono({1, 2}), mono({0, 3})};
    const std::vector<size_t> mapping{0, 1};
    const std::vector<double> coeffs{1.0, 1.0};
    const std::vector<size_t> gate_indices{0, 1};
    const auto built = sharded::build_graph<kModes>(options,
                                                    graph_shards,
                                                    ctx,
                                                    {.generators = circuit,
                                                     .parameter_mapping = mapping,
                                                     .gen_coeffs = coeffs,
                                                     .gate_indices = gate_indices,
                                                     .only_rotate_len_k = {}});
    for (const auto &outcome : {propagated, built}) {
        if (outcome.error) {
            try {
                std::rethrow_exception(outcome.error);
            }
            catch (const std::exception &e) {
                std::println(stderr, "[link_export_probe] sharded construction chain: {}", e.what());
            }
        }
    }
    size_t layers = 0;
    for (const auto &s : graph_shards) {
        layers += s->graph.layers();
    }

    // A rotation preserves the coefficient norm: cos^2 + sin^2 of the one source term.
    const auto correct = !propagated.error && propagated.mutation_started && !built.error && rows == 2
                         && std::abs(norm - 1.0) < 1e-12 && layers == 2 * threads;
    std::println(stderr,
                 "[link_export_probe] sharded construction chain: shards={} rows={} norm={} layers={} correct={}",
                 threads,
                 rows,
                 norm,
                 layers,
                 correct);
    return correct;
}
