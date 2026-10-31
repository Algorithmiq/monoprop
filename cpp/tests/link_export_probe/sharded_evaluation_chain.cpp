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

// Chain (i) of link_export_probe.cpp, in its own translation unit so that detail/sharded/Evaluation.h is the first
// monoprop header: an installed consumer compiles it self-contained, with OpenMP flags from the imported target, and
// calls the exported sharded evaluators across the shared-library boundary.

#include "monoprop/detail/sharded/Evaluation.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <print>
#include <span>
#include <vector>

#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/Construction.h"

namespace {

constexpr size_t kModes = 4;
using Mono = monoprop::Monomial<kModes>;

auto mono(const monoprop::VecZ &slots) -> Mono {
    return monoprop::indices_to_bitset<kModes>(slots);
}

auto report(const char *what, const std::exception_ptr &error) -> bool {
    if (!error) {
        return true;
    }
    try {
        std::rethrow_exception(error);
    }
    catch (const std::exception &e) {
        std::println(stderr, "[link_export_probe] sharded evaluation chain: {}: {}", what, e.what());
    }
    return false;
}

} // namespace

// Builds a two-gate graph on the captured budget's shards (structurally, then extended coefficient-informed, which
// replays the existing graph in-team as its seed), retains a functional, and checks ev_and_grad_sharded() against
// ev_sharded() and a central difference of it.
auto run_sharded_evaluation_chain() -> bool {
    namespace sharded = monoprop::detail::sharded;
    const auto options = monoprop::detail::parallel::capture_thread_budget();
    const auto threads = static_cast<size_t>(options.threads);
    const auto router = monoprop::routing::make_router<kModes>(1, threads);

    // H = i g0 g1 + 0.5 i g2 g3 + 0.25 (identity); state with modes 0 and 1 occupied.
    monoprop::OperatorDict op;
    op[monoprop::VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    op[monoprop::VecZ{2, 3}] = std::complex<double>{0.0, 0.5};
    op[monoprop::VecZ{}] = std::complex<double>{0.25, 0.0};
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

    const std::vector<Mono> first{mono({1, 2}), mono({0, 3})};
    const std::vector<size_t> first_mapping{0, 1};
    const std::vector<double> ones{1.0, 1.0};
    const std::vector<size_t> first_gates{0, 1};
    bool ok = report("build_graph",
                     sharded::build_graph<kModes>(options,
                                                  shards,
                                                  ctx,
                                                  {.generators = first,
                                                   .parameter_mapping = first_mapping,
                                                   .gen_coeffs = ones,
                                                   .gate_indices = first_gates,
                                                   .only_rotate_len_k = {}})
                         .error);

    // Extend coefficient-informed: gate 2 is driven by parameter 2; the existing layers replay at parameters 0 and 1.
    const std::vector<double> params{0.3, -0.2, 0.45};
    const std::vector<Mono> second{mono({2, 5})};
    const std::vector<size_t> second_mapping{2};
    const std::vector<double> second_coeff{0.5};
    const std::vector<size_t> second_gates{2};
    const std::vector<double> build_angle{params[2] * 0.5};
    // Heisenberg seed angles: the existing layers' mapped parameters in replay order (optimizer order reversed).
    const std::vector<double> seed_angles{params[1], params[0]};
    ok = ok
         && report("build_graph_informed",
                   sharded::build_graph_informed<kModes>(options,
                                                         shards,
                                                         ctx,
                                                         {.graph = {.generators = second,
                                                                    .parameter_mapping = second_mapping,
                                                                    .gen_coeffs = second_coeff,
                                                                    .gate_indices = second_gates,
                                                                    .only_rotate_len_k = {}},
                                                          .mapped_params = build_angle,
                                                          .seed_params = std::span<const double>(seed_angles)})
                       .error);

    // Optimizer order is the reverse of the graph's layer order.
    const size_t layers = shards.front()->graph.layers();
    monoprop::VecZ mapping(layers);
    monoprop::VecD gen(layers);
    for (size_t l = 0; l < layers; ++l) {
        const auto t = shards.front()->graph.get_layer_traversal(l);
        mapping[layers - 1 - l] = t.param_index();
        gen[layers - 1 - l] = t.gen_coeff();
    }
    auto retained = sharded::prepare_retained<kModes>(options,
                                                      shards,
                                                      {.core_term = 0.25,
                                                       .parameter_mapping = mapping,
                                                       .gen_coeffs = gen,
                                                       .pare_threshold = {},
                                                       .basis = monoprop::Basis::Majorana,
                                                       .schrodinger = false,
                                                       .rank = 0});
    ok = ok && report("prepare_retained", retained.error) && retained.retained.has_value();
    if (!ok) {
        std::println(stderr, "[link_export_probe] sharded evaluation chain: shards={} correct=false", threads);
        return false;
    }
    const auto &r = *retained.retained;
    const auto energy = [&](const monoprop::VecD &p) {
        const auto requests = r.requests(p);
        return sharded::ev_sharded(requests, r.callbacks, options, MPI_COMM_SELF);
    };
    const monoprop::VecD point(params.begin(), params.end());
    const auto requests = r.requests(point);
    const auto [value, grad] = sharded::ev_and_grad_sharded(requests, r.callbacks, options, MPI_COMM_SELF);
    double worst = 0.0;
    for (size_t i = 0; i < grad.size(); ++i) {
        auto plus = point;
        auto minus = point;
        plus[i] += 1e-6;
        minus[i] -= 1e-6;
        worst = std::max(worst, std::abs(grad[i] - ((energy(plus) - energy(minus)) / 2e-6)));
    }
    const bool correct = layers == 3 && grad.size() == 3 && std::abs(value - energy(point)) < 1e-12 && worst < 1e-6;
    std::println(stderr,
                 "[link_export_probe] sharded evaluation chain: shards={} layers={} value={} worst_fd={} correct={}",
                 threads,
                 layers,
                 value,
                 worst,
                 correct);
    return correct;
}
