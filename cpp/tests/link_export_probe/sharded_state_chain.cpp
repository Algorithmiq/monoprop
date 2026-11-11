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

// Chain (g) of link_export_probe.cpp, in its own translation unit so that detail/sharded/State.h is the first and
// only monoprop header: an installed consumer compiles it self-contained, with OpenMP flags from the imported target.

#include "monoprop/detail/sharded/State.h"

#include <complex>
#include <cstddef>
#include <memory>
#include <print>
#include <stdexcept>

#include "monoprop/detail/parallel/ThreadBudget.h"

// Seeds both pictures at the captured budget, copies the Heisenberg shards on their owners and checks the counts
// against the unsharded totals, then checks that an initializer failure reaches the caller.
auto run_sharded_state_chain() -> bool {
    namespace sharded = monoprop::detail::sharded;
    constexpr size_t kModes = 4;
    const auto options = monoprop::detail::parallel::capture_thread_budget();
    const auto threads = static_cast<size_t>(options.threads);
    const auto router = monoprop::routing::make_router<kModes>(1, threads);

    monoprop::OperatorDict op;
    op[monoprop::VecZ{}] = std::complex<double>{1.5, 0.0};
    op[monoprop::VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    op[monoprop::VecZ{2, 3}] = std::complex<double>{0.0, -0.5};
    op[monoprop::VecZ{1, 2}] = std::complex<double>{0.0, 0.25};
    const monoprop::VecZ state{0, 1};
    const auto core = sharded::validate_initial_operator<kModes>(op, monoprop::Basis::Majorana, kModes);

    const auto heisenberg = sharded::OperatorSeed<kModes>{.initial_operator = op,
                                                          .initial_state = state,
                                                          .router = router,
                                                          .basis = monoprop::Basis::Majorana,
                                                          .logical_num_modes = kModes,
                                                          .paired = std::nullopt,
                                                          .inline_width = 4};
    const auto seeded = sharded::seed_shards(options, heisenberg, 0);
    const auto copied = sharded::copy_shards(options, seeded);

    auto schrodinger = heisenberg;
    schrodinger.paired = sharded::paired_basis_bounds(3, kModes, router.flat_world());
    const auto paired = sharded::seed_shards(options, schrodinger, 0);

    auto failed = false;
    try {
        (void)sharded::make_shards<kModes>(options, [&](size_t shard) {
            if (shard + 1 == threads) {
                throw std::runtime_error("expected");
            }
            return std::make_unique<sharded::ShardState<kModes>>(monoprop::detail::MPOperator<kModes>{}, false);
        });
    }
    catch (const std::runtime_error &) {
        failed = true;
    }

    const auto correct = core == 1.5 && sharded::total_size(seeded) == 3 && sharded::total_size(copied) == 3
                         && sharded::total_size(paired) == monoprop::paired_op_size(2, kModes)
                         && sharded::operator_memory_usage(copied).total_bytes() > 0 && failed;
    std::println(stderr,
                 "[link_export_probe] sharded state chain: shards={} heisenberg={} schrodinger={} correct={}",
                 threads,
                 sharded::total_size(seeded),
                 sharded::total_size(paired),
                 correct);
    return correct;
}
