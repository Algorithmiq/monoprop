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

#pragma once

#include <bit>
#include <cstdlib>
#include <format>
#include <functional>
#include <string>
#include <utility>

#include <boost/test/unit_test.hpp>

#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"

/*
 * Boost.Test preconditions for cases that need a particular MPI world. A case whose precondition
 * fails is reported as skipped, with the reason, instead of passing without checking anything;
 * CTest turns that report into a skip for the per-case serial entries.
 */
namespace test_utils {

namespace rank_detail {

inline auto world_size_precondition(std::function<bool(int)> accepts, std::string need)
    -> boost::unit_test::decorator::precondition {
    return boost::unit_test::precondition(
        [accepts = std::move(accepts), need = std::move(need)](boost::unit_test::test_unit_id) {
            const int ranks = monoprop::mpi::size(MPI_COMM_WORLD);
            boost::test_tools::assertion_result result(accepts(ranks));
            result.message() << "needs " << need << ", MPI_COMM_WORLD has " << ranks;
            return result;
        });
}

} // namespace rank_detail

/// Requires at least @p count ranks in MPI_COMM_WORLD.
inline auto min_ranks(int count) -> boost::unit_test::decorator::precondition {
    return rank_detail::world_size_precondition([count](int ranks) { return ranks >= count; },
                                                std::format("at least {} ranks", count));
}

/// Requires a power-of-two rank count of at least @p count, which the XOR pairing needs.
inline auto power_of_two_ranks(int count = 2) -> boost::unit_test::decorator::precondition {
    return rank_detail::world_size_precondition(
        [count](int ranks) { return ranks >= count && std::has_single_bit(static_cast<unsigned>(ranks)); },
        std::format("a power-of-two rank count of at least {}", count));
}

/// Requires an even rank count, so that `rank ^ 1` pairs every rank.
inline auto even_ranks() -> boost::unit_test::decorator::precondition {
    return rank_detail::world_size_precondition([](int ranks) { return ranks >= 2 && ranks % 2 == 0; },
                                                "an even rank count");
}

/// Requires monoprop_ROUTING to resolve to linear routing.
inline auto linear_routing() -> boost::unit_test::decorator::precondition {
    return boost::unit_test::precondition([](boost::unit_test::test_unit_id) {
        boost::test_tools::assertion_result result(monoprop::routing::linear_requested());
        result.message() << "needs linear routing, but monoprop_ROUTING selects splitmix";
        return result;
    });
}

/// Requires monoprop_ROUTING to be unset, so that the shipped default is what runs.
inline auto default_routing() -> boost::unit_test::decorator::precondition {
    return boost::unit_test::precondition([](boost::unit_test::test_unit_id) {
        const char *mode = std::getenv("monoprop_ROUTING");
        boost::test_tools::assertion_result result(mode == nullptr || *mode == '\0');
        result.message() << "needs monoprop_ROUTING unset, but it is set";
        return result;
    });
}

} // namespace test_utils
