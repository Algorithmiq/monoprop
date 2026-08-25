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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <memory>
#include <optional>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// A derived MonomialPropagator must keep its own type through partitioning: both its partition
// children (built via the injected child_factory) and clones of a partitioned facade (built via the
// virtual clone_() hook) have to come out as the derived type, not the base.

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 4;

// A toy subclass exercising both extension points: `child_factory` builds more of itself, and
// `clone_()` overrides the base's default (which would otherwise slice a copy down to Base).
class DerivedPropagator : public MonomialPropagator {
public:
    using Base = MonomialPropagator;

    DerivedPropagator(const OperatorDict &initial_operator,
                      unsigned int cutoff,
                      const VecZ &initial_state,
                      mpi::Comm comm,
                      size_t partitions)
        : Base(initial_operator,
               cutoff,
               initial_state,
               kNumModes,
               std::nullopt,
               comm,
               std::nullopt,
               std::nullopt,
               CutoffType::Length,
               std::nullopt,
               Basis::Majorana,
               partitions,
               /*storage_num_modes=*/kNumModes,
               Base::PartitionChildFactory{[=](mpi::Comm partition_comm) -> std::unique_ptr<Base> {
                   return std::make_unique<DerivedPropagator>(initial_operator,
                                                              cutoff,
                                                              initial_state,
                                                              partition_comm,
                                                              /*partitions=*/1);
               }}) {}

    DerivedPropagator(const DerivedPropagator &) = default;

    // Test-only passthrough: is_partition_facade() is protected on Base.
    auto is_facade() const -> bool { return this->is_partition_facade(); }

    auto children_are_all_derived() -> bool {
        const auto flags =
            this->map_partitions_([](Base &p) { return dynamic_cast<DerivedPropagator *>(&p) != nullptr; });
        return std::ranges::all_of(flags, [](bool ok) { return ok; });
    }

    auto indexed_children_are_all_derived() -> bool {
        const auto flags = this->map_partitions_indexed_(
            [](int, Base &p) { return dynamic_cast<DerivedPropagator *>(&p) != nullptr; });
        return std::ranges::all_of(flags, [](bool ok) { return ok; });
    }

protected:
    auto clone_() const -> std::unique_ptr<Base> override { return std::make_unique<DerivedPropagator>(*this); }
};

} // namespace

BOOST_AUTO_TEST_CASE(partition_facade_children_are_derived_type) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    DerivedPropagator sim(data.hamiltonian, kCutoff, data.initial_state, MPI_COMM_SELF, /*partitions=*/4);

    BOOST_TEST(sim.is_facade());
    BOOST_TEST(sim.children_are_all_derived());
    BOOST_TEST(sim.indexed_children_are_all_derived());
}

BOOST_AUTO_TEST_CASE(partition_facade_copy_children_are_derived_type) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    DerivedPropagator sim(data.hamiltonian, kCutoff, data.initial_state, MPI_COMM_SELF, /*partitions=*/4);

    DerivedPropagator copy(sim); // exercises PartitionGroup's copy ctor -> clone_()
    BOOST_TEST(copy.is_facade());
    BOOST_TEST(copy.children_are_all_derived());
}
