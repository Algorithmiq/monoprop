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

/*
 * The public C++ surface of MonomialPropagator, checked at compile time in both builds.
 *
 * The sharded OpenMP candidate (monoprop_SHARDED_OPENMP_PROTOTYPE) has no partition count, child factory or facade
 * helpers, and no replacement thread or shard argument; the legacy (default) build still has all of them. Each probe
 * is asserted with the build's expected value, so the same probe must come out true in the legacy build and false in
 * the candidate: a probe that is false for an unrelated reason (a missing header, an access error, a typo) fails one
 * of the two builds instead of passing silently. Positive controls in both builds pin the constructor and the
 * ordinary extension hooks that must survive.
 */

#include <boost/test/unit_test.hpp>

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/EnvConfig.h"

// Declared, never defined, by this test: complete only if the library defines it.
namespace monoprop {
class MultiPartitionUnsupported;
} // namespace monoprop

namespace {

using namespace monoprop;

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
constexpr bool kLegacy = false;
#else
constexpr bool kLegacy = true;
#endif

using MP = MonomialPropagator<8>;

template <class T>
concept Complete = requires { sizeof(T); };

// The process-cached environment settings: the legacy runtime also keeps a permissive thread count there, for its
// automatic partition count; the candidate only captures the budget, strictly, at construction (ThreadBudget.h).
template <class S>
concept HasCachedThreadCount = requires(const S &s) { s.num_threads; };

template <class S>
concept HasCachedRouting = requires(const S &s) { s.routing_mode; };

template <class P>
concept HasPartitionFactory = requires { typename P::PartitionChildFactory; };

// Positive control for the nested-type probe: a public nested alias both builds keep.
template <class P>
concept HasLayerData = requires { typename P::LayerData; };

// The public constructor with the eleven mathematical arguments, followed by `Tail`.
template <class... Tail>
constexpr bool kConstructibleWith = std::is_constructible_v<MP,
                                                            const OperatorDict &,
                                                            unsigned int,
                                                            const VecZ &,
                                                            std::optional<unsigned int>,
                                                            mpi::Comm,
                                                            std::optional<double>,
                                                            std::optional<double>,
                                                            CutoffType,
                                                            std::optional<std::vector<VecZ>>,
                                                            size_t,
                                                            Basis,
                                                            Tail...>;

using LegacyFactory = std::function<std::unique_ptr<MP>(mpi::Comm)>;

/*
 * Protected members are visible from a derived class only, so the probes live in one. `Self` keeps each requirement
 * dependent: a missing member is then a false requirement, not a hard error.
 */
struct Probe : MP {
    using MP::MP;

    template <class Self = Probe>
    static constexpr bool kFacadeHelpers =
        requires(Self &p) { p.is_partition_facade(); } || requires(const Self &p) { p.first_partition_(); }
        || requires(Self &p) { p.for_each_partition_([](MP &) {}); }
        || requires(Self &p) { p.map_partitions_([](MP &) { return 0; }); }
        || requires(Self &p) { p.map_partitions_indexed_([](int, MP &) { return 0; }); }
        || requires(Self &p) { p.concat_partitions_([](MP &) { return VecD{}; }); }
        || requires(const Self &p) { p.sum_partitions_([](const MP &) { return size_t{0}; }); }
        || requires(const Self &p) {
               p.fold_partitions_([](const MP &) { return size_t{0}; }, [](size_t &, size_t) {});
           };

    // Each facade helper on its own, so a partial removal fails too.
    template <class Self = Probe>
    static constexpr bool kEveryFacadeHelper =
        requires(Self &p) { p.is_partition_facade(); } && requires(const Self &p) { p.first_partition_(); }
        && requires(Self &p) { p.for_each_partition_([](MP &) {}); }
        && requires(Self &p) { p.map_partitions_([](MP &) { return 0; }); }
        && requires(Self &p) { p.map_partitions_indexed_([](int, MP &) { return 0; }); }
        && requires(Self &p) { p.concat_partitions_([](MP &) { return VecD{}; }); }
        && requires(const Self &p) { p.sum_partitions_([](const MP &) { return size_t{0}; }); }
        && requires(const Self &p) {
               p.fold_partitions_([](const MP &) { return size_t{0}; }, [](size_t &, size_t) {});
           };

    // The ordinary extension hooks every build keeps.
    template <class Self = Probe>
    static constexpr bool kCloneHook = requires(const Self &p) {
        { p.clone_() } -> std::same_as<std::unique_ptr<MP>>;
    };

    template <class Self = Probe>
    static constexpr bool kInitialOperatorHook = requires(Self &p, const OperatorDict &op) {
        { p.apply_initial_operator_(op) } -> std::same_as<std::pair<MonomialList<8>, VecD>>;
    };
};

// --- constructor ---------------------------------------------------------------------------------------------------

static_assert(kConstructibleWith<>, "the eleven-argument constructor must stay");
static_assert(std::is_constructible_v<MP,
                                      const OperatorDict &,
                                      unsigned int,
                                      const VecZ &,
                                      std::optional<unsigned int>,
                                      mpi::Comm>,
              "the defaulted tail must stay defaulted");
static_assert(kConstructibleWith<size_t> == kLegacy, "only the legacy build takes a partition count");
static_assert(kConstructibleWith<size_t, LegacyFactory> == kLegacy, "only the legacy build takes a child factory");
static_assert(kConstructibleWith<size_t, std::nullptr_t> == kLegacy);
// No replacement thread or shard count of any integral type.
static_assert(kConstructibleWith<int> == kLegacy);
static_assert(kConstructibleWith<unsigned int> == kLegacy);
static_assert(!kConstructibleWith<size_t, LegacyFactory, int>);

// --- nested types and helpers --------------------------------------------------------------------------------------

static_assert(HasLayerData<MP>);
static_assert(HasPartitionFactory<MP> == kLegacy, "PartitionChildFactory is legacy-only");
static_assert(Probe::kFacadeHelpers<> == kLegacy, "no partition-facade helper may remain in the candidate");
static_assert(Probe::kEveryFacadeHelper<> == kLegacy);
static_assert(Probe::kCloneHook<>, "the virtual clone hook stays");
static_assert(Probe::kInitialOperatorHook<>, "the protected initial-operator hook stays");
static_assert(std::has_virtual_destructor_v<MP>);

// --- environment ---------------------------------------------------------------------------------------------------

static_assert(HasCachedRouting<config::Settings>);
static_assert(HasCachedThreadCount<config::Settings> == kLegacy, "the candidate has no cached thread-count reader");

// --- the raw-accessor exception ------------------------------------------------------------------------------------

static_assert(Complete<MultiPartitionUnsupported> == kLegacy, "the partition-named exception is legacy-only");
#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
static_assert(std::is_base_of_v<std::runtime_error, MultiShardUnsupported>);
#endif

} // namespace

// The compile-time checks above are the test; this case reports which surface the runner was built against.
BOOST_AUTO_TEST_CASE(sharded_api_surface_matches_the_build) {
    BOOST_TEST_MESSAGE((kLegacy ? "legacy partition surface" : "sharded candidate surface"));
    BOOST_TEST(Probe::kFacadeHelpers<> == kLegacy);
    BOOST_TEST(kConstructibleWith<size_t> == kLegacy);
}
