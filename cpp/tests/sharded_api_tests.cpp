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
 * The public C++ surface of MonomialPropagator after the removal of the partition runtime, checked at compile time.
 *
 * There is one runtime: no partition count, child factory, facade helper, partition-named exception, cached thread
 * count or replacement thread/shard argument, and no development selector. Every negative probe is paired with a
 * positive control that applies the same probe to a test-local type which does have the member, so a probe that is
 * false for an unrelated reason (a typo, an access error, a missing header) fails instead of passing silently.
 * Positive controls on the propagator itself pin the constructor and the ordinary extension hooks that must survive.
 */

#include <boost/test/unit_test.hpp>

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/EnvConfig.h"

// The removed development selector must not reach the library, the tests or an installed consumer.
#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
#error "monoprop_SHARDED_OPENMP_PROTOTYPE is defined: the removed runtime selector came back"
#endif

// The removed runtime's headers are gone from the include path.
#if __has_include("monoprop/detail/partition/PartitionGroup.h") \
                  || __has_include("monoprop/detail/partition/CpuTopology.h")
#error "a partition-runtime header is back on the include path"
#endif
#if __has_include("monoprop/detail/mpi/ShmComm.h")                                            \
                  || __has_include("monoprop/detail/mpi/HybridComm.h")                        \
                                   || __has_include("monoprop/detail/mpi/PartitionBarrier.h") \
                                                    || __has_include("monoprop/detail/mpi/CpuRelax.h")
#error "an in-process communicator or custom-barrier header is back on the include path"
#endif
// Positive control: the include path itself works.
#if !__has_include("monoprop/detail/sharded/Team.h")
#error "the include-path probe cannot see a header that exists"
#endif

// Declared, never defined, by this test: complete only if the library defines them.
namespace monoprop {
class MultiPartitionUnsupported;
class PartitionCountMismatch;
} // namespace monoprop

namespace {

using namespace monoprop;

using MP = MonomialPropagator<8>;

template <class T>
concept Complete = requires { sizeof(T); };

// The process-cached environment settings hold only the routing knobs; the budget is captured, strictly, at
// construction (ThreadBudget.h).
template <class S>
concept HasCachedThreadCount = requires(const S &s) { s.num_threads; };

template <class S>
concept HasCachedRouting = requires(const S &s) { s.routing_mode; };

template <class P>
concept HasPartitionFactory = requires { typename P::PartitionChildFactory; };

template <class P>
concept HasLayerData = requires { typename P::LayerData; };

// `Target` constructed from the eleven mathematical arguments, followed by `Tail`.
template <class Target, class... Tail>
constexpr bool kConstructible = std::is_constructible_v<Target,
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

using RemovedFactory = std::function<std::unique_ptr<MP>(mpi::Comm)>;

/*
 * Protected members are visible from a derived class only, so the probes are static members of a class derived from
 * the one probed. `Self` keeps each requirement dependent: a missing member is then a false requirement, not a hard
 * error.
 */
template <class Base>
struct ProbeOf : Base {
    using Base::Base;

    template <class Self = ProbeOf>
    static constexpr bool kFacade = requires(Self &p) { p.is_partition_facade(); } || requires(const Self &p) {
        p.first_partition_();
    } || requires(Self &p) { p.for_each_partition_([](MP &) {}); } || requires(Self &p) {
        p.map_partitions_([](MP &) { return 0; });
    } || requires(Self &p) { p.map_partitions_indexed_([](int, MP &) { return 0; }); } || requires(Self &p) {
        p.concat_partitions_([](MP &) { return VecD{}; });
    } || requires(const Self &p) {
        p.sum_partitions_([](const MP &) { return size_t{0}; });
    } || requires(const Self &p) { p.fold_partitions_([](const MP &) { return size_t{0}; }, [](size_t &, size_t) {}); };

    template <class Self = ProbeOf>
    static constexpr bool kClone = requires(const Self &p) {
        { p.clone_() } -> std::same_as<std::unique_ptr<MP>>;
    };

    template <class Self = ProbeOf>
    static constexpr bool kInitialOperator = requires(Self &p, const OperatorDict &op) {
        { p.apply_initial_operator_(op) } -> std::same_as<std::pair<MonomialList<8>, VecD>>;
    };
};

using Probe = ProbeOf<MP>;

// Positive controls: test-local types that do have what the probes look for.
struct FacadeBase : MP {
    using MP::MP;

protected:
    auto is_partition_facade() -> bool { return true; }
};
using FacadeControl = ProbeOf<FacadeBase>;

struct TailControl {
    TailControl(const OperatorDict &,
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
                size_t = 0,
                RemovedFactory = nullptr) {}
    using PartitionChildFactory = RemovedFactory;
};

struct SettingsControl {
    std::optional<int> num_threads;
    std::optional<config::RoutingMode> routing_mode;
};

// --- constructor ---------------------------------------------------------------------------------------------------

static_assert(kConstructible<MP>, "the eleven-argument constructor must stay");
static_assert(std::is_constructible_v<MP,
                                      const OperatorDict &,
                                      unsigned int,
                                      const VecZ &,
                                      std::optional<unsigned int>,
                                      mpi::Comm>,
              "the defaulted tail must stay defaulted");
static_assert(!kConstructible<MP, size_t>, "no partition count");
static_assert(!kConstructible<MP, size_t, RemovedFactory>, "no child factory");
static_assert(!kConstructible<MP, size_t, std::nullptr_t>);
// No replacement thread or shard count of any integral type.
static_assert(!kConstructible<MP, int>);
static_assert(!kConstructible<MP, unsigned int>);
static_assert(kConstructible<TailControl, size_t> && kConstructible<TailControl, size_t, RemovedFactory>
                  && kConstructible<TailControl, int>,
              "positive control for the constructor probes");

// --- nested types and helpers --------------------------------------------------------------------------------------

static_assert(HasLayerData<MP>);
static_assert(!HasPartitionFactory<MP>, "PartitionChildFactory is removed");
static_assert(HasPartitionFactory<TailControl>, "positive control for the nested-type probe");
static_assert(!Probe::kFacade<>, "no partition-facade helper may remain");
static_assert(FacadeControl::kFacade<>, "positive control for the facade probe");
static_assert(Probe::kClone<>, "the virtual clone hook stays");
static_assert(Probe::kInitialOperator<>, "the protected initial-operator hook stays");
static_assert(std::has_virtual_destructor_v<MP>);

// --- environment ---------------------------------------------------------------------------------------------------

static_assert(HasCachedRouting<config::Settings>);
static_assert(!HasCachedThreadCount<config::Settings>, "no cached, permissive thread-count reader");
static_assert(HasCachedThreadCount<SettingsControl>, "positive control for the settings probe");

// --- exceptions ----------------------------------------------------------------------------------------------------

static_assert(!Complete<MultiPartitionUnsupported>, "the partition-named raw-accessor exception is removed");
static_assert(!Complete<PartitionCountMismatch>, "the partition-count agreement is removed");
static_assert(Complete<MultiShardUnsupported>, "positive control: the raw-accessor exception");
static_assert(std::is_base_of_v<std::runtime_error, MultiShardUnsupported>);
static_assert(Complete<InvalidPropagatorError> && Complete<PropagatorConfigError>);

} // namespace

// The compile-time checks above are the test; this case keeps the file visible to the runner and reports the surface.
BOOST_AUTO_TEST_CASE(sharded_api_surface_has_no_partition_runtime) {
    BOOST_TEST_MESSAGE("sharded OpenMP surface: no partition runtime, selector or replacement thread argument");
    BOOST_TEST(!Probe::kFacade<>);
    BOOST_TEST(FacadeControl::kFacade<>);
    BOOST_TEST((!kConstructible<MP, size_t>));
}
