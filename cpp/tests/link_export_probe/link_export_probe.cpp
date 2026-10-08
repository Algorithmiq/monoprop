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
//  (b) the constructor's shape -- the eleven mathematical arguments construct, no partition, factory, thread or shard
//      argument does, and PartitionChildFactory is gone -- checked at compile time, then a construction that works.
// It also checks the installed usage requirements a consumer inherits:
//  (c) the OpenMP worksharing template (detail/parallel/Workshare.h), instantiated in this translation
//      unit, so OpenMP compile and link flags must reach the consumer through the imported target alone;
//  (d) the Pauli basis, alongside the Majorana graph chain in (a);
//  (e) the ordinary constructor, whose template calls the out-of-line capture_thread_budget(),
//      require_thread_support() and require_initializing_thread(), and operations reaching the exported evolve and
//      derivative entry points with their trailing Options; plus the raw-access rule (the sole shard at T = 1,
//      MultiShardUnsupported at T > 1, after which the object still answers).
//  (f) the fixed-team phase primitive (detail/sharded/Team.h), instantiated here like (c): its region,
//      barriers and masked construct must compile and link through the imported target's OpenMP flags.
//  (g) owner-initialized shard state (detail/sharded/State.h) in sharded_state_chain.cpp, whose only monoprop
//      include is that header: seeding, owner copies, counts and a failing initializer through the imported target.
//  (h) direct-buffer construction (detail/sharded/Construction.h) in sharded_construction_chain.cpp, whose only
//      monoprop include is that header: one-team propagation and graph construction over the captured budget's
//      shards through the imported target.
//  (i) sharded evaluation (detail/sharded/Evaluation.h) in sharded_evaluation_chain.cpp, whose first monoprop include
//      is that header: a coefficient-informed extension replaying its seed in-team, a retained functional, and runtime
//      calls to the exported ev_sharded() and ev_and_grad_sharded() through the imported target.
//  (j) the inherited configuration: no runtime selector reaches the consumer, and it sees monoprop_ENABLE_MPI exactly
//      when it expected an MPI package (monoprop_EXPECT_MPI, when set). The root runs every public operation family
//      and its exports and aggregates.
//  (k) an ordinary derived class -- virtual clone_() and update_initial_operator() overrides, the protected
//      apply_initial_operator_() -- used through the base interface.
//  (l) MPI builds launched on several ranks: the root on MPI_COMM_WORLD against the same root on MPI_COMM_SELF --
//      energies, gradients, global term counts and every exported key -- with T-dependent raw access.

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/parallel/Workshare.h"
#include "monoprop/detail/sharded/Team.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <optional>
#include <print>
#include <stdexcept>
#include <vector>

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
#error "monoprop::monoprop carries monoprop_SHARDED_OPENMP_PROTOTYPE: the removed runtime selector came back"
#endif

#if defined(monoprop_EXPECT_MPI)
#if monoprop_EXPECT_MPI && !defined(monoprop_ENABLE_MPI)
#error "expected an MPI package, but monoprop::monoprop did not carry monoprop_ENABLE_MPI"
#elif !monoprop_EXPECT_MPI && defined(monoprop_ENABLE_MPI)
#error "expected an MPI-off package, but monoprop::monoprop carries monoprop_ENABLE_MPI"
#endif
#endif

#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>

#include "monoprop/detail/mpi/MPIUtils.h"

// Forces every member function of MonomialPropagator<NumModes> to be compiled for these two
// representative widths, regardless of which ones main() below happens to call.
template class monoprop::MonomialPropagator<2>;
template class monoprop::MonomialPropagator<6>;

// Defined in sharded_state_chain.cpp.
auto run_sharded_state_chain() -> bool;
// Defined in sharded_construction_chain.cpp.
auto run_sharded_construction_chain() -> bool;
// Defined in sharded_evaluation_chain.cpp.
auto run_sharded_evaluation_chain() -> bool;

namespace {

using namespace monoprop;

template <class P>
concept HasPartitionFactory = requires { typename P::PartitionChildFactory; };

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

// The constructor has no partition, factory, thread or shard argument at all.
auto run_constructor_shape_chain() -> void {
    constexpr size_t kModes = 6;
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};

    using MP = MonomialPropagator<kModes>;
    using Factory = std::function<std::unique_ptr<MP>(mpi::Comm)>;
    constexpr auto constructible_with = []<class... Tail>() {
        return std::is_constructible_v<MP,
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
    };
    static_assert(constructible_with.template operator()<>());
    static_assert(!constructible_with.template operator()<size_t>());
    static_assert(!constructible_with.template operator()<int>());
    static_assert(!constructible_with.template operator()<size_t, Factory>());
    static_assert(!HasPartitionFactory<MP>);
    MP sim(ham,
           2 * kModes,
           VecZ{0, 1},
           /*schrodinger_cutoff=*/std::nullopt,
           MPI_COMM_SELF,
           /*lower_atol=*/std::nullopt,
           /*upper_atol=*/std::nullopt,
           CutoffType::Length,
           /*basis_change=*/std::nullopt,
           /*logical_num_modes=*/kModes,
           Basis::Majorana);
    std::println(stderr,
                 "[link_export_probe] constructor chain: no partition surface; eleven-argument size={}",
                 sim.size());
}

auto run_raw_access_chain() -> void {
    constexpr size_t kModes = 2;
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    MonomialPropagator<kModes> sim(ham, 2 * kModes, VecZ{0, 1}, std::nullopt, MPI_COMM_SELF);
    sim.build_graph(std::vector<VecZ>{{0, 2}, {1, 3}}, VecZ{0, 1}, VecD{1.0, 1.0});
    const auto [value, grad] = sim.expectation_value_and_gradient(VecD{0.1, 0.2});
    const auto budget = monoprop::detail::parallel::capture_thread_budget();
    // Raw access is the sole shard's at T = 1 and rejected above it; a rejection changes nothing.
    bool raw_ok = false;
    try {
        raw_ok = sim.mp_op().size() == sim.size() && sim.indexing().size() == sim.size()
                 && sim.graph().layers() == sim.graph_layers() && sim.graph_data().size() == sim.graph_layers()
                 && budget.threads == 1;
    }
    catch (const MultiShardUnsupported &) {
        raw_ok = budget.threads > 1 && sim.expectation_value(VecD{0.1, 0.2}) == value;
    }
    std::println(stderr,
                 "[link_export_probe] raw access chain: budget={} value={} grad_size={} raw_access_rule={}",
                 budget.threads,
                 value,
                 grad.size(),
                 raw_ok);
    if (!raw_ok || grad.size() != 2) {
        std::println(stderr, "[link_export_probe] raw access chain: FAILED");
        std::exit(1);
    }
}

// The integrated root through the imported target only: every public operation family, checked against itself
// (graph against graph-free propagation, direct against retained evaluation, unique exported keys).
auto run_sharded_root_chain() -> bool {
    constexpr size_t kModes = 6;
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    ham[VecZ{2, 3}] = std::complex<double>{0.0, 0.5};
    ham[VecZ{0, 3, 4, 7}] = std::complex<double>{0.25, 0.0};
    ham[VecZ{}] = std::complex<double>{1.5, 0.0};
    const VecZ state{0, 1, 2};
    const std::vector<VecZ> gates{{0, 2}, {1, 4}, {3, 5}, {2, 7}, {0, 9}};
    const VecZ mapping{0, 1, 0, 2, 1};
    const VecD gen{1.0, -0.5, 1.0, 0.75, 1.0};
    const VecD params{0.3, -0.2, 0.7};
    const auto make = [&] { return MonomialPropagator<kModes>(ham, 2 * kModes, state, std::nullopt, MPI_COMM_SELF); };
    auto graph = make();
    graph.build_graph(gates, mapping, gen);
    auto direct = make();
    direct.propagate(gates, mapping, gen, params);
    const double energy = graph.expectation_value(params);
    const auto [value, gradient] = graph.expectation_value_and_gradient(params);
    const auto pared = graph.expectation_value_and_gradient_functional(1e-3)(params);
    auto informed = make();
    informed.build_graph(gates, mapping, gen, std::nullopt, params);
    auto copy = graph;
    copy.update_initial_operator(OperatorDict{{VecZ{0, 1}, std::complex<double>{0.0, 2.0}}});
    const auto terms = graph.evolved_operator_terms(params, 0.0);
    std::vector<VecZ> keys;
    for (const auto &[key, coeff] : terms) {
        keys.push_back(key);
    }
    std::ranges::sort(keys);
    const bool unique = std::ranges::adjacent_find(keys) == keys.end();
    const auto contracted = graph.contract_partially(params, true);
    const double after = graph.expectation_value(VecD{});
    const double reference = direct.expectation_value(VecD{});
    const auto close = [](double a, double b) { return std::abs(a - b) <= 1e-10 * (1.0 + std::abs(a)); };
    const auto threads = monoprop::detail::parallel::capture_thread_budget().threads;
    const bool correct = close(energy, reference) && energy == value && gradient.size() == params.size()
                         && close(pared.first, energy) && close(after, energy) && unique && terms.size() == graph.size()
                         && contracted.size() == graph.size() && graph.graph_layers() == 0
                         && copy.expectation_value(params) != energy && informed.graph_layers() == gates.size();
    std::println(stderr,
                 "[link_export_probe] sharded root chain: team={} energy={} terms={} correct={}",
                 threads,
                 energy,
                 terms.size(),
                 correct);
    return correct;
}

// An ordinary extension through the imported target: virtual clone_() and update_initial_operator() overrides, and the
// protected apply_initial_operator_() returning this rank's share, all used through the base interface.
class DerivedRoot final : public MonomialPropagator<6> {
public:
    using MonomialPropagator<6>::MonomialPropagator;
    DerivedRoot(const DerivedRoot &) = default;
    auto update_initial_operator(const OperatorDict &op_dict) -> void override {
        ++updates;
        share = apply_initial_operator_(op_dict).first.size();
    }
    size_t updates = 0;
    size_t share = 0;

protected:
    auto clone_() const -> std::unique_ptr<MonomialPropagator<6>> override {
        return std::make_unique<DerivedRoot>(*this);
    }
};

class CloneProbe final : public MonomialPropagator<6> {
public:
    static auto clone_of(const MonomialPropagator<6> &p) -> std::unique_ptr<MonomialPropagator<6>> {
        return (p.*(&CloneProbe::clone_))();
    }
};

auto hook_operator() -> OperatorDict {
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    ham[VecZ{2, 3}] = std::complex<double>{0.0, 0.5};
    ham[VecZ{0, 3, 4, 7}] = std::complex<double>{0.25, 0.0};
    ham[VecZ{1, 2, 5, 6}] = std::complex<double>{-0.125, 0.0};
    return ham;
}

auto run_derived_hooks_chain(MPI_Comm comm) -> bool {
    const auto ham = hook_operator();
    DerivedRoot sim(ham, 12, VecZ{0, 1, 2}, std::nullopt, comm);
    sim.build_graph(std::vector<VecZ>{{0, 2}, {1, 4}, {3, 5}}, VecZ{0, 1, 0}, VecD{1.0, -0.5, 1.0});
    const VecD params{0.3, -0.2};
    const double energy = sim.expectation_value(params);
    const auto clone = CloneProbe::clone_of(sim);
    MonomialPropagator<6> &base = sim;
    OperatorDict scaled;
    for (const auto &[key, value] : ham) {
        scaled[key] = value * 2.0;
    }
    base.update_initial_operator(scaled);
    const auto global_share = mpi::allreduce_sum<size_t>(sim.share, mpi::Comm(comm));
    const bool correct =
        dynamic_cast<const DerivedRoot *>(clone.get()) != nullptr && sim.updates == 1 && global_share == ham.size()
        && clone->expectation_value(params) == energy
        && std::abs(sim.expectation_value(params) - (2.0 * energy)) <= 1e-12 * (1.0 + std::abs(energy));
    std::println(stderr,
                 "[link_export_probe] derived hooks chain: ranks={} share={} global_share={} correct={}",
                 mpi::size(mpi::Comm(comm)),
                 sim.share,
                 global_share,
                 correct);
    return correct;
}

// Several ranks: the root on the world communicator against one process's root, through the imported target.
auto run_multirank_root_chain() -> bool {
    const auto world = mpi::Comm(MPI_COMM_WORLD);
    const auto ranks = mpi::size(world);
    // A launch that names its rank count must get it: a single-rank run cannot stand in for this chain.
    if (const char *expected = std::getenv("monoprop_TEST_EXPECT_RANKS");
        expected != nullptr && std::atoi(expected) != ranks) {
        std::println(stderr, "[link_export_probe] multirank root chain: expected {} ranks, got {}", expected, ranks);
        return false;
    }
    if (ranks < 2) {
        std::println(stderr, "[link_export_probe] multirank root chain: skipped on one rank");
        return true;
    }
    const auto ham = hook_operator();
    const std::vector<VecZ> gates{{0, 2}, {1, 4}, {3, 5}, {2, 7}, {0, 9}, {6, 11}};
    const VecZ mapping{0, 1, 0, 2, 1, 2};
    const VecD gen{1.0, -0.5, 1.0, 0.75, 1.0, 0.5};
    const VecD params{0.3, -0.2, 0.7};
    const auto make = [&](MPI_Comm comm) { return MonomialPropagator<6>(ham, 12, VecZ{0, 1, 2}, std::nullopt, comm); };
    auto distributed = make(MPI_COMM_WORLD);
    auto single = make(MPI_COMM_SELF);
    distributed.build_graph(gates, mapping, gen);
    single.build_graph(gates, mapping, gen);
    const auto close = [](double a, double b) {
        return std::abs(a - b) <= 1e-9 + (1e-7 * std::max(std::abs(a), std::abs(b)));
    };
    const auto [dv, dg] = distributed.expectation_value_and_gradient(params);
    const auto [sv, sg] = single.expectation_value_and_gradient(params);
    bool correct = close(dv, sv) && dg.size() == sg.size();
    for (size_t i = 0; correct && i < dg.size(); ++i) {
        correct = close(dg[i], sg[i]);
    }
    const auto local = distributed.evolved_operator_terms(params, 0.0);
    std::map<VecZ, std::complex<double>> reference;
    for (const auto &[key, coeff] : single.evolved_operator_terms(params, 0.0)) {
        reference.emplace(key, coeff);
    }
    size_t unknown = 0;
    for (const auto &[key, coeff] : local) {
        const auto it = reference.find(key);
        unknown += it == reference.end() || !close(it->second.real(), coeff.real()) ? 1 : 0;
    }
    const auto global_terms = mpi::allreduce_sum<size_t>(local.size(), world);
    const auto global_size = mpi::allreduce_sum<size_t>(distributed.size(), world);
    const auto global_unknown = mpi::allreduce_sum<size_t>(unknown, world);
    correct = correct && global_terms == reference.size() && global_size == single.size() && global_unknown == 0;
    // Raw access depends on T alone, whatever the rank count.
    const auto threads = monoprop::detail::parallel::capture_thread_budget().threads;
    bool raw = false;
    try {
        raw = distributed.mp_op().size() == distributed.size() && threads == 1;
    }
    catch (const MultiShardUnsupported &) {
        raw = threads > 1;
    }
    const bool hooks = run_derived_hooks_chain(MPI_COMM_WORLD);
    std::println(stderr,
                 "[link_export_probe] multirank root chain: ranks={} team={} energy={} reference={} global_terms={} "
                 "reference_terms={} global_size={} reference_size={} unknown={} raw_access_rule={}",
                 ranks,
                 threads,
                 dv,
                 sv,
                 global_terms,
                 reference.size(),
                 global_size,
                 single.size(),
                 global_unknown,
                 raw);
    correct = correct && raw && hooks;
    std::println(stderr, "[link_export_probe] multirank root chain: correct={}", correct);
    return correct;
}

} // namespace

auto main() -> int {
    monoprop::mpi::init();
    run_graph_build_chain();
    run_constructor_shape_chain();
    run_openmp_workshare_chain();
    run_sharded_team_chain();
    if (!run_sharded_state_chain()) {
        std::println(stderr, "[link_export_probe] sharded state chain: FAILED");
        return 1;
    }
    if (!run_sharded_construction_chain()) {
        std::println(stderr, "[link_export_probe] sharded construction chain: FAILED");
        return 1;
    }
    if (!run_sharded_evaluation_chain()) {
        std::println(stderr, "[link_export_probe] sharded evaluation chain: FAILED");
        return 1;
    }
    run_pauli_chain();
    run_raw_access_chain();
    if (!run_sharded_root_chain()) {
        std::println(stderr, "[link_export_probe] sharded root chain: FAILED");
        return 1;
    }
    if (!run_derived_hooks_chain(MPI_COMM_SELF)) {
        std::println(stderr, "[link_export_probe] derived hooks chain: FAILED");
        return 1;
    }
    if (!run_multirank_root_chain()) {
        std::println(stderr, "[link_export_probe] multirank root chain: FAILED");
        return 1;
    }
    monoprop::mpi::finalize();
    std::println(stderr, "[link_export_probe] OK");
    return 0;
}
