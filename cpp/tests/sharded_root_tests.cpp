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
 * The integrated sharded OpenMP prototype root (monoprop_SHARDED_OPENMP_PROTOTYPE): MonomialPropagator over the T
 * shard states of one rank, through its public operations. Compiled only in a prototype build. The team size T is the
 * captured launch budget; cpp/tests/CMakeLists.txt reruns every case in fresh processes at T = 1, 2 and 4
 * (sharded_root_env_t*), and the serial per-case registrations run at T = 1. Cases that need a nonprimary worker
 * return early at T = 1.
 *
 * Private inspection, phase observation and failure injection go through PropagatorTestAccess (shards(), the (1, T)
 * router, and the test-only RootObserver every seam reports to). Observers record the executing worker inside the
 * protected phase; assertions run after the team has joined. Numerical parity against the legacy runtime at the same
 * (1, T) geometry, and across T, is checked from separate processes by tests/test_sharded_openmp.py.
 */

#include <boost/test/unit_test.hpp>

#include <cstdlib>
#include <string>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "AllocationProbe.h"
#include "PropagatorTestAccess.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/RootObserver.h"

namespace {

using namespace monoprop;
using namespace test_utils;
namespace sharded = monoprop::detail::sharded;

constexpr size_t kN = 12;
using MP = MonomialPropagator<kN>;
using Access = monoprop::detail::PropagatorTestAccess<kN>;

auto launch_team() -> size_t {
    const char *text = std::getenv("monoprop_TEST_EXPECT_TEAM");
    return text == nullptr ? 1 : static_cast<size_t>(std::stoul(text));
}

auto lih() -> const CaseData & {
    static const auto data = load_case_data<kN>("lih_fermionic_spin_exact.msgpack");
    return data;
}

struct Injected : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class Seam : std::uint8_t { shard, construction, evaluation, root };

struct Visit {
    Seam seam;
    int work;
    size_t step;
    size_t shard;
    int worker;
    int team;
    int level;
    std::thread::id thread;
};

// Records every observed visit with its executing worker, and optionally throws at one armed visit.
class Recorder final : public sharded::RootObserver {
public:
    struct Arm {
        Seam seam;
        int work;
        size_t shard;
        std::optional<size_t> step; // any step when empty
    };

    auto arm(Arm target) -> void {
        arm_ = target;
        fired_ = false;
    }
    auto disarm() -> void { arm_.reset(); }
    [[nodiscard]] auto fired() const -> bool { return fired_; }
    auto clear() -> void {
        const std::lock_guard lock(mutex_);
        visits_.clear();
    }
    [[nodiscard]] auto visits() const -> std::vector<Visit> {
        const std::lock_guard lock(mutex_);
        return visits_;
    }

    auto visit(sharded::EvaluationWork work, size_t step, size_t shard) const -> void override {
        note(Seam::evaluation, static_cast<int>(work), step, shard);
    }
    auto shard_work(sharded::ShardWork work, bool end, size_t shard) const -> void override {
        note(Seam::shard, (2 * static_cast<int>(work)) + (end ? 1 : 0), sharded::kNoStep, shard);
    }
    auto construction(sharded::ConstructionWork work, size_t step, size_t shard) const -> void override {
        note(Seam::construction, static_cast<int>(work), step, shard);
    }
    auto root(sharded::RootWork work, size_t shard) const -> void override {
        note(Seam::root, static_cast<int>(work), sharded::kNoStep, shard);
    }

private:
    auto note(Seam seam, int work, size_t step, size_t shard) const -> void {
        bool fire = false;
        {
            const std::lock_guard lock(mutex_);
            visits_.push_back(Visit{seam,
                                    work,
                                    step,
                                    shard,
                                    omp_get_thread_num(),
                                    omp_get_num_threads(),
                                    omp_get_level(),
                                    std::this_thread::get_id()});
            if (arm_ && !fired_ && arm_->seam == seam && arm_->work == work && arm_->shard == shard
                && (!arm_->step || *arm_->step == step)) {
                fired_ = true;
                fire = true;
            }
        }
        if (fire) {
            throw Injected("injected root-seam failure");
        }
    }

    mutable std::mutex mutex_;
    mutable std::vector<Visit> visits_;
    std::optional<Arm> arm_;
    mutable bool fired_ = false;
};

struct Config {
    bool schrodinger = false;
    std::optional<double> lower_atol = std::nullopt;
    std::optional<std::vector<VecZ>> basis_change = std::nullopt;
    Basis basis = Basis::Majorana;
};

auto make(const Config &config = {}, const sharded::RootObserver *observer = nullptr) -> std::unique_ptr<MP> {
    const auto &data = lih();
    return Access::construct_observed(observer,
                                      data.hamiltonian,
                                      4U,
                                      data.initial_state,
                                      config.schrodinger ? std::optional<unsigned int>{6U} : std::nullopt,
                                      mpi::Comm(MPI_COMM_SELF),
                                      config.lower_atol,
                                      std::optional<double>{},
                                      CutoffType::Length,
                                      config.basis_change,
                                      kN,
                                      config.basis,
                                      size_t{0},
                                      typename MP::PartitionChildFactory{});
}

auto build(MP &sim) -> void {
    const auto &data = lih();
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
}

auto params() -> const VecD & {
    return lih().parameters;
}

// The root's own caller-side work, outside any team.
auto caller_work(const Visit &v) -> bool {
    return v.seam == Seam::root
           && (v.work == static_cast<int>(sharded::RootWork::combine)
               || v.work == static_cast<int>(sharded::RootWork::export_block));
}

// Every owner visit of shard t ran on worker t of a level-1 team of T workers; caller work ran outside any team.
auto check_owner_visits(const std::vector<Visit> &visits, size_t team, const char *what) -> void {
    BOOST_TEST_CONTEXT(what) {
        BOOST_TEST(!visits.empty());
        std::set<std::thread::id> threads;
        for (const auto &v : visits) {
            if (caller_work(v)) {
                BOOST_TEST(v.level == 0);
                continue;
            }
            BOOST_TEST(static_cast<size_t>(v.worker) == v.shard);
            BOOST_TEST(static_cast<size_t>(v.team) == team);
            BOOST_TEST(v.level == 1);
            threads.insert(v.thread);
        }
        BOOST_TEST(threads.size() == team);
    }
}

auto count(const std::vector<Visit> &visits, Seam seam, int work, size_t shard) -> size_t {
    return static_cast<size_t>(std::ranges::count_if(visits, [&](const Visit &v) {
        return v.seam == seam && v.work == work && v.shard == shard;
    }));
}

auto energy_of(MP &sim) -> double {
    return sim.expectation_value(params());
}

} // namespace

BOOST_AUTO_TEST_CASE(sharded_root_raw_accessors_follow_the_launch_team) {
    const auto data = load_case_data<8>("random_exact.msgpack");
    MonomialPropagator<8> sim(data.hamiltonian, 16, data.initial_state, std::nullopt, MPI_COMM_SELF);
    const auto &shards = monoprop::detail::PropagatorTestAccess<8>::shards(sim);
    BOOST_TEST(shards.size() == launch_team());
    if (launch_team() == 1) {
        // The actual sole shard, not a copy or a merge.
        BOOST_TEST(&sim.mp_op() == &shards.front()->op);
        BOOST_TEST(&sim.graph() == &shards.front()->graph);
        BOOST_TEST(&sim.indexing() == shards.front()->op.store.get());
        BOOST_TEST(sim.graph_data().size() == sim.graph_layers());
        return;
    }
    const auto remedy = [](const MultiPartitionUnsupported &e) {
        return std::string(e.what()).find("monoprop_NUM_THREADS=1") != std::string::npos
               && std::string(e.what()).find("partitions=1") == std::string::npos;
    };
    BOOST_CHECK_EXCEPTION(sim.mp_op(), MultiPartitionUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(std::as_const(sim).mp_op(), MultiPartitionUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.graph(), MultiPartitionUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.indexing(), MultiPartitionUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.graph_data(), MultiPartitionUnsupported, remedy);
    // The rejection mutates nothing.
    BOOST_TEST(!monoprop::detail::PropagatorTestAccess<8>::is_invalid(sim));
}

BOOST_AUTO_TEST_CASE(sharded_root_rejects_explicit_partitions) {
    const auto data = load_case_data<8>("random_exact.msgpack");
    const auto construct = [&](size_t partitions, MonomialPropagator<8>::PartitionChildFactory factory) {
        return MonomialPropagator<8>(data.hamiltonian,
                                     16,
                                     data.initial_state,
                                     std::nullopt,
                                     MPI_COMM_SELF,
                                     std::nullopt,
                                     std::nullopt,
                                     CutoffType::Length,
                                     std::nullopt,
                                     8,
                                     Basis::Majorana,
                                     partitions,
                                     std::move(factory));
    };
    for (const size_t partitions : {1, 2, 4}) {
        BOOST_CHECK_EXCEPTION(construct(partitions, nullptr), PropagatorConfigError, [](const auto &e) {
            return std::string(e.what()).find("monoprop_NUM_THREADS") != std::string::npos;
        });
    }
    BOOST_CHECK_THROW(construct(0,
                                [](mpi::Comm) -> std::unique_ptr<MonomialPropagator<8>> {
                                    return nullptr;
                                }),
                      PropagatorConfigError);
    // The obsolete environment selector is rejected, whatever its value; it never selects a geometry.
    for (const char *value : {"off", "1", "4", "auto", ""}) {
        setenv("monoprop_PARTITIONS", value, 1);
        BOOST_CHECK_EXCEPTION(construct(0, nullptr), PropagatorConfigError, [](const auto &e) {
            return std::string(e.what()).find("monoprop_PARTITIONS") != std::string::npos;
        });
        unsetenv("monoprop_PARTITIONS");
    }
    BOOST_CHECK_NO_THROW(construct(0, nullptr));
}

BOOST_AUTO_TEST_CASE(sharded_root_owns_t_routed_shards) {
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto sim = make({.schrodinger = schrodinger});
            build(*sim);
            const size_t team = launch_team();
            const auto &shards = Access::shards(*sim);
            const auto &router = Access::router(*sim);
            BOOST_TEST(static_cast<size_t>(Access::options(*sim).threads) == team);
            BOOST_TEST(shards.size() == team);
            BOOST_TEST(router.ranks() == 1U);
            BOOST_TEST(router.partitions() == team);
            std::set<VecZ> keys;
            size_t rows = 0;
            for (size_t t = 0; t < team; ++t) {
                BOOST_TEST(shards[t]->graph.layers() == sim->graph_layers());
                shards[t]->op.store->for_each([&](const auto &mono, size_t) {
                    BOOST_TEST(router.dest<kN>(mono) == t); // flat owner rank * T + t at rank 0
                    BOOST_TEST(keys.insert(bitset_to_indices<kN>(mono)).second); // no key on two shards
                    ++rows;
                });
            }
            BOOST_TEST(rows == sim->size());
            // Heisenberg keeps the identity out of every shard; it is rank-level metadata, counted once.
            if (!schrodinger) {
                BOOST_TEST(!keys.contains(VecZ{}));
                BOOST_TEST(sim->core_term() == Access::core_term(*sim));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_aggregates_cover_every_shard_once) {
    auto sim = make();
    build(*sim);
    const auto &shards = Access::shards(*sim);
    size_t rows = 0;
    std::pair<size_t, size_t> graph_size{0, 0};
    auto op_memory = monoprop::detail::MPOperatorMemoryBreakdown<kN>{};
    auto graph_memory = GraphMemoryBreakdown{};
    for (const auto &state : shards) {
        rows += state->size();
        op_memory += state->operator_memory_usage();
        graph_memory += state->graph_memory_usage();
        graph_size.second += state->graph.total_cycles();
        BOOST_TEST(state->operator_memory_usage().matched_scratch_bytes == state->matched.memory_bytes());
    }
    BOOST_TEST(sim->size() == rows);
    BOOST_TEST(sim->operator_memory_usage().total_bytes() == op_memory.total_bytes());
    BOOST_TEST(sim->operator_memory_usage().matched_scratch_bytes == op_memory.matched_scratch_bytes);
    BOOST_TEST(sim->operator_memory_usage().inverted_index_bytes == op_memory.inverted_index_bytes);
    BOOST_TEST(sim->graph_memory_usage().total_bytes() == graph_memory.total_bytes());
    BOOST_TEST(sim->graph_memory_usage().cross_rank_bytes == graph_memory.cross_rank_bytes);
    BOOST_TEST(sim->graph_size().second == graph_size.second);
    // Shared metadata is read once: layers and gates are not multiplied by T.
    BOOST_TEST(sim->graph_layers() == lih().majoranas.size());
    BOOST_TEST(sim->n_gates() == lih().majoranas.size());
    BOOST_TEST(sim->parameter_mapping().size() == lih().majoranas.size());
}

BOOST_AUTO_TEST_CASE(sharded_root_owners_do_the_work_of_every_operation) {
    const size_t team = launch_team();
    if (team == 1) {
        return; // needs a nonprimary worker
    }
    Recorder recorder;
    auto sim = make({}, &recorder);
    auto visits = recorder.visits();
    check_owner_visits(visits, team, "seeding");
    for (size_t t = 0; t < team; ++t) {
        BOOST_TEST(count(visits, Seam::shard, 2 * static_cast<int>(sharded::ShardWork::seed), t) == 1U);
        BOOST_TEST(count(visits, Seam::shard, 2 * static_cast<int>(sharded::ShardWork::seed) + 1, t) == 1U);
    }

    const size_t gates = lih().majoranas.size();
    const auto expect = [&](const char *what, Seam seam, int work, size_t per_shard) {
        visits = recorder.visits();
        check_owner_visits(visits, team, what);
        for (size_t t = 0; t < team; ++t) {
            BOOST_TEST_CONTEXT(what << " shard " << t) { BOOST_TEST(count(visits, seam, work, t) == per_shard); }
        }
        recorder.clear();
    };
    recorder.clear();
    build(*sim);
    expect("build_graph", Seam::construction, static_cast<int>(sharded::ConstructionWork::traverse), gates);

    const size_t layers = sim->graph_layers();
    static_cast<void>(sim->expectation_value_and_gradient(params()));
    expect("gradient: retained preparation", Seam::evaluation, static_cast<int>(sharded::EvaluationWork::retain), 1);
    static_cast<void>(sim->expectation_value_and_gradient(params()));
    expect("gradient: reverse derivatives",
           Seam::evaluation,
           static_cast<int>(sharded::EvaluationWork::reverse_finish),
           layers);
    static_cast<void>(sim->expectation_value(params()));
    expect("energy: forward publication", Seam::evaluation, static_cast<int>(sharded::EvaluationWork::publish), layers);
    const auto functional = sim->expectation_value_and_gradient_functional(1e-3);
    recorder.clear();
    static_cast<void>(functional(params()));
    expect("pared functional: reverse accumulation",
           Seam::evaluation,
           static_cast<int>(sharded::EvaluationWork::reverse_accumulate),
           layers);
    static_cast<void>(sim->contract_partially(params(), false));
    expect("contraction: replay finish", Seam::evaluation, static_cast<int>(sharded::EvaluationWork::finish), layers);

    auto informed = make({.lower_atol = 1e-4}, &recorder);
    recorder.clear();
    informed->build_graph(lih().majoranas, lih().param_inds, lih().gen_coeffs, std::nullopt, params());
    expect("informed build: new-layer replay",
           Seam::construction,
           static_cast<int>(sharded::ConstructionWork::replay),
           2 * gates);
    // Extending a built graph with coefficients replays the existing layers as the seed, in the same team.
    VecD doubled = params();
    doubled.insert(doubled.end(), params().begin(), params().end());
    VecZ shifted = lih().param_inds;
    for (auto &p : shifted) {
        p += params().size();
    }
    informed->build_graph(lih().majoranas, shifted, lih().gen_coeffs, std::nullopt, doubled);
    visits = recorder.visits();
    check_owner_visits(visits, team, "informed seed replay");
    for (size_t t = 0; t < team; ++t) {
        BOOST_TEST(count(visits, Seam::construction, static_cast<int>(sharded::ConstructionWork::seed), t)
                   == 1 + (3 * gates)); // the seed phase, then publish, cosine and finish per existing layer
    }
    recorder.clear();

    auto direct = make({}, &recorder);
    recorder.clear();
    direct->propagate(lih().majoranas, lih().param_inds, lih().gen_coeffs, params());
    expect("propagate: fused apply", Seam::construction, static_cast<int>(sharded::ConstructionWork::apply), gates);

    sim->update_initial_operator(OperatorDict{{lih().hamiltonian.begin()->first, lih().hamiltonian.begin()->second}});
    expect("initial-operator update", Seam::root, static_cast<int>(sharded::RootWork::initial_operator), 1);
    sim->set_parameter_mapping(VecZ(layers, 0));
    expect("remap", Seam::root, static_cast<int>(sharded::RootWork::remap), 1);

    const MP copy(*sim);
    expect("copy", Seam::shard, 2 * static_cast<int>(sharded::ShardWork::copy), 1);
}

BOOST_AUTO_TEST_CASE(sharded_root_opaque_cutoff_traverses_on_the_primary) {
    const size_t team = launch_team();
    if (team == 1) {
        return;
    }
    // A basis-change cutoff is an opaque closure: never called from two threads at once.
    std::vector<VecZ> identity_basis;
    for (size_t i = 0; i < 2 * kN; ++i) {
        identity_basis.push_back(VecZ{i});
    }
    Recorder recorder;
    auto sim = make({.basis_change = identity_basis}, &recorder);
    recorder.clear();
    build(*sim);
    for (const auto &v : recorder.visits()) {
        if (v.seam == Seam::construction && v.work == static_cast<int>(sharded::ConstructionWork::traverse)) {
            BOOST_TEST(v.worker == 0);
        }
        else {
            BOOST_TEST(static_cast<size_t>(v.worker) == v.shard);
        }
    }
    auto reference = make();
    build(*reference);
    BOOST_TEST(energy_of(*sim) == energy_of(*reference));
}

BOOST_AUTO_TEST_CASE(sharded_root_export_pairs_each_shard_with_its_block) {
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto sim = make({.schrodinger = schrodinger});
            build(*sim);
            const auto contracted = sim->contract_partially(params(), false);
            const auto terms = sim->evolved_operator_terms(params(), 0.0);
            const auto &shards = Access::shards(*sim);
            // Blocks are concatenated in shard order, each indexed by its own shard's rows.
            std::vector<std::pair<VecZ, std::complex<double>>> expected;
            size_t offset = 0;
            for (const auto &state : shards) {
                state->op.store->for_each([&](const auto &mono, size_t idx) {
                    const auto decoded = algebra_decode_coeff<kN>(Basis::Majorana, contracted[offset + idx], mono);
                    expected.emplace_back(bitset_to_indices<kN>(mono),
                                          std::complex<double>(std::round(decoded.real() * 1e12) / 1e12,
                                                               std::round(decoded.imag() * 1e12) / 1e12));
                });
                offset += state->size();
            }
            BOOST_TEST(offset == contracted.size());
            BOOST_TEST(terms.size() == expected.size());
            BOOST_TEST((terms == expected));
            // atol = 0 keeps every stored row, zeros included, and no key twice.
            std::set<VecZ> keys;
            for (const auto &[key, coeff] : terms) {
                BOOST_TEST(keys.insert(key).second);
            }
            BOOST_TEST(keys.size() == sim->size());
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_contraction_updates_the_right_picture) {
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto sim = make({.schrodinger = schrodinger});
            build(*sim);
            const auto &shards = Access::shards(*sim);
            const auto picture = [&](size_t t) -> const VecD & {
                return schrodinger ? shards[t]->op.state_coeffs : shards[t]->op.op_coeffs;
            };
            std::vector<VecD> before;
            for (size_t t = 0; t < shards.size(); ++t) {
                before.push_back(picture(t));
            }
            const auto layers = sim->graph_layers();
            const auto evolved = sim->contract_partially(params(), false);
            // Non-inplace consumes nothing.
            BOOST_TEST(sim->graph_layers() == layers);
            for (size_t t = 0; t < shards.size(); ++t) {
                BOOST_TEST((picture(t) == before[t]));
            }
            const auto consumed = sim->contract_partially(params(), true);
            BOOST_TEST((consumed == evolved));
            BOOST_TEST(sim->graph_layers() == 0U);
            size_t offset = 0;
            for (size_t t = 0; t < shards.size(); ++t) {
                BOOST_TEST(shards[t]->graph.layers() == 0U);
                const VecD block(evolved.begin() + static_cast<std::ptrdiff_t>(offset),
                                 evolved.begin() + static_cast<std::ptrdiff_t>(offset + picture(t).size()));
                BOOST_TEST((picture(t) == block));
                offset += picture(t).size();
            }
            // A further build starts from the contracted picture.
            build(*sim);
            BOOST_TEST(std::isfinite(energy_of(*sim)));
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_initial_operator_updates_route_to_owners) {
    auto sim = make();
    build(*sim);
    const auto epoch = Access::epoch(*sim);
    OperatorDict scaled;
    for (const auto &[key, value] : lih().hamiltonian) {
        scaled[key] = value * 2.0;
    }
    const auto stale = sim->expectation_value_functional();
    sim->update_initial_operator(scaled);
    BOOST_TEST(Access::epoch(*sim) == epoch + 1); // once, not once per shard
    BOOST_CHECK_THROW(stale(params()), std::exception);
    BOOST_TEST(!Access::is_invalid(*sim));
    const auto &router = Access::router(*sim);
    const auto &shards = Access::shards(*sim);
    for (size_t t = 0; t < shards.size(); ++t) {
        for (const auto &[mono, coeff] : shards[t]->op.init_op_map) {
            BOOST_TEST(router.dest<kN>(mono) == t);
        }
    }
    auto reference = make();
    build(*reference);
    // Linear in the operator: every coefficient doubled, the identity included.
    BOOST_TEST(energy_of(*sim) == 2.0 * energy_of(*reference), boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(sharded_root_remap_replaces_immutable_cores) {
    auto sim = make();
    build(*sim);
    const MP before(*sim);
    const auto &shards = Access::shards(*sim);
    const auto &old_shards = Access::shards(before);
    // The copy shares the source's immutable cores.
    BOOST_TEST(&shards.front()->graph.get_layer(0).core() == &old_shards.front()->graph.get_layer(0).core());
    const size_t layers = sim->graph_layers();
    sim->set_parameter_mapping(VecZ(layers, 0));
    for (size_t t = 0; t < shards.size(); ++t) {
        for (size_t l = 0; l < layers; ++l) {
            BOOST_TEST(&shards[t]->graph.get_layer(l).core() != &old_shards[t]->graph.get_layer(l).core());
            BOOST_TEST(shards[t]->graph.get_layer_traversal(l).param_index() == 0U);
        }
    }
    BOOST_TEST((before.parameter_mapping() != sim->parameter_mapping()));
}

BOOST_AUTO_TEST_CASE(sharded_root_copies_are_owner_allocated_and_independent) {
    auto sim = make();
    build(*sim);
    const double energy = energy_of(*sim);
    auto copy = std::make_unique<MP>(*sim);
    const auto clone = Access::clone(*sim);
    BOOST_TEST(Access::options(*copy).threads == Access::options(*sim).threads);
    for (size_t t = 0; t < Access::shards(*sim).size(); ++t) {
        BOOST_TEST(Access::shards(*copy)[t].get() != Access::shards(*sim)[t].get());
        BOOST_TEST(Access::shards(*copy)[t]->op.store.get() != Access::shards(*sim)[t]->op.store.get());
    }
    copy->update_lower_atol(1e-3);
    build(*copy);
    BOOST_TEST(energy_of(*sim) == energy);
    BOOST_TEST(clone->expectation_value(params()) == energy);
    copy.reset();
    BOOST_TEST(energy_of(*sim) == energy);
}

namespace {

// An ordinary extension: the virtual clone and initial-operator hooks, without the obsolete partition factory.
class Derived final : public MP {
public:
    using MP::MP;
    Derived(const Derived &) = default;

    auto update_initial_operator(const OperatorDict &op_dict) -> void override {
        const auto [terms, coeffs] = apply_initial_operator_(op_dict);
        last_terms = terms.size();
        last_coeffs = coeffs.size();
    }

    size_t last_terms = 0;
    size_t last_coeffs = 0;

protected:
    auto clone_() const -> std::unique_ptr<MP> override { return std::make_unique<Derived>(*this); }
};

} // namespace

BOOST_AUTO_TEST_CASE(sharded_root_virtual_hooks_keep_working) {
    const auto &data = lih();
    Derived sim(data.hamiltonian, 4U, data.initial_state, std::nullopt, MPI_COMM_SELF);
    build(sim);
    const double energy = sim.expectation_value(params());
    const auto clone = Access::clone(sim);
    BOOST_TEST(dynamic_cast<const Derived *>(clone.get()) != nullptr);
    BOOST_TEST(clone->expectation_value(params()) == energy);
    // Through the base interface: the override runs and the protected hook returns this rank's routed share,
    // gathered from every shard (the identity is rank-level metadata and is not among the terms).
    MP &base = sim;
    base.update_initial_operator(data.hamiltonian);
    const size_t routed = data.hamiltonian.size() - (data.hamiltonian.contains(VecZ{}) ? 1 : 0);
    BOOST_TEST(sim.last_terms == routed);
    BOOST_TEST(sim.last_coeffs == routed);
    BOOST_TEST(sim.expectation_value(params()) == energy);
    BOOST_TEST(clone->expectation_value(params()) == energy);
}

BOOST_AUTO_TEST_CASE(sharded_root_rejections_before_mutation_leave_it_usable) {
    auto sim = make();
    build(*sim);
    const double energy = energy_of(*sim);
    const auto &data = lih();
    BOOST_CHECK_THROW(sim->build_graph(data.majoranas, VecZ{0}, data.gen_coeffs), std::exception);
    BOOST_CHECK_THROW(sim->build_graph({VecZ{0, 2 * kN}}, VecZ{0}, VecD{1.0}), std::exception);
    BOOST_CHECK_THROW(sim->propagate(data.majoranas, data.param_inds, data.gen_coeffs, params()), std::exception);
    BOOST_CHECK_THROW(sim->expectation_value(VecD{0.1}), std::exception);
    BOOST_CHECK_THROW(sim->contract_partially(VecD{}, false), std::exception);
    BOOST_CHECK_THROW(sim->update_initial_operator(OperatorDict{{VecZ{0, 2 * kN}, 1.0}}), std::exception);
    BOOST_CHECK_THROW(sim->set_parameter_mapping(VecZ{0}), std::exception);
    BOOST_CHECK_THROW(sim->update_basis_change(std::vector<VecZ>{VecZ{0}}), std::exception);
    BOOST_CHECK_THROW(sim->build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, VecD{0.1}),
                      std::exception);
    BOOST_TEST(!Access::is_invalid(*sim));
    BOOST_TEST(energy_of(*sim) == energy);
    BOOST_TEST(sim->graph_layers() == data.majoranas.size());
}

namespace {

// Arms one seam visit on the last shard (a nonprimary owner at T > 1), runs `op`, and checks the original error
// arrives, the object is invalid with every dependent operation refused, and an earlier copy keeps working.
template <typename Prepare, typename Op>
auto check_post_mutation_failure(const char *what, Recorder::Arm target, Prepare &&prepare, Op &&op) -> void {
    BOOST_TEST_CONTEXT(what) {
        Recorder recorder;
        auto sim = make({}, &recorder);
        prepare(*sim);
        MP earlier(*sim);
        Access::observe(earlier, nullptr);
        const VecD at = earlier.graph_layers() > 0 ? params() : VecD{};
        const double earlier_energy = earlier.expectation_value(at);
        recorder.arm(target);
        BOOST_CHECK_THROW(op(*sim), Injected);
        BOOST_TEST(recorder.fired());
        recorder.disarm();
        BOOST_TEST(Access::is_invalid(*sim));
        BOOST_CHECK_THROW(sim->size(), InvalidPropagatorError);
        BOOST_CHECK_THROW(energy_of(*sim), InvalidPropagatorError);
        BOOST_CHECK_THROW(build(*sim), InvalidPropagatorError);
        BOOST_CHECK_THROW(MP{*sim}, InvalidPropagatorError);
        BOOST_TEST(!Access::is_invalid(earlier));
        BOOST_TEST(earlier.expectation_value(at) == earlier_energy);
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sharded_root_failures_after_mutation_invalidate_the_root) {
    const size_t last = launch_team() - 1;
    const auto none = [](MP &) {};
    const auto built = [](MP &sim) { build(sim); };
    using CW = sharded::ConstructionWork;
    using EW = sharded::EvaluationWork;
    using RW = sharded::RootWork;
    check_post_mutation_failure("construction traversal",
                                {Seam::construction, static_cast<int>(CW::traverse), last, 1},
                                none,
                                [](MP &sim) { build(sim); });
    check_post_mutation_failure("construction publication",
                                {Seam::construction, static_cast<int>(CW::finalize), last, 2},
                                none,
                                [](MP &sim) { build(sim); });
    check_post_mutation_failure("propagation apply",
                                {Seam::construction, static_cast<int>(CW::apply), last, 0},
                                none,
                                [](MP &sim) {
                                    sim.propagate(lih().majoranas, lih().param_inds, lih().gen_coeffs, params());
                                });
    check_post_mutation_failure(
        "informed seed replay",
        {Seam::construction, static_cast<int>(CW::seed), last, 1},
        built,
        [](MP &sim) {
            VecD doubled = params();
            doubled.insert(doubled.end(), params().begin(), params().end());
            VecZ shifted = lih().param_inds;
            for (auto &p : shifted) {
                p += params().size();
            }
            sim.build_graph(lih().majoranas, shifted, lih().gen_coeffs, std::nullopt, doubled);
        });
    check_post_mutation_failure("retained preparation",
                                {Seam::evaluation, static_cast<int>(EW::retain), last, std::nullopt},
                                built,
                                [](MP &sim) { static_cast<void>(sim.expectation_value_functional(1e-3)); });
    check_post_mutation_failure("forward replay of a direct energy",
                                {Seam::evaluation, static_cast<int>(EW::finish), last, 1},
                                built,
                                [](MP &sim) { static_cast<void>(sim.expectation_value(params())); });
    check_post_mutation_failure("reverse derivative of a direct gradient",
                                {Seam::evaluation, static_cast<int>(EW::reverse_finish), last, std::nullopt},
                                built,
                                [](MP &sim) { static_cast<void>(sim.expectation_value_and_gradient(params())); });
    check_post_mutation_failure("result combination after the team",
                                {Seam::root, static_cast<int>(RW::combine), 0, std::nullopt},
                                built,
                                [](MP &sim) { static_cast<void>(sim.expectation_value_and_gradient(params())); });
    check_post_mutation_failure("partial-contraction replay",
                                {Seam::evaluation, static_cast<int>(EW::finish), last, std::nullopt},
                                built,
                                [](MP &sim) { static_cast<void>(sim.contract_partially(params(), true)); });
    check_post_mutation_failure("decoded export",
                                {Seam::root, static_cast<int>(RW::export_block), last, std::nullopt},
                                built,
                                [](MP &sim) { static_cast<void>(sim.evolved_operator_terms(params(), 0.0)); });
    check_post_mutation_failure("initial-operator owner phase",
                                {Seam::root, static_cast<int>(RW::initial_operator), last, std::nullopt},
                                built,
                                [](MP &sim) { sim.update_initial_operator(lih().hamiltonian); });
    check_post_mutation_failure("remap owner phase",
                                {Seam::root, static_cast<int>(RW::remap), last, std::nullopt},
                                built,
                                [](MP &sim) { sim.set_parameter_mapping(VecZ(sim.graph_layers(), 0)); });
}

BOOST_AUTO_TEST_CASE(sharded_root_functional_failures_invalidate_their_owner) {
    const size_t last = launch_team() - 1;
    Recorder recorder;
    auto sim = make({}, &recorder);
    build(*sim);
    const auto functional = sim->expectation_value_and_gradient_functional();
    const auto value = functional(params());
    // Argument errors are checked before the team: the owner stays usable.
    BOOST_CHECK_THROW(functional(VecD{0.1}), std::exception);
    BOOST_TEST(!Access::is_invalid(*sim));
    recorder.arm({Seam::evaluation, static_cast<int>(sharded::EvaluationWork::reverse_accumulate), last, std::nullopt});
    BOOST_CHECK_THROW(functional(params()), Injected);
    BOOST_TEST(Access::is_invalid(*sim));
    BOOST_CHECK_THROW(functional(params()), InvalidPropagatorError);
    static_cast<void>(value);
}

BOOST_AUTO_TEST_CASE(sharded_root_functional_epoch_and_graph_guards) {
    auto sim = make();
    build(*sim);
    const auto energy = sim->expectation_value_functional();
    const auto pared = sim->expectation_value_and_gradient_functional(1e-3);
    const double before = energy(params());
    sim->update_initial_operator(lih().hamiltonian);
    BOOST_CHECK_THROW(energy(params()), std::exception);
    BOOST_CHECK_THROW(pared(params()), std::exception);
    const auto fresh = sim->expectation_value_functional();
    BOOST_TEST(fresh(params()) == before);
    // An unpared functional aliases the live graphs, so appending layers is detected.
    build(*sim);
    BOOST_CHECK_THROW(fresh(params()), std::exception);
    BOOST_TEST(!Access::is_invalid(*sim));
}

BOOST_AUTO_TEST_CASE(sharded_root_copy_and_seed_failures_are_contained) {
    const size_t last = launch_team() - 1;
    Recorder recorder;
    recorder.arm({Seam::shard, 2 * static_cast<int>(sharded::ShardWork::seed), last, std::nullopt});
    BOOST_CHECK_THROW(make({}, &recorder), Injected);
    recorder.disarm();
    auto sim = make({}, &recorder);
    build(*sim);
    const double energy = energy_of(*sim);
    recorder.arm({Seam::shard, 2 * static_cast<int>(sharded::ShardWork::copy) + 1, last, std::nullopt});
    BOOST_CHECK_THROW(MP{*sim}, Injected);
    recorder.disarm();
    // A failed copy leaves its source intact and valid.
    BOOST_TEST(!Access::is_invalid(*sim));
    BOOST_TEST(energy_of(*sim) == energy);
}

BOOST_AUTO_TEST_CASE(sharded_root_allocation_failures_are_catchable) {
    if (!allocation::available()) {
        return; // the probe is compiled out under this sanitizer
    }
    // Every operator new on the calling thread (the caller and, inside teams, worker 0) fails in turn. Each failure
    // must surface as std::bad_alloc: before mutation the object is unchanged, after it the object is invalid.
    const auto data = load_case_data<8>("random_exact.msgpack");
    using Small = MonomialPropagator<8>;
    using SmallAccess = monoprop::detail::PropagatorTestAccess<8>;
    const auto make_small = [&] { return Small(data.hamiltonian, 16, data.initial_state, std::nullopt, MPI_COMM_SELF); };
    auto reference = make_small();
    reference.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double expected = reference.expectation_value(data.parameters);
    size_t failures = 0;
    for (size_t nth = 1; nth < 20000; ++nth) {
        auto sim = make_small();
        allocation::arm_failure(nth);
        bool threw = false;
        try {
            sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            static_cast<void>(sim.expectation_value_and_gradient(data.parameters));
        }
        catch (const std::bad_alloc &) {
            threw = true;
        }
        const bool pending = allocation::disarm_failure();
        if (!threw) {
            if (pending) {
                break; // the armed allocation never happened: the sweep is complete
            }
            // A nothrow allocation (a standard algorithm's temporary buffer) failed and its caller fell back: the
            // results must be unaffected.
            BOOST_TEST_CONTEXT("nth=" << nth) {
                BOOST_TEST(!SmallAccess::is_invalid(sim));
                BOOST_TEST(sim.expectation_value(data.parameters) == expected);
            }
            continue;
        }
        ++failures;
        if (!SmallAccess::is_invalid(sim)) {
            if (sim.graph_layers() == data.majoranas.size()) {
                BOOST_TEST(sim.expectation_value(data.parameters) == expected);
            }
            else {
                BOOST_TEST(sim.graph_layers() == 0U);
            }
        }
        else {
            BOOST_CHECK_THROW(sim.size(), InvalidPropagatorError);
        }
    }
    BOOST_TEST(failures > 0U);
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_launch_is_rejected) {
    const auto data = load_case_data<8>("random_exact.msgpack");
    const auto construct = [&] {
        return MonomialPropagator<8>(data.hamiltonian, 16, data.initial_state, std::nullopt, MPI_COMM_WORLD);
    };
    if (mpi::size(mpi::Comm(MPI_COMM_WORLD)) == 1) {
        BOOST_CHECK_NO_THROW(construct());
        return;
    }
    // Every rank rejects locally, before any collective, so no rank is left waiting.
    BOOST_CHECK_EXCEPTION(construct(), PropagatorConfigError, [](const auto &e) {
        return std::string(e.what()).find("one MPI rank") != std::string::npos;
    });
}

BOOST_AUTO_TEST_CASE(sharded_root_empty_operators_and_identity_generators) {
    auto empty = Access::construct_observed(nullptr,
                                            OperatorDict{},
                                            4U,
                                            VecZ{},
                                            std::optional<unsigned int>{},
                                            mpi::Comm(MPI_COMM_SELF),
                                            std::optional<double>{},
                                            std::optional<double>{},
                                            CutoffType::Length,
                                            std::optional<std::vector<VecZ>>{},
                                            kN,
                                            Basis::Majorana,
                                            size_t{0},
                                            typename MP::PartitionChildFactory{});
    BOOST_TEST(empty->size() == 0U);
    empty->build_graph({VecZ{0, 1}, VecZ{}}, VecZ{0, 1}, VecD{1.0, 1.0});
    BOOST_TEST(empty->graph_layers() == 2U);
    BOOST_TEST(empty->expectation_value(VecD{0.3, 0.2}) == 0.0);
    const auto [value, gradient] = empty->expectation_value_and_gradient(VecD{0.3, 0.2});
    BOOST_TEST(value == 0.0);
    BOOST_TEST(gradient.size() == 2U);
    BOOST_TEST(empty->evolved_operator_terms(VecD{0.3, 0.2}, 0.0).empty());
    BOOST_TEST(empty->contract_partially(VecD{0.3, 0.2}, true).empty());

    // The replicated identity is rank-level metadata: added once to every result, at every T, never exported by
    // a shard.
    auto core_only = Access::construct_observed(nullptr,
                                                OperatorDict{{VecZ{}, std::complex<double>{2.5, 0.0}}},
                                                4U,
                                                VecZ{},
                                                std::optional<unsigned int>{},
                                                mpi::Comm(MPI_COMM_SELF),
                                                std::optional<double>{},
                                                std::optional<double>{},
                                                CutoffType::Length,
                                                std::optional<std::vector<VecZ>>{},
                                                kN,
                                                Basis::Majorana,
                                                size_t{0},
                                                typename MP::PartitionChildFactory{});
    core_only->build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0});
    BOOST_TEST(core_only->core_term() == 2.5);
    BOOST_TEST(core_only->expectation_value(VecD{0.3}) == 2.5);
    BOOST_TEST(core_only->expectation_value_and_gradient(VecD{0.3}).first == 2.5);
    BOOST_TEST(core_only->expectation_value_functional(1e-3)(VecD{0.3}) == 2.5);
    BOOST_TEST(core_only->evolved_operator_terms(VecD{0.3}, 0.0).empty());
}

#endif // monoprop_SHARDED_OPENMP_PROTOTYPE
