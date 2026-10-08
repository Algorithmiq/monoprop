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
 * The sharded OpenMP root: MonomialPropagator over the T shard states of one rank, through its public operations. The
 * team size T is the captured launch budget; cpp/tests/CMakeLists.txt reruns every case in fresh processes at T = 1, 2
 * and 4 (sharded_root_env_t*), and the serial per-case registrations run at T = 1. Cases that need a nonprimary worker
 * return early at T = 1.
 *
 * Private inspection, phase observation and failure injection go through PropagatorTestAccess (shards(), the (1, T)
 * router, and the test-only RootObserver every seam reports to). Observers record the executing worker inside the
 * protected phase; assertions run after the team has joined. Numerical parity against an independently built runtime at
 * the same (1, T) geometry, and across T, is checked from separate processes by tests/test_sharded_openmp.py.
 */

#include <boost/test/unit_test.hpp>

#include <cstdlib>
#include <string>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"

#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "AllocationProbe.h"
#include "PropagatorTestAccess.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"
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
                                      config.basis);
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

// Every owner visit of shard t ran on worker t of a level-1 team of T workers (a one-worker team runs on the caller,
// at level 0, without a region); caller work ran outside any team.
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
            BOOST_TEST(v.level == (team > 1 ? 1 : 0));
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
    const auto remedy = [](const MultiShardUnsupported &e) {
        return std::string(e.what()).find("monoprop_NUM_THREADS=1") != std::string::npos
               && std::string(e.what()).find("partitions=1") == std::string::npos;
    };
    BOOST_CHECK_EXCEPTION(sim.mp_op(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(std::as_const(sim).mp_op(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.graph(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.indexing(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.graph_data(), MultiShardUnsupported, remedy);
    // The rejection mutates nothing: the object stays valid and keeps answering.
    BOOST_TEST(!monoprop::detail::PropagatorTestAccess<8>::is_invalid(sim));
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_CHECK_EXCEPTION(sim.graph(), MultiShardUnsupported, remedy);
    BOOST_TEST(std::isfinite(sim.expectation_value(data.parameters)));
    BOOST_TEST(!monoprop::detail::PropagatorTestAccess<8>::is_invalid(sim));
}

// At T = 1 on one rank the root evaluates its sole shard with the low-level serial evaluator; with an observer attached
// it runs the sharded phases. Both give bitwise the same energies and gradients through every entry point, both
// pictures, pared and unpared, and a bad parameter vector fails the same way on both, leaving the object valid.
BOOST_AUTO_TEST_CASE(sharded_root_sole_shard_serial_evaluation_matches_the_phases) {
    if (launch_team() != 1) {
        return; // the serial path is only for a sole shard
    }
    using Access = monoprop::detail::PropagatorTestAccess<8>;
    const auto data = load_case_data<8>("random_exact.msgpack");
    const sharded::RootObserver phases; // observes nothing; its presence keeps the sharded phases
    const auto bits = [](double v) { return std::bit_cast<uint64_t>(v); };
    const auto same = [&](const std::pair<double, VecD> &a, const std::pair<double, VecD> &b) {
        return bits(a.first) == bits(b.first) && a.second.size() == b.second.size()
               && std::ranges::equal(a.second, b.second, [&](double x, double y) { return bits(x) == bits(y); });
    };
    for (const std::optional<unsigned int> schrodinger :
         {std::optional<unsigned int>{}, std::optional<unsigned int>{16}}) {
        for (const std::optional<double> pare : {std::optional<double>{}, std::optional<double>{1e-3}}) {
            BOOST_TEST_CONTEXT("schrodinger " << schrodinger.has_value() << " pare " << pare.has_value()) {
                MonomialPropagator<8> sim(data.hamiltonian, 16, data.initial_state, schrodinger, MPI_COMM_SELF);
                sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
                const auto energy = sim.expectation_value_functional(pare);
                const auto gradient = sim.expectation_value_and_gradient_functional(pare);
                const std::pair<double, VecD> serial_e{energy(data.parameters), {}};
                const auto serial_g = gradient(data.parameters);
                const auto serial_direct = sim.expectation_value_and_gradient(data.parameters);
                Access::observe(sim, &phases);
                const std::pair<double, VecD> phased_e{energy(data.parameters), {}};
                const auto phased_g = gradient(data.parameters);
                const auto phased_direct = sim.expectation_value_and_gradient(data.parameters);
                Access::observe(sim, nullptr);
                BOOST_TEST(same(serial_e, phased_e));
                BOOST_TEST(same(serial_g, phased_g));
                BOOST_TEST(same(serial_direct, phased_direct));
                BOOST_TEST(!serial_g.second.empty());
                // Too few parameters for the mapping: the same diagnostic on both paths, nothing mutated.
                const VecD short_params(1, 0.1);
                std::string serial_message;
                std::string phased_message;
                try {
                    static_cast<void>(gradient(short_params));
                }
                catch (const std::exception &e) {
                    serial_message = e.what();
                }
                Access::observe(sim, &phases);
                try {
                    static_cast<void>(gradient(short_params));
                }
                catch (const std::exception &e) {
                    phased_message = e.what();
                }
                Access::observe(sim, nullptr);
                BOOST_TEST(!serial_message.empty());
                BOOST_TEST(serial_message == phased_message);
                BOOST_TEST(!Access::is_invalid(sim));
                BOOST_TEST(same(gradient(data.parameters), serial_g));
            }
        }
    }
}

// An invalid owner fails its validity guard before the shard-count check, at every T.
BOOST_AUTO_TEST_CASE(sharded_root_raw_accessors_check_validity_first) {
    auto sim = make();
    build(*sim);
    Access::set_cutoff_fn(*sim, [](const Monomial<kN> &) -> bool { throw Injected("injected cutoff failure"); });
    BOOST_CHECK_THROW(build(*sim), Injected);
    BOOST_TEST_REQUIRE(Access::is_invalid(*sim));
    BOOST_CHECK_THROW(sim->mp_op(), InvalidPropagatorError);
    BOOST_CHECK_THROW(std::as_const(*sim).mp_op(), InvalidPropagatorError);
    BOOST_CHECK_THROW(sim->indexing(), InvalidPropagatorError);
    BOOST_CHECK_THROW(std::as_const(*sim).indexing(), InvalidPropagatorError);
    BOOST_CHECK_THROW(sim->graph(), InvalidPropagatorError);
    BOOST_CHECK_THROW(sim->graph_data(), InvalidPropagatorError);
}

namespace {

// Sets one environment variable for a scope (or unsets it, for nullopt), then restores the previous state.
class ScopedEnv {
public:
    ScopedEnv(const char *name, std::optional<std::string> value) : name_(name) {
        if (const char *old = std::getenv(name)) {
            old_ = std::string(old);
        }
        if (value) {
            ::setenv(name, value->c_str(), 1);
        }
        else {
            ::unsetenv(name);
        }
    }
    ScopedEnv(const ScopedEnv &) = delete;
    auto operator=(const ScopedEnv &) -> ScopedEnv & = delete;
    ~ScopedEnv() {
        if (old_) {
            ::setenv(name_, old_->c_str(), 1);
        }
        else {
            ::unsetenv(name_);
        }
    }

private:
    const char *name_;
    std::optional<std::string> old_;
};

} // namespace

// The obsolete partition selector is not read at all: whatever it holds while a root is constructed and used, the
// root has the launch's T shards and answers bitwise as with the variable unset. Only the captured budget decides.
BOOST_AUTO_TEST_CASE(sharded_root_ignores_the_obsolete_partition_variable) {
    double energy = 0.0;
    std::vector<std::pair<VecZ, std::complex<double>>> terms;
    {
        const ScopedEnv unset("monoprop_PARTITIONS", std::nullopt);
        auto sim = make();
        build(*sim);
        energy = energy_of(*sim);
        terms = sim->evolved_operator_terms(params(), 0.0);
    }
    for (const char *value : {"off", "auto", "1", "2", "4", "0", "", "-3", "not-a-count"}) {
        BOOST_TEST_CONTEXT("monoprop_PARTITIONS=\"" << value << "\"") {
            const ScopedEnv set("monoprop_PARTITIONS", std::string(value));
            std::unique_ptr<MP> sim;
            BOOST_CHECK_NO_THROW(sim = make());
            BOOST_TEST_REQUIRE(sim != nullptr);
            BOOST_TEST(Access::shards(*sim).size() == launch_team());
            BOOST_TEST(Access::options(*sim).threads == static_cast<int>(launch_team()));
            build(*sim);
            BOOST_TEST(energy_of(*sim) == energy);
            BOOST_TEST((sim->evolved_operator_terms(params(), 0.0) == terms));
            const MP copy(*sim);
            BOOST_TEST(Access::shards(copy).size() == launch_team());
        }
    }
}

// Registered as sharded_root_env_openmp_default_t3: monoprop_NUM_THREADS unset and OMP_NUM_THREADS=3. Neither the test
// runner nor the library may supply the variable, so the root takes omp_get_max_threads() and runs that many owners.
BOOST_AUTO_TEST_CASE(sharded_root_unset_budget_takes_the_openmp_default) {
    const char *expected = std::getenv("monoprop_TEST_EXPECT_UNSET_BUDGET");
    if (expected == nullptr) {
        BOOST_TEST_MESSAGE("not the unset-budget launch");
        return;
    }
    const auto team = static_cast<size_t>(std::stoul(expected));
    BOOST_TEST(std::getenv("monoprop_NUM_THREADS") == nullptr);
    BOOST_TEST(static_cast<size_t>(omp_get_max_threads()) == team);
    Recorder recorder;
    auto sim = make({}, &recorder);
    BOOST_TEST(static_cast<size_t>(Access::options(*sim).threads) == team);
    BOOST_TEST(Access::shards(*sim).size() == team);
    build(*sim);
    static_cast<void>(energy_of(*sim));
    check_owner_visits(recorder.visits(), team, "seeding, construction and evaluation at the OpenMP default");
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
                    BOOST_TEST(router.dest<kN>(mono) == t);                      // flat owner rank * T + t at rank 0
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
            BOOST_TEST_CONTEXT(what << " shard " << t) {
                BOOST_TEST(count(visits, seam, work, t) == per_shard);
            }
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
        ++calls;
        std::tie(last_terms, last_coeffs) = apply_initial_operator_(op_dict);
    }

    size_t calls = 0;
    MonomialList<kN> last_terms;
    VecD last_coeffs;

protected:
    auto clone_() const -> std::unique_ptr<MP> override { return std::make_unique<Derived>(*this); }
};

} // namespace

BOOST_AUTO_TEST_CASE(sharded_root_virtual_hooks_keep_working) {
    const auto &data = lih();
    Derived sim(data.hamiltonian, 4U, data.initial_state, std::nullopt, MPI_COMM_SELF);
    build(sim);
    const double energy = sim.expectation_value(params());
    // clone_() keeps the derived type, with its own shards and the source's T.
    const auto clone = Access::clone(sim);
    BOOST_TEST(dynamic_cast<const Derived *>(clone.get()) != nullptr);
    BOOST_TEST(Access::options(*clone).threads == Access::options(sim).threads);
    BOOST_TEST(Access::shards(*clone).size() == launch_team());
    for (size_t t = 0; t < Access::shards(sim).size(); ++t) {
        BOOST_TEST(Access::shards(*clone)[t]->op.store.get() != Access::shards(sim)[t]->op.store.get());
    }
    BOOST_TEST(clone->expectation_value(params()) == energy);
    Derived copy(sim);

    // Through the base interface: the override runs, and the protected hook returns this rank's routed share, every
    // shard's block in shard order (the identity is rank-level metadata and is not among the terms).
    OperatorDict scaled;
    for (const auto &[key, value] : data.hamiltonian) {
        scaled[key] = value * 0.5;
    }
    MP &base = sim;
    base.update_initial_operator(scaled);
    BOOST_TEST(sim.calls == 1U);
    const size_t routed = data.hamiltonian.size() - (data.hamiltonian.contains(VecZ{}) ? 1 : 0);
    BOOST_TEST(sim.last_terms.size() == routed);
    BOOST_TEST(sim.last_coeffs.size() == routed);
    // The returned share is each key once, routed to one of this rank's shards, with its encoded new coefficient.
    const auto &router = Access::router(sim);
    std::set<VecZ> returned;
    for (size_t i = 0; i < sim.last_terms.size(); ++i) {
        const auto &mono = sim.last_terms[i];
        const auto key = bitset_to_indices<kN>(mono);
        BOOST_TEST(returned.insert(key).second);
        BOOST_TEST(router.dest<kN>(mono) < launch_team());
        BOOST_TEST_REQUIRE(scaled.contains(key));
        BOOST_TEST(sim.last_coeffs[i] == algebra_encode_coeff<kN>(Basis::Majorana, scaled.at(key), mono));
    }
    // The update reached the source only: the copy and the clone answer as before.
    const double updated = sim.expectation_value(params());
    BOOST_TEST(updated != energy);
    BOOST_TEST(clone->expectation_value(params()) == energy);
    BOOST_TEST(copy.expectation_value(params()) == energy);
    BOOST_TEST(copy.calls == 0U);
    // And the other way round: updating the copy leaves the source alone.
    copy.update_initial_operator(data.hamiltonian);
    BOOST_TEST(copy.calls == 1U);
    BOOST_TEST(copy.expectation_value(params()) == energy);
    BOOST_TEST(sim.expectation_value(params()) == updated);
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
    check_post_mutation_failure(
        "propagation apply",
        {Seam::construction, static_cast<int>(CW::apply), last, 0},
        none,
        [](MP &sim) { sim.propagate(lih().majoranas, lih().param_inds, lih().gen_coeffs, params()); });
    check_post_mutation_failure("informed seed replay",
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
    const auto make_small = [&] {
        return Small(data.hamiltonian, 16, data.initial_state, std::nullopt, MPI_COMM_SELF);
    };
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

// --- Multi-rank roots (P ranks x T threads; cpp/tests/CMakeLists.txt launches these under mpiexec) -------------------

namespace {

auto world_size() -> size_t {
    return static_cast<size_t>(mpi::size(mpi::Comm(MPI_COMM_WORLD)));
}

auto world_rank() -> size_t {
    return static_cast<size_t>(mpi::rank(mpi::Comm(MPI_COMM_WORLD)));
}

// A linear router needs a power-of-two rank count; splitmix routes any.
auto routable_world() -> bool {
    return !routing::linear_requested() || std::has_single_bit(world_size());
}

// Multi-rank cases run only in their dedicated launches, which name the world size (cpp/tests/CMakeLists.txt).
auto dedicated_launch() -> bool {
    const char *text = std::getenv("monoprop_TEST_SHARDED_RANKS");
    return text != nullptr && std::stoul(text) == world_size();
}

auto has_routable_ranks(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    boost::test_tools::assertion_result result(world_size() >= 2 && dedicated_launch() && routable_world());
    result.message() << "the launch has " << world_size() << " rank(s)"
                     << (routing::linear_requested() ? " under linear routing" : "")
                     << "; this case needs at least 2 ranks the router accepts";
    return result;
}

auto make_on(mpi::Comm comm, const Config &config = {}, const sharded::RootObserver *observer = nullptr)
    -> std::unique_ptr<MP> {
    const auto &data = lih();
    return Access::construct_observed(observer,
                                      data.hamiltonian,
                                      4U,
                                      data.initial_state,
                                      config.schrodinger ? std::optional<unsigned int>{6U} : std::nullopt,
                                      comm,
                                      config.lower_atol,
                                      std::optional<double>{},
                                      CutoffType::Length,
                                      config.basis_change,
                                      kN,
                                      config.basis);
}

// The plan's general tolerance for cross-geometry comparisons.
auto close(double a, double b) -> bool {
    return std::abs(a - b) <= 1e-9 + (1e-7 * std::max(std::abs(a), std::abs(b)));
}

auto key_hash(const VecZ &key) -> uint64_t {
    uint64_t h = 0x9E37'79B9'7F4A'7C15ULL;
    for (const size_t i : key) {
        h = (h ^ i) * 0x100'0000'01B3ULL;
        h ^= h >> 29;
    }
    return h;
}

/*
 * The rank-local terms of `world`, gathered implicitly: every local key must be in `reference` (one process's whole
 * map) with a close value, and over all ranks the key count and the wrapping sum of key hashes must equal the
 * reference's, so no key is missing or owned twice.
 */
auto check_global_terms(const std::vector<std::pair<VecZ, std::complex<double>>> &local,
                        const std::vector<std::pair<VecZ, std::complex<double>>> &reference,
                        const char *what) -> void {
    BOOST_TEST_CONTEXT(what << " on rank " << world_rank() << " of " << world_size()) {
        std::map<VecZ, std::complex<double>> want(reference.begin(), reference.end());
        BOOST_TEST(want.size() == reference.size());
        size_t missing = 0;
        size_t off = 0;
        uint64_t hashes = 0;
        for (const auto &[key, value] : local) {
            hashes += key_hash(key);
            const auto it = want.find(key);
            if (it == want.end()) {
                ++missing;
                continue;
            }
            if (!close(value.real(), it->second.real()) || !close(value.imag(), it->second.imag())) {
                ++off;
            }
        }
        BOOST_TEST(missing == 0U);
        BOOST_TEST(off == 0U);
        uint64_t want_hashes = 0;
        for (const auto &[key, value] : reference) {
            want_hashes += key_hash(key);
        }
        const auto comm = mpi::Comm(MPI_COMM_WORLD);
        BOOST_TEST(mpi::allreduce_sum<size_t>(local.size(), comm) == reference.size());
        BOOST_TEST(mpi::allreduce_sum<uint64_t>(hashes, comm) == want_hashes);
    }
}

auto check_close_vectors(const VecD &got, const VecD &want, const char *what) -> void {
    BOOST_TEST_CONTEXT(what) {
        BOOST_TEST_REQUIRE(got.size() == want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            BOOST_TEST(close(got[i], want[i]), "component " << i << ": " << got[i] << " vs " << want[i]);
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sharded_root_multirank_rejects_an_unroutable_geometry) {
    if (world_size() < 2 || !dedicated_launch()) {
        BOOST_TEST_MESSAGE("not a dedicated multi-rank launch");
        return;
    }
    if (routable_world()) {
        BOOST_CHECK_NO_THROW(make_on(mpi::Comm(MPI_COMM_WORLD)));
        return;
    }
    // Every rank derives the same verdict from the agreed mode and the shared size, so all of them throw together.
    BOOST_CHECK_THROW(make_on(mpi::Comm(MPI_COMM_WORLD)), routing::UnroutableGeometry);
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_owns_its_flat_owners, *boost::unit_test::precondition(has_routable_ranks)) {
    const size_t team = launch_team();
    auto world = make_on(mpi::Comm(MPI_COMM_WORLD));
    auto self = make_on(mpi::Comm(MPI_COMM_SELF));
    const auto &router = Access::router(*world);
    BOOST_TEST(router.ranks() == world_size());
    BOOST_TEST(router.partitions() == team);
    const auto &shards = Access::shards(*world);
    BOOST_TEST_REQUIRE(shards.size() == team);
    size_t misrouted = 0;
    for (size_t t = 0; t < team; ++t) {
        shards[t]->op.store->for_each(
            [&](const auto &mono, size_t) { misrouted += router.dest<kN>(mono) == (world_rank() * team) + t ? 0 : 1; });
    }
    BOOST_TEST(misrouted == 0U);
    // The seeded rows tile the one-process operator, and so do the built rows.
    BOOST_TEST(mpi::allreduce_sum<size_t>(world->size(), mpi::Comm(MPI_COMM_WORLD)) == self->size());
    build(*world);
    build(*self);
    BOOST_TEST(mpi::allreduce_sum<size_t>(world->size(), mpi::Comm(MPI_COMM_WORLD)) == self->size());
    BOOST_TEST(world->graph_layers() == self->graph_layers());
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_operations_reuse_the_owners_rounds,
                     *boost::unit_test::precondition(has_routable_ranks)) {
    using Kind = sharded::PhysicalRounds::Kind;
    const auto &data = lih();
    const size_t team = launch_team();
    auto world = make_on(mpi::Comm(MPI_COMM_WORLD));
    const sharded::PhysicalRounds *rounds = Access::rounds(*world);
    BOOST_TEST_REQUIRE(rounds != nullptr);
    BOOST_TEST(rounds->peek(Kind::queries).threads() == 0U);
    // Graph construction creates the owner's query and answer rounds; evaluation, its replay round; later calls reuse
    // them: the same objects, not rounds of their own.
    build(*world);
    BOOST_TEST(rounds->peek(Kind::queries).threads() == team);
    BOOST_TEST(rounds->peek(Kind::graph_answers).threads() == team);
    const sharded::PhysicalExchange *queries = &rounds->peek(Kind::queries);
    static_cast<void>(world->expectation_value(data.parameters));
    BOOST_TEST(rounds->peek(Kind::replay).threads() == team);
    BOOST_TEST(rounds->peek(Kind::queries).live() == 0);
    BOOST_TEST(&rounds->peek(Kind::queries) == queries);
    // A copy never shares them.
    auto copy = std::make_unique<MP>(*world);
    BOOST_TEST(Access::rounds(*copy) != rounds);
    BOOST_TEST(Access::rounds(*copy)->peek(Kind::queries).threads() == 0U);
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_matches_one_process, *boost::unit_test::precondition(has_routable_ranks)) {
    const auto &data = lih();
    for (const Config config : {Config{},
                                Config{.schrodinger = true},
                                Config{.lower_atol = 1e-3},
                                Config{.schrodinger = true, .lower_atol = 1e-4}}) {
        BOOST_TEST_CONTEXT("schrodinger=" << config.schrodinger << " lower_atol=" << config.lower_atol.has_value()) {
            auto world = make_on(mpi::Comm(MPI_COMM_WORLD), config);
            auto self = make_on(mpi::Comm(MPI_COMM_SELF), config);
            // Structural construction, or coefficient-informed when an atol is set.
            if (config.lower_atol) {
                world->build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, data.parameters);
                self->build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, data.parameters);
            }
            else {
                build(*world);
                build(*self);
            }
            const auto &p = params();
            BOOST_TEST(close(world->expectation_value(p), self->expectation_value(p)));
            const auto [wv, wg] = world->expectation_value_and_gradient(p);
            const auto [sv, sg] = self->expectation_value_and_gradient(p);
            BOOST_TEST(close(wv, sv));
            check_close_vectors(wg, sg, "gradient");
            const auto [pv, pg] = world->expectation_value_and_gradient_functional(1e-3)(p);
            const auto [qv, qg] = self->expectation_value_and_gradient_functional(1e-3)(p);
            BOOST_TEST(close(pv, qv));
            check_close_vectors(pg, qg, "pared gradient");
            check_global_terms(world->evolved_operator_terms(p, 0.0), self->evolved_operator_terms(p, 0.0), "export");
            // A copy evaluates alike and mutates independently.
            auto copy = std::make_unique<MP>(*world);
            BOOST_TEST(close(copy->expectation_value(p), wv));
            // Contract in place, then extend the contracted picture.
            BOOST_TEST(mpi::allreduce_sum<size_t>(world->contract_partially(p, true).size(), mpi::Comm(MPI_COMM_WORLD))
                       == self->contract_partially(p, true).size());
            BOOST_TEST(close(world->expectation_value({}), self->expectation_value({})));
            build(*world);
            build(*self);
            BOOST_TEST(close(world->expectation_value(p), self->expectation_value(p)));
            BOOST_TEST(close(copy->expectation_value(p), wv));
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_propagation_matches_one_process,
                     *boost::unit_test::precondition(has_routable_ranks)) {
    const auto &data = lih();
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto world = make_on(mpi::Comm(MPI_COMM_WORLD), Config{.schrodinger = schrodinger});
            auto self = make_on(mpi::Comm(MPI_COMM_SELF), Config{.schrodinger = schrodinger});
            world->propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
            self->propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
            BOOST_TEST(close(world->expectation_value({}), self->expectation_value({})));
            check_global_terms(world->evolved_operator_terms({}, 0.0), self->evolved_operator_terms({}, 0.0), "map");
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_updates_route_to_their_owners,
                     *boost::unit_test::precondition(has_routable_ranks)) {
    const auto &data = lih();
    auto world = make_on(mpi::Comm(MPI_COMM_WORLD));
    auto self = make_on(mpi::Comm(MPI_COMM_SELF));
    build(*world);
    build(*self);
    OperatorDict scaled;
    for (const auto &[key, value] : data.hamiltonian) {
        scaled[key] = value * 0.5;
    }
    world->update_initial_operator(scaled);
    self->update_initial_operator(scaled);
    BOOST_TEST(close(world->expectation_value(params()), self->expectation_value(params())));
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_mpi_runs_on_the_primary_and_owners_on_their_workers,
                     *boost::unit_test::precondition(has_routable_ranks)) {
    const size_t team = launch_team();
    Recorder recorder;
    auto world = make_on(mpi::Comm(MPI_COMM_WORLD), Config{.lower_atol = 1e-3}, &recorder);
    const auto &data = lih();
    world->build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, data.parameters);
    static_cast<void>(world->expectation_value_and_gradient(params()));
    const auto visits = recorder.visits();
    size_t exchanges = 0;
    for (const auto &v : visits) {
        if (v.seam == Seam::construction && v.work == static_cast<int>(sharded::ConstructionWork::exchange)) {
            ++exchanges;
            // A physical round's MPI calls run on the team's primary, the caller's thread.
            BOOST_TEST(v.worker == 0);
            BOOST_TEST(v.thread == std::this_thread::get_id());
        }
    }
    BOOST_TEST(exchanges > 0U);
    std::vector<Visit> owners;
    std::ranges::copy_if(visits, std::back_inserter(owners), [](const Visit &v) {
        return !(v.seam == Seam::construction && v.work == static_cast<int>(sharded::ConstructionWork::exchange));
    });
    check_owner_visits(owners, team, "multi-rank construction and evaluation");
}

// Raw access depends on T alone: at T = 1 every rank hands out its actual sole shard, whatever P; above it every rank
// rejects. The ordinary extension hooks work over P ranks, each rank applying and returning its own share only.
BOOST_AUTO_TEST_CASE(sharded_root_multirank_raw_accessors_and_hooks,
                     *boost::unit_test::precondition(has_routable_ranks)) {
    const size_t team = launch_team();
    const auto &data = lih();
    auto world = make_on(mpi::Comm(MPI_COMM_WORLD));
    build(*world);
    const auto &shards = Access::shards(*world);
    if (team == 1) {
        BOOST_TEST(&world->mp_op() == &shards.front()->op);
        BOOST_TEST(&std::as_const(*world).mp_op() == &shards.front()->op);
        BOOST_TEST(&world->indexing() == shards.front()->op.store.get());
        BOOST_TEST(&world->graph() == &shards.front()->graph);
        BOOST_TEST(world->graph_data().size() == world->graph_layers());
        BOOST_TEST(world->mp_op().size() == world->size());
    }
    else {
        BOOST_CHECK_THROW(world->mp_op(), MultiShardUnsupported);
        BOOST_CHECK_THROW(world->indexing(), MultiShardUnsupported);
        BOOST_CHECK_THROW(world->graph(), MultiShardUnsupported);
        BOOST_CHECK_THROW(world->graph_data(), MultiShardUnsupported);
    }
    BOOST_TEST(!Access::is_invalid(*world));
    BOOST_TEST(std::isfinite(world->expectation_value(params())));

    Derived derived(data.hamiltonian, 4U, data.initial_state, std::nullopt, MPI_COMM_WORLD);
    build(derived);
    const double energy = derived.expectation_value(params());
    const auto clone = Access::clone(derived);
    BOOST_TEST(dynamic_cast<const Derived *>(clone.get()) != nullptr);
    MP &base = derived;
    base.update_initial_operator(data.hamiltonian);
    BOOST_TEST(derived.calls == 1U);
    const auto &router = Access::router(derived);
    size_t foreign = 0;
    for (const auto &mono : derived.last_terms) {
        const size_t owner = router.dest<kN>(mono);
        foreign += owner / team == world_rank() ? 0 : 1;
    }
    BOOST_TEST(foreign == 0U);
    const size_t routed = data.hamiltonian.size() - (data.hamiltonian.contains(VecZ{}) ? 1 : 0);
    BOOST_TEST(mpi::allreduce_sum<size_t>(derived.last_terms.size(), mpi::Comm(MPI_COMM_WORLD)) == routed);
    BOOST_TEST(close(derived.expectation_value(params()), energy));
    BOOST_TEST(clone->expectation_value(params()) == energy);
}

BOOST_AUTO_TEST_CASE(sharded_root_multirank_rejections_before_mutation_leave_it_usable,
                     *boost::unit_test::precondition(has_routable_ranks)) {
    // Replicated arguments fail the same validation on every rank, before any team or collective.
    auto world = make_on(mpi::Comm(MPI_COMM_WORLD));
    build(*world);
    const double before = world->expectation_value(params());
    BOOST_CHECK_THROW(world->expectation_value(VecD{0.1}), std::exception);
    BOOST_CHECK_THROW(world->propagate({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, VecD{0.1}), GraphStateConflict);
    BOOST_TEST(!Access::is_invalid(*world));
    BOOST_TEST(world->expectation_value(params()) == before);
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
                                            Basis::Majorana);
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
                                                Basis::Majorana);
    core_only->build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0});
    BOOST_TEST(core_only->core_term() == 2.5);
    BOOST_TEST(core_only->expectation_value(VecD{0.3}) == 2.5);
    BOOST_TEST(core_only->expectation_value_and_gradient(VecD{0.3}).first == 2.5);
    BOOST_TEST(core_only->expectation_value_functional(1e-3)(VecD{0.3}) == 2.5);
    BOOST_TEST(core_only->evolved_operator_terms(VecD{0.3}, 0.0).empty());
}
