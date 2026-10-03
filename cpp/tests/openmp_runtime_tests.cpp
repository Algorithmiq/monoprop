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

// Construction-time thread budgets, their per-object plumbing, and the invalid-owner rule.
//
// Budgets are captured from monoprop_NUM_THREADS when an object is constructed. Cases that need a specific
// launch environment are also registered as separate fresh-process CTest entries (see CMakeLists.txt). The
// pure parser and capture cases set the variable around one call with ScopedEnv. In the legacy build, the
// one-store prototype (explicit partitions=1) is the object under test, and ScopedEnv selects its budget too;
// in the sharded candidate build every propagator captures the budget, and its T is its shard count, so no
// candidate integration case changes the budget inside a process: those cases run at the launch's T. These
// tests check what reaches each object; kernel-level parallel execution is covered by openmp_kernel_tests.cpp.

#include <boost/test/unit_test.hpp>

#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif

#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "PropagatorTestAccess.h"
#include "TestUtilities.h"
#include "monoprop/Evolution.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/OperationFailure.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/parallel/Workshare.h"

namespace {

using namespace monoprop;
using monoprop::detail::parallel::capture_thread_budget;
using monoprop::detail::parallel::Options;
using monoprop::detail::parallel::resolve_thread_budget;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 4;
constexpr const char *kBudgetVariable = "monoprop_NUM_THREADS";

using Access = monoprop::detail::PropagatorTestAccess<kNumModes>;
using Propagator = MonomialPropagator<kNumModes>;

// Sets (or unsets, for nullopt) one environment variable for a scope, then restores it.
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

// A distinct type, so a test can tell the original exception from anything the library substitutes.
class InjectedFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
// The candidate root at the launch's budget.
auto make_prototype(const test_utils::CaseData &data) -> Propagator {
    return Propagator(data.hamiltonian, kCutoff, data.initial_state, std::nullopt, MPI_COMM_SELF);
}
#else
auto make_sim(const test_utils::CaseData &data,
              size_t partitions,
              std::optional<unsigned int> schrodinger = std::nullopt) -> Propagator {
    return Propagator(data.hamiltonian,
                      kCutoff,
                      data.initial_state,
                      schrodinger,
                      MPI_COMM_SELF,
                      std::nullopt,
                      std::nullopt,
                      CutoffType::Length,
                      std::nullopt,
                      kNumModes,
                      Basis::Majorana,
                      partitions);
}

// The one-store prototype: explicit partitions=1 on an ordinary communicator.
auto make_prototype(const test_utils::CaseData &data) -> Propagator {
    return make_sim(data, 1);
}
#endif

auto load() -> test_utils::CaseData {
    return test_utils::load_case_data<kNumModes>("random_exact.msgpack");
}

auto build_all(Propagator &sim, const test_utils::CaseData &data) -> void {
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
}

// What the launch environment should resolve to, independently of the library's capture.
auto expected_launch_budget() -> int {
    if (const char *expected = std::getenv("monoprop_TEST_EXPECT_BUDGET")) {
        return std::stoi(expected);
    }
    const char *configured = std::getenv(kBudgetVariable);
    return resolve_thread_budget(configured ? std::optional<std::string_view>(configured) : std::nullopt,
                                 omp_get_max_threads())
        .threads;
}

struct RuntimeSettings {
    int max_threads = 0;
    int dynamic = 0;
    int max_active_levels = 0;
    int thread_limit = 0;
    omp_sched_t schedule = omp_sched_static;
    int chunk = 0;
    omp_proc_bind_t proc_bind = omp_proc_bind_false;

    auto operator==(const RuntimeSettings &) const -> bool = default;
};

auto runtime_settings() -> RuntimeSettings {
    RuntimeSettings s;
    s.max_threads = omp_get_max_threads();
    s.dynamic = omp_get_dynamic();
    s.max_active_levels = omp_get_max_active_levels();
    s.thread_limit = omp_get_thread_limit();
    omp_get_schedule(&s.schedule, &s.chunk);
    s.proc_bind = omp_get_proc_bind();
    return s;
}

auto binding_disabled(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    boost::test_tools::assertion_result result(omp_get_proc_bind() == omp_proc_bind_false);
    result.message() << "OpenMP binding is enabled (OMP_PROC_BIND); run with OMP_PROC_BIND=false";
    return result;
}

// Every state-consuming entry an invalid owner must refuse, each run on its own.
template <typename Check>
auto for_each_state_entry(Propagator &sim, const test_utils::CaseData &data, Check check) -> void {
    check("build_graph", [&] { sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs); });
    check("propagate", [&] { sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters); });
    check("expectation_value", [&] { (void)sim.expectation_value(data.parameters); });
    check("expectation_value_and_gradient", [&] { (void)sim.expectation_value_and_gradient(data.parameters); });
    check("expectation_value_functional", [&] { (void)sim.expectation_value_functional(); });
    check("expectation_value_and_gradient_functional", [&] { (void)sim.expectation_value_and_gradient_functional(); });
    check("contract_partially", [&] { (void)sim.contract_partially(data.parameters, false); });
    check("evolved_operator_terms", [&] { (void)sim.evolved_operator_terms(data.parameters, 0.0); });
    check("update_initial_operator", [&] { sim.update_initial_operator(data.hamiltonian); });
    check("set_parameter_mapping", [&] { sim.set_parameter_mapping(sim.parameter_mapping()); });
    check("update_cutoff", [&] { sim.update_cutoff(kCutoff); });
    check("size", [&] { (void)sim.size(); });
    check("graph_size", [&] { (void)sim.graph_size(); });
    check("graph_data", [&] { (void)sim.graph_data(); });
    check("copy", [&] { const Propagator copy(sim); });
    check("clone", [&] { (void)Access::clone(sim); });
}

auto require_invalid(Propagator &sim, const test_utils::CaseData &data) -> void {
    for_each_state_entry(sim, data, [](const char *what, auto &&call) {
        BOOST_TEST_CONTEXT(what) {
            BOOST_CHECK_THROW(call(), InvalidPropagatorError);
        }
    });
}

} // namespace

// --- resolve_thread_budget: the pure parser --------------------------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_budget_absent_uses_runtime_default) {
    BOOST_TEST(resolve_thread_budget(std::nullopt, 1).threads == 1);
    BOOST_TEST(resolve_thread_budget(std::nullopt, 3).threads == 3);
}

BOOST_AUTO_TEST_CASE(openmp_budget_absent_rejects_a_nonpositive_runtime_default) {
    BOOST_CHECK_THROW((void)resolve_thread_budget(std::nullopt, 0), std::invalid_argument);
    BOOST_CHECK_THROW((void)resolve_thread_budget(std::nullopt, -1), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(openmp_budget_explicit_value_overrides_the_default) {
    BOOST_TEST(resolve_thread_budget("8", 2).threads == 8);
    // A valid explicit value wins even over an invalid fallback.
    BOOST_TEST(resolve_thread_budget("1", 0).threads == 1);
    BOOST_TEST(resolve_thread_budget("1", -1).threads == 1);
    // Leading zeros are harmless, however many.
    BOOST_TEST(resolve_thread_budget("0008", 2).threads == 8);
    BOOST_TEST(resolve_thread_budget("0000000000000000000000000000000000000003", 1).threads == 3);
}

BOOST_AUTO_TEST_CASE(openmp_budget_rejects_malformed_values) {
    for (const std::string_view text :
         {"", "0", "00", "-1", "-0", "abc", "2,3", " 3", "3 ", "+3", "\t3", "3\n", "0x10", "1e3", "3.0", "3a"}) {
        BOOST_TEST_CONTEXT("value=\"" << text << "\"") {
            BOOST_CHECK_EXCEPTION((void)resolve_thread_budget(text, 2),
                                  std::invalid_argument,
                                  [](const std::invalid_argument &e) {
                                      return std::string_view(e.what()).find(kBudgetVariable) != std::string_view::npos;
                                  });
        }
    }
    // Non-ASCII digits (Arabic-Indic three, U+0663) are not decimal digits here.
    BOOST_CHECK_THROW((void)resolve_thread_budget("\xd9\xa3", 2), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(openmp_budget_accepts_int_max_without_creating_a_team) {
    // Parsing only: no region is opened, so no INT_MAX-sized team is ever requested.
    BOOST_TEST(resolve_thread_budget("2147483647", 1).threads == INT_MAX);
    BOOST_TEST(resolve_thread_budget("0002147483647", 1).threads == INT_MAX);
}

BOOST_AUTO_TEST_CASE(openmp_budget_rejects_overflow) {
    for (const std::string_view text :
         {"2147483648", "4294967297", "18446744073709551617", "99999999999999999999999"}) {
        BOOST_TEST_CONTEXT("value=" << text) {
            BOOST_CHECK_THROW((void)resolve_thread_budget(text, 2), std::invalid_argument);
        }
    }
}

BOOST_AUTO_TEST_CASE(openmp_budget_has_no_arbitrary_thread_cap) {
    // The legacy parser silently ignored anything above one million.
    BOOST_TEST(resolve_thread_budget("1000001", 1).threads == 1000001);
}

// --- capture_thread_budget: one environment read per call ---------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_capture_reads_the_environment) {
    {
        const ScopedEnv env(kBudgetVariable, "5");
        BOOST_TEST(capture_thread_budget().threads == 5);
    }
    {
        const ScopedEnv env(kBudgetVariable, std::nullopt);
        BOOST_TEST(capture_thread_budget().threads == omp_get_max_threads());
    }
    {
        // Present but empty is an error, never the fallback.
        const ScopedEnv env(kBudgetVariable, "");
        BOOST_CHECK_THROW((void)capture_thread_budget(), std::invalid_argument);
    }
    {
        const ScopedEnv env(kBudgetVariable, "two");
        BOOST_CHECK_THROW((void)capture_thread_budget(), std::invalid_argument);
    }
}

// --- per-object capture ------------------------------------------------------------------------------------

// Registered again as fresh processes with monoprop_NUM_THREADS=1, =3, and unset with OMP_NUM_THREADS=3.
BOOST_AUTO_TEST_CASE(openmp_prototype_budget_matches_the_launch_environment) {
    const auto data = load();
    const auto sim = make_prototype(data);
    BOOST_TEST(Access::options(sim).threads == expected_launch_budget());
#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
    // The candidate's shard count is its budget.
    BOOST_TEST(Access::shards(sim).size() == static_cast<size_t>(expected_launch_budget()));
    // The unset-variable launch: nothing supplied the variable, so the OpenMP default decided.
    if (std::getenv("monoprop_TEST_EXPECT_UNSET_BUDGET") != nullptr) {
        BOOST_TEST(std::getenv(kBudgetVariable) == nullptr);
        BOOST_TEST(Access::options(sim).threads == omp_get_max_threads());
    }
#endif
}

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
// The candidate reads the variable once, at construction: copies and clones never reread it, so even an unparseable
// value set afterwards reaches neither them nor the source, while a new construction does read it. No second budget
// is ever created in this process.
BOOST_AUTO_TEST_CASE(openmp_prototype_captures_once_and_copies_preserve_it) {
    const auto data = load();
    auto sim = make_prototype(data);
    const int team = Access::options(sim).threads;
    build_all(sim, data);
    const double energy = sim.expectation_value(data.parameters);
    const ScopedEnv env(kBudgetVariable, "not-a-budget");
    const Propagator copy(sim);
    BOOST_TEST(Access::options(copy).threads == team);
    BOOST_TEST(Access::shards(copy).size() == static_cast<size_t>(team));
    BOOST_TEST(Access::options(*Access::clone(sim)).threads == team);
    BOOST_TEST(sim.expectation_value(data.parameters) == energy);
    BOOST_TEST(Propagator(copy).expectation_value(data.parameters) == energy);
    BOOST_CHECK_THROW((void)make_prototype(data), PropagatorConfigError);
}
#else
BOOST_AUTO_TEST_CASE(openmp_prototype_captures_once_and_copies_preserve_it) {
    const auto data = load();
    std::optional<Propagator> sim;
    {
        const ScopedEnv env(kBudgetVariable, "3");
        sim.emplace(make_prototype(data));
    }
    BOOST_TEST(Access::options(*sim).threads == 3);
    build_all(*sim, data);

    // A later environment change reaches new objects only.
    const ScopedEnv env(kBudgetVariable, "5");
    const Propagator copy(*sim);
    BOOST_TEST(Access::options(copy).threads == 3);
    BOOST_TEST(Access::options(*Access::clone(*sim)).threads == 3);
    BOOST_TEST(Access::options(make_prototype(data)).threads == 5);
    BOOST_TEST(Access::options(*sim).threads == 3);
}

// Legacy only: the candidate's retained functionals evaluate in the owner's team (sharded_root_owners_do_the_work_of_
// every_operation) and have no replaceable evaluation body.
BOOST_AUTO_TEST_CASE(openmp_retained_functionals_carry_the_owner_budget) {
    const auto data = load();
    std::optional<Propagator> sim;
    {
        const ScopedEnv env(kBudgetVariable, "3");
        sim.emplace(make_prototype(data));
    }
    build_all(*sim, data);
    auto copy = *sim;

    int seen = 0;
    const auto record =
        [&seen](const EvalRequest &request, mpi::Comm, const monoprop::detail::CosCallbacks &) -> double {
        seen = request.parallel.threads;
        return 0.0;
    };
    const ScopedEnv env(kBudgetVariable, "5");
    auto from_owner = Access::make_functional(*sim, record);
    auto from_copy = Access::make_functional(copy, record);
    (void)from_owner(data.parameters);
    BOOST_TEST(seen == 3);
    seen = 0;
    (void)from_copy(data.parameters);
    BOOST_TEST(seen == 3);
    // The public functionals evaluate through the same request and still agree with a serial object.
    auto serial = [&] {
        const ScopedEnv one(kBudgetVariable, "1");
        return make_prototype(data);
    }();
    build_all(serial, data);
    BOOST_TEST(test_utils::near(sim->expectation_value_functional()(data.parameters),
                                serial.expectation_value_functional()(data.parameters)));
}
#endif

BOOST_AUTO_TEST_CASE(openmp_invalid_budget_is_a_config_error_at_construction) {
    const auto data = load();
    for (const char *value : {"0", "abc", "", "-2", "2147483648"}) {
        const ScopedEnv env(kBudgetVariable, value);
        BOOST_TEST_CONTEXT("monoprop_NUM_THREADS=\"" << value << "\"") {
            BOOST_CHECK_EXCEPTION((void)make_prototype(data),
                                  PropagatorConfigError,
                                  [](const PropagatorConfigError &e) {
                                      return std::string_view(e.what()).find(kBudgetVariable) != std::string_view::npos;
                                  });
        }
    }
}

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Legacy only: the facade and automatic partition paths; the candidate has neither.
BOOST_AUTO_TEST_CASE(openmp_legacy_paths_keep_serial_options) {
    const auto data = load();
    const ScopedEnv env(kBudgetVariable, "3");
    // A two-partition facade and its Shm children (built with partitions=1 over a non-ordinary comm).
    const auto facade = make_sim(data, 2);
    BOOST_TEST(Access::options(facade).threads == 1);
    BOOST_REQUIRE(Access::partition_count(facade) == 2);
    for (int r = 0; r < 2; ++r) {
        BOOST_TEST(Access::options(Access::partition(facade, r)).threads == 1);
    }
    // Automatic partition selection keeps the legacy path whatever it resolves to.
    const auto automatic = make_sim(data, 0);
    BOOST_TEST(Access::options(automatic).threads == 1);
    for (int r = 0; r < Access::partition_count(automatic); ++r) {
        BOOST_TEST(Access::options(Access::partition(automatic, r)).threads == 1);
    }
}

// Registered again with OMP_THREAD_LIMIT=2 and monoprop_NUM_THREADS=3. Legacy only: a team limited below T is outside
// the candidate's supported launch, so no candidate operation runs under it (for_blocks keeps its own reduced-team
// contract in openmp_workshare_tests.cpp and openmp_env_kernels_*).
BOOST_AUTO_TEST_CASE(openmp_runtime_limited_team_keeps_the_captured_budget) {
    const auto data = load();
    const auto sim = make_prototype(data);
    const auto options = Access::options(sim);
    BOOST_TEST(options.threads == expected_launch_budget());

    constexpr size_t kBlocks = 64;
    std::vector<int> worker(kBlocks, -1);
    monoprop::detail::parallel::for_blocks(kBlocks, options, [&](size_t b) { worker[b] = omp_get_thread_num(); });
    const std::set<int> distinct(worker.begin(), worker.end());
    BOOST_TEST(!distinct.contains(-1));
    BOOST_TEST(distinct.size() <= static_cast<size_t>(std::min(options.threads, omp_get_thread_limit())));
    // A smaller actual team is not a budget change.
    BOOST_TEST(Access::options(sim).threads == options.threads);
}
#endif

BOOST_AUTO_TEST_CASE(openmp_propagator_work_leaves_openmp_settings_unchanged) {
    const auto data = load();
    const auto before = runtime_settings();
    {
#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
        const ScopedEnv env(kBudgetVariable, "3");
#endif
        auto sim = make_prototype(data);
        build_all(sim, data);
        (void)sim.expectation_value_and_gradient(data.parameters);
        const auto copy = sim;
        (void)sim.contract_partially(data.parameters, true);
    }
    BOOST_TEST((runtime_settings() == before));
}

// Registered again with OMP_PROC_BIND=false; skipped where the runtime was asked to bind.
BOOST_AUTO_TEST_CASE(openmp_propagator_work_does_not_change_affinity,
                     *boost::unit_test::precondition(binding_disabled)) {
#ifdef __linux__
    cpu_set_t before;
    CPU_ZERO(&before);
    BOOST_REQUIRE(sched_getaffinity(0, sizeof(before), &before) == 0);
    {
        const auto data = load();
#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
        const ScopedEnv env(kBudgetVariable, "3");
#endif
        auto sim = make_prototype(data);
        build_all(sim, data);
        (void)sim.expectation_value_and_gradient(data.parameters);
        monoprop::detail::parallel::for_blocks(8, Access::options(sim), [](size_t) {});
    }
    cpu_set_t after;
    CPU_ZERO(&after);
    BOOST_REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
    BOOST_TEST(CPU_EQUAL(&before, &after));
#else
    BOOST_TEST_MESSAGE("affinity masks are only compared on Linux");
#endif
}

// --- old low-level call sites keep compiling and default to serial Options ---------------------------------

BOOST_AUTO_TEST_CASE(openmp_low_level_calls_default_to_serial_options) {
    const auto data = load();
    auto sim = make_prototype(data);
    build_all(sim, data);
    const auto &graph = sim.graph();
    const auto view = graph.replay_view();
    const auto mapped = map_params(data.parameters, sim.parameter_mapping(), VecD(sim.graph_layers(), 1.0), 1.0, true);
    const auto cos = build_cos_callbacks<kNumModes>(sim.mp_op().inverted_index(), view, Basis::Majorana);
    const auto cos_threaded =
        build_cos_callbacks<kNumModes>(sim.mp_op().inverted_index(), view, Basis::Majorana, Options{.threads = 3});

    const VecD op = sim.mp_op().get_operator();
    const VecD by_default = evolve_operator(VecD(op), view, mapped, MPI_COMM_SELF, cos.scale);
    const VecD explicit_serial = evolve_operator(VecD(op), view, mapped, MPI_COMM_SELF, cos.scale, Options{});
    const VecD threaded =
        evolve_operator(VecD(op), view, mapped, MPI_COMM_SELF, cos_threaded.scale, Options{.threads = 3});
    BOOST_CHECK_EQUAL_COLLECTIONS(by_default.begin(), by_default.end(), explicit_serial.begin(), explicit_serial.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(by_default.begin(), by_default.end(), threaded.begin(), threaded.end());

    VecD s1(op.size(), 0.5), h1 = by_default, s2 = s1, h2 = h1;
    const size_t last = sim.graph_layers() - 1;
    const LayerAngle angle{.gen_coeff = 1.0, .param = mapped[last]};
    const double d1 = state_operator_derivative_local(s1, h1, view, last, angle, MPI_COMM_SELF, cos.accumulate);
    const double d2 = state_operator_derivative_local(s2,
                                                      h2,
                                                      view,
                                                      last,
                                                      angle,
                                                      MPI_COMM_SELF,
                                                      cos_threaded.accumulate,
                                                      {},
                                                      Options{.threads = 3});
    BOOST_TEST(d1 == d2);
    BOOST_CHECK_EQUAL_COLLECTIONS(h1.begin(), h1.end(), h2.begin(), h2.end());

    CosMask mask;
    mask.blocks.emplace_back(0, uint64_t{0b1011});
    mask.total_count = 3;
    VecD a(64, 2.0), b = a;
    monoprop::detail::scale_cos_mask(a.data(), mask, 0.5);
    monoprop::detail::scale_cos_mask(b.data(), mask, 0.5, Options{.threads = 3});
    BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());

    EvalState state = EvalState::dense(VecD(op.size(), 0.25));
    const auto mapping = sim.parameter_mapping();
    const VecD gens(mapping.size(), 1.0);
    const EvalRequest serial_request{.e_core = 0.0,
                                     .state = state,
                                     .op = op,
                                     .parameter_mapping = mapping,
                                     .gen_coeffs = gens,
                                     .graph = view,
                                     .params = data.parameters};
    BOOST_TEST(serial_request.parallel.threads == 1);
}

// --- invalid-owner enforcement ------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_failed_build_invalidates_the_owner) {
    const auto data = load();
    const size_t half = data.majoranas.size() / 2;
    BOOST_REQUIRE(half > 0);
    const std::vector<VecZ> first(data.majoranas.begin(), data.majoranas.begin() + static_cast<std::ptrdiff_t>(half));
    const std::vector<VecZ> rest(data.majoranas.begin() + static_cast<std::ptrdiff_t>(half), data.majoranas.end());
    const VecZ first_map(data.param_inds.begin(), data.param_inds.begin() + static_cast<std::ptrdiff_t>(half));
    const VecZ rest_map(data.param_inds.begin() + static_cast<std::ptrdiff_t>(half), data.param_inds.end());
    const VecD first_gen(data.gen_coeffs.begin(), data.gen_coeffs.begin() + static_cast<std::ptrdiff_t>(half));
    const VecD rest_gen(data.gen_coeffs.begin() + static_cast<std::ptrdiff_t>(half), data.gen_coeffs.end());

    auto sim = make_prototype(data);
    sim.build_graph(first, first_map, first_gen);
    const VecD params(expected_num_params(sim.parameter_mapping()), 0.3);
    const double before = sim.expectation_value(params);
    Propagator early(sim);
    auto retained = sim.expectation_value_functional();

    Access::set_cutoff_fn(sim, [](const Monomial<kNumModes> &) -> bool {
        throw InjectedFailure("injected cutoff failure");
    });
    // The original exception reaches a single-rank caller unchanged.
    BOOST_CHECK_THROW(sim.build_graph(rest, rest_map, rest_gen), InjectedFailure);
    BOOST_TEST(Access::is_invalid(sim));

    require_invalid(sim, data);
    BOOST_CHECK_THROW((void)retained(params), InvalidPropagatorError);
    // Immutable configuration stays readable.
    BOOST_TEST(sim.cutoff() == kCutoff);
    BOOST_TEST(!sim.schrodinger());
    // An independent earlier copy is unaffected.
    BOOST_TEST(test_utils::near(early.expectation_value(params), before));
}

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Legacy only (a replaceable evaluation body): the candidate's equivalents inject failures into its real evaluation
// phases (sharded_root_functional_failures_invalidate_their_owner,
// sharded_root_failures_after_mutation_invalidate_the_root).
BOOST_AUTO_TEST_CASE(openmp_failed_functional_evaluation_invalidates_the_owner) {
    const auto data = load();
    auto sim = make_prototype(data);
    build_all(sim, data);
    const double before = sim.expectation_value(data.parameters);
    Propagator early(sim);
    auto retained = sim.expectation_value_and_gradient_functional();

    auto failing =
        Access::make_functional(sim,
                                [](const EvalRequest &, mpi::Comm, const monoprop::detail::CosCallbacks &) -> double {
                                    throw InjectedFailure("injected evaluation failure");
                                });
    BOOST_CHECK_THROW((void)failing(data.parameters), InjectedFailure);
    BOOST_TEST(Access::is_invalid(sim));
    BOOST_CHECK_THROW((void)retained(data.parameters), InvalidPropagatorError);
    require_invalid(sim, data);
    BOOST_TEST(test_utils::near(early.expectation_value(data.parameters), before));
}

// A worker exception from the worksharing helper, inside a real evaluation, on one rank.
BOOST_AUTO_TEST_CASE(openmp_single_rank_worker_throw_keeps_the_original_exception) {
    const auto data = load();
    auto sim = make_prototype(data);
    build_all(sim, data);
    auto failing = Access::make_functional(
        sim,
        [](const EvalRequest &request, mpi::Comm, const monoprop::detail::CosCallbacks &) -> double {
            monoprop::detail::parallel::for_blocks(8, Options{.threads = 2}, [](size_t block) {
                if (block == 5) {
                    throw InjectedFailure("injected worker failure");
                }
            });
            return request.e_core;
        });
    BOOST_CHECK_THROW((void)failing(data.parameters), InjectedFailure);
    BOOST_TEST(Access::is_invalid(sim));
    BOOST_CHECK_THROW((void)sim.expectation_value(data.parameters), InvalidPropagatorError);
}
#endif

BOOST_AUTO_TEST_CASE(openmp_validation_errors_leave_the_owner_usable) {
    const auto data = load();
    auto sim = make_prototype(data);
    build_all(sim, data);
    const double before = sim.expectation_value(data.parameters);
    const auto check_usable = [&](const char *what) {
        BOOST_TEST_CONTEXT(what) {
            BOOST_TEST(!Access::is_invalid(sim));
            BOOST_TEST(test_utils::near(sim.expectation_value(data.parameters), before));
        }
    };

    BOOST_CHECK_THROW(sim.build_graph(data.majoranas, VecZ{0}, data.gen_coeffs), std::exception);
    check_usable("mismatched mapping length");

    // A generator index past the system: rejected before any gate runs, even behind a valid gate.
    std::vector<VecZ> bad_gates{data.majoranas.front(), VecZ{0, 2 * kNumModes}};
    BOOST_CHECK_THROW(sim.build_graph(bad_gates, VecZ{0, 0}, VecD{1.0, 1.0}), std::exception);
    check_usable("out-of-range generator index in build_graph");
    auto fresh = make_prototype(data);
    BOOST_CHECK_THROW(fresh.propagate(bad_gates, VecZ{0, 0}, VecD{1.0, 1.0}, VecD{0.1}), std::exception);
    BOOST_TEST(!Access::is_invalid(fresh));

    BOOST_CHECK_THROW(sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters), std::exception);
    check_usable("propagate on a stored graph");

    BOOST_CHECK_THROW((void)sim.expectation_value(VecD{0.1}), std::exception);
    check_usable("wrong parameter length");
    auto retained = sim.expectation_value_functional();
    BOOST_CHECK_THROW((void)retained(VecD{0.1}), std::exception);
    check_usable("wrong parameter length through a retained functional");

    sim.update_upper_atol(0.5);
    BOOST_CHECK_THROW(sim.update_lower_atol(1.0), PropagatorConfigError);
    sim.update_upper_atol(std::nullopt);
    check_usable("crossed atol pair");
    BOOST_CHECK_THROW(sim.set_parameter_mapping(VecZ{0}), std::exception);
    check_usable("wrong mapping length");

    OperatorDict bad_operator = data.hamiltonian;
    bad_operator[VecZ{0, 2 * kNumModes}] = 1.0;
    BOOST_CHECK_THROW(sim.update_initial_operator(bad_operator), std::exception);
    check_usable("out-of-range initial-operator index");
    BOOST_TEST(test_utils::near(retained(data.parameters), before));
}

BOOST_AUTO_TEST_CASE(openmp_invalid_owner_can_still_be_destroyed) {
    const auto data = load();
    auto owner = std::make_unique<Propagator>(make_prototype(data));
    build_all(*owner, data);
    Access::set_cutoff_fn(*owner, [](const Monomial<kNumModes> &) -> bool { throw InjectedFailure("injected"); });
    BOOST_CHECK_THROW(build_all(*owner, data), InjectedFailure);
    BOOST_CHECK_NO_THROW(owner.reset());
}

// --- failure boundary helper -----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(openmp_operation_failed_rethrows_the_original_on_one_rank) {
    const auto error = std::make_exception_ptr(InjectedFailure("original"));
    BOOST_CHECK_EXCEPTION(mpi::operation_failed(MPI_COMM_SELF, error), InjectedFailure, [](const InjectedFailure &e) {
        return std::string_view(e.what()) == "original";
    });
    int calls = 0;
    BOOST_CHECK_THROW(mpi::guard_distributed(MPI_COMM_SELF,
                                             [&] {
                                                 ++calls;
                                                 throw InjectedFailure("guarded");
                                             }),
                      InjectedFailure);
    BOOST_TEST(calls == 1);
    BOOST_TEST(mpi::guard_distributed(MPI_COMM_SELF, [] { return 7; }) == 7);
}

#ifdef monoprop_ENABLE_MPI
BOOST_AUTO_TEST_CASE(openmp_thread_level_decisions_are_pure) {
    const int levels[] = {MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE};
    for (const int provided : levels) {
        BOOST_TEST_CONTEXT("provided=" << mpi::thread_level_name(provided)) {
            BOOST_TEST(mpi::thread_level_satisfies(provided, MPI_THREAD_SERIALIZED)
                       == (provided >= MPI_THREAD_SERIALIZED));
            BOOST_TEST(mpi::thread_level_satisfies(provided, MPI_THREAD_FUNNELED) == (provided >= MPI_THREAD_FUNNELED));
        }
    }
    // Coexistence keeps SERIALIZED as the requested and required level until Hybrid is removed.
    BOOST_TEST(mpi::kRequiredThreadLevel == MPI_THREAD_SERIALIZED);
    BOOST_TEST(std::string(mpi::thread_level_name(MPI_THREAD_FUNNELED)) == "MPI_THREAD_FUNNELED");
}

BOOST_AUTO_TEST_CASE(openmp_initialized_mpi_meets_the_required_level) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    BOOST_TEST_MESSAGE("MPI provided thread support: " << mpi::thread_level_name(provided));
    BOOST_TEST(mpi::thread_level_satisfies(provided, mpi::kRequiredThreadLevel));
    BOOST_CHECK_NO_THROW(mpi::require_thread_support());
    int is_main = 0;
    MPI_Is_thread_main(&is_main);
    BOOST_TEST(is_main != 0);
    // The initializing thread passes the guard; a wrong thread is covered by mpi_failure_driver.
    mpi::require_initializing_thread();
}
#endif
