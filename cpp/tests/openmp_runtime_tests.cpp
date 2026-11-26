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

// Construction-time thread budgets and the invalid-owner rule. Budgets are captured from monoprop_NUM_THREADS when an
// object is constructed; the pure parser and capture cases set the variable around one call with ScopedEnv. The
// integration cases run at the launch's T (cpp/tests/CMakeLists.txt reruns them at T = 4).

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "PropagatorTestAccess.h"
#include "ShardedTestSupport.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/Validation.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/parallel/ThreadBudget.h"

namespace {

using namespace monoprop;
using monoprop::detail::parallel::capture_thread_budget;
using monoprop::detail::parallel::resolve_thread_budget;
using sharded_test::ScopedEnv;

constexpr size_t num_modes = 8;
constexpr unsigned int fixture_cutoff = 4;
constexpr const char *budget_variable = "monoprop_NUM_THREADS";

using Access = monoprop::detail::PropagatorTestAccess<num_modes>;
using Propagator = MonomialPropagator<num_modes>;

// A distinct type, so a test can tell the original exception from anything the library substitutes.
class InjectedFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The root at the launch's budget.
auto make_root(const test_utils::CaseData &data) -> Propagator {
    return Propagator(data.hamiltonian, fixture_cutoff, data.initial_state, std::nullopt, MPI_COMM_SELF);
}

auto load() -> test_utils::CaseData {
    return test_utils::load_case_data<num_modes>("random_exact.msgpack");
}

auto build_all(Propagator &sim, const test_utils::CaseData &data) -> void {
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
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
    check("update_cutoff", [&] { sim.update_cutoff(fixture_cutoff); });
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

BOOST_AUTO_TEST_CASE(openmp_budget_rejects_malformed_values) {
    const auto rejects = [](std::string_view text) {
        BOOST_TEST_CONTEXT("value=\"" << text << "\"") {
            BOOST_CHECK_EXCEPTION((void)resolve_thread_budget(text, 2),
                                  std::invalid_argument,
                                  [](const std::invalid_argument &e) {
                                      return std::string_view(e.what()).find(budget_variable) != std::string_view::npos;
                                  });
        }
    };
    for (const std::string_view text :
         {"", "0", "00", "-1", "-0", "abc", "2,3", " 3", "3 ", "+3", "\t3", "3\n", "0x10", "1e3", "3.0", "3a"}) {
        rejects(text);
    }
    // Overflow, and non-ASCII digits (Arabic-Indic three, U+0663), which are not decimal digits here.
    for (const std::string_view text :
         {"2147483648", "4294967297", "18446744073709551617", "99999999999999999999999", "\xd9\xa3"}) {
        rejects(text);
    }
}

BOOST_AUTO_TEST_CASE(openmp_capture_reads_the_environment) {
    {
        const ScopedEnv env(budget_variable, "5");
        BOOST_TEST(capture_thread_budget().threads == 5);
    }
    {
        const ScopedEnv env(budget_variable, std::nullopt);
        BOOST_TEST(capture_thread_budget().threads == omp_get_max_threads());
    }
    // Present but empty or malformed is an error, never the fallback.
    for (const char *bad : {"", "two"}) {
        const ScopedEnv env(budget_variable, bad);
        BOOST_CHECK_THROW((void)capture_thread_budget(), std::invalid_argument);
    }
}

// The root reads the variable once, at construction, and its T is its shard count: copies and clones never reread it,
// so even an unparseable value set afterwards reaches neither them nor the source, while a new construction does read
// it. No second budget is ever created in this process.
BOOST_AUTO_TEST_CASE(openmp_root_captures_the_launch_budget_once) {
    const auto data = load();
    auto sim = make_root(data);
    const int team = Access::options(sim).threads;
    BOOST_TEST(team == sharded_test::expected_team().value_or(omp_get_max_threads()));
    BOOST_TEST(Access::shards(sim).size() == static_cast<size_t>(team));
    build_all(sim, data);
    const double energy = sim.expectation_value(data.parameters);
    const ScopedEnv env(budget_variable, "not-a-budget");
    const Propagator copy(sim);
    BOOST_TEST(Access::options(copy).threads == team);
    BOOST_TEST(Access::shards(copy).size() == static_cast<size_t>(team));
    BOOST_TEST(Access::options(*Access::clone(sim)).threads == team);
    BOOST_TEST(sim.expectation_value(data.parameters) == energy);
    BOOST_TEST(Propagator(copy).expectation_value(data.parameters) == energy);
    BOOST_CHECK_THROW((void)make_root(data), PropagatorConfigError);
}

BOOST_AUTO_TEST_CASE(openmp_propagator_work_leaves_openmp_settings_unchanged) {
    const auto data = load();
    const auto before = runtime_settings();
    {
        auto sim = make_root(data);
        build_all(sim, data);
        (void)sim.expectation_value_and_gradient(data.parameters);
        const auto copy = sim;
        (void)sim.contract_partially(data.parameters, true);
    }
    BOOST_TEST((runtime_settings() == before));
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

    auto sim = make_root(data);
    sim.build_graph(first, first_map, first_gen);
    const VecD params(expected_num_params(sim.parameter_mapping()), 0.3);
    const double before = sim.expectation_value(params);
    Propagator early(sim);
    auto retained = sim.expectation_value_functional();

    Access::set_cutoff_fn(sim, [](const Monomial<num_modes> &) -> bool {
        throw InjectedFailure("injected cutoff failure");
    });
    // The original exception reaches a single-rank caller unchanged.
    BOOST_CHECK_THROW(sim.build_graph(rest, rest_map, rest_gen), InjectedFailure);
    BOOST_TEST(Access::is_invalid(sim));

    require_invalid(sim, data);
    BOOST_CHECK_THROW((void)retained(params), InvalidPropagatorError);
    // Immutable configuration stays readable.
    BOOST_TEST(sim.cutoff() == fixture_cutoff);
    BOOST_TEST(!sim.schrodinger());
    // An independent earlier copy is unaffected.
    BOOST_TEST(test_utils::near(early.expectation_value(params), before));
}
