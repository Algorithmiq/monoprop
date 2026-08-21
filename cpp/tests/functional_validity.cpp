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

// What each public mutator does to a functional that was built before it ran. The table is the
// contract: one row per mutating method, asserted for the value and the gradient functional, with
// and without a pare threshold, in both pictures.

#include <boost/test/unit_test.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/functional/Control.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;
namespace tt = boost::test_tools;

namespace {

constexpr size_t kNumModes = 2;
using Prop = MonomialPropagator<kNumModes>;

// Small enough to reason about by hand, large enough that both gates rotate something.
constexpr double kPareThreshold = 1e-12;
const VecD kBaseParams{0.3, 0.7};
// One gate per Hamiltonian term, and the terms carry different weights, so the answer is not
// symmetric under swapping the two angles -- which is what makes the set_parameter_mapping row bite.
const std::vector<VecZ> kBaseGates{VecZ{0}, VecZ{2}};

// The weight mutate_update_initial_operator() writes onto term (0, 1). A re-weighted propagator must
// answer exactly like one built with it from the start, so both sides read it from here.
constexpr double kReweightedFirstWeight = 2.75;

// The identity weight the core-term cases start from; every other case leaves the row out entirely,
// which is what makes a re-weight that also leaves it out a no-op there.
constexpr double kCoreTerm = 0.25;

// `partitions` is passed explicitly, so it wins over the suite-wide monoprop_PARTITIONS=off.
// `core_term` adds the identity row, which only the core-term cases below need.
auto make_propagator(bool schrodinger,
                     size_t partitions = 1,
                     double first_weight = 1.0,
                     std::optional<double> core_term = std::nullopt) -> Prop {
    OperatorDict initial_ham;
    initial_ham[VecZ{0, 1}] = std::complex<double>{0.0, first_weight};
    initial_ham[VecZ{2, 3}] = std::complex<double>{0.0, 0.5};
    if (core_term.has_value()) {
        initial_ham[VecZ{}] = std::complex<double>{*core_term, 0.0};
    }
    const auto cutoff = static_cast<unsigned int>(2 * kNumModes);
    return Prop(initial_ham,
                cutoff,
                VecZ{0, 1},
                schrodinger ? std::optional<unsigned int>{cutoff} : std::nullopt,
                MPI_COMM_SELF,
                std::nullopt,
                std::nullopt,
                CutoffType::Support,
                std::nullopt,
                kNumModes,
                Basis::Majorana,
                partitions);
}

auto build_base_graph(Prop &prop) -> void {
    prop.build_graph(kBaseGates, VecZ{0, 1}, VecD{1.0, 1.0});
}

// The mutators, one per public mutating method. Each runs against a propagator that already carries
// the two-layer base graph, except propagate(), which refuses a non-empty graph (see
// propagate_on_non_empty_graph_leaves_functional_valid).

auto mutate_build_graph(Prop &prop) -> void {
    prop.build_graph({VecZ{1}}, VecZ{0}, VecD{1.0});
}

auto mutate_propagate(Prop &prop) -> void {
    prop.propagate({VecZ{0}}, VecZ{0}, VecD{1.0}, VecD{0.4});
}

auto mutate_contract_partially(Prop &prop) -> void {
    prop.contract_partially(kBaseParams, /*inplace=*/true);
}

auto mutate_update_initial_operator(Prop &prop) -> void {
    OperatorDict updated;
    updated[VecZ{0, 1}] = std::complex<double>{0.0, kReweightedFirstWeight};
    updated[VecZ{2, 3}] = std::complex<double>{0.0, 0.5};
    prop.update_initial_operator(updated);
}

// Swaps which parameter drives which layer, so the answer moves whenever the two angles differ.
auto mutate_set_parameter_mapping(Prop &prop) -> void {
    prop.set_parameter_mapping(VecZ{1, 0});
}

auto mutate_update_cutoff(Prop &prop) -> void {
    prop.update_cutoff(2);
}

auto mutate_update_cutoff_type(Prop &prop) -> void {
    prop.update_cutoff_type(CutoffType::Length);
}

auto mutate_update_basis_change(Prop &prop) -> void {
    prop.update_basis_change(std::vector<VecZ>{VecZ{0}, VecZ{1}, VecZ{0, 1, 2}, VecZ{0, 1, 3}});
}

auto mutate_update_lower_atol(Prop &prop) -> void {
    prop.update_lower_atol(1e-12);
}

auto mutate_update_upper_atol(Prop &prop) -> void {
    prop.update_upper_atol(1e-3);
}

// What calling the pre-built functional does after the row's mutator ran.
enum class Outcome : std::uint8_t {
    Stale,          // throws, reporting the propagator moved under the functional
    Answers,        // returns, and returns exactly what it returned before the mutation
    Refreshes,      // returns, and now returns what a functional built after the mutation returns
    RefusesRefresh, // throws, reporting weights it cannot follow rather than a moved structure
};

struct MutatorRow {
    std::string_view method; // the public method this row covers
    void (*apply)(Prop &);
    bool needs_empty_graph; // build the functional with no graph, so the mutator is accepted
    // Paring only changes the verdict where the keep-set came from the operator coefficients, so the
    // Schrodinger-pared column is the only one that can differ from the general one.
    Outcome outcome;           // no pare_threshold in either picture, or pared Heisenberg
    Outcome pared_schrodinger; // pare_threshold == kPareThreshold, Schrodinger: pares from `op`
    std::string_view rationale;
};

constexpr std::array kMutatorTable{
    MutatorRow{.method = "build_graph",
               .apply = &mutate_build_graph,
               .needs_empty_graph = false,
               .outcome = Outcome::Stale,
               .pared_schrodinger = Outcome::Stale,
               .rationale = "Appending a layer moves the structure revision, which a pared plan reads as "
                            "readily as an exact one."},
    MutatorRow{.method = "propagate",
               .apply = &mutate_propagate,
               .needs_empty_graph = true,
               .outcome = Outcome::Stale,
               .pared_schrodinger = Outcome::Stale,
               .rationale = "Re-evolves the operator in place. It leaves the layer count at zero, so the "
                            "revision is the only thing that sees it."},
    MutatorRow{.method = "contract_partially",
               .apply = &mutate_contract_partially,
               .needs_empty_graph = false,
               .outcome = Outcome::Stale,
               .pared_schrodinger = Outcome::Stale,
               .rationale = "Consumes the folded layers and rewrites the coefficients. Only inplace=true "
                            "bumps; see contract_partially_out_of_place_keeps_functional_valid."},
    MutatorRow{.method = "update_initial_operator",
               .apply = &mutate_update_initial_operator,
               .needs_empty_graph = false,
               .outcome = Outcome::Refreshes,
               .pared_schrodinger = Outcome::RefusesRefresh,
               .rationale = "A re-weight moves no structure, so the functional follows the new "
                            "coefficients -- unless its keep-set was thresholded from those very "
                            "coefficients, which is Schrodinger with a pare threshold."},
    MutatorRow{.method = "set_parameter_mapping",
               .apply = &mutate_set_parameter_mapping,
               .needs_empty_graph = false,
               .outcome = Outcome::Stale,
               .pared_schrodinger = Outcome::Stale,
               .rationale = "Relabels the layers in place, which changes neither the layer count nor the "
                            "operator -- the revision is the only thing that sees it."},
    MutatorRow{.method = "update_cutoff",
               .apply = &mutate_update_cutoff,
               .needs_empty_graph = false,
               .outcome = Outcome::Answers,
               .pared_schrodinger = Outcome::Answers,
               .rationale = "Intended: a cutoff gates the next build and changes nothing the plan holds."},
    MutatorRow{.method = "update_cutoff_type",
               .apply = &mutate_update_cutoff_type,
               .needs_empty_graph = false,
               .outcome = Outcome::Answers,
               .pared_schrodinger = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
    MutatorRow{.method = "update_basis_change",
               .apply = &mutate_update_basis_change,
               .needs_empty_graph = false,
               .outcome = Outcome::Answers,
               .pared_schrodinger = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
    MutatorRow{.method = "update_lower_atol",
               .apply = &mutate_update_lower_atol,
               .needs_empty_graph = false,
               .outcome = Outcome::Answers,
               .pared_schrodinger = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
    MutatorRow{.method = "update_upper_atol",
               .apply = &mutate_update_upper_atol,
               .needs_empty_graph = false,
               .outcome = Outcome::Answers,
               .pared_schrodinger = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
};

// A mutating method with no row would leave its effect on a live functional unrecorded, so the count
// is pinned to the roster on the class. Adding a mutator means bumping that constant, which breaks
// this build until a row lands here.
constexpr auto distinct_methods() -> size_t {
    size_t distinct = 0;
    for (size_t i = 0; i < kMutatorTable.size(); ++i) {
        bool seen_earlier = false;
        for (size_t j = 0; j < i; ++j) {
            seen_earlier = seen_earlier || kMutatorTable[j].method == kMutatorTable[i].method;
        }
        distinct += seen_earlier ? 0U : 1U;
    }
    return distinct;
}

static_assert(kMutatorTable.size() == Prop::num_mutating_methods && distinct_methods() == kMutatorTable.size(),
              "cpp/tests/functional_validity.cpp must carry one row per public mutating method of "
              "MonomialPropagator; see num_mutating_methods in MonomialPropagator.h.");

// Both functional kinds behind one signature; the gradient kind is judged on its value component.
auto make_call(Prop &prop, bool gradient, std::optional<double> pare_threshold) -> std::function<double(const VecD &)> {
    if (gradient) {
        auto fn = prop.expectation_value_and_gradient_functional(pare_threshold);
        return [fn = std::move(fn)](const VecD &params) { return fn(params).first; };
    }
    auto fn = prop.expectation_value_functional(pare_threshold);
    return [fn = std::move(fn)](const VecD &params) { return fn(params); };
}

auto expected_outcome(const MutatorRow &row, bool schrodinger, std::optional<double> pare_threshold) -> Outcome {
    return pare_threshold.has_value() && schrodinger ? row.pared_schrodinger : row.outcome;
}

auto run_row(const MutatorRow &row, bool schrodinger, bool gradient, std::optional<double> pare_threshold) -> void {
    auto prop = make_propagator(schrodinger);
    if (!row.needs_empty_graph) {
        build_base_graph(prop);
    }
    const VecD params = row.needs_empty_graph ? VecD{} : kBaseParams;

    auto call = make_call(prop, gradient, pare_threshold);
    const double before = call(params);

    row.apply(prop);

    switch (expected_outcome(row, schrodinger, pare_threshold)) {
        case Outcome::Stale:
            BOOST_CHECK_EXCEPTION(call(params), std::runtime_error, [](const std::runtime_error &e) {
                BOOST_TEST_INFO("message: " << e.what());
                return std::string_view(e.what()).find("MP object has been modified") != std::string_view::npos;
            });
            return;
        case Outcome::RefusesRefresh:
            BOOST_CHECK_EXCEPTION(call(params), std::runtime_error, [](const std::runtime_error &e) {
                BOOST_TEST_INFO("message: " << e.what());
                return std::string_view(e.what()).find("cannot follow the new weights") != std::string_view::npos;
            });
            return;
        case Outcome::Answers: {
            // The plan replays its own snapshot, so the number cannot have moved.
            double after = 0.0;
            BOOST_REQUIRE_NO_THROW(after = call(params));
            BOOST_TEST(after == before, tt::tolerance(1e-12));
            return;
        }
        case Outcome::Refreshes: {
            // The plan reads the propagator's live weights, so it must now agree exactly with a functional
            // built after the mutation -- and disagree with what it answered before it.
            double after = 0.0;
            BOOST_REQUIRE_NO_THROW(after = call(params));
            BOOST_TEST(after == make_call(prop, gradient, pare_threshold)(params));
            BOOST_TEST(after != before);
            return;
        }
    }
}

auto run_table(bool schrodinger, std::optional<double> pare_threshold) -> void {
    for (const auto &row : kMutatorTable) {
        for (const bool gradient : {false, true}) {
            BOOST_TEST_CONTEXT("method=" << row.method << " gradient=" << gradient << " rationale=" << row.rationale) {
                run_row(row, schrodinger, gradient, pare_threshold);
            }
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(functional_validity_table_heisenberg_exact) {
    run_table(/*schrodinger=*/false, std::nullopt);
}

BOOST_AUTO_TEST_CASE(functional_validity_table_heisenberg_pared) {
    run_table(/*schrodinger=*/false, kPareThreshold);
}

BOOST_AUTO_TEST_CASE(functional_validity_table_schrodinger_exact) {
    run_table(/*schrodinger=*/true, std::nullopt);
}

BOOST_AUTO_TEST_CASE(functional_validity_table_schrodinger_pared) {
    run_table(/*schrodinger=*/true, kPareThreshold);
}

// contract_partially(inplace=false) mutates nothing, so it belongs outside the table: it must never
// invalidate a functional.
BOOST_AUTO_TEST_CASE(contract_partially_out_of_place_keeps_functional_valid) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, std::nullopt);
    const double before = call(kBaseParams);

    prop.contract_partially(kBaseParams, /*inplace=*/false);

    BOOST_TEST(call(kBaseParams) == before, tt::tolerance(1e-12));
}

// propagate() refuses to run on top of a stored graph, so it cannot invalidate a functional built
// against one: the rejection leaves the propagator untouched.
BOOST_AUTO_TEST_CASE(propagate_on_non_empty_graph_leaves_functional_valid) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, std::nullopt);
    const double before = call(kBaseParams);

    BOOST_CHECK_THROW(mutate_propagate(prop), std::runtime_error);

    BOOST_TEST(call(kBaseParams) == before, tt::tolerance(1e-12));
}

// Two of the table's Stale rows, spelled out: the propagator's own answer moves, and the functional
// refuses the call rather than following it half way.

BOOST_AUTO_TEST_CASE(set_parameter_mapping_invalidates_functional_it_desynchronises) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, std::nullopt);
    const double before = call(kBaseParams);

    mutate_set_parameter_mapping(prop);

    BOOST_TEST(prop.expectation_value(kBaseParams) != before);
    BOOST_CHECK_THROW(call(kBaseParams), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(pared_functional_is_invalidated_by_build_graph) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, kPareThreshold);
    const double before = call(kBaseParams);

    mutate_build_graph(prop);

    BOOST_TEST(prop.expectation_value(kBaseParams) != before);
    BOOST_CHECK_THROW(call(kBaseParams), std::runtime_error);
}

// A functional reads its propagator's operator index directly, so it cannot outlive it. The control
// block records the destruction, which is the only thing left to read once the handles are dangling.
BOOST_AUTO_TEST_CASE(functional_reports_a_destroyed_propagator) {
    auto prop = std::make_unique<Prop>(make_propagator(/*schrodinger=*/false));
    build_base_graph(*prop);
    auto call = make_call(*prop, /*gradient=*/false, std::nullopt);
    BOOST_CHECK_NO_THROW(call(kBaseParams));

    prop.reset();

    BOOST_CHECK_EXCEPTION(call(kBaseParams), std::runtime_error, [](const std::runtime_error &e) {
        return std::string_view(e.what()).find("has been destroyed") != std::string_view::npos;
    });
}

// The functional objects report the axis they were built against, so a caller can size its parameter
// vector without going back to the propagator.
BOOST_AUTO_TEST_CASE(functional_reports_its_parameter_axis) {
    auto prop = make_propagator(/*schrodinger=*/false);
    BOOST_TEST(prop.expectation_value_functional().num_params() == 0U);

    build_base_graph(prop);
    BOOST_TEST(prop.expectation_value_functional().num_params() == kBaseParams.size());
    BOOST_TEST(prop.expectation_value_and_gradient_functional().num_params() == kBaseParams.size());
    BOOST_TEST(prop.expectation_value_functional(kPareThreshold).num_params() == kBaseParams.size());
}

// The contract read off the object, for callers that hold a functional and not the propagator: only a
// Schrodinger plan pared against the operator's own coefficients refuses to follow a re-weight.
BOOST_AUTO_TEST_CASE(functional_reports_whether_it_follows_the_weights) {
    for (const bool schrodinger : {false, true}) {
        auto prop = make_propagator(schrodinger);
        build_base_graph(prop);
        BOOST_TEST(prop.expectation_value_functional().follows_weights());
        BOOST_TEST(prop.expectation_value_and_gradient_functional().follows_weights());
        BOOST_TEST(prop.expectation_value_functional(kPareThreshold).follows_weights() == !schrodinger);
    }
    // A facade answers for its partitions, which were all built the same way.
    auto facade = make_propagator(/*schrodinger=*/true, /*partitions=*/2);
    build_base_graph(facade);
    BOOST_TEST(facade.expectation_value_functional().follows_weights());
    BOOST_TEST(!facade.expectation_value_functional(kPareThreshold).follows_weights());
}

// A facade's plan holds the partition group by raw pointer, so it needs the facade's own control block:
// the children cannot report a destruction that takes their group with it.
BOOST_AUTO_TEST_CASE(fanned_out_functional_reports_a_destroyed_facade) {
    auto facade = std::make_unique<Prop>(make_propagator(/*schrodinger=*/false, /*partitions=*/2));
    build_base_graph(*facade);
    auto call = make_call(*facade, /*gradient=*/false, std::nullopt);
    BOOST_CHECK_NO_THROW(call(kBaseParams));

    facade.reset();

    BOOST_CHECK_EXCEPTION(call(kBaseParams), std::runtime_error, [](const std::runtime_error &e) {
        return std::string_view(e.what()).find("has been destroyed") != std::string_view::npos;
    });
}

// The facade bumps its own revision as it fans a mutation out, so the error is reported on the calling
// thread rather than surfacing out of a partition's collective.
BOOST_AUTO_TEST_CASE(fanned_out_functional_is_invalidated_by_build_graph) {
    auto facade = make_propagator(/*schrodinger=*/false, /*partitions=*/2);
    build_base_graph(facade);
    auto call = make_call(facade, /*gradient=*/false, std::nullopt);
    BOOST_CHECK_NO_THROW(call(kBaseParams));

    mutate_build_graph(facade);

    BOOST_CHECK_EXCEPTION(call(kBaseParams), std::runtime_error, [](const std::runtime_error &e) {
        return std::string_view(e.what()).find("build_graph()") != std::string_view::npos;
    });
}

// The refresh, end to end and at full precision: a re-weighted propagator's functional must answer what
// a propagator built with those coefficients answers, to the last bit. The two run the same arithmetic
// over the same store order, so anything less than equality means the refresh reached a different
// vector -- there is no rounding to hide behind here.
namespace {

auto check_refresh_matches_fresh_propagator(bool gradient, std::optional<double> pare_threshold, size_t partitions)
    -> void {
    auto reweighted = make_propagator(/*schrodinger=*/false, partitions);
    build_base_graph(reweighted);
    auto call = make_call(reweighted, gradient, pare_threshold);
    const double before = call(kBaseParams);
    mutate_update_initial_operator(reweighted);

    auto fresh = make_propagator(/*schrodinger=*/false, partitions, kReweightedFirstWeight);
    build_base_graph(fresh);

    BOOST_TEST(call(kBaseParams) == make_call(fresh, gradient, pare_threshold)(kBaseParams));
    BOOST_TEST(call(kBaseParams) != before);
}

} // namespace

BOOST_AUTO_TEST_CASE(reweighted_functional_matches_a_fresh_propagator) {
    for (const bool gradient : {false, true}) {
        for (const auto pare_threshold : {std::optional<double>{}, std::optional<double>{kPareThreshold}}) {
            for (const size_t partitions : {1U, 2U}) {
                BOOST_TEST_CONTEXT("gradient=" << gradient << " pared=" << pare_threshold.has_value()
                                               << " partitions=" << partitions) {
                    check_refresh_matches_fresh_propagator(gradient, pare_threshold, partitions);
                }
            }
        }
    }
}

// The gradient follows the weights too, component by component: a re-weight scales what the backward
// pass carries, so a value-only check would pass on a gradient that stayed behind.
BOOST_AUTO_TEST_CASE(reweighted_gradient_matches_a_fresh_propagator) {
    auto reweighted = make_propagator(/*schrodinger=*/false);
    build_base_graph(reweighted);
    auto fn = reweighted.expectation_value_and_gradient_functional(std::nullopt);
    const auto before = fn(kBaseParams);
    mutate_update_initial_operator(reweighted);

    auto fresh = make_propagator(/*schrodinger=*/false, /*partitions=*/1, kReweightedFirstWeight);
    build_base_graph(fresh);
    const auto expected = fresh.expectation_value_and_gradient_functional(std::nullopt)(kBaseParams);

    const auto after = fn(kBaseParams);
    BOOST_TEST(after.second == expected.second, tt::per_element());
    BOOST_CHECK(after.second != before.second);
}

// Invariant 4: Schrodinger thresholds its keep-set from the operator coefficients, so new coefficients
// select a different keep-set and the plan cannot replay this one. It says so rather than answering for a
// paring nobody asked for. Heisenberg thresholds the state, which a re-weight leaves alone.
BOOST_AUTO_TEST_CASE(pared_schrodinger_functional_refuses_to_follow_a_reweight) {
    auto prop = make_propagator(/*schrodinger=*/true);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, kPareThreshold);
    BOOST_CHECK_NO_THROW(call(kBaseParams));

    mutate_update_initial_operator(prop);

    BOOST_CHECK_EXCEPTION(call(kBaseParams), std::runtime_error, [](const std::runtime_error &e) {
        const std::string_view what(e.what());
        BOOST_TEST_INFO("message: " << what);
        return what.find("pares its graph") != std::string_view::npos
               && what.find("cannot follow the new weights") != std::string_view::npos;
    });
    // The unpared plan over the same propagator follows the re-weight, so the refusal is the paring's and
    // not the picture's.
    BOOST_CHECK_NO_THROW(make_call(prop, /*gradient=*/false, std::nullopt)(kBaseParams));
}

// A rejected re-weight is not a refresh: MPOperator::update_initial_operator throws for a term the
// operator does not hold, and core_term_ may already carry the new value by then. A functional must
// report that rather than answer from weights the propagator disagrees with.
BOOST_AUTO_TEST_CASE(a_failed_reweight_invalidates_the_functional) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, std::nullopt);
    BOOST_CHECK_NO_THROW(call(kBaseParams));

    OperatorDict unknown_term;
    unknown_term[VecZ{0, 2}] = std::complex<double>{0.0, 1.0};
    BOOST_CHECK_THROW(prop.update_initial_operator(unknown_term), std::runtime_error);

    BOOST_CHECK_EXCEPTION(call(kBaseParams), std::runtime_error, [](const std::runtime_error &e) {
        return std::string_view(e.what()).find("update_initial_operator()") != std::string_view::npos;
    });
}

// Building a second functional must not look like a re-weight to the first: both are built over the one
// published weight set, so the pared Schrodinger plan -- the only one that refuses to follow a new set --
// keeps answering.
BOOST_AUTO_TEST_CASE(building_another_functional_is_not_a_reweight) {
    auto prop = make_propagator(/*schrodinger=*/true);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, kPareThreshold);
    const double before = call(kBaseParams);

    auto other = make_call(prop, /*gradient=*/true, std::nullopt);
    BOOST_CHECK_NO_THROW(other(kBaseParams));

    BOOST_TEST(call(kBaseParams) == before);
}

// The scope guard behind the failure-path rule, over a bare control block: a scope that returns
// normally leaves the revision alone, and one that throws bumps it once, recording its site.
BOOST_AUTO_TEST_CASE(bump_on_unwind_bumps_only_when_the_scope_throws) {
    detail::FunctionalControl control;

    {
        auto guard = detail::BumpOnUnwind(control, "site()");
    }
    BOOST_TEST(control.structure_revision.load() == 0U);
    BOOST_TEST((control.last_structural_change.load() == nullptr));

    BOOST_CHECK_THROW(
        [&] {
            auto guard = detail::BumpOnUnwind(control, "site()");
            throw std::runtime_error("part-way");
        }(),
        std::runtime_error);
    BOOST_TEST(control.structure_revision.load() == 1U);
    BOOST_TEST(std::string_view(control.last_structural_change.load()) == std::string_view("site()"));

    // Disarmed is how a known no-op opts out, so it must stay silent on both paths.
    BOOST_CHECK_THROW(
        [&] {
            auto guard = detail::BumpOnUnwind(control, "other()", /*armed=*/false);
            throw std::runtime_error("part-way");
        }(),
        std::runtime_error);
    BOOST_TEST(control.structure_revision.load() == 1U);
}

// A gate generator is bounds-checked only as the gate loop reaches it, so a bad one in a multi-gate
// call throws with the earlier gates already committed: valid, out of range, valid appends before it
// throws in both pictures (Heisenberg walks the gates in reverse).
namespace {

const std::vector<VecZ> kPartWayGates{VecZ{0}, VecZ{2 * kNumModes + 1}, VecZ{2}};
const VecZ kPartWayMapping{0, 1, 2};
const VecD kPartWayCoeffs{1.0, 1.0, 1.0};

auto reports_site(std::string_view site) {
    return [site](const std::runtime_error &e) {
        const std::string_view what(e.what());
        BOOST_TEST_INFO("message: " << what);
        return what.find(site) != std::string_view::npos;
    };
}

} // namespace

BOOST_AUTO_TEST_CASE(a_part_way_build_graph_failure_invalidates_the_functional) {
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto prop = make_propagator(schrodinger);
            build_base_graph(prop);
            auto call = make_call(prop, /*gradient=*/false, std::nullopt);
            BOOST_CHECK_NO_THROW(call(kBaseParams));
            const size_t layers_before = prop.graph_layers();

            BOOST_CHECK_THROW(prop.build_graph(kPartWayGates, kPartWayMapping, kPartWayCoeffs), std::runtime_error);

            // The graph grew, so the call did mutate on its way to the throw.
            BOOST_TEST(prop.graph_layers() > layers_before);
            BOOST_CHECK_EXCEPTION(call(kBaseParams), std::runtime_error, reports_site("build_graph()"));
        }
    }
}

// propagate() folds into the operator instead of appending, so nothing counts the mutation: without
// the failure-path bump the functional would keep answering for coefficients that had moved.
BOOST_AUTO_TEST_CASE(a_part_way_propagate_failure_invalidates_the_functional) {
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto prop = make_propagator(schrodinger);
            auto call = make_call(prop, /*gradient=*/false, std::nullopt);
            BOOST_CHECK_NO_THROW(call(VecD{}));

            BOOST_CHECK_THROW(prop.propagate(kPartWayGates, kPartWayMapping, kPartWayCoeffs, VecD{0.3, 0.7, 0.5}),
                              std::runtime_error);

            BOOST_CHECK_EXCEPTION(call(VecD{}), std::runtime_error, reports_site("propagate()"));
        }
    }
}

// A call that appends or folds nothing is not a mutation. The facade decides that before it fans out,
// so both partition counts must agree -- this is the off/auto divergence the table exists to rule out.
BOOST_AUTO_TEST_CASE(no_op_mutators_keep_a_functional_valid) {
    for (const size_t partitions : {1U, 2U}) {
        BOOST_TEST_CONTEXT("partitions=" << partitions) {
            auto prop = make_propagator(/*schrodinger=*/false, partitions);
            auto call = make_call(prop, /*gradient=*/false, std::nullopt);
            const double before = call(VecD{});

            // No graph, so there is nothing to fold and no layer to retire; the return is the current
            // coefficients either way, which is what it was before the call.
            const VecD folded = prop.contract_partially(VecD{}, /*inplace=*/true);
            BOOST_TEST(folded == prop.contract_partially(VecD{}, /*inplace=*/false), tt::per_element());
            prop.build_graph({}, VecZ{}, VecD{});
            prop.propagate({}, VecZ{}, VecD{}, VecD{});
            BOOST_TEST(prop.graph_layers() == 0U);

            BOOST_TEST(call(VecD{}) == before, tt::tolerance(1e-12));
        }
    }
}

// The core term describes the dict that committed, like every other row: a re-weight that leaves the
// identity out zeroes it, which is what MPOperator::update_initial_operator does with the rows the dict
// omits. The live functional and the propagator's own answer must both see that.
BOOST_AUTO_TEST_CASE(a_reweight_that_drops_the_core_term_zeroes_it) {
    auto prop = make_propagator(/*schrodinger=*/false, /*partitions=*/1, /*first_weight=*/1.0, kCoreTerm);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, std::nullopt);
    BOOST_TEST(prop.core_term() == kCoreTerm);

    mutate_update_initial_operator(prop); // carries no identity row

    BOOST_TEST(prop.core_term() == 0.0);
    auto fresh = make_propagator(/*schrodinger=*/false, /*partitions=*/1, kReweightedFirstWeight);
    build_base_graph(fresh);
    BOOST_TEST(call(kBaseParams) == make_call(fresh, /*gradient=*/false, std::nullopt)(kBaseParams));
    BOOST_TEST(call(kBaseParams) == prop.expectation_value(kBaseParams));
}

// The other half of the same rule: a dict the store rejects commits nothing, so the core term stays
// where it was and the propagator's own expectation value is untouched.
BOOST_AUTO_TEST_CASE(a_rejected_reweight_leaves_the_core_term_alone) {
    auto prop = make_propagator(/*schrodinger=*/false, /*partitions=*/1, /*first_weight=*/1.0, kCoreTerm);
    build_base_graph(prop);
    const double before = prop.expectation_value(kBaseParams);

    OperatorDict rejected;
    rejected[VecZ{}] = std::complex<double>{2.0, 0.0};
    rejected[VecZ{0, 2}] = std::complex<double>{0.0, 1.0}; // a term the operator does not hold
    BOOST_CHECK_THROW(prop.update_initial_operator(rejected), std::runtime_error);

    BOOST_TEST(prop.core_term() == kCoreTerm);
    BOOST_TEST(prop.expectation_value(kBaseParams) == before, tt::tolerance(1e-12));
}
