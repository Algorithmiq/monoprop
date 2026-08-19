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
// and without a pare threshold, in both pictures. A cell that says Answers and carries a defect note
// records behaviour that is wrong today and is fixed by the stage the note names.

#include <boost/test/unit_test.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "monoprop/MonomialPropagator.h"
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

auto make_propagator(bool schrodinger) -> Prop {
    OperatorDict initial_ham;
    initial_ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    initial_ham[VecZ{2, 3}] = std::complex<double>{0.0, 0.5};
    const auto cutoff = static_cast<unsigned int>(2 * kNumModes);
    return Prop(initial_ham,
                cutoff,
                VecZ{0, 1},
                schrodinger ? std::optional<unsigned int>{cutoff} : std::nullopt,
                MPI_COMM_SELF,
                std::nullopt,
                std::nullopt,
                CutoffType::Support,
                std::nullopt);
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
    updated[VecZ{0, 1}] = std::complex<double>{0.0, 2.75};
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
    Stale,   // throws, reporting the propagator moved under the functional
    Answers, // returns, and returns exactly what it returned before the mutation
};

struct MutatorRow {
    std::string_view method;    // the public method this row covers
    void (*apply)(Prop &);      //
    bool needs_empty_graph;     // build the functional with no graph, so the mutator is accepted
    Outcome exact;              // pare_threshold == nullopt
    Outcome pared;              // pare_threshold == kPareThreshold
    std::string_view rationale; //
};

constexpr std::array kMutatorTable{
    MutatorRow{.method = "build_graph",
               .apply = &mutate_build_graph,
               .needs_empty_graph = false,
               .exact = Outcome::Stale,
               .pared = Outcome::Answers,
               .rationale = "DEFECT (fixed in stage 2): the pared plan owns its layers, so its layer "
                            "count cannot move and the live-graph check never fires."},
    MutatorRow{.method = "propagate",
               .apply = &mutate_propagate,
               .needs_empty_graph = true,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
               .rationale = "DEFECT (fixed in stage 2): propagate() leaves the layer count at zero and "
                            "does not touch the epoch, so neither check sees the re-evolved operator."},
    MutatorRow{.method = "contract_partially",
               .apply = &mutate_contract_partially,
               .needs_empty_graph = false,
               .exact = Outcome::Stale,
               .pared = Outcome::Answers,
               .rationale = "DEFECT (fixed in stage 2): as build_graph, the pared plan's layer count is fixed."},
    MutatorRow{.method = "update_initial_operator",
               .apply = &mutate_update_initial_operator,
               .needs_empty_graph = false,
               .exact = Outcome::Stale,
               .pared = Outcome::Stale,
               .rationale = "The epoch check fires. Stage 4 turns this into a weight refresh, except for "
                            "a pared Schrodinger plan."},
    MutatorRow{.method = "set_parameter_mapping",
               .apply = &mutate_set_parameter_mapping,
               .needs_empty_graph = false,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
               .rationale = "DEFECT (fixed in stage 2): relabelling is in place, so the layer count and "
                            "the epoch both hold and the plan keeps replaying the old labels."},
    MutatorRow{.method = "update_cutoff",
               .apply = &mutate_update_cutoff,
               .needs_empty_graph = false,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
               .rationale = "Intended: a cutoff gates the next build and changes nothing the plan holds."},
    MutatorRow{.method = "update_cutoff_type",
               .apply = &mutate_update_cutoff_type,
               .needs_empty_graph = false,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
    MutatorRow{.method = "update_basis_change",
               .apply = &mutate_update_basis_change,
               .needs_empty_graph = false,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
    MutatorRow{.method = "update_lower_atol",
               .apply = &mutate_update_lower_atol,
               .needs_empty_graph = false,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
               .rationale = "Intended: as update_cutoff."},
    MutatorRow{.method = "update_upper_atol",
               .apply = &mutate_update_upper_atol,
               .needs_empty_graph = false,
               .exact = Outcome::Answers,
               .pared = Outcome::Answers,
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

static_assert(distinct_methods() == Prop::num_mutating_methods,
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

auto run_row(const MutatorRow &row, bool schrodinger, bool gradient, std::optional<double> pare_threshold) -> void {
    auto prop = make_propagator(schrodinger);
    if (!row.needs_empty_graph) {
        build_base_graph(prop);
    }
    const VecD params = row.needs_empty_graph ? VecD{} : kBaseParams;

    auto call = make_call(prop, gradient, pare_threshold);
    const double before = call(params);

    row.apply(prop);

    const auto expected = pare_threshold.has_value() ? row.pared : row.exact;
    if (expected == Outcome::Stale) {
        BOOST_CHECK_EXCEPTION(call(params), std::runtime_error, [](const std::runtime_error &e) {
            BOOST_TEST_INFO("message: " << e.what());
            return std::string_view(e.what()).find("MP object has been modified") != std::string_view::npos;
        });
        return;
    }
    // Answers: the plan replays its own snapshot, so the number cannot have moved.
    double after = 0.0;
    BOOST_REQUIRE_NO_THROW(after = call(params));
    BOOST_TEST(after == before, tt::tolerance(1e-12));
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

// The two Answers-with-a-defect rows above only say the number did not move. These two say why that
// is wrong: the propagator's own answer *did* move, so the functional is now reporting a circuit
// nobody asked about. Stage 2 turns both calls into throws.

BOOST_AUTO_TEST_CASE(set_parameter_mapping_silently_desynchronises_functional) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, std::nullopt);
    const double before = call(kBaseParams);

    mutate_set_parameter_mapping(prop);

    BOOST_TEST(prop.expectation_value(kBaseParams) != before);
    BOOST_TEST(call(kBaseParams) == before, tt::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(pared_functional_silently_survives_build_graph) {
    auto prop = make_propagator(/*schrodinger=*/false);
    build_base_graph(prop);
    auto call = make_call(prop, /*gradient=*/false, kPareThreshold);
    const double before = call(kBaseParams);

    mutate_build_graph(prop);

    BOOST_TEST(prop.expectation_value(kBaseParams) != before);
    BOOST_TEST(call(kBaseParams) == before, tt::tolerance(1e-12));
}
