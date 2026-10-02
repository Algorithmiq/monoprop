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

// First on purpose: the header must compile on its own.
#include "monoprop/detail/sharded/Evaluation.h"

/*
 * Snapshot-safe replay, energy, gradients, retained/pared functionals and coefficient-informed construction over T
 * shards of one process (detail/sharded/Evaluation.h, Construction.h). The team size T is the captured launch budget;
 * cpp/tests/CMakeLists.txt reruns these cases in fresh processes at T = 1, 2 and 4. Cases that need a nonprimary worker
 * are left out of T = 1.
 *
 * Oracles:
 * - The legacy partition facade (partitions = T over an in-process communicator; child t is flat owner t at geometry
 *   (1, T)), or at T = 1 the single store: per-shard evolved coefficients, local terms, energies and every gradient
 *   component, bitwise. The two paths share the extracted kernels (LayerReplay.h), so they can agree on a shared bug:
 * - an independent coefficient-map propagator (algebra primitives only) for exact-cutoff energies, the frozen exact
 *   energy of tests/data/random_exact.msgpack, closed-form single-gate fits and central finite differences;
 * - across T, global retained maps of the in-process T = 1 store.
 *
 * Workers never call BOOST_TEST. Observers write per-shard slots and assertions run after the team has joined.
 */

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "AllocationProbe.h"
#include "PauliTestOracle.h"
#include "PropagatorTestAccess.h"
#include "TestUtilities.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/LayerReplay.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/pare/PareGraph.h"
#include "monoprop/detail/sharded/Construction.h"
#include "monoprop/detail/sharded/State.h"
#include "monoprop/detail/sharded/Team.h"

namespace {

using namespace monoprop;
namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;
namespace replay = monoprop::detail::replay;
namespace allocation = test_utils::allocation;
using E = sharded::EvaluationWork;
using W = sharded::ConstructionWork;
using Key = std::vector<uint64_t>;

constexpr size_t kN = 8;
constexpr double kExactAtol = 1e-9; // the frozen exact-energy tolerance of the construction and fused-sweep tests

auto team_options() -> parallel::Options {
    return parallel::capture_thread_budget();
}

auto team_size() -> size_t {
    return static_cast<size_t>(team_options().threads);
}

auto has_nonprimary_worker(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const auto threads = parallel::capture_thread_budget().threads;
    boost::test_tools::assertion_result result(threads >= 2);
    result.message() << "the captured team budget is " << threads << "; this case needs a nonprimary worker";
    return result;
}

auto has_allocation_probe(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    boost::test_tools::assertion_result result(allocation::available());
    result.message() << "this build's sanitizer runtime owns operator new/delete, so no allocation probe is linked";
    return result;
}

template <size_t N>
auto key_of(const Monomial<N> &m) -> Key {
    Key key;
    for (size_t w = 0; w < Monomial<N>::num_words(); ++w) {
        key.push_back(m.word(w));
    }
    return key;
}

auto bits(double v) -> uint64_t {
    return std::bit_cast<uint64_t>(v);
}

auto bits_of(const VecD &v) -> std::vector<uint64_t> {
    std::vector<uint64_t> out;
    out.reserve(v.size());
    for (const double x : v) {
        out.push_back(bits(x));
    }
    return out;
}

auto message_of(const std::exception_ptr &error) -> std::string {
    try {
        std::rethrow_exception(error);
    }
    catch (const std::exception &e) {
        return e.what();
    }
    catch (...) {
        return "non-standard exception";
    }
}

// --- Fixtures -------------------------------------------------------------------------------------------------

struct Fixture {
    std::string name;
    OperatorDict op;
    VecZ initial_state;
    std::optional<unsigned int> schrodinger_cutoff = std::nullopt;
    Basis basis = Basis::Majorana;
    CutoffType cutoff_type = CutoffType::Length;
    unsigned int cutoff = 4;
    size_t logical = kN;
    std::optional<double> lower_atol = std::nullopt;
    std::optional<double> upper_atol = std::nullopt;
    std::optional<std::vector<VecZ>> basis_change = std::nullopt;
};

struct Circuit {
    std::vector<VecZ> gates;
    VecZ mapping;
    VecD gen_coeffs;
    VecD params;
};

struct Case {
    Fixture f;
    Circuit c;
    std::optional<size_t> k = std::nullopt; // only_rotate_len_k
};

auto add_term(OperatorDict &op, Basis basis, const VecZ &indices, double value) -> void {
    op[indices] = algebra_decode_coeff<kN>(basis, std::complex<double>(value, 0.0), indices_to_bitset<kN>(indices));
}

// Pairs, a ladder of quartics and long terms (which spill past the packed inline width), plus identity.
auto majorana_operator(size_t logical) -> OperatorDict {
    OperatorDict op;
    const size_t slots = 2 * logical;
    auto value = 0.25;
    for (size_t i = 0; i < slots; ++i) {
        for (size_t j = i + 1; j < slots; j += 2) {
            add_term(op, Basis::Majorana, {i, j}, value);
            value += 0.125;
        }
    }
    for (size_t i = 0; i + 3 < slots; ++i) {
        add_term(op, Basis::Majorana, {i, i + 1, i + 2, i + 3}, -0.5 - (0.03125 * static_cast<double>(i)));
    }
    add_term(op, Basis::Majorana, {0, 1, 2, 3, 4, 5}, 0.75);
    add_term(op, Basis::Majorana, {1, 3, 5, 7, 9, 11}, -1.25);
    add_term(op, Basis::Majorana, {0, 1, 2, 3, 4, 5, 6, 7}, 2.0);
    add_term(op, Basis::Majorana, {}, 3.5);
    return op;
}

auto pauli_operator() -> OperatorDict {
    OperatorDict op;
    auto value = 0.5;
    const auto add = [&](const std::string &pauli) {
        add_term(op, Basis::Pauli, pauli_oracle::slots_of_string(pauli), value);
        value += 0.25;
    };
    for (size_t q = 0; q < kN; ++q) {
        for (const char p : {'X', 'Z'}) {
            std::string s(kN, 'I');
            s[q] = p;
            add(s);
        }
    }
    for (size_t q = 0; q + 1 < kN; ++q) {
        std::string s(kN, 'I');
        s[q] = 'Z';
        s[q + 1] = 'Z';
        add(s);
    }
    add("XYZXYIII");
    add("ZZZZZZZZ");
    add_term(op, Basis::Pauli, {}, -1.75);
    return op;
}

// Two-, three- (odd), four- and six-slot generators, a repeated gate, a repeated parameter and an identity gate.
auto majorana_circuit(size_t logical) -> Circuit {
    const size_t s = 2 * logical;
    Circuit c;
    c.gates = {{0, 1}, {2, 5}, {1, 3, 4, 6}, {0, s - 1}, {}, {3}, {2, 4, s - 3}, {5, 6, 7, s - 2}, {0, 1}, {1, 2}};
    c.mapping = {0, 1, 2, 3, 4, 5, 6, 2, 7, 8};
    for (size_t i = 0; i < c.gates.size(); ++i) {
        c.gen_coeffs.push_back(i % 3 == 0 ? -1.0 : 0.5 + (0.25 * static_cast<double>(i)));
    }
    for (size_t i = 0; i < 9; ++i) {
        c.params.push_back(0.11 + (0.07 * static_cast<double>(i)));
    }
    return c;
}

auto pauli_circuit() -> Circuit {
    Circuit c;
    for (const std::string g : {"XXIIIIII", "YZXIIIII", "ZIIIIIII", "IIXYIIII", "IIIIIIII", "ZZZZIIII", "IYIIIXII"}) {
        c.gates.push_back(pauli_oracle::slots_of_string(g));
    }
    c.mapping.resize(c.gates.size());
    std::iota(c.mapping.begin(), c.mapping.end(), size_t{0});
    for (size_t i = 0; i < c.gates.size(); ++i) {
        c.gen_coeffs.push_back(0.75 - (0.125 * static_cast<double>(i)));
        c.params.push_back(0.2 + (0.05 * static_cast<double>(i)));
    }
    return c;
}

auto data_circuit(const test_utils::CaseData &data) -> Circuit {
    return {.gates = data.majoranas,
            .mapping = data.param_inds,
            .gen_coeffs = data.gen_coeffs,
            .params = data.parameters};
}

// A basis change that permutes the slots pairwise.
auto swap_basis(size_t logical) -> std::vector<VecZ> {
    std::vector<VecZ> rows;
    for (size_t i = 0; i < 2 * logical; ++i) {
        rows.push_back({i ^ 1U});
    }
    return rows;
}

// Every basis and picture, trimming cutoffs, coefficient cutoffs, length caps (two-pass fallback), a basis change,
// odd Majorana generators and native Pauli folding.
auto legacy_cases() -> std::vector<Case> {
    const auto data = test_utils::load_case_data<kN>("random_exact.msgpack");
    const Fixture maj{.name = "majorana-heisenberg-logical-6",
                      .op = majorana_operator(6),
                      .initial_state = {0, 3},
                      .logical = 6};
    auto maj_s = maj;
    maj_s.name = "majorana-schrodinger-logical-6";
    maj_s.schrodinger_cutoff = 5;
    auto maj_atol = maj;
    maj_atol.name = "majorana-heisenberg-atol";
    maj_atol.lower_atol = 0.02;
    maj_atol.upper_atol = 0.3;
    auto maj_s_atol = maj_s;
    maj_s_atol.name = "majorana-schrodinger-atol";
    maj_s_atol.lower_atol = 0.001;
    maj_s_atol.upper_atol = 0.05;
    auto maj_bc = maj;
    maj_bc.name = "majorana-heisenberg-basis-change";
    maj_bc.basis_change = swap_basis(6);
    const Fixture rnd{.name = "majorana-heisenberg-random-exact",
                      .op = data.hamiltonian,
                      .initial_state = data.initial_state,
                      .cutoff = 6};
    auto rnd_s = rnd;
    rnd_s.name = "majorana-schrodinger-random-exact";
    rnd_s.schrodinger_cutoff = 6;
    const Fixture pauli{.name = "pauli-heisenberg",
                        .op = pauli_operator(),
                        .initial_state = {},
                        .basis = Basis::Pauli,
                        .cutoff_type = CutoffType::Support,
                        .cutoff = 3};
    auto pauli_s = pauli;
    pauli_s.name = "pauli-schrodinger";
    pauli_s.schrodinger_cutoff = 5;
    return {
        {.f = maj, .c = majorana_circuit(6)},
        {.f = maj_s, .c = majorana_circuit(6)},
        {.f = maj_atol, .c = majorana_circuit(6)},
        {.f = maj_s_atol, .c = majorana_circuit(6)},
        {.f = maj, .c = majorana_circuit(6), .k = 4},
        {.f = maj_s, .c = majorana_circuit(6), .k = 2},
        {.f = maj_bc, .c = majorana_circuit(6)},
        {.f = rnd, .c = data_circuit(data)},
        {.f = rnd_s, .c = data_circuit(data)},
        {.f = pauli, .c = pauli_circuit()},
        {.f = pauli_s, .c = pauli_circuit()},
    };
}

auto label(const Case &c) -> std::string {
    return c.k ? std::format("{} k={}", c.f.name, *c.k) : c.f.name;
}

// Parameter vectors on the case's axis: the case's own, a deep-amplification set (every layer records what the layer
// below rotates), and one with vanishing cosines on two consecutive layers (duplicate record indices).
auto parameter_sets(const Circuit &c) -> std::vector<std::pair<std::string, VecD>> {
    std::vector<std::pair<std::string, VecD>> sets = {{"case", c.params}};
    VecD deep(c.params.size());
    for (size_t i = 0; i < deep.size(); ++i) {
        deep[i] = 0.7 + (0.013 * static_cast<double>(i));
    }
    sets.emplace_back("deep", deep);
    if (c.gen_coeffs.size() >= 2) {
        // The last two gates' layers are the first two replayed in Heisenberg and the last two in Schrödinger; both
        // are driven to cos(2 * gen * param) ~ 6e-17.
        VecD vanish = c.params;
        const size_t n = c.gates.size();
        for (const size_t gate : {n - 1, n - 2}) {
            const double g = c.gen_coeffs[gate];
            if (g != 0.0) {
                vanish[c.mapping[gate]] = std::numbers::pi / (4.0 * g);
            }
        }
        sets.emplace_back("vanishing", vanish);
    }
    return sets;
}

// --- Legacy oracle ----------------------------------------------------------------------------------------------

template <size_t N>
auto legacy(const Fixture &f, size_t partitions) -> MonomialPropagator<N> {
    return MonomialPropagator<N>(f.op,
                                 f.cutoff,
                                 f.initial_state,
                                 f.schrodinger_cutoff,
                                 MPI_COMM_SELF,
                                 f.lower_atol,
                                 f.upper_atol,
                                 f.cutoff_type,
                                 f.basis_change,
                                 f.logical,
                                 f.basis,
                                 partitions);
}

template <size_t N>
using Access = monoprop::detail::PropagatorTestAccess<N>;

// Per legacy store: the legacy energy evaluation's evolved operator and its local term at `params`.
struct LegacyShard {
    VecD evolved;
    double local = 0.0;
};

template <size_t N>
auto legacy_shards(MonomialPropagator<N> &p, size_t threads, const VecD &params) -> std::vector<LegacyShard> {
    std::vector<LegacyShard> out(threads);
    Access<N>::for_each_store(p, [&](size_t r, MonomialPropagator<N> &store) {
        const auto state = Access<N>::evaluation_state(store);
        out[r].evolved = Access<N>::evaluation_operator(store, params);
        out[r].local = state.dot(out[r].evolved);
    });
    return out;
}

// Per legacy store: contract_partially(params, false), the picture's partial contraction.
template <size_t N>
auto legacy_contractions(MonomialPropagator<N> &p, size_t threads, const VecD &params) -> std::vector<VecD> {
    std::vector<VecD> out(threads);
    Access<N>::for_each_store(p, [&](size_t r, MonomialPropagator<N> &store) {
        out[r] = store.contract_partially(params, false);
    });
    return out;
}

template <size_t N>
auto legacy_owner(MonomialPropagator<N> &p, size_t shard) -> const MonomialPropagator<N> & {
    return Access<N>::partition_count(p) == 0 ? p : Access<N>::partition(p, static_cast<int>(shard));
}

// --- Sharded setup ----------------------------------------------------------------------------------------------

template <size_t N>
auto make_cutoff_fn(const Fixture &f) -> CutoffFn<N> {
    if (f.basis_change) {
        MonomialList<N> basis;
        for (size_t i = 0; i < 2 * f.logical; ++i) {
            basis.push_back(indices_to_bitset_checked<N>((*f.basis_change)[i], 2 * f.logical));
        }
        return monoprop::detail::cutoff_function_basis_change<N>(f.cutoff_type, f.cutoff, basis, f.logical);
    }
    return monoprop::detail::cutoff_function<N>(f.cutoff_type, f.cutoff, f.logical);
}

template <size_t N>
struct Sharded {
    Fixture f;
    CutoffFn<N> cutoff_fn;
    routing::Router router;
    sharded::Shards<N> shards;

    [[nodiscard]] auto ctx() const -> sharded::ConstructionContext<N> {
        return {.cutoff_fn = cutoff_fn,
                .router = router,
                .lower_atol = f.lower_atol,
                .upper_atol = f.upper_atol,
                .basis = f.basis,
                .schrodinger = f.schrodinger_cutoff.has_value(),
                .rank = 0};
    }
    [[nodiscard]] auto schrodinger() const -> bool { return f.schrodinger_cutoff.has_value(); }
    [[nodiscard]] auto core() const -> double {
        return sharded::validate_initial_operator<N>(f.op, f.basis, f.logical);
    }
};

template <size_t N>
auto seed(const Fixture &f, parallel::Options options) -> Sharded<N> {
    const auto router = routing::make_router<N>(1, static_cast<size_t>(options.threads));
    const auto cutoff_fn = make_cutoff_fn<N>(f);
    std::optional<sharded::PairedBasisBounds> paired;
    if (f.schrodinger_cutoff) {
        paired = sharded::paired_basis_bounds(*f.schrodinger_cutoff, f.logical, router.flat_world());
    }
    const auto width_fn = monoprop::detail::cutoff_function<N>(f.cutoff_type, f.cutoff, f.logical);
    const auto seed_inputs = sharded::OperatorSeed<N>{
        .initial_operator = f.op,
        .initial_state = f.initial_state,
        .router = router,
        .basis = f.basis,
        .logical_num_modes = f.logical,
        .paired = paired,
        .inline_width = sharded::packed_inline_width<N>(paired.has_value(), f.basis_change ? cutoff_fn : width_fn)};
    return {.f = f, .cutoff_fn = cutoff_fn, .router = router, .shards = sharded::seed_shards(options, seed_inputs, 0)};
}

template <size_t N>
auto generators(const Fixture &f, const Circuit &c) -> std::vector<Monomial<N>> {
    std::vector<Monomial<N>> out;
    for (const auto &g : c.gates) {
        out.push_back(indices_to_bitset_checked<N>(g, 2 * f.logical));
    }
    return out;
}

auto require_success(const std::exception_ptr &error, const char *what) -> void {
    if (error) {
        BOOST_FAIL(what << " failed: " << message_of(error));
    }
}

template <size_t N>
auto run_graph(Sharded<N> &s, const Circuit &c, std::optional<size_t> k, size_t gate_offset = 0) -> void {
    const auto gens = generators<N>(s.f, c);
    VecZ gate_indices(c.gates.size());
    std::iota(gate_indices.begin(), gate_indices.end(), gate_offset);
    const auto ctx = s.ctx();
    const auto outcome = sharded::build_graph<N>(team_options(),
                                                 s.shards,
                                                 ctx,
                                                 {.generators = gens,
                                                  .parameter_mapping = c.mapping,
                                                  .gen_coeffs = c.gen_coeffs,
                                                  .gate_indices = gate_indices,
                                                  .only_rotate_len_k = k});
    require_success(outcome.error, "build_graph");
}

// Optimizer-order gate arrays of a shard graph, as MonomialPropagator::graph_gate_arrays_() reads them.
auto gate_arrays(const MPGraph &graph) -> std::pair<VecZ, VecD> {
    const size_t count = graph.layers();
    VecZ mapping(count);
    VecD gen(count);
    for (size_t layer = 0; layer < count; ++layer) {
        const auto t = graph.get_layer_traversal(layer);
        mapping[count - 1 - layer] = t.param_index();
        gen[count - 1 - layer] = t.gen_coeff();
    }
    return {std::move(mapping), std::move(gen)};
}

// contract_partially()'s replay angles for a picture over a graph's layers.
auto contraction_angles(const MPGraph &graph, bool schrodinger, const VecD &params) -> VecD {
    const auto [mapping, gen] = gate_arrays(graph);
    return schrodinger ? map_params(params, mapping, gen, -1.0) : map_params(params, mapping, gen, 1.0, true);
}

// Coefficient-informed construction on the shards, with the angles the legacy build_graph derives.
template <size_t N, class Observer = sharded::NoConstructionObserver>
auto run_informed(Sharded<N> &s,
                  const Circuit &c,
                  const VecD &parameters,
                  std::optional<size_t> k,
                  size_t gate_offset = 0,
                  const Observer &observer = {}) -> sharded::ConstructionOutcome {
    const auto gens = generators<N>(s.f, c);
    VecZ gate_indices(c.gates.size());
    std::iota(gate_indices.begin(), gate_indices.end(), gate_offset);
    const auto mapped = map_params(parameters, c.mapping, c.gen_coeffs, 1.0);
    std::optional<VecD> seed_angles;
    if (s.shards.front()->graph.layers() > 0) {
        const auto [existing, _] = gate_arrays(s.shards.front()->graph);
        const size_t m = expected_num_params(existing);
        const VecD prefix(parameters.begin(), parameters.begin() + static_cast<std::ptrdiff_t>(m));
        seed_angles = contraction_angles(s.shards.front()->graph, s.schrodinger(), prefix);
    }
    const auto ctx = s.ctx();
    return sharded::build_graph_informed<N>(
        team_options(),
        s.shards,
        ctx,
        {.graph = {.generators = gens,
                   .parameter_mapping = c.mapping,
                   .gen_coeffs = c.gen_coeffs,
                   .gate_indices = gate_indices,
                   .only_rotate_len_k = k},
         .mapped_params = mapped,
         .seed_params = seed_angles ? std::optional<std::span<const double>>(*seed_angles) : std::nullopt},
        observer);
}

template <size_t N>
auto retained_context(const Sharded<N> &s, std::optional<double> pare) -> sharded::RetainedContext {
    auto [mapping, gen] = gate_arrays(s.shards.front()->graph);
    return {.core_term = s.core(),
            .parameter_mapping = std::move(mapping),
            .gen_coeffs = std::move(gen),
            .pare_threshold = pare,
            .basis = s.f.basis,
            .schrodinger = s.schrodinger(),
            .rank = 0};
}

template <size_t N>
auto retain(Sharded<N> &s,
            std::optional<double> pare = std::nullopt,
            const sharded::EvaluationObserver *observer = nullptr) -> sharded::RetainedEvaluation {
    auto outcome = sharded::prepare_retained<N>(team_options(), s.shards, retained_context(s, pare), observer);
    require_success(outcome.error, "prepare_retained");
    BOOST_TEST_REQUIRE(outcome.retained.has_value());
    return std::move(*outcome.retained);
}

auto evaluate(const sharded::RetainedEvaluation &r,
              const VecD &params,
              bool gradient,
              const sharded::EvaluationObserver *observer = nullptr) -> sharded::EvaluationOutcome {
    const auto requests = r.requests(params);
    return sharded::evaluate_shards(requests, r.callbacks, team_options(), gradient, observer);
}

auto energy(const sharded::RetainedEvaluation &r, const VecD &params) -> double {
    const auto requests = r.requests(params);
    return sharded::ev_sharded(requests, r.callbacks, team_options(), MPI_COMM_SELF);
}

auto energy_and_gradient(const sharded::RetainedEvaluation &r, const VecD &params) -> std::pair<double, VecD> {
    const auto requests = r.requests(params);
    return sharded::ev_and_grad_sharded(requests, r.callbacks, team_options(), MPI_COMM_SELF);
}

// Per shard: the energy evaluation's evolved operator, through replay_shards().
auto evaluation_operators(const sharded::RetainedEvaluation &r, const VecD &params) -> std::vector<VecD> {
    std::vector<sharded::ReplayRequest> requests;
    for (const auto &shard : r.shards) {
        requests.push_back({.coeffs = shard.op, .graph = shard.graph->replay_view()});
    }
    const auto mapped = map_params(params, r.parameter_mapping, r.gen_coeffs, 1.0, true);
    auto outcome = sharded::replay_shards(requests, mapped, r.callbacks, team_options());
    require_success(outcome.error, "replay_shards");
    return std::move(outcome.coeffs);
}

// --- Observation ------------------------------------------------------------------------------------------------

struct Visit {
    E work;
    size_t step;
    int worker;
    int team;
    int level;
    std::thread::id thread;
};

struct FailAt {
    E work;
    size_t step;
    size_t shard;
};

struct AllocAt {
    E work;
    size_t step;
    size_t shard;
    size_t nth;
};

// Per-shard visit logs and optional injected failures. Slot s is written by the thread performing shard s's work:
// its owner, or the primary inside an exclusive phase; never two threads between the same checkpoints.
class Recorder final : public sharded::EvaluationObserver {
public:
    explicit Recorder(size_t threads, std::vector<FailAt> fail = {}, std::optional<AllocAt> alloc = std::nullopt)
        : logs_(threads),
          fail_(std::move(fail)),
          alloc_(alloc) {
        for (auto &log : logs_) {
            log.reserve(8192);
        }
    }

    auto visit(E work, size_t step, size_t shard) const -> void override {
        (void)allocation::disarm_failure();
        logs_[shard].push_back(
            {work, step, omp_get_thread_num(), omp_get_num_threads(), omp_get_level(), std::this_thread::get_id()});
        for (const auto &f : fail_) {
            if (f.work == work && f.step == step && f.shard == shard) {
                throw std::runtime_error(
                    std::format("injected failure: work {} step {} shard {}", static_cast<int>(work), step, shard));
            }
        }
        if (alloc_ && alloc_->work == work && alloc_->step == step && alloc_->shard == shard) {
            allocation::arm_failure(alloc_->nth);
        }
    }

    [[nodiscard]] auto logs() const -> const std::vector<std::vector<Visit>> & { return logs_; }

    [[nodiscard]] auto count(E work) const -> size_t {
        size_t n = 0;
        for (const auto &log : logs_) {
            n += static_cast<size_t>(std::ranges::count_if(log, [&](const Visit &v) { return v.work == work; }));
        }
        return n;
    }

private:
    mutable std::vector<std::vector<Visit>> logs_;
    std::vector<FailAt> fail_;
    std::optional<AllocAt> alloc_;
};

// After an injected allocation failure, no worker thread keeps an armed failure.
auto disarm_team() -> void {
    static_cast<void>(sharded::run_team(team_options(), [](size_t, sharded::TeamFailure &) noexcept {
        (void)allocation::disarm_failure();
    }));
}

// --- Independent references -------------------------------------------------------------------------------------

using RetainedMap = std::map<Key, double>;

template <size_t N>
auto mono_of(const Key &key) -> Monomial<N> {
    Monomial<N> m;
    for (size_t w = 0; w < key.size(); ++w) {
        m.data()[w] = key[w];
    }
    return m;
}

// Independent propagation on a map: each anticommuting key M moves to cos(2a) M + sin(2a) phase(M) (M ^ G). Absent
// partners enter with 0 (an operator has no implicit terms).
template <size_t N>
auto reference_operator(RetainedMap map, const std::vector<Monomial<N>> &gens, const VecD &mapped, Basis basis)
    -> RetainedMap {
    const size_t n = gens.size();
    for (size_t step = 0; step < n; ++step) {
        const size_t idx = n - 1 - step;
        const auto &gen = gens[idx];
        const double c = std::cos(2 * mapped[idx]);
        const double sn = std::sin(2 * mapped[idx]);
        with_algebra<N>(basis, [&]<typename A>() {
            const auto fold = A::fold_generator(gen);
            const bool odd = A::fold_needs_odd_correction(gen);
            const auto ctx = A::make_gen_context(gen);
            const auto anti = [&](const Monomial<N> &m) {
                return ((m & fold).count() + (odd ? m.count() : 0)) % 2 == 1;
            };
            RetainedMap next;
            for (const auto &[key, v] : map) {
                const auto m = mono_of<N>(key);
                if (!anti(m)) {
                    next[key] += v;
                    continue;
                }
                const auto partner = m ^ gen;
                const int phase =
                    A::emit_phase(A::rotation_sign(ctx, m, partner), m.count(), gen.count(), (m & gen).count());
                next[key] += c * v;
                next[key_of<N>(partner)] += sn * static_cast<double>(phase) * v;
            }
            map = std::move(next);
        });
    }
    return map;
}

// <psi| U^dagger O U |psi> from the reference Heisenberg operator: the identity once, plus each term's state score.
template <size_t N>
auto reference_energy(const Fixture &f, const Circuit &c, const VecD &params) -> double {
    RetainedMap map;
    double core = 0.0;
    for (const auto &[indices, coefficient] : f.op) {
        const auto mono = indices_to_bitset_checked<N>(indices, 2 * f.logical);
        const double encoded = algebra_encode_coeff<N>(f.basis, coefficient, mono);
        if (indices.empty()) {
            core = encoded;
            continue;
        }
        map[key_of<N>(mono)] = encoded;
    }
    const auto mapped = map_params(params, c.mapping, c.gen_coeffs, 1.0);
    const auto evolved = reference_operator<N>(map, generators<N>(f, c), mapped, f.basis);
    const auto mask = initial_state_mask<N>(f.initial_state);
    double total = core;
    for (const auto &[key, v] : evolved) {
        const auto m = mono_of<N>(key);
        bool diagonal = true;
        for (size_t mode = 0; mode < N; ++mode) {
            if (m.test(2 * mode) != m.test((2 * mode) + 1)) {
                diagonal = false;
                break;
            }
        }
        if (diagonal) {
            total += v * algebra_state_phase<N>(f.basis, m, mask);
        }
    }
    return total;
}

template <class Fn>
auto central_differences(Fn &&fn, const VecD &params, double eps = 1e-5) -> VecD {
    VecD grad(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        auto plus = params;
        auto minus = params;
        plus[i] += eps;
        minus[i] -= eps;
        grad[i] = (fn(plus) - fn(minus)) / (2 * eps);
    }
    return grad;
}

auto close(double a, double b, double atol, double rtol = 0.0) -> bool {
    return std::abs(a - b) <= atol + (rtol * std::max(std::abs(a), std::abs(b)));
}

// Layout census of a set of shard graphs: what the fixtures exercise.
struct Census {
    size_t self_pairs = 0;   // layers with a nonempty self slot
    size_t cross_slots = 0;  // occupied slots of another local shard
    size_t multi_peer = 0;   // layers with two or more partner slots
    size_t empty_layers = 0; // layers with no occupied slot at all
    size_t empty_owners = 0; // shards without rows
};

template <size_t N>
auto census(const sharded::Shards<N> &shards) -> Census {
    Census c;
    for (size_t t = 0; t < shards.size(); ++t) {
        const auto &state = *shards[t];
        c.empty_owners += state.op.size() == 0 ? 1 : 0;
        for (size_t l = 0; l < state.graph.layers(); ++l) {
            const auto layer = state.graph.get_layer_traversal(l);
            size_t partners = 0;
            size_t occupied = 0;
            layer.for_each_occupied_slot([&](size_t slot, const monoprop::detail::CrossRankSlotView &view) {
                ++occupied;
                if (slot != t && view.sin_send_count > 0) {
                    ++partners;
                }
            });
            c.self_pairs += layer.cross_rank_self_slot().sin_send_count >= 2 ? 1 : 0;
            c.cross_slots += partners;
            c.multi_peer += partners >= 2 ? 1 : 0;
            c.empty_layers += occupied == 0 ? 1 : 0;
        }
    }
    return c;
}

} // namespace

// --- Fixed-geometry equality with the legacy partitions ---------------------------------------------------------

// At (1, T), for every basis/picture fixture and parameter set (case angles, deep amplification, vanishing cosines):
// per-shard evolved operators and local terms, the energy and every gradient component equal the legacy partitions'
// bitwise, including through ev_sharded()/ev_and_grad_sharded().
BOOST_AUTO_TEST_CASE(sharded_evaluation_matches_legacy_partitions) {
    const auto options = team_options();
    const size_t threads = team_size();
    size_t records_below = 0;
    size_t cosine_sets = 0;
    for (const auto &cs : legacy_cases()) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto old = legacy<kN>(cs.f, threads);
            old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
            auto s = seed<kN>(cs.f, options);
            run_graph<kN>(s, cs.c, cs.k);
            const auto r = retain<kN>(s);
            for (const auto &[name, params] : parameter_sets(cs.c)) {
                BOOST_TEST_CONTEXT(name) {
                    std::vector<uint8_t> wanted;
                    const bool vanishes =
                        replay::plan_cos_records(map_params(params, r.parameter_mapping, r.gen_coeffs, 1.0, true),
                                                 wanted);
                    cosine_sets += vanishes ? 1 : 0;
                    records_below += static_cast<size_t>(std::ranges::count_if(wanted, [](uint8_t w) {
                        return (w & replay::kRecordRotationsBelow) != 0;
                    }));
                    const auto want = legacy_shards<kN>(old, threads, params);
                    const auto got = evaluation_operators(r, params);
                    const auto outcome = evaluate(r, params, false);
                    require_success(outcome.error, "evaluate");
                    for (size_t t = 0; t < threads; ++t) {
                        BOOST_TEST((bits_of(got[t]) == bits_of(want[t].evolved)), "shard " << t);
                        BOOST_TEST(bits(outcome.contributions[t]) == bits(want[t].local), "shard " << t);
                    }
                    const double e = old.expectation_value(params);
                    BOOST_TEST(bits(energy(r, params)) == bits(e));
                    const auto [ge, grad] = old.expectation_value_and_gradient(params);
                    const auto [se, sgrad] = energy_and_gradient(r, params);
                    BOOST_TEST(bits(se) == bits(ge));
                    BOOST_TEST((bits_of(sgrad) == bits_of(grad)));
                    const auto gout = evaluate(r, params, true);
                    require_success(gout.error, "evaluate gradient");
                    BOOST_TEST((bits_of(sharded::combine_gradients(gout.gradients)) == bits_of(grad)));
                }
            }
        }
    }
    // The parameter sets reach both record branches.
    BOOST_TEST(records_below > 0U);
    BOOST_TEST(cosine_sets > 0U);
}

// A small cross-shard fixture: every shard both publishes and reads partner endpoints in the same layer, so a finish
// that read a partner's live (already cos-scaled) coefficients instead of its snapshot would be off by a factor cos.
// Checked in both directions against the independent map propagator and its central differences.
BOOST_AUTO_TEST_CASE(sharded_evaluation_peers_read_published_snapshots,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    Fixture f{.name = "cross-shard", .op = {}, .initial_state = {0, 2}, .cutoff = 2 * kN, .logical = kN};
    for (size_t i = 0; i + 1 < 2 * kN; ++i) {
        add_term(f.op, Basis::Majorana, {i, i + 1}, 0.3 + (0.05 * static_cast<double>(i)));
    }
    add_term(f.op, Basis::Majorana, {0, 1, 2, 3}, -0.4);
    add_term(f.op, Basis::Majorana, {}, 0.25);
    const Circuit c{.gates = {{1, 2}, {3, 4, 5, 6}, {0, 3}, {2, 7}},
                    .mapping = {0, 1, 2, 3},
                    .gen_coeffs = {0.5, -0.75, 1.0, 0.25},
                    .params = {0.4, 0.9, -0.3, 1.1}};
    auto s = seed<kN>(f, options);
    run_graph<kN>(s, c, std::nullopt);
    const auto layout = census<kN>(s.shards);
    BOOST_TEST_REQUIRE(layout.cross_slots > 0U);
    const auto r = retain<kN>(s);
    const auto reference = [&](const VecD &p) { return reference_energy<kN>(f, c, p); };
    const auto [e, grad] = energy_and_gradient(r, c.params);
    BOOST_TEST(close(energy(r, c.params), reference(c.params), kExactAtol));
    BOOST_TEST(close(e, reference(c.params), kExactAtol));
    const auto fd = central_differences(reference, c.params);
    for (size_t i = 0; i < grad.size(); ++i) {
        BOOST_TEST(close(grad[i], fd[i], 1e-6), "component " << i << ": " << grad[i] << " vs " << fd[i]);
    }
}

// Whole self pairs, other-local-shard and multi-peer layouts, empty owners and no-work layers are all exercised at
// T >= 2, and evaluation over them matches the legacy partitions.
BOOST_AUTO_TEST_CASE(sharded_evaluation_covers_every_layout, *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const size_t threads = team_size();
    Census total;
    const auto accumulate = [&](const Census &c) {
        total.self_pairs += c.self_pairs;
        total.cross_slots += c.cross_slots;
        total.multi_peer += c.multi_peer;
        total.empty_layers += c.empty_layers;
        total.empty_owners += c.empty_owners;
    };
    Fixture two{.name = "two-terms", .op = {}, .initial_state = {0}, .cutoff = 4, .logical = 6};
    add_term(two.op, Basis::Majorana, {0, 1}, 0.5);
    add_term(two.op, Basis::Majorana, {}, 1.5);
    const Circuit few{.gates = {{}, {1, 2}}, .mapping = {0, 1}, .gen_coeffs = {1.0, 0.5}, .params = {0.3, 0.4}};
    std::vector<Case> cases = legacy_cases();
    cases.push_back({.f = two, .c = few});
    for (const auto &cs : cases) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto s = seed<kN>(cs.f, options);
            run_graph<kN>(s, cs.c, cs.k);
            accumulate(census<kN>(s.shards));
            auto old = legacy<kN>(cs.f, threads);
            old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
            const auto r = retain<kN>(s);
            const auto [ge, grad] = old.expectation_value_and_gradient(cs.c.params);
            const auto [se, sgrad] = energy_and_gradient(r, cs.c.params);
            BOOST_TEST(bits(se) == bits(ge));
            BOOST_TEST((bits_of(sgrad) == bits_of(grad)));
        }
    }
    BOOST_TEST(total.self_pairs > 0U);
    BOOST_TEST(total.cross_slots > 0U);
    BOOST_TEST(total.empty_layers > 0U);
    if (threads >= 4) {
        // Two partner slots need at least three shards.
        BOOST_TEST(total.multi_peer > 0U);
        BOOST_TEST(total.empty_owners > 0U);
    }
}

// The replicated identity is added once, never once per shard; a Schrödinger identity basis row is an ordinary routed
// row with a zero operator coefficient. Empty parameters contract the un-evolved operator and need no callbacks.
BOOST_AUTO_TEST_CASE(sharded_evaluation_identity_once_and_empty_parameters) {
    const auto options = team_options();
    const size_t threads = team_size();
    for (const auto &cs : legacy_cases()) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto s = seed<kN>(cs.f, options);
            auto r = retain<kN>(s);
            BOOST_TEST(r.num_params == 0U);
            // No callbacks at all: empty parameters must not need them.
            r.callbacks.assign(threads, monoprop::detail::CosCallbacks{});
            const VecD none;
            const auto outcome = evaluate(r, none, true);
            require_success(outcome.error, "evaluate");
            double expected = r.core_term;
            double folded = 0.0;
            for (size_t t = 0; t < threads; ++t) {
                const double local = r.shards[t].state.dot(r.shards[t].op);
                BOOST_TEST(bits(outcome.contributions[t]) == bits(local));
                BOOST_TEST(outcome.gradients[t].empty());
                folded += local;
            }
            expected += folded;
            BOOST_TEST(bits(energy(r, none)) == bits(expected));
            const auto [e, g] = energy_and_gradient(r, none);
            BOOST_TEST(bits(e) == bits(expected));
            BOOST_TEST(g.empty());
            auto old = legacy<kN>(cs.f, threads);
            BOOST_TEST(bits(old.expectation_value(none)) == bits(expected));
            // The identity is not a shard row in Heisenberg; in Schrödinger its basis row carries no operator weight.
            for (const auto &state : s.shards) {
                if (const auto row = state->op.store->find(Monomial<kN>{})) {
                    BOOST_TEST(cs.f.schrodinger_cutoff.has_value());
                    BOOST_TEST(state->op.get_operator()[*row] == 0.0);
                }
            }
        }
    }
}

// Missing callbacks and mismatched arguments are rejected before the team: no observer visit, nothing mutated.
BOOST_AUTO_TEST_CASE(sharded_evaluation_rejects_invalid_arguments_before_the_team) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    const auto r = retain<kN>(s);
    const auto requests = r.requests(cs.c.params);
    const Recorder recorder(threads);
    const auto run = [&](std::span<const EvalRequest> req,
                         std::span<const monoprop::detail::CosCallbacks> cos,
                         bool g) { return sharded::evaluate_shards(req, cos, options, g, &recorder); };
    const auto accepted = [&](std::span<const EvalRequest> req,
                              std::span<const monoprop::detail::CosCallbacks> cos,
                              bool g) { return sharded::evaluate_shards(req, cos, options, g); };
    auto no_scale = r.callbacks;
    no_scale.back().scale = nullptr;
    BOOST_CHECK_THROW((void)run(requests, no_scale, false), MissingLayerCallback);
    auto no_acc = r.callbacks;
    no_acc.front().accumulate = nullptr;
    BOOST_CHECK_NO_THROW(require_success(accepted(requests, no_acc, false).error, "energy without accumulate"));
    BOOST_CHECK_THROW((void)run(requests, no_acc, true), MissingLayerCallback);
    // Indices are needed only where a cosine vanishes.
    auto no_indices = r.callbacks;
    no_indices.back().indices = nullptr;
    BOOST_CHECK_NO_THROW(
        require_success(accepted(requests, no_indices, true).error, "gradient without vanishing cosine"));
    const auto vanishing = parameter_sets(cs.c).back().second;
    const auto vanish_requests = r.requests(vanishing);
    BOOST_CHECK_THROW((void)run(vanish_requests, no_indices, true), MissingLayerCallback);
    // Counts and replicated inputs.
    BOOST_CHECK_THROW((void)run(std::span(requests).first(threads - 1), r.callbacks, false), std::invalid_argument);
    BOOST_CHECK_THROW((void)run(requests, std::span(r.callbacks).first(threads - 1), false), std::invalid_argument);
    if (threads > 1) {
        auto other = cs.c.params;
        other[0] += 1.0;
        auto mixed = r.requests(cs.c.params);
        const auto odd = r.requests(other);
        std::vector<EvalRequest> combined;
        combined.push_back(mixed[0]);
        for (size_t t = 1; t < threads; ++t) {
            combined.push_back(odd[t]);
        }
        BOOST_CHECK_THROW((void)run(combined, r.callbacks, false), std::invalid_argument);
    }
    const VecD too_short(1, 0.1);
    BOOST_CHECK_THROW((void)run(r.requests(too_short), r.callbacks, false), std::invalid_argument);
    // A state longer than its operator.
    const VecD short_op(1, 1.0);
    std::vector<EvalRequest> longer;
    for (size_t t = 0; t < threads; ++t) {
        longer.push_back(EvalRequest{.e_core = r.core_term,
                                     .state = r.shards[t].state,
                                     .op = r.shards[t].state.length() > 1 ? short_op : r.shards[t].op,
                                     .parameter_mapping = r.parameter_mapping,
                                     .gen_coeffs = r.gen_coeffs,
                                     .graph = r.shards[t].graph->replay_view(),
                                     .params = cs.c.params});
    }
    BOOST_CHECK_THROW((void)run(longer, r.callbacks, false), EvalStateArgumentError);
    BOOST_CHECK_THROW((void)sharded::evaluate_shards(requests, r.callbacks, parallel::Options{.threads = 0}, false),
                      std::invalid_argument);
    // Replay: window lengths and a missing scale.
    std::vector<sharded::ReplayRequest> replays;
    for (const auto &shard : r.shards) {
        replays.push_back({.coeffs = shard.op, .graph = shard.graph->replay_view()});
    }
    const VecD one_angle(1, 0.2);
    BOOST_CHECK_THROW((void)sharded::replay_shards(replays, one_angle, r.callbacks, options), std::invalid_argument);
    const auto mapped = map_params(cs.c.params, r.parameter_mapping, r.gen_coeffs, 1.0, true);
    BOOST_CHECK_THROW((void)sharded::replay_shards(replays, mapped, no_scale, options), MissingLayerCallback);
    for (const auto &log : recorder.logs()) {
        BOOST_TEST(log.empty());
    }
}

// Exact-cutoff graphs against the independent map propagator: energies (and the frozen exact energy of
// random_exact), gradients against central differences of the reference, and single-gate closed-form fits.
BOOST_AUTO_TEST_CASE(sharded_evaluation_matches_independent_references) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto data = test_utils::load_case_data<kN>("random_exact.msgpack");
    std::vector<std::pair<Fixture, Circuit>> cases;
    for (const auto sc : {std::optional<unsigned int>{}, std::optional<unsigned int>{16}}) {
        cases.push_back({{.name = "random-exact",
                          .op = data.hamiltonian,
                          .initial_state = data.initial_state,
                          .schrodinger_cutoff = sc,
                          .cutoff = 2 * kN},
                         data_circuit(data)});
        cases.push_back({{.name = "majorana-logical-6",
                          .op = majorana_operator(6),
                          .initial_state = {0, 3},
                          .schrodinger_cutoff = sc ? std::optional<unsigned int>{12} : std::nullopt,
                          .cutoff = 12,
                          .logical = 6},
                         majorana_circuit(6)});
        cases.push_back({{.name = "pauli",
                          .op = pauli_operator(),
                          .initial_state = {},
                          .schrodinger_cutoff = sc,
                          .basis = Basis::Pauli,
                          .cutoff_type = CutoffType::Support,
                          .cutoff = kN},
                         pauli_circuit()});
    }
    // Named elements, not structured bindings: Clang cannot capture a binding in a lambda under OpenMP.
    for (const auto &entry : cases) {
        const Fixture &f = entry.first;
        const Circuit &c = entry.second;
        const bool schrodinger = f.schrodinger_cutoff.has_value();
        BOOST_TEST_CONTEXT(f.name << (schrodinger ? " schrodinger" : " heisenberg") << " T=" << threads) {
            auto s = seed<kN>(f, options);
            run_graph<kN>(s, c, std::nullopt);
            const auto r = retain<kN>(s);
            const auto reference = [&](const VecD &p) { return reference_energy<kN>(f, c, p); };
            const double e = energy(r, c.params);
            BOOST_TEST(close(e, reference(c.params), kExactAtol, 1e-12));
            if (f.name == "random-exact") {
                BOOST_CHECK_SMALL(e - data.actual_expval, kExactAtol);
            }
            const auto [ge, grad] = energy_and_gradient(r, c.params);
            BOOST_TEST(close(ge, e, 1e-12, 1e-12));
            const auto fd = central_differences(reference, c.params);
            for (size_t i = 0; i < grad.size(); ++i) {
                BOOST_TEST(close(grad[i], fd[i], 1e-6, 1e-6), "component " << i << ": " << grad[i] << " vs " << fd[i]);
            }
        }
    }
    // Closed form: with one gate, E(theta) = a + b cos(2 g theta) + c sin(2 g theta); fit a, b, c from three angles and
    // check two further angles and the gradient, for an odd Majorana generator and a Pauli generator.
    const std::vector<std::pair<Fixture, VecZ>> single = {
        {{.name = "odd-majorana", .op = majorana_operator(6), .initial_state = {0, 3}, .cutoff = 12, .logical = 6},
         {1, 4, 6}},
        {{.name = "pauli-xy",
          .op = pauli_operator(),
          .initial_state = {},
          .basis = Basis::Pauli,
          .cutoff_type = CutoffType::Support,
          .cutoff = kN},
         pauli_oracle::slots_of_string("YIIIIIII")},
    };
    for (const auto &[f, gate] : single) {
        BOOST_TEST_CONTEXT(f.name << " closed form T=" << threads) {
            const double g = 0.8;
            const Circuit c{.gates = {gate}, .mapping = {0}, .gen_coeffs = {g}, .params = {0.0}};
            auto s = seed<kN>(f, options);
            run_graph<kN>(s, c, std::nullopt);
            const auto r = retain<kN>(s);
            const auto at = [&](double theta) { return energy(r, VecD{theta}); };
            const double e0 = at(0.0);
            const double eq = at(std::numbers::pi / (4 * g)); // cos = 0, sin = 1
            const double eh = at(std::numbers::pi / (2 * g)); // cos = -1, sin = 0
            const double a = 0.5 * (e0 + eh);
            const double b = 0.5 * (e0 - eh);
            const double cc = eq - a;
            for (const double theta : {0.17, -0.61}) {
                const double want = a + (b * std::cos(2 * g * theta)) + (cc * std::sin(2 * g * theta));
                BOOST_TEST(close(at(theta), want, 1e-12, 1e-12));
                const auto [e, grad] = energy_and_gradient(r, VecD{theta});
                const double dwant = 2 * g * ((-b * std::sin(2 * g * theta)) + (cc * std::cos(2 * g * theta)));
                BOOST_TEST(close(grad[0], dwant, 1e-12, 1e-10), grad[0] << " vs " << dwant);
            }
            // The gate really moves the energy. (An odd Majorana generator never maps an even term onto a diagonal
            // partner, so its sine coefficient vanishes and only the cosine term carries the angle.)
            BOOST_TEST(std::abs(b) + std::abs(cc) > 1e-6);
        }
    }
}

// The substantive cases of tests/test_deep_circuit_gradient.py through the sharded engine: a ZZ observable over eight
// qubits propagated through layers of every ZZ term and an X on every qubit, with a pruning cutoff. Depth, vanishing
// cosines (theta = pi/4 at unit generator coefficient) and angles that record nothing; each against central
// differences of the sharded energy and bitwise against the legacy partitions.
BOOST_AUTO_TEST_CASE(sharded_evaluation_deep_circuit_gradients) {
    const auto options = team_options();
    const size_t threads = team_size();
    Fixture f{.name = "deep-zz",
              .op = {},
              .initial_state = {},
              .basis = Basis::Pauli,
              .cutoff_type = CutoffType::Support,
              .cutoff = 4};
    std::vector<std::string> zz;
    for (size_t i = 0; i < kN; ++i) {
        for (size_t j = i + 1; j < kN; ++j) {
            std::string p(kN, 'I');
            p[i] = 'Z';
            p[j] = 'Z';
            zz.push_back(p);
            add_term(f.op, Basis::Pauli, pauli_oracle::slots_of_string(p), 0.5);
        }
    }
    const auto circuit = [&](size_t layers) {
        Circuit c;
        for (size_t k = 0; k < layers; ++k) {
            for (const auto &p : zz) {
                c.gates.push_back(pauli_oracle::slots_of_string(p));
                c.mapping.push_back(k);
                c.gen_coeffs.push_back(0.5);
            }
            for (size_t q = 0; q < kN; ++q) {
                std::string p(kN, 'I');
                p[q] = 'X';
                c.gates.push_back(pauli_oracle::slots_of_string(p));
                c.mapping.push_back(layers + k);
                c.gen_coeffs.push_back(-1.0);
            }
        }
        return c;
    };
    struct Run {
        size_t layers;
        double theta;
        uint8_t flags; // the record branches the angles must reach
    };
    const std::vector<Run> runs = {{4, 1.0, replay::kRecordRotationsBelow},
                                   {8, 1.0, replay::kRecordRotationsBelow},
                                   {1, std::numbers::pi / 4, replay::kRecordCosineSet},
                                   {2, std::numbers::pi / 4, replay::kRecordCosineSet},
                                   {4, std::numbers::pi / 4, replay::kRecordCosineSet},
                                   {4, 0.05, 0},
                                   {8, 0.05, 0}};
    for (const auto &run : runs) {
        BOOST_TEST_CONTEXT("layers " << run.layers << " theta " << run.theta << " T=" << threads) {
            auto c = circuit(run.layers);
            c.params.assign(2 * run.layers, run.theta);
            auto s = seed<kN>(f, options);
            run_graph<kN>(s, c, std::nullopt);
            const auto r = retain<kN>(s);
            // The cutoff prunes, or the checks would be vacuous.
            size_t cos_only = 0;
            for (const auto &state : s.shards) {
                for (size_t l = 0; l < state->graph.layers(); ++l) {
                    const auto t = state->graph.get_layer_traversal(l);
                    const auto mask = monoprop::detail::full_cos_mask<kN>(state->op.inverted_index(), t, f.basis);
                    cos_only += mask.total_count > t.total_rotation_endpoints() ? 1 : 0;
                }
            }
            BOOST_TEST_REQUIRE(cos_only > 0U);
            std::vector<uint8_t> wanted;
            (void)replay::plan_cos_records(map_params(c.params, r.parameter_mapping, r.gen_coeffs, 1.0, true), wanted);
            uint8_t reached = 0;
            for (const uint8_t w : wanted) {
                reached |= w;
            }
            BOOST_TEST((reached & run.flags) == run.flags);
            if (run.flags == 0) {
                BOOST_TEST(reached == 0);
            }
            const auto [e, grad] = energy_and_gradient(r, c.params);
            BOOST_TEST(std::ranges::all_of(grad, [](double x) { return std::isfinite(x); }));
            const auto fd = central_differences([&](const VecD &p) { return energy(r, p); }, c.params);
            for (size_t i = 0; i < grad.size(); ++i) {
                BOOST_TEST(close(grad[i], fd[i], 1e-6), "component " << i << ": " << grad[i] << " vs " << fd[i]);
            }
            auto old = legacy<kN>(f, threads);
            old.build_graph(c.gates, c.mapping, c.gen_coeffs, std::nullopt, std::nullopt, std::nullopt);
            const auto [le, lgrad] = old.expectation_value_and_gradient(c.params);
            BOOST_TEST(bits(e) == bits(le));
            BOOST_TEST((bits_of(grad) == bits_of(lgrad)));
        }
    }
}

// Repeated parameters accumulate per shard in the reverse order, and two consecutive vanishing layers record the same
// indices twice (rotations below and the cosine set): restoration is idempotent and the gradient stays exact.
BOOST_AUTO_TEST_CASE(sharded_evaluation_repeated_parameters_and_duplicate_records) {
    const auto options = team_options();
    const size_t threads = team_size();
    for (const auto &cs : {legacy_cases()[0], legacy_cases()[1], legacy_cases()[9]}) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto s = seed<kN>(cs.f, options);
            run_graph<kN>(s, cs.c, cs.k);
            const auto r = retain<kN>(s);
            // Drive every other replayed layer to a vanishing cosine: consecutive vanishing layers pile both flags up.
            VecD params = cs.c.params;
            for (size_t gate = 0; gate + 1 < cs.c.gates.size(); gate += 2) {
                if (cs.c.gen_coeffs[gate] != 0.0) {
                    params[cs.c.mapping[gate]] = std::numbers::pi / (4.0 * cs.c.gen_coeffs[gate]);
                }
            }
            std::vector<uint8_t> wanted;
            (void)replay::plan_cos_records(map_params(params, r.parameter_mapping, r.gen_coeffs, 1.0, true), wanted);
            const bool both = std::ranges::any_of(wanted, [](uint8_t w) {
                return w == (replay::kRecordRotationsBelow | replay::kRecordCosineSet);
            });
            BOOST_TEST(both);
            auto old = legacy<kN>(cs.f, threads);
            old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
            const auto [le, lgrad] = old.expectation_value_and_gradient(params);
            const auto [e, grad] = energy_and_gradient(r, params);
            BOOST_TEST(bits(e) == bits(le));
            BOOST_TEST((bits_of(grad) == bits_of(lgrad)));
            // A repeated parameter's component is the sum over its layers: perturbing the parameter moves the energy
            // by the whole component.
            const auto fd = central_differences([&](const VecD &p) { return energy(r, p); }, params, 1e-6);
            for (size_t i = 0; i < grad.size(); ++i) {
                if (std::ranges::count(cs.c.mapping, i) > 1) {
                    BOOST_TEST(close(grad[i], fd[i], 1e-5, 1e-5),
                               "repeated " << i << ": " << grad[i] << " vs " << fd[i]);
                }
            }
        }
    }
}

// Pared functionals equal the legacy partitions' (bitwise, energy and gradient) for positive, zero and negative
// thresholds in both pictures; cross-owner lists are the unchanged shared cores; some layers really store a pruned
// mask; and the owner seam uses the flat owner, not a communicator rank.
BOOST_AUTO_TEST_CASE(sharded_evaluation_pared_functionals_match_legacy) {
    const auto options = team_options();
    const size_t threads = team_size();
    size_t stored = 0;
    size_t empty_stored = 0;
    for (const auto &cs : {legacy_cases()[0], legacy_cases()[1], legacy_cases()[7], legacy_cases()[9]}) {
        auto s = seed<kN>(cs.f, options);
        run_graph<kN>(s, cs.c, cs.k);
        auto old = legacy<kN>(cs.f, threads);
        old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
        for (const double threshold : {1e-3, 0.05, 0.0, -1.0}) {
            BOOST_TEST_CONTEXT(label(cs) << " threshold " << threshold << " T=" << threads) {
                const auto r = retain<kN>(s, threshold);
                const auto ev = old.expectation_value_functional(threshold);
                const auto evg = old.expectation_value_and_gradient_functional(threshold);
                for (const auto &[name, params] : parameter_sets(cs.c)) {
                    BOOST_TEST_CONTEXT(name) {
                        BOOST_TEST(bits(energy(r, params)) == bits(ev(params)));
                        const auto [le, lgrad] = evg(params);
                        const auto [e, grad] = energy_and_gradient(r, params);
                        BOOST_TEST(bits(e) == bits(le));
                        BOOST_TEST((bits_of(grad) == bits_of(lgrad)));
                    }
                }
                for (size_t t = 0; t < threads; ++t) {
                    const MPGraph &pared = *r.shards[t].graph;
                    const MPGraph &live = s.shards[t]->graph;
                    BOOST_TEST_REQUIRE(pared.layers() == live.layers());
                    for (size_t l = 0; l < pared.layers(); ++l) {
                        BOOST_TEST(pared.get_layer(l).shared_core() == live.get_layer(l).shared_core());
                        stored += pared.get_layer(l).pruned_cos() != nullptr ? 1 : 0;
                    }
                    // The owner seam at the flat slot reproduces the retained pared graph exactly.
                    const auto &state = *s.shards[t];
                    const auto &index = state.op.inverted_index();
                    const auto full = [&](size_t i) {
                        return monoprop::detail::full_cos_mask<kN>(index, live.get_layer_traversal(i), cs.f.basis);
                    };
                    const bool pic = cs.f.schrodinger_cutoff.has_value();
                    const auto keep =
                        pic ? indices_above(r.shards[t].op, threshold) : r.shards[t].state.indices_above(threshold);
                    const size_t count = pic ? r.shards[t].op.size() : r.shards[t].state.length();
                    const auto again = monoprop::detail::pare_graph_owner(live, keep, count, pic, t, full);
                    for (size_t l = 0; l < pared.layers(); ++l) {
                        const CosMask *a = pared.get_layer(l).pruned_cos();
                        const CosMask *b = again.get_layer(l).pruned_cos();
                        const bool same =
                            (a == nullptr && b == nullptr) || (a != nullptr && b != nullptr && a->blocks == b->blocks);
                        BOOST_TEST(same, "shard " << t << " layer " << l);
                        empty_stored += (a != nullptr && a->blocks.empty()) ? 1 : 0;
                    }
                }
            }
        }
    }
    BOOST_TEST(stored > 0U);
    BOOST_TEST_MESSAGE("pared layers storing a mask: " << stored << ", of which empty: " << empty_stored);
}

// Stages every partner block in a run of its own, so small layers exercise the multi-run staging of the finishes.
class OneBlockRuns final : public sharded::EvaluationObserver {
public:
    auto visit(E /*work*/, size_t /*step*/, size_t /*shard*/) const -> void override {}
    [[nodiscard]] auto staging_run() const noexcept -> size_t override { return 1; }
};

/*
 * Partner blocks staged one per run give bitwise the results of the default runs, which hold a whole small layer: the
 * staging changes where the finishes read the partners' values, never which values or in which order. Covers forward
 * and reverse finishes (energy, gradient) and the in-team forward replay (replay_shards()).
 */
BOOST_AUTO_TEST_CASE(sharded_evaluation_staging_runs_do_not_change_results) {
    const auto options = team_options();
    const size_t threads = team_size();
    const OneBlockRuns one_block_runs;
    size_t multi_partner_layers = 0;
    for (const auto &cs : {legacy_cases()[0], legacy_cases()[1], legacy_cases()[7], legacy_cases()[9]}) {
        auto s = seed<kN>(cs.f, options);
        run_graph<kN>(s, cs.c, cs.k);
        const auto r = retain<kN>(s);
        for (size_t t = 0; t < threads; ++t) {
            const MPGraph &graph = *r.shards[t].graph;
            for (size_t l = 0; l < graph.layers(); ++l) {
                size_t partners = 0;
                graph.get_layer_traversal(l).for_each_occupied_slot(
                    [&](size_t rank, const auto & /*slot*/) { partners += rank != t ? 1 : 0; });
                multi_partner_layers += partners > 1 ? 1 : 0;
            }
        }
        for (const auto &[name, params] : parameter_sets(cs.c)) {
            BOOST_TEST_CONTEXT(label(cs) << " " << name << " T=" << threads) {
                const auto requests = r.requests(params);
                for (const bool gradient : {false, true}) {
                    const auto plain = sharded::evaluate_shards(requests, r.callbacks, options, gradient);
                    const auto runs =
                        sharded::evaluate_shards(requests, r.callbacks, options, gradient, &one_block_runs);
                    require_success(plain.error, "evaluate_shards");
                    require_success(runs.error, "evaluate_shards, one block per run");
                    BOOST_TEST((bits_of(runs.contributions) == bits_of(plain.contributions)));
                    BOOST_TEST_REQUIRE(runs.gradients.size() == plain.gradients.size());
                    for (size_t t = 0; t < plain.gradients.size(); ++t) {
                        BOOST_TEST((bits_of(runs.gradients[t]) == bits_of(plain.gradients[t])), "shard " << t);
                    }
                }
                std::vector<sharded::ReplayRequest> replays;
                for (const auto &shard : r.shards) {
                    replays.push_back({.coeffs = shard.op, .graph = shard.graph->replay_view()});
                }
                const auto mapped = map_params(params, r.parameter_mapping, r.gen_coeffs, 1.0, true);
                const auto plain = sharded::replay_shards(replays, mapped, r.callbacks, options);
                const auto runs = sharded::replay_shards(replays, mapped, r.callbacks, options, &one_block_runs);
                require_success(plain.error, "replay_shards");
                require_success(runs.error, "replay_shards, one block per run");
                BOOST_TEST_REQUIRE(runs.coeffs.size() == plain.coeffs.size());
                for (size_t t = 0; t < plain.coeffs.size(); ++t) {
                    BOOST_TEST((bits_of(runs.coeffs[t]) == bits_of(plain.coeffs[t])), "shard " << t);
                }
            }
        }
    }
    // With three or more shards some owner has several partners in a layer, so one block per run means several runs.
    if (threads >= 3) {
        BOOST_TEST(multi_partner_layers > 0U);
    }
}

/*
 * CosCallbacks::count reports exactly what CosCallbacks::indices appends, for recomputed folds, pared stored masks and
 * empty stored masks, so replay::reserve_records() reserves the records once: recording every layer with both flags
 * fills the reservation exactly and never reallocates.
 */
BOOST_AUTO_TEST_CASE(sharded_evaluation_cosine_counts_reserve_records_exactly) {
    const auto options = team_options();
    size_t nonempty = 0;
    for (const auto &cs : {legacy_cases()[0], legacy_cases()[1], legacy_cases()[7], legacy_cases()[9]}) {
        auto s = seed<kN>(cs.f, options);
        run_graph<kN>(s, cs.c, cs.k);
        const auto r = retain<kN>(s, 0.05);
        for (size_t t = 0; t < s.shards.size(); ++t) {
            const auto &state = *s.shards[t];
            const auto &index = state.op.inverted_index();
            std::vector<Layer> empty_layers;
            for (size_t l = 0; l < state.graph.layers(); ++l) {
                empty_layers.emplace_back(state.graph.get_layer(l).shared_core(), CosMask{});
            }
            const MPGraph empty_masks(false, std::move(empty_layers));
            for (const auto &[name, view] : {std::pair{"recomputed", state.graph.replay_view()},
                                             std::pair{"pared", r.shards[t].graph->replay_view()},
                                             std::pair{"empty stored", empty_masks.replay_view()}}) {
                BOOST_TEST_CONTEXT(label(cs) << " shard " << t << " " << name) {
                    const auto cos = monoprop::detail::make_cos_callbacks<kN>(index, view, cs.f.basis);
                    BOOST_TEST_REQUIRE(static_cast<bool>(cos.count));
                    std::vector<uint8_t> wanted(view.layers());
                    for (size_t l = 0; l < view.layers(); ++l) {
                        std::vector<TermIndex> indices;
                        cos.indices(l, indices);
                        BOOST_TEST(cos.count(l) == indices.size(), "layer " << l);
                        nonempty += indices.empty() ? 0 : 1;
                        wanted[l] = replay::kRecordCosineSet | (l > 0 ? replay::kRecordRotationsBelow : 0);
                    }
                    replay::CosRecords records;
                    replay::reset_records(records, view.layers());
                    replay::reserve_records(records, wanted, view, cos.count);
                    const size_t reserved = records.values.capacity();
                    const auto *const storage = records.values.data();
                    const VecD op(state.op.size(), 1.0);
                    for (size_t l = 0; l < view.layers(); ++l) {
                        replay::record_pre_layer(records, wanted, view, cos.indices, l, op);
                    }
                    BOOST_TEST(records.values.size() == reserved);
                    BOOST_TEST(records.indices.size() == records.indices.capacity());
                    BOOST_TEST(records.values.data() == storage);
                }
            }
        }
    }
    BOOST_TEST(nonempty > 0U);
}

// A stored empty pruned cosine mask replays nothing: it is not a request to recompute the full mask.
BOOST_AUTO_TEST_CASE(sharded_evaluation_empty_stored_mask_applies_nothing) {
    const auto options = team_options();
    auto s = seed<kN>(legacy_cases()[0].f, options);
    run_graph<kN>(s, legacy_cases()[0].c, std::nullopt);
    const auto &state = *s.shards[0];
    std::vector<Layer> layers;
    for (size_t l = 0; l < state.graph.layers(); ++l) {
        layers.emplace_back(state.graph.get_layer(l).shared_core(), CosMask{});
    }
    const MPGraph empty_masks(false, std::move(layers));
    const auto view = empty_masks.replay_view();
    const auto &index = state.op.inverted_index();
    const auto cos = monoprop::detail::make_cos_callbacks<kN>(index, view, Basis::Majorana);
    const auto recompute = monoprop::detail::make_cos_callbacks<kN>(index, state.graph.replay_view(), Basis::Majorana);
    size_t changed_by_recompute = 0;
    for (size_t l = 0; l < view.layers(); ++l) {
        VecD coeffs(state.op.size());
        std::iota(coeffs.begin(), coeffs.end(), 1.0);
        const VecD before = coeffs;
        cos.scale(l, coeffs.data(), 0.5);
        BOOST_TEST((coeffs == before), "layer " << l);
        std::vector<TermIndex> indices;
        cos.indices(l, indices);
        BOOST_TEST(indices.empty());
        VecD st = before;
        VecD op = before;
        BOOST_TEST(cos.accumulate(l, st.data(), op.data(), 0.5, 2.0) == 0.0);
        recompute.scale(l, coeffs.data(), 0.5);
        changed_by_recompute += coeffs != before ? 1 : 0;
    }
    BOOST_TEST(changed_by_recompute > 0U);
}

// Two differently sized instances evaluated interleaved and repeatedly give bitwise-stable results through the
// retained thread-local scratch; a copy mutated afterwards leaves the original's retained evaluation unchanged; and
// retained callbacks stay exact after the owning index grows, since parity words are fetched at use time.
BOOST_AUTO_TEST_CASE(sharded_evaluation_repeated_instances_copies_and_index_growth) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto small_case = legacy_cases()[9];
    const auto large_case = legacy_cases()[7];
    auto small = seed<kN>(small_case.f, options);
    run_graph<kN>(small, small_case.c, std::nullopt);
    auto large = seed<kN>(large_case.f, options);
    run_graph<kN>(large, large_case.c, std::nullopt);
    const auto rs = retain<kN>(small);
    const auto rl = retain<kN>(large);
    const auto first_small = energy_and_gradient(rs, small_case.c.params);
    const auto first_large = energy_and_gradient(rl, large_case.c.params);
    for (size_t round = 0; round < 3; ++round) {
        const auto a = energy_and_gradient(rl, large_case.c.params);
        const auto b = energy_and_gradient(rs, small_case.c.params);
        const auto c = energy(rl, large_case.c.params);
        BOOST_TEST(bits(a.first) == bits(first_large.first));
        BOOST_TEST((bits_of(a.second) == bits_of(first_large.second)));
        BOOST_TEST(bits(b.first) == bits(first_small.first));
        BOOST_TEST((bits_of(b.second) == bits_of(first_small.second)));
        BOOST_TEST(bits(c) == bits(first_large.first));
    }

    // Copies are independent: extend the copy's graph, the original's retained evaluation is unchanged.
    Sharded<kN> copy{.f = large.f,
                     .cutoff_fn = large.cutoff_fn,
                     .router = large.router,
                     .shards = sharded::copy_shards(options, large.shards)};
    run_graph<kN>(copy, pauli_circuit(), std::nullopt, large_case.c.gates.size());
    const auto again = energy_and_gradient(rl, large_case.c.params);
    BOOST_TEST(bits(again.first) == bits(first_large.first));
    BOOST_TEST((bits_of(again.second) == bits_of(first_large.second)));

    // Index growth under retained callbacks: an owned graph (negative threshold keeps every recomputing layer) keeps
    // its layer count while the shards' store grows.
    const auto owned = retain<kN>(large, -1.0);
    const auto before = energy_and_gradient(owned, large_case.c.params);
    std::vector<size_t> rows_before;
    for (const auto &state : large.shards) {
        rows_before.push_back(state->op.size());
    }
    run_graph<kN>(large, majorana_circuit(kN), std::nullopt, large_case.c.gates.size());
    size_t grown = 0;
    for (size_t t = 0; t < threads; ++t) {
        grown += large.shards[t]->op.size() > rows_before[t] ? 1 : 0;
    }
    BOOST_TEST_REQUIRE(grown > 0U);
    BOOST_CHECK_THROW(rl.validate_call(large_case.c.params), std::runtime_error); // the aliasing graph grew
    BOOST_CHECK_NO_THROW(owned.validate_call(large_case.c.params));
    const auto after = energy_and_gradient(owned, large_case.c.params);
    BOOST_TEST(bits(after.first) == bits(before.first));
    BOOST_TEST((bits_of(after.second) == bits_of(before.second)));
}

// Partial contraction in both pictures: per-shard replay_shards() with contract_partially()'s angles and windows equals
// every legacy child's contract_partially(params, false) bitwise.
BOOST_AUTO_TEST_CASE(sharded_evaluation_partial_contraction_matches_legacy) {
    const auto options = team_options();
    const size_t threads = team_size();
    for (const auto &cs : legacy_cases()) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            const bool schrodinger = cs.f.schrodinger_cutoff.has_value();
            auto s = seed<kN>(cs.f, options);
            run_graph<kN>(s, cs.c, cs.k);
            auto old = legacy<kN>(cs.f, threads);
            old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
            const auto want = legacy_contractions<kN>(old, threads, cs.c.params);
            std::vector<VecD> start;
            std::vector<MPGraphView> views;
            std::vector<monoprop::detail::CosCallbacks> callbacks;
            for (auto &state : s.shards) {
                start.push_back(state->op.current_picture(schrodinger));
                views.push_back(state->graph.slice_view(state->graph.layers()));
            }
            for (size_t t = 0; t < threads; ++t) {
                callbacks.push_back(
                    monoprop::detail::make_cos_callbacks<kN>(s.shards[t]->op.inverted_index(), views[t], cs.f.basis));
            }
            std::vector<sharded::ReplayRequest> requests;
            for (size_t t = 0; t < threads; ++t) {
                requests.push_back({.coeffs = start[t], .graph = views[t]});
            }
            const auto angles = contraction_angles(s.shards.front()->graph, schrodinger, cs.c.params);
            const auto outcome = sharded::replay_shards(requests, angles, callbacks, options);
            require_success(outcome.error, "replay_shards");
            for (size_t t = 0; t < threads; ++t) {
                BOOST_TEST((bits_of(outcome.coeffs[t]) == bits_of(want[t])), "shard " << t);
            }
        }
    }
}

// Coefficient-informed construction, first and incremental (seed replay), in both pictures with cutoff-sensitive
// atols and length caps: every shard's rows, coefficients, caches and graph layers equal the legacy child's after the
// legacy coefficient-informed build_graph; the informed map differs from the structural one; and across T the global
// retained keys agree with the in-process single store.
BOOST_AUTO_TEST_CASE(sharded_evaluation_informed_construction_matches_legacy) {
    const auto options = team_options();
    const size_t threads = team_size();
    size_t sensitive = 0;
    const auto cases = legacy_cases();
    for (const auto &cs : {cases[2], cases[3], cases[4], cases[5], cases[7], cases[9], cases[10]}) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            const bool schrodinger = cs.f.schrodinger_cutoff.has_value();
            // First build (picture seed), then an incremental one on the extended axis (replay seed).
            // The engine's axis: the existing graph replays the prefix [0, m) as the seed, the new gates index past it.
            const Circuit second = cs.f.basis == Basis::Pauli ? pauli_circuit() : majorana_circuit(cs.f.logical);
            const VecD first_params = cs.c.params;
            const size_t first_count = expected_num_params(cs.c.mapping);
            Circuit second_circuit = second;
            for (auto &m : second_circuit.mapping) {
                m += first_count;
            }
            VecD second_params = cs.c.params;
            second_params.insert(second_params.end(), second.params.begin(), second.params.end());
            static_cast<void>(schrodinger);

            auto old = legacy<kN>(cs.f, threads);
            old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, first_params, cs.k);
            auto s = seed<kN>(cs.f, options);
            require_success(run_informed<kN>(s, cs.c, first_params, cs.k).error, "informed build");
            const auto compare = [&](const char *what) {
                for (size_t t = 0; t < threads; ++t) {
                    const auto want = Access<kN>::shard_state(legacy_owner(old, t));
                    const auto &got = *s.shards[t];
                    BOOST_TEST_CONTEXT(what << " shard " << t) {
                        BOOST_TEST_REQUIRE(got.op.size() == want->op.size());
                        for (size_t i = 0; i < got.op.size(); ++i) {
                            BOOST_TEST((key_of<kN>(got.op.store->row(i)) == key_of<kN>(want->op.store->row(i))));
                        }
                        BOOST_TEST((bits_of(got.op.op_coeffs) == bits_of(want->op.op_coeffs)));
                        BOOST_TEST((bits_of(got.op.state_coeffs) == bits_of(want->op.state_coeffs)));
                        BOOST_TEST_REQUIRE(got.graph.layers() == want->graph.layers());
                        for (size_t l = 0; l < got.graph.layers(); ++l) {
                            const auto &a = got.graph.get_layer(l).core();
                            const auto &b = want->graph.get_layer(l).core();
                            BOOST_TEST((a.cross_rank.sin_send_indices == b.cross_rank.sin_send_indices));
                            BOOST_TEST(a.scaled_count == b.scaled_count);
                            BOOST_TEST(a.param_index == b.param_index);
                            BOOST_TEST(bits(a.gen_coeff) == bits(b.gen_coeff));
                            BOOST_TEST(a.gate_index == b.gate_index);
                            BOOST_TEST((a.generator_words == b.generator_words));
                        }
                    }
                }
            };
            compare("first");
            const size_t offset = old.n_gates();
            old.build_graph(second_circuit.gates,
                            second_circuit.mapping,
                            second_circuit.gen_coeffs,
                            std::nullopt,
                            second_params,
                            std::nullopt);
            require_success(run_informed<kN>(s, second_circuit, second_params, std::nullopt, offset).error,
                            "incremental informed build");
            compare("incremental");

            // Cutoff sensitivity: the structural build of the same circuit keeps a different retained key set.
            auto structural = seed<kN>(cs.f, options);
            run_graph<kN>(structural, cs.c, cs.k);
            auto informed = seed<kN>(cs.f, options);
            require_success(run_informed<kN>(informed, cs.c, first_params, cs.k).error, "informed build");
            sensitive += sharded::total_size(structural.shards) != sharded::total_size(informed.shards) ? 1 : 0;

            // Across T: the informed global key set equals the in-process single store's.
            auto single = legacy<kN>(cs.f, 1);
            single.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, first_params, cs.k);
            std::set<Key> want_keys;
            std::set<Key> got_keys;
            const auto one = Access<kN>::shard_state(single);
            for (size_t i = 0; i < one->op.size(); ++i) {
                want_keys.insert(key_of<kN>(one->op.store->row(i)));
            }
            for (const auto &state : informed.shards) {
                for (size_t i = 0; i < state->op.size(); ++i) {
                    got_keys.insert(key_of<kN>(state->op.store->row(i)));
                }
            }
            BOOST_TEST((got_keys == want_keys));
        }
    }
    BOOST_TEST(sensitive > 0U);
}

namespace {

// Captures each shard's evolving coefficients after every new-layer replay of an informed build.
struct ReplayCapture {
    std::vector<VecD> *last;
    std::vector<size_t> *calls;
    auto visit(W /*work*/, size_t /*step*/, size_t /*shard*/) const -> void {}
    [[nodiscard]] auto kernels(size_t) const -> monoprop::detail::NoRangeObserver { return {}; }
    auto replayed(size_t shard, size_t /*step*/, const VecD &coeffs) const -> void {
        (*last)[shard] = coeffs;
        ++(*calls)[shard];
    }
};

// The evolving coefficients an informed build ends with equal an independent replay of the finished graph at the
// picture's contraction angles (Schrödinger negated), from the live picture: rows created by a later layer lie past
// every earlier layer's scaled_count, so earlier layers never touch them.
template <size_t N>
auto check_replayed_coefficients(Sharded<N> &s, const std::vector<VecD> &last, const VecD &params) -> void {
    const bool schrodinger = s.schrodinger();
    std::vector<VecD> start;
    std::vector<MPGraphView> views;
    std::vector<monoprop::detail::CosCallbacks> callbacks;
    for (auto &state : s.shards) {
        start.push_back(state->op.current_picture(schrodinger));
        views.push_back(state->graph.slice_view(state->graph.layers()));
    }
    std::vector<sharded::ReplayRequest> requests;
    for (size_t t = 0; t < s.shards.size(); ++t) {
        callbacks.push_back(
            monoprop::detail::make_cos_callbacks<N>(s.shards[t]->op.inverted_index(), views[t], s.f.basis));
        requests.push_back({.coeffs = start[t], .graph = views[t]});
    }
    const auto angles = contraction_angles(s.shards.front()->graph, schrodinger, params);
    const auto outcome = sharded::replay_shards(requests, angles, callbacks, team_options());
    require_success(outcome.error, "replay_shards");
    for (size_t t = 0; t < s.shards.size(); ++t) {
        BOOST_TEST((bits_of(last[t]) == bits_of(outcome.coeffs[t])), "shard " << t);
    }
}

} // namespace

// Cutoff decisions follow the replayed coefficients: sweeping the lower atol over the range where rows are dropped,
// every informed build equals the legacy children's rows and coefficients and the sweep really changes the retained row
// count; and the evolving coefficients themselves equal an independent replay of the finished graph.
BOOST_AUTO_TEST_CASE(sharded_evaluation_informed_decisions_follow_the_replayed_coefficients) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cases = legacy_cases();
    for (const auto &base : {cases[1], cases[0], cases[10]}) {
        std::set<size_t> sizes;
        for (size_t i = 0; i < 24; ++i) {
            auto cs = base;
            cs.f.lower_atol = 1e-4 * std::pow(10.0, static_cast<double>(i) / 6.0);
            // A larger angle on the second gate, so the replayed sine terms of the first gates matter.
            cs.c.params[1] = 0.9;
            BOOST_TEST_CONTEXT(label(cs) << " lower_atol " << *cs.f.lower_atol << " T=" << threads) {
                auto old = legacy<kN>(cs.f, threads);
                old.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, cs.c.params, cs.k);
                auto s = seed<kN>(cs.f, options);
                std::vector<VecD> last(threads);
                std::vector<size_t> calls(threads, 0);
                require_success(run_informed<kN>(s, cs.c, cs.c.params, cs.k, 0, ReplayCapture{&last, &calls}).error,
                                "informed build");
                sizes.insert(sharded::total_size(s.shards));
                for (const size_t n : calls) {
                    BOOST_TEST(n == cs.c.gates.size());
                }
                check_replayed_coefficients<kN>(s, last, cs.c.params);
                for (size_t t = 0; t < threads; ++t) {
                    const auto want = Access<kN>::shard_state(legacy_owner(old, t));
                    const auto &got = *s.shards[t];
                    BOOST_TEST_REQUIRE(got.op.size() == want->op.size(), "shard " << t);
                    for (size_t row = 0; row < got.op.size(); ++row) {
                        BOOST_TEST((key_of<kN>(got.op.store->row(row)) == key_of<kN>(want->op.store->row(row))));
                    }
                    BOOST_TEST((bits_of(got.op.state_coeffs) == bits_of(want->op.state_coeffs)));
                }
            }
        }
        BOOST_TEST(sizes.size() > 2U, base.f.name << ": the sweep must change the retained row count");
    }
}

// Actual participation, asserted after the join: every evaluation, replay, retained-preparation and informed seed or
// replay visit of shard t ran on worker t at nesting level 1 in a team of T, and every step is present.
BOOST_AUTO_TEST_CASE(sharded_evaluation_owners_do_their_own_work) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[1]; // Schrödinger, cross-owner traffic at T >= 2
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    const Recorder retain_log(threads);
    const auto r = retain<kN>(s, 1e-3, &retain_log);
    const size_t layers = r.expected_layers;
    const auto check = [&](const Recorder &rec, const std::string &what) {
        BOOST_TEST_CONTEXT(what) {
            std::set<std::thread::id> owners;
            for (size_t t = 0; t < threads; ++t) {
                BOOST_TEST_REQUIRE(!rec.logs()[t].empty());
                for (const auto &v : rec.logs()[t]) {
                    BOOST_TEST(v.worker == static_cast<int>(t));
                    BOOST_TEST(v.team == static_cast<int>(threads));
                    BOOST_TEST(v.level == 1);
                }
                owners.insert(rec.logs()[t].front().thread);
            }
            BOOST_TEST(owners.size() == threads);
        }
    };
    check(retain_log, "retain");
    BOOST_TEST(retain_log.count(E::retain) == threads);

    const Recorder energy_log(threads);
    require_success(evaluate(r, cs.c.params, false, &energy_log).error, "evaluate");
    check(energy_log, "energy");
    for (const E work : {E::publish, E::cosine, E::finish}) {
        BOOST_TEST(energy_log.count(work) == threads * layers);
    }
    BOOST_TEST(energy_log.count(E::contribution) == threads);
    BOOST_TEST(energy_log.count(E::reverse_publish) == 0U);

    const Recorder grad_log(threads);
    require_success(evaluate(r, cs.c.params, true, &grad_log).error, "evaluate gradient");
    check(grad_log, "gradient");
    for (const E work :
         {E::record, E::publish, E::cosine, E::finish, E::reverse_publish, E::reverse_accumulate, E::reverse_finish}) {
        BOOST_TEST(grad_log.count(work) == threads * layers, "work " << static_cast<int>(work));
    }
    // Every step of every shard, in phase order.
    for (size_t t = 0; t < threads; ++t) {
        std::vector<size_t> finishes;
        for (const auto &v : grad_log.logs()[t]) {
            if (v.work == E::finish || v.work == E::reverse_finish) {
                finishes.push_back(v.step);
            }
        }
        std::vector<size_t> want(2 * layers);
        std::iota(want.begin(), want.end(), size_t{0});
        BOOST_TEST((finishes == want));
    }

    // Informed construction's seed replay and new-layer replays.
    struct ConstructionLog {
        std::vector<std::vector<Visit>> *logs;
        auto visit(W work, size_t step, size_t shard) const -> void {
            if (work == W::seed || work == W::replay) {
                (*logs)[shard].push_back({E::frame,
                                          step,
                                          omp_get_thread_num(),
                                          omp_get_num_threads(),
                                          omp_get_level(),
                                          std::this_thread::get_id()});
            }
        }
        [[nodiscard]] auto kernels(size_t) const -> monoprop::detail::NoRangeObserver { return {}; }
    };
    std::vector<std::vector<Visit>> clogs(threads);
    auto inf = seed<kN>(cs.f, options);
    require_success(run_informed<kN>(inf, cs.c, cs.c.params, cs.k).error, "first informed");
    const size_t first_layers = inf.shards.front()->graph.layers();
    require_success(run_informed<kN>(
                        inf,
                        cs.c,
                        [&] {
                            VecD p = cs.c.params;
                            p.insert(p.end(), cs.c.params.begin(), cs.c.params.end());
                            return p;
                        }(),
                        cs.k,
                        first_layers,
                        ConstructionLog{&clogs})
                        .error,
                    "incremental informed");
    for (size_t t = 0; t < threads; ++t) {
        // One seed-preparation visit, three per seed replay step, and two replay visits per gate.
        BOOST_TEST(clogs[t].size() == 1 + (3 * first_layers) + (2 * cs.c.gates.size()));
        for (const auto &v : clogs[t]) {
            BOOST_TEST(v.worker == static_cast<int>(t));
            BOOST_TEST(v.level == 1);
        }
    }
}

// Opaque callback sets run every callback call on the primary, one shard at a time, while snapshots, publication and
// finishes stay on the owners; the results are bitwise those of the owner-parallel run.
BOOST_AUTO_TEST_CASE(sharded_evaluation_opaque_callbacks_run_on_the_primary) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    auto r = retain<kN>(s);
    const auto params = parameter_sets(cs.c).back().second; // vanishing: the indices callback runs too
    const auto want = energy_and_gradient(r, params);
    // Per-shard call logs: every call is made by the primary, so each vector has one writer.
    std::vector<std::vector<int>> calls(threads);
    std::vector<monoprop::detail::CosCallbacks> opaque(threads);
    for (size_t t = 0; t < threads; ++t) {
        const auto inner = r.callbacks[t];
        auto *log = &calls[t];
        opaque[t].scale = [inner, log](size_t l, double *c, double v) {
            log->push_back(omp_get_thread_num());
            inner.scale(l, c, v);
        };
        opaque[t].accumulate = [inner, log](size_t l, double *st, double *h, double v, double sec) {
            log->push_back(omp_get_thread_num());
            return inner.accumulate(l, st, h, v, sec);
        };
        opaque[t].indices = [inner, log](size_t l, std::vector<TermIndex> &out) {
            log->push_back(omp_get_thread_num());
            inner.indices(l, out);
        };
    }
    r.callbacks = opaque;
    const Recorder rec(threads);
    const auto requests = r.requests(params);
    const auto outcome = sharded::evaluate_shards(requests, r.callbacks, options, true, &rec);
    require_success(outcome.error, "opaque evaluation");
    const auto got = energy_and_gradient(r, params);
    BOOST_TEST(bits(got.first) == bits(want.first));
    BOOST_TEST((bits_of(got.second) == bits_of(want.second)));
    for (size_t t = 0; t < threads; ++t) {
        BOOST_TEST(!calls[t].empty());
        BOOST_TEST(std::ranges::all_of(calls[t], [](int worker) { return worker == 0; }));
        for (const auto &v : rec.logs()[t]) {
            const bool callback_work = v.work == E::record || v.work == E::cosine || v.work == E::reverse_accumulate;
            BOOST_TEST(v.worker == (callback_work ? 0 : static_cast<int>(t)), "work " << static_cast<int>(v.work));
        }
    }
}

// The ascending-shard fold is the reference association, visible with order-sensitive values.
BOOST_AUTO_TEST_CASE(sharded_evaluation_combination_folds_in_shard_order) {
    const std::vector<double> values = {1e16, 1.0, -1e16, 1.0};
    BOOST_TEST(sharded::combine_contributions(values) == ((((0.0 + 1e16) + 1.0) + -1e16) + 1.0));
    BOOST_TEST(sharded::combine_contributions(values) != 2.0);
    const std::vector<VecD> gradients = {{1e16, 1.0}, {1.0, 1e16}, {-1e16, 1.0}, {1.0, -1e16}};
    const auto folded = sharded::combine_gradients(gradients);
    BOOST_TEST(folded[0] == ((((0.0 + 1e16) + 1.0) + -1e16) + 1.0));
    BOOST_TEST(folded[1] == ((((0.0 + 1.0) + 1e16) + 1.0) + -1e16));
    BOOST_TEST(sharded::combine_contributions(std::vector<double>{-0.0}) == 0.0);
    BOOST_TEST(!std::signbit(sharded::combine_contributions(std::vector<double>{-0.0})));
    BOOST_CHECK_THROW((void)sharded::combine_gradients(std::vector<VecD>{{1.0}, {1.0, 2.0}}), std::invalid_argument);
    // The evaluator adds the identity to the fold of its per-shard terms, once.
    const auto options = team_options();
    const auto cs = legacy_cases()[7];
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    const auto r = retain<kN>(s);
    const auto outcome = evaluate(r, cs.c.params, false);
    require_success(outcome.error, "evaluate");
    BOOST_TEST(bits(energy(r, cs.c.params))
               == bits(r.core_term + sharded::combine_contributions(outcome.contributions)));
}

// --- Failures -----------------------------------------------------------------------------------------------------

namespace {

// The phase a visit belongs to: frame (-1), then phase p for step work (finish of p - 1 belongs to phase p), and the
// opaque/exclusive work has no separate phase in the owner-parallel runs used here.
auto phase_of(const Visit &v, size_t layers) -> long long {
    switch (v.work) {
        case E::frame:
        case E::retain:
            return -1;
        case E::finish:
        case E::reverse_finish:
            return static_cast<long long>(v.step) + 1;
        case E::contribution:
            return static_cast<long long>(layers);
        case E::record:
        case E::publish:
        case E::cosine:
        case E::reverse_publish:
        case E::reverse_accumulate:
            return static_cast<long long>(v.step);
    }
    return 1LL << 40;
}

auto no_later_work(const Recorder &rec, long long limit, size_t layers) -> bool {
    for (const auto &log : rec.logs()) {
        for (const auto &v : log) {
            if (phase_of(v, layers) > limit) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

// A failure in any evaluation phase, on the primary or the last owner, joins the team, suppresses every later phase,
// returns the original exception from evaluate_shards(), and is rethrown unchanged by ev_sharded() and
// ev_and_grad_sharded(); independent copies survive.
BOOST_AUTO_TEST_CASE(sharded_evaluation_failure_in_each_phase_suppresses_later_work) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[1];
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    const auto copy = sharded::copy_shards(options, s.shards);
    const auto r = retain<kN>(s);
    const size_t layers = r.expected_layers;
    const auto params = parameter_sets(cs.c)[1].second;
    const auto reference = energy_and_gradient(r, params);
    struct Site {
        E work;
        size_t step;
        bool gradient;
    };
    const std::vector<Site> sites = {{E::frame, sharded::kNoStep, false},
                                     {E::publish, 0, false},
                                     {E::cosine, 2, false},
                                     {E::finish, 1, false},
                                     {E::contribution, sharded::kNoStep, false},
                                     {E::record, 3, true},
                                     {E::reverse_publish, layers + 1, true},
                                     {E::reverse_accumulate, layers + 2, true},
                                     {E::reverse_finish, layers, true},
                                     {E::finish, layers - 1, true}};
    for (const auto &site : sites) {
        for (const size_t shard : {size_t{0}, threads - 1}) {
            BOOST_TEST_CONTEXT("work " << static_cast<int>(site.work) << " step " << site.step << " shard " << shard) {
                const std::string message = std::format("injected failure: work {} step {} shard {}",
                                                        static_cast<int>(site.work),
                                                        site.step,
                                                        shard);
                const Recorder rec(threads, {{site.work, site.step, shard}});
                const auto outcome = evaluate(r, params, site.gradient, &rec);
                BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
                BOOST_TEST(message_of(outcome.error) == message);
                const Visit probe{site.work, site.step, 0, 0, 0, {}};
                BOOST_TEST(no_later_work(rec, phase_of(probe, layers), layers));
                const Recorder again(threads, {{site.work, site.step, shard}});
                const auto requests = r.requests(params);
                try {
                    if (site.gradient) {
                        (void)sharded::ev_and_grad_sharded(requests, r.callbacks, options, MPI_COMM_SELF, &again);
                    }
                    else {
                        (void)sharded::ev_sharded(requests, r.callbacks, options, MPI_COMM_SELF, &again);
                    }
                    BOOST_FAIL("the failure was not rethrown");
                }
                catch (const std::runtime_error &e) {
                    BOOST_TEST(std::string(e.what()) == message);
                }
            }
        }
    }
    // After every failure: an independent copy made earlier evaluates identically, and so does the original.
    Sharded<kN> c{.f = s.f,
                  .cutoff_fn = s.cutoff_fn,
                  .router = s.router,
                  .shards = sharded::copy_shards(options, copy)};
    const auto rc = retain<kN>(c);
    const auto from_copy = energy_and_gradient(rc, params);
    BOOST_TEST(bits(from_copy.first) == bits(reference.first));
    BOOST_TEST((bits_of(from_copy.second) == bits_of(reference.second)));
    const auto after = energy_and_gradient(r, params);
    BOOST_TEST(bits(after.first) == bits(reference.first));
    BOOST_TEST((bits_of(after.second) == bits_of(reference.second)));
}

// Real failures inside callbacks (forward cosine pass, reverse accumulation, record indices), inside the retained
// preparation (paring) and inside informed construction (seed replay, new-layer replay): each joins, suppresses later
// work and returns the original exception, with mutation reported where it began.
BOOST_AUTO_TEST_CASE(sharded_evaluation_callback_paring_and_construction_failures) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    auto r = retain<kN>(s);
    const auto params = parameter_sets(cs.c).back().second;
    std::vector<uint8_t> wanted;
    (void)replay::plan_cos_records(map_params(params, r.parameter_mapping, r.gen_coeffs, 1.0, true), wanted);
    const auto vanishing = std::ranges::find_if(wanted, [](uint8_t w) { return (w & replay::kRecordCosineSet) != 0; });
    BOOST_TEST_REQUIRE((vanishing != wanted.end()));
    const auto vanishing_layer = static_cast<size_t>(vanishing - wanted.begin());
    struct Throwing {
        size_t shard;
        int which; // 0 scale, 1 accumulate, 2 indices
        size_t layer;
    };
    for (const Throwing site :
         {Throwing{threads - 1, 0, 3}, Throwing{0, 1, 2}, Throwing{threads - 1, 2, vanishing_layer}}) {
        BOOST_TEST_CONTEXT("callback " << site.which << " shard " << site.shard) {
            auto callbacks = r.callbacks;
            auto &target = callbacks[site.shard];
            const auto inner = target;
            // Still owner-parallel: this wrapper only adds a throw to the built-in closures.
            if (site.which == 0) {
                target.scale = [inner, site](size_t l, double *c, double v) {
                    if (l == site.layer) {
                        throw std::runtime_error("injected callback failure");
                    }
                    inner.scale(l, c, v);
                };
            }
            else if (site.which == 1) {
                target.accumulate = [inner, site](size_t l, double *st, double *h, double v, double sec) {
                    if (l == site.layer) {
                        throw std::runtime_error("injected callback failure");
                    }
                    return inner.accumulate(l, st, h, v, sec);
                };
            }
            else {
                target.indices = [inner, site](size_t l, std::vector<TermIndex> &out) {
                    if (l == site.layer) {
                        throw std::runtime_error("injected callback failure");
                    }
                    inner.indices(l, out);
                };
            }
            const Recorder rec(threads);
            const auto requests = r.requests(params);
            const auto outcome = sharded::evaluate_shards(requests, callbacks, options, true, &rec);
            BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
            BOOST_TEST(message_of(outcome.error) == "injected callback failure");
            BOOST_TEST(rec.count(E::contribution) == (site.which == 1 ? threads : 0U));
        }
    }

    // Paring: the last owner fails inside its retained preparation; mutation had started.
    {
        const Recorder rec(threads, {{E::retain, sharded::kNoStep, threads - 1}});
        const auto outcome = sharded::prepare_retained<kN>(options, s.shards, retained_context(s, 1e-3), &rec);
        BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
        BOOST_TEST(outcome.mutation_started);
        BOOST_TEST(!outcome.retained.has_value());
        BOOST_TEST(message_of(outcome.error)
                   == std::format("injected failure: work {} step {} shard {}",
                                  static_cast<int>(E::retain),
                                  sharded::kNoStep,
                                  threads - 1));
    }

    // Informed construction: a seed-replay step and a new-layer replay fail; nothing after them runs, and an earlier
    // copy survives unchanged.
    struct ConstructionFail {
        std::vector<std::vector<std::pair<W, size_t>>> *logs;
        W work;
        size_t step;
        size_t shard;
        size_t *seen;
        auto visit(W w, size_t st, size_t sh) const -> void {
            (*logs)[sh].emplace_back(w, st);
            if (w == work && st == step && sh == shard) {
                ++*seen;
                if (*seen == 1) {
                    throw std::runtime_error("injected construction failure");
                }
            }
        }
        [[nodiscard]] auto kernels(size_t) const -> monoprop::detail::NoRangeObserver { return {}; }
    };
    for (const auto &[work, step] : {std::pair{W::seed, size_t{2}}, std::pair{W::replay, size_t{1}}}) {
        BOOST_TEST_CONTEXT("construction work " << static_cast<int>(work)) {
            auto inf = seed<kN>(cs.f, options);
            require_success(run_informed<kN>(inf, cs.c, cs.c.params, cs.k).error, "first informed");
            const auto copy = sharded::copy_shards(options, inf.shards);
            const size_t copy_rows = sharded::total_size(copy);
            std::vector<std::vector<std::pair<W, size_t>>> logs(threads);
            size_t seen = 0;
            VecD p = cs.c.params;
            p.insert(p.end(), cs.c.params.begin(), cs.c.params.end());
            const auto outcome = run_informed<kN>(inf,
                                                  cs.c,
                                                  p,
                                                  cs.k,
                                                  inf.shards.front()->graph.layers(),
                                                  ConstructionFail{&logs, work, step, threads - 1, &seen});
            BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
            BOOST_TEST(message_of(outcome.error) == "injected construction failure");
            BOOST_TEST(outcome.mutation_started);
            for (const auto &log : logs) {
                BOOST_TEST(std::ranges::none_of(log, [](const auto &v) { return v.first == W::caches; }));
            }
            BOOST_TEST(sharded::total_size(copy) == copy_rows);
        }
    }
}

// Real allocation failures at every allocation of one owner's frame, snapshot publication and reverse work: each is a
// catchable std::bad_alloc after the join, and the retained scratch left behind does not change later results.
BOOST_AUTO_TEST_CASE(sharded_evaluation_allocation_failures_are_contained,
                     *boost::unit_test::precondition(has_allocation_probe)) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[1];
    auto s = seed<kN>(cs.f, options);
    run_graph<kN>(s, cs.c, cs.k);
    const auto r = retain<kN>(s);
    const size_t layers = r.expected_layers;
    // Vanishing cosines: the records hold a cosine-set layer's indices. The frame phase reserves them exactly
    // (replay::reserve_records), so the frame allocates and the record of that layer does not.
    const auto params = parameter_sets(cs.c).back().second;
    const auto reference = energy_and_gradient(r, params);
    std::vector<uint8_t> wanted;
    (void)replay::plan_cos_records(map_params(params, r.parameter_mapping, r.gen_coeffs, 1.0, true), wanted);
    const auto vanishing = std::ranges::find_if(wanted, [](uint8_t w) { return (w & replay::kRecordCosineSet) != 0; });
    BOOST_TEST_REQUIRE((vanishing != wanted.end()));
    // The scratch sites run on shard 0, the fresh host thread, whose thread-local scratch is always cold. libomp may
    // hand a new root thread pooled workers whose scratch is already warm; the frame site's per-call gradient still
    // allocates there, so it keeps a nonprimary owner covered.
    struct Site {
        AllocAt at;
        bool allocates;
    };
    const std::vector<Site> sites = {{{E::frame, sharded::kNoStep, threads - 1, 0}, true},
                                     {{E::frame, sharded::kNoStep, 0, 0}, true},
                                     {{E::publish, 1, 0, 0}, true},
                                     {{E::reverse_publish, layers + 1, 0, 0}, true},
                                     {{E::record, static_cast<size_t>(vanishing - wanted.begin()), 0, 0}, false}};
    for (const Site &entry : sites) {
        const AllocAt &site = entry.at;
        const bool allocates = entry.allocates;
        size_t injected = 0;
        for (size_t nth = 1; nth < 200; ++nth) {
            bool failed = false;
            bool bad_alloc = false;
            // A fresh host thread gets a fresh OpenMP team, whose workers start with cold thread-local scratch. It
            // makes no Boost.Test call; the main thread asserts after the join.
            std::thread host([&] {
                const Recorder rec(threads, {}, AllocAt{site.work, site.step, site.shard, nth});
                const auto requests = r.requests(params);
                const auto outcome = sharded::evaluate_shards(requests, r.callbacks, options, true, &rec);
                disarm_team();
                failed = static_cast<bool>(outcome.error);
                if (failed) {
                    try {
                        std::rethrow_exception(outcome.error);
                    }
                    catch (const std::bad_alloc &) {
                        bad_alloc = true;
                    }
                    catch (...) {
                    }
                }
            });
            host.join();
            if (failed) {
                ++injected;
                BOOST_TEST(bad_alloc);
            }
            if (!failed) {
                break;
            }
        }
        BOOST_TEST_MESSAGE("allocation site work " << static_cast<int>(site.work) << " shard " << site.shard << ": "
                                                   << injected << " injected failures");
        BOOST_TEST((injected > 0U) == allocates);
    }
    const auto after = energy_and_gradient(r, params);
    BOOST_TEST(bits(after.first) == bits(reference.first));
    BOOST_TEST((bits_of(after.second) == bits_of(reference.second)));
}
