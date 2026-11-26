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

#pragma once

// The operator, circuit and shard-seeding fixtures shared by the sharded state, construction and evaluation suites,
// their phase observation, and the references they are compared against: the rank-level root and an independent
// coefficient-map propagator.

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <format>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "PauliTestOracle.h"
#include "PropagatorTestAccess.h"
#include "ShardedTestSupport.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/sharded/Construction.h"
#include "monoprop/detail/sharded/Exchange.h"
#include "monoprop/detail/sharded/State.h"

namespace sharded_test {

// --- Fixtures -------------------------------------------------------------------------------------------------

struct Fixture {
    std::string name;
    OperatorDict op;
    VecZ initial_state;
    std::optional<unsigned int> schrodinger_cutoff = std::nullopt;
    Basis basis = Basis::Majorana;
    CutoffType cutoff_type = CutoffType::Length;
    unsigned int cutoff = 4;
    size_t logical = num_modes;
    std::optional<double> lower_atol = std::nullopt;
    std::optional<double> upper_atol = std::nullopt;
    std::optional<std::vector<VecZ>> basis_change = std::nullopt;
    bool opaque = false; // wrap the cutoff in an opaque closure
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

// Pairs, a ladder of quartics and long terms (which spill past the packed inline width), plus identity.
inline auto majorana_operator(size_t logical) -> OperatorDict {
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

inline auto pauli_operator() -> OperatorDict {
    OperatorDict op;
    auto value = 0.5;
    const auto add = [&](const std::string &pauli) {
        add_term(op, Basis::Pauli, pauli_oracle::slots_of_string(pauli), value);
        value += 0.25;
    };
    for (size_t q = 0; q < num_modes; ++q) {
        for (const char p : {'X', 'Z'}) {
            std::string s(num_modes, 'I');
            s[q] = p;
            add(s);
        }
    }
    for (size_t q = 0; q + 1 < num_modes; ++q) {
        std::string s(num_modes, 'I');
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
inline auto majorana_circuit(size_t logical) -> Circuit {
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

inline auto pauli_circuit() -> Circuit {
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

inline auto data_circuit(const test_utils::CaseData &data) -> Circuit {
    return {.gates = data.majoranas,
            .mapping = data.param_inds,
            .gen_coeffs = data.gen_coeffs,
            .params = data.parameters};
}

// A basis change that permutes the slots pairwise, so the opaque closure really differs from the plain cutoff.
inline auto swap_basis(size_t logical) -> std::vector<VecZ> {
    std::vector<VecZ> rows;
    for (size_t i = 0; i < 2 * logical; ++i) {
        rows.push_back({i ^ 1U});
    }
    return rows;
}

// Every basis and picture, trimming cutoffs, coefficient cutoffs, length caps (two-pass fallback), a basis change,
// odd Majorana generators and native Pauli folding.
inline auto fixture_cases() -> std::vector<Case> {
    const auto data = test_utils::load_case_data<num_modes>("random_exact.msgpack");
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

inline auto label(const Case &c) -> std::string {
    return c.k ? std::format("{} k={}", c.f.name, *c.k) : c.f.name;
}

// Exact-cutoff cases in both pictures: random_exact (`data`), Majorana at six logical modes and Pauli.
inline auto exact_reference_cases(const test_utils::CaseData &data) -> std::vector<std::pair<Fixture, Circuit>> {
    std::vector<std::pair<Fixture, Circuit>> cases;
    for (const auto sc : {std::optional<unsigned int>{}, std::optional<unsigned int>{16}}) {
        cases.push_back({{.name = "random-exact",
                          .op = data.hamiltonian,
                          .initial_state = data.initial_state,
                          .schrodinger_cutoff = sc,
                          .cutoff = 2 * num_modes},
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
                          .cutoff = num_modes},
                         pauli_circuit()});
    }
    return cases;
}

// --- Sharded setup ----------------------------------------------------------------------------------------------

template <size_t N>
auto make_cutoff_fn(const Fixture &f) -> CutoffFn<N> {
    CutoffFn<N> fn;
    if (f.basis_change) {
        MonomialList<N> basis;
        for (size_t i = 0; i < 2 * f.logical; ++i) {
            basis.push_back(indices_to_bitset_checked<N>((*f.basis_change)[i], 2 * f.logical));
        }
        fn = monoprop::detail::cutoff_function_basis_change<N>(f.cutoff_type, f.cutoff, basis, f.logical);
    }
    else {
        fn = monoprop::detail::cutoff_function<N>(f.cutoff_type, f.cutoff, f.logical);
    }
    if (f.opaque) {
        return [inner = fn](const Monomial<N> &m) { return inner(m); };
    }
    return fn;
}

// Shards seeded through the production seeding path, with the context that goes with them.
template <size_t N>
struct Sharded {
    Fixture f;
    CutoffFn<N> cutoff_fn;
    routing::Router router;
    sharded::Shards<N> shards;
    sharded::PhysicalWorld world{}; // one process unless a multi-rank case seeds over MPI_COMM_WORLD

    [[nodiscard]] auto ctx() const -> sharded::ConstructionContext<N> {
        return {.cutoff_fn = cutoff_fn,
                .router = router,
                .lower_atol = f.lower_atol,
                .upper_atol = f.upper_atol,
                .max_sins = std::nullopt,
                .basis = f.basis,
                .schrodinger = f.schrodinger_cutoff.has_value(),
                .world = world};
    }
    [[nodiscard]] auto schrodinger() const -> bool { return f.schrodinger_cutoff.has_value(); }
    [[nodiscard]] auto core() const -> double {
        return sharded::validate_initial_operator<N>(f.op, f.basis, f.logical);
    }
};

// Caller-side preparation of the seeding inputs, as the rank-level constructor performs it. Views `f`.
template <size_t N>
auto operator_seed(const Fixture &f, const routing::Router &router) -> sharded::OperatorSeed<N> {
    std::optional<sharded::PairedBasisBounds> paired;
    if (f.schrodinger_cutoff) {
        paired = sharded::paired_basis_bounds(*f.schrodinger_cutoff, f.logical, router.flat_world());
    }
    // The inline width only selects the row packing (overflow spills losslessly), so the plain cutoff serves here.
    const auto width_fn = f.basis_change ? make_cutoff_fn<N>(f)
                                         : monoprop::detail::cutoff_function<N>(f.cutoff_type, f.cutoff, f.logical);
    return {.initial_operator = f.op,
            .initial_state = f.initial_state,
            .router = router,
            .basis = f.basis,
            .logical_num_modes = f.logical,
            .paired = paired,
            .inline_width = sharded::packed_inline_width<N>(paired.has_value(), width_fn)};
}

template <size_t N>
auto seed(const Fixture &f, parallel::Options options, const sharded::PhysicalWorld &world = {}) -> Sharded<N> {
    const auto router = routing::make_router<N>(world.ranks, static_cast<size_t>(options.threads));
    return {.f = f,
            .cutoff_fn = make_cutoff_fn<N>(f),
            .router = router,
            .shards = sharded::seed_shards(options, operator_seed<N>(f, router), world.rank),
            .world = world};
}

template <size_t N>
auto generators(const Fixture &f, const Circuit &c) -> std::vector<Monomial<N>> {
    std::vector<Monomial<N>> out;
    for (const auto &g : c.gates) {
        out.push_back(indices_to_bitset_checked<N>(g, 2 * f.logical));
    }
    return out;
}

// Gate indices offset, offset + 1, ... for the gates of `c`.
inline auto gate_indices(const Circuit &c, size_t offset) -> VecZ {
    VecZ indices(c.gates.size());
    std::iota(indices.begin(), indices.end(), offset);
    return indices;
}

// Graph construction on the shards, with the gate indices following `gate_offset`.
template <size_t N, class Observer = sharded::NoConstructionObserver>
auto run_graph(Sharded<N> &s,
               const Circuit &c,
               std::optional<size_t> k,
               const Observer &observer = {},
               size_t gate_offset = 0) -> sharded::ConstructionOutcome {
    const auto gens = generators<N>(s.f, c);
    const auto indices = gate_indices(c, gate_offset);
    const auto ctx = s.ctx();
    return sharded::build_graph<N>(team_options(),
                                   s.shards,
                                   ctx,
                                   {.generators = gens,
                                    .parameter_mapping = c.mapping,
                                    .gen_coeffs = c.gen_coeffs,
                                    .gate_indices = indices,
                                    .only_rotate_len_k = k},
                                   observer);
}

// run_graph() without an observer, failing the case if the construction fails.
template <size_t N>
auto build_checked(Sharded<N> &s, const Circuit &c, std::optional<size_t> k = std::nullopt, size_t gate_offset = 0)
    -> void {
    require_success(run_graph<N>(s, c, k, {}, gate_offset).error, "build_graph");
}

// `f` seeded at the launch's team, with `c` built on it.
template <size_t N>
auto seed_and_build(const Fixture &f, const Circuit &c, std::optional<size_t> k = std::nullopt) -> Sharded<N> {
    auto s = seed<N>(f, team_options());
    build_checked<N>(s, c, k);
    return s;
}

// The rank-level reference at geometry (1, T): the root over MPI_COMM_SELF, built through its public operations at the
// launch's T. Its shards come from the same seam, so a case that compares against it checks the root's wiring of the
// seam, and keeps its independent checks.
template <size_t N>
auto reference_root(const Fixture &f, size_t threads) -> MonomialPropagator<N> {
    auto p = MonomialPropagator<N>(f.op,
                                   f.cutoff,
                                   f.initial_state,
                                   f.schrodinger_cutoff,
                                   MPI_COMM_SELF,
                                   f.lower_atol,
                                   f.upper_atol,
                                   f.cutoff_type,
                                   f.basis_change,
                                   f.logical,
                                   f.basis);
    BOOST_TEST_REQUIRE(monoprop::detail::PropagatorTestAccess<N>::shards(p).size() == threads);
    return p;
}

// --- Observation ------------------------------------------------------------------------------------------------

// One observed phase visit, with the worker, team, nesting level and thread that executed it.
template <class Work>
struct Visit {
    Work work;
    size_t step;
    int worker;
    int team;
    int level;
    std::thread::id thread;
};

template <class Work>
struct FailAt {
    Work work;
    size_t step;
    size_t shard;
};

template <class Work>
struct AllocAt {
    Work work;
    size_t step;
    size_t shard;
    size_t nth;
};

// Records a visit in `log`, then throws the failure armed at it, or arms the allocation failure armed at it.
template <class Work>
auto record_visit(std::vector<Visit<Work>> &log,
                  Work work,
                  size_t step,
                  size_t shard,
                  const std::vector<FailAt<Work>> &fail,
                  const std::optional<AllocAt<Work>> &alloc) -> void {
    // A pending armed allocation failure never leaks into a later phase.
    (void)allocation::disarm_failure();
    log.push_back(
        {work, step, omp_get_thread_num(), omp_get_num_threads(), omp_get_level(), std::this_thread::get_id()});
    for (const auto &f : fail) {
        if (f.work == work && f.step == step && f.shard == shard) {
            throw std::runtime_error(
                std::format("injected failure: work {} step {} shard {}", static_cast<int>(work), step, shard));
        }
    }
    if (alloc && alloc->work == work && alloc->step == step && alloc->shard == shard) {
        allocation::arm_failure(alloc->nth);
    }
}

// --- Independent reference --------------------------------------------------------------------------------------

// Independent propagation on a map: each anticommuting key M moves to cos(2a) M + sin(2a) phase(M) (M ^ G). An absent
// partner enters with its implicit value: 0 for an operator, and for a state its initial score.
template <size_t N>
auto reference_propagate(RetainedMap map,
                         const std::vector<Monomial<N>> &gens,
                         const VecD &mapped,
                         bool schrodinger,
                         Basis basis,
                         const VecZ &initial_state) -> RetainedMap {
    const auto mask = initial_state_mask<N>(initial_state);
    const size_t n = gens.size();
    for (size_t step = 0; step < n; ++step) {
        const size_t idx = schrodinger ? step : n - 1 - step;
        const auto &gen = gens[idx];
        const double angle = schrodinger ? -mapped[idx] : mapped[idx];
        const double c = std::cos(2 * angle);
        const double sn = std::sin(2 * angle);
        with_algebra<N>(basis, [&]<typename A>() {
            const auto fold = A::fold_generator(gen);
            const bool odd = A::fold_needs_odd_correction(gen);
            const auto ctx = A::make_gen_context(gen);
            const auto anti = [&](const Monomial<N> &m) {
                return ((m & fold).count() + (odd ? m.count() : 0)) % 2 == 1;
            };
            // Only a state's paired partner has a nonzero implicit value; a zero one would change no sum (up to the
            // sign of a zero).
            if (schrodinger) {
                std::vector<Monomial<N>> partners;
                for (const auto &[key, v] : map) {
                    const auto m = monoprop::detail::generator_from_words<N>(key);
                    if (anti(m) && is_paired<N>(m ^ gen) && !map.contains(key_of<N>(m ^ gen))) {
                        partners.push_back(m ^ gen);
                    }
                }
                for (const auto &m : partners) {
                    map.emplace(key_of<N>(m), algebra_state_phase<N>(basis, m, mask));
                }
            }
            RetainedMap next;
            for (const auto &[key, v] : map) {
                const auto m = monoprop::detail::generator_from_words<N>(key);
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

} // namespace sharded_test
