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
#include "monoprop/detail/sharded/Construction.h"

/*
 * Direct-buffer construction and graph-free propagation over T shards of one process (detail/sharded/Construction.h).
 * The team size T is the captured launch budget; cpp/tests/CMakeLists.txt reruns these cases in fresh processes at
 * T = 1, 2 and 4. Cases that need a nonprimary worker are left out of T = 1.
 *
 * Oracles:
 * - The legacy partition facade (partitions = T over an in-process communicator; child t is flat owner t at geometry
 *   (1, T)), or at T = 1 the single-store propagator. Shared phases mean the two paths can agree on a shared bug, so:
 * - an independent insertion-order reference (plain sets and vectors, no engine code) for rows and IDs;
 * - an independent coefficient-map propagator, and the frozen exact energy of tests/data/random_exact.msgpack;
 * - across T, the global retained map of the in-process T = 1 store, compared with the plan's map tolerance.
 *
 * Workers never call BOOST_TEST. Observers write per-shard slots (visits, streams, kernel ranges) and assertions run
 * after the team has joined. AllocationProbe.h injects real allocation failures on a chosen owner.
 */

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "AllocationProbe.h"
#include "KernelTestSupport.h"
#include "PauliTestOracle.h"
#include "PropagatorTestAccess.h"
#include "TestUtilities.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"
#include "monoprop/detail/sharded/State.h"
#include "monoprop/detail/sharded/Team.h"

namespace {

using namespace monoprop;
namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;
namespace allocation = test_utils::allocation;
using monoprop::detail::KernelRange;
using W = sharded::ConstructionWork;
using Key = std::vector<uint64_t>;

constexpr size_t kN = 8;
constexpr double kExactAtol = 1e-9; // fused_cos_sweep_tests' exactness tolerance against the frozen energy

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

auto bits(TermIndex v) -> uint64_t {
    return v;
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

// A basis change that permutes the slots pairwise, so the opaque closure really differs from the plain cutoff.
auto swap_basis(size_t logical) -> std::vector<VecZ> {
    std::vector<VecZ> rows;
    for (size_t i = 0; i < 2 * logical; ++i) {
        rows.push_back({i ^ 1U});
    }
    return rows;
}

// Legacy comparisons: every basis and picture, cutoffs that trim, coefficient cutoffs, a length cap (two-pass
// fallback) and the basis-change closure.
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

// --- Legacy oracle ----------------------------------------------------------------------------------------------

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
template <size_t N>
auto legacy(const Fixture &f, size_t partitions, MPI_Comm comm = MPI_COMM_SELF) -> MonomialPropagator<N> {
    return MonomialPropagator<N>(f.op,
                                 f.cutoff,
                                 f.initial_state,
                                 f.schrodinger_cutoff,
                                 comm,
                                 f.lower_atol,
                                 f.upper_atol,
                                 f.cutoff_type,
                                 f.basis_change,
                                 f.logical,
                                 f.basis,
                                 partitions);
}

template <size_t N>
auto legacy_owner(MonomialPropagator<N> &p, size_t shard) -> const MonomialPropagator<N> & {
    using Access = monoprop::detail::PropagatorTestAccess<N>;
    return Access::partition_count(p) == 0 ? p : Access::partition(p, static_cast<int>(shard));
}
#endif

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

// Shards seeded through the real S1 path, with the context that goes with them.
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
                .basis = f.basis,
                .schrodinger = f.schrodinger_cutoff.has_value(),
                .world = world};
    }
};

template <size_t N>
auto seed(const Fixture &f, parallel::Options options, const sharded::PhysicalWorld &world = {}) -> Sharded<N> {
    const auto router = routing::make_router<N>(world.ranks, static_cast<size_t>(options.threads));
    const auto cutoff_fn = make_cutoff_fn<N>(f);
    std::optional<sharded::PairedBasisBounds> paired;
    if (f.schrodinger_cutoff) {
        paired = sharded::paired_basis_bounds(*f.schrodinger_cutoff, f.logical, router.flat_world());
    }
    // The legacy constructor sizes rows from the plain cutoff (the basis change never affects the width).
    const auto width_fn = monoprop::detail::cutoff_function<N>(f.cutoff_type, f.cutoff, f.logical);
    const auto seed_inputs = sharded::OperatorSeed<N>{
        .initial_operator = f.op,
        .initial_state = f.initial_state,
        .router = router,
        .basis = f.basis,
        .logical_num_modes = f.logical,
        .paired = paired,
        .inline_width = sharded::packed_inline_width<N>(paired.has_value(), f.basis_change ? cutoff_fn : width_fn)};
    return {.f = f,
            .cutoff_fn = cutoff_fn,
            .router = router,
            .shards = sharded::seed_shards(options, seed_inputs, world.rank),
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

template <size_t N, class Observer = sharded::NoConstructionObserver>
auto run_graph(Sharded<N> &s, const Circuit &c, std::optional<size_t> k, const Observer &observer = {})
    -> sharded::ConstructionOutcome {
    const auto gens = generators<N>(s.f, c);
    VecZ gate_indices(c.gates.size());
    std::iota(gate_indices.begin(), gate_indices.end(), size_t{0});
    const auto ctx = s.ctx();
    return sharded::build_graph<N>(team_options(),
                                   s.shards,
                                   ctx,
                                   {.generators = gens,
                                    .parameter_mapping = c.mapping,
                                    .gen_coeffs = c.gen_coeffs,
                                    .gate_indices = gate_indices,
                                    .only_rotate_len_k = k},
                                   observer);
}

template <size_t N, class Observer = sharded::NoConstructionObserver>
auto run_propagate(Sharded<N> &s, const Circuit &c, std::optional<size_t> k, const Observer &observer = {})
    -> sharded::ConstructionOutcome {
    const auto gens = generators<N>(s.f, c);
    const auto mapped = map_params(c.params, c.mapping, c.gen_coeffs, 1.0);
    const auto ctx = s.ctx();
    return sharded::propagate<N>(team_options(),
                                 s.shards,
                                 ctx,
                                 {.generators = gens, .mapped_params = mapped, .only_rotate_len_k = k},
                                 observer);
}

auto require_success(const sharded::ConstructionOutcome &outcome) -> void {
    if (outcome.error) {
        try {
            std::rethrow_exception(outcome.error);
        }
        catch (const std::exception &e) {
            BOOST_FAIL("the construction failed: " << e.what());
        }
    }
}

// --- Exact shard digests ----------------------------------------------------------------------------------------

struct LayerDigest {
    std::vector<std::vector<uint64_t>> occupied; // (slot, sin_send_count, in_count)
    std::vector<TermIndex> sin_send;
    bool binary = false;
    size_t phase_count = 0;
    std::vector<uint64_t> phase_words;
    std::vector<int8_t> phase_values;
    size_t world = 0;
    size_t self_pos = 0;
    size_t self_offset = 0;
    std::vector<uint64_t> generator;
    uint64_t scaled_count = 0;
    size_t param_index = 0;
    uint64_t gen_coeff = 0;
    size_t gate_index = 0;
    auto operator==(const LayerDigest &) const -> bool = default;
};

struct ShardDigest {
    std::vector<Key> rows;
    std::vector<uint64_t> op_coeffs;
    std::vector<uint64_t> state_coeffs;
    std::vector<TermIndex> state_rows;
    std::vector<uint64_t> state_vals;
    std::vector<std::pair<Key, uint64_t>> pending;
    size_t index_rows = 0;
    std::vector<LayerDigest> layers;
    auto operator==(const ShardDigest &) const -> bool = default;
};

auto layer_digest(const LayerCore &core) -> LayerDigest {
    LayerDigest d;
    for (const auto &e : core.cross_rank.occupied) {
        d.occupied.push_back({e.slot, e.sin_send_count, e.in_count});
    }
    d.sin_send = core.cross_rank.sin_send_indices;
    d.binary = core.cross_rank.sin_recv_phases.uses_binary_phases;
    d.phase_count = core.cross_rank.sin_recv_phases.total_count;
    d.phase_words = core.cross_rank.sin_recv_phases.phase_words;
    d.phase_values = core.cross_rank.sin_recv_phases.phase_values;
    d.world = core.cross_rank.world_size;
    d.self_pos = core.cross_rank.self_pos;
    d.self_offset = core.cross_rank.self_offset;
    d.generator = core.generator_words;
    d.scaled_count = core.scaled_count;
    d.param_index = core.param_index;
    d.gen_coeff = bits(core.gen_coeff);
    d.gate_index = core.gate_index;
    return d;
}

template <size_t N>
auto digest(const sharded::ShardState<N> &s) -> ShardDigest {
    ShardDigest d;
    for (size_t i = 0; i < s.op.store->size(); ++i) {
        d.rows.push_back(key_of<N>(s.op.store->row(i)));
    }
    for (const double v : s.op.op_coeffs) {
        d.op_coeffs.push_back(bits(v));
    }
    for (const double v : s.op.state_coeffs) {
        d.state_coeffs.push_back(bits(v));
    }
    d.state_rows = s.op.state_rows_;
    for (const double v : s.op.state_vals_) {
        d.state_vals.push_back(bits(v));
    }
    for (const auto &[m, v] : s.op.init_op_map) {
        d.pending.emplace_back(key_of<N>(m), bits(v));
    }
    std::ranges::sort(d.pending);
    d.index_rows = s.op.inverted_index_ ? s.op.inverted_index_->rows() : 0;
    for (size_t l = 0; l < s.graph.layers(); ++l) {
        d.layers.push_back(layer_digest(s.graph.get_layer(l).core()));
    }
    return d;
}

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
// The rank-level reference at geometry (1, T) in a candidate build: the candidate root over MPI_COMM_SELF, built
// through its public operations at the launch's T. Its shards come from the same seam, so a case that compares against
// it checks the root's wiring of the seam, and keeps its independent checks; the legacy build compares against the
// legacy partitions instead.
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

template <size_t N>
auto reference_digest(MonomialPropagator<N> &p, size_t shard) -> ShardDigest {
    return digest<N>(*monoprop::detail::PropagatorTestAccess<N>::shards(p).at(shard));
}
#else
template <size_t N>
auto legacy_digest(MonomialPropagator<N> &p, size_t shard) -> ShardDigest {
    return digest<N>(*monoprop::detail::PropagatorTestAccess<N>::shard_state(legacy_owner(p, shard)));
}

// The rank-level reference at geometry (1, T): the legacy partitions.
template <size_t N>
auto reference_root(const Fixture &f, size_t threads) -> MonomialPropagator<N> {
    return legacy<N>(f, threads);
}

template <size_t N>
auto reference_digest(MonomialPropagator<N> &p, size_t shard) -> ShardDigest {
    return legacy_digest<N>(p, shard);
}
#endif

// --- Global retained maps ---------------------------------------------------------------------------------------

using RetainedMap = std::map<Key, double>;

template <size_t N>
auto picture_map(const sharded::ShardState<N> &s, bool schrodinger, RetainedMap &out) -> void {
    const VecD &c = schrodinger ? s.op.state_coeffs : s.op.op_coeffs;
    BOOST_REQUIRE_EQUAL(c.size(), s.op.store->size());
    for (size_t i = 0; i < s.op.store->size(); ++i) {
        const bool fresh = out.emplace(key_of<N>(s.op.store->row(i)), c[i]).second;
        BOOST_REQUIRE_MESSAGE(fresh, "a retained key is owned twice");
    }
}

template <size_t N>
auto global_map(const sharded::Shards<N> &shards, bool schrodinger) -> RetainedMap {
    RetainedMap out;
    for (const auto &s : shards) {
        picture_map<N>(*s, schrodinger, out);
    }
    return out;
}

// The plan's general map tolerance: identical keys, abs(a - b) <= 1e-9 + 1e-7 * max(|a|, |b|).
auto check_maps_agree(const RetainedMap &got, const RetainedMap &want, const std::string &what) -> void {
    BOOST_TEST_CONTEXT(what) {
        BOOST_TEST(got.size() == want.size());
        size_t missing = 0;
        size_t off = 0;
        for (const auto &[key, value] : want) {
            const auto it = got.find(key);
            if (it == got.end()) {
                ++missing;
                continue;
            }
            const double scale = std::max(std::abs(value), std::abs(it->second));
            if (!(std::abs(value - it->second) <= 1e-9 + (1e-7 * scale))) {
                ++off;
            }
        }
        BOOST_TEST(missing == 0U);
        BOOST_TEST(off == 0U);
    }
}

// --- Observation ------------------------------------------------------------------------------------------------

struct Visit {
    W work;
    size_t step;
    int worker;
    int team;
    int level;
    std::thread::id thread;
};

struct PassLog {
    bool leaders = true;
    std::vector<std::vector<uint64_t>> payload;
    std::vector<std::vector<uint64_t>> sources;
    std::vector<std::vector<uint64_t>> values;
    std::vector<std::vector<uint64_t>> answers;
    std::vector<std::vector<uint64_t>> received;
    auto operator==(const PassLog &) const -> bool = default;
};

struct ShardLog {
    std::vector<Visit> visits;
    std::vector<PassLog> passes;
    kernel_test::PhaseLog ranges;
};

struct KernelFailure {
    size_t shard = 0;
    KernelRange kind = KernelRange::decode;
    size_t call = 0;
    size_t range = 0;
};

// The per-shard range observer: kernel ranges, and the streams of the exchange phases.
struct KernelProbe {
    ShardLog *log = nullptr;
    bool capture = true;
    std::optional<KernelFailure> fail = std::nullopt;
    std::optional<bool> fail_publication = std::nullopt; // throw once this pass's payload is built

    auto prepare(KernelRange kind, size_t ranges) const -> void {
        kernel_test::AccumulatingObserver{&log->ranges}.prepare(kind, ranges);
    }
    // An armed failure fires in range `fail->range` of the `fail->call`-th call of its kind that has any range.
    auto visit(KernelRange kind, size_t range) const -> void {
        kernel_test::AccumulatingObserver{&log->ranges}.visit(kind, range);
        if (fail && kind == fail->kind && range == fail->range) {
            const auto calls = log->ranges.of(kind);
            const auto nonempty = std::ranges::count_if(calls, [](const auto *call) { return !call->worker.empty(); });
            if (static_cast<size_t>(nonempty) == fail->call + 1) {
                throw std::runtime_error("injected kernel failure");
            }
        }
    }
    template <class T>
    static auto flatten(const mpi::WindowVec<T> &blocks) -> std::vector<std::vector<uint64_t>> {
        std::vector<std::vector<uint64_t>> out;
        for (const auto &block : blocks) {
            std::vector<uint64_t> words;
            for (const auto v : block) {
                words.push_back(bits(v));
            }
            out.push_back(std::move(words));
        }
        return out;
    }
    static auto flatten_ids(const mpi::WindowVec<VecZ> &blocks) -> std::vector<std::vector<uint64_t>> {
        std::vector<std::vector<uint64_t>> out;
        for (const auto &block : blocks) {
            out.emplace_back(block.begin(), block.end());
        }
        return out;
    }
    auto exchange_prepared(bool leaders,
                           const mpi::WindowVec<VecZ> &payload,
                           const mpi::WindowVec<std::vector<size_t>> &sources,
                           const mpi::WindowVec<std::vector<double>> &values) const -> void {
        if (capture) {
            PassLog pass;
            pass.leaders = leaders;
            pass.payload = flatten_ids(payload);
            pass.sources = flatten_ids(sources);
            pass.values = flatten(values);
            log->passes.push_back(std::move(pass));
        }
        if (fail_publication == leaders) {
            throw std::runtime_error("injected publication failure");
        }
    }
    template <class R>
    auto exchange_resolved(bool /*leaders*/, const mpi::WindowVec<std::vector<R>> &answers) const -> void {
        if (capture) {
            log->passes.back().answers = flatten(answers);
        }
    }
    template <class R>
    auto exchange_consumed(bool /*leaders*/, const mpi::WindowVec<std::span<const R>> &received) const -> void {
        if (capture) {
            log->passes.back().received = flatten(received);
        }
    }
};

struct FailAt {
    W work;
    size_t step;
    size_t shard;
};

struct AllocAt {
    W work;
    size_t step;
    size_t shard;
    size_t nth;
};

// The construction observer: per-shard visit logs, and optional injected failures.
struct Probe {
    std::vector<ShardLog> *logs = nullptr;
    bool capture = true;
    std::vector<FailAt> fail = {};
    std::optional<AllocAt> alloc = std::nullopt;
    std::optional<KernelFailure> kernel_fail = std::nullopt;
    std::optional<std::pair<size_t, bool>> publication_fail = std::nullopt; // (shard, leaders)

    auto visit(W work, size_t step, size_t shard) const -> void {
        // A pending armed allocation failure never leaks into a later phase.
        (void)allocation::disarm_failure();
        auto &log = (*logs)[shard];
        log.visits.push_back(
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
    [[nodiscard]] auto kernels(size_t shard) const -> KernelProbe {
        KernelProbe probe{.log = &(*logs)[shard], .capture = capture};
        if (kernel_fail && kernel_fail->shard == shard) {
            probe.fail = kernel_fail;
        }
        if (publication_fail && publication_fail->first == shard) {
            probe.fail_publication = publication_fail->second;
        }
        return probe;
    }
};

auto make_logs(size_t threads) -> std::vector<ShardLog> {
    std::vector<ShardLog> logs(threads);
    for (auto &log : logs) {
        log.visits.reserve(4096);
    }
    return logs;
}

// After an injected allocation failure, no worker thread keeps an armed failure.
auto disarm_team() -> void {
    static_cast<void>(sharded::run_team(team_options(), [](size_t, sharded::TeamFailure &) noexcept {
        (void)allocation::disarm_failure();
    }));
}

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Legacy streams, captured gate by gate on every legacy store with the same per-shard probe.
template <size_t N>
auto legacy_streams(const Case &cs, size_t threads, bool fused) -> std::vector<ShardLog> {
    using Access = monoprop::detail::PropagatorTestAccess<N>;
    auto p = legacy<N>(cs.f, threads);
    auto logs = make_logs(threads);
    const auto mapped = map_params(cs.c.params, cs.c.mapping, cs.c.gen_coeffs, 1.0);
    const bool schrodinger = cs.f.schrodinger_cutoff.has_value();
    const size_t n = cs.c.gates.size();
    for (size_t step = 0; step < n; ++step) {
        const size_t idx = schrodinger ? step : n - 1 - step;
        Access::for_each_store(p, [&](size_t r, MonomialPropagator<N> &store) {
            const KernelProbe probe{.log = &logs[r]};
            if (fused) {
                Access::contract_gate_observed(store, cs.c.gates[idx], cs.k, mapped[idx], probe);
            }
            else {
                (void)Access::graph_gate_observed(store, cs.c.gates[idx], cs.k, probe);
            }
        });
    }
    return logs;
}
#endif

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Whether any pass of any shard published a nonempty payload block, so a stream comparison is not vacuous.
auto any_payload(const std::vector<ShardLog> &logs) -> bool {
    for (const auto &log : logs) {
        for (const auto &pass : log.passes) {
            if (std::ranges::any_of(pass.payload, [](const auto &block) { return !block.empty(); })) {
                return true;
            }
        }
    }
    return false;
}
#endif

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
auto passes_equal(const std::vector<ShardLog> &got, const std::vector<ShardLog> &want) -> bool {
    if (got.size() != want.size()) {
        return false;
    }
    for (size_t t = 0; t < got.size(); ++t) {
        if (got[t].passes != want[t].passes) {
            return false;
        }
    }
    return true;
}
#endif

} // namespace

// --- Fixed-geometry equality with the legacy partitions ---------------------------------------------------------

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Graph construction at (1, T): every shard's rows and IDs, coefficients, caches and graph layers (endpoints, signs,
// generator words, scaled_count, gate metadata) equal the legacy child's, and so do the query, source and answer
// streams of every pass of every gate.
BOOST_AUTO_TEST_CASE(sharded_construction_graph_matches_legacy_partitions) {
    const auto options = team_options();
    const size_t threads = team_size();
    for (const auto &cs : legacy_cases()) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto p = legacy<kN>(cs.f, threads);
            p.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
            auto s = seed<kN>(cs.f, options);
            const size_t seeded = sharded::total_size(s.shards);
            auto logs = make_logs(threads);
            require_success(run_graph<kN>(s, cs.c, cs.k, Probe{.logs = &logs}));
            BOOST_TEST(sharded::total_size(s.shards) > seeded);
            BOOST_TEST((threads == 1 || any_payload(logs)));
            size_t layers = 0;
            for (size_t t = 0; t < threads; ++t) {
                const auto got = digest<kN>(*s.shards[t]);
                BOOST_TEST((got == legacy_digest<kN>(p, t)), "shard " << t);
                layers += got.layers.size();
            }
            BOOST_TEST(layers == threads * cs.c.gates.size());
            BOOST_TEST(passes_equal(logs, legacy_streams<kN>(cs, threads, false)));
        }
    }
}
#endif

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Graph-free propagation at (1, T): every shard's coefficients equal the legacy child's bit for bit (zeros and
// near-cutoff rows included), as do rows, caches and the fused query/value, source and answer streams.
BOOST_AUTO_TEST_CASE(sharded_construction_propagation_matches_legacy_partitions) {
    const auto options = team_options();
    const size_t threads = team_size();
    for (const auto &cs : legacy_cases()) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto p = legacy<kN>(cs.f, threads);
            p.propagate(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, cs.c.params, cs.k);
            auto s = seed<kN>(cs.f, options);
            const bool schrodinger = cs.f.schrodinger_cutoff.has_value();
            auto before = sharded::copy_shards(options, s.shards);
            for (auto &state : before) {
                (void)state->op.current_picture(schrodinger);
            }
            const auto initial = global_map<kN>(before, schrodinger);
            auto logs = make_logs(threads);
            require_success(run_propagate<kN>(s, cs.c, cs.k, Probe{.logs = &logs}));
            BOOST_TEST((threads == 1 || any_payload(logs)));
            // The coefficients evolved and the store grew: equality below is not between untouched states.
            BOOST_TEST(global_map<kN>(s.shards, schrodinger).size() > initial.size());
            for (size_t t = 0; t < threads; ++t) {
                BOOST_TEST((digest<kN>(*s.shards[t]) == legacy_digest<kN>(p, t)), "shard " << t);
                BOOST_TEST(s.shards[t]->graph.layers() == 0U);
            }
            BOOST_TEST(passes_equal(logs, legacy_streams<kN>(cs, threads, true)));
        }
    }
}
#endif

// --- Independent references -------------------------------------------------------------------------------------

namespace {

// The ordering contract, restated without engine code: per gate, each destination appends cross-owner leader
// misses (ascending source shard, source row order), then cross-owner follower misses (followers a leader matched
// are dropped), then its deferred same-shard leader misses, then its same-shard follower misses.
template <size_t N>
struct OrderReference {
    std::vector<std::vector<Monomial<N>>> rows;     // per shard, in row order
    std::map<Key, std::pair<size_t, size_t>> where; // key -> (shard, row)
    size_t cross_matches = 0;                       // leader hits that marked a row of another shard
    size_t self_misses = 0;
    size_t cross_misses = 0;

    auto append(size_t shard, const Monomial<N> &m) -> void {
        where.emplace(key_of<N>(m), std::make_pair(shard, rows[shard].size()));
        rows[shard].push_back(m);
    }

    auto gate(const Monomial<N> &gen,
              const routing::Router &router,
              Basis basis,
              const std::function<bool(const Monomial<N> &)> &cutoff) -> void {
        const size_t threads = rows.size();
        const auto [fold, odd] = with_algebra<N>(basis, [&]<typename A>() {
            return std::pair{A::fold_generator(gen), A::fold_needs_odd_correction(gen)};
        });
        if (!gen.any()) {
            return;
        }
        const size_t pivot = gen.find_first();
        struct Query {
            size_t src;
            Monomial<N> partner;
            size_t dest;
        };
        // [source][dest] leaders and followers, in source row order.
        std::vector<std::vector<std::vector<Query>>> lead(threads, std::vector<std::vector<Query>>(threads));
        auto foll = lead;
        for (size_t s = 0; s < threads; ++s) {
            for (size_t i = 0; i < rows[s].size(); ++i) {
                const auto &m = rows[s][i];
                const size_t overlap = (m & fold).count() + (odd ? m.count() : 0);
                if (overlap % 2 == 0) {
                    continue;
                }
                const auto partner = m ^ gen;
                if (!cutoff(partner)) {
                    continue;
                }
                const size_t d = router.dest<N>(partner);
                (m.test(pivot) ? foll : lead)[s][d].push_back({i, partner, d});
            }
        }
        const auto pre = rows;
        const auto pre_hit = [&](size_t d, const Monomial<N> &m) -> std::optional<size_t> {
            const auto it = where.find(key_of<N>(m));
            if (it == where.end() || it->second.first != d || it->second.second >= pre[d].size()) {
                return std::nullopt;
            }
            return it->second.second;
        };
        std::vector<std::vector<bool>> marked(threads);
        for (size_t d = 0; d < threads; ++d) {
            marked[d].assign(pre[d].size(), false);
        }
        std::vector<std::vector<Monomial<N>>> self_lead_miss(threads);
        std::vector<std::vector<Monomial<N>>> self_foll_miss(threads);
        // Leaders: same-shard resolution is frozen (deferred misses), cross-owner misses insert at once.
        for (size_t d = 0; d < threads; ++d) {
            for (const auto &q : lead[d][d]) {
                if (const auto hit = pre_hit(d, q.partner)) {
                    marked[d][*hit] = true;
                }
                else {
                    self_lead_miss[d].push_back(q.partner);
                }
            }
        }
        for (size_t d = 0; d < threads; ++d) {
            for (size_t s = 0; s < threads; ++s) {
                if (s == d) {
                    continue;
                }
                for (const auto &q : lead[s][d]) {
                    if (const auto hit = pre_hit(d, q.partner)) {
                        marked[d][*hit] = true;
                        ++cross_matches;
                    }
                    else {
                        append(d, q.partner);
                        ++cross_misses;
                    }
                }
            }
        }
        // Followers: filtered by the source shard's completed marks.
        for (size_t d = 0; d < threads; ++d) {
            for (const auto &q : foll[d][d]) {
                if (!marked[d][q.src] && !pre_hit(d, q.partner)) {
                    self_foll_miss[d].push_back(q.partner);
                }
            }
        }
        for (size_t d = 0; d < threads; ++d) {
            for (size_t s = 0; s < threads; ++s) {
                if (s == d) {
                    continue;
                }
                for (const auto &q : foll[s][d]) {
                    if (!marked[s][q.src] && !pre_hit(d, q.partner)) {
                        append(d, q.partner);
                        ++cross_misses;
                    }
                }
            }
        }
        for (size_t d = 0; d < threads; ++d) {
            for (const auto &m : self_lead_miss[d]) {
                append(d, m);
                ++self_misses;
            }
            for (const auto &m : self_foll_miss[d]) {
                append(d, m);
                ++self_misses;
            }
        }
    }
};

} // namespace

// Rows and IDs per shard follow the ordering contract as an independent reference computes it, including followers
// matched by a leader of another shard (which must occur at T >= 2 on this fixture).
BOOST_AUTO_TEST_CASE(sharded_construction_rows_follow_the_ordering_contract) {
    const auto options = team_options();
    const size_t threads = team_size();
    auto cases = legacy_cases();
    cases.resize(2);                    // Majorana, both pictures
    cases.push_back(legacy_cases()[9]); // Pauli Heisenberg
    size_t cross_matches = 0;
    for (const auto &cs : cases) {
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto s = seed<kN>(cs.f, options);
            OrderReference<kN> ref;
            ref.rows.resize(threads);
            for (size_t t = 0; t < threads; ++t) {
                for (size_t i = 0; i < s.shards[t]->op.store->size(); ++i) {
                    ref.append(t, s.shards[t]->op.store->row(i));
                }
            }
            const auto gens = generators<kN>(cs.f, cs.c);
            const auto plain = monoprop::detail::cutoff_function<kN>(cs.f.cutoff_type, cs.f.cutoff, cs.f.logical);
            const bool schrodinger = cs.f.schrodinger_cutoff.has_value();
            for (size_t step = 0; step < gens.size(); ++step) {
                ref.gate(gens[schrodinger ? step : gens.size() - 1 - step], s.router, cs.f.basis, plain);
            }
            require_success(run_graph<kN>(s, cs.c, cs.k));
            for (size_t t = 0; t < threads; ++t) {
                std::vector<Key> got;
                std::vector<Key> want;
                for (size_t i = 0; i < s.shards[t]->op.store->size(); ++i) {
                    got.push_back(key_of<kN>(s.shards[t]->op.store->row(i)));
                }
                for (const auto &m : ref.rows[t]) {
                    want.push_back(key_of<kN>(m));
                }
                BOOST_TEST((got == want), "shard " << t);
            }
            BOOST_TEST(ref.self_misses > 0U);
            if (threads >= 2) {
                BOOST_TEST(ref.cross_misses > 0U);
            }
            cross_matches += ref.cross_matches;
        }
    }
    // A follower dropped because a leader of another shard matched it.
    BOOST_TEST((threads < 2 || cross_matches > 0U));
}

namespace {

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
    const auto implicit = [&](const Monomial<N> &m) -> double {
        if (!schrodinger) {
            return 0.0;
        }
        for (size_t mode = 0; mode < N; ++mode) {
            if (m.test(2 * mode) != m.test((2 * mode) + 1)) {
                return 0.0;
            }
        }
        return algebra_state_phase<N>(basis, m, mask);
    };
    const auto mono_of = [](const Key &key) {
        Monomial<N> m;
        for (size_t w = 0; w < key.size(); ++w) {
            m.data()[w] = key[w];
        }
        return m;
    };
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
            std::vector<Monomial<N>> partners;
            for (const auto &[key, v] : map) {
                const auto m = mono_of(key);
                if (anti(m) && !map.contains(key_of<N>(m ^ gen))) {
                    partners.push_back(m ^ gen);
                }
            }
            for (const auto &m : partners) {
                map.emplace(key_of<N>(m), implicit(m));
            }
            RetainedMap next;
            for (const auto &[key, v] : map) {
                const auto m = mono_of(key);
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

// <psi| O |psi> from shard coefficients: Heisenberg sums the evolved operator against the initial state's scores;
// Schrödinger sums the evolved state against the Hamiltonian's rows. The identity term is added once.
template <size_t N>
auto shard_energy(sharded::Shards<N> &shards, bool schrodinger, double core) -> double {
    auto energy = core;
    for (auto &s : shards) {
        auto op = s->op; // the dense oracles mutate caches
        if (schrodinger) {
            const auto &h = op.get_operator();
            for (size_t i = 0; i < op.size(); ++i) {
                energy += op.state_coeffs[i] * h[i];
            }
        }
        else {
            const auto state = op.materialize_state();
            for (size_t i = 0; i < op.size(); ++i) {
                energy += op.op_coeffs[i] * state[i];
            }
        }
    }
    return energy;
}

} // namespace

// Exact-cutoff propagation against the independent map propagator (both pictures, both bases, odd Majorana
// generators, native Pauli folding), and the frozen exact energy of random_exact in both pictures.
BOOST_AUTO_TEST_CASE(sharded_construction_propagation_matches_an_independent_reference) {
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
    for (const auto &[f, c] : cases) {
        const bool schrodinger = f.schrodinger_cutoff.has_value();
        BOOST_TEST_CONTEXT(f.name << (schrodinger ? " schrodinger" : " heisenberg") << " T=" << threads) {
            auto s = seed<kN>(f, options);
            for (auto &state : s.shards) {
                (void)state->op.current_picture(schrodinger);
            }
            const auto initial = global_map<kN>(s.shards, schrodinger);
            require_success(run_propagate<kN>(s, c, std::nullopt));
            const auto mapped = map_params(c.params, c.mapping, c.gen_coeffs, 1.0);
            const auto want =
                reference_propagate<kN>(initial, generators<kN>(f, c), mapped, schrodinger, f.basis, f.initial_state);
            check_maps_agree(global_map<kN>(s.shards, schrodinger), want, "reference map");
            if (f.name == "random-exact") {
                const auto core = sharded::validate_initial_operator<kN>(f.op, f.basis, f.logical);
                BOOST_CHECK_SMALL(shard_energy<kN>(s.shards, schrodinger, core) - data.actual_expval, kExactAtol);
            }
        }
    }
}

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// Across launches the geometry changes, so raw row IDs may differ; the global retained maps (keys, coefficients,
// stored zeros) must agree with the in-process single store, and every graph layer must record the same number of
// rotations.
BOOST_AUTO_TEST_CASE(sharded_construction_maps_agree_across_team_sizes) {
    const auto options = team_options();
    const size_t threads = team_size();
    for (const auto &cs : legacy_cases()) {
        const bool schrodinger = cs.f.schrodinger_cutoff.has_value();
        BOOST_TEST_CONTEXT(label(cs) << " T=" << threads) {
            auto single = legacy<kN>(cs.f, 1);
            single.propagate(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, cs.c.params, cs.k);
            RetainedMap want;
            picture_map<kN>(*monoprop::detail::PropagatorTestAccess<kN>::shard_state(single), schrodinger, want);
            auto s = seed<kN>(cs.f, options);
            require_success(run_propagate<kN>(s, cs.c, cs.k));
            check_maps_agree(global_map<kN>(s.shards, schrodinger), want, "propagated map");

            auto single_graph = legacy<kN>(cs.f, 1);
            single_graph.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
            const auto one = monoprop::detail::PropagatorTestAccess<kN>::shard_state(single_graph);
            auto g = seed<kN>(cs.f, options);
            require_success(run_graph<kN>(g, cs.c, cs.k));
            std::set<Key> got_keys;
            std::set<Key> want_keys;
            for (const auto &state : g.shards) {
                for (size_t i = 0; i < state->op.store->size(); ++i) {
                    got_keys.insert(key_of<kN>(state->op.store->row(i)));
                }
            }
            for (size_t i = 0; i < one->op.store->size(); ++i) {
                want_keys.insert(key_of<kN>(one->op.store->row(i)));
            }
            BOOST_TEST((got_keys == want_keys));
            for (size_t l = 0; l < one->graph.layers(); ++l) {
                size_t entries = 0;
                for (const auto &state : g.shards) {
                    entries += state->graph.get_layer(l).core().cross_rank.sin_send_indices.size();
                }
                BOOST_TEST(entries == one->graph.get_layer(l).core().cross_rank.sin_send_indices.size(), "layer " << l);
            }
        }
    }
}
#endif

// --- Participation and the opaque path --------------------------------------------------------------------------

namespace {

auto check_owner_participation(const std::vector<ShardLog> &logs, size_t gates, bool opaque, const std::string &what)
    -> void {
    const size_t threads = logs.size();
    std::set<std::thread::id> threads_seen;
    BOOST_TEST_CONTEXT(what) {
        for (size_t t = 0; t < threads; ++t) {
            const auto &visits = logs[t].visits;
            BOOST_TEST_REQUIRE(!visits.empty());
            BOOST_TEST((visits.front().work == W::frame));
            BOOST_TEST((visits.back().work == W::caches));
            std::set<size_t> steps;
            std::set<std::thread::id> owner_threads;
            bool owner_work = true;
            bool one_team = true;
            for (const auto &v : visits) {
                one_team = one_team && v.team == static_cast<int>(threads) && v.level == 1;
                if (v.step != sharded::kNoStep) {
                    steps.insert(v.step);
                }
                const int expected = (opaque && v.work == W::traverse) ? 0 : static_cast<int>(t);
                owner_work = owner_work && v.worker == expected;
                if (v.worker == static_cast<int>(t)) {
                    owner_threads.insert(v.thread);
                }
            }
            BOOST_TEST(one_team, "shard " << t);
            BOOST_TEST(owner_work, "shard " << t);
            // Every gate of the circuit ran inside the same team, on the same owner thread.
            BOOST_TEST(steps.size() == gates, "shard " << t);
            BOOST_TEST(owner_threads.size() == 1U, "shard " << t);
            threads_seen.insert(owner_threads.begin(), owner_threads.end());
            // The kernels of shard t ran serially on their owner (or, for an opaque traversal, the primary).
            for (const auto &call : logs[t].ranges.calls) {
                const int expected = (opaque && call.kind == KernelRange::scan) ? 0 : static_cast<int>(t);
                for (size_t r = 0; r < call.worker.size(); ++r) {
                    BOOST_TEST(call.worker[r] == expected, "shard " << t << " kind " << static_cast<int>(call.kind));
                    BOOST_TEST(call.level[r] == 1);
                }
            }
        }
        BOOST_TEST(threads_seen.size() == threads);
    }
}

auto count_visits(const std::vector<ShardLog> &logs, W work) -> size_t {
    size_t n = 0;
    for (const auto &log : logs) {
        n += static_cast<size_t>(std::ranges::count_if(log.visits, [&](const Visit &v) { return v.work == work; }));
    }
    return n;
}

auto count_ranges(const std::vector<ShardLog> &logs, KernelRange kind) -> size_t {
    size_t n = 0;
    for (const auto &log : logs) {
        for (const auto &call : log.ranges.of(kind)) {
            n += call->worker.size();
        }
    }
    return n;
}

} // namespace

// Each owner performs its own traversal, resolution/publication, finalization and application, for every gate, inside
// one team of T workers at level 1, on one thread per owner; its kernels run serially on it.
BOOST_AUTO_TEST_CASE(sharded_construction_owners_do_their_own_work) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    for (const bool fused : {false, true}) {
        auto s = seed<kN>(cs.f, options);
        auto logs = make_logs(threads);
        require_success(fused ? run_propagate<kN>(s, cs.c, cs.k, Probe{.logs = &logs})
                              : run_graph<kN>(s, cs.c, cs.k, Probe{.logs = &logs}));
        check_owner_participation(logs, cs.c.gates.size(), false, fused ? "propagate" : "graph");
        const size_t gates = cs.c.gates.size();
        BOOST_TEST(count_visits(logs, W::traverse) == threads * gates);
        BOOST_TEST(count_visits(logs, W::finalize) == threads * gates);
        BOOST_TEST(count_visits(logs, W::apply) == (fused ? threads * gates : 0U));
        // One identity gate: no exchange; with one shard, no cross-owner resolution at all.
        const size_t exchanging = threads > 1 ? threads * (gates - 1) : 0U;
        BOOST_TEST(count_visits(logs, W::resolve_leaders) == exchanging);
        BOOST_TEST(count_visits(logs, W::consume_followers) == exchanging);
        BOOST_TEST(count_ranges(logs, KernelRange::scan) > 0U);
        BOOST_TEST(count_ranges(logs, KernelRange::self_probe) > 0U);
        if (threads > 1) {
            BOOST_TEST(count_ranges(logs, KernelRange::decode) > 0U);
            BOOST_TEST(count_ranges(logs, KernelRange::scatter) > 0U);
        }
        if (fused) {
            BOOST_TEST(count_ranges(logs, KernelRange::fused_apply) > 0U);
        }
    }
}

// Opaque cutoffs (a closure over a typed one, and the basis-change closure) traverse every shard on the primary while
// the owners keep every other phase; the results equal the typed-cutoff run and the legacy partitions.
BOOST_AUTO_TEST_CASE(sharded_construction_opaque_cutoff_traverses_on_the_primary) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cases = legacy_cases();
    for (const auto &base : {cases[0], cases[1], cases[6]}) {
        auto opaque = base;
        opaque.f.opaque = true;
        for (const bool fused : {false, true}) {
            BOOST_TEST_CONTEXT(label(base) << (fused ? " propagate" : " graph") << " T=" << threads) {
                auto s = seed<kN>(opaque.f, options);
                auto logs = make_logs(threads);
                require_success(fused ? run_propagate<kN>(s, opaque.c, opaque.k, Probe{.logs = &logs})
                                      : run_graph<kN>(s, opaque.c, opaque.k, Probe{.logs = &logs}));
                check_owner_participation(logs, opaque.c.gates.size(), true, "opaque");
                // Each traversal names its target shard, and the primary executed all of them.
                for (size_t t = 0; t < threads; ++t) {
                    for (const auto &v : logs[t].visits) {
                        if (v.work == W::traverse) {
                            BOOST_TEST(v.worker == 0);
                        }
                    }
                }
                auto p = reference_root<kN>(base.f, threads);
                if (fused) {
                    p.propagate(base.c.gates, base.c.mapping, base.c.gen_coeffs, base.c.params, base.k);
                }
                else {
                    p.build_graph(base.c.gates, base.c.mapping, base.c.gen_coeffs, std::nullopt, std::nullopt, base.k);
                }
                for (size_t t = 0; t < threads; ++t) {
                    BOOST_TEST((digest<kN>(*s.shards[t]) == reference_digest<kN>(p, t)), "shard " << t);
                }
            }
        }
    }
}

// --- Degenerate inputs ------------------------------------------------------------------------------------------

// An empty circuit opens no team and changes nothing; an empty operator and identity-only circuits leave every shard
// (including empty owners) consistent with the legacy partitions; empty owners still reach every checkpoint.
BOOST_AUTO_TEST_CASE(sharded_construction_empty_operators_circuits_and_identity) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    {
        auto s = seed<kN>(cs.f, options);
        std::vector<ShardDigest> before;
        for (const auto &state : s.shards) {
            before.push_back(digest<kN>(*state));
        }
        auto logs = make_logs(threads);
        const auto outcome = run_graph<kN>(s, Circuit{}, std::nullopt, Probe{.logs = &logs});
        BOOST_TEST(!outcome.error);
        BOOST_TEST(!outcome.mutation_started);
        const auto outcome2 = run_propagate<kN>(s, Circuit{}, std::nullopt, Probe{.logs = &logs});
        BOOST_TEST(!outcome2.mutation_started);
        for (size_t t = 0; t < threads; ++t) {
            BOOST_TEST((digest<kN>(*s.shards[t]) == before[t]));
            BOOST_TEST(logs[t].visits.empty());
        }
    }
    Fixture empty = cs.f;
    empty.name = "empty-operator";
    empty.op = {};
    Fixture two = cs.f;
    two.name = "two-terms";
    two.op = {};
    add_term(two.op, Basis::Majorana, {0, 1}, 0.5);
    add_term(two.op, Basis::Majorana, {2, 3}, -0.25);
    const Circuit identity{.gates = {{}, {}}, .mapping = {0, 1}, .gen_coeffs = {1.0, 1.0}, .params = {0.3, 0.4}};
    for (const auto &f : {empty, two, cs.f}) {
        for (const auto &c : {identity, cs.c}) {
            for (const bool fused : {false, true}) {
                BOOST_TEST_CONTEXT(f.name << " gates=" << c.gates.size() << (fused ? " propagate" : " graph")) {
                    auto s = seed<kN>(f, options);
                    auto logs = make_logs(threads);
                    require_success(fused ? run_propagate<kN>(s, c, std::nullopt, Probe{.logs = &logs})
                                          : run_graph<kN>(s, c, std::nullopt, Probe{.logs = &logs}));
                    auto p = reference_root<kN>(f, threads);
                    if (fused) {
                        p.propagate(c.gates, c.mapping, c.gen_coeffs, c.params);
                    }
                    else {
                        p.build_graph(c.gates, c.mapping, c.gen_coeffs);
                    }
                    for (size_t t = 0; t < threads; ++t) {
                        BOOST_TEST((digest<kN>(*s.shards[t]) == reference_digest<kN>(p, t)), "shard " << t);
                        // Empty owners reach every checkpoint: the frame, every gate's finish, the caches.
                        BOOST_TEST(static_cast<size_t>(
                                       std::ranges::count_if(logs[t].visits,
                                                             [](const Visit &v) { return v.work == W::finalize; }))
                                   == c.gates.size());
                    }
                }
            }
        }
    }
}

// --- The shared span-helper seam --------------------------------------------------------------------------------

namespace {

template <size_t N>
auto make_store(const std::vector<Monomial<N>> &terms) -> monoprop::detail::MPOperator<N> {
    monoprop::detail::MPOperator<N> op;
    if (!terms.empty()) {
        monoprop::detail::insert_absent_terms<N>(
            op,
            terms.size(),
            [&](size_t k) -> const Monomial<N> & { return terms[k]; },
            [&](size_t k, size_t base) { assign_row<N>(*op.store, base + k, terms[k]); });
    }
    return op;
}

// A Majorana term with a coefficient its basis can encode (real or imaginary by its length).
template <size_t N>
auto add_random_term(OperatorDict &op, const Monomial<N> &m, double value) -> void {
    VecZ idx;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        idx.push_back(b);
    }
    op[idx] = algebra_decode_coeff<N>(Basis::Majorana, std::complex<double>(value, 0.0), m);
}

template <size_t N>
auto random_term(std::mt19937_64 &rng, size_t k) -> Monomial<N> {
    Monomial<N> m;
    std::uniform_int_distribution<size_t> slot(0, (2 * N) - 1);
    while (m.count() < k) {
        m.set(slot(rng));
    }
    return m;
}

template <size_t N>
auto encode(const std::vector<Monomial<N>> &keys, bool fused, size_t salt) -> VecZ {
    using QW = monoprop::detail::QueryWire<N>;
    VecZ buf;
    for (size_t q = 0; q < keys.size(); ++q) {
        std::vector<uint16_t> pos;
        for (size_t b = keys[q].find_first(); b < keys[q].size(); b = keys[q].find_next(b)) {
            pos.push_back(static_cast<uint16_t>(b));
        }
        QW::push(buf, pos, static_cast<int>((q + salt) % 3) - 1);
        if (fused) {
            QW::push_value(buf, 0.25 + static_cast<double>(q + salt));
        }
    }
    return buf;
}

// One resolver owner's state after resolving `incoming` through the owning-VecZ path or the published-span path.
template <size_t N, bool Fused>
auto resolve_both(const std::vector<Monomial<N>> &seeds,
                  const mpi::WindowVec<VecZ> &incoming,
                  size_t owner,
                  size_t world) -> std::pair<std::vector<std::vector<uint64_t>>, std::vector<std::vector<uint64_t>>> {
    using Sink = std::conditional_t<Fused, monoprop::detail::ContractSink<N>, monoprop::detail::GraphSink<N>>;
    const auto run = [&](bool spans) {
        auto op = make_store<N>(seeds);
        monoprop::detail::MatchedEpochSet matched;
        monoprop::detail::FusedContract fc;
        VecD coeffs(seeds.size(), 0.5);
        const auto window = incoming.window();
        auto engine = [&] {
            if constexpr (Fused) {
                return monoprop::detail::LayerBuildEngine<N, Sink>(op,
                                                                   world,
                                                                   owner,
                                                                   matched,
                                                                   op.size(),
                                                                   Sink{.R = world,
                                                                        .my_rank = owner,
                                                                        .fc = fc,
                                                                        .op_coeffs = coeffs,
                                                                        .fused_scale = false,
                                                                        .inv_cos = 1.0,
                                                                        .schrodinger = false,
                                                                        .basis = Basis::Majorana},
                                                                   window);
            }
            else {
                return monoprop::detail::LayerBuildEngine<N, Sink>(op,
                                                                   world,
                                                                   owner,
                                                                   matched,
                                                                   op.size(),
                                                                   Sink(world, owner),
                                                                   window);
            }
        }();
        mpi::WindowVec<std::vector<typename Sink::Response>> answers;
        if (spans) {
            answers = engine.resolve_published(mpi::views_of(incoming), true);
        }
        else {
            answers = monoprop::detail::resolve_incoming<N>(incoming, op, true, matched, op.size(), engine.sink);
        }
        std::vector<std::vector<uint64_t>> out;
        for (const auto &block : answers) {
            std::vector<uint64_t> words;
            for (const auto v : block) {
                words.push_back(bits(v));
            }
            out.push_back(std::move(words));
        }
        for (size_t i = 0; i < op.size(); ++i) {
            out.push_back(key_of<N>(op.store->row(i)));
        }
        for (const auto &h : fc.cross_half) {
            out.push_back({h.local_idx, bits(h.v_partner), static_cast<uint64_t>(h.phase_signed), h.is_insert});
        }
        for (size_t i = 0; i < op.size() && i < matched.epoch_.size(); ++i) {
            out.push_back({static_cast<uint64_t>(i < seeds.size() && matched.is_marked(i))});
        }
        return out;
    };
    return {run(false), run(true)};
}

} // namespace

// The published-view helpers keep a nonzero window base, give empty senders and absent publishers empty views, and
// alias rather than copy.
BOOST_AUTO_TEST_CASE(sharded_construction_published_views_cover_the_window) {
    const mpi::SlotWindow window{.base = 2, .count = 3};
    mpi::WindowVec<VecZ> blocks(window);
    blocks.at_slot(2) = {7, 8};
    blocks.at_slot(4) = {9};
    const auto views = mpi::views_of(blocks);
    BOOST_TEST((views.window() == window));
    BOOST_TEST(views.at_slot(2).data() == blocks.at_slot(2).data());
    BOOST_TEST(views.at_slot(3).empty());
    BOOST_TEST(views.at_slot(4).size() == 1U);

    // Owners 2..4 of a process whose first flat slot is 2 publish buffers over windows of their own.
    mpi::WindowVec<VecZ> from2(mpi::SlotWindow{.base = 2, .count = 3});
    from2.at_slot(3) = {1, 2, 3};
    mpi::WindowVec<VecZ> from4(mpi::SlotWindow{.base = 4, .count = 1}); // does not reach owner 3
    from4.at_slot(4) = {5};
    const std::vector<const mpi::WindowVec<VecZ> *> published{&from2, nullptr, &from4};
    const auto gathered = sharded::gather_published<size_t>(mpi::SlotWindow{.base = 1, .count = 5},
                                                            3,
                                                            2,
                                                            std::span<const mpi::WindowVec<VecZ> *const>(published));
    BOOST_TEST((gathered.window() == mpi::SlotWindow{.base = 1, .count = 5}));
    BOOST_TEST(gathered.at_slot(1).empty()); // not a local owner
    BOOST_TEST(gathered.at_slot(2).data() == from2.at_slot(3).data());
    BOOST_TEST(gathered.at_slot(3).empty()); // published nothing
    BOOST_TEST(gathered.at_slot(4).empty()); // its window does not reach owner 3
    BOOST_TEST(gathered.at_slot(5).empty()); // beyond the local owners
}

// resolve_published over read-only views equals resolve_incoming over owning blocks, exactly, for Plain and Fused
// records, at 255/256/257 queries per sender, with an empty sender and a nonzero window base, for narrow (8-mode) and
// wide, escaped, multiword (250-mode) positions; and a malformed Plain stream is rejected before any mutation.
BOOST_AUTO_TEST_CASE(sharded_construction_published_resolution_matches_owning_blocks) {
    const auto check = [&]<size_t N>(size_t min_k, size_t max_k) {
        std::mt19937_64 rng(N + max_k);
        std::vector<Monomial<N>> seeds;
        std::set<Key> used;
        const auto fresh = [&](size_t k) {
            for (;;) {
                const auto m = random_term<N>(rng, k);
                if (used.insert(key_of<N>(m)).second) {
                    return m;
                }
            }
        };
        const auto width = [&](size_t i) { return min_k + (i % (max_k - min_k + 1)); };
        for (size_t i = 0; i < 600; ++i) {
            seeds.push_back(fresh(width(i)));
        }
        // Mixed, all-hit and all-miss senders.
        // Clang cannot capture a structured binding under OpenMP, so the pair is unpacked by name.
        for (const auto &shape : {std::pair{size_t{255}, 0},
                                  std::pair{size_t{256}, 0},
                                  std::pair{size_t{257}, 0},
                                  std::pair{size_t{256}, 1},
                                  std::pair{size_t{257}, 2}}) {
            const size_t count = shape.first;
            const int pattern = shape.second;
            const mpi::SlotWindow window{.base = 3, .count = 3};
            std::vector<std::vector<Monomial<N>>> per_sender(3);
            const auto hit = [&](size_t q, size_t modulus) {
                return pattern == 1 || (pattern == 0 && q % modulus == 0);
            };
            for (size_t q = 0; q < count; ++q) {
                per_sender[0].push_back(hit(q, 2) ? seeds[q] : fresh(width(q)));
                per_sender[2].push_back(hit(q, 3) ? seeds[300 + (q % 290)] : fresh(width(q * 7)));
            }
            // Distinct queries overall, as XOR with a fixed generator guarantees in production.
            std::set<Key> distinct;
            for (auto &sender : per_sender) {
                std::erase_if(sender, [&](const Monomial<N> &m) { return !distinct.insert(key_of<N>(m)).second; });
            }
            for (const bool fused : {false, true}) {
                mpi::WindowVec<VecZ> incoming(window);
                for (size_t k = 0; k < 3; ++k) {
                    incoming[mpi::WindowIndex{k}] = encode<N>(per_sender[k], fused, k);
                }
                // Owner 4 resolves; slot 4 (its own) is the empty sender.
                const auto [owning, viewed] = fused ? resolve_both<N, true>(seeds, incoming, 4, 6)
                                                    : resolve_both<N, false>(seeds, incoming, 4, 6);
                BOOST_TEST((owning == viewed),
                           "N=" << N << " count=" << count << " pattern=" << pattern << " fused=" << fused);
            }
        }
    };
    check.template operator()<kN>(3, 6);
    check.template operator()<250>(1, 40); // positions above 255, escaped counts (k >= 31), multiword keys

    // The retained malformed Plain QueryWire<128> fixture: a header past its one-word stream.
    using Engine = monoprop::detail::LayerBuildEngine<128, monoprop::detail::GraphSink<128>>;
    std::mt19937_64 rng128(3);
    auto op = make_store<128>({random_term<128>(rng128, 4)});
    monoprop::detail::MatchedEpochSet matched;
    const mpi::SlotWindow window{.base = 0, .count = 2};
    Engine engine(op, 2, 0, matched, op.size(), monoprop::detail::GraphSink<128>(2, 0), window);
    mpi::WindowVec<VecZ> malformed(window);
    malformed.at_slot(1) = VecZ{size_t{0x8037e}};
    BOOST_CHECK_THROW((void)engine.resolve_published(mpi::views_of(malformed), true),
                      monoprop::detail::MalformedQueryStream);
    BOOST_TEST(op.size() == 1U);
    // A view over the wrong window is refused before anything is read.
    mpi::WindowVec<VecZ> shifted(mpi::SlotWindow{.base = 1, .count = 2});
    BOOST_CHECK_THROW((void)engine.resolve_published(mpi::views_of(shifted), true), std::invalid_argument);
}

// --- Large streams and wide terms through the orchestration -----------------------------------------------------

// Same-shard streams beyond one 4096-query self window, fresh growth and reindexing, through the whole team: the
// result equals the legacy partitions exactly, and at T = 1 the self stream exceeds 4096 queries.
BOOST_AUTO_TEST_CASE(sharded_construction_large_self_streams_match_legacy) {
    constexpr size_t N = 16;
    const auto options = team_options();
    const size_t threads = team_size();
    std::mt19937_64 rng(17);
    Fixture f{.name = "large", .op = {}, .initial_state = {0, 1}, .cutoff = 2 * N, .logical = N};
    std::set<Key> used;
    while (f.op.size() < 20000) {
        const auto m = random_term<N>(rng, 2 + (2 * (rng() % 4)));
        if (used.insert(key_of<N>(m)).second) {
            add_random_term<N>(f.op, m, 0.001 * static_cast<double>(f.op.size() % 97));
        }
    }
    const Circuit c{.gates = {{0, 1}, {3, 17}, {2, 5, 9, 30}},
                    .mapping = {0, 1, 2},
                    .gen_coeffs = {1.0, 0.5, -1.0},
                    .params = {0.1, 0.2, 0.3}};
    for (const bool fused : {false, true}) {
        auto s = seed<N>(f, options);
        auto logs = make_logs(threads);
        require_success(fused ? run_propagate<N>(s, c, std::nullopt, Probe{.logs = &logs, .capture = true})
                              : run_graph<N>(s, c, std::nullopt, Probe{.logs = &logs, .capture = true}));
        auto p = reference_root<N>(f, threads);
        if (fused) {
            p.propagate(c.gates, c.mapping, c.gen_coeffs, c.params);
        }
        else {
            p.build_graph(c.gates, c.mapping, c.gen_coeffs);
        }
        for (size_t t = 0; t < threads; ++t) {
            BOOST_TEST((digest<N>(*s.shards[t]) == reference_digest<N>(p, t)), "shard " << t);
        }
        // A full self window probes 4096 queries in 16 blocks; a stream reaching one exceeds a single window.
        size_t full_windows = 0;
        for (const auto &log : logs) {
            for (const auto *call : log.ranges.of(KernelRange::self_probe)) {
                full_windows += static_cast<size_t>(call->worker.size() == 16);
            }
        }
        if (threads == 1) {
            BOOST_TEST(full_windows > 0U);
        }
    }
}

// Wide positions (250 modes: positions above 255, more than 31 positions per record, multiword keys) through graph
// construction and propagation, against the legacy partitions.
BOOST_AUTO_TEST_CASE(sharded_construction_wide_terms_match_legacy) {
    constexpr size_t N = 250;
    const auto options = team_options();
    const size_t threads = team_size();
    std::mt19937_64 rng(250);
    Fixture f{.name = "wide", .op = {}, .initial_state = {0, 1, 2}, .cutoff = 2 * N, .logical = N};
    std::set<Key> used;
    while (f.op.size() < 300) {
        const auto m = random_term<N>(rng, 2 * (1 + (rng() % 20)));
        if (used.insert(key_of<N>(m)).second) {
            add_random_term<N>(f.op, m, 0.01 * static_cast<double>(1 + (f.op.size() % 13)));
        }
    }
    const Circuit c{.gates = {{1, 300}, {7, 260, 261, 499}, {2}, {40, 41}},
                    .mapping = {0, 1, 2, 3},
                    .gen_coeffs = {1.0, -0.5, 0.25, 1.0},
                    .params = {0.3, 0.2, 0.1, 0.4}};
    for (const bool fused : {false, true}) {
        auto s = seed<N>(f, options);
        require_success(fused ? run_propagate<N>(s, c, std::nullopt) : run_graph<N>(s, c, std::nullopt));
        auto p = reference_root<N>(f, threads);
        if (fused) {
            p.propagate(c.gates, c.mapping, c.gen_coeffs, c.params);
        }
        else {
            p.build_graph(c.gates, c.mapping, c.gen_coeffs);
        }
        for (size_t t = 0; t < threads; ++t) {
            BOOST_TEST((digest<N>(*s.shards[t]) == reference_digest<N>(p, t)), "shard " << t);
        }
    }
}

// --- Failures -----------------------------------------------------------------------------------------------------

namespace {

// The checkpoint group a work item belongs to, so "later work" is ordered across owners.
auto group_of(W work) -> int {
    switch (work) {
        case W::frame:
            return -2;
        case W::picture:
            return -1;
        case W::traverse:
        case W::prepare_leaders:
            return 0;
        case W::resolve_leaders:
            return 1;
        case W::consume_leaders:
        case W::prepare_followers:
            return 2;
        case W::resolve_followers:
            return 3;
        case W::consume_followers:
        case W::finalize:
        case W::apply:
            return 4;
        case W::caches:
            return 5;
        case W::seed:
        case W::replay:
            return 6; // informed construction only; never reported by these cases
    }
    return 7;
}

auto order_of(size_t step, W work) -> std::pair<long long, int> {
    const long long s = step == sharded::kNoStep ? (work == W::caches ? 1LL << 40 : -1) : static_cast<long long>(step);
    return {s, group_of(work)};
}

// Nothing ran after the failing checkpoint: every visit is at or before the failing (step, group).
auto no_later_work(const std::vector<ShardLog> &logs, size_t step, W work) -> bool {
    const auto limit = order_of(step, work);
    for (const auto &log : logs) {
        for (const auto &v : log.visits) {
            if (order_of(v.step, v.work) > limit) {
                return false;
            }
        }
    }
    return true;
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

} // namespace

// A failure in any phase, on the primary or the last owner, joins the team, suppresses every later phase body on every
// worker, returns the original exception, reports whether mutation started, and leaves an earlier copy intact.
BOOST_AUTO_TEST_CASE(sharded_construction_failure_in_each_phase_suppresses_later_work) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    const auto cross_works = {W::resolve_leaders, W::consume_leaders, W::resolve_followers, W::consume_followers};
    for (const bool fused : {false, true}) {
        std::vector<W> works =
            {W::frame, W::traverse, W::prepare_leaders, W::prepare_followers, W::finalize, W::caches};
        if (fused) {
            works.push_back(W::picture);
            works.push_back(W::apply);
        }
        if (threads > 1) {
            works.insert(works.end(), cross_works.begin(), cross_works.end());
        }
        for (const W work : works) {
            for (const size_t shard : {size_t{0}, threads - 1}) {
                const bool loop = work != W::frame && work != W::picture && work != W::caches;
                const size_t step = loop ? 1 : sharded::kNoStep;
                BOOST_TEST_CONTEXT((fused ? "propagate" : "graph")
                                   << " work " << static_cast<int>(work) << " shard " << shard) {
                    auto s = seed<kN>(cs.f, options);
                    const auto copy = sharded::copy_shards(options, s.shards);
                    std::vector<ShardDigest> copy_before;
                    for (const auto &state : copy) {
                        copy_before.push_back(digest<kN>(*state));
                    }
                    auto logs = make_logs(threads);
                    const Probe probe{.logs = &logs, .capture = false, .fail = {{work, step, shard}}};
                    const auto outcome =
                        fused ? run_propagate<kN>(s, cs.c, cs.k, probe) : run_graph<kN>(s, cs.c, cs.k, probe);
                    BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
                    BOOST_TEST(message_of(outcome.error)
                               == std::format("injected failure: work {} step {} shard {}",
                                              static_cast<int>(work),
                                              step,
                                              shard));
                    BOOST_TEST(outcome.mutation_started == (work != W::frame));
                    BOOST_TEST(no_later_work(logs, step, work));
                    for (size_t t = 0; t < threads; ++t) {
                        BOOST_TEST((digest<kN>(*copy[t]) == copy_before[t]));
                    }
                    if (work == W::frame) {
                        // Nothing mutated: the shards still equal the untouched copy.
                        for (size_t t = 0; t < threads; ++t) {
                            BOOST_TEST((digest<kN>(*s.shards[t]) == copy_before[t]));
                        }
                    }
                }
            }
        }
    }
}

// Real failures inside the kernels and phases: a decode worker of a destination, the fused apply of the last owner,
// and a publication after the payload was built; each joins and suppresses later work.
BOOST_AUTO_TEST_CASE(sharded_construction_kernel_decode_and_publication_failures,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    struct Scenario {
        const char *name;
        bool fused;
        std::optional<KernelFailure> kernel;
        std::optional<std::pair<size_t, bool>> publication;
        const char *message;
    };
    const std::vector<Scenario> scenarios = {
        {"decode",
         false,
         KernelFailure{threads - 1, KernelRange::decode, 1, 0},
         std::nullopt,
         "injected kernel failure"},
        {"scatter", true, KernelFailure{1, KernelRange::scatter, 0, 0}, std::nullopt, "injected kernel failure"},
        {"fused apply",
         true,
         KernelFailure{threads - 1, KernelRange::fused_apply, 2, 0},
         std::nullopt,
         "injected kernel failure"},
        {"leader publication", false, std::nullopt, std::pair{threads - 1, true}, "injected publication failure"},
        {"follower publication", true, std::nullopt, std::pair{size_t{0}, false}, "injected publication failure"},
    };
    for (const auto &sc : scenarios) {
        BOOST_TEST_CONTEXT(sc.name) {
            auto s = seed<kN>(cs.f, options);
            const auto copy = sharded::copy_shards(options, s.shards);
            const auto copy_digest = digest<kN>(*copy[0]);
            auto logs = make_logs(threads);
            const Probe probe{.logs = &logs,
                              .capture = false,
                              .kernel_fail = sc.kernel,
                              .publication_fail = sc.publication};
            const auto outcome =
                sc.fused ? run_propagate<kN>(s, cs.c, cs.k, probe) : run_graph<kN>(s, cs.c, cs.k, probe);
            BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
            BOOST_TEST(message_of(outcome.error) == sc.message);
            BOOST_TEST(outcome.mutation_started);
            BOOST_TEST((digest<kN>(*copy[0]) == copy_digest));
            // No shard started the caches phase after the failure.
            BOOST_TEST(count_visits(logs, W::caches) == 0U);
        }
    }
}

// Several owners failing in the same phase: the lowest-numbered owner's exception is returned.
BOOST_AUTO_TEST_CASE(sharded_construction_concurrent_failures_select_the_lowest_owner,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    auto s = seed<kN>(cs.f, options);
    auto logs = make_logs(threads);
    const Probe probe{.logs = &logs,
                      .capture = false,
                      .fail = {{W::resolve_followers, 2, threads - 1}, {W::resolve_followers, 2, 1}}};
    const auto outcome = run_graph<kN>(s, cs.c, cs.k, probe);
    BOOST_TEST_REQUIRE(static_cast<bool>(outcome.error));
    BOOST_TEST(message_of(outcome.error)
               == std::format("injected failure: work {} step 2 shard 1", static_cast<int>(W::resolve_followers)));
    BOOST_TEST(no_later_work(logs, 2, W::resolve_followers));
}

// Real allocation failures at every allocation of one owner's frame, traversal, resolution and finish: each is a
// catchable std::bad_alloc after the join, with no leak; a frame failure mutates nothing; earlier copies survive.
BOOST_AUTO_TEST_CASE(sharded_construction_allocation_failures_are_contained,
                     *boost::unit_test::precondition(has_allocation_probe)) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[1]; // Schrödinger: fresh-insert scoring in the apply
    struct Site {
        W work;
        size_t step;
        size_t shard;
    };
    std::vector<Site> sites = {{W::frame, sharded::kNoStep, threads - 1},
                               {W::traverse, 2, 0},
                               {W::finalize, 3, threads - 1}};
    sites.push_back({W::apply, 2, 0}); // propagation only; graph construction has no apply
    if (threads > 1) {
        sites.push_back({W::resolve_leaders, 1, threads - 1});
        sites.push_back({W::consume_followers, 1, 0});
    }
    for (const bool fused : {false, true}) {
        // Warm every worker's retained thread-local scan scratch, which later runs reuse and never free.
        {
            auto warm = seed<kN>(cs.f, options);
            require_success(fused ? run_propagate<kN>(warm, cs.c, cs.k) : run_graph<kN>(warm, cs.c, cs.k));
        }
        for (const auto &site : sites) {
            if (site.work == W::apply && !fused) {
                continue;
            }
            size_t injected = 0;
            for (size_t nth = 1; nth < 2000; ++nth) {
                bool failed = false;
                {
                    const allocation::LiveBytes live;
                    {
                        auto s = seed<kN>(cs.f, options);
                        const auto copy = sharded::copy_shards(options, s.shards);
                        const auto copy_digest = digest<kN>(*copy[threads - 1]);
                        auto logs = make_logs(threads);
                        const Probe probe{.logs = &logs,
                                          .capture = false,
                                          .alloc = AllocAt{site.work, site.step, site.shard, nth}};
                        const auto outcome =
                            fused ? run_propagate<kN>(s, cs.c, cs.k, probe) : run_graph<kN>(s, cs.c, cs.k, probe);
                        disarm_team();
                        failed = static_cast<bool>(outcome.error);
                        if (failed) {
                            ++injected;
                            bool bad_alloc = false;
                            try {
                                std::rethrow_exception(outcome.error);
                            }
                            catch (const std::bad_alloc &) {
                                bad_alloc = true;
                            }
                            catch (...) {
                            }
                            BOOST_TEST(bad_alloc);
                            BOOST_TEST(outcome.mutation_started == (site.work != W::frame));
                            BOOST_TEST((digest<kN>(*copy[threads - 1]) == copy_digest));
                        }
                    }
                    BOOST_TEST(live.net() == 0);
                }
                if (!failed) {
                    break;
                }
            }
            BOOST_TEST_MESSAGE("allocation site work " << static_cast<int>(site.work) << " shard " << site.shard
                                                       << (fused ? " propagate" : " graph") << ": " << injected
                                                       << " injected failures");
            BOOST_TEST(injected > 0U);
        }
    }
}

// Arguments are checked before the team, with nothing mutated.
BOOST_AUTO_TEST_CASE(sharded_construction_rejects_invalid_arguments_before_the_team) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto cs = legacy_cases()[0];
    auto s = seed<kN>(cs.f, options);
    const auto before = digest<kN>(*s.shards[0]);
    auto logs = make_logs(threads);
    const auto gens = generators<kN>(cs.f, cs.c);
    VecZ gate_indices(gens.size());
    const auto good = s.ctx();
    const auto graph =
        [&](const sharded::ConstructionContext<kN> &ctx, std::span<const double> coeffs, std::optional<size_t> k) {
            return sharded::build_graph<kN>(options,
                                            s.shards,
                                            ctx,
                                            {.generators = gens,
                                             .parameter_mapping = cs.c.mapping,
                                             .gen_coeffs = coeffs,
                                             .gate_indices = gate_indices,
                                             .only_rotate_len_k = k},
                                            Probe{.logs = &logs});
        };
    BOOST_CHECK_THROW((void)graph(good, std::span<const double>(cs.c.gen_coeffs).first(2), std::nullopt),
                      std::invalid_argument);
    BOOST_CHECK_THROW((void)graph(good, cs.c.gen_coeffs, 2 * kN + 1), std::runtime_error);
    auto two_ranks = good;
    two_ranks.router = routing::Router::splitmix(2 * threads);
    BOOST_CHECK_THROW((void)graph(two_ranks, cs.c.gen_coeffs, std::nullopt), std::invalid_argument);
    auto wrong_rank = good;
    wrong_rank.world.rank = 1;
    BOOST_CHECK_THROW((void)graph(wrong_rank, cs.c.gen_coeffs, std::nullopt), std::invalid_argument);
    sharded::Shards<kN> short_shards;
    BOOST_CHECK_THROW(
        (void)sharded::propagate<kN>(options,
                                     short_shards,
                                     good,
                                     {.generators = gens, .mapped_params = {}, .only_rotate_len_k = std::nullopt}),
        std::invalid_argument);
    BOOST_TEST((digest<kN>(*s.shards[0]) == before));
    for (const auto &log : logs) {
        BOOST_TEST(log.visits.empty());
    }
}

// --- Multi-rank: P ranks x T threads against the legacy hybrid -------------------------------------------------------
//
// Launched by cpp/tests/CMakeLists.txt with monoprop_TEST_SHARDED_RANKS = P under mpiexec (never by the whole-suite MPI
// variants, which run at the default budget). The legacy facade at partitions = T over MPI_COMM_WORLD (the one-store
// path at T = 1) routes over the same (P, T) flat owners: child t of rank r is flat owner r * T + t. Every shard of
// every rank must equal its legacy child bit for bit: rows in ID order (so local sender blocks never jump ahead of
// lower-numbered remote senders, and deferred same-shard misses come last), coefficients, caches and the layers'
// partner layouts, whose sin_send lists pin the answers to the queries they retrace.

namespace {

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
auto multirank_launch(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    const char *text = std::getenv("monoprop_TEST_SHARDED_RANKS");
    const int ranks = mpi::size(mpi::Comm(MPI_COMM_WORLD));
    boost::test_tools::assertion_result result(text != nullptr && ranks >= 2 && std::stoi(text) == ranks);
    result.message()
        << "needs a dedicated multi-rank launch (monoprop_TEST_SHARDED_RANKS = the world size, at least 2)";
    return result;
}
#endif

} // namespace

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
BOOST_AUTO_TEST_CASE(sharded_construction_multirank_matches_the_legacy_hybrid,
                     *boost::unit_test::precondition(multirank_launch)) {
    const auto options = team_options();
    const size_t threads = team_size();
    const auto world = sharded::PhysicalWorld::of(mpi::Comm(MPI_COMM_WORLD));
    size_t remote_rows = 0;
    for (const auto &cs : legacy_cases()) {
        BOOST_TEST_CONTEXT(label(cs) << " rank " << world.rank << " of " << world.ranks << " T=" << threads) {
            for (const bool propagate : {false, true}) {
                BOOST_TEST_CONTEXT((propagate ? "propagate" : "build_graph")) {
                    auto p = legacy<kN>(cs.f, threads, MPI_COMM_WORLD);
                    auto s = seed<kN>(cs.f, options, world);
                    if (propagate) {
                        p.propagate(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, cs.c.params, cs.k);
                        require_success(run_propagate<kN>(s, cs.c, cs.k));
                    }
                    else {
                        p.build_graph(cs.c.gates, cs.c.mapping, cs.c.gen_coeffs, std::nullopt, std::nullopt, cs.k);
                        require_success(run_graph<kN>(s, cs.c, cs.k));
                    }
                    for (size_t t = 0; t < threads; ++t) {
                        const auto got = digest<kN>(*s.shards[t]);
                        BOOST_TEST((got == legacy_digest<kN>(p, t)), "shard " << t);
                        for (const auto &layer : got.layers) {
                            for (const auto &entry : layer.occupied) {
                                remote_rows += entry[0] / threads != world.rank ? entry[1] : 0;
                            }
                        }
                    }
                }
            }
        }
    }
    // The fixtures really cross ranks: some layer has partners on another process.
    BOOST_TEST(mpi::allreduce_sum<size_t>(remote_rows, mpi::Comm(MPI_COMM_WORLD)) > 0U);
}
#endif
