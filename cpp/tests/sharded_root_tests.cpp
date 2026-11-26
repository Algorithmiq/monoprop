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
 * team size T is the captured launch budget; cpp/tests/CMakeLists.txt runs every case at T = 1 and reruns them in a
 * fresh process at T = 4. Shard inspection goes through PropagatorTestAccess. Numerical parity across T is checked
 * from separate processes by tests/test_sharded_openmp.py.
 */

#include <boost/test/unit_test.hpp>

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "PropagatorTestAccess.h"
#include "ShardedTestSupport.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t num_modes = 12;
using MP = MonomialPropagator<num_modes>;
using Access = monoprop::detail::PropagatorTestAccess<num_modes>;

auto launch_team() -> size_t {
    return static_cast<size_t>(sharded_test::expected_team().value_or(1));
}

auto lih() -> const CaseData & {
    static const auto data = load_case_data<num_modes>("lih_fermionic_spin_exact.msgpack");
    return data;
}

struct Config {
    bool schrodinger = false;
    std::optional<double> lower_atol = std::nullopt;
    std::optional<OperatorDict> op = std::nullopt;    // LiH's Hamiltonian when empty
    std::optional<VecZ> initial_state = std::nullopt; // LiH's initial state when empty
};

auto make_on(mpi::Comm comm, const Config &config = {}) -> std::unique_ptr<MP> {
    const auto &data = lih();
    return std::make_unique<MP>(config.op ? *config.op : data.hamiltonian,
                                4U,
                                config.initial_state ? *config.initial_state : data.initial_state,
                                config.schrodinger ? std::optional<unsigned int>{6U} : std::nullopt,
                                comm,
                                config.lower_atol);
}

auto make(const Config &config = {}) -> std::unique_ptr<MP> {
    return make_on(mpi::Comm(MPI_COMM_SELF), config);
}

auto build(MP &sim) -> void {
    const auto &data = lih();
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
}

// make(), with the LiH circuit built.
auto make_built(const Config &config = {}) -> std::unique_ptr<MP> {
    auto sim = make(config);
    build(*sim);
    return sim;
}

auto params() -> const VecD & {
    return lih().parameters;
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
        return std::string(e.what()).find("monoprop_NUM_THREADS=1") != std::string::npos;
    };
    BOOST_CHECK_EXCEPTION(sim.mp_op(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(std::as_const(sim).mp_op(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.graph(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.indexing(), MultiShardUnsupported, remedy);
    BOOST_CHECK_EXCEPTION(sim.graph_data(), MultiShardUnsupported, remedy);
    // The rejection mutates nothing: the object stays valid and keeps answering.
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_CHECK_EXCEPTION(sim.graph(), MultiShardUnsupported, remedy);
    BOOST_TEST(std::isfinite(sim.expectation_value(data.parameters)));
    BOOST_TEST(!monoprop::detail::PropagatorTestAccess<8>::is_invalid(sim));
}

BOOST_AUTO_TEST_CASE(sharded_root_owns_t_routed_shards) {
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto sim = make_built({.schrodinger = schrodinger});
            const size_t team = launch_team();
            const auto &shards = Access::shards(*sim);
            const auto &router = Access::router(*sim);
            BOOST_TEST(static_cast<size_t>(Access::options(*sim).threads) == team);
            BOOST_TEST(shards.size() == team);
            BOOST_TEST(router.ranks() == 1U);
            BOOST_TEST(router.shards_per_rank() == team);
            std::set<VecZ> keys;
            size_t rows = 0;
            for (size_t t = 0; t < team; ++t) {
                BOOST_TEST(shards[t]->graph.layers() == sim->graph_layers());
                shards[t]->op.store->for_each([&](const auto &mono, size_t) {
                    BOOST_TEST(router.dest<num_modes>(mono) == t); // flat owner rank * T + t at rank 0
                    BOOST_TEST(keys.insert(bitset_to_indices<num_modes>(mono)).second); // no key on two shards
                    ++rows;
                });
            }
            BOOST_TEST(rows == sim->size());
            // Heisenberg keeps the identity out of every shard; it is rank-level metadata, counted once.
            if (!schrodinger) {
                BOOST_TEST(!keys.contains(VecZ{}));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_rejections_before_mutation_leave_it_usable) {
    auto sim = make_built();
    const double energy = energy_of(*sim);
    const auto retained = sim->expectation_value_functional();
    const auto &data = lih();
    BOOST_CHECK_THROW(sim->build_graph(data.majoranas, VecZ{0}, data.gen_coeffs), std::exception);
    // A generator index past the system is rejected before any gate runs, even behind a valid gate.
    const std::vector<VecZ> bad_gates{data.majoranas.front(), VecZ{0, 2 * num_modes}};
    BOOST_CHECK_THROW(sim->build_graph(bad_gates, VecZ{0, 0}, VecD{1.0, 1.0}), std::exception);
    auto fresh = make();
    BOOST_CHECK_THROW(fresh->propagate(bad_gates, VecZ{0, 0}, VecD{1.0, 1.0}, VecD{0.1}), std::exception);
    BOOST_TEST(!Access::is_invalid(*fresh));
    BOOST_CHECK_THROW(sim->propagate(data.majoranas, data.param_inds, data.gen_coeffs, params()), std::exception);
    BOOST_CHECK_THROW(sim->expectation_value(VecD{0.1}), std::exception);
    BOOST_CHECK_THROW((void)retained(VecD{0.1}), std::exception);
    BOOST_CHECK_THROW(sim->contract_partially(VecD{}, false), std::exception);
    BOOST_CHECK_THROW(sim->update_initial_operator(OperatorDict{{VecZ{0, 2 * num_modes}, 1.0}}), std::exception);
    // Past the cutoff, so no shard holds the term: Heisenberg rejects it before the epoch moves.
    BOOST_CHECK_THROW(sim->update_initial_operator(OperatorDict{{VecZ{0, 2, 4, 6, 8, 10, 12, 14}, 1.0}}),
                      std::runtime_error);
    BOOST_CHECK_THROW(sim->set_parameter_mapping(VecZ{0}), std::exception);
    BOOST_CHECK_THROW(sim->update_basis_change(std::vector<VecZ>{VecZ{0}}), std::exception);
    BOOST_CHECK_THROW(sim->build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, VecD{0.1}),
                      std::exception);
    sim->update_upper_atol(0.5);
    BOOST_CHECK_THROW(sim->update_lower_atol(1.0), PropagatorConfigError);
    sim->update_upper_atol(std::nullopt);
    BOOST_TEST(!Access::is_invalid(*sim));
    BOOST_TEST(energy_of(*sim) == energy);
    BOOST_TEST(near(retained(params()), energy));
    BOOST_TEST(sim->graph_layers() == data.majoranas.size());
}

// --- Multi-rank roots (P ranks x T threads) ----------------------------------------------------------------------

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

auto routable(boost::unit_test::test_unit_id) -> boost::test_tools::assertion_result {
    boost::test_tools::assertion_result result(routable_world());
    result.message() << "the linear router rejects " << world_size() << " ranks";
    return result;
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
            if (!near(value.real(), it->second.real()) || !near(value.imag(), it->second.imag())) {
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
            BOOST_TEST(near(got[i], want[i]), "component " << i << ": " << got[i] << " vs " << want[i]);
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sharded_root_multirank_matches_one_process,
                     *min_ranks(2) * boost::unit_test::precondition(routable)) {
    const auto &data = lih();
    for (const Config &config : {Config{},
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
            BOOST_TEST(near(world->expectation_value(p), self->expectation_value(p)));
            const auto [wv, wg] = world->expectation_value_and_gradient(p);
            const auto [sv, sg] = self->expectation_value_and_gradient(p);
            BOOST_TEST(near(wv, sv));
            check_close_vectors(wg, sg, "gradient");
            const auto [pv, pg] = world->expectation_value_and_gradient_functional(1e-3)(p);
            const auto [qv, qg] = self->expectation_value_and_gradient_functional(1e-3)(p);
            BOOST_TEST(near(pv, qv));
            check_close_vectors(pg, qg, "pared gradient");
            check_global_terms(world->evolved_operator_terms(p, 0.0), self->evolved_operator_terms(p, 0.0), "export");
            // A copy evaluates alike and mutates independently.
            auto copy = std::make_unique<MP>(*world);
            BOOST_TEST(near(copy->expectation_value(p), wv));
            // Contract in place, then extend the contracted picture.
            BOOST_TEST(mpi::allreduce_sum<size_t>(world->contract_partially(p, true).size(), mpi::Comm(MPI_COMM_WORLD))
                       == self->contract_partially(p, true).size());
            BOOST_TEST(near(world->expectation_value({}), self->expectation_value({})));
            build(*world);
            build(*self);
            BOOST_TEST(near(world->expectation_value(p), self->expectation_value(p)));
            BOOST_TEST(near(copy->expectation_value(p), wv));
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_root_empty_operators_and_identity_generators) {
    // Roots with no initial state.
    auto empty = make({.op = OperatorDict{}, .initial_state = VecZ{}});
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
    auto core_only = make({.op = OperatorDict{{VecZ{}, std::complex<double>{2.5, 0.0}}}, .initial_state = VecZ{}});
    core_only->build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0});
    BOOST_TEST(core_only->core_term() == 2.5);
    BOOST_TEST(core_only->expectation_value(VecD{0.3}) == 2.5);
    BOOST_TEST(core_only->expectation_value_and_gradient(VecD{0.3}).first == 2.5);
    BOOST_TEST(core_only->expectation_value_functional(1e-3)(VecD{0.3}) == 2.5);
    BOOST_TEST(core_only->evolved_operator_terms(VecD{0.3}, 0.0).empty());
}
