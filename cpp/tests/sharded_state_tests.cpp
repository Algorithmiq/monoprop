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
#include "monoprop/detail/sharded/State.h"

/*
 * Owner-initialized shard state (detail/sharded/State.h). The team size T is the captured launch budget;
 * cpp/tests/CMakeLists.txt reruns these cases in fresh processes at T = 1, 2 and 4. Cases that need a nonprimary
 * worker skip below T = 2.
 *
 * Oracles: the legacy partition facade (partitions = T over an in-process communicator, whose child t is flat owner
 * t at geometry (1, T)) and, at T = 1, the ordinary single-store propagator. They are constructed on the test thread,
 * outside any sharded team, and are references only. Those cases compile in the legacy (default) build only, where
 * both runtimes exist; the sharded candidate build compiles every other case, with the candidate root (the same
 * seam, driven through the public constructor at the launch's T) wherever a built source or an aggregate reference
 * is needed, and an independent reference for seeding.
 *
 * Workers never call BOOST_TEST. Initializers and the library's test observer write per-owner slots; assertions run
 * after the team has joined. AllocationProbe.h supplies per-thread allocation counts and real allocation failures at
 * the n-th operator new of a chosen owner. Owner-thread observations show where construction executes, not physical
 * memory placement.
 */

#include <boost/test/unit_test.hpp>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <format>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "AllocationProbe.h"
#include "PauliTestOracle.h"
#include "PropagatorTestAccess.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/monomial_propagator/MonomialPropagatorCommon.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/parallel/ThreadBudget.h"

namespace {

using namespace monoprop;
namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;
namespace allocation = test_utils::allocation;
using namespace std::chrono_literals;

constexpr size_t kN = 8;
using State = sharded::ShardState<kN>;
using Shards = sharded::Shards<kN>;
using Operator = monoprop::detail::MPOperator<kN>;
using Propagator = MonomialPropagator<kN>;
using Access = monoprop::detail::PropagatorTestAccess<kN>;
using Mono = Monomial<kN>;

// Monomials have no ordering of their own; any strict order serves the key-set comparisons.
struct MonoLess {
    auto operator()(const Mono &a, const Mono &b) const -> bool {
        return std::lexicographical_compare(a.data(),
                                            a.data() + Mono::num_words(),
                                            b.data(),
                                            b.data() + Mono::num_words());
    }
};

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
};

// Stores the coefficient whose encoding is `value`, so every fixture is encodable in its basis.
auto add_term(OperatorDict &op, Basis basis, const VecZ &indices, double value) -> void {
    op[indices] = algebra_decode_coeff<kN>(basis, std::complex<double>(value, 0.0), indices_to_bitset<kN>(indices));
}

// Pairs, a ladder of quartics and a few six- and eight-slot terms over the first 2 * logical slots, plus identity.
// Under a length-4 cutoff the long terms exceed the packed inline width and spill to the overflow map.
auto majorana_operator(size_t logical) -> OperatorDict {
    OperatorDict op;
    const size_t slots = 2 * logical;
    auto value = 0.25;
    for (size_t i = 0; i < slots; ++i) {
        for (size_t j = i + 1; j < slots; ++j) {
            add_term(op, Basis::Majorana, {i, j}, value);
            value += 0.125;
        }
    }
    for (size_t i = 0; i + 3 < slots; ++i) {
        add_term(op, Basis::Majorana, {i, i + 1, i + 2, i + 3}, -0.5 - (0.03125 * static_cast<double>(i)));
    }
    add_term(op, Basis::Majorana, {0, 1, 2, 3, 4, 5}, 0.75);
    add_term(op, Basis::Majorana, {1, 3, 5, 7, 9, 11}, -1.25);
    add_term(op, Basis::Majorana, {0, 2, 4, 6, 8, 10}, 1.5);
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
        for (const char p : {'X', 'Y', 'Z'}) {
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
    add("IYYYYYII");
    add("ZZZZZZZZ");
    add_term(op, Basis::Pauli, {}, -1.75);
    return op;
}

auto nontrivial_fixtures() -> std::vector<Fixture> {
    const auto data = test_utils::load_case_data<kN>("random_exact.msgpack");
    return {
        {.name = "majorana-heisenberg-logical-6",
         .op = majorana_operator(6),
         .initial_state = {0, 3},
         .schrodinger_cutoff = std::nullopt,
         .logical = 6},
        // Odd: rounds up to three pairs, so the paired basis has sum C(6, 0..3) = 42 rows including identity. The
        // unpaired Hamiltonian terms stay pending in init_op_map.
        {.name = "majorana-schrodinger-logical-6",
         .op = majorana_operator(6),
         .initial_state = {0, 3},
         .schrodinger_cutoff = 5,
         .logical = 6},
        {.name = "majorana-heisenberg-random-exact", .op = data.hamiltonian, .initial_state = data.initial_state},
        {.name = "majorana-schrodinger-random-exact",
         .op = data.hamiltonian,
         .initial_state = data.initial_state,
         .schrodinger_cutoff = 6},
        {.name = "pauli-heisenberg",
         .op = pauli_operator(),
         .initial_state = {},
         .schrodinger_cutoff = std::nullopt,
         .basis = Basis::Pauli,
         .cutoff_type = CutoffType::Support,
         .cutoff = 3},
        {.name = "pauli-schrodinger",
         .op = pauli_operator(),
         .initial_state = {},
         .schrodinger_cutoff = 5,
         .basis = Basis::Pauli,
         .cutoff_type = CutoffType::Support,
         .cutoff = 3},
    };
}

auto degenerate_fixtures() -> std::vector<Fixture> {
    OperatorDict identity;
    add_term(identity, Basis::Majorana, {}, 2.0);
    OperatorDict pauli_identity;
    add_term(pauli_identity, Basis::Pauli, {}, -2.0);
    return {
        {.name = "majorana-heisenberg-empty", .op = {}, .initial_state = {0, 1}},
        {.name = "majorana-heisenberg-identity-only", .op = identity, .initial_state = {0, 1}},
        {.name = "majorana-schrodinger-identity-only",
         .op = identity,
         .initial_state = {0, 1},
         .schrodinger_cutoff = 3,
         .logical = 4},
        {.name = "pauli-heisenberg-identity-only",
         .op = pauli_identity,
         .initial_state = {},
         .schrodinger_cutoff = std::nullopt,
         .basis = Basis::Pauli,
         .cutoff_type = CutoffType::Support,
         .cutoff = 3},
    };
}

auto all_fixtures() -> std::vector<Fixture> {
    auto fixtures = nontrivial_fixtures();
    for (auto &fixture : degenerate_fixtures()) {
        fixtures.push_back(std::move(fixture));
    }
    return fixtures;
}

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
// The candidate root at the launch's T over MPI_COMM_SELF: its shards are seeded by the seam under test.
auto root(const Fixture &f) -> Propagator {
    return Propagator(f.op,
                      f.cutoff,
                      f.initial_state,
                      f.schrodinger_cutoff,
                      MPI_COMM_SELF,
                      std::nullopt,
                      std::nullopt,
                      f.cutoff_type,
                      std::nullopt,
                      f.logical,
                      f.basis);
}

// The rank-level reference for aggregates: the candidate root, which must run at the launch's T.
auto oracle_of(const Fixture &f, size_t threads) -> Propagator {
    auto sim = root(f);
    BOOST_TEST_REQUIRE(Access::shards(sim).size() == threads);
    return sim;
}
#else
// The legacy propagator at `partitions` partitions: a facade over in-process children for partitions > 1, the
// single-store propagator for partitions = 1.
auto legacy(const Fixture &f, size_t partitions) -> Propagator {
    return Propagator(f.op,
                      f.cutoff,
                      f.initial_state,
                      f.schrodinger_cutoff,
                      MPI_COMM_SELF,
                      std::nullopt,
                      std::nullopt,
                      f.cutoff_type,
                      std::nullopt,
                      f.logical,
                      f.basis,
                      partitions);
}

// The legacy store that flat owner `shard` holds at geometry (1, T).
auto legacy_owner(const Propagator &p, size_t shard) -> const Propagator & {
    return Access::partition_count(p) == 0 ? p : Access::partition(p, static_cast<int>(shard));
}

// The rank-level reference for aggregates: the legacy partitions at geometry (1, T).
auto oracle_of(const Fixture &f, size_t threads) -> Propagator {
    return legacy(f, threads);
}
#endif

// Caller-side preparation, as the rank-level constructor performs it.
auto make_seed(const Fixture &f, routing::Router router) -> sharded::OperatorSeed<kN> {
    const auto cutoff_fn = monoprop::detail::cutoff_function<kN>(f.cutoff_type, f.cutoff, f.logical);
    std::optional<sharded::PairedBasisBounds> paired;
    if (f.schrodinger_cutoff) {
        paired = sharded::paired_basis_bounds(*f.schrodinger_cutoff, f.logical, router.flat_world());
    }
    return {.initial_operator = f.op,
            .initial_state = f.initial_state,
            .router = router,
            .basis = f.basis,
            .logical_num_modes = f.logical,
            .paired = paired,
            .inline_width = sharded::packed_inline_width<kN>(f.schrodinger_cutoff.has_value(), cutoff_fn)};
}

auto local_router(size_t threads) -> routing::Router {
    return routing::make_router<kN>(1, threads);
}

// --- Owner observation ----------------------------------------------------------------------------------------

// What one owner saw inside the library's own seed or copy body, written only by that owner.
struct alignas(64) OwnerSlot {
    std::atomic<int> begins{0};
    std::atomic<int> ends{0};
    int worker = -1;
    int team = 0;
    std::thread::id thread;
    sharded::ShardWork work = sharded::ShardWork::seed;
    size_t bytes_at_begin = 0;
    size_t bytes = 0; //!< Allocated on this thread between begin() and end().
    bool armed = false;
    bool unfired = false; //!< An armed failure was still pending at end().
};

// Test observer for seed_shards/copy_shards. Optionally arms a real allocation failure on one owner.
struct OwnerProbe {
    std::vector<OwnerSlot> *slots = nullptr;
    std::optional<size_t> fail_shard = std::nullopt;
    size_t fail_nth = 0;

    auto begin(sharded::ShardWork work, size_t shard) const -> void {
        auto &slot = slots->at(shard);
        slot.begins.fetch_add(1);
        slot.worker = omp_get_thread_num();
        slot.team = omp_get_num_threads();
        slot.thread = std::this_thread::get_id();
        slot.work = work;
        slot.bytes_at_begin = allocation::thread_bytes();
        if (fail_shard == shard) {
            slot.armed = true;
            allocation::arm_failure(fail_nth);
        }
    }

    auto end(sharded::ShardWork /*work*/, size_t shard) const -> void {
        auto &slot = slots->at(shard);
        slot.bytes = allocation::thread_bytes() - slot.bytes_at_begin;
        if (slot.armed) {
            slot.unfired = allocation::disarm_failure();
        }
        slot.ends.fetch_add(1);
    }
};

// Every owner ran exactly one begin/end of `work` on its own worker, one distinct thread each, worker 0 the caller.
auto check_owner_bodies(const std::vector<OwnerSlot> &slots, sharded::ShardWork work, const std::string &what) -> void {
    const auto threads = slots.size();
    std::set<std::thread::id> distinct;
    for (size_t shard = 0; shard < threads; ++shard) {
        const auto &slot = slots[shard];
        BOOST_TEST_CONTEXT(what << " shard " << shard) {
            BOOST_TEST(slot.begins.load() == 1);
            BOOST_TEST(slot.ends.load() == 1);
            BOOST_TEST(slot.worker == static_cast<int>(shard));
            BOOST_TEST(slot.team == static_cast<int>(threads));
            BOOST_TEST((slot.work == work));
            if (allocation::available()) {
                BOOST_TEST(slot.bytes > 0U, "the owner body allocated its own state");
            }
        }
        distinct.insert(slot.thread);
    }
    BOOST_TEST(distinct.size() == threads);
    BOOST_TEST((slots.at(0).thread == std::this_thread::get_id()));
}

// --- State comparison ---------------------------------------------------------------------------------------

auto describe(const Mono &mono) -> std::string {
    std::string out = "{";
    for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
        out += std::format("{} ", b);
    }
    return out + "}";
}

// First difference between two operators, or empty. Rows, keys, lookups, overflow placement, coefficients, pending
// entries, sparse/dense state, the inverted index and every memory-accounting field (so capacities and reserves too).
auto operator_difference(const Operator &a, const Operator &b) -> std::string {
    if (a.size() != b.size()) {
        return std::format("size {} != {}", a.size(), b.size());
    }
    if (a.store->overflow_size() != b.store->overflow_size()) {
        return std::format("overflow rows {} != {}", a.store->overflow_size(), b.store->overflow_size());
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const auto row_a = a.store->row(i);
        if (row_a != b.store->row(i)) {
            return std::format("row {}: {} != {}", i, describe(row_a), describe(b.store->row(i)));
        }
        if (a.store->row_positions(i).inlined() != b.store->row_positions(i).inlined()) {
            return std::format("row {} inline placement differs", i);
        }
        if (a.store->find(row_a) != std::optional<size_t>(i) || b.store->find(row_a) != std::optional<size_t>(i)) {
            return std::format("row {} lookup does not return its own index", i);
        }
    }
    if (a.op_coeffs != b.op_coeffs) {
        return "op_coeffs differ";
    }
    if (a.state_rows_ != b.state_rows_ || a.state_vals_ != b.state_vals_
        || a.state_scored_rows_ != b.state_scored_rows_) {
        return "sparse state differs";
    }
    if (a.state_coeffs != b.state_coeffs) {
        return "dense state differs";
    }
    if (a.init_op_map.size() != b.init_op_map.size()) {
        return std::format("pending entries {} != {}", a.init_op_map.size(), b.init_op_map.size());
    }
    for (const auto &[key, value] : a.init_op_map) {
        const auto found = b.init_op_map.find(key);
        if (found == b.init_op_map.end() || found->second != value) {
            return std::format("pending entry {} differs", describe(key));
        }
    }
    if (a.initial_state != b.initial_state || a.basis != b.basis) {
        return "initial state or basis differs";
    }
    if (a.inverted_index_.has_value() != b.inverted_index_.has_value()) {
        return "inverted index presence differs";
    }
    if (a.inverted_index_) {
        const auto &ia = *a.inverted_index_;
        const auto &ib = *b.inverted_index_;
        if (ia.row_count != ib.row_count || ia.row_parity_ != ib.row_parity_) {
            return "inverted index rows differ";
        }
        for (size_t c = 0; c < ia.cols.size(); ++c) {
            if (ia.cols[c].is_dense != ib.cols[c].is_dense || ia.cols[c].words != ib.cols[c].words
                || ia.cols[c].set_rows != ib.cols[c].set_rows) {
                return std::format("inverted index column {} differs", c);
            }
        }
    }
    const auto ma = monoprop::detail::estimate_memory_usage(a);
    const auto mb = monoprop::detail::estimate_memory_usage(b);
    const auto fields_a = std::vector<size_t>{ma.operator_terms_bytes,
                                              ma.op_coeffs_bytes,
                                              ma.state_coeffs_bytes,
                                              ma.indexing_bytes,
                                              ma.init_operator_bytes,
                                              ma.initial_state_bytes,
                                              ma.inverted_index_bytes,
                                              ma.operator_terms_slack_bytes,
                                              ma.init_operator_entries,
                                              ma.state_coeffs_nonzero};
    const auto fields_b = std::vector<size_t>{mb.operator_terms_bytes,
                                              mb.op_coeffs_bytes,
                                              mb.state_coeffs_bytes,
                                              mb.indexing_bytes,
                                              mb.init_operator_bytes,
                                              mb.initial_state_bytes,
                                              mb.inverted_index_bytes,
                                              mb.operator_terms_slack_bytes,
                                              mb.init_operator_entries,
                                              mb.state_coeffs_nonzero};
    for (size_t k = 0; k < fields_a.size(); ++k) {
        if (fields_a[k] != fields_b[k]) {
            return std::format("memory accounting field {}: {} != {}", k, fields_a[k], fields_b[k]);
        }
    }
    return {};
}

// Graph layers compare by shared core identity: a copy shares its immutable cores with its source.
auto graph_difference(const MPGraph &a, const MPGraph &b) -> std::string {
    if (a.is_schrodinger() != b.is_schrodinger() || a.layers() != b.layers()) {
        return std::format("graph shape differs ({} vs {} layers)", a.layers(), b.layers());
    }
    for (size_t i = 0; i < a.layers(); ++i) {
        const auto &la = a.get_layer(i);
        const auto &lb = b.get_layer(i);
        if (la.shared_core() != lb.shared_core()) {
            return std::format("layer {} core differs", i);
        }
        if ((la.pruned_cos() == nullptr) != (lb.pruned_cos() == nullptr)) {
            return std::format("layer {} cosine form differs", i);
        }
    }
    return {};
}

auto state_difference(const State &a, const State &b) -> std::string {
    if (auto d = operator_difference(a.op, b.op); !d.empty()) {
        return d;
    }
    if (auto d = graph_difference(a.graph, b.graph); !d.empty()) {
        return d;
    }
    if (a.matched.epoch_ != b.matched.epoch_ || a.matched.cur_ != b.matched.cur_) {
        return "matched scratch differs";
    }
    return {};
}

auto shards_difference(const Shards &a, const Shards &b) -> std::string {
    if (a.size() != b.size()) {
        return std::format("shard count {} != {}", a.size(), b.size());
    }
    for (size_t shard = 0; shard < a.size(); ++shard) {
        if (!a[shard] || !b[shard]) {
            return std::format("shard {} is null", shard);
        }
        if (auto d = state_difference(*a[shard], *b[shard]); !d.empty()) {
            return std::format("shard {}: {}", shard, d);
        }
    }
    return {};
}

// A caller-side deep snapshot through the plain copy constructor, as an independent reference.
auto snapshot(const Shards &shards) -> Shards {
    Shards out;
    for (const auto &state : shards) {
        out.push_back(std::make_unique<State>(*state));
    }
    return out;
}

#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
// Shards bridged from a built candidate root: a caller-side deep snapshot of its own T shards.
auto bridged_shards(const Propagator &p, size_t threads) -> Shards {
    BOOST_TEST_REQUIRE(Access::shards(p).size() == threads);
    return snapshot(Access::shards(p));
}
#else
// Shards bridged from a built legacy propagator: owner t holds legacy flat owner t's operator, graph and scratch.
auto bridged_shards(const Propagator &p, size_t threads) -> Shards {
    Shards out;
    for (size_t shard = 0; shard < threads; ++shard) {
        out.push_back(Access::shard_state(legacy_owner(p, shard)));
    }
    return out;
}
#endif

// A partition facade (legacy) or root (candidate) at T with a built graph: nontrivial grown stores, layers, caches and
// matched scratch. The Hamiltonian's long terms exceed the inline width, so the stores have overflow rows too.
struct BuiltSource {
    test_utils::CaseData data;
    Fixture fixture;
    Propagator propagator;
};

auto built_source(size_t threads) -> BuiltSource {
    auto data = test_utils::load_case_data<kN>("random_exact.msgpack");
    auto fixture = Fixture{.name = "built", .op = majorana_operator(kN), .initial_state = data.initial_state};
    auto propagator = oracle_of(fixture, threads);
    propagator.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    return {std::move(data), std::move(fixture), std::move(propagator)};
}

class ScopedEnv {
public:
    ScopedEnv(const char *name, const char *value) : name_(name) {
        if (const char *old = std::getenv(name)) {
            old_ = old;
        }
        setenv(name, value, 1);
    }
    ScopedEnv(const ScopedEnv &) = delete;
    auto operator=(const ScopedEnv &) -> ScopedEnv & = delete;
    ~ScopedEnv() {
        if (old_) {
            setenv(name_, old_->c_str(), 1);
        }
        else {
            unsetenv(name_);
        }
    }

private:
    const char *name_;
    std::optional<std::string> old_;
};

struct WorkerError {
    size_t shard;
};

class InjectedFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// An initializer body for make_shards that builds a small real state on the owner.
auto small_state(bool schrodinger) -> std::unique_ptr<State> {
    auto op = Operator{};
    op.store->push_back(indices_to_bitset<kN>({0, 1}));
    op.store->emplace(indices_to_bitset<kN>({0, 1}), 0);
    op.op_coeffs.assign(1, 1.0);
    return std::make_unique<State>(std::move(op), schrodinger);
}

} // namespace

// --- Ownership and initialization -----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(sharded_state_make_shards_initializes_each_owner_once) {
    const auto options = team_options();
    const auto threads = team_size();
    auto slots = std::vector<OwnerSlot>(threads);
    auto shards = sharded::make_shards<kN>(options, [&](size_t shard) {
        auto &slot = slots.at(shard);
        slot.begins.fetch_add(1);
        slot.worker = omp_get_thread_num();
        slot.team = omp_get_num_threads();
        slot.thread = std::this_thread::get_id();
        slot.bytes_at_begin = allocation::thread_bytes();
        auto state = small_state(false);
        slot.bytes = allocation::thread_bytes() - slot.bytes_at_begin;
        slot.ends.fetch_add(1);
        return state;
    });
    check_owner_bodies(slots, sharded::ShardWork::seed, "make_shards");

    BOOST_REQUIRE(shards.size() == threads);
    std::set<const void *> stores;
    std::vector<const State *> addresses;
    for (const auto &state : shards) {
        BOOST_REQUIRE(state != nullptr);
        stores.insert(state->op.store.get());
        addresses.push_back(state.get());
        BOOST_TEST(state->size() == 1U);
    }
    BOOST_TEST(stores.size() == threads);
    // Moving the owning container moves pointer metadata only.
    const auto moved = std::move(shards);
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST(moved[shard].get() == addresses[shard]);
    }
}

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
BOOST_AUTO_TEST_CASE(sharded_state_seed_matches_the_partition_oracle) {
    const auto options = team_options();
    const auto threads = team_size();
    for (const auto &fixture : all_fixtures()) {
        BOOST_TEST_CONTEXT(fixture.name) {
            const auto oracle = legacy(fixture, threads);
            const auto seed = make_seed(fixture, local_router(threads));
            auto slots = std::vector<OwnerSlot>(threads);
            const auto shards = sharded::seed_shards(options, seed, 0, OwnerProbe{.slots = &slots});
            check_owner_bodies(slots, sharded::ShardWork::seed, fixture.name);
            BOOST_REQUIRE(shards.size() == threads);
            for (size_t shard = 0; shard < threads; ++shard) {
                const auto &reference = legacy_owner(oracle, shard);
                const auto &state = *shards[shard];
                BOOST_TEST_CONTEXT("shard " << shard) {
                    const auto difference = operator_difference(state.op, reference.mp_op());
                    BOOST_TEST(difference.empty(), difference);
                    BOOST_TEST(state.graph.layers() == 0U);
                    BOOST_TEST(state.graph.is_schrodinger() == fixture.schrodinger_cutoff.has_value());
                    BOOST_TEST(state.matched.epoch_.empty());
                }
            }
            // The identity coefficient is rank-level metadata, identical to the legacy core term.
            BOOST_TEST(sharded::validate_initial_operator<kN>(fixture.op, fixture.basis, fixture.logical)
                       == oracle.core_term());
        }
    }
}
#endif

// An oracle independent of the extracted seed (which the legacy constructor now shares): rows, coefficients, pending
// entries, inline width and reservation derived here from the routing, enumeration and store primitives alone.
BOOST_AUTO_TEST_CASE(sharded_state_seed_matches_an_independent_reference) {
    const auto options = team_options();
    const auto threads = team_size();
    for (const size_t ranks : {size_t{1}, size_t{2}}) {
        const auto router = routing::make_router<kN>(ranks, threads);
        for (const auto &fixture : all_fixtures()) {
            BOOST_TEST_CONTEXT(fixture.name << " ranks " << ranks) {
                const auto seed = make_seed(fixture, router);
                // The width rule by hand: Schrödinger takes the default; a length cutoff bounds positions, a support
                // cutoff bounds qubits (two slots each).
                const size_t width = fixture.schrodinger_cutoff ? size_t{11}
                                     : fixture.cutoff_type == CutoffType::Length
                                         ? std::min<size_t>(fixture.cutoff, 32)
                                         : std::min<size_t>(2 * fixture.cutoff, 32);
                BOOST_TEST(seed.inline_width == width);
                std::vector<Mono> paired_basis;
                size_t paired_reserve = 0;
                if (fixture.schrodinger_cutoff) {
                    const auto effective = std::min<size_t>(*fixture.schrodinger_cutoff, 2 * fixture.logical);
                    const auto pairs = (effective + 1) / 2;
                    paired_basis = generate_paired_op<kN>(pairs, fixture.logical);
                    paired_reserve = std::max<size_t>(1, paired_basis.size() / (ranks * threads));
                }
                const size_t rank = ranks - 1;
                const auto shards = sharded::seed_shards(options, seed, rank);
                for (size_t shard = 0; shard < threads; ++shard) {
                    const auto flat = (rank * threads) + shard;
                    const auto owned = [&](const Mono &mono) { return router.dest<kN>(mono) == flat; };
                    std::vector<Mono> rows;
                    std::map<Mono, double, MonoLess> coefficients;
                    for (const auto &[indices, coefficient] : fixture.op) {
                        const auto mono = indices_to_bitset<kN>(indices);
                        if (indices.empty() || !owned(mono)) {
                            continue;
                        }
                        coefficients[mono] = algebra_encode_coeff<kN>(fixture.basis, coefficient, mono);
                        if (!fixture.schrodinger_cutoff) {
                            rows.push_back(mono);
                        }
                    }
                    for (const auto &mono : paired_basis) {
                        if (owned(mono)) {
                            rows.push_back(mono);
                        }
                    }
                    auto store = monoprop::detail::OperatorIndex<kN>(width);
                    store.reserve(fixture.schrodinger_cutoff ? paired_reserve : std::max<size_t>(1, rows.size()));
                    for (size_t i = 0; i < rows.size(); ++i) {
                        store.push_back(rows[i]);
                        store.emplace(rows[i], i);
                    }

                    const auto &op = shards[shard]->op;
                    BOOST_TEST_CONTEXT("shard " << shard) {
                        BOOST_REQUIRE(op.size() == rows.size());
                        size_t wrong_rows = 0;
                        size_t wrong_coefficients = 0;
                        auto pending = coefficients;
                        for (size_t i = 0; i < rows.size(); ++i) {
                            wrong_rows += op.store->row(i) != rows[i] ? 1 : 0;
                            const auto found = pending.find(rows[i]);
                            const auto expected = found == pending.end() ? 0.0 : found->second;
                            wrong_coefficients += op.op_coeffs.at(i) != expected ? 1 : 0;
                            if (found != pending.end()) {
                                pending.erase(found);
                            }
                        }
                        BOOST_TEST(wrong_rows == 0U);
                        BOOST_TEST(wrong_coefficients == 0U);
                        BOOST_TEST(op.init_op_map.size() == pending.size());
                        for (const auto &[key, value] : pending) {
                            BOOST_TEST(op.init_op_map.count(key) == 1U);
                            BOOST_TEST((op.init_op_map.count(key) == 1 && op.init_op_map.at(key) == value));
                        }
                        BOOST_TEST(op.store->overflow_size() == store.overflow_size());
                        BOOST_TEST(op.store->memory_bytes() == store.memory_bytes());
                        BOOST_TEST(op.store->index_estimated_memory_bytes() == store.index_estimated_memory_bytes());
                        BOOST_TEST(op.initial_state == fixture.initial_state);
                    }
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_state_routed_rows_belong_to_their_flat_owner) {
    const auto options = team_options();
    const auto threads = team_size();
    struct Geometry {
        size_t ranks;
        bool linear;
    };
    for (const auto geometry : {Geometry{1, true}, Geometry{2, true}, Geometry{4, true}, Geometry{3, false}}) {
        const auto router = routing::Router::for_modes<kN>(geometry.ranks, threads, geometry.linear);
        for (const auto &fixture : all_fixtures()) {
            BOOST_TEST_CONTEXT(fixture.name << " ranks " << geometry.ranks << " linear " << geometry.linear) {
                const auto seed = make_seed(fixture, router);
                std::set<Mono, MonoLess> expected;
                if (seed.paired) {
                    for (const auto &mono : generate_paired_op<kN>(seed.paired->max_pairs, fixture.logical)) {
                        expected.insert(mono);
                    }
                }
                else {
                    for (const auto &[indices, coefficient] : fixture.op) {
                        if (!indices.empty()) {
                            expected.insert(indices_to_bitset<kN>(indices));
                        }
                    }
                }
                std::map<Mono, size_t, MonoLess> owners;
                size_t rows = 0;
                size_t misrouted = 0;
                size_t identity_pending = 0;
                size_t identity_coefficient_nonzero = 0;
                for (size_t rank = 0; rank < geometry.ranks; ++rank) {
                    // One rank's shards at a time, as each rank of the launch would hold them.
                    const auto shards = sharded::seed_shards(options, seed, rank);
                    for (size_t shard = 0; shard < threads; ++shard) {
                        const auto flat = (rank * threads) + shard;
                        const auto &op = shards[shard]->op;
                        for (size_t i = 0; i < op.size(); ++i) {
                            const auto key = op.store->row(i);
                            misrouted += router.dest<kN>(key) != flat ? 1 : 0;
                            owners.emplace(key, flat);
                            ++rows;
                            if (key.none() && op.op_coeffs.at(i) != 0.0) {
                                ++identity_coefficient_nonzero;
                            }
                        }
                        identity_pending += op.init_op_map.count(Mono{});
                    }
                }
                BOOST_TEST(misrouted == 0U);
                BOOST_TEST(rows == owners.size(), "no key is held by two owners");
                BOOST_TEST(owners.size() == expected.size());
                BOOST_TEST(std::ranges::all_of(expected, [&](const Mono &key) { return owners.contains(key); }));
                // The identity core is never seeded into a shard. Schrödinger's identity basis row is ordinary.
                BOOST_TEST(identity_pending == 0U);
                BOOST_TEST(identity_coefficient_nonzero == 0U);
                BOOST_TEST(owners.contains(Mono{}) == seed.paired.has_value());
                if (seed.paired) {
                    BOOST_TEST(owners.at(Mono{}) == router.dest<kN>(Mono{}));
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_state_empty_owners_participate, *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const auto threads = team_size();
    OperatorDict op;
    add_term(op, Basis::Majorana, {}, 1.0);
    add_term(op, Basis::Majorana, {0, 1}, 0.5);
    const auto fixture = Fixture{.name = "one-term", .op = op, .initial_state = {0, 1}};
    const auto seed = make_seed(fixture, local_router(threads));
    auto slots = std::vector<OwnerSlot>(threads);
    const auto shards = sharded::seed_shards(options, seed, 0, OwnerProbe{.slots = &slots});
    check_owner_bodies(slots, sharded::ShardWork::seed, "one-term");
    size_t empty = 0;
    for (const auto &state : shards) {
        BOOST_REQUIRE(state != nullptr);
        empty += state->size() == 0 ? 1 : 0;
        BOOST_TEST(state->op.inverted_index_.has_value());
    }
    BOOST_TEST(empty == threads - 1);
    BOOST_TEST(sharded::total_size(shards) == 1U);
}

BOOST_AUTO_TEST_CASE(sharded_state_keeps_sparse_heisenberg_and_dense_schrodinger_caches) {
    const auto options = team_options();
    const auto threads = team_size();
    size_t pending = 0;
    for (const auto &fixture : nontrivial_fixtures()) {
        BOOST_TEST_CONTEXT(fixture.name) {
            const auto shards = sharded::seed_shards(options, make_seed(fixture, local_router(threads)), 0);
            for (const auto &state : shards) {
                const auto &op = state->op;
                BOOST_TEST(op.state_scored_rows_ == op.size());
                BOOST_TEST(op.op_coeffs.size() == op.size());
                BOOST_TEST(op.inverted_index_.has_value());
                BOOST_TEST(op.inverted_index_->rows() == op.size());
                if (fixture.schrodinger_cutoff) {
                    BOOST_TEST(op.state_coeffs.size() == op.size(), "the live Schrödinger vector is dense");
                    pending += op.init_op_map.size();
                }
                else {
                    BOOST_TEST(op.state_coeffs.empty(), "Heisenberg keeps the sparse state only");
                    BOOST_TEST(op.init_op_map.empty());
                }
            }
        }
    }
    // Unpaired Hamiltonian terms have no Schrödinger row yet and stay pending.
    BOOST_TEST(pending > 0U);
}

BOOST_AUTO_TEST_CASE(sharded_state_counts_and_accounting_sum_over_shards) {
    const auto options = team_options();
    const auto threads = team_size();
    for (const auto &fixture : all_fixtures()) {
        BOOST_TEST_CONTEXT(fixture.name) {
            const auto oracle = oracle_of(fixture, threads);
            const auto shards = sharded::seed_shards(options, make_seed(fixture, local_router(threads)), 0);
            BOOST_TEST(sharded::total_size(shards) == oracle.size());
            const auto ours = sharded::operator_memory_usage(shards);
            const auto theirs = oracle.operator_memory_usage();
            BOOST_TEST(ours.total_bytes() == theirs.total_bytes());
            BOOST_TEST(ours.operator_terms_bytes == theirs.operator_terms_bytes);
            BOOST_TEST(ours.indexing_bytes == theirs.indexing_bytes);
            BOOST_TEST(ours.inverted_index_bytes == theirs.inverted_index_bytes);
            BOOST_TEST(ours.state_coeffs_bytes == theirs.state_coeffs_bytes);
            BOOST_TEST(ours.init_operator_entries == theirs.init_operator_entries);
            BOOST_TEST(ours.matched_scratch_bytes == theirs.matched_scratch_bytes);
            BOOST_TEST(sharded::graph_memory_usage(shards).total_bytes() == oracle.graph_memory_usage().total_bytes());
        }
    }
    // A built state reports its matched scratch per shard.
    auto source = built_source(threads);
    const auto shards = bridged_shards(source.propagator, threads);
    const auto ours = sharded::operator_memory_usage(shards);
    const auto theirs = source.propagator.operator_memory_usage();
    BOOST_TEST(ours.total_bytes() == theirs.total_bytes());
    BOOST_TEST(ours.matched_scratch_bytes == theirs.matched_scratch_bytes);
    BOOST_TEST(ours.matched_scratch_bytes > 0U);
    BOOST_TEST(sharded::graph_memory_usage(shards).total_bytes()
               == source.propagator.graph_memory_usage().total_bytes());
    BOOST_TEST(sharded::total_size(shards) == source.propagator.size());
}

BOOST_AUTO_TEST_CASE(sharded_state_rejects_invalid_arguments_before_the_team) {
    const auto options = team_options();
    const auto threads = team_size();
    auto calls = std::atomic<int>{0};
    const auto counting = [&](size_t) {
        calls.fetch_add(1);
        return small_state(false);
    };
    BOOST_CHECK_THROW((void)sharded::make_shards<kN>(parallel::Options{.threads = 0}, counting), std::invalid_argument);
    BOOST_CHECK_THROW((void)sharded::make_shards<kN>(parallel::Options{.threads = -3}, counting),
                      std::invalid_argument);
    BOOST_TEST(calls.load() == 0);

    // An initializer that publishes nothing is a failure, reported after the join.
    BOOST_CHECK_THROW((void)sharded::make_shards<kN>(options, [](size_t) { return std::unique_ptr<State>{}; }),
                      std::logic_error);

    const auto fixture = nontrivial_fixtures().front();
    const auto mismatched = make_seed(fixture, routing::make_router<kN>(1, threads + 1));
    BOOST_CHECK_THROW((void)sharded::seed_shards(options, mismatched, 0), std::invalid_argument);
    const auto seed = make_seed(fixture, local_router(threads));
    BOOST_CHECK_THROW((void)sharded::seed_shards(options, seed, 1), std::invalid_argument);

    auto source = sharded::seed_shards(options, seed, 0);
    BOOST_CHECK_THROW((void)sharded::copy_shards(parallel::Options{.threads = options.threads + 1}, source),
                      std::invalid_argument);
    source.back().reset();
    BOOST_CHECK_THROW((void)sharded::copy_shards(options, source), std::invalid_argument);

    // Caller-side validation of the logical-mode bound and the coefficient encoding.
    OperatorDict out_of_range;
    add_term(out_of_range, Basis::Majorana, {0, 12}, 1.0);
    BOOST_CHECK_THROW((void)sharded::validate_initial_operator<kN>(out_of_range, Basis::Majorana, 6), std::exception);
    BOOST_CHECK_NO_THROW((void)sharded::validate_initial_operator<kN>(out_of_range, Basis::Majorana, kN));
    OperatorDict non_hermitian;
    non_hermitian[VecZ{0, 1}] = std::complex<double>(1.0, 0.0);
    BOOST_CHECK_THROW((void)sharded::validate_initial_operator<kN>(non_hermitian, Basis::Majorana, kN), std::exception);
}

// --- Copying and lifecycle ------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(sharded_state_copy_is_exact_and_owner_allocated) {
    const auto options = team_options();
    const auto threads = team_size();
    auto source = built_source(threads);
    const auto shards = bridged_shards(source.propagator, threads);
    size_t overflow = 0;
    size_t layers = 0;
    for (const auto &state : shards) {
        overflow += state->op.store->overflow_size();
        layers += state->graph.layers();
    }
    BOOST_TEST_MESSAGE("copy source: rows " << sharded::total_size(shards) << ", overflow rows " << overflow
                                            << ", layers " << layers);
    BOOST_REQUIRE(overflow > 0U);
    BOOST_REQUIRE(layers > 0U);

    auto slots = std::vector<OwnerSlot>(threads);
    const auto copy = sharded::copy_shards(options, shards, OwnerProbe{.slots = &slots});
    check_owner_bodies(slots, sharded::ShardWork::copy, "copy");
    for (size_t shard = 0; shard < threads; ++shard) {
        BOOST_TEST_MESSAGE("copy owner " << shard << ": worker " << slots[shard].worker << ", bytes "
                                         << slots[shard].bytes);
    }
    const auto difference = shards_difference(copy, shards);
    BOOST_TEST(difference.empty(), difference);
    for (size_t shard = 0; shard < threads; ++shard) {
        const auto &a = *shards[shard];
        const auto &b = *copy[shard];
        BOOST_TEST_CONTEXT("shard " << shard) {
            BOOST_TEST(&a != &b);
            BOOST_TEST(a.op.store.get() != b.op.store.get());
            BOOST_TEST((a.op.op_coeffs.empty() || a.op.op_coeffs.data() != b.op.op_coeffs.data()));
            BOOST_TEST((a.matched.epoch_.empty() || a.matched.epoch_.data() != b.matched.epoch_.data()));
            // The copy's bytes were requested inside the owner's copy body: at least its packed rows.
            if (allocation::available()) {
                BOOST_TEST(slots[shard].bytes >= a.op.store->size());
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_state_copies_mutate_independently) {
    const auto options = team_options();
    const auto threads = team_size();
    auto source_propagator = built_source(threads);
    auto source = bridged_shards(source_propagator.propagator, threads);
    const auto reference = snapshot(source);
    auto copy = sharded::copy_shards(options, source);

    const auto mutate = [](Shards &shards) {
        for (auto &state : shards) {
            auto &op = state->op;
            // Touch the inverted index first so later growth has a materialized cache to diverge from.
            (void)op.inverted_index();
            for (size_t i = 0; i < op.size(); ++i) {
                if (!op.store->row_positions(i).inlined()) {
                    op.store->set(i, indices_to_bitset<kN>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
                    break;
                }
            }
            if (op.size() > 0) {
                op.store->set(0, indices_to_bitset<kN>({15}));
                op.op_coeffs.at(0) += 1.0;
            }
            op.store->push_back(indices_to_bitset<kN>({14, 15}));
            op.reindex_after_growth(op.size() - 1, 1);
            op.op_coeffs.push_back(2.0);
            op.state_coeffs.assign(op.size(), 0.5);
            op.init_op_map[indices_to_bitset<kN>({13})] = 4.0;
            state->matched.begin_gate(op.size());
            state->matched.mark(0);
            if (state->graph.layers() > 0) {
                (void)state->graph.slice_graph(1, /*contract=*/true);
            }
        }
    };

    mutate(copy);
    auto difference = shards_difference(source, reference);
    BOOST_CHECK_MESSAGE(difference.empty(), "source after copy mutation: " << difference);

    auto second = sharded::copy_shards(options, source);
    mutate(source);
    difference = shards_difference(second, reference);
    BOOST_CHECK_MESSAGE(difference.empty(), "copy after source mutation: " << difference);
    BOOST_TEST(!shards_difference(source, reference).empty(), "the mutation is visible where it was applied");
}

BOOST_AUTO_TEST_CASE(sharded_state_graph_cores_are_shared_and_lists_independent) {
    const auto options = team_options();
    const auto threads = team_size();
    auto built = std::make_unique<BuiltSource>(built_source(threads));
    auto source = bridged_shards(built->propagator, threads);
    auto copy = sharded::copy_shards(options, source);
    std::vector<std::shared_ptr<const LayerCore>> cores;
    std::vector<size_t> layer_counts;
    for (size_t shard = 0; shard < threads; ++shard) {
        const auto &a = source[shard]->graph;
        const auto &b = copy[shard]->graph;
        BOOST_REQUIRE(a.layers() == b.layers());
        layer_counts.push_back(b.layers());
        for (size_t i = 0; i < a.layers(); ++i) {
            BOOST_TEST(a.get_layer(i).shared_core() == b.get_layer(i).shared_core());
            cores.push_back(b.get_layer(i).shared_core());
        }
        // A layer-list change in the copy leaves the source's list alone.
        if (b.layers() > 0) {
            (void)copy[shard]->graph.slice_graph(1, /*contract=*/true);
            BOOST_TEST(source[shard]->graph.layers() == layer_counts.back());
            BOOST_TEST(copy[shard]->graph.layers() == layer_counts.back() - 1);
        }
    }
    const auto use_counts_before = [&] {
        std::vector<long> counts;
        for (const auto &core : cores) {
            counts.push_back(core.use_count());
        }
        return counts;
    }();
    // Destroying the source releases only its references; the copy's remaining layers stay readable.
    source.clear();
    built.reset();
    size_t index = 0;
    for (size_t shard = 0; shard < threads; ++shard) {
        const auto &graph = copy[shard]->graph;
        for (size_t i = 0; i < graph.layers(); ++i) {
            const auto traversal = graph.get_layer_traversal(i);
            BOOST_TEST(traversal.scaled_count() <= copy[shard]->op.size());
        }
        for (size_t i = 0; i < layer_counts[shard]; ++i, ++index) {
            BOOST_TEST(cores[index].use_count() < use_counts_before[index]);
        }
    }
}

BOOST_AUTO_TEST_CASE(sharded_state_copy_keeps_the_captured_budget) {
    const auto options = team_options();
    const auto threads = team_size();
    const auto fixture = nontrivial_fixtures().front();
    const auto source = sharded::seed_shards(options, make_seed(fixture, local_router(threads)), 0);
#ifdef monoprop_SHARDED_OPENMP_PROTOTYPE
    auto legacy_source = oracle_of(fixture, threads);
#else
    auto legacy_source = legacy(fixture, 1);
#endif
    const auto legacy_budget = Access::options(legacy_source).threads;

    // A reread of the environment would now fail, so neither copy may capture a fresh budget.
    const ScopedEnv poisoned("monoprop_NUM_THREADS", "0");
    BOOST_CHECK_THROW((void)parallel::capture_thread_budget(), std::invalid_argument);
    auto slots = std::vector<OwnerSlot>(threads);
    const auto copy = sharded::copy_shards(options, source, OwnerProbe{.slots = &slots});
    check_owner_bodies(slots, sharded::ShardWork::copy, "copy under a changed environment");
    BOOST_TEST(copy.size() == threads);
    const auto difference = shards_difference(copy, source);
    BOOST_TEST(difference.empty(), difference);
    // Preservation: the rank-level copy keeps the source's captured budget.
    const Propagator legacy_copy(legacy_source);
    BOOST_TEST(Access::options(legacy_copy).threads == legacy_budget);
}

// Preservation evidence for the rank-level owner guard (MonomialPropagator::require_valid_), which S1 reuses rather
// than duplicating per shard: an invalid owner's state is rejected before anything is copied, while copies made
// before the failure stay exact and usable.
#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
BOOST_AUTO_TEST_CASE(sharded_state_invalid_source_is_rejected_by_the_owner_guard) {
    const auto options = team_options();
    auto data = test_utils::load_case_data<kN>("random_exact.msgpack");
    auto sim = legacy(Fixture{.name = "guard", .op = data.hamiltonian, .initial_state = data.initial_state}, 1);
    const auto half = data.majoranas.size() / 2;
    const std::vector<VecZ> first(data.majoranas.begin(), data.majoranas.begin() + static_cast<std::ptrdiff_t>(half));
    const VecZ first_map(data.param_inds.begin(), data.param_inds.begin() + static_cast<std::ptrdiff_t>(half));
    const VecD first_gen(data.gen_coeffs.begin(), data.gen_coeffs.begin() + static_cast<std::ptrdiff_t>(half));
    sim.build_graph(first, first_map, first_gen);

    auto early = sharded::make_shards<kN>(options, [&](size_t) { return Access::shard_state(sim); });
    const auto early_reference = snapshot(early);
    const Propagator early_propagator(sim);

    Access::set_cutoff_fn(sim, [](const Mono &) -> bool { throw InjectedFailure("injected cutoff failure"); });
    BOOST_CHECK_THROW(sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs), InjectedFailure);
    BOOST_REQUIRE(Access::is_invalid(sim));

    BOOST_CHECK_THROW((void)Access::shard_state(sim), InvalidPropagatorError);
    BOOST_CHECK_THROW((void)sharded::make_shards<kN>(options, [&](size_t) { return Access::shard_state(sim); }),
                      InvalidPropagatorError);
    BOOST_CHECK_THROW(const Propagator copy(sim), InvalidPropagatorError);
    BOOST_CHECK_THROW((void)Access::clone(sim), InvalidPropagatorError);

    const auto difference = shards_difference(early, early_reference);
    BOOST_TEST(difference.empty(), difference);
    auto later = sharded::copy_shards(options, early);
    BOOST_TEST(shards_difference(later, early_reference).empty());
    BOOST_TEST(early_propagator.size() == early_reference.front()->size());
}
#endif

// --- Failure paths --------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(sharded_state_initializer_failure_on_the_primary_is_returned) {
    const auto options = team_options();
    const auto threads = team_size();
    long long net = -1;
    auto caught = false;
    {
        const allocation::LiveBytes live;
        try {
            (void)sharded::make_shards<kN>(options, [](size_t shard) {
                auto state = small_state(false); // unwinds inside the failing initializer
                if (shard == 0) {
                    throw InjectedFailure("primary initializer failure");
                }
                return state;
            });
        }
        catch (const InjectedFailure &e) {
            // Compared in place: a copied message would itself be a tracked allocation.
            caught = std::string_view(e.what()) == "primary initializer failure";
        }
        net = live.net();
    }
    BOOST_TEST(caught);
    if (allocation::available()) {
        BOOST_TEST(net == 0, "published sibling shards are released after the join");
    }
    // A fresh construction afterwards succeeds.
    const auto shards = sharded::make_shards<kN>(options, [](size_t) { return small_state(true); });
    BOOST_TEST(shards.size() == threads);
}

BOOST_AUTO_TEST_CASE(sharded_state_initializer_failure_on_a_nonprimary_owner,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const auto threads = team_size();
    for (const bool after_publication : {false, true}) {
        BOOST_TEST_CONTEXT("after publication " << after_publication) {
            auto published = std::atomic<size_t>{0};
            auto waited_out = std::atomic<bool>{false};
            long long net = -1;
            auto caught = std::optional<size_t>{};
            {
                const allocation::LiveBytes live;
                try {
                    (void)sharded::make_shards<kN>(options, [&](size_t shard) {
                        if (shard + 1 == threads) {
                            if (after_publication) {
                                // Bounded wait for every sibling to finish its state; nobody waits on this owner.
                                const auto deadline = std::chrono::steady_clock::now() + 10s;
                                while (published.load() + 1 < threads) {
                                    if (std::chrono::steady_clock::now() > deadline) {
                                        waited_out.store(true);
                                        break;
                                    }
                                    std::this_thread::yield();
                                }
                            }
                            throw WorkerError{shard};
                        }
                        auto state = small_state(false);
                        published.fetch_add(1);
                        return state;
                    });
                }
                catch (const WorkerError &e) {
                    caught = e.shard;
                }
                net = live.net();
            }
            BOOST_TEST(caught.value_or(0) == threads - 1);
            BOOST_TEST(!waited_out.load());
            if (allocation::available()) {
                BOOST_TEST(net == 0);
            }
        }
    }
    const auto shards = sharded::make_shards<kN>(options, [](size_t) { return small_state(false); });
    BOOST_TEST(sharded::total_size(shards) == threads);
}

BOOST_AUTO_TEST_CASE(sharded_state_concurrent_failures_select_the_lowest_owner,
                     *boost::unit_test::precondition(has_nonprimary_worker)) {
    const auto options = team_options();
    const auto threads = team_size();
    for (const size_t stride : {size_t{1}, size_t{2}}) {
        auto caught = std::optional<size_t>{};
        try {
            (void)sharded::make_shards<kN>(options, [&](size_t shard) {
                auto state = small_state(false);
                if (shard % stride == stride - 1) {
                    throw WorkerError{shard};
                }
                return state;
            });
        }
        catch (const WorkerError &e) {
            caught = e.shard;
        }
        BOOST_CHECK_MESSAGE(caught.value_or(threads) == stride - 1, "stride " << stride);
    }
}

// Real allocation failures at every operator new of one owner's seed body. Each must reach the caller as the
// original std::bad_alloc after the join (an allocating noexcept constructor would terminate instead), with every
// published sibling released. The sweep ends at the first count the body does not reach, which must seed exactly.
BOOST_AUTO_TEST_CASE(sharded_state_seed_allocation_failures_are_catchable,
                     *boost::unit_test::precondition(has_allocation_probe)) {
    const auto options = team_options();
    const auto threads = team_size();
    const auto fixture = nontrivial_fixtures().at(1);
    const auto seed = make_seed(fixture, local_router(threads));
    const auto reference = sharded::seed_shards(options, seed, 0);
    for (const size_t failing : std::set<size_t>{0, threads - 1}) {
        BOOST_TEST_CONTEXT("failing owner " << failing) {
            size_t failures = 0;
            size_t leaks = 0;
            size_t other_errors = 0;
            bool completed = false;
            for (size_t nth = 1; nth <= 100000 && !completed; ++nth) {
                auto slots = std::vector<OwnerSlot>(threads);
                const allocation::LiveBytes live;
                try {
                    auto shards =
                        sharded::seed_shards(options,
                                             seed,
                                             0,
                                             OwnerProbe{.slots = &slots, .fail_shard = failing, .fail_nth = nth});
                    completed = slots[failing].unfired;
                    if (completed) {
                        const auto difference = shards_difference(shards, reference);
                        BOOST_TEST(difference.empty(), difference);
                    }
                }
                catch (const std::bad_alloc &) {
                    ++failures;
                }
                catch (...) {
                    ++other_errors;
                }
                leaks += live.net() != 0 ? 1 : 0;
            }
            BOOST_TEST_MESSAGE("allocation failures injected and caught: " << failures);
            BOOST_TEST(completed);
            BOOST_TEST(failures > 0U);
            BOOST_TEST(other_errors == 0U);
            BOOST_TEST(leaks == 0U);
        }
    }
}

// The same sweep over one owner's copy body: every failure leaves the source and an earlier independent copy exact.
BOOST_AUTO_TEST_CASE(sharded_state_copy_allocation_failures_leave_sources_intact,
                     *boost::unit_test::precondition(has_allocation_probe)) {
    const auto options = team_options();
    const auto threads = team_size();
    auto built = built_source(threads);
    const auto source = bridged_shards(built.propagator, threads);
    const auto reference = snapshot(source);
    const auto earlier = sharded::copy_shards(options, source);
    for (const size_t failing : std::set<size_t>{0, threads - 1}) {
        BOOST_TEST_CONTEXT("failing owner " << failing) {
            size_t failures = 0;
            size_t leaks = 0;
            size_t other_errors = 0;
            size_t damaged = 0;
            bool completed = false;
            for (size_t nth = 1; nth <= 100000 && !completed; ++nth) {
                auto slots = std::vector<OwnerSlot>(threads);
                {
                    const allocation::LiveBytes live;
                    try {
                        auto copy =
                            sharded::copy_shards(options,
                                                 source,
                                                 OwnerProbe{.slots = &slots, .fail_shard = failing, .fail_nth = nth});
                        completed = slots[failing].unfired;
                        if (completed) {
                            const auto difference = shards_difference(copy, reference);
                            BOOST_TEST(difference.empty(), difference);
                        }
                    }
                    catch (const std::bad_alloc &) {
                        ++failures;
                    }
                    catch (...) {
                        ++other_errors;
                    }
                    leaks += live.net() != 0 ? 1 : 0;
                }
                damaged +=
                    (!shards_difference(source, reference).empty() || !shards_difference(earlier, reference).empty())
                        ? 1
                        : 0;
            }
            BOOST_TEST_MESSAGE("allocation failures injected and caught: " << failures);
            BOOST_TEST(completed);
            BOOST_TEST(failures > 0U);
            BOOST_TEST(other_errors == 0U);
            BOOST_TEST(leaks == 0U);
            BOOST_TEST(damaged == 0U);
        }
    }
}

// Preservation audit of the shared construction path: the legacy single-store constructor now seeds through the
// extracted helpers, so every allocation failure on its thread must surface as std::bad_alloc without leaking.
#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
BOOST_AUTO_TEST_CASE(sharded_state_legacy_constructor_allocation_failures_are_catchable,
                     *boost::unit_test::precondition(has_allocation_probe)) {
    for (const auto &fixture : {nontrivial_fixtures().at(0), nontrivial_fixtures().at(1)}) {
        BOOST_TEST_CONTEXT(fixture.name) {
            const auto reference = legacy(fixture, 0);
            size_t failures = 0;
            size_t leaks = 0;
            size_t other_errors = 0;
            bool completed = false;
            for (size_t nth = 1; nth <= 100000 && !completed; ++nth) {
                const allocation::LiveBytes live;
                try {
                    allocation::arm_failure(nth);
                    const auto sim = legacy(fixture, 0);
                    completed = allocation::disarm_failure();
                    if (completed) {
                        const auto difference = operator_difference(sim.mp_op(), reference.mp_op());
                        BOOST_TEST(difference.empty(), difference);
                    }
                }
                catch (const std::bad_alloc &) {
                    ++failures;
                }
                catch (...) {
                    ++other_errors;
                }
                allocation::disarm_failure();
                leaks += live.net() != 0 ? 1 : 0;
            }
            BOOST_TEST_MESSAGE("allocation failures injected and caught: " << failures);
            BOOST_TEST(completed);
            BOOST_TEST(failures > 0U);
            BOOST_TEST(other_errors == 0U);
            BOOST_TEST(leaks == 0U);
        }
    }
}
#endif
