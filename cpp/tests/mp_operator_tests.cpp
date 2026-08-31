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

// White-box tests for detail::MPOperator<N> built directly (append_term / init_op_map / basis) rather
// than through a simulator, with the algebra primitives (is_paired, algebra_state_phase,
// encode_pauli_coeff) as the oracle. They pin composition -- incremental scoring, slot placement, the
// init-map drain, the picture/basis branches -- not the phase math (majorana_cutoff_tests.cpp).

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <complex>
#include <optional>
#include <utility>
#include <vector>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

#include "TestOperator.h"

using namespace monoprop;
using cd = std::complex<double>;

namespace {

auto build_indexed_op(const std::vector<Monomial<8>> &terms, Basis basis = Basis::Majorana) -> detail::MPOperator<8> {
    return test_utils::indexed_operator<8>(terms, basis);
}

// Independent expected state vector: score paired rows with the basis' state phase, 0 otherwise.
auto expected_state(detail::MPOperator<8> &op, Basis basis, const VecZ &initial_state) -> VecD {
    const auto state_mask = initial_state_mask<8>(initial_state);
    VecD expected(op.size(), 0.0);
    for (size_t i = 0; i < op.size(); ++i) {
        const auto row = op.with_store([i](const auto &rows) { return materialize_row<8>(rows, i); });
        if (is_paired<8>(row)) {
            expected[i] = algebra_state_phase<8>(basis, row, state_mask);
        }
    }
    return expected;
}

// Independent expected sparse state: ascending rows that score nonzero, and their phases.
auto expected_sparse_state(detail::MPOperator<8> &op, Basis basis, const VecZ &initial_state) -> std::pair<VecZ, VecD> {
    const auto dense = expected_state(op, basis, initial_state);
    std::pair<VecZ, VecD> expected;
    for (size_t i = 0; i < dense.size(); ++i) {
        if (dense[i] != 0.0) {
            expected.first.push_back(i);
            expected.second.push_back(dense[i]);
        }
    }
    return expected;
}

auto sparse_state_equals(const detail::MPOperator<8>::SparseState &sparse, const std::pair<VecZ, VecD> &expected)
    -> bool {
    return std::ranges::equal(sparse.rows, expected.first, {}, [](TermIndex r) { return static_cast<size_t>(r); })
           && std::ranges::equal(sparse.values, expected.second);
}

} // namespace

BOOST_AUTO_TEST_CASE(mp_operator_get_state_scores_paired_terms_majorana_and_pauli) {
    const VecZ initial_state = {0, 1}; // occupied modes
    for (const Basis basis : {Basis::Majorana, Basis::Pauli}) {
        detail::MPOperator<8> op;
        op.basis = basis;
        op.initial_state = initial_state;

        Monomial<8> identity;     // empty -> paired
        Monomial<8> paired_mode0; // raw bits {0,1} -> mode 0 paired
        paired_mode0.set(0);
        paired_mode0.set(1);
        Monomial<8> unpaired; // raw bit {0} only -> not paired
        unpaired.set(0);

        op.append_term(identity);
        op.append_term(paired_mode0);
        op.append_term(unpaired);

        // The sparse form is the resting representation: paired rows only, ascending.
        const auto sparse = op.sparse_state();
        BOOST_CHECK(sparse_state_equals(sparse, expected_sparse_state(op, basis, initial_state)));
        BOOST_REQUIRE_EQUAL(sparse.rows.size(), 2U); // rows 0 and 1; the unpaired row 2 is absent
        BOOST_CHECK_EQUAL(sparse.rows[0], 0U);
        BOOST_CHECK_EQUAL(sparse.rows[1], 1U);

        const VecD state = op.materialize_state();
        BOOST_REQUIRE_EQUAL(state.size(), 3U);
        BOOST_CHECK(state == expected_state(op, basis, initial_state));
        BOOST_CHECK(op.dense_state() == state);

        // Structural, oracle-independent: paired rows carry a unit phase, the unpaired row is zero.
        BOOST_CHECK_EQUAL(std::abs(state[0]), 1.0);
        BOOST_CHECK_EQUAL(std::abs(state[1]), 1.0);
        BOOST_CHECK_EQUAL(state[2], 0.0);
    }
}

BOOST_AUTO_TEST_CASE(mp_operator_get_state_scores_only_new_terms_incrementally) {
    const VecZ initial_state = {0};
    detail::MPOperator<8> op;
    op.initial_state = initial_state;

    Monomial<8> a;
    a.set(0);
    a.set(1); // paired
    op.append_term(a);
    const VecD first = op.dense_state();
    BOOST_REQUIRE_EQUAL(first.size(), 1U);
    const double a_score = first[0];
    BOOST_CHECK_EQUAL(op.state_scored_rows_, 1U);

    // Stands in for evolution mutating the live vector: the incremental pass must not rewrite an
    // already-scored row, so this value has to survive.
    op.state_coeffs[0] = 7.5;

    Monomial<8> b;
    b.set(2);
    b.set(3); // paired
    op.append_term(b);
    const VecD second = op.dense_state(); // must score only row 1, leave row 0 untouched
    BOOST_REQUIRE_EQUAL(second.size(), 2U);
    BOOST_CHECK_EQUAL(second[0], 7.5);
    BOOST_CHECK_EQUAL(second[1], expected_state(op, Basis::Majorana, initial_state)[1]);

    // The sparse set was extended, not rebuilt: row 0 still carries its original state score.
    const auto sparse = op.sparse_state();
    BOOST_CHECK(sparse_state_equals(sparse, expected_sparse_state(op, Basis::Majorana, initial_state)));
    BOOST_REQUIRE_EQUAL(sparse.rows.size(), 2U);
    BOOST_CHECK_EQUAL(sparse.values[0], a_score);

    // Idempotent when nothing was appended.
    BOOST_CHECK(op.dense_state() == second);
}

BOOST_AUTO_TEST_CASE(mp_operator_get_operator_drains_present_terms_from_init_map) {
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({a, b});

    const auto absent = indices_to_bitset<8>({4, 5});
    op.init_op_map[a] = 3.0;      // present in store -> should land on row 0 and be erased
    op.init_op_map[absent] = 9.0; // absent from store -> stays pending

    const VecD &coeffs = op.get_operator();
    BOOST_REQUIRE_EQUAL(coeffs.size(), 2U);
    BOOST_CHECK_EQUAL(coeffs[0], 3.0);
    BOOST_CHECK_EQUAL(coeffs[1], 0.0);                                // b was not in the init map
    BOOST_CHECK(op.init_op_map.find(a) == op.init_op_map.end());      // drained
    BOOST_CHECK(op.init_op_map.find(absent) != op.init_op_map.end()); // retained

    // Second call is a no-op fast path (size already matches).
    BOOST_CHECK(op.get_operator() == coeffs);
}

// erase/clear leave bucket_count(), so init_operator_bytes must fall, not just the entry count.
BOOST_AUTO_TEST_CASE(mp_operator_get_operator_releases_init_map_when_fully_bound) {
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({a, b});

    op.init_op_map.reserve(4096); // buckets far in excess of the two live entries
    op.init_op_map[a] = 3.0;
    op.init_op_map[b] = 5.0;

    const auto before = detail::estimate_memory_usage<8>(op);
    BOOST_REQUIRE_EQUAL(before.init_operator_entries, 2U);

    const VecD &coeffs = op.get_operator();
    BOOST_REQUIRE_EQUAL(coeffs.size(), 2U);
    BOOST_CHECK_EQUAL(coeffs[0], 3.0);
    BOOST_CHECK_EQUAL(coeffs[1], 5.0);

    const auto after = detail::estimate_memory_usage<8>(op);
    BOOST_CHECK_EQUAL(after.init_operator_entries, 0U);
    BOOST_CHECK_LT(after.init_operator_bytes, before.init_operator_bytes);
    BOOST_CHECK_EQUAL(op.init_op_map.bucket_count(), 0U); // released, not merely shrunk to the minimum
}

// Partial bind: the pending entry survives with its value and the bucket array shrinks to the remainder.
BOOST_AUTO_TEST_CASE(mp_operator_get_operator_shrinks_init_map_when_partially_bound) {
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    const auto absent = indices_to_bitset<8>({4, 5});
    auto op = build_indexed_op({a, b});

    op.init_op_map.reserve(4096);
    op.init_op_map[a] = 3.0;
    op.init_op_map[absent] = 9.0;

    const auto before = detail::estimate_memory_usage<8>(op);
    (void)op.get_operator();
    const auto after = detail::estimate_memory_usage<8>(op);

    BOOST_REQUIRE_EQUAL(after.init_operator_entries, 1U);
    const auto found = op.init_op_map.find(absent);
    BOOST_REQUIRE(found != op.init_op_map.end());
    BOOST_CHECK_EQUAL(found->second, 9.0);
    BOOST_CHECK(op.init_op_map.find(a) == op.init_op_map.end());
    BOOST_CHECK_LT(after.init_operator_bytes, before.init_operator_bytes);
}

// Nothing bound: the map is left exactly as it was, buckets included.
BOOST_AUTO_TEST_CASE(mp_operator_get_operator_keeps_init_map_when_nothing_bound) {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 1})});

    const auto absent = indices_to_bitset<8>({4, 5});
    op.init_op_map[absent] = 9.0;

    (void)op.get_operator();
    const auto after = detail::estimate_memory_usage<8>(op);
    BOOST_CHECK_EQUAL(after.init_operator_entries, 1U);
    BOOST_CHECK(op.init_op_map.find(absent) != op.init_op_map.end());
}

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_heisenberg_branches_pauli) {
    const auto present = indices_to_bitset<8>({0, 2});
    auto op = build_indexed_op({present}, Basis::Pauli); // row 0 indexed
    op.init_op_map[indices_to_bitset<8>({4, 6})] = 0.0;  // seed a pending term

    OperatorDict dict;
    dict[VecZ{0, 2}] = cd(1.5, 0.0); // present in store -> row coeff
    dict[VecZ{4, 6}] = cd(2.5, 0.0); // in init_op_map -> stays pending

    const auto grad = op.update_initial_operator(dict, /*schrodinger=*/false);
    BOOST_REQUIRE_EQUAL(op.op_coeffs.size(), 1U);
    BOOST_CHECK_EQUAL(op.op_coeffs[0], encode_pauli_coeff(cd(1.5, 0.0))); // Pauli encode path
    BOOST_CHECK(op.init_op_map.find(indices_to_bitset<8>({4, 6})) != op.init_op_map.end());
    BOOST_CHECK(op.init_op_map.find(present) == op.init_op_map.end());
    BOOST_CHECK_EQUAL(grad.first.size(), 2U); // every supplied term recorded in the grad arrays
}

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_heisenberg_rejects_absent_term) {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 2})}, Basis::Pauli);

    OperatorDict dict;
    dict[VecZ{1, 3, 5}] = cd(1.0, 0.0); // absent from both store and init_op_map
    BOOST_CHECK_THROW(op.update_initial_operator(dict, /*schrodinger=*/false), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_schrodinger_admits_absent_term) {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 2})}, Basis::Pauli);

    OperatorDict dict;
    const auto fresh = indices_to_bitset<8>({1, 3, 5});
    dict[VecZ{1, 3, 5}] = cd(4.0, 0.0);
    op.update_initial_operator(dict, /*schrodinger=*/true);
    BOOST_CHECK(op.init_op_map.find(fresh) != op.init_op_map.end());
}

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_majorana_encode_identity_term) {
    // The Majorana codec divides by the term's hermitian phase, which is 1 for the identity term, so
    // a real coefficient round-trips as itself without tripping the non-Hermitian guard.
    const Monomial<8> identity;             // empty
    auto op = build_indexed_op({identity}); // basis defaults to Majorana

    OperatorDict dict;
    dict[VecZ{}] = cd(2.75, 0.0);
    op.update_initial_operator(dict, /*schrodinger=*/false);
    BOOST_REQUIRE_EQUAL(op.op_coeffs.size(), 1U);
    BOOST_CHECK_EQUAL(op.op_coeffs[0], algebra_encode_coeff<8>(Basis::Majorana, cd(2.75, 0.0), identity));
    BOOST_CHECK_EQUAL(op.op_coeffs[0], 2.75);
}

BOOST_AUTO_TEST_CASE(mp_operator_insert_absent_terms_grows_and_indexes) {
    const auto e0 = indices_to_bitset<8>({0, 1});
    const auto e1 = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({e0, e1});

    const std::vector<Monomial<8>> fresh = {indices_to_bitset<8>({4, 5}),
                                            indices_to_bitset<8>({6, 7}),
                                            indices_to_bitset<8>({0, 3})};

    const size_t base = op.with_store([&](auto &rows) {
        return detail::insert_absent_terms<8>(
            op,
            rows,
            fresh.size(),
            [&](size_t k) -> const Monomial<8> & { return fresh[k]; },
            [&](size_t k, size_t b) { assign_row<8>(rows, b + k, fresh[k]); });
    });

    BOOST_CHECK_EQUAL(base, 2U);
    BOOST_CHECK_EQUAL(op.size(), 5U);
    for (const auto &f : fresh) {
        BOOST_CHECK(op.find(f).has_value());
    }
    BOOST_CHECK(op.find(e0).has_value()); // existing rows intact
    BOOST_CHECK(op.find(e1).has_value());
}

BOOST_AUTO_TEST_CASE(mp_operator_append_term_after_materialization_rebuilds_inverted_index) {
    detail::MPOperator<8> op;
    op.append_term(indices_to_bitset<8>({0, 1}));
    BOOST_CHECK_EQUAL(op.inverted_index().rows(), 1U); // materializes the index (rows == size)

    // append_term does not sync the index; the next inverted_index() sees rows() != store size and
    // rebuilds against the grown store.
    op.append_term(indices_to_bitset<8>({2, 3}));
    BOOST_CHECK_EQUAL(op.inverted_index().rows(), 2U);
}

BOOST_AUTO_TEST_CASE(mp_operator_estimate_memory_usage_tracks_inverted_index_presence) {
    detail::MPOperator<8> op;
    op.append_term(indices_to_bitset<8>({0, 1}));
    op.append_term(indices_to_bitset<8>({2, 3}));

    const auto before = detail::estimate_memory_usage<8>(op);
    BOOST_CHECK_GT(before.total_bytes(), 0U);
    BOOST_CHECK_GT(before.operator_terms_bytes, 0U);
    BOOST_CHECK_EQUAL(before.inverted_index_bytes, 0U); // absent arm

    (void)op.inverted_index();
    const auto after = detail::estimate_memory_usage<8>(op);
    BOOST_CHECK_GT(after.inverted_index_bytes, 0U); // present arm
}

// matched_scratch_bytes is summed by total_bytes() and accumulated by operator+= for the facade's sum.
BOOST_AUTO_TEST_CASE(mp_operator_breakdown_counts_matched_scratch_in_total_and_sum) {
    detail::MPOperatorMemoryBreakdown<8> acc;
    acc.op_coeffs_bytes = 100;
    acc.matched_scratch_bytes = 7;
    BOOST_CHECK_EQUAL(acc.total_bytes(), 107U);

    detail::MPOperatorMemoryBreakdown<8> other;
    other.op_coeffs_bytes = 20;
    other.matched_scratch_bytes = 3;

    acc += other;
    BOOST_CHECK_EQUAL(acc.matched_scratch_bytes, 10U);
    BOOST_CHECK_EQUAL(acc.total_bytes(), 130U);

    // An operator on its own has no stamp array to report.
    auto bare = build_indexed_op({indices_to_bitset<8>({0, 1})});
    BOOST_CHECK_EQUAL(detail::estimate_memory_usage<8>(bare).matched_scratch_bytes, 0U);
}

// epoch_ is empty until the first begin_gate, so this must apply a gate before the bytes can be nonzero.
BOOST_AUTO_TEST_CASE(mp_operator_breakdown_matched_scratch_nonzero_after_a_gate) {
    constexpr size_t kModes = 2;
    OperatorDict ham;
    ham[VecZ{0, 1}] = cd{0.0, 1.0};
    VecZ initial_state{0, 1};
    auto sim = MonomialPropagator<kModes>(ham,
                                          2 * kModes,
                                          initial_state,
                                          std::nullopt,
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Length,
                                          std::nullopt);
    BOOST_CHECK_EQUAL(sim.operator_memory_usage().matched_scratch_bytes, 0U); // no gate applied yet

    const std::vector<VecZ> monos{{0}};
    sim.build_graph(monos, VecZ{0}, VecD{1.0});

    const auto live = sim.operator_memory_usage();
    BOOST_CHECK_GT(live.matched_scratch_bytes, 0U);

    auto without = live;
    without.matched_scratch_bytes = 0;
    BOOST_CHECK_EQUAL(live.total_bytes() - without.total_bytes(), live.matched_scratch_bytes);
}

// init_operator_entries is a count: accumulated by operator+= but never summed into total_bytes().
BOOST_AUTO_TEST_CASE(mp_operator_breakdown_keeps_init_operator_entries_out_of_total) {
    detail::MPOperatorMemoryBreakdown<8> acc;
    acc.op_coeffs_bytes = 100;
    acc.init_operator_entries = 2;
    BOOST_CHECK_EQUAL(acc.total_bytes(), 100U);

    detail::MPOperatorMemoryBreakdown<8> other;
    other.op_coeffs_bytes = 20;
    other.init_operator_entries = 5;

    acc += other;
    BOOST_CHECK_EQUAL(acc.init_operator_entries, 7U);
    BOOST_CHECK_EQUAL(acc.total_bytes(), 120U);
}

BOOST_AUTO_TEST_CASE(mp_operator_copy_constructor_clones_store_and_coeffs) {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 1}), indices_to_bitset<8>({2, 3})});
    op.initial_state = {0};
    (void)op.sparse_state();

    detail::MPOperator<8> copy(op); // deep copy via clone()
    BOOST_CHECK_EQUAL(copy.size(), op.size());
    BOOST_CHECK_EQUAL(copy.state_scored_rows_, op.state_scored_rows_);
    BOOST_CHECK(copy.state_rows_ == op.state_rows_);
    BOOST_CHECK(copy.state_vals_ == op.state_vals_);
    BOOST_CHECK(copy.materialize_state() == op.materialize_state());
    BOOST_CHECK(copy.find(indices_to_bitset<8>({0, 1})).has_value());
    // Mutating the copy must not touch the original (independent stores).
    copy.append_term(indices_to_bitset<8>({4, 5}));
    BOOST_CHECK_EQUAL(op.size(), 2U);
    BOOST_CHECK_EQUAL(copy.size(), 3U);
}
