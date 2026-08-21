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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <complex>
#include <utility>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/operator/MPOperator.h"

using namespace monoprop;
using cd = std::complex<double>;

namespace {

// Build an MPOperator whose store rows are also indexed (findable). append_term writes a row only;
// find() needs the hash index, which only the insert_absent_terms path populates.
auto build_indexed_op(const std::vector<Monomial<8>> &terms, Basis basis = Basis::Majorana) -> detail::MPOperator<8> {
    detail::MPOperator<8> op;
    op.basis = basis;
    detail::insert_absent_terms<8>(
        op,
        terms.size(),
        [&](size_t k) -> const Monomial<8> & { return terms[k]; },
        [&](size_t k, size_t base) { assign_row<8>(*op.store, base + k, terms[k]); });
    return op;
}

// Independent expected state vector: score paired rows with the basis' state phase, 0 otherwise.
auto expected_state(detail::MPOperator<8> &op, Basis basis, const VecZ &initial_state) -> VecD {
    const auto state_mask = initial_state_mask<8>(initial_state);
    VecD expected(op.size(), 0.0);
    for (size_t i = 0; i < op.size(); ++i) {
        const auto row = materialize_row<8>(*op.store, i);
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

TEST_CASE("mp_operator_get_state_scores_paired_terms_majorana_and_pauli") {
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
        CHECK(sparse_state_equals(sparse, expected_sparse_state(op, basis, initial_state)));
        REQUIRE((sparse.rows.size()) == (2U)); // rows 0 and 1; the unpaired row 2 is absent
        CHECK((sparse.rows[0]) == (0U));
        CHECK((sparse.rows[1]) == (1U));

        const VecD state = op.materialize_state();
        REQUIRE((state.size()) == (3U));
        CHECK(state == expected_state(op, basis, initial_state));
        CHECK(op.dense_state() == state);

        // Structural, oracle-independent: paired rows carry a unit phase, the unpaired row is zero.
        CHECK((std::abs(state[0])) == (1.0));
        CHECK((std::abs(state[1])) == (1.0));
        CHECK((state[2]) == (0.0));
    }
}

TEST_CASE("mp_operator_get_state_scores_only_new_terms_incrementally") {
    const VecZ initial_state = {0};
    detail::MPOperator<8> op;
    op.initial_state = initial_state;

    Monomial<8> a;
    a.set(0);
    a.set(1); // paired
    op.append_term(a);
    const VecD first = op.dense_state();
    REQUIRE((first.size()) == (1U));
    const double a_score = first[0];
    CHECK((op.state_scored_rows_) == (1U));

    // Stands in for evolution mutating the live vector: the incremental pass must not rewrite an
    // already-scored row, so this value has to survive.
    op.state_coeffs[0] = 7.5;

    Monomial<8> b;
    b.set(2);
    b.set(3); // paired
    op.append_term(b);
    const VecD second = op.dense_state(); // must score only row 1, leave row 0 untouched
    REQUIRE((second.size()) == (2U));
    CHECK((second[0]) == (7.5));
    CHECK((second[1]) == (expected_state(op, Basis::Majorana, initial_state)[1]));

    // The sparse set was extended, not rebuilt: row 0 still carries its original state score.
    const auto sparse = op.sparse_state();
    CHECK(sparse_state_equals(sparse, expected_sparse_state(op, Basis::Majorana, initial_state)));
    REQUIRE((sparse.rows.size()) == (2U));
    CHECK((sparse.values[0]) == (a_score));

    // Idempotent when nothing was appended.
    CHECK(op.dense_state() == second);
}

TEST_CASE("mp_operator_get_operator_drains_present_terms_from_init_map") {
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({a, b});

    const auto absent = indices_to_bitset<8>({4, 5});
    op.init_op_map[a] = 3.0;      // present in store -> should land on row 0 and be erased
    op.init_op_map[absent] = 9.0; // absent from store -> stays pending

    const VecD &coeffs = op.get_operator();
    REQUIRE((coeffs.size()) == (2U));
    CHECK((coeffs[0]) == (3.0));
    CHECK((coeffs[1]) == (0.0));                                // b was not in the init map
    CHECK(op.init_op_map.find(a) == op.init_op_map.end());      // drained
    CHECK(op.init_op_map.find(absent) != op.init_op_map.end()); // retained

    // Second call is a no-op fast path (size already matches).
    CHECK(op.get_operator() == coeffs);
}

TEST_CASE("mp_operator_update_initial_operator_heisenberg_branches_pauli") {
    const auto present = indices_to_bitset<8>({0, 2});
    auto op = build_indexed_op({present}, Basis::Pauli); // row 0 indexed
    op.init_op_map[indices_to_bitset<8>({4, 6})] = 0.0;  // seed a pending term

    OperatorDict dict;
    dict[VecZ{0, 2}] = cd(1.5, 0.0); // present in store -> row coeff
    dict[VecZ{4, 6}] = cd(2.5, 0.0); // in init_op_map -> stays pending

    const auto grad = op.update_initial_operator(dict, /*schrodinger=*/false);
    REQUIRE((op.op_coeffs.size()) == (1U));
    CHECK((op.op_coeffs[0]) == (encode_pauli_coeff(cd(1.5, 0.0)))); // Pauli encode path
    CHECK(op.init_op_map.find(indices_to_bitset<8>({4, 6})) != op.init_op_map.end());
    CHECK(op.init_op_map.find(present) == op.init_op_map.end());
    CHECK((grad.first.size()) == (2U)); // every supplied term recorded in the grad arrays
}

TEST_CASE("mp_operator_update_initial_operator_heisenberg_rejects_absent_term") {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 2})}, Basis::Pauli);

    OperatorDict dict;
    dict[VecZ{1, 3, 5}] = cd(1.0, 0.0); // absent from both store and init_op_map
    CHECK_THROWS_AS(op.update_initial_operator(dict, /*schrodinger=*/false), std::runtime_error);
}

TEST_CASE("mp_operator_update_initial_operator_schrodinger_admits_absent_term") {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 2})}, Basis::Pauli);

    OperatorDict dict;
    const auto fresh = indices_to_bitset<8>({1, 3, 5});
    dict[VecZ{1, 3, 5}] = cd(4.0, 0.0);
    op.update_initial_operator(dict, /*schrodinger=*/true);
    CHECK(op.init_op_map.find(fresh) != op.init_op_map.end());
}

TEST_CASE("mp_operator_update_initial_operator_majorana_encode_identity_term") {
    // The Majorana codec divides by the term's hermitian phase, which is 1 for the identity term, so
    // a real coefficient round-trips as itself without tripping the non-Hermitian guard.
    const Monomial<8> identity;             // empty
    auto op = build_indexed_op({identity}); // basis defaults to Majorana

    OperatorDict dict;
    dict[VecZ{}] = cd(2.75, 0.0);
    op.update_initial_operator(dict, /*schrodinger=*/false);
    REQUIRE((op.op_coeffs.size()) == (1U));
    CHECK((op.op_coeffs[0]) == (algebra_encode_coeff<8>(Basis::Majorana, cd(2.75, 0.0), identity)));
    CHECK((op.op_coeffs[0]) == (2.75));
}

TEST_CASE("mp_operator_insert_absent_terms_grows_and_indexes") {
    const auto e0 = indices_to_bitset<8>({0, 1});
    const auto e1 = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({e0, e1});

    const std::vector<Monomial<8>> fresh = {indices_to_bitset<8>({4, 5}),
                                            indices_to_bitset<8>({6, 7}),
                                            indices_to_bitset<8>({0, 3})};

    const size_t base = detail::insert_absent_terms<8>(
        op,
        fresh.size(),
        [&](size_t k) -> const Monomial<8> & { return fresh[k]; },
        [&](size_t k, size_t b) { assign_row<8>(*op.store, b + k, fresh[k]); });

    CHECK((base) == (2U));
    CHECK((op.size()) == (5U));
    for (const auto &f : fresh) {
        CHECK(op.store->find(f).has_value());
    }
    CHECK(op.store->find(e0).has_value()); // existing rows intact
    CHECK(op.store->find(e1).has_value());
}

TEST_CASE("mp_operator_append_term_after_materialization_rebuilds_inverted_index") {
    detail::MPOperator<8> op;
    op.append_term(indices_to_bitset<8>({0, 1}));
    CHECK((op.inverted_index().rows()) == (1U)); // materializes the index (rows == size)

    // append_term does not sync the index; the next inverted_index() sees rows() != store size and
    // rebuilds against the grown store.
    op.append_term(indices_to_bitset<8>({2, 3}));
    CHECK((op.inverted_index().rows()) == (2U));
}

TEST_CASE("mp_operator_estimate_memory_usage_tracks_inverted_index_presence") {
    detail::MPOperator<8> op;
    op.append_term(indices_to_bitset<8>({0, 1}));
    op.append_term(indices_to_bitset<8>({2, 3}));

    const auto before = detail::estimate_memory_usage<8>(op);
    CHECK((before.total_bytes()) > (0U));
    CHECK((before.operator_terms_bytes) > (0U));
    CHECK((before.inverted_index_bytes) == (0U)); // absent arm

    (void)op.inverted_index();
    const auto after = detail::estimate_memory_usage<8>(op);
    CHECK((after.inverted_index_bytes) > (0U)); // present arm
}

TEST_CASE("mp_operator_copy_constructor_clones_store_and_coeffs") {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 1}), indices_to_bitset<8>({2, 3})});
    op.initial_state = {0};
    (void)op.sparse_state();

    detail::MPOperator<8> copy(op); // deep copy via clone()
    CHECK((copy.size()) == (op.size()));
    CHECK((copy.state_scored_rows_) == (op.state_scored_rows_));
    CHECK(copy.state_rows_ == op.state_rows_);
    CHECK(copy.state_vals_ == op.state_vals_);
    CHECK(copy.materialize_state() == op.materialize_state());
    CHECK(copy.store->find(indices_to_bitset<8>({0, 1})).has_value());
    // Mutating the copy must not touch the original (independent stores).
    copy.append_term(indices_to_bitset<8>({4, 5}));
    CHECK((op.size()) == (2U));
    CHECK((copy.size()) == (3U));
}
