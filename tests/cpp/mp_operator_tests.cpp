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

// White-box tests for detail::MPOperator<N>, built directly (append_term / init_op_map / basis)
// rather than through a full simulator. Oracles are the independent algebra primitives
// (is_paired, algebra_hf_phase, encode_pauli_coeff), so these verify MPOperator's COMPOSITION
// (incremental scoring, slot placement, the init-map drain, the picture/basis branches) rather
// than re-deriving the phase math (covered in majorana_cutoff_tests.cpp).

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <complex>
#include <utility>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/operator/MPOperator.h"

using namespace monoprop;
using cd = std::complex<double>;

namespace {

// Build an MPOperator whose store rows are also INDEXED (findable). The row store and the keyless
// hash index are separate in OperatorIndex: push_back/append_term writes a row only, while find()
// needs the index that bulk_insert populates. insert_absent_terms is the production grow→assign→
// bulk_insert path, so it yields a store where find() works — required by every find-driven method.
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

// Independent expected state vector: score paired rows with the basis' HF phase, 0 otherwise.
auto expected_state(detail::MPOperator<8> &op, Basis basis, const VecZ &hf) -> VecD {
    const auto hf_mask = get_hf_mask<8>(hf);
    VecD expected(op.size(), 0.0);
    for (size_t i = 0; i < op.size(); ++i) {
        const auto row = materialize_row<8>(*op.store, i);
        if (is_paired<8>(row)) {
            expected[i] = algebra_hf_phase<8>(basis, row, hf_mask);
        }
    }
    return expected;
}

// Independent expected SPARSE state: the ascending rows that score nonzero, and their phases.
auto expected_sparse_state(detail::MPOperator<8> &op, Basis basis, const VecZ &hf) -> std::pair<VecZ, VecD> {
    const auto dense = expected_state(op, basis, hf);
    std::pair<VecZ, VecD> expected;
    for (size_t i = 0; i < dense.size(); ++i) {
        if (dense[i] != 0.0) {
            expected.first.push_back(i);
            expected.second.push_back(dense[i]);
        }
    }
    return expected;
}

// Compare a SparseState view against (rows, values) oracle vectors.
auto sparse_state_equals(const detail::MPOperator<8>::SparseState &sparse, const std::pair<VecZ, VecD> &expected)
    -> bool {
    return std::ranges::equal(sparse.rows, expected.first, {}, [](TermIndex r) { return static_cast<size_t>(r); })
           && std::ranges::equal(sparse.values, expected.second);
}

} // namespace

// ── state scoring: paired-only, ±1 phases, both algebra branches, sparse and dense surfaces ──────

BOOST_AUTO_TEST_CASE(mp_operator_get_state_scores_paired_terms_majorana_and_pauli) {
    const VecZ hf = {0, 1}; // occupied modes
    for (const Basis basis : {Basis::Majorana, Basis::Pauli}) {
        detail::MPOperator<8> op;
        op.basis = basis;
        op.slater_determinant = hf;

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
        BOOST_CHECK(sparse_state_equals(sparse, expected_sparse_state(op, basis, hf)));
        BOOST_REQUIRE_EQUAL(sparse.rows.size(), 2U); // rows 0 and 1; the unpaired row 2 is absent
        BOOST_CHECK_EQUAL(sparse.rows[0], 0U);
        BOOST_CHECK_EQUAL(sparse.rows[1], 1U);

        // Both dense surfaces must scatter to exactly the same vector.
        const VecD state = op.materialize_state();
        BOOST_REQUIRE_EQUAL(state.size(), 3U);
        BOOST_CHECK(state == expected_state(op, basis, hf));
        BOOST_CHECK(op.dense_state() == state);

        // Structural, oracle-independent: paired rows carry a unit phase, the unpaired row is zero.
        BOOST_CHECK_EQUAL(std::abs(state[0]), 1.0);
        BOOST_CHECK_EQUAL(std::abs(state[1]), 1.0);
        BOOST_CHECK_EQUAL(state[2], 0.0);
    }
}

BOOST_AUTO_TEST_CASE(mp_operator_get_state_scores_only_new_terms_incrementally) {
    const VecZ hf = {0};
    detail::MPOperator<8> op;
    op.slater_determinant = hf;

    Monomial<8> a;
    a.set(0);
    a.set(1); // paired
    op.append_term(a);
    const VecD first = op.dense_state(); // scores row 0
    BOOST_REQUIRE_EQUAL(first.size(), 1U);
    const double a_score = first[0];
    BOOST_CHECK_EQUAL(op.hf_scored_rows_, 1U);

    // Stand in for evolution mutating the live (Schrödinger) vector; the incremental pass must not
    // rewrite an already-scored row, so this value has to survive.
    op.state_coeffs[0] = 7.5;

    Monomial<8> b;
    b.set(2);
    b.set(3); // paired
    op.append_term(b);
    const VecD second = op.dense_state(); // must score only row 1, leave row 0 untouched
    BOOST_REQUIRE_EQUAL(second.size(), 2U);
    BOOST_CHECK_EQUAL(second[0], 7.5); // unchanged
    BOOST_CHECK_EQUAL(second[1], expected_state(op, Basis::Majorana, hf)[1]);

    // The sparse set was EXTENDED, not rebuilt: row 0 still carries its original HF score.
    const auto sparse = op.sparse_state();
    BOOST_CHECK(sparse_state_equals(sparse, expected_sparse_state(op, Basis::Majorana, hf)));
    BOOST_REQUIRE_EQUAL(sparse.rows.size(), 2U);
    BOOST_CHECK_EQUAL(sparse.values[0], a_score);

    // Idempotent when nothing was appended.
    BOOST_CHECK(op.dense_state() == second);
}

// ── get_operator: lazy sizing + init-map drain ───────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(mp_operator_get_operator_drains_present_terms_from_init_map) {
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({a, b}); // rows 0,1 indexed

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

// ── update_initial_operator: picture branches + Pauli coeff encode ───────────────────────────────

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_heisenberg_branches_pauli) {
    const auto present = indices_to_bitset<8>({0, 2});
    auto op = build_indexed_op({present}, Basis::Pauli); // row 0 indexed
    op.init_op_map[indices_to_bitset<8>({4, 6})] = 0.0;  // seed a pending term

    FermiOperatorMap dict;
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
    // store has only this term, init_op_map empty
    auto op = build_indexed_op({indices_to_bitset<8>({0, 2})}, Basis::Pauli);

    FermiOperatorMap dict;
    dict[VecZ{1, 3, 5}] = cd(1.0, 0.0); // absent from BOTH store and init_op_map
    BOOST_CHECK_THROW(op.update_initial_operator(dict, /*schrodinger=*/false), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_schrodinger_admits_absent_term) {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 2})}, Basis::Pauli);

    FermiOperatorMap dict;
    const auto fresh = indices_to_bitset<8>({1, 3, 5});
    dict[VecZ{1, 3, 5}] = cd(4.0, 0.0);
    // Schrödinger admits an unknown term (goes to pending) rather than throwing.
    op.update_initial_operator(dict, /*schrodinger=*/true);
    BOOST_CHECK(op.init_op_map.find(fresh) != op.init_op_map.end());
}

BOOST_AUTO_TEST_CASE(mp_operator_update_initial_operator_majorana_encode_identity_term) {
    // The Majorana codec divides by the term's hermitian phase; for the identity term that phase is
    // 1, so a real coefficient round-trips as itself without tripping the non-Hermitian guard.
    const Monomial<8> identity;             // empty
    auto op = build_indexed_op({identity}); // basis defaults to Majorana

    FermiOperatorMap dict;
    dict[VecZ{}] = cd(2.75, 0.0);
    op.update_initial_operator(dict, /*schrodinger=*/false);
    BOOST_REQUIRE_EQUAL(op.op_coeffs.size(), 1U);
    BOOST_CHECK_EQUAL(op.op_coeffs[0], algebra_encode_coeff<8>(Basis::Majorana, cd(2.75, 0.0), identity));
    BOOST_CHECK_EQUAL(op.op_coeffs[0], 2.75);
}

// ── insert_absent_terms / inverted index / memory estimate / copy ────────────────────────────────

BOOST_AUTO_TEST_CASE(mp_operator_insert_absent_terms_grows_and_indexes) {
    const auto e0 = indices_to_bitset<8>({0, 1});
    const auto e1 = indices_to_bitset<8>({2, 3});
    auto op = build_indexed_op({e0, e1}); // two existing indexed rows

    const std::vector<Monomial<8>> fresh = {indices_to_bitset<8>({4, 5}),
                                            indices_to_bitset<8>({6, 7}),
                                            indices_to_bitset<8>({0, 3})};

    const size_t base = detail::insert_absent_terms<8>(
        op,
        fresh.size(),
        [&](size_t k) -> const Monomial<8> & { return fresh[k]; },
        [&](size_t k, size_t b) { assign_row<8>(*op.store, b + k, fresh[k]); });

    BOOST_CHECK_EQUAL(base, 2U);
    BOOST_CHECK_EQUAL(op.size(), 5U);
    for (const auto &f : fresh) {
        BOOST_CHECK(op.store->find(f).has_value());
    }
    BOOST_CHECK(op.store->find(e0).has_value()); // existing rows intact
    BOOST_CHECK(op.store->find(e1).has_value());
}

BOOST_AUTO_TEST_CASE(mp_operator_append_term_after_materialization_rebuilds_inverted_index) {
    detail::MPOperator<8> op;
    op.append_term(indices_to_bitset<8>({0, 1}));
    BOOST_CHECK_EQUAL(op.inverted_index().rows(), 1U); // materializes the index (rows == size)

    // append_term no longer syncs the index incrementally; the next inverted_index() sees it stale
    // (rows() != store size) and rebuilds it against the grown store.
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

    (void)op.inverted_index(); // materialize it
    const auto after = detail::estimate_memory_usage<8>(op);
    BOOST_CHECK_GT(after.inverted_index_bytes, 0U); // present arm
}

BOOST_AUTO_TEST_CASE(mp_operator_copy_constructor_clones_store_and_coeffs) {
    auto op = build_indexed_op({indices_to_bitset<8>({0, 1}), indices_to_bitset<8>({2, 3})});
    op.slater_determinant = {0};
    (void)op.sparse_state();

    detail::MPOperator<8> copy(op); // deep copy via clone()
    BOOST_CHECK_EQUAL(copy.size(), op.size());
    BOOST_CHECK_EQUAL(copy.hf_scored_rows_, op.hf_scored_rows_);
    BOOST_CHECK(copy.hf_rows_ == op.hf_rows_);
    BOOST_CHECK(copy.hf_vals_ == op.hf_vals_);
    BOOST_CHECK(copy.materialize_state() == op.materialize_state());
    BOOST_CHECK(copy.store->find(indices_to_bitset<8>({0, 1})).has_value());
    // Mutating the copy must not touch the original (independent stores).
    copy.append_term(indices_to_bitset<8>({4, 5}));
    BOOST_CHECK_EQUAL(op.size(), 2U);
    BOOST_CHECK_EQUAL(copy.size(), 3U);
}
