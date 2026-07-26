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

#include <algorithm>
#include <ranges>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <format>
#include "monoprop/detail/print_compat.h"

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/OperatorIndex.h"

// Forward-declared (not #included) to break an include cycle with algebra/Algebra.h; the definitions
// are visible wherever the MPOperator methods are instantiated (via MonomialPropagatorImpl.h). `Rows`
// is either MonomialList or the packed operator-row container, read through the backend-agnostic accessors.
namespace monoprop {
template <size_t NumModes, typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op) -> VecZ;

template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> Monomial<NumModes>;

// The two algebra-generic entry points this header calls (HF scoring + the real<->double coeff codec).
// Each binds the runtime Basis to its algebra model internally, so the Majorana/Pauli choice lives in
// ONE place (the policy layer) rather than scattered `if (basis == Basis::Pauli)` branches here.
template <size_t NumModes, typename Rows>
auto algebra_score_hf(Basis basis, const VecZ &paired_inds, const VecZ &hf, const Rows &store, VecD &out) -> void;

template <size_t NumModes>
auto algebra_encode_coeff(Basis basis, const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> double;
} // namespace monoprop

namespace monoprop::detail {

/// Thrown by set-coefficient / update paths when a requested operator term is absent from the store.
class OperatorTermNotFound : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// The propagated operator: the term store (entropy-packed rows + keyless hash index), its
/// coefficient vectors, the initial-operator map, and the lazily-built even-parity scan inverted index.
template <size_t NumModes>
struct MPOperator {
    // The store is non-copyable/non-movable, so it is heap-owned by unique_ptr (keeping MPOperator
    // itself cheaply movable). Always non-null. Rows go through the backend-agnostic accessors.
    std::unique_ptr<OperatorIndex<NumModes>> store = std::make_unique<OperatorIndex<NumModes>>();
    VecD op_coeffs = {};
    VecD state_coeffs = {};
    MonomialMap<NumModes> init_op_map = {};
    VecZ slater_determinant = {};
    // Operator basis: Majorana monomials (default) or native Pauli strings. Bound to its algebra model
    // at each use (drives the coeff codec and HF scoring); set once at propagator construction.
    Basis basis = Basis::Majorana;
    mutable std::optional<InvertedIndex<NumModes>> inverted_index_ = std::nullopt;

    MPOperator() noexcept = default;
    MPOperator(MPOperator &&) noexcept = default;
    MPOperator &operator=(MPOperator &&) noexcept = default;

    // Deep copy via the copy constructor only: the non-copyable `store` is rebuilt via clone(),
    // everything else is plain value data. Copy assignment stays implicitly deleted (unique_ptr member).
    MPOperator(const MPOperator &other)
        : store(other.store->clone()),
          op_coeffs(other.op_coeffs),
          state_coeffs(other.state_coeffs),
          init_op_map(other.init_op_map),
          slater_determinant(other.slater_determinant),
          basis(other.basis),
          inverted_index_(other.inverted_index_) {}

    auto size() const -> size_t { return store->size(); }

    // Append one term to the store. The lazy inverted index is NOT kept in sync here: appends happen
    // during setup, before the index is first materialized, so a later append simply makes
    // inverted_index() rebuild via its rows() != store->size() guard.
    auto append_term(const Monomial<NumModes> &maj) -> void { store->push_back(maj); }

    // Resync the even-parity inverted index after a bulk growth of `store`, preserving the
    // has_value() ⟹ rows()==store.size() invariant.
    auto reindex_after_growth(size_t base, size_t n) -> void {
        if (inverted_index_.has_value()) {
            inverted_index_->append_rows(*store, base, n);
        }
    }

    // The lazily-built even-parity inverted index, rebuilt whenever it is stale (rows() != store size).
    // That rebuild is also the only mechanism that resyncs an append_term made after materialization.
    auto inverted_index() const -> const InvertedIndex<NumModes> & {
        if (!inverted_index_.has_value() || inverted_index_->rows() != store->size()) {
            inverted_index_.emplace();
            inverted_index_->rebuild(*store);
        }
        return *inverted_index_;
    }

    /**
     * @brief Lazily materialize the operator coefficients aligned with the store's row indexing.
     *
     * Drains pending init_op_map terms into op_coeffs, erasing them AFTER the lookup loop (the flat_map
     * is not iterable while mutating). A no-op once op_coeffs is already in sync with the store.
     */
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        op_coeffs.resize(size(), 0.0);

        if (init_op_map.empty()) {
            return op_coeffs;
        }

        std::vector<Monomial<NumModes>> del;
        for (const auto &kv : init_op_map) {
            const auto &maj = kv.first;
            const auto coeff = kv.second;
            if (const auto found = store->find(maj)) {
                op_coeffs[*found] = coeff;
                del.push_back(maj);
            }
        }

        for (const auto &maj : del) {
            init_op_map.erase(maj);
        }

        return op_coeffs;
    }

    /**
     * @brief Lazily materialize the state (initial reference) coefficients aligned with the store.
     *
     * Scores ONLY the newly-appended terms [old_size, size()); a new term is nonzero only if fully
     * paired with the Slater determinant, in which case it receives that term's Hartree-Fock phase.
     */
    auto get_state() -> const VecD & {
        if (state_coeffs.size() == size()) {
            return state_coeffs;
        }

        size_t cur_len = state_coeffs.size();
        size_t new_elements = size() - cur_len;
        state_coeffs.resize(size(), 0.0);

        VecZ new_inds(new_elements);
        std::iota(new_inds.begin(), new_inds.end(), cur_len);

        const auto paired_inds = is_fully_paired<NumModes>(new_inds, *store);

        // Score the diagonal ⟨b|·|b⟩ coefficient of each fully-paired term; the algebra picks the phase
        // (algebra_score_hf binds the basis to its model once, then loops).
        algebra_score_hf<NumModes>(basis, paired_inds, slater_determinant, *store, state_coeffs);

        return state_coeffs;
    }

    /**
     * @brief Rewrite the initial Hamiltonian from a new coefficient dictionary.
     *
     * Each term lands on its existing evolved-operator row, or in the pending map if not yet
     * materialized. Heisenberg REJECTS a term absent from both (new Majoranas may have no graph paths);
     * Schrödinger admits them freely (the state was already evolved). Overwrites init_op_map/op_coeffs
     * and returns the gradient operator (the supplied terms and their encoded coefficients, in order).
     */
    auto update_initial_operator(const FermiOperatorMap &op_dict, bool schrodinger)
        -> std::pair<MonomialList<NumModes>, VecD> {
        MonomialMap<NumModes> new_op_map;
        std::pair<MonomialList<NumModes>, VecD> new_grad_op;
        VecD new_op_coeffs(size(), 0.0);

        for (const auto &[k, v] : op_dict) {
            const auto maj = indices_to_bitset<NumModes>(k);
            const auto rank_evolved_op = store->find(maj);
            const auto rank_init_op = init_op_map.find(maj);
            const auto coeff = algebra_encode_coeff<NumModes>(basis, v, maj);

            if (!schrodinger) {
                if (rank_init_op != init_op_map.end()) {
                    new_op_map[maj] = coeff;
                }
                else if (rank_evolved_op) {
                    new_op_coeffs[*rank_evolved_op] = coeff;
                }
                else {
                    const auto term_repr = std::format("[{}]", join_with_separator(k, ", "));
                    throw OperatorTermNotFound(std::format("Operator term {} not found in the operator.", term_repr));
                }
            }
            else {
                if (rank_evolved_op) {
                    new_op_coeffs[*rank_evolved_op] = coeff;
                }
                else {
                    new_op_map[maj] = coeff;
                }
            }
            new_grad_op.first.push_back(maj);
            new_grad_op.second.push_back(coeff);
        }

        init_op_map = std::move(new_op_map);
        op_coeffs = std::move(new_op_coeffs);
        return new_grad_op;
    }
};

// Insert `n` provably-distinct, currently-absent terms into `op` in one batch — the grow → scatter →
// index → resync quartet shared by every miss-insert site. Callers pass pairwise-distinct keys, so
// bulk_insert can skip duplicate probes and slot k deterministically lands at base+k. Call AFTER any
// pass that reads pre-insert op state (op.size() must equal the returned base).
template <size_t NumModes, typename KeyAt, typename PerSlot>
inline auto insert_absent_terms(MPOperator<NumModes> &op, size_t n, KeyAt &&key_at, PerSlot &&per_slot) -> size_t {
    const size_t base = op.store->grow_rows_geometric(n);
    for (size_t k = 0; k < n; ++k) {
        per_slot(k, base);
    }
    op.store->bulk_insert(n, base, std::forward<KeyAt>(key_at));
    op.reindex_after_growth(base, n);
    return base;
}

template <typename FlatMap>
inline auto unordered_flat_map_storage_bytes(const FlatMap &map) -> size_t {
    return sizeof(FlatMap) + map.bucket_count() * (sizeof(typename FlatMap::value_type) + sizeof(unsigned char));
}

template <size_t NumModes>
struct MPOperatorMemoryBreakdown final {
    size_t operator_terms_bytes = 0;
    size_t op_coeffs_bytes = 0;
    size_t state_coeffs_bytes = 0;
    size_t indexing_bytes = 0;
    size_t init_operator_bytes = 0;
    size_t slater_determinant_bytes = 0;
    size_t inverted_index_bytes = 0;

    // Diagnostics: breakdowns OF the fields above, deliberately excluded from total_bytes() so they
    // can never double-count. They exist to size compression choices, which turn on *which part* of a
    // field holds the bytes -- a field total cannot answer that.
    size_t inverted_index_dense_bytes = 0;  ///< of inverted_index_bytes: full-height bitmap columns
    size_t inverted_index_sparse_bytes = 0; ///< of inverted_index_bytes: ascending set-row lists
    size_t inverted_index_dense_columns = 0;
    size_t inverted_index_delta_bytes = 0;  ///< inverted index if every column were delta+varint coded
    size_t inverted_index_oracle_bytes = 0; ///< ... picking min(bitmap, delta) per column
    size_t inverted_index_delta_wins = 0;   ///< columns where delta beats the current representation
    size_t operator_terms_slack_bytes = 0; ///< of operator_terms_bytes: unused geometric-growth capacity
    size_t state_coeffs_nonzero = 0;       ///< entries of state_coeffs that are not exactly 0.0

    auto total_bytes() const -> size_t {
        return operator_terms_bytes + op_coeffs_bytes + state_coeffs_bytes + indexing_bytes + init_operator_bytes
               + slater_determinant_bytes + inverted_index_bytes;
    }

    // Field-wise sum, so a sharded propagator can aggregate its per-shard operator breakdowns.
    auto operator+=(const MPOperatorMemoryBreakdown &o) -> MPOperatorMemoryBreakdown & {
        operator_terms_bytes += o.operator_terms_bytes;
        op_coeffs_bytes += o.op_coeffs_bytes;
        state_coeffs_bytes += o.state_coeffs_bytes;
        indexing_bytes += o.indexing_bytes;
        init_operator_bytes += o.init_operator_bytes;
        slater_determinant_bytes += o.slater_determinant_bytes;
        inverted_index_bytes += o.inverted_index_bytes;
        inverted_index_dense_bytes += o.inverted_index_dense_bytes;
        inverted_index_sparse_bytes += o.inverted_index_sparse_bytes;
        inverted_index_dense_columns += o.inverted_index_dense_columns;
        inverted_index_delta_bytes += o.inverted_index_delta_bytes;
        inverted_index_oracle_bytes += o.inverted_index_oracle_bytes;
        inverted_index_delta_wins += o.inverted_index_delta_wins;
        operator_terms_slack_bytes += o.operator_terms_slack_bytes;
        state_coeffs_nonzero += o.state_coeffs_nonzero;
        return *this;
    }
};

template <size_t NumModes>
inline auto estimate_memory_usage(const MPOperator<NumModes> &op) -> MPOperatorMemoryBreakdown<NumModes> {
    MPOperatorMemoryBreakdown<NumModes> breakdown;
    // Packed rows store stride bytes/row (+ overflow side-map), not sizeof(Monomial); ask directly.
    breakdown.operator_terms_bytes = op.store->memory_bytes();
    breakdown.op_coeffs_bytes = op.op_coeffs.capacity() * sizeof(double);
    breakdown.state_coeffs_bytes = op.state_coeffs.capacity() * sizeof(double);
    breakdown.indexing_bytes = op.store->index_estimated_memory_bytes();
    breakdown.init_operator_bytes = unordered_flat_map_storage_bytes(op.init_op_map);
    breakdown.slater_determinant_bytes = op.slater_determinant.capacity() * sizeof(size_t);
    if (op.inverted_index_.has_value()) {
        breakdown.inverted_index_bytes = op.inverted_index_->memory_bytes();
        const auto tiers = op.inverted_index_->tier_memory_bytes();
        breakdown.inverted_index_dense_bytes = tiers[0];
        breakdown.inverted_index_sparse_bytes = tiers[1];
        breakdown.inverted_index_dense_columns = tiers[2];
        const auto delta = op.inverted_index_->delta_coded_bytes();
        breakdown.inverted_index_delta_bytes = delta[0];
        breakdown.inverted_index_oracle_bytes = delta[1];
        breakdown.inverted_index_delta_wins = delta[2];
    }
    breakdown.operator_terms_slack_bytes = op.store->slack_bytes();
    breakdown.state_coeffs_nonzero =
        static_cast<size_t>(std::ranges::count_if(op.state_coeffs, [](double c) { return c != 0.0; }));
    return breakdown;
}

} // namespace monoprop::detail
