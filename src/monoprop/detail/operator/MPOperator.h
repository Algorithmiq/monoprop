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
#include <numeric>
#include <optional>
#include <ranges>
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
template <size_t NumModes, typename Rows, typename Sink>
auto algebra_score_hf(Basis basis, const VecZ &paired_inds, const VecZ &hf, const Rows &store, Sink &&sink) -> void;

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
    // ── The reference (Hartree-Fock) state, in its resting SPARSE form ───────────────────────────
    // Only fully-paired terms score nonzero (see score_new_state_rows_), which on production models is
    // ~0.07% of the rows -- a dense vector here is 99.9% zeros. hf_rows_ is strictly ASCENDING: rows are
    // scored in ascending order and the set is only ever appended to.
    std::vector<TermIndex> hf_rows_ = {};
    VecD hf_vals_ = {};         ///< parallel to hf_rows_; every entry is a unit phase (+-1), never 0
    size_t hf_scored_rows_ = 0; ///< rows [0, hf_scored_rows_) have been scored into hf_rows_/hf_vals_
    // The DENSE state vector. Heisenberg: stays empty (the sparse form above is the whole truth) unless
    // a caller explicitly asks dense_state() to cache one. SCHRODINGER: this is the live coefficient
    // vector that evolution mutates in place, seeded by dense_state()'s first materialization.
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
          hf_rows_(other.hf_rows_),
          hf_vals_(other.hf_vals_),
          hf_scored_rows_(other.hf_scored_rows_),
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

    /// A read-only view of the sparse reference state: ascending rows and their matching scores.
    struct SparseState {
        std::span<const TermIndex> rows; ///< strictly ascending row indices with a nonzero score
        std::span<const double> values;  ///< parallel to `rows`
    };

    /**
     * @brief The reference (initial) state in its sparse form, scored up to date.
     *
     * The cheap accessor: no per-row storage is touched for the ~99.9% of rows that score zero.
     */
    auto sparse_state() -> SparseState {
        score_new_state_rows_();
        return SparseState{std::span<const TermIndex>(hf_rows_), std::span<const double>(hf_vals_)};
    }

    /**
     * @brief Scatter the sparse state into a FRESH dense vector of length size(); nothing is cached.
     *
     * For consumers that genuinely need a dense `VecD` (the evaluation functional). Prefer this over
     * dense_state() in the Heisenberg picture: it leaves no dense copy behind on the operator.
     */
    auto materialize_state() -> VecD {
        score_new_state_rows_();
        VecD dense(size(), 0.0);
        scatter_state_rows_from_(0, dense);
        return dense;
    }

    /**
     * @brief The dense state vector, materialized once and then CACHED in `state_coeffs`.
     *
     * This is the Schrödinger picture's live coefficient vector: the first call seeds it from the HF
     * scores, and evolution then overwrites it in place. Subsequent calls only EXTEND it -- rows scored
     * before are left exactly as the caller (or evolution) left them, and just the newly-appended rows
     * receive their HF score. Heisenberg callers that only need a value should use materialize_state().
     */
    auto dense_state() -> const VecD & {
        score_new_state_rows_();
        if (state_coeffs.size() == size()) {
            return state_coeffs;
        }
        const size_t cur_len = state_coeffs.size();
        state_coeffs.resize(size(), 0.0);
        scatter_state_rows_from_(cur_len, state_coeffs);
        return state_coeffs;
    }

    /// Trim the slack of every state representation once the term count has stabilized.
    auto shrink_state_to_fit() -> void {
        hf_rows_.shrink_to_fit();
        hf_vals_.shrink_to_fit();
        state_coeffs.shrink_to_fit();
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

    /**
     * @brief Score the newly-appended terms [hf_scored_rows_, size()) into the sparse HF set.
     *
     * A new term is nonzero only if fully paired with the Slater determinant, in which case it receives
     * that term's Hartree-Fock phase. Already-scored rows are never revisited, and because the new rows
     * are scored in ascending order and only appended, hf_rows_ stays globally ascending.
     */
    auto score_new_state_rows_() -> void {
        if (hf_scored_rows_ == size()) {
            return;
        }

        VecZ new_inds(size() - hf_scored_rows_);
        std::iota(new_inds.begin(), new_inds.end(), hf_scored_rows_);

        const auto paired_inds = is_fully_paired<NumModes>(new_inds, *store);
        hf_rows_.reserve(hf_rows_.size() + paired_inds.size());
        hf_vals_.reserve(hf_vals_.size() + paired_inds.size());

        // Score the diagonal ⟨b|·|b⟩ coefficient of each fully-paired term; the algebra picks the phase
        // (algebra_score_hf binds the basis to its model once, then loops).
        algebra_score_hf<NumModes>(basis, paired_inds, slater_determinant, *store, [this](size_t row, double phase) {
            hf_rows_.push_back(static_cast<TermIndex>(row));
            hf_vals_.push_back(phase);
        });

        hf_scored_rows_ = size();
    }

    /// Write the scored entries with row >= @p first_row into @p out (sized >= size()); ascending
    /// hf_rows_ makes the starting entry a binary search rather than a full scan.
    auto scatter_state_rows_from_(size_t first_row, VecD &out) const -> void {
        const auto first = std::ranges::lower_bound(hf_rows_, static_cast<TermIndex>(first_row));
        for (auto it = first; it != hf_rows_.end(); ++it) {
            out[*it] = hf_vals_[static_cast<size_t>(std::distance(hf_rows_.begin(), it))];
        }
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
    size_t operator_terms_slack_bytes = 0;  ///< of operator_terms_bytes: unused geometric-growth capacity
    /// of state_coeffs_bytes: entries of the state that are not exactly 0.0 -- the sparse HF entry count
    /// at rest, or the dense vector's true nonzero count once a live (Schrödinger) vector exists.
    size_t state_coeffs_nonzero = 0;

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
    // Every representation of the state at once: the sparse HF set (the resting form) plus the dense
    // vector, which is empty unless a live Schrödinger vector or an explicit dense_state() cache exists.
    breakdown.state_coeffs_bytes = op.state_coeffs.capacity() * sizeof(double)
                                   + op.hf_rows_.capacity() * sizeof(TermIndex)
                                   + op.hf_vals_.capacity() * sizeof(double);
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
    // HF phases are unit-magnitude, so at rest the scored-entry count IS the nonzero count; once a dense
    // vector exists it has been evolved and only a scan can answer.
    breakdown.state_coeffs_nonzero =
        op.state_coeffs.empty()
            ? op.hf_rows_.size()
            : static_cast<size_t>(std::ranges::count_if(op.state_coeffs, [](double c) { return c != 0.0; }));
    return breakdown;
}

} // namespace monoprop::detail
