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
#include <print>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/TermLookup.h"

// Forward-declared to break an include cycle with algebra/Algebra.h.
namespace monoprop {
template <size_t NumModes, typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op) -> VecZ;

template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> Monomial<NumModes>;

// Each binds the runtime Basis to its algebra model internally, so no basis branch is needed here.
template <size_t NumModes, typename Rows, typename Sink>
auto algebra_score_state(Basis basis,
                         const VecZ &paired_inds,
                         const VecZ &initial_state,
                         const Rows &store,
                         Sink &&sink) -> void;

template <size_t NumModes>
auto algebra_encode_coeff(Basis basis, const std::complex<double> &coeff, const Monomial<NumModes> &mono) -> double;
} // namespace monoprop

namespace monoprop::detail {

class OperatorTermNotFound : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <size_t NumModes>
struct MPOperator {
    // The store is non-copyable/non-movable, so it is heap-owned by unique_ptr (keeping MPOperator
    // itself cheaply movable). Always non-null.
    std::unique_ptr<OperatorIndex<NumModes>> store{std::make_unique<OperatorIndex<NumModes>>()};
    VecD op_coeffs;
    // Only fully-paired terms score nonzero (see score_new_state_rows_), which on production models is
    // ~0.07% of the rows -- a dense vector here is 99.9% zeros. state_rows_ is strictly ascending: rows are
    // scored in ascending order and the set is only ever appended to.
    std::vector<TermIndex> state_rows_;
    VecD state_vals_;               // parallel to state_rows_; every entry is a unit phase (+-1), never 0
    size_t state_scored_rows_{0uz}; // rows [0, state_scored_rows_) have been scored into state_rows_/state_vals_
    // The dense state: empty in Heisenberg unless a caller asks dense_state() to cache one; in Schrödinger
    // it is the live coefficient vector evolution mutates in place.
    VecD state_coeffs;
    MonomialMap<NumModes> init_op_map{};
    VecZ initial_state;
    // Set once at propagator construction.
    Basis basis{Basis::Majorana};
    mutable std::optional<InvertedIndex<NumModes>> inverted_index_{std::nullopt};

    MPOperator() noexcept = default;
    MPOperator(MPOperator &&) noexcept = default;
    MPOperator &operator=(MPOperator &&) noexcept = default;

    MPOperator(const MPOperator &other)
        : store(other.store->clone()),
          op_coeffs(other.op_coeffs),
          state_rows_(other.state_rows_),
          state_vals_(other.state_vals_),
          state_scored_rows_(other.state_scored_rows_),
          state_coeffs(other.state_coeffs),
          init_op_map(other.init_op_map),
          initial_state(other.initial_state),
          basis(other.basis),
          inverted_index_(other.inverted_index_) {}

    auto size() const -> size_t { return store->size(); }

    // Does not keep the lazy inverted index in sync: appends happen during setup, before the index is
    // first materialized, so a later append just makes inverted_index() rebuild via its staleness guard.
    auto append_term(const Monomial<NumModes> &mono) -> void { store->push_back(mono); }

    // Resync the inverted index after a bulk growth of `store`, preserving has_value() ⟹ rows()==store.size().
    auto reindex_after_growth(size_t base, size_t n) -> void {
        if (inverted_index_.has_value()) {
            inverted_index_->append_rows(*store, base, n);
        }
    }

    auto inverted_index() const -> const InvertedIndex<NumModes> & {
        if (!inverted_index_.has_value() || inverted_index_->rows() != store->size()) {
            inverted_index_.emplace();
            inverted_index_->rebuild(*store);
        }
        return *inverted_index_;
    }

    // erase/clear keep bucket_count(), which init_operator_bytes reports, so drained buckets must be released.
    // A pending term was absent from every row the last drain saw, so only the rows grown since then can
    // hold it: the lookup is built over those alone.
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        const size_t first_new = op_coeffs.size();
        op_coeffs.resize(size(), 0.0);

        if (init_op_map.empty()) {
            return op_coeffs;
        }

        const auto lookup = build_term_lookup<NumModes>(*store, first_new, size());
        const auto before = init_op_map.size();
        erase_if(init_op_map, [this, &lookup](const auto &kv) {
            const auto found = lookup.find(kv.first);
            if (found != lookup.end()) {
                op_coeffs[found->second] = kv.second;
            }
            return found != lookup.end();
        });
        if (init_op_map.size() != before) {
            init_op_map.rehash(0);
        }

        return op_coeffs;
    }

    struct SparseState {
        std::span<const TermIndex> rows; // strictly ascending row indices with a nonzero score
        std::span<const double> values;  // parallel to `rows`
    };

    auto sparse_state() -> SparseState {
        score_new_state_rows_();
        return SparseState{std::span<const TermIndex>(state_rows_), std::span<const double>(state_vals_)};
    }

    // Off every library path: evaluation carries the sparse form and densifies only inside the gradient's
    // reverse pass. Kept as the tests' dense oracle, and it cannot defer to EvalState::scatter_into --
    // MPFunctions.h, where EvalState lives, already includes this header.
    auto materialize_state() -> VecD {
        score_new_state_rows_();
        VecD dense(size(), 0.0);
        scatter_state_rows_from_(0, dense);
        return dense;
    }

    // Caches in state_coeffs -- the Schrödinger live vector, which evolution mutates in place, so later
    // calls only extend it and never rewrite an already-scored row, and a caller that snapshots it must
    // copy.
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

    auto shrink_state_to_fit() -> void {
        state_rows_.shrink_to_fit();
        state_vals_.shrink_to_fit();
        state_coeffs.shrink_to_fit();
    }

    // Each term lands on its existing evolved-operator row, or in the pending map if not yet materialized.
    // Heisenberg rejects a term absent from both (new monomials may have no graph paths); Schrödinger
    // admits them freely (the state was already evolved). Returns the supplied terms with their encoded
    // coefficients, in order.
    auto update_initial_operator(const OperatorDict &op_dict, bool schrodinger)
        -> std::pair<MonomialList<NumModes>, VecD> {
        MonomialMap<NumModes> new_op_map;
        std::pair<MonomialList<NumModes>, VecD> new_grad_op;
        VecD new_op_coeffs(size(), 0.0);
        const auto lookup = build_term_lookup<NumModes>(*store, 0, size());

        for (const auto &[k, v] : op_dict) {
            // Unchecked by design: the only caller bounds-checks against its logical_num_modes_.
            const auto mono = indices_to_bitset<NumModes>(k);
            const auto found = lookup.find(mono);
            const std::optional<size_t> rank_evolved_op =
                found == lookup.end() ? std::nullopt : std::optional<size_t>{static_cast<size_t>(found->second)};
            const auto rank_init_op = init_op_map.find(mono);
            const auto coeff = algebra_encode_coeff<NumModes>(basis, v, mono);

            if (!schrodinger) {
                if (rank_init_op != init_op_map.end()) {
                    new_op_map[mono] = coeff;
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
                    new_op_map[mono] = coeff;
                }
            }
            new_grad_op.first.push_back(mono);
            new_grad_op.second.push_back(coeff);
        }

        init_op_map = std::move(new_op_map);
        op_coeffs = std::move(new_op_coeffs);
        return new_grad_op;
    }

    auto score_new_state_rows_() -> void {
        if (state_scored_rows_ == size()) {
            return;
        }

        VecZ new_inds(size() - state_scored_rows_);
        std::iota(new_inds.begin(), new_inds.end(), state_scored_rows_); // NOLINT(modernize-use-ranges)
        const auto paired_inds = is_fully_paired<NumModes>(new_inds, *store);
        state_rows_.reserve(state_rows_.size() + paired_inds.size());
        state_vals_.reserve(state_vals_.size() + paired_inds.size());

        // The algebra picks the diagonal ⟨b|·|b⟩ phase of each fully-paired term.
        algebra_score_state<NumModes>(basis, paired_inds, initial_state, *store, [this](size_t row, double phase) {
            state_rows_.push_back(static_cast<TermIndex>(row));
            state_vals_.push_back(phase);
        });

        state_scored_rows_ = size();
    }

    // Write the scored entries with row >= first_row into `out` (sized >= size()); ascending state_rows_
    // makes the starting entry a binary search rather than a full scan.
    auto scatter_state_rows_from_(size_t first_row, VecD &out) const -> void {
        const auto first = std::ranges::lower_bound(state_rows_, static_cast<TermIndex>(first_row));
        for (auto it = first; it != state_rows_.end(); ++it) {
            out[*it] = state_vals_[static_cast<size_t>(std::distance(state_rows_.begin(), it))];
        }
    }
};

// Callers must pass pairwise-distinct, currently-absent terms: slot k deterministically lands at base+k
// and nothing checks for a duplicate. Call after any pass that reads pre-insert op state (op.size() must
// equal the returned base). per_slot(k, base) writes row base+k.
template <size_t NumModes, typename PerSlot>
inline auto insert_absent_terms(MPOperator<NumModes> &op, size_t n, PerSlot &&per_slot) -> size_t {
    const size_t base = op.store->grow_rows_geometric(n);
    for (size_t k = 0; k < n; ++k) {
        per_slot(k, base);
    }
    op.reindex_after_growth(base, n);
    return base;
}

template <typename FlatMap>
inline auto unordered_flat_map_storage_bytes(const FlatMap &map) -> size_t {
    return sizeof(FlatMap) + map.bucket_count() * (sizeof(typename FlatMap::value_type) + sizeof(unsigned char));
}

template <size_t NumModes>
struct MPOperatorMemoryBreakdown final {
    size_t operator_terms_bytes{0uz};
    size_t op_coeffs_bytes{0uz};
    size_t state_coeffs_bytes{0uz};
    size_t indexing_bytes{0uz};
    size_t init_operator_bytes{0uz};
    size_t initial_state_bytes{0uz};
    size_t inverted_index_bytes{0uz};
    // The per-gate layer-build scratch (the join, the row marks and the buffers). Propagator-owned, so 0 unless
    // MonomialPropagator fills it in.
    size_t gate_scratch_bytes{0uz};

    // Diagnostics: breakdowns of the fields above, deliberately excluded from total_bytes() so they can
    // never double-count.
    size_t inverted_index_dense_bytes{0uz};  // of inverted_index_bytes: full-height bitmap columns
    size_t inverted_index_sparse_bytes{0uz}; // of inverted_index_bytes: ascending set-row lists
    size_t inverted_index_dense_columns{0uz};
    size_t operator_terms_slack_bytes{0uz}; // of operator_terms_bytes: unused geometric-growth capacity
    // of state_coeffs_bytes: entries of the state that are not exactly 0.0
    size_t state_coeffs_nonzero{0uz};
    // Live entries behind init_operator_bytes, which is bucket_count(): bytes with no entries are dead buckets.
    size_t init_operator_entries{0uz};

    auto total_bytes() const -> size_t {
        return operator_terms_bytes + op_coeffs_bytes + state_coeffs_bytes + indexing_bytes + init_operator_bytes
               + initial_state_bytes + inverted_index_bytes + gate_scratch_bytes;
    }

    auto operator+=(const MPOperatorMemoryBreakdown &o) -> MPOperatorMemoryBreakdown & {
        operator_terms_bytes += o.operator_terms_bytes;
        op_coeffs_bytes += o.op_coeffs_bytes;
        state_coeffs_bytes += o.state_coeffs_bytes;
        indexing_bytes += o.indexing_bytes;
        init_operator_bytes += o.init_operator_bytes;
        initial_state_bytes += o.initial_state_bytes;
        inverted_index_bytes += o.inverted_index_bytes;
        gate_scratch_bytes += o.gate_scratch_bytes;
        inverted_index_dense_bytes += o.inverted_index_dense_bytes;
        inverted_index_sparse_bytes += o.inverted_index_sparse_bytes;
        inverted_index_dense_columns += o.inverted_index_dense_columns;
        operator_terms_slack_bytes += o.operator_terms_slack_bytes;
        state_coeffs_nonzero += o.state_coeffs_nonzero;
        init_operator_entries += o.init_operator_entries;
        return *this;
    }
};

template <size_t NumModes>
inline auto estimate_memory_usage(const MPOperator<NumModes> &op) -> MPOperatorMemoryBreakdown<NumModes> {
    MPOperatorMemoryBreakdown<NumModes> breakdown;
    breakdown.operator_terms_bytes = op.store->memory_bytes();
    breakdown.op_coeffs_bytes = op.op_coeffs.capacity() * sizeof(double);
    // Every representation of the state at once: the sparse scored set plus the dense vector.
    breakdown.state_coeffs_bytes = op.state_coeffs.capacity() * sizeof(double)
                                   + op.state_rows_.capacity() * sizeof(TermIndex)
                                   + op.state_vals_.capacity() * sizeof(double);
    // No persistent key index exists any more (see OperatorIndex.h); the field stays so the breakdown's
    // shape is stable, and reports the store object itself.
    breakdown.indexing_bytes = sizeof(*op.store);
    breakdown.init_operator_bytes = unordered_flat_map_storage_bytes(op.init_op_map);
    breakdown.init_operator_entries = op.init_op_map.size();
    breakdown.initial_state_bytes = op.initial_state.capacity() * sizeof(size_t);
    if (op.inverted_index_.has_value()) {
        breakdown.inverted_index_bytes = op.inverted_index_->memory_bytes();
        const auto tiers = op.inverted_index_->tier_memory_bytes();
        breakdown.inverted_index_dense_bytes = tiers[0];
        breakdown.inverted_index_sparse_bytes = tiers[1];
        breakdown.inverted_index_dense_columns = tiers[2];
    }
    breakdown.operator_terms_slack_bytes = op.store->slack_bytes();
    // State phases are unit-magnitude, so at rest the scored count IS the nonzero count; a live vector needs a scan.
    breakdown.state_coeffs_nonzero =
        op.state_coeffs.empty()
            ? op.state_rows_.size()
            : static_cast<size_t>(std::ranges::count_if(op.state_coeffs, [](double c) { return c != 0.0; }));
    return breakdown;
}

} // namespace monoprop::detail
