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
#include "monoprop/detail/operator/TermTable.h"

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

/*! @brief Reserve for a coefficient array that parallels the row store, on the store's own policy.
 *
 *  Left to itself std::vector doubles, so a coefficient array appended alongside the rows settles at up
 *  to 8 B/term more capacity than the rows it parallels -- and never returns it mid-run, only at the
 *  quiescence shrink_to_fit. Growing at 1.5x keeps the two in step. Capacity is unobservable, so this
 *  is exactly a memory choice: the array stays one contiguous vector, append-only, with every fresh
 *  slot zero-filled by the resize that follows.
 */
inline auto reserve_coeffs_geometric(VecD &coeffs, size_t new_size) -> void {
    const size_t cap = coeffs.capacity();
    if (cap < new_size) {
        coeffs.reserve(std::max(new_size, cap + (cap / 2) + 1));
    }
}

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
    // Peak of (capacity - size) * 8 over the current propagate/build_graph call, sampled at the growth
    // sites. Kept here rather than derived in estimate_memory_usage() because the peak is transient:
    // the shrink at the end of each call erases it.
    size_t op_coeffs_slack_hwm_{0uz};
    MonomialMap<NumModes> init_op_map{};
    VecZ initial_state;
    // Set once at propagator construction.
    Basis basis{Basis::Majorana};
    mutable std::optional<InvertedIndex<NumModes>> inverted_index_{std::nullopt};
    // The join key -> row table the gate probe reads (TermTable.h). Lazy and kept in step exactly as
    // the inverted index is: materialised by the first term_table(), appended by reindex_after_growth,
    // and rebuilt by its staleness guard after any growth that bypassed that door.
    mutable std::optional<TermTable> term_table_{std::nullopt};

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
          op_coeffs_slack_hwm_(other.op_coeffs_slack_hwm_),
          init_op_map(other.init_op_map),
          initial_state(other.initial_state),
          basis(other.basis),
          inverted_index_(other.inverted_index_),
          term_table_(other.term_table_) {}

    auto size() const -> size_t { return store->size(); }

    /*! @brief Records op_coeffs' current growth slack into the per-call high-water mark.
     *
     *  Called right after every growth, because that is the only moment the slack exists to be seen:
     *  the array is shrunk to fit at the end of each propagate or build_graph call, so a caller reading
     *  the ledger between calls finds nothing left of it.
     */
    auto observe_op_coeffs_slack() -> void {
        op_coeffs_slack_hwm_ =
            std::max(op_coeffs_slack_hwm_, (op_coeffs.capacity() - op_coeffs.size()) * sizeof(double));
    }

    //! Opens a new measurement window for observe_op_coeffs_slack(): one propagate or build_graph call.
    auto reset_op_coeffs_slack_hwm() -> void { op_coeffs_slack_hwm_ = 0uz; }

    // Does not keep the lazy inverted index in sync: appends happen during setup, before the index is
    // first materialized, so a later append just makes inverted_index() rebuild via its staleness guard.
    auto append_term(const Monomial<NumModes> &mono) -> void { store->push_back(mono); }

    // Resync the inverted index and the term table after a bulk growth of `store`, preserving
    // has_value() ⟹ rows()==store.size() for both.
    auto reindex_after_growth(size_t base, size_t n) -> void {
        if (inverted_index_.has_value()) {
            inverted_index_->append_rows(*store, base, n);
        }
        if (term_table_.has_value()) {
            term_table_->append_rows(*store, base, n);
        }
    }

    auto inverted_index() const -> const InvertedIndex<NumModes> & {
        if (!inverted_index_.has_value() || inverted_index_->rows() != store->size()) {
            inverted_index_.emplace();
            inverted_index_->rebuild(*store);
        }
        return *inverted_index_;
    }

    //! The join key -> row table over every stored row, built or caught up on first use like the above.
    auto term_table() const -> const TermTable & {
        if (!term_table_.has_value() || term_table_->rows() != store->size()) {
            term_table_.emplace();
            term_table_->rebuild(*store);
        }
        return *term_table_;
    }

    // erase/clear keep bucket_count(), which init_operator_bytes reports, so drained buckets must be released.
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        reserve_coeffs_geometric(op_coeffs, size());
        op_coeffs.resize(size(), 0.0);
        observe_op_coeffs_slack();

        if (init_op_map.empty()) {
            return op_coeffs;
        }

        const auto before = init_op_map.size();
        const TermTable &table = term_table();
        erase_if(init_op_map, [this, &table](const auto &kv) {
            const size_t found = table.find(*store, kv.first);
            if (found != TermTable::kNotFound) {
                op_coeffs[found] = kv.second;
            }
            return found != TermTable::kNotFound;
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
        reserve_coeffs_geometric(state_coeffs, size());
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
        const TermTable &table = term_table();

        for (const auto &[k, v] : op_dict) {
            // Unchecked by design: the only caller bounds-checks against its logical_num_modes_.
            const auto mono = indices_to_bitset<NumModes>(k);
            const size_t rank_evolved_op = table.find(*store, mono);
            const bool in_evolved_op = rank_evolved_op != TermTable::kNotFound;
            const auto rank_init_op = init_op_map.find(mono);
            const auto coeff = algebra_encode_coeff<NumModes>(basis, v, mono);

            if (!schrodinger) {
                if (rank_init_op != init_op_map.end()) {
                    new_op_map[mono] = coeff;
                }
                else if (in_evolved_op) {
                    new_op_coeffs[rank_evolved_op] = coeff;
                }
                else {
                    const auto term_repr = std::format("[{}]", join_with_separator(k, ", "));
                    throw OperatorTermNotFound(std::format("Operator term {} not found in the operator.", term_repr));
                }
            }
            else {
                if (in_evolved_op) {
                    new_op_coeffs[rank_evolved_op] = coeff;
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
    // The join key -> row table (TermTable): 4-byte slots at a load of 0.35-0.7, so 5.7-11.4 B/term
    // depending on where the row count sits between two doublings. 0 while the table is unmaterialised.
    size_t indexing_bytes{0uz};
    size_t init_operator_bytes{0uz};
    size_t initial_state_bytes{0uz};
    size_t inverted_index_bytes{0uz};
    // The layer build's per-gate scratch (GateScratch), whose capacity survives from gate to gate.
    // Propagator-owned, so 0 unless MonomialPropagator fills it in.
    size_t gate_scratch_bytes{0uz};

    // Diagnostics: breakdowns of the fields above, deliberately excluded from total_bytes() so they can
    // never double-count.
    size_t inverted_index_dense_bytes{0uz};  // of inverted_index_bytes: full-height bitmap columns
    size_t inverted_index_sparse_bytes{0uz}; // of inverted_index_bytes: ascending set-row lists
    size_t inverted_index_dense_columns{0uz};
    size_t operator_terms_slack_bytes{0uz}; // of operator_terms_bytes: unused capacity (a chunk's tail)
    // The row store's two tiers: how many rows are too wide for the inline slot, the inline width they
    // are measured against, and how often the store has had to re-lay itself at a wider one. Together
    // they say whether the width guessed from the model's cutoff held.
    size_t row_wide_rows{0uz};
    size_t row_inline_width{0uz};
    size_t row_restrides{0uz};
    // What the chunk pools have mapped, and how much of that is chunks they have not handed out. Not a
    // subset of any field above: a pool maps whole arenas and keeps one while a single chunk is live,
    // so these bytes are what the kernel charges even where no named field prices them.
    size_t pool_mapped_bytes{0uz};
    size_t pool_free_chunk_bytes{0uz};
    /*! @brief The transport's persistent staging: the in-process transports' grow-only payload buffers
     *  and their (R, S)-fixed tables. Zero for plain MPI, whose payload buffers are the gate's own.
     *
     *  A per-RANK figure, so exactly one partition reports it and the sum over partitions counts a
     *  rank's transport once (MonomialPropagator::operator_memory_usage). Not in total_bytes(): it is
     *  the comm's memory, not the operator's, and total_bytes() is an operator figure.
     */
    size_t wire_staging_bytes{0uz};
    /*! @brief The widest instant of the gate exchange's per-gate buffers over the last call.
     *
     *  Those buffers die with their gate or are resized under the next one, so no resting figure can name
     *  their peak. Not in total_bytes(): it is a high-water mark over a call, not a resident size, and it
     *  overlaps the part of gate_scratch_bytes whose capacity the scratch keeps for reuse.
     */
    size_t gate_buffers_hwm_bytes{0uz};
    // of op_coeffs_bytes: the most capacity the coefficient array held beyond its live rows at any
    // point in the last propagate or build_graph call. A high-water mark, not a resting figure: the
    // array is shrunk to fit at the end of every call, so measured at quiescence the slack is always 0.
    size_t op_coeffs_slack_bytes{0uz};
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
        row_wide_rows += o.row_wide_rows;
        row_restrides += o.row_restrides;
        // Every partition sizes its rows from the same cutoff, so the width is shared, not summed.
        row_inline_width = std::max(row_inline_width, o.row_inline_width);
        pool_mapped_bytes += o.pool_mapped_bytes;
        pool_free_chunk_bytes += o.pool_free_chunk_bytes;
        // Summed although it is per-rank, because only one partition reports a nonzero: see the field.
        wire_staging_bytes += o.wire_staging_bytes;
        // Summed, not maxed: the partitions run their gates concurrently, so the per-process transient
        // peak is the sum of theirs. An upper bound, and it errs the safe way.
        gate_buffers_hwm_bytes += o.gate_buffers_hwm_bytes;
        // Summed, not maxed, across partitions: the partitions grow together within a call, so the sum
        // is the figure a per-process footprint wants. An upper bound, and it errs the safe way.
        op_coeffs_slack_bytes += o.op_coeffs_slack_bytes;
        state_coeffs_nonzero += o.state_coeffs_nonzero;
        init_operator_entries += o.init_operator_entries;
        return *this;
    }
};

template <size_t NumModes>
inline auto estimate_memory_usage(const MPOperator<NumModes> &op) -> MPOperatorMemoryBreakdown<NumModes> {
    MPOperatorMemoryBreakdown<NumModes> breakdown;
    breakdown.operator_terms_bytes = op.store->memory_bytes();
    breakdown.row_wide_rows = op.store->wide_size();
    breakdown.row_inline_width = op.store->inline_width();
    breakdown.row_restrides = op.store->restrides();
    breakdown.pool_mapped_bytes = op.store->pool_mapped_bytes();
    breakdown.pool_free_chunk_bytes = op.store->pool_free_chunk_bytes();
    breakdown.op_coeffs_bytes = op.op_coeffs.capacity() * sizeof(double);
    // Every representation of the state at once: the sparse scored set plus the dense vector.
    breakdown.state_coeffs_bytes = op.state_coeffs.capacity() * sizeof(double)
                                   + op.state_rows_.capacity() * sizeof(TermIndex)
                                   + op.state_vals_.capacity() * sizeof(double);
    breakdown.indexing_bytes = op.term_table_.has_value() ? op.term_table_->memory_bytes() : 0uz;
    breakdown.init_operator_bytes = unordered_flat_map_storage_bytes(op.init_op_map);
    breakdown.init_operator_entries = op.init_op_map.size();
    breakdown.initial_state_bytes = op.initial_state.capacity() * sizeof(size_t);
    if (op.inverted_index_.has_value()) {
        breakdown.inverted_index_bytes = op.inverted_index_->memory_bytes();
        const auto tiers = op.inverted_index_->tier_memory_bytes();
        breakdown.inverted_index_dense_bytes = tiers[0];
        breakdown.inverted_index_sparse_bytes = tiers[1];
        breakdown.inverted_index_dense_columns = tiers[2];
        breakdown.pool_mapped_bytes += op.inverted_index_->pool_mapped_bytes();
        breakdown.pool_free_chunk_bytes += op.inverted_index_->pool_free_chunk_bytes();
    }
    breakdown.operator_terms_slack_bytes = op.store->slack_bytes();
    breakdown.op_coeffs_slack_bytes = op.op_coeffs_slack_hwm_;
    // State phases are unit-magnitude, so at rest the scored count IS the nonzero count; a live vector needs a scan.
    breakdown.state_coeffs_nonzero =
        op.state_coeffs.empty()
            ? op.state_rows_.size()
            : static_cast<size_t>(std::ranges::count_if(op.state_coeffs, [](double c) { return c != 0.0; }));
    return breakdown;
}

} // namespace monoprop::detail
