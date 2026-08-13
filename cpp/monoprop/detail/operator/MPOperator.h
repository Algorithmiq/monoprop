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
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/OperatorIndex.h"

// Forward-declared to break an include cycle with algebra/Algebra.h.
namespace monoprop {
template <typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op, size_t num_bits) -> VecZ;

// inline here too, matching the definition in AlgebraCommon.h: a declaration that disagreed would
// leave TUs holding only this one expecting an out-of-line definition that nothing emits.
inline auto indices_to_bitset(const VecZ &arr, size_t num_bits) -> Bitset;

// Branches on the runtime Basis internally, so no basis branch is needed here.
template <typename Rows, typename Sink>
auto algebra_score_state(Basis basis,
                         const VecZ &paired_inds,
                         const VecZ &initial_state,
                         const Rows &store,
                         size_t num_bits,
                         Sink &&sink) -> void;

auto algebra_encode_coeff(Basis basis, const std::complex<double> &coeff, const MonomialLike auto &mono) -> double;
} // namespace monoprop

namespace monoprop::detail {

class OperatorTermNotFound : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MPOperator {
    // The row store is one of two backends, chosen per propagator from its storage mode count
    // (SparseRowStore::preferred_for_modes) and then fixed for the propagator's lifetime. Exactly one
    // of these is non-null.
    //
    // Two pointers rather than the compile-time alias this was, because the choice is data. And rather
    // than a virtual interface, because the scan asks the store for a row per anticommuting term: a
    // branch or an indirect call on that path is not affordable. with_store() binds the concrete type
    // once per layer instead -- the same shape as with_algebra() for a runtime Basis, and the reason
    // build_layer is a template. Everything off that path goes through the forwarding accessors below,
    // which pay one well-predicted branch.
    //
    // Heap-owned because neither store is copyable or movable (single-writer, and their views borrow
    // their arrays), which keeps MPOperator itself cheaply movable.
    std::unique_ptr<OperatorIndex> dense_rows = nullptr;
    std::unique_ptr<SparseRowStore> sparse_rows = nullptr;
    VecD op_coeffs = {};
    // Only fully-paired terms score nonzero (see score_new_state_rows_), which on production models is
    // ~0.07% of the rows -- a dense vector here is 99.9% zeros. state_rows_ is strictly ascending: rows are
    // scored in ascending order and the set is only ever appended to.
    std::vector<TermIndex> state_rows_ = {};
    VecD state_vals_ = {};         // parallel to state_rows_; every entry is a unit phase (+-1), never 0
    size_t state_scored_rows_ = 0; // rows [0, state_scored_rows_) have been scored into state_rows_/state_vals_
    // The dense state: empty in Heisenberg unless a caller asks dense_state() to cache one; in Schrödinger
    // it is the live coefficient vector evolution mutates in place.
    VecD state_coeffs = {};
    MonomialMap init_op_map = {};
    VecZ initial_state = {};
    // Set once at propagator construction.
    Basis basis = Basis::Majorana;
    mutable std::optional<InvertedIndex> inverted_index_ = std::nullopt;

    // num_bits is the storage bit width of every monomial this operator holds. No default constructor:
    // the width used to come free from NumModes, and a default-constructed store would be a width-0
    // one that silently mis-sizes every monomial built from it. The backend starts dense and the
    // propagator replaces it via set_store() once the cutoff -- and with it the row width -- is known.
    explicit MPOperator(size_t num_bits) : dense_rows(std::make_unique<OperatorIndex>(num_bits)) {}

    MPOperator(MPOperator &&) noexcept = default;
    MPOperator &operator=(MPOperator &&) noexcept = default;

    MPOperator(const MPOperator &other)
        : dense_rows(other.dense_rows ? other.dense_rows->clone() : nullptr),
          sparse_rows(other.sparse_rows ? other.sparse_rows->clone() : nullptr),
          op_coeffs(other.op_coeffs),
          state_rows_(other.state_rows_),
          state_vals_(other.state_vals_),
          state_scored_rows_(other.state_scored_rows_),
          state_coeffs(other.state_coeffs),
          init_op_map(other.init_op_map),
          initial_state(other.initial_state),
          basis(other.basis),
          inverted_index_(other.inverted_index_) {}

    // Binds the live store to a concrete type for the duration of the call. Both arms are instantiated,
    // so `f` must be a generic lambda and must return the same type from each.
    template <class F>
    [[gnu::always_inline]] auto with_store(F &&f) -> decltype(auto) {
        if (sparse_rows) {
            return f(*sparse_rows);
        }
        return f(*dense_rows);
    }
    template <class F>
    [[gnu::always_inline]] auto with_store(F &&f) const -> decltype(auto) {
        if (sparse_rows) {
            return f(*sparse_rows);
        }
        return f(*dense_rows);
    }

    // Installs a backend, dropping the lazy inverted index with it: the index addresses the old rows,
    // and leaving it would let a stale one answer for the new store until its row count happened to
    // disagree. One overload per backend rather than a tag, so a call site names the choice.
    auto set_store(std::unique_ptr<OperatorIndex> rows) -> void {
        dense_rows = std::move(rows);
        sparse_rows.reset();
        inverted_index_.reset();
    }
    auto set_store(std::unique_ptr<SparseRowStore> rows) -> void {
        sparse_rows = std::move(rows);
        dense_rows.reset();
        inverted_index_.reset();
    }
    [[nodiscard]] auto rows_are_sparse() const -> bool { return sparse_rows != nullptr; }

    auto size() const -> size_t {
        return with_store([](const auto &rows) { return rows.size(); });
    }

    // Off the store, not a member of its own, so the width driving row reconstruction and the width
    // driving the monomials handed to it cannot drift apart. The copy constructor needs no extra
    // work for the same reason: clone() carries the width across.
    [[nodiscard]] auto num_bits() const -> size_t {
        return with_store([](const auto &rows) { return rows.num_bits(); });
    }
    // The per-word loops are sized in words, not bits.
    [[nodiscard]] auto num_words() const -> size_t { return (num_bits() + 63) / 64; }

    // Does not keep the lazy inverted index in sync: appends happen during setup, before the index is
    // first materialized, so a later append just makes inverted_index() rebuild via its staleness guard.
    auto append_term(const Bitset &mono) -> void {
        with_store([&](auto &rows) { rows.push_back(mono); });
    }

    // Both are setup-path forwards, kept here rather than exposing a store, so nothing outside has to
    // know which backend is live.
    auto reserve_terms(size_t n) -> void {
        with_store([&](auto &rows) { rows.reserve(n); });
    }
    auto index_term(const Bitset &mono, size_t row) -> void {
        with_store([&](auto &rows) { rows.emplace(mono, row); });
    }
    [[nodiscard]] auto find(const Bitset &mono) const -> std::optional<size_t> {
        return with_store([&](const auto &rows) { return rows.find(mono); });
    }
    // This rank's terms as fn(monomial, row), in the index's slot order. Materializes each row.
    template <typename Fn>
    auto for_each_term(Fn &&fn) const -> void {
        with_store([&](const auto &rows) { rows.for_each(fn); });
    }

    // Resync the inverted index after a bulk growth of the store, preserving has_value() ⟹ rows()==size().
    auto reindex_after_growth(size_t base, size_t n) -> void {
        if (inverted_index_.has_value()) {
            with_store([&](const auto &rows) { inverted_index_->append_rows(rows, base, n); });
        }
    }

    auto inverted_index() const -> const InvertedIndex & {
        if (!inverted_index_.has_value() || inverted_index_->rows() != size()) {
            // Column count is the storage bit width, taken off the store so it cannot drift from the
            // monomials whose positions rebuild() scatters.
            inverted_index_.emplace(num_bits());
            with_store([&](const auto &rows) { inverted_index_->rebuild(rows); });
        }
        return *inverted_index_;
    }

    // Pending init_op_map terms are erased after the lookup loop: the flat_map is not iterable while
    // mutating.
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        op_coeffs.resize(size(), 0.0);

        if (init_op_map.empty()) {
            return op_coeffs;
        }

        MonomialList del;
        for (const auto &kv : init_op_map) {
            const auto &mono = kv.first;
            const auto coeff = kv.second;
            if (const auto found = find(mono)) {
                op_coeffs[*found] = coeff;
                del.push_back(mono);
            }
        }

        for (const auto &mono : del) {
            init_op_map.erase(mono);
        }
        // Erasing does not give the slot array back, and a drained map is the normal end state: every
        // initial-operator term is materialized as a row by the time the caches are warmed, so without
        // this the propagator carries an empty map sized for the whole initial operator for its whole
        // life -- 2.5 MB behind zero entries for a 20k-term observable. Assignment, not clear(), because
        // clear() is what keeps the capacity.
        if (init_op_map.empty()) {
            init_op_map = MonomialMap{};
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
    auto update_initial_operator(const OperatorDict &op_dict, bool schrodinger) -> std::pair<MonomialList, VecD> {
        MonomialMap new_op_map;
        std::pair<MonomialList, VecD> new_grad_op;
        VecD new_op_coeffs(size(), 0.0);

        for (const auto &[k, v] : op_dict) {
            // Unchecked by design: the only caller bounds-checks against its logical_num_modes_.
            const auto mono = indices_to_bitset(k, num_bits());
            const auto rank_evolved_op = find(mono);
            const auto rank_init_op = init_op_map.find(mono);
            const auto coeff = algebra_encode_coeff(basis, v, mono);

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
        std::iota(new_inds.begin(), new_inds.end(), state_scored_rows_);

        with_store([&](const auto &rows) {
            const auto paired_inds = is_fully_paired(new_inds, rows, num_bits());
            state_rows_.reserve(state_rows_.size() + paired_inds.size());
            state_vals_.reserve(state_vals_.size() + paired_inds.size());

            // The algebra picks the diagonal ⟨b|·|b⟩ phase of each fully-paired term.
            algebra_score_state(basis, paired_inds, initial_state, rows, num_bits(), [this](size_t row, double phase) {
                state_rows_.push_back(static_cast<TermIndex>(row));
                state_vals_.push_back(phase);
            });
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

// Callers must pass pairwise-distinct, currently-absent keys: bulk_insert then skips duplicate probes and
// slot k deterministically lands at base+k. Call after any pass that reads pre-insert op state
// (op.size() must equal the returned base).
//
// `store` is passed alongside `op` rather than taken off it: every caller is inside build_layer, which
// has already bound the concrete backend, and re-entering with_store() here would bind it a second time
// per insert batch for nothing.
inline auto insert_absent_terms(auto &op, auto &store, size_t n, auto &&key_at, auto &&per_slot) -> size_t {
    const size_t base = store.grow_rows_geometric(n);
    for (size_t k = 0; k < n; ++k) {
        per_slot(k, base);
    }
    store.bulk_insert(n, base, std::forward<decltype(key_at)>(key_at));
    op.reindex_after_growth(base, n);
    return base;
}

template <typename FlatMap>
inline auto unordered_flat_map_storage_bytes(const FlatMap &map) -> size_t {
    return sizeof(FlatMap) + map.bucket_count() * (sizeof(typename FlatMap::value_type) + sizeof(unsigned char));
}

// The slot array plus what the keys own outside it. A monomial key wider than Bitset's inline capacity
// points at its own allocation, so slots alone under-report a wide operator's map by more than the slots
// themselves: 20k keys at 1024 modes hold 5.1 MB of words behind 2.5 MB of slots.
inline auto monomial_map_bytes(const MonomialMap &map) -> size_t {
    size_t total = unordered_flat_map_storage_bytes(map);
    for (const auto &kv : map) {
        total += kv.first.heap_bytes();
    }
    return total;
}

// No width parameter: nothing in here is width-dependent, and it never was -- every field is a byte
// count.
struct MPOperatorMemoryBreakdown final {
    size_t operator_terms_bytes = 0;
    size_t op_coeffs_bytes = 0;
    size_t state_coeffs_bytes = 0;
    size_t indexing_bytes = 0;
    size_t init_operator_bytes = 0;
    size_t initial_state_bytes = 0;
    size_t inverted_index_bytes = 0;

    // Diagnostics: breakdowns of the fields above, deliberately excluded from total_bytes() so they can
    // never double-count.
    size_t inverted_index_dense_bytes = 0;  // of inverted_index_bytes: full-height bitmap columns
    size_t inverted_index_sparse_bytes = 0; // of inverted_index_bytes: ascending set-row lists
    // of inverted_index_bytes: the Column vector itself, one entry per bit position. The only term that
    // scales with the mode count instead of the operator, so it is what a width sweep has to watch.
    size_t inverted_index_columns_bytes = 0;
    size_t inverted_index_dense_columns = 0;
    size_t operator_terms_slack_bytes = 0; // of operator_terms_bytes: unused geometric-growth capacity
    // of state_coeffs_bytes: entries of the state that are not exactly 0.0
    size_t state_coeffs_nonzero = 0;
    // Live entries of init_op_map. Not derivable from init_operator_bytes: a flat map keeps its bucket
    // count across erases, so the byte figure cannot tell a full map from a drained one.
    size_t init_operator_entries = 0;

    auto total_bytes() const -> size_t {
        return operator_terms_bytes + op_coeffs_bytes + state_coeffs_bytes + indexing_bytes + init_operator_bytes
               + initial_state_bytes + inverted_index_bytes;
    }

    auto operator+=(const MPOperatorMemoryBreakdown &o) -> MPOperatorMemoryBreakdown & {
        operator_terms_bytes += o.operator_terms_bytes;
        op_coeffs_bytes += o.op_coeffs_bytes;
        state_coeffs_bytes += o.state_coeffs_bytes;
        indexing_bytes += o.indexing_bytes;
        init_operator_bytes += o.init_operator_bytes;
        initial_state_bytes += o.initial_state_bytes;
        inverted_index_bytes += o.inverted_index_bytes;
        inverted_index_dense_bytes += o.inverted_index_dense_bytes;
        inverted_index_sparse_bytes += o.inverted_index_sparse_bytes;
        inverted_index_columns_bytes += o.inverted_index_columns_bytes;
        inverted_index_dense_columns += o.inverted_index_dense_columns;
        operator_terms_slack_bytes += o.operator_terms_slack_bytes;
        state_coeffs_nonzero += o.state_coeffs_nonzero;
        init_operator_entries += o.init_operator_entries;
        return *this;
    }
};

inline auto estimate_memory_usage(const MPOperator &op) -> MPOperatorMemoryBreakdown {
    MPOperatorMemoryBreakdown breakdown;
    op.with_store([&](const auto &rows) {
        breakdown.operator_terms_bytes = rows.memory_bytes();
        breakdown.indexing_bytes = rows.index_estimated_memory_bytes();
        breakdown.operator_terms_slack_bytes = rows.slack_bytes();
    });
    breakdown.op_coeffs_bytes = op.op_coeffs.capacity() * sizeof(double);
    // Every representation of the state at once: the sparse scored set plus the dense vector.
    breakdown.state_coeffs_bytes = op.state_coeffs.capacity() * sizeof(double)
                                   + op.state_rows_.capacity() * sizeof(TermIndex)
                                   + op.state_vals_.capacity() * sizeof(double);
    breakdown.init_operator_bytes = monomial_map_bytes(op.init_op_map);
    breakdown.init_operator_entries = op.init_op_map.size();
    breakdown.initial_state_bytes = op.initial_state.capacity() * sizeof(size_t);
    if (op.inverted_index_.has_value()) {
        breakdown.inverted_index_bytes = op.inverted_index_->memory_bytes();
        const auto tiers = op.inverted_index_->tier_memory_bytes();
        breakdown.inverted_index_dense_bytes = tiers[0];
        breakdown.inverted_index_sparse_bytes = tiers[1];
        breakdown.inverted_index_columns_bytes = op.inverted_index_->columns_bytes();
        breakdown.inverted_index_dense_columns = tiers[2];
    }
    // State phases are unit-magnitude, so at rest the scored count IS the nonzero count; a live vector needs a scan.
    breakdown.state_coeffs_nonzero =
        op.state_coeffs.empty()
            ? op.state_rows_.size()
            : static_cast<size_t>(std::ranges::count_if(op.state_coeffs, [](double c) { return c != 0.0; }));
    return breakdown;
}

} // namespace monoprop::detail
