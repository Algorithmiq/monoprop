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
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <format>
#include "monoprop/detail/print_compat.h"

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/OperatorIndex.h"

// Forward declarations of MajoranaAlgebra.h helpers (namespace monoprop) so this header need not
// include it. `Rows` is either the generic MajoranaVector (plain vector) or the packed operator-row
// container; both are read through the backend-agnostic row accessors.
namespace monoprop {
template <size_t NumModes, typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op) -> VecZ;

template <size_t NumModes, typename Rows>
auto get_hf_phases(const VecZ &paired_inds, const VecZ &hf, const Rows &op) -> VecD;

template <size_t NumModes>
auto encode_coeff(const std::complex<double> &coeff, const MajoranaSet<NumModes> &maj) -> double;

template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> MajoranaSet<NumModes>;

// Pauli-basis helpers (defined in PauliAlgebra.h). Forward-declared here so this header stays free of a
// PauliAlgebra.h include (which would form a cycle through TypeAliases.h); every TU that instantiates the
// MPOperator methods below also includes PauliAlgebra.h transitively (via CosineRecompute.h).
template <size_t NumModes>
auto get_hf_mask(const VecZ &hf) -> MajoranaSet<NumModes>;

template <size_t NumModes>
auto pauli_hf_phase(const MajoranaSet<NumModes> &maj, const MajoranaSet<NumModes> &hf_mask) -> double;

auto encode_pauli_coeff(const std::complex<double> &coeff) -> double;
} // namespace monoprop

namespace monoprop::detail {

/// The propagated operator: the term store (entropy-packed rows + keyless hash index), its
/// coefficient vectors, the initial-operator map, and the lazily-built even-parity scan inverted index.
template <size_t NumModes>
struct MPOperator {
    // Operator rows are stored entropy-packed (position-list rows, ALWAYS — every NumModes);
    // all row reads/writes go through the backend-agnostic accessors (materialize_row/assign_row/
    // row_popcount/for_each_row_position) or the container's own packed API.
    // The store is non-copyable/non-movable, so it lives on the heap: an MPOperator owns it by
    // unique_ptr, keeping MPOperator itself cheaply movable. Always non-null.
    std::unique_ptr<OperatorIndex<NumModes>> store = std::make_unique<OperatorIndex<NumModes>>();
    VecD op_coeffs = {};
    VecD state_coeffs = {};
    MajoranaOperator<NumModes> init_op_map = {};
    VecZ slater_determinant = {};
    // Operator basis: Majorana monomials (default) or native Pauli strings. Selects the coefficient
    // encoding (identity for Pauli) and the ⟨b|·|b⟩ scoring (pauli_hf_phase vs hf_phase) in get_state /
    // the fused resolver. Set once at propagator construction (MonomialPropagator ctor).
    Basis basis = Basis::Majorana;
    mutable std::optional<InvertedIndex<NumModes>> inverted_index_ = std::nullopt;

    MPOperator() noexcept = default;
    MPOperator(MPOperator &&) noexcept = default;
    MPOperator &operator=(MPOperator &&) noexcept = default;

    // Deep copy via the copy constructor only (enables the simulator's deep copy / __deepcopy__).
    // `store` is owned by unique_ptr and the OperatorIndex is non-copyable, so it is rebuilt via
    // clone(); everything else is plain value data. Copy assignment stays implicitly deleted
    // (unique_ptr member) -- copy construction is all deepcopy needs.
    MPOperator(const MPOperator &other)
        : store(other.store->clone()),
          op_coeffs(other.op_coeffs),
          state_coeffs(other.state_coeffs),
          init_op_map(other.init_op_map),
          slater_determinant(other.slater_determinant),
          basis(other.basis),
          inverted_index_(other.inverted_index_) {}

    auto size() const -> size_t { return store->size(); }

    auto append_term(const MajoranaSet<NumModes> &maj) -> void {
        store->push_back(maj);
        if (inverted_index_.has_value()) {
            inverted_index_->append_row(maj);
        }
    }

    // After a bulk PARALLEL growth of `store` (slots [base, base+n) already filled by the caller),
    // bring the even-parity inverted index back in sync via the atomics-free word-partitioned append,
    // preserving the has_value() ⟹ rows()==store.size() invariant (rows()==base holds pre-growth).
    auto reindex_after_growth(size_t base, size_t n) -> void {
        if (inverted_index_.has_value()) {
            inverted_index_->append_rows_from_op_disjoint(*store, base, n);
        }
    }

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
     * Resizes op_coeffs to the current term count and drains any pending terms from init_op_map into
     * it: each pending (term, coeff) is looked up in the store and written at its row index, then
     * erased from init_op_map (erase after the lookup loop — the flat_map is not iterable while
     * mutating). A no-op once op_coeffs is already in sync with the store.
     *
     * @return Const reference to the row-indexed coefficient vector (valid until the operator grows).
     */
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        op_coeffs.resize(size(), 0.0);

        if (init_op_map.empty()) {
            return op_coeffs;
        }

        // Match keys in ascending map order; the found entries are erased afterward (the flat_map is not
        // safely mutable mid-iteration), so the erase order is deterministic (ascending map order).
        std::vector<MajoranaSet<NumModes>> del;
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
     * Extends state_coeffs to the current term count and scores ONLY the newly-appended terms
     * [old_size, size()); existing entries are left untouched. A new term is nonzero only if it is
     * fully paired with respect to the Slater determinant, in which case it receives that term's
     * Hartree–Fock phase.
     *
     * @return Const reference to the state coefficient vector (valid until the operator grows).
     */
    auto get_state() -> const VecD & {
        if (state_coeffs.size() == size()) {
            return state_coeffs;
        }

        size_t cur_len = state_coeffs.size();
        size_t new_elements = size() - cur_len;
        state_coeffs.resize(size(), 0.0);

        // Only score the newly-appended terms [cur_len, size); already-set coeffs stay untouched.
        VecZ new_inds(new_elements);
        std::iota(new_inds.begin(), new_inds.end(), cur_len);

        const auto paired_inds = is_fully_paired<NumModes>(new_inds, *store);

        // A Z-only (paired) Pauli scores ⟨b|P|b⟩ = (−1)^{|Z∩occ|} with no Majorana pairing sign, so
        // Pauli uses pauli_hf_phase rather than hf_phase. The occupancy mask marks slot 2q; for a paired
        // term slots 2q and 2q+1 agree, so the same get_hf_mask feeds both phases (see pauli_hf_phase).
        if (basis == Basis::Pauli) {
            const auto hf_mask = get_hf_mask<NumModes>(slater_determinant);
            for (size_t i = 0; i < paired_inds.size(); ++i) {
                const auto &row = materialize_row<NumModes>(*store, paired_inds[i]);
                state_coeffs[paired_inds[i]] = pauli_hf_phase<NumModes>(row, hf_mask);
            }
            return state_coeffs;
        }

        const auto hf_phases = get_hf_phases<NumModes>(paired_inds, slater_determinant, *store);

        for (size_t i = 0; i < paired_inds.size(); ++i) {
            state_coeffs[paired_inds[i]] = hf_phases[i];
        }

        return state_coeffs;
    }

    /**
     * @brief Rewrite the initial Hamiltonian from a new coefficient dictionary.
     *
     * Places each term either directly on its existing evolved-operator row (new_op_coeffs) or in the
     * pending map (new_op_map) for terms not yet materialized. The picture governs unknown terms:
     *   - Heisenberg (schrodinger == false): a term absent from BOTH the pending map and the evolved
     *     store is rejected (throws) — new Majoranas may have no paths in the evolution graph.
     *   - Schrödinger: terms may be introduced freely, since the state was already evolved to build
     *     the graph.
     * Overwrites the operator's internal init_op_map and op_coeffs with the result.
     *
     * @param op_dict     New Hamiltonian terms (Fermionic operator → coefficient).
     * @param schrodinger Whether the simulator is in the Schrödinger picture.
     * @return Tuple {new_op_map (pending terms), new_op_coeffs (row-indexed coeffs),
     *         new_grad_op (parallel (majorana, coeff) arrays of every supplied term)}.
     */
    auto update_initial_operator(const FermiOperatorMap &op_dict, bool schrodinger)
        -> std::tuple<MajoranaOperator<NumModes>, VecD, std::pair<MajoranaVector<NumModes>, VecD>> {
        // Update the Hamiltonian with new elements for the specified rank
        MajoranaOperator<NumModes> new_op_map;
        std::pair<MajoranaVector<NumModes>, VecD> new_grad_op;
        VecD new_op_coeffs(size(), 0.0);

        for (const auto &[k, v] : op_dict) {
            const auto maj = indices_to_bitset<NumModes>(k);
            const auto rank_evolved_op = store->find(maj);
            const auto rank_init_op = init_op_map.find(maj);
            const auto coeff = (basis == Basis::Pauli) ? encode_pauli_coeff(v) : encode_coeff<NumModes>(v, maj);

            // in heisenberg picture, we cannot change the initial hamiltonian if the majorana is not present
            // this is because these paths from new majoranas may not be present in the evolution graph
            if (!schrodinger) {
                if (rank_init_op != init_op_map.end()) {
                    new_op_map[maj] = coeff;
                }
                else if (rank_evolved_op) {
                    new_op_coeffs[*rank_evolved_op] = coeff;
                }
                else {
                    const auto term_repr = std::format("[{}]", join_with_separator(k, ", "));
                    throw std::runtime_error(std::format("Operator term {} not found in the operator.", term_repr));
                }
            }
            // otherwise, in schrodinger picture, we can change the initial hamiltonian freely as the state has
            // been evolved to construct the graph
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

        // Update the internal state for the specified rank
        init_op_map = new_op_map;
        op_coeffs = new_op_coeffs;
        return {std::move(new_op_map), std::move(new_op_coeffs), std::move(new_grad_op)};
    }
};

// Insert `n` provably-distinct, currently-absent terms into `op` in one deterministic batch — the
// grow → scatter → index → resync quartet shared by every miss-insert site (cross-rank incoming misses
// and deferred self-misses). Steps: grow the row store by `n` (returning the insert base = old size);
// have the caller scatter each term's packed row + any side records via `per_slot(k, base)` (writing
// the disjoint slot base+k); bulk-insert the keys from `key_at(k)`; resync the inverted index. The
// base+k assignment is byte-identical to a serial loop because callers pass pairwise-distinct keys
// (source ⊕ G over distinct terms, ⊕G injective); atomics-free (disjoint op slots / map shards /
// inverted-index words). Call AFTER any pass that reads pre-insert op state — op.size() must equal the
// returned base. `key_at(k) -> const MajoranaSet<NumModes>&`, `per_slot(k, base) -> void`.
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
        return *this;
    }
};

template <size_t NumModes>
inline auto estimate_memory_usage(const MPOperator<NumModes> &op) -> MPOperatorMemoryBreakdown<NumModes> {
    MPOperatorMemoryBreakdown<NumModes> breakdown;
    // Packed rows store stride bytes/row (+ overflow side-map), not sizeof(MajoranaSet); ask directly.
    breakdown.operator_terms_bytes = op.store->memory_bytes();
    breakdown.op_coeffs_bytes = op.op_coeffs.capacity() * sizeof(double);
    breakdown.state_coeffs_bytes = op.state_coeffs.capacity() * sizeof(double);
    breakdown.indexing_bytes = op.store->index_estimated_memory_bytes();
    breakdown.init_operator_bytes = unordered_flat_map_storage_bytes(op.init_op_map);
    breakdown.slater_determinant_bytes = op.slater_determinant.capacity() * sizeof(size_t);
    if (op.inverted_index_.has_value()) {
        breakdown.inverted_index_bytes = op.inverted_index_->memory_bytes();
    }
    return breakdown;
}

} // namespace monoprop::detail
