#pragma once

#include <algorithm>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include <format>
#include "monoprop/detail/print_compat.h"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/Sidecar.h"

namespace monoprop {

// Forward declarations for functions used in MPOperator. `Rows` is either the generic
// MajoranaVector (plain vector) or the packed operator-row container; both are read through
// the backend-agnostic row accessors.
template <size_t NumModes, typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op) -> VecZ;

template <size_t NumModes, typename Rows>
auto get_hf_phases(const VecZ &paired_inds, const VecZ &hf, const Rows &op) -> VecD;

template <size_t NumModes>
auto encode_coeff(const std::complex<double> &coeff, const MajoranaSet<NumModes> &maj) -> double;

template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> MajoranaSet<NumModes>;

template <size_t NumModes>
struct MPOperator {
    // Operator rows are stored entropy-packed (position-list rows, ALWAYS — every NumModes);
    // all row reads/writes go through the backend-agnostic accessors (materialize_row/assign_row/
    // row_popcount/for_each_row_position) or the container's own packed API.
    // The store is non-movable (RowEq holds a fixed back-pointer), so it lives on the heap: an
    // MPOperator owns it by unique_ptr, keeping MPOperator itself cheaply movable. Always non-null.
    std::unique_ptr<detail::OperatorIndex<NumModes>> op = std::make_unique<detail::OperatorIndex<NumModes>>();
    VecD op_coeffs = {};
    VecD state_coeffs = {};
    MajoranaOperator<NumModes> init_op_map_ = {};
    VecZ slater_determinant_ = {};
    mutable std::optional<EvenParityMajoranaScanSidecar<NumModes>> even_parity_scan_sidecar_ = std::nullopt;

    MPOperator() noexcept = default;
    MPOperator(MPOperator &&) noexcept = default;
    MPOperator &operator=(MPOperator &&) noexcept = default;

    // Deep copy via the copy constructor only (enables the simulator's deep copy / __deepcopy__).
    // `op` is owned by unique_ptr and the OperatorIndex is non-copyable (self back-pointer), so it is
    // rebuilt via clone(); everything else is plain value data. Copy assignment stays implicitly
    // deleted (unique_ptr member) -- copy construction is all deepcopy needs.
    MPOperator(const MPOperator &other)
        : op(other.op->clone()),
          op_coeffs(other.op_coeffs),
          state_coeffs(other.state_coeffs),
          init_op_map_(other.init_op_map_),
          slater_determinant_(other.slater_determinant_),
          even_parity_scan_sidecar_(other.even_parity_scan_sidecar_) {}

    auto size() const -> size_t { return op->size(); }

    auto append_term(const MajoranaSet<NumModes> &maj) -> void {
        op->push_back(maj);
        if (even_parity_scan_sidecar_.has_value()) {
            even_parity_scan_sidecar_->append_row(maj);
        }
    }

    // After a bulk PARALLEL growth of `op` (slots [base, base+n) already filled by the caller),
    // bring the even-parity sidecar back in sync via the atomics-free word-partitioned append,
    // preserving the has_value() ⟹ rows()==op.size() invariant (rows()==base holds pre-growth).
    auto resync_sidecar_after_bulk_growth(size_t base, size_t n) -> void {
        if (even_parity_scan_sidecar_.has_value()) {
            even_parity_scan_sidecar_->append_rows_from_op_disjoint(*op, base, n);
        }
    }

    auto even_parity_scan_sidecar() const -> const EvenParityMajoranaScanSidecar<NumModes> & {
        if (!even_parity_scan_sidecar_.has_value() || even_parity_scan_sidecar_->rows() != op->size()) {
            even_parity_scan_sidecar_.emplace();
            even_parity_scan_sidecar_->rebuild(*op);
        }
        return *even_parity_scan_sidecar_;
    }

    /**
     * @brief Gets the current Hamiltonian coefficients
     *
     * Retrieves or initializes the Hamiltonian coefficients vector.
     * Translates operator format representation to vector format as needed.
     *
     * @return Constant reference to the Hamiltonian coefficients vector
     */
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        op_coeffs.resize(size(), 0.0);

        if (init_op_map_.empty()) {
            return op_coeffs;
        }

        // Snapshot keys/values to enable parallel iteration (the flat_map itself isn't safely iterable
        // while erasing). Matched entries are erased afterward, single-threaded, to avoid concurrent
        // map mutation.
        std::vector<std::pair<MajoranaSet<NumModes>, double>> items;
        items.reserve(init_op_map_.size());
        for (const auto &kv : init_op_map_) {
            items.emplace_back(kv.first, kv.second);
        }

        tbb::enumerable_thread_specific<std::vector<MajoranaSet<NumModes>>> tls_del;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, items.size()), [&](const tbb::blocked_range<size_t> &r) {
            auto &local_del = tls_del.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto &maj = items[i].first;
                const auto coeff = items[i].second;
                if (const auto i = op->find(maj)) {
                    op_coeffs[*i] = coeff;
                    local_del.push_back(maj);
                }
            }
        });

        // Batch erase processed entries (do erases single-threaded).
        size_t total = 0;
        for (auto &v : tls_del) {
            total += v.size();
        }
        std::vector<MajoranaSet<NumModes>> del_list;
        del_list.reserve(total);
        for (auto &v : tls_del) {
            del_list.insert(del_list.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
        }
        for (const auto &maj : del_list) {
            init_op_map_.erase(maj);
        }

        return op_coeffs;
    }

    /**
     * @brief Gets the current state coefficients
     *
     * Retrieves or initializes the state vector based on the slater determinant.
     * Extends the vector as needed when the Hamiltonian size changes.
     *
     * @return Constant reference to the state coefficients vector
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

        const auto paired_inds = is_fully_paired<NumModes>(new_inds, *op);
        const auto hf_phases = get_hf_phases<NumModes>(paired_inds, slater_determinant_, *op);

        tbb::parallel_for(size_t{0}, paired_inds.size(), [&](size_t i) {
            state_coeffs[paired_inds[i]] = hf_phases[i];
        });

        return state_coeffs;
    }

    /**
     * @brief Updates the initial Hamiltonian with new terms
     *
     * @param op_dict New Hamiltonian terms to update
     * @param schrodinger Whether in Schrodinger picture
     * @return Tuple containing (new_op_map, new_op_coeffs, new_grad_op)
     */
    auto update_initial_operator(const FermiOperatorMap &op_dict, bool schrodinger)
        -> std::tuple<MajoranaOperator<NumModes>, VecD, std::pair<MajoranaVector<NumModes>, VecD>> {
        // Update the Hamiltonian with new elements for the specified rank
        MajoranaOperator<NumModes> new_op_map;
        std::pair<MajoranaVector<NumModes>, VecD> new_grad_op;
        VecD new_op_coeffs(size(), 0.0);

        for (const auto &[k, v] : op_dict) {
            const auto maj = indices_to_bitset<NumModes>(k);
            const auto rank_evolved_op = op->find(maj);
            const auto rank_init_op = init_op_map_.find(maj);
            const auto coeff = encode_coeff<NumModes>(v, maj);

            // in heisenberg picture, we cannot change the initial hamiltonian if the majorana is not present
            // this is because these paths from new majoranas may not be present in the evolution graph
            if (!schrodinger) {
                if (rank_init_op != init_op_map_.end()) {
                    new_op_map[maj] = coeff;
                }
                else if (rank_evolved_op) {
                    new_op_coeffs[*rank_evolved_op] = coeff;
                }
                else {
                    const auto term_repr = std::format("[{}]", join_with_separator(k, ", "));
                    throw std::runtime_error(
                        std::format("Operator term {} not found in the operator.", term_repr));
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
        init_op_map_ = new_op_map;
        op_coeffs = new_op_coeffs;
        return {std::move(new_op_map), std::move(new_op_coeffs), std::move(new_grad_op)};
    }
};

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
    size_t even_parity_scan_sidecar_bytes = 0;

    auto total_bytes() const -> size_t {
        return operator_terms_bytes + op_coeffs_bytes + state_coeffs_bytes + indexing_bytes + init_operator_bytes
               + slater_determinant_bytes + even_parity_scan_sidecar_bytes;
    }
};

template <size_t NumModes>
inline auto estimate_memory_usage(const MPOperator<NumModes> &mbs_op) -> MPOperatorMemoryBreakdown<NumModes> {
    MPOperatorMemoryBreakdown<NumModes> breakdown;
    // Packed rows store stride bytes/row (+ overflow side-map), not sizeof(MajoranaSet); ask directly.
    breakdown.operator_terms_bytes = mbs_op.op->memory_bytes();
    breakdown.op_coeffs_bytes = mbs_op.op_coeffs.capacity() * sizeof(double);
    breakdown.state_coeffs_bytes = mbs_op.state_coeffs.capacity() * sizeof(double);
    breakdown.indexing_bytes = mbs_op.op->index_estimated_memory_bytes();
    breakdown.init_operator_bytes = unordered_flat_map_storage_bytes(mbs_op.init_op_map_);
    breakdown.slater_determinant_bytes = mbs_op.slater_determinant_.capacity() * sizeof(size_t);
    if (mbs_op.even_parity_scan_sidecar_.has_value()) {
        breakdown.even_parity_scan_sidecar_bytes = mbs_op.even_parity_scan_sidecar_->memory_bytes();
    }
    return breakdown;
}

} // namespace monoprop
