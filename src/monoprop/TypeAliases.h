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

#include <bit>
#include <complex>
#include <cstddef>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <boost/unordered/unordered_flat_map.hpp>

#include "monoprop/Bitset.h"
#include "monoprop/Utilities.h"

namespace monoprop {
/*!
 * @brief Bitset container for a multi-qubit operator in Majorana basis.
 * @tparam NumModes Number of Fermionic modes.
 */
template <size_t NumModes>
using MajoranaSet = Bitset<2 * NumModes>;

/*!
 * @brief Scratchpad for storing Majorana sets in the operator to estimate.
 * @tparam NumModes Number of Fermionic modes.
 */
template <size_t NumModes>
using MajoranaVector = std::vector<MajoranaSet<NumModes>>;

template <size_t NumModes>
struct PrehashedMajoranaLookup {
    const MajoranaSet<NumModes> &key;
    size_t hash;
};

/*!
 * @brief
 * @tparam NumModes
 */
template <size_t NumModes>
struct MPHash final {
    using is_transparent = void;

    auto operator()(const MajoranaSet<NumModes> &arr) const noexcept -> size_t {
        return SplitmixHash<MajoranaSet<NumModes>>{}(arr);
    }

    auto operator()(const PrehashedMajoranaLookup<NumModes> &lookup) const noexcept -> size_t { return lookup.hash; }
};

template <size_t NumModes>
struct MPEqual final {
    using is_transparent = void;

    auto operator()(const MajoranaSet<NumModes> &lhs, const MajoranaSet<NumModes> &rhs) const noexcept -> bool {
        return lhs == rhs;
    }

    auto operator()(const PrehashedMajoranaLookup<NumModes> &lhs, const MajoranaSet<NumModes> &rhs) const noexcept
        -> bool {
        return lhs.key == rhs;
    }

    auto operator()(const MajoranaSet<NumModes> &lhs, const PrehashedMajoranaLookup<NumModes> &rhs) const noexcept
        -> bool {
        return lhs == rhs.key;
    }

    auto operator()(const PrehashedMajoranaLookup<NumModes> &lhs,
                    const PrehashedMajoranaLookup<NumModes> &rhs) const noexcept -> bool {
        return lhs.key == rhs.key;
    }
};

template <size_t NumModes>
using MajoranaOperator = boost::unordered_flat_map<MajoranaSet<NumModes>, double, MPHash<NumModes>, MPEqual<NumModes>>;

template <size_t NumModes>
using IndexMap = boost::unordered_flat_map<MajoranaSet<NumModes>, size_t, MPHash<NumModes>, MPEqual<NumModes>>;

/*!
 * @brief Sharded index map for cache-friendly lookup and shard-local updates.
 *
 * Wraps P independent IndexMap shards, routing keys by hash.
 * - find() is lock-free for concurrent reads
 * - emplace() routes to the correct shard
 * - callers can parallelize inserts when threads operate on disjoint shards
 *
 * The shard count is a power of two for fast modulo via bitwise AND.
 */
template <size_t NumModes>
class ShardedIndexMap {
public:
    using key_type = MajoranaSet<NumModes>;
    using mapped_type = size_t;
    using value_type = std::pair<const key_type, mapped_type>;
    using hasher = MPHash<NumModes>;
    using shard_type = IndexMap<NumModes>;
    using iterator = typename shard_type::iterator;
    using const_iterator = typename shard_type::const_iterator;
    using lookup_type = PrehashedMajoranaLookup<NumModes>;

    explicit ShardedIndexMap(size_t num_shards = 0) { reset(num_shards); }

    auto reset(size_t num_shards) -> void {
        num_shards_ = num_shards < 2 ? 1 : std::bit_ceil(num_shards);
        mask_ = num_shards_ - 1;
        shards_.resize(num_shards_);
        for (auto &s : shards_) {
            s.clear();
        }
    }

    auto num_shards() const -> size_t { return num_shards_; }

    // Route a key to its shard index.
    auto shard_for(const key_type &key) const -> size_t { return hasher{}(key)&mask_; }
    auto shard_for_hash(size_t hash) const -> size_t { return hash & mask_; }

    // Lookup — routes to correct shard then probes. Lock-free for concurrent reads.
    auto find(const key_type &key) const -> const_iterator {
        const auto hash = hasher{}(key);
        return find_prehashed(key, hash);
    }

    auto find(const key_type &key) -> iterator {
        const auto hash = hasher{}(key);
        return find_prehashed(key, hash);
    }

    auto find_prehashed(const key_type &key, size_t hash) const -> const_iterator {
        return shards_[shard_for_hash(hash)].find(lookup_type{key, hash});
    }

    auto find_prehashed(const key_type &key, size_t hash) -> iterator {
        return shards_[shard_for_hash(hash)].find(lookup_type{key, hash});
    }

    // Returns end() iterator for the shard that `key` would route to.
    // For use with: if (it != map.end_for(key))
    auto end_for(const key_type &key) const -> const_iterator { return shards_[shard_for(key)].end(); }

    auto end_for(const key_type &key) -> iterator { return shards_[shard_for(key)].end(); }

    auto end_for_hash(size_t hash) const -> const_iterator { return shards_[shard_for_hash(hash)].end(); }

    auto end_for_hash(size_t hash) -> iterator { return shards_[shard_for_hash(hash)].end(); }

    // Insert — routes to correct shard.
    auto emplace(const key_type &key, mapped_type value) { return shards_[shard_for(key)].emplace(key, value); }

    // Subscript operator — routes to correct shard.
    auto operator[](const key_type &key) -> mapped_type & { return shards_[shard_for(key)][key]; }

    // Reserve across all shards (divides evenly).
    auto reserve(size_t total) -> void {
        const size_t per_shard = (total + num_shards_ - 1) / num_shards_;
        for (auto &s : shards_)
            s.reserve(per_shard);
    }

    auto size() const -> size_t {
        size_t total = 0;
        for (const auto &s : shards_)
            total += s.size();
        return total;
    }

    auto empty() const -> bool {
        for (const auto &s : shards_) {
            if (!s.empty())
                return false;
        }
        return true;
    }

    auto estimated_memory_bytes() const -> size_t {
        size_t bytes = sizeof(ShardedIndexMap);
        bytes += shards_.capacity() * sizeof(shard_type);
        for (const auto &shard : shards_) {
            bytes += shard.bucket_count() * (sizeof(value_type) + sizeof(unsigned char));
        }
        return bytes;
    }

    auto clear() -> void {
        for (auto &s : shards_)
            s.clear();
    }

    // Direct shard access for parallel batch operations.
    auto shard(size_t idx) -> shard_type & { return shards_[idx]; }
    auto shard(size_t idx) const -> const shard_type & { return shards_[idx]; }

    // Iteration across all shards (for serialization, debugging, etc.)
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (const auto &s : shards_) {
            for (const auto &kv : s) {
                fn(kv.first, kv.second);
            }
        }
    }

private:
    size_t num_shards_ = 1;
    size_t mask_ = 0;
    std::vector<shard_type> shards_{1};
};

using VecCD = std::vector<std::complex<double>>;

using VecD = std::vector<double>;

using VecI = std::vector<int>;

using VecZ = std::vector<size_t>;

using FermiOperatorMap = std::map<VecZ, std::complex<double>>;

using CyclesType = std::vector<std::vector<std::pair<size_t, size_t>>>;

/**
 * @brief A local cycle where both source and target indices are on the same rank.
 */
struct LocalCycle {
    size_t src;
    size_t tgt;
    int phase;
};

/**
 * @brief Cross-rank cycles for a single remote rank, stored contiguously.
 * outgoing: cycles where we own src (send src values, receive tgt updates)
 * incoming: cycles where we own tgt (send tgt values, receive src updates)
 *
 * For single-communication evolution, we send [outgoing values] + [incoming values]
 * and receive [incoming shadow values] + [outgoing shadow values] in one alltoallv.
 */
struct CrossRankCycles {
    // Outgoing: we own source indices
    VecZ out_indices; // source indices
    VecI out_phases;  // phases for outgoing

    // Incoming: we own target indices
    VecZ in_indices; // target indices
    VecI in_phases;  // phases for incoming

    auto empty() const -> bool { return out_indices.empty() && in_indices.empty(); }
    auto out_size() const -> size_t { return out_indices.size(); }
    auto in_size() const -> size_t { return in_indices.size(); }
};

/**
 * @brief Result of splitting cycles into local and cross-rank storage.
 *
 * Uses flattened CrossRankCycles structure for efficient storage.
 */
struct SplitCycleResult {
    std::vector<LocalCycle> local_cycles;    // Local cycles (src, tgt, phase)
    std::vector<CrossRankCycles> cross_rank; // Indexed by remote rank
};

/**
 * @brief Structural truncation criterion applied to Majorana monomials after each gate.
 *
 * Both criteria share one rule: a *fully paired* monomial -- one whose support
 * consists entirely of complete pairs m_{2j-1} m_{2j} on a mode -- is always kept,
 * regardless of the cutoff. Fully paired monomials are exactly the terms that can
 * contribute to an expectation value against a computational-basis state or Slater
 * determinant, so discarding them would throw away signal. The criteria differ only
 * in how they measure the remaining, partially paired monomials.
 */
enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

template <size_t NumModes>
using CutoffFn = std::function<bool(const MajoranaSet<NumModes> &)>;

// Forward declarations for functions used in MPOperator
template <size_t NumModes>
auto is_fully_paired(const VecZ &inds, const MajoranaVector<NumModes> &op) -> VecZ;

template <size_t NumModes>
auto get_hf_phases(const VecZ &paired_inds, const VecZ &hf, const MajoranaVector<NumModes> &op) -> VecD;

template <size_t NumModes>
auto encode_coeff(const std::complex<double> &coeff, const MajoranaSet<NumModes> &maj) -> double;

template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> MajoranaSet<NumModes>;

template <size_t NumModes>
struct MPOperator {
    MajoranaVector<NumModes> op = {};
    VecD op_coeffs = {};
    VecD state_coeffs = {};
    ShardedIndexMap<NumModes> indexing{};
    MajoranaOperator<NumModes> init_op_map_ = {};
    VecZ slater_determinant_ = {};

    auto size() const -> size_t { return op.size(); }

    /**
     * @brief Gets the current operator coefficients
     *
     * Retrieves or initializes the operator coefficients vector.
     * Translates operator format representation to vector format as needed.
     *
     * @return Constant reference to the operator coefficients vector
     */
    auto get_operator() -> const VecD & {
        if (size() == op_coeffs.size()) {
            return op_coeffs;
        }

        op_coeffs.resize(size(), 0.0);

        if (init_op_map_.empty()) {
            return op_coeffs;
        }

        // Snapshot keys/values to enable parallel iteration.
        std::vector<std::pair<MajoranaSet<NumModes>, double>> items;
        items.reserve(init_op_map_.size());
        for (const auto &kv : init_op_map_) {
            items.emplace_back(kv.first, kv.second);
        }

        tbb::enumerable_thread_specific<std::vector<MajoranaSet<NumModes>>> tls_del;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, items.size()),
                          [this, &tls_del, &items](const tbb::blocked_range<size_t> &r) {
                              auto &local_del = tls_del.local();
                              for (auto i = r.begin(); i != r.end(); ++i) {
                                  const auto &maj = items[i].first;
                                  const auto coeff = items[i].second;
                                  if (const auto it = indexing.find(maj); it != indexing.end_for(maj)) {
                                      op_coeffs[it->second] = coeff;
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
     * Extends the vector as needed when the operator size changes.
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

        // Create vector with new indices only once
        VecZ new_inds(new_elements);
        std::iota(new_inds.begin(), new_inds.end(), cur_len);

        const auto paired_inds = is_fully_paired<NumModes>(new_inds, op);
        const auto hf_phases = get_hf_phases<NumModes>(paired_inds, slater_determinant_, op);

        tbb::parallel_for(size_t{0}, paired_inds.size(), [this, &paired_inds, &hf_phases](size_t i) {
            state_coeffs[paired_inds[i]] = hf_phases[i];
        });

        return state_coeffs;
    }

    /**
     * @brief Updates the initial operator with new terms
     *
     * @param op_dict New operator terms to update
     * @param schrodinger Whether in Schrodinger picture
     * @return Tuple containing (new_op_map, new_op_coeffs, new_grad_op)
     */
    auto update_initial_operator(const FermiOperatorMap &op_dict, bool schrodinger)
        -> std::tuple<MajoranaOperator<NumModes>, VecD, std::pair<MajoranaVector<NumModes>, VecD>> {
        // Update the operator with new elements for the specified rank
        MajoranaOperator<NumModes> new_op_map;
        std::pair<MajoranaVector<NumModes>, VecD> new_grad_op;
        VecD new_op_coeffs(size(), 0.0);

        for (const auto &[k, v] : op_dict) {
            const auto maj = indices_to_bitset<NumModes>(k);
            const auto rank_evolved_op = indexing.find(maj);
            const auto rank_init_op = init_op_map_.find(maj);
            const auto coeff = encode_coeff<NumModes>(v, maj);

            // in heisenberg picture, we cannot change the initial operator if the majorana is not present
            // this is because these paths from new majoranas may not be present in the evolution graph
            if (!schrodinger) {
                if (rank_init_op != init_op_map_.end()) {
                    new_op_map[maj] = coeff;
                }
                else if (rank_evolved_op != indexing.end_for(maj)) {
                    new_op_coeffs[rank_evolved_op->second] = coeff;
                }
                else {
                    const auto term_repr = std::format("[{}]", join_with_separator(k, ", "));
                    throw std::runtime_error(std::format("Operator term {} not found in the operator.", term_repr));
                }
            }
            // otherwise, in schrodinger picture, we can change the initial operator freely as the state has
            // been evolved to construct the graph
            else {
                if (rank_evolved_op != indexing.end_for(maj)) {
                    new_op_coeffs[rank_evolved_op->second] = coeff;
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

    auto total_bytes() const -> size_t {
        return operator_terms_bytes + op_coeffs_bytes + state_coeffs_bytes + indexing_bytes + init_operator_bytes
               + slater_determinant_bytes;
    }
};

template <size_t NumModes>
inline auto estimate_memory_usage(const MPOperator<NumModes> &mp_op) -> MPOperatorMemoryBreakdown<NumModes> {
    MPOperatorMemoryBreakdown<NumModes> breakdown;
    breakdown.operator_terms_bytes = mp_op.op.capacity() * sizeof(MajoranaSet<NumModes>);
    breakdown.op_coeffs_bytes = mp_op.op_coeffs.capacity() * sizeof(double);
    breakdown.state_coeffs_bytes = mp_op.state_coeffs.capacity() * sizeof(double);
    breakdown.indexing_bytes = mp_op.indexing.estimated_memory_bytes();
    breakdown.init_operator_bytes = unordered_flat_map_storage_bytes(mp_op.init_op_map_);
    breakdown.slater_determinant_bytes = mp_op.slater_determinant_.capacity() * sizeof(size_t);
    return breakdown;
}
} // namespace monoprop
