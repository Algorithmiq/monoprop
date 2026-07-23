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

#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

inline constexpr size_t kMissingIndex = std::numeric_limits<size_t>::max();

template <size_t NumModes>
inline constexpr size_t kWords = MajoranaSet<NumModes>::num_words();

template <typename Func>
inline auto parallel_for_indices(size_t count, Func &&func, size_t grain_size = 256) -> void {
    threading::parallel_for_indices(count, std::forward<Func>(func), grain_size);
}

inline auto effective_parallelism() -> size_t {
    return threading::effective_parallelism();
}

inline auto empty_coeffs() -> const VecD & {
    static const VecD coeffs;
    return coeffs;
}

inline auto collect_cycle_sources(const CyclesType &cycles_by_target) -> VecZ {
    VecZ cycle_sources;
    size_t total_cycles = 0;
    for (const auto &cycles_for_target : cycles_by_target) {
        total_cycles += cycles_for_target.size();
    }
    cycle_sources.reserve(total_cycles);
    for (const auto &cycles_for_target : cycles_by_target) {
        for (const auto &cycle : cycles_for_target) {
            cycle_sources.push_back(cycle.first);
        }
    }
    return cycle_sources;
}

inline auto sort_unique_indices(VecZ &indices) -> void {
    if (indices.empty()) {
        return;
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
}

inline auto build_excluded_index_mask(const VecZ &excluded) -> std::pair<std::vector<unsigned char>, size_t> {
    size_t max_idx = 0;
    for (const auto idx : excluded) {
        max_idx = std::max(max_idx, idx);
    }

    std::vector<unsigned char> mask(max_idx + 1, 0);
    for (const auto idx : excluded) {
        mask[idx] = 1;
    }

    return {std::move(mask), max_idx};
}

inline auto remove_excluded_indices(VecZ indices, const VecZ &excluded) -> VecZ {
    if (indices.empty() || excluded.empty()) {
        return indices;
    }

    const auto [is_excluded, max_idx] = build_excluded_index_mask(excluded);

    size_t write = 0;
    for (size_t read = 0; read < indices.size(); ++read) {
        const bool keep = indices[read] > max_idx || !is_excluded[indices[read]];
        indices[write] = indices[read];
        write += static_cast<size_t>(keep);
    }
    indices.resize(write);

    return indices;
}

inline auto build_filtered_compressed_cosine_data(const VecZ &indices, const VecZ &excluded) -> CompressedCosineData {
    if (indices.empty()) {
        return {};
    }

    if (excluded.empty()) {
        return build_compressed_cosine_data(indices);
    }

    CompressedCosineData filtered_data;
    filtered_data.total_count = 0;
    reserve_compressed_cosine_data(filtered_data, indices.size());

    const auto [is_excluded, max_idx] = build_excluded_index_mask(excluded);

    PendingIndexRun pending_run;
    for (const auto idx : indices) {
        if (idx <= max_idx && is_excluded[idx]) {
            continue;
        }

        append_cosine_index(filtered_data, idx, pending_run);
        ++filtered_data.total_count;
    }

    finish_pending_cosine_run(filtered_data, pending_run);
    return filtered_data;
}

inline auto should_process_anticommuting(size_t maj_pop, size_t gen_maj_pop, size_t overlap, int only_rotate_len_k)
    -> bool {
    if (only_rotate_len_k > 0 && maj_pop > static_cast<size_t>(only_rotate_len_k)) {
        return false;
    }
    return majs_anticommute(maj_pop, gen_maj_pop, overlap);
}

template <typename T>
inline auto reset_nested_buffers(std::vector<std::vector<T>> &buffers, size_t size) -> void {
    if (buffers.size() != size) {
        buffers.resize(size);
    }
    for (auto &buffer : buffers) {
        buffer.clear();
    }
}

} // namespace monoprop::detail

namespace monoprop {

template <size_t NumModes>
struct AnticommutingOpData {
    MajoranaSet<NumModes> original_maj;
    MajoranaSet<NumModes> new_maj;
    size_t cycle_index;
    size_t maj_pop = 0;
    size_t overlap = 0;
    int phase;
    bool is_below_sin_atol;

    AnticommutingOpData() = default;

    AnticommutingOpData(const MajoranaSet<NumModes> &original_maj,
                        const MajoranaSet<NumModes> &new_maj,
                        size_t cycle_index,
                        size_t maj_pop,
                        size_t overlap,
                        int phase,
                        bool is_below_sin_atol)
        : original_maj(original_maj),
          new_maj(new_maj),
          cycle_index(cycle_index),
          maj_pop(maj_pop),
          overlap(overlap),
          phase(phase),
          is_below_sin_atol(is_below_sin_atol) {}

    AnticommutingOpData(const MajoranaSet<NumModes> &new_maj, size_t cycle_index, int phase, bool is_below_sin_atol)
        : new_maj(new_maj),
          cycle_index(cycle_index),
          phase(phase),
          is_below_sin_atol(is_below_sin_atol) {}
};

template <size_t NumModes>
struct CollectedAnticommutingOps {
    VecZ cos_inds;
    std::vector<std::vector<AnticommutingOpData<NumModes>>> ops_by_target;
};

struct LookupResponsePlan {
    std::vector<VecZ> responses;
};

inline auto split_and_exchange_cycles(CyclesType &cycles_by_target, std::vector<VecI> &phases_by_target, MPI_Comm comm)
    -> SplitCycleResult {
    const int num_ranks = mpi::size(comm);
    const int my_rank = mpi::rank(comm);

    SplitCycleResult result;
    result.cross_rank.resize(static_cast<size_t>(num_ranks));

    if (num_ranks == 1) {
        if (!cycles_by_target.empty()) {
            const auto &cycles = cycles_by_target[0];
            const auto &phases = phases_by_target[0];
            result.local_cycles.reserve(cycles.size());
            for (size_t i = 0; i < cycles.size(); ++i) {
                result.local_cycles.push_back({cycles[i].first, cycles[i].second, phases[i]});
            }
        }
        return result;
    }

    if (static_cast<size_t>(my_rank) < cycles_by_target.size()) {
        const auto &cycles = cycles_by_target[static_cast<size_t>(my_rank)];
        const auto &phases = phases_by_target[static_cast<size_t>(my_rank)];
        result.local_cycles.reserve(cycles.size());
        for (size_t i = 0; i < cycles.size(); ++i) {
            result.local_cycles.push_back({cycles[i].first, cycles[i].second, phases[i]});
        }
    }

    static thread_local std::vector<VecZ> send_buffers;
    static thread_local std::vector<VecZ> recv_buffers;
    detail::reset_nested_buffers(send_buffers, static_cast<size_t>(num_ranks));
    detail::parallel_for_indices(
        cycles_by_target.size(),
        [&my_rank, &cycles_by_target, &phases_by_target, &result](size_t rank) {
            if (static_cast<int>(rank) == my_rank) {
                return;
            }

            const auto &cycles = cycles_by_target[rank];
            const auto &phases = phases_by_target[rank];
            if (cycles.empty()) {
                return;
            }

            auto &cross_rank = result.cross_rank[rank];
            cross_rank.out_indices.resize(cycles.size());
            cross_rank.out_phases.resize(cycles.size());

            auto &buffer = send_buffers[rank];
            buffer.resize(cycles.size() * 2);

            detail::parallel_for_indices(cycles.size(), [&cross_rank, &cycles, &phases, &buffer](size_t idx) {
                cross_rank.out_indices[idx] = cycles[idx].first;
                cross_rank.out_phases[idx] = phases[idx];
                const size_t base = 2 * idx;
                buffer[base] = cycles[idx].second;
                buffer[base + 1] = static_cast<size_t>(phases[idx]);
            });
        },
        1);

    mpi::alltoallv_into(send_buffers, recv_buffers, comm);

    detail::parallel_for_indices(
        static_cast<size_t>(num_ranks),
        [&my_rank, &result](size_t rank) {
            if (static_cast<int>(rank) == my_rank) {
                return;
            }

            const auto &buffer = recv_buffers[rank];
            const size_t count = buffer.size() / 2;
            if (count == 0) {
                return;
            }

            auto &cross_rank = result.cross_rank[rank];
            cross_rank.in_indices.resize(count);
            cross_rank.in_phases.resize(count);

            detail::parallel_for_indices(count, [&cross_rank, &buffer](size_t idx) {
                cross_rank.in_indices[idx] = buffer[2 * idx];
                cross_rank.in_phases[idx] = static_cast<int>(buffer[2 * idx + 1]);
            });
        },
        1);

    return result;
}

template <size_t NumModes>
struct EvolveMajResult {
    CyclesType cycles;
    CyclesType half_cycles;
    std::vector<VecI> phases;
    std::vector<VecI> half_phases;
    std::vector<MajoranaVector<NumModes>> half_op;
    VecZ cos_inds;
    std::optional<CompressedCosineData> compressed_cos_data;

    auto resize_for_ranks(size_t num_ranks) -> void {
        cycles.resize(num_ranks);
        half_cycles.resize(num_ranks);
        phases.resize(num_ranks);
        half_phases.resize(num_ranks);
        half_op.resize(num_ranks);
    }
};

struct CutoffContext {
    bool check_atol = false;
    bool check_upper_atol = false;
    double atol_value = 0.0;
    double upper_atol_value = 0.0;
    double abs_sin_val = 1.0;
    double abs_cos_val = 1.0;
    bool use_coeff_checks = false;

    auto abs_coeff_for(size_t i, const VecD &coeffs) const -> double {
        return use_coeff_checks ? std::abs(i < coeffs.size() ? coeffs[i] : 0.0) : 0.0;
    }
    auto is_above_upper(double abs_coeff) const -> bool {
        return check_upper_atol && (abs_cos_val * abs_coeff >= upper_atol_value);
    }
    auto is_below_sin(double abs_coeff) const -> bool { return check_atol && (abs_sin_val * abs_coeff <= atol_value); }
};

template <size_t NumModes>
struct LocalScanResult {
    std::vector<std::pair<size_t, size_t>> cycles;
    std::vector<std::pair<size_t, size_t>> half_cycles;
    VecI phases;
    VecI half_phases;
    MajoranaVector<NumModes> half_op;
    VecZ cos_inds;
    CompressedCosineData compressed_cos_data;
    detail::PendingIndexRun pending_cos_run;
};

template <size_t NumModes>
inline auto reserve_evolve_target(EvolveMajResult<NumModes> &result, size_t target, size_t additional) -> void {
    result.cycles[target].reserve(result.cycles[target].size() + additional);
    result.phases[target].reserve(result.phases[target].size() + additional);
    result.half_cycles[target].reserve(result.half_cycles[target].size() + additional);
    result.half_phases[target].reserve(result.half_phases[target].size() + additional);
    result.half_op[target].reserve(result.half_op[target].size() + additional);
}

template <size_t NumModes>
inline auto append_half_term(EvolveMajResult<NumModes> &result,
                             size_t target,
                             size_t source_idx,
                             const MajoranaSet<NumModes> &new_maj,
                             int phase) -> void {
    result.half_op[target].push_back(new_maj);
    result.half_phases[target].push_back(phase);
    result.half_cycles[target].push_back({source_idx, 0});
}

template <size_t NumModes>
inline auto append_cycle_term(EvolveMajResult<NumModes> &result,
                              size_t target,
                              size_t source_idx,
                              size_t target_idx,
                              int phase) -> void {
    result.cycles[target].push_back({source_idx, target_idx});
    result.phases[target].push_back(phase);
}

inline auto append_cos_index_blocks(VecZ &dst, const std::vector<VecZ> &blocks) -> void {
    size_t total_size = dst.size();
    for (const auto &block : blocks) {
        total_size += block.size();
    }
    dst.reserve(total_size);

    for (const auto &block : blocks) {
        dst.insert(dst.end(), block.begin(), block.end());
    }
}

template <size_t NumModes>
inline auto reset_local_scan_results(tbb::enumerable_thread_specific<LocalScanResult<NumModes>> &local_results)
    -> void {
    for (auto &r : local_results) {
        r.cycles.clear();
        r.half_cycles.clear();
        r.phases.clear();
        r.half_phases.clear();
        r.half_op.clear();
        r.cos_inds.clear();
        r.compressed_cos_data.reset();
        r.pending_cos_run = {};
    }
}

template <size_t NumModes>
inline auto process_majorana_lookup_target(EvolveMajResult<NumModes> &result,
                                           std::vector<VecZ> &cos_inds_by_target,
                                           const MajoranaVector<NumModes> &source_op,
                                           const std::vector<AnticommutingOpData<NumModes>> &ops,
                                           const VecZ &responses,
                                           size_t target_rank) -> void {
    const size_t response_count = std::min(ops.size(), responses.size());
    if (response_count == 0) {
        return;
    }

    auto &cos_inds_local = cos_inds_by_target[target_rank];
    cos_inds_local.reserve(response_count);
    reserve_evolve_target(result, target_rank, response_count);

    for (size_t idx = 0; idx < response_count; ++idx) {
        const auto &op_data = ops[idx];
        const auto found_idx = responses[idx];
        if (found_idx != detail::kMissingIndex) {
            const auto &source_maj = source_op[op_data.cycle_index];
            if (asymmetric_bitset_compare<NumModes>(op_data.new_maj, source_maj)) {
                append_cycle_term(result, target_rank, op_data.cycle_index, found_idx, op_data.phase);
            }
            continue;
        }

        if (op_data.is_below_sin_atol) {
            cos_inds_local.push_back(op_data.cycle_index);
            continue;
        }

        append_half_term(result, target_rank, op_data.cycle_index, op_data.new_maj, op_data.phase);
    }
}

template <size_t NumModes>
inline auto asymmetric_bitset_compare(const MajoranaSet<NumModes> &lhs, const MajoranaSet<NumModes> &rhs) -> bool {
    for (size_t word = 0; word < detail::kWords<NumModes>; ++word) {
        const auto lhs_word = lhs.word(word);
        const auto rhs_word = rhs.word(word);
        const auto diff = lhs_word ^ rhs_word;
        if (diff != 0) {
            const auto lowest_diff_bit = std::countr_zero(diff);
            return ((lhs_word >> lowest_diff_bit) & 1ULL) != 0;
        }
    }
    return false;
}

template <size_t NumModes, bool IncludeOriginalMaj>
inline auto collect_anticommuting_ops_for_rank_core(const MajoranaVector<NumModes> &op,
                                                    const MajoranaSet<NumModes> &gen_maj,
                                                    const CutoffFn<NumModes> &cutoff_fn,
                                                    int only_rotate_len_k,
                                                    const std::optional<double> &atol,
                                                    const std::optional<double> &upper_atol,
                                                    double sin_val,
                                                    double cos_val,
                                                    const VecD &coeffs,
                                                    size_t num_ranks)
    -> std::pair<VecZ, std::vector<std::vector<AnticommutingOpData<NumModes>>>> {
    const detail::CutoffEvaluator<NumModes> cutoff_eval{cutoff_fn};

    const auto gen_maj_pop = gen_maj.count();
    const CutoffContext cutoff_ctx{.check_atol = atol.has_value(),
                                   .check_upper_atol = upper_atol.has_value(),
                                   .atol_value = atol.value_or(0.0),
                                   .upper_atol_value = upper_atol.value_or(0.0),
                                   .abs_sin_val = std::abs(sin_val),
                                   .abs_cos_val = std::abs(cos_val),
                                   .use_coeff_checks = atol.has_value() || upper_atol.has_value()};

    constexpr size_t grain_size = 512;
    const size_t n = op.size();
    const size_t safe_num_ranks = std::max<size_t>(1, num_ranks);
    const size_t max_threads = detail::effective_parallelism();

    struct LocalResult {
        VecZ cos_updates;
        std::vector<std::vector<AnticommutingOpData<NumModes>>> ops_by_target;
    };

    const size_t est_terms_per_thread = n == 0 ? 0 : n / max_threads;
    const size_t est_per_thread_per_rank =
        est_terms_per_thread == 0 ? 0 : std::max<size_t>(64, est_terms_per_thread / (safe_num_ranks * 4));
    const size_t est_cos_updates_per_thread =
        est_terms_per_thread == 0 ? 0 : std::max<size_t>(256, est_terms_per_thread / 4);
    tbb::combinable<LocalResult> local_results([safe_num_ranks, est_per_thread_per_rank, est_cos_updates_per_thread]() {
        LocalResult r;
        r.cos_updates.reserve(est_cos_updates_per_thread);
        r.ops_by_target.resize(safe_num_ranks);
        for (auto &rank_vec : r.ops_by_target) {
            rank_vec.reserve(est_per_thread_per_rank);
        }
        return r;
    });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n, grain_size),
        [&local_results,
         &op,
         &gen_maj,
         gen_maj_pop,
         only_rotate_len_k,
         &cutoff_ctx,
         &cutoff_eval,
         &coeffs,
         safe_num_ranks](const tbb::blocked_range<size_t> &range) {
            auto &local = local_results.local();
            auto &rank_ops = local.ops_by_target;
            for (size_t i = range.begin(); i < range.end(); ++i) {
                const auto &maj = op[i];
                const auto maj_pop = maj.count();
                const auto overlap = maj.count_and(gen_maj);

                if (!detail::should_process_anticommuting(maj_pop, gen_maj_pop, overlap, only_rotate_len_k)) {
                    continue;
                }

                const auto new_maj = maj ^ gen_maj;
                const auto ac = cutoff_ctx.abs_coeff_for(i, coeffs);

                if (!cutoff_ctx.is_above_upper(ac) && !cutoff_eval(new_maj)) {
                    local.cos_updates.push_back(i);
                    continue;
                }

                const auto phase = get_multiplicative_phase<NumModes>(maj, gen_maj, maj_pop, gen_maj_pop, overlap);
                const auto target_rank = find_rank<NumModes>(new_maj, safe_num_ranks);
                const auto below_sin = cutoff_ctx.is_below_sin(ac);
                if constexpr (IncludeOriginalMaj) {
                    rank_ops[target_rank].push_back({maj, new_maj, i, maj_pop, overlap, phase, below_sin});
                }
                else {
                    rank_ops[target_rank].push_back({new_maj, i, phase, below_sin});
                }
            }
        });

    std::vector<LocalResult *> locals;
    locals.reserve(max_threads);
    local_results.combine_each([&locals](LocalResult &local) { locals.push_back(&local); });

    const size_t num_locals = locals.size();
    size_t total_cos_updates = 0;
    std::vector<size_t> rank_sizes(safe_num_ranks, 0);
    std::vector<size_t> cos_offsets(num_locals);
    std::vector<std::vector<size_t>> ops_offsets(num_locals, std::vector<size_t>(safe_num_ranks));

    for (size_t t = 0; t < num_locals; ++t) {
        cos_offsets[t] = total_cos_updates;
        total_cos_updates += locals[t]->cos_updates.size();
        for (size_t r = 0; r < safe_num_ranks; ++r) {
            ops_offsets[t][r] = rank_sizes[r];
            rank_sizes[r] += locals[t]->ops_by_target[r].size();
        }
    }

    VecZ cos_updates(total_cos_updates);
    std::vector<std::vector<AnticommutingOpData<NumModes>>> ops_by_target(safe_num_ranks);
    for (size_t r = 0; r < safe_num_ranks; ++r) {
        ops_by_target[r].resize(rank_sizes[r]);
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_locals, 1),
                      [&locals, &cos_updates, &cos_offsets, &ops_by_target, &ops_offsets, safe_num_ranks](
                          const tbb::blocked_range<size_t> &range) {
                          for (size_t t = range.begin(); t < range.end(); ++t) {
                              auto *lp = locals[t];
                              if (!lp->cos_updates.empty()) {
                                  std::memcpy(cos_updates.data() + cos_offsets[t],
                                              lp->cos_updates.data(),
                                              lp->cos_updates.size() * sizeof(size_t));
                              }
                              for (size_t r = 0; r < safe_num_ranks; ++r) {
                                  const size_t cnt = lp->ops_by_target[r].size();
                                  if (cnt > 0) {
                                      std::copy_n(std::make_move_iterator(lp->ops_by_target[r].begin()),
                                                  cnt,
                                                  ops_by_target[r].begin() + ops_offsets[t][r]);
                                  }
                              }
                          }
                      });

    return {std::move(cos_updates), std::move(ops_by_target)};
}

template <size_t NumModes, bool IncludeOriginalMaj = false>
inline auto collect_anticommuting_ops_for_rank_sparse(const MajoranaVector<NumModes> &op,
                                                      const MajoranaSet<NumModes> &gen_maj,
                                                      const CutoffFn<NumModes> &cutoff_fn,
                                                      int only_rotate_len_k,
                                                      const std::optional<double> &atol,
                                                      const std::optional<double> &upper_atol,
                                                      double sin_val,
                                                      double cos_val,
                                                      const VecD &coeffs,
                                                      size_t num_ranks) -> CollectedAnticommutingOps<NumModes> {
    auto [cos_updates, ops_by_target] =
        collect_anticommuting_ops_for_rank_core<NumModes, IncludeOriginalMaj>(op,
                                                                              gen_maj,
                                                                              cutoff_fn,
                                                                              only_rotate_len_k,
                                                                              atol,
                                                                              upper_atol,
                                                                              sin_val,
                                                                              cos_val,
                                                                              coeffs,
                                                                              num_ranks);

    CollectedAnticommutingOps<NumModes> result;
    result.cos_inds = std::move(cos_updates);
    result.ops_by_target = std::move(ops_by_target);
    return result;
}

template <size_t NumModes>
inline auto exchange_majorana_lookups(const std::vector<std::vector<AnticommutingOpData<NumModes>>> &ops_by_target,
                                      const ShardedIndexMap<NumModes> &local_indexing,
                                      MPI_Comm comm) -> LookupResponsePlan {
    const int num_ranks = mpi::size(comm);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));

    LookupResponsePlan empty_result;
    if (num_ranks == 1) {
        empty_result.responses.resize(static_cast<size_t>(num_ranks));
        return empty_result;
    }

    constexpr size_t W = detail::kWords<NumModes>;

    const size_t rank_count = static_cast<size_t>(num_ranks);
    static thread_local std::vector<VecZ> send_buffers;
    static thread_local std::vector<VecZ> incoming_queries;
    static thread_local std::vector<VecZ> response_buffers;
    detail::reset_nested_buffers(send_buffers, rank_count);
    detail::reset_nested_buffers(response_buffers, rank_count);

    for (size_t target = 0; target < rank_count; ++target) {
        const auto &ops = ops_by_target[target];
        if (ops.empty() || target == my_rank) {
            continue;
        }

        auto &buffer = send_buffers[target];
        buffer.resize(ops.size() * W);

        detail::parallel_for_indices(ops.size(), [&ops, &buffer](size_t idx) {
            mpi_detail::write_majorana_words<NumModes>(ops[idx].new_maj, buffer, idx * W);
        });
    }

    VecZ local_responses;
    const auto &local_ops = ops_by_target[my_rank];
    if (!local_ops.empty()) {
        local_responses.resize(local_ops.size());
        detail::parallel_for_indices(local_ops.size(), [&local_ops, &local_indexing, &local_responses](size_t q) {
            const auto &query = local_ops[q].new_maj;
            const auto hash = MPHash<NumModes>{}(query);
            size_t found_idx = detail::kMissingIndex;
            if (const auto it = local_indexing.find_prehashed(query, hash); it != local_indexing.end_for_hash(hash)) {
                found_idx = it->second;
            }
            local_responses[q] = found_idx;
        });
    }

    mpi::alltoallv_into(send_buffers, incoming_queries, comm);

    for (size_t source = 0; source < incoming_queries.size(); ++source) {
        const auto &buffer = incoming_queries[source];
        if (buffer.empty()) {
            continue;
        }
        auto &responses = response_buffers[source];
        const size_t num_queries = buffer.size() / W;
        responses.resize(num_queries);

        detail::parallel_for_indices(num_queries, [&buffer, &local_indexing, &responses](size_t q) {
            const auto maj = mpi_detail::read_majorana_from_words<NumModes>(buffer, q * W);
            const auto hash = MPHash<NumModes>{}(maj);
            size_t found_idx = detail::kMissingIndex;
            if (const auto it = local_indexing.find_prehashed(maj, hash); it != local_indexing.end_for_hash(hash)) {
                found_idx = it->second;
            }
            responses[q] = found_idx;
        });
    }

    LookupResponsePlan result;
    mpi::alltoallv_into(response_buffers, result.responses, comm);
    result.responses[my_rank] = std::move(local_responses);
    return result;
}

} // namespace monoprop
