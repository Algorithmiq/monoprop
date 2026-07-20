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

#include <cstddef>

#include <monoprop/TypeAliases.h>

namespace monoprop {

template <size_t NumModes>
auto evolve_maj_single_rank(const MPOperator<NumModes> &local_mp_op,
                            const MajoranaSet<NumModes> &gen_maj,
                            const CutoffFn<NumModes> &cutoff_fn,
                            const std::optional<double> &atol,
                            std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                            const std::optional<double> &upper_atol,
                            const std::optional<double> &param,
                            int only_rotate_len_k,
                            MPI_Comm comm,
                            size_t num_ranks) -> EvolveMajResult<NumModes> {
    const detail::CutoffEvaluator<NumModes> cutoff_eval{cutoff_fn};

    const auto gen_maj_pop = gen_maj.count();

    const auto check_atol = atol.has_value() && local_coeffs.has_value() && param.has_value();
    const auto check_upper_atol = upper_atol.has_value() && local_coeffs.has_value();
    const auto sin_val = param.has_value() ? std::sin(2 * param.value()) : 1.0;
    const auto cos_val = param.has_value() ? std::cos(2 * param.value()) : 1.0;
    const CutoffContext cutoff_ctx{.check_atol = check_atol,
                                   .check_upper_atol = check_upper_atol,
                                   .atol_value = atol.value_or(0.0),
                                   .upper_atol_value = upper_atol.value_or(0.0),
                                   .abs_sin_val = std::abs(sin_val),
                                   .abs_cos_val = std::abs(cos_val),
                                   .use_coeff_checks = check_atol || check_upper_atol};

    if (num_ranks == 1) {
        const auto &op = local_mp_op.op;
        const auto &indexing = local_mp_op.indexing;
        const auto &coeffs = local_coeffs ? local_coeffs->get() : detail::empty_coeffs();
        const size_t n = op.size();

        const size_t num_threads = detail::effective_parallelism();
        const size_t grain_size = std::max<size_t>(64, n / (num_threads * 16));
        const size_t estimated_local_cos_terms =
            n == 0 ? 0 : std::max<size_t>(64, n / std::max<size_t>(size_t{1}, num_threads));

        static tbb::enumerable_thread_specific<LocalScanResult<NumModes>> local_results;
        reset_local_scan_results(local_results);

        {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, grain_size),
                [&check_upper_atol,
                 &estimated_local_cos_terms,
                 &op,
                 &gen_maj,
                 &gen_maj_pop,
                 &cutoff_ctx,
                 &cutoff_eval,
                 &indexing,
                 &coeffs,
                 &only_rotate_len_k](const tbb::blocked_range<size_t> &range) {
                    auto &result = local_results.local();
                    if (!check_upper_atol && result.compressed_cos_data.chunk_bases.capacity() == 0
                        && result.compressed_cos_data.span_offsets.capacity() == 0) {
                        detail::reserve_compressed_cosine_data(result.compressed_cos_data, estimated_local_cos_terms);
                    }

                    auto append_cos_update = [&check_upper_atol, &result](size_t idx) {
                        if (check_upper_atol) {
                            result.cos_inds.push_back(idx);
                            return;
                        }
                        detail::append_cosine_index(result.compressed_cos_data, idx, result.pending_cos_run);
                        ++result.compressed_cos_data.total_count;
                    };

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
                            append_cos_update(i);
                            continue;
                        }

                        const auto hash = MPHash<NumModes>{}(new_maj);
                        const auto found_it = indexing.find_prehashed(new_maj, hash);
                        if (found_it == indexing.end_for_hash(hash)) {
                            if (cutoff_ctx.is_below_sin(ac)) {
                                append_cos_update(i);
                                continue;
                            }
                            const auto phase =
                                get_multiplicative_phase<NumModes>(maj, gen_maj, maj_pop, gen_maj_pop, overlap);
                            result.half_op.push_back(new_maj);
                            result.half_phases.push_back(phase);
                            result.half_cycles.push_back({i, 0});
                        }
                        else if (asymmetric_bitset_compare<NumModes>(new_maj, maj)) {
                            const auto phase =
                                get_multiplicative_phase<NumModes>(maj, gen_maj, maj_pop, gen_maj_pop, overlap);
                            result.phases.push_back(phase);
                            result.cycles.push_back({i, found_it->second});
                        }
                    }
                });
        }

        EvolveMajResult<NumModes> result;
        result.resize_for_ranks(1);

        {
            size_t total_cycles = 0, total_half_cycles = 0, total_cos_inds = 0;
            std::vector<LocalScanResult<NumModes> *> locals;
            locals.reserve(num_threads);
            local_results.combine_each([&check_upper_atol, &locals, &total_cycles, &total_half_cycles, &total_cos_inds](
                                           LocalScanResult<NumModes> &local) {
                if (!check_upper_atol) {
                    detail::finish_pending_cosine_run(local.compressed_cos_data, local.pending_cos_run);
                }
                locals.push_back(&local);
                total_cycles += local.cycles.size();
                total_half_cycles += local.half_cycles.size();
                total_cos_inds += check_upper_atol ? local.cos_inds.size() : local.compressed_cos_data.total_count;
            });

            const size_t num_locals = locals.size();
            std::vector<size_t> cyc_off(num_locals), hc_off(num_locals);
            {
                size_t co = 0;
                size_t hco = 0;
                for (size_t t = 0; t < num_locals; ++t) {
                    cyc_off[t] = co;
                    hc_off[t] = hco;
                    co += locals[t]->cycles.size();
                    hco += locals[t]->half_cycles.size();
                }
            }

            struct PersistentBufs {
                std::vector<std::pair<size_t, size_t>> cycles, half_cycles;
                VecI phases, half_phases;
                MajoranaVector<NumModes> half_op;
                VecZ cos_inds;
            };
            static PersistentBufs bufs[2];
            static size_t buf_idx = 0;
            auto &b = bufs[buf_idx ^= 1];

            auto ensure = [](auto &v, size_t n) {
                if (v.capacity() < n) {
                    v.reserve(std::max(n, v.capacity() * 2));
                }
                v.resize(n);
            };
            ensure(b.cycles, total_cycles);
            ensure(b.phases, total_cycles);
            ensure(b.half_cycles, total_half_cycles);
            ensure(b.half_phases, total_half_cycles);
            ensure(b.half_op, total_half_cycles);
            const bool need_raw_cos_inds = check_upper_atol;
            if (need_raw_cos_inds) {
                ensure(b.cos_inds, total_cos_inds);
            }

            tbb::parallel_for(tbb::blocked_range<size_t>(0, num_locals, 1),
                              [&locals, &b, &cyc_off, &hc_off](const tbb::blocked_range<size_t> &range) {
                                  for (size_t t = range.begin(); t < range.end(); ++t) {
                                      auto *lp = locals[t];
                                      const size_t nc = lp->cycles.size();
                                      const size_t nhc = lp->half_cycles.size();

                                      if (nc > 0) {
                                          std::copy_n(lp->cycles.data(), nc, b.cycles.data() + cyc_off[t]);
                                          std::copy_n(lp->phases.data(), nc, b.phases.data() + cyc_off[t]);
                                      }
                                      if (nhc > 0) {
                                          std::copy_n(lp->half_cycles.data(), nhc, b.half_cycles.data() + hc_off[t]);
                                          std::copy_n(lp->half_phases.data(), nhc, b.half_phases.data() + hc_off[t]);
                                          std::copy_n(lp->half_op.data(), nhc, b.half_op.data() + hc_off[t]);
                                      }
                                  }
                              });

            result.cycles[0].swap(b.cycles);
            result.phases[0].swap(b.phases);
            result.half_cycles[0].swap(b.half_cycles);
            result.half_phases[0].swap(b.half_phases);
            result.half_op[0].swap(b.half_op);
            if (need_raw_cos_inds) {
                std::vector<size_t> ci_off(num_locals);
                size_t cio = 0;
                for (size_t t = 0; t < num_locals; ++t) {
                    ci_off[t] = cio;
                    cio += locals[t]->cos_inds.size();
                }

                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, num_locals, 1),
                    [&locals, &b, &ci_off](const tbb::blocked_range<size_t> &range) {
                        for (size_t t = range.begin(); t < range.end(); ++t) {
                            auto *lp = locals[t];
                            const size_t nci = lp->cos_inds.size();
                            if (nci > 0) {
                                std::memcpy(b.cos_inds.data() + ci_off[t], lp->cos_inds.data(), nci * sizeof(size_t));
                            }
                        }
                    });

                result.cos_inds.swap(b.cos_inds);
            }
            else {
                CompressedCosineData compressed_cos_data;
                detail::reserve_compressed_cosine_data(compressed_cos_data, total_cos_inds);
                for (const auto *local : locals) {
                    detail::append_compressed_cosine_data(compressed_cos_data, local->compressed_cos_data);
                }
                result.compressed_cos_data = std::move(compressed_cos_data);
            }
        }

        return result;
    }

    const auto &coeffs = local_coeffs ? local_coeffs->get() : detail::empty_coeffs();
    auto collected_ops =
        collect_anticommuting_ops_for_rank_sparse<NumModes>(local_mp_op.op,
                                                            gen_maj,
                                                            cutoff_fn,
                                                            only_rotate_len_k,
                                                            check_atol ? atol : std::optional<double>{},
                                                            check_upper_atol ? upper_atol : std::optional<double>{},
                                                            sin_val,
                                                            cos_val,
                                                            coeffs,
                                                            num_ranks);
    auto &anticommuting_ops = collected_ops.ops_by_target;

    EvolveMajResult<NumModes> result;
    result.resize_for_ranks(num_ranks);
    result.cos_inds = std::move(collected_ops.cos_inds);
    std::vector<VecZ> cos_inds_by_target(num_ranks);

    auto lookup_responses = exchange_majorana_lookups<NumModes>(anticommuting_ops, local_mp_op.indexing, comm);

    detail::parallel_for_indices(
        num_ranks,
        [&result, &cos_inds_by_target, &local_mp_op, &anticommuting_ops, &lookup_responses](size_t target_rank) {
            process_majorana_lookup_target(result,
                                           cos_inds_by_target,
                                           local_mp_op.op,
                                           anticommuting_ops[target_rank],
                                           lookup_responses.responses[target_rank],
                                           target_rank);
        },
        1);

    append_cos_index_blocks(result.cos_inds, cos_inds_by_target);

    return result;
}

template <size_t NumModes>
auto evolve_maj(const MPOperator<NumModes> &local_mp_op,
                const MajoranaSet<NumModes> &gen_maj,
                const CutoffFn<NumModes> &cutoff_fn,
                const std::optional<double> &atol,
                std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                const std::optional<double> &upper_atol,
                const std::optional<double> &param,
                int only_rotate_len_k,
                MPI_Comm comm) -> EvolveMajResult<NumModes> {
    const int num_ranks = mpi::size(comm);
    const int my_rank = mpi::rank(comm);
    const auto check_upper_atol = upper_atol.has_value() && local_coeffs.has_value();

    auto rank_result = evolve_maj_single_rank<NumModes>(local_mp_op,
                                                        gen_maj,
                                                        cutoff_fn,
                                                        atol,
                                                        local_coeffs,
                                                        upper_atol,
                                                        param,
                                                        only_rotate_len_k,
                                                        comm,
                                                        static_cast<size_t>(num_ranks));

    if (check_upper_atol) {
        const auto &my_cycles = rank_result.cycles[my_rank];
        VecZ cycle_targets(my_cycles.size());
        threading::parallel_for_indices(my_cycles.size(), [&cycle_targets, &my_cycles](size_t i) {
            cycle_targets[i] = my_cycles[i].second;
        });
        rank_result.cos_inds = detail::remove_excluded_indices(std::move(rank_result.cos_inds), cycle_targets);
        rank_result.compressed_cos_data.reset();
    }

    return rank_result;
}

} // namespace monoprop
