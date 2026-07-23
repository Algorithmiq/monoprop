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

#include "monoprop/Evolution.h"

#include <cmath>
#include <utility>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop {
namespace {

struct DerivativeContrib {
    double cos = 0.0;
    double sin = 0.0;
};

struct DerivativeCosineCacheContext {
    const double *data = nullptr;
    size_t num_layers = 0;
    size_t layer_stride = 0;
};

auto derivative_cosine_cache_context() -> DerivativeCosineCacheContext & {
    static thread_local DerivativeCosineCacheContext ctx;
    return ctx;
}

struct TrigValues {
    double cos_val;
    double sin_val;
    double sec_val;
    double der_cos_val;
    double der_sin_val;

    explicit TrigValues(double param, double gen_coeff = 1.0) {
        const double g = 2.0 * gen_coeff;
        cos_val = std::cos(g * param);
        sin_val = std::sin(g * param);
        sec_val = 1.0 / cos_val;
        der_cos_val = -g * sin_val;
        der_sin_val = g * cos_val;
    }
};

auto combine_derivative_contrib(const DerivativeContrib &a, const DerivativeContrib &b) -> DerivativeContrib {
    return {.cos = a.cos + b.cos, .sin = a.sin + b.sin};
}

auto accumulate_cosine_span(double *state_ptr,
                            double *op_ptr,
                            const double *cached_op_ptr,
                            size_t count,
                            double cos_val,
                            double sec_val) -> double {
    double local = 0.0;
    for (size_t idx = 0; idx < count; ++idx) {
        const double cached_state = state_ptr[idx];
        if (cached_op_ptr != nullptr) {
            const double cached_op = cached_op_ptr[idx];
            local += cached_state * cached_op;
            op_ptr[idx] = cached_op;
        }
        else {
            op_ptr[idx] *= sec_val;
            local += cached_state * op_ptr[idx];
        }
        state_ptr[idx] = cached_state * cos_val;
    }
    return local;
}

auto scale_cosine_span(double *coeffs, size_t count, double cos_val) -> void {
    for (size_t idx = 0; idx < count; ++idx) {
        coeffs[idx] *= cos_val;
    }
}

auto accumulate_cosine_span_range(double *state_data,
                                  double *operator_data,
                                  const double *cached_op_data,
                                  const CompressedCosineData &cos_data,
                                  size_t begin,
                                  size_t end,
                                  double cos_val,
                                  double sec_val) -> double {
    double total = 0.0;

    detail::for_each_cosine_span_range(
        cos_data,
        begin,
        end,
        [&total, state_data, operator_data, cached_op_data, cos_val, sec_val](size_t start, uint8_t count) {
            total += accumulate_cosine_span(state_data + start,
                                            operator_data + start,
                                            cached_op_data != nullptr ? (cached_op_data + start) : nullptr,
                                            count,
                                            cos_val,
                                            sec_val);
        });

    return total;
}

auto scale_cosine_span_range(double *coeff_data,
                             const CompressedCosineData &cos_data,
                             size_t begin,
                             size_t end,
                             double cos_val) -> void {
    detail::for_each_cosine_span_range(cos_data, begin, end, [coeff_data, cos_val](size_t start, uint8_t count) {
        scale_cosine_span(coeff_data + start, count, cos_val);
    });
}

struct FlatExchangeBuffers {
    VecD send_buffer;
    VecD recv_buffer;
};

auto acquire_flat_exchange_buffers() -> FlatExchangeBuffers & {
    struct Scratch {
        FlatExchangeBuffers buffers;
    };
    static thread_local Scratch scratch;
    return scratch.buffers;
}

// Derive the derivative exchange layout (2x scale of evolution layout) using thread-local storage
// to avoid storing it permanently in LayerStorage.
auto acquire_derivative_layout(const LayerExchangeLayout &evol) -> LayerExchangeLayout & {
    static thread_local LayerExchangeLayout layout;
    layout.total_count = evol.total_count * 2;
    const size_t n = evol.counts.size();
    layout.counts.resize(n);
    layout.displs.resize(n);
    for (size_t i = 0; i < n; ++i) {
        layout.counts[i] = evol.counts[i] * 2;
        layout.displs[i] = evol.displs[i] * 2;
    }
    return layout;
}

template <typename Body>
auto for_each_cross_rank_range(const LayerTraversal &layer,
                               bool outgoing,
                               size_t rank,
                               size_t begin,
                               size_t end,
                               Body &&body) -> void {
    if (outgoing) {
        layer.for_each_cross_rank_out_range(rank, begin, end, std::forward<Body>(body));
        return;
    }

    layer.for_each_cross_rank_in_range(rank, begin, end, std::forward<Body>(body));
}

auto resize_flat_exchange_buffers(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers) -> void {
    buffers.send_buffer.resize(layout.total_count);
    buffers.recv_buffer.resize(layout.total_count);
}

auto active_evolution_exchange_layout(const LayerTraversal &layer, MPI_Comm comm) -> const LayerExchangeLayout * {
    if (mpi::size(comm) == 1) {
        return nullptr;
    }

    const auto &layout = layer.evolution_exchange_layout();
    return layout.total_count == 0 ? nullptr : &layout;
}

auto execute_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, MPI_Comm comm) -> void;

template <typename Pack, typename Apply>
auto with_cross_rank_exchange(const LayerExchangeLayout &layout, MPI_Comm comm, Pack &&pack, Apply &&apply) -> void {
    const int my_rank = mpi::rank(comm);

    auto &buffers = acquire_flat_exchange_buffers();
    resize_flat_exchange_buffers(layout, buffers);
    pack(my_rank, layout, buffers.send_buffer);
    execute_flat_exchange(layout, buffers, comm);
    apply(my_rank, layout, buffers.recv_buffer);
}

auto execute_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, MPI_Comm comm) -> void {
    if (layout.total_count == 0) {
        return;
    }
#ifdef monoprop_ENABLE_MPI
    MPI_Alltoallv(buffers.send_buffer.data(),
                  layout.counts.data(),
                  layout.displs.data(),
                  MPI_DOUBLE,
                  buffers.recv_buffer.data(),
                  layout.counts.data(),
                  layout.displs.data(),
                  MPI_DOUBLE,
                  comm);
#else
    (void)comm;
    buffers.recv_buffer = buffers.send_buffer;
#endif
}

// ─── Cross-rank derivative helpers ──────────────────────────────────────────

auto pack_cross_rank_derivative_payload_impl(VecD &state,
                                             VecD &op,
                                             const LayerTraversal &layer,
                                             int my_rank,
                                             const LayerExchangeLayout &layout,
                                             VecD &send_buffer) -> void {
    threading::parallel_for_cross_rank_ranges(
        layer,
        my_rank,
        true,
        [&layer, &layout, &state, &op, &send_buffer](size_t rank, size_t begin, size_t end) {
            const size_t base = static_cast<size_t>(layout.displs[rank]);
            for_each_cross_rank_range(layer,
                                      true,
                                      rank,
                                      begin,
                                      end,
                                      [base, &state, &op, &send_buffer](size_t logical_idx, size_t value_idx, int) {
                                          const size_t pair_offset = 2 * logical_idx;
                                          send_buffer[base + pair_offset] = state[value_idx];
                                          send_buffer[base + pair_offset + 1] = op[value_idx];
                                      });
        });

    threading::parallel_for_cross_rank_ranges(
        layer,
        my_rank,
        false,
        [&layer, &layout, &state, &op, &send_buffer](size_t rank, size_t begin, size_t end) {
            const size_t base = static_cast<size_t>(layout.displs[rank]) + (2 * layer.cross_rank_out_size(rank));
            for_each_cross_rank_range(layer,
                                      false,
                                      rank,
                                      begin,
                                      end,
                                      [base, &state, &op, &send_buffer](size_t logical_idx, size_t value_idx, int) {
                                          const size_t pair_offset = 2 * logical_idx;
                                          send_buffer[base + pair_offset] = state[value_idx];
                                          send_buffer[base + pair_offset + 1] = op[value_idx];
                                      });
        });
}

auto apply_cross_rank_derivative_exchange_impl(VecD &state,
                                               VecD &op,
                                               const LayerTraversal &layer,
                                               const LayerExchangeLayout &layout,
                                               const TrigValues &trig,
                                               const VecD &recv_buffer,
                                               int my_rank) -> DerivativeContrib {
    const auto out_contrib = threading::parallel_reduce_cross_rank_ranges(
        layer,
        my_rank,
        true,
        DerivativeContrib{},
        [&layer, &recv_buffer, &layout, &state, &op, &trig](size_t rank,
                                                            size_t begin,
                                                            size_t end,
                                                            DerivativeContrib local) {
            const auto *rv = recv_buffer.data() + layout.displs[rank];
            const size_t offset = 2 * layer.cross_rank_in_size(rank);
            for_each_cross_rank_range(
                layer,
                true,
                rank,
                begin,
                end,
                [rv, offset, &state, &op, &trig, &local](size_t logical_idx, size_t value_idx, int phase_int) {
                    const size_t pair_offset = 2 * logical_idx;
                    const double phase = static_cast<double>(phase_int);
                    const double ps = trig.sin_val * phase;
                    const double s_old = state[value_idx];
                    const double h_new = (op[value_idx] * trig.cos_val) + (ps * rv[offset + pair_offset + 1]);
                    local.cos += (h_new * s_old);
                    local.sin += (h_new * rv[offset + pair_offset] * phase);
                    op[value_idx] = h_new;
                    state[value_idx] = (s_old * trig.cos_val) + (ps * rv[offset + pair_offset]);
                });
            return local;
        },
        combine_derivative_contrib);

    const auto in_contrib = threading::parallel_reduce_cross_rank_ranges(
        layer,
        my_rank,
        false,
        DerivativeContrib{},
        [&layer, &recv_buffer, &layout, &state, &op, &trig](size_t rank,
                                                            size_t begin,
                                                            size_t end,
                                                            DerivativeContrib local) {
            const auto *rv = recv_buffer.data() + layout.displs[rank];
            for_each_cross_rank_range(
                layer,
                false,
                rank,
                begin,
                end,
                [rv, &state, &op, &trig, &local](size_t logical_idx, size_t value_idx, int phase_int) {
                    const size_t pair_offset = 2 * logical_idx;
                    const double phase = static_cast<double>(phase_int);
                    const double ps = trig.sin_val * phase;
                    const double s_old = state[value_idx];
                    const double h_new = (op[value_idx] * trig.cos_val) - (ps * rv[pair_offset + 1]);
                    local.cos += (h_new * s_old);
                    local.sin -= (h_new * rv[pair_offset] * phase);
                    op[value_idx] = h_new;
                    state[value_idx] = (s_old * trig.cos_val) - (ps * rv[pair_offset]);
                });
            return local;
        },
        combine_derivative_contrib);

    return {out_contrib.cos + in_contrib.cos, out_contrib.sin + in_contrib.sin};
}

// ─── Cross-rank evolution helpers ───────────────────────────────────────────

auto pack_cross_rank_evolution_payload_impl(VecD &op,
                                            const LayerTraversal &layer,
                                            int my_rank,
                                            const LayerExchangeLayout &layout,
                                            VecD &send_buffer) -> void {
    threading::parallel_for_cross_rank_ranges(
        layer,
        my_rank,
        true,
        [&layer, &layout, &op, &send_buffer](size_t rank, size_t begin, size_t end) {
            const size_t base = static_cast<size_t>(layout.displs[rank]);
            for_each_cross_rank_range(layer,
                                      true,
                                      rank,
                                      begin,
                                      end,
                                      [base, &op, &send_buffer](size_t logical_idx, size_t value_idx, int) {
                                          send_buffer[base + logical_idx] = op[value_idx];
                                      });
        });

    threading::parallel_for_cross_rank_ranges(
        layer,
        my_rank,
        false,
        [&layer, &layout, &op, &send_buffer](size_t rank, size_t begin, size_t end) {
            const size_t base = static_cast<size_t>(layout.displs[rank]) + layer.cross_rank_out_size(rank);
            for_each_cross_rank_range(layer,
                                      false,
                                      rank,
                                      begin,
                                      end,
                                      [base, &op, &send_buffer](size_t logical_idx, size_t value_idx, int) {
                                          send_buffer[base + logical_idx] = op[value_idx];
                                      });
        });
}

auto apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              const LayerExchangeLayout &layout,
                                              double cos_val,
                                              double sin_val,
                                              const VecD &recv_buffer,
                                              int my_rank) -> void {
    threading::parallel_for_cross_rank_ranges(
        layer,
        my_rank,
        true,
        [&layer, &recv_buffer, &layout, &op, cos_val, sin_val](size_t rank, size_t begin, size_t end) {
            const auto *rv = recv_buffer.data() + layout.displs[rank];
            const size_t offset = layer.cross_rank_in_size(rank);
            for_each_cross_rank_range(
                layer,
                true,
                rank,
                begin,
                end,
                [rv, offset, &op, cos_val, sin_val](size_t logical_idx, size_t value_idx, int phase_int) {
                    const double phase = static_cast<double>(phase_int);
                    op[value_idx] = (cos_val * op[value_idx]) - ((sin_val * phase) * rv[offset + logical_idx]);
                });
        });

    threading::parallel_for_cross_rank_ranges(
        layer,
        my_rank,
        false,
        [&layer, &recv_buffer, &layout, &op, cos_val, sin_val](size_t rank, size_t begin, size_t end) {
            const auto *rv = recv_buffer.data() + layout.displs[rank];
            for_each_cross_rank_range(layer,
                                      false,
                                      rank,
                                      begin,
                                      end,
                                      [rv, &op, cos_val, sin_val](size_t logical_idx, size_t value_idx, int phase_int) {
                                          const double phase = static_cast<double>(phase_int);
                                          op[value_idx] =
                                              (cos_val * op[value_idx]) + ((sin_val * phase) * rv[logical_idx]);
                                      });
        });
}

auto synchronize_cross_rank_operator_impl(VecD &op,
                                          const LayerTraversal &layer,
                                          double cos_val,
                                          double sin_val,
                                          MPI_Comm comm) -> void {
    const auto *layout = active_evolution_exchange_layout(layer, comm);
    if (layout == nullptr) {
        return;
    }

    with_cross_rank_exchange(
        *layout,
        comm,
        [&op, &layer](int my_rank, const LayerExchangeLayout &active_layout, VecD &send_buffer) {
            pack_cross_rank_evolution_payload_impl(op, layer, my_rank, active_layout, send_buffer);
        },
        [&op, &layer, cos_val, sin_val](int my_rank,
                                        const LayerExchangeLayout &active_layout,
                                        const VecD &recv_buffer) {
            apply_cross_rank_evolution_exchange_impl(op, layer, active_layout, cos_val, sin_val, recv_buffer, my_rank);
        });
}

auto accumulate_cosine_derivative(VecD &state,
                                  VecD &op,
                                  const LayerTraversal &layer,
                                  size_t layer_idx,
                                  double cos_val,
                                  double sec_val) -> double {
    const auto &cos_data = layer.cos_data();
    auto *const state_data = state.data();
    auto *const operator_data = op.data();
    const auto &cache_ctx = derivative_cosine_cache_context();
    const double *cached_layer_data = (cache_ctx.data != nullptr && layer_idx < cache_ctx.num_layers)
                                          ? (cache_ctx.data + layer_idx * cache_ctx.layer_stride)
                                          : nullptr;
    return threading::parallel_reduce_ranges(
        layer.cos_span_count(),
        0.0,
        [state_data, operator_data, cached_layer_data, &cos_data, cos_val, sec_val](size_t begin,
                                                                                    size_t end,
                                                                                    double local) {
            return local
                   + accumulate_cosine_span_range(state_data,
                                                  operator_data,
                                                  cached_layer_data,
                                                  cos_data,
                                                  begin,
                                                  end,
                                                  cos_val,
                                                  sec_val);
        },
        [](double lhs, double rhs) { return lhs + rhs; },
        threading::range_grain_size(layer.cos_span_count(), 1));
}

auto accumulate_cycle_derivative(VecD &state, VecD &op, const LayerTraversal &layer, double sin_val, double cos_val)
    -> DerivativeContrib {
    if (layer.local_cycle_count() == 0) {
        return {};
    }

    const auto contrib = threading::parallel_reduce_ranges(
        layer.local_cycle_count(),
        DerivativeContrib{},
        [&layer, &state, &op, sin_val, cos_val](size_t begin, size_t end, DerivativeContrib local) {
            layer.for_each_local_cycle_range(
                begin,
                end,
                [&state, &op, sin_val, cos_val, &local](size_t, size_t src, size_t tgt, int phase_int) {
                    const double phase = static_cast<double>(phase_int);
                    const double ps = sin_val * phase;
                    const double s0 = state[src], s1 = state[tgt];
                    const double h0 = op[src], h1 = op[tgt];
                    const double nh0 = (h0 * cos_val) + (ps * h1);
                    const double nh1 = (h1 * cos_val) - (ps * h0);
                    op[src] = nh0;
                    op[tgt] = nh1;
                    local.cos += (nh0 * s0) + (nh1 * s1);
                    local.sin += ((nh0 * s1) - (nh1 * s0)) * phase;
                    state[src] = (s0 * cos_val) + (ps * s1);
                    state[tgt] = (s1 * cos_val) - (ps * s0);
                });
            return local;
        },
        combine_derivative_contrib);

    return contrib;
}

auto apply_local_cycle_evolution(VecD &op, const LayerTraversal &layer, double cos_val, double sin_val) -> void {
    threading::parallel_for_ranges(layer.local_cycle_count(),
                                   [&layer, &op, cos_val, sin_val](size_t begin, size_t end) {
                                       layer.for_each_local_cycle_range(
                                           begin,
                                           end,
                                           [&op, cos_val, sin_val](size_t, size_t src, size_t tgt, int phase_int) {
                                               const double p = sin_val * static_cast<double>(phase_int);
                                               const double o1 = op[src], o2 = op[tgt];
                                               op[src] = (o1 * cos_val) - (p * o2);
                                               op[tgt] = (o2 * cos_val) + (p * o1);
                                           });
                                   });
}

auto accumulate_cross_rank_derivatives_impl(VecD &state,
                                            VecD &op,
                                            const LayerTraversal &layer,
                                            const TrigValues &trig,
                                            MPI_Comm comm) -> DerivativeContrib {
    const auto *evolution_layout = active_evolution_exchange_layout(layer, comm);
    if (evolution_layout == nullptr) {
        return {};
    }

    DerivativeContrib contrib{};
    auto &layout = acquire_derivative_layout(*evolution_layout);
    with_cross_rank_exchange(
        layout,
        comm,
        [&state, &op, &layer](int my_rank, const LayerExchangeLayout &active_layout, VecD &send_buffer) {
            pack_cross_rank_derivative_payload_impl(state, op, layer, my_rank, active_layout, send_buffer);
        },
        [&state, &op, &layer, &trig, &contrib](int my_rank,
                                               const LayerExchangeLayout &active_layout,
                                               const VecD &recv_buffer) {
            contrib =
                apply_cross_rank_derivative_exchange_impl(state, op, layer, active_layout, trig, recv_buffer, my_rank);
        });
    return contrib;
}

} // namespace

// ─── Public API ─────────────────────────────────────────────────────────────

// ─── Derivative & evolution dispatch ────────────────────────────────────────

template <typename GraphType>
auto state_operator_derivative_local_impl(VecD &state,
                                          VecD &op,
                                          const GraphType &graph,
                                          size_t layer_idx,
                                          double gen_coeff,
                                          double param,
                                          MPI_Comm comm) -> double {
    const TrigValues trig(param, gen_coeff);
    const auto layer = graph.get_layer_traversal(layer_idx);

    double cos_contrib = accumulate_cosine_derivative(state, op, layer, layer_idx, trig.cos_val, trig.sec_val);
    const auto cycle_contrib = accumulate_cycle_derivative(state, op, layer, trig.sin_val, trig.cos_val);
    const auto cross_rank_contrib = accumulate_cross_rank_derivatives_impl(state, op, layer, trig, comm);
    const auto derivative_contrib = combine_derivative_contrib(cycle_contrib, cross_rank_contrib);

    return ((cos_contrib + derivative_contrib.cos) * trig.der_cos_val) + (derivative_contrib.sin * trig.der_sin_val);
}

auto state_operator_derivative(VecD &state,
                               VecD &op,
                               const MPGraph &graph,
                               size_t layer_idx,
                               double gen_coeff,
                               double param,
                               MPI_Comm comm) -> double {
    return mpi::allreduce_sum(state_operator_derivative_local_impl(state, op, graph, layer_idx, gen_coeff, param, comm),
                              comm);
}

auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraph &graph,
                                     size_t layer_idx,
                                     double gen_coeff,
                                     double param,
                                     MPI_Comm comm) -> double {
    return state_operator_derivative_local_impl(state, op, graph, layer_idx, gen_coeff, param, comm);
}

auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraphView &graph,
                                     size_t layer_idx,
                                     double gen_coeff,
                                     double param,
                                     MPI_Comm comm) -> double {
    return state_operator_derivative_local_impl(state, op, graph, layer_idx, gen_coeff, param, comm);
}

auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPExecutionPlan &graph,
                                     size_t layer_idx,
                                     double gen_coeff,
                                     double param,
                                     MPI_Comm comm) -> double {
    return state_operator_derivative_local_impl(state, op, graph, layer_idx, gen_coeff, param, comm);
}

auto set_derivative_cosine_cache(const double *cache_data, size_t num_layers, size_t layer_stride) -> void {
    auto &ctx = derivative_cosine_cache_context();
    ctx.data = cache_data;
    ctx.num_layers = num_layers;
    ctx.layer_stride = layer_stride;
}

auto clear_derivative_cosine_cache() -> void {
    auto &ctx = derivative_cosine_cache_context();
    ctx = {};
}

template <typename GraphType>
auto evolve_step_impl(VecD &op, const GraphType &graph, double param, size_t layer_idx, MPI_Comm comm) -> void {
    const double cos_val = std::cos(2 * param), sin_val = std::sin(2 * param);
    const auto layer = graph.get_layer_traversal(layer_idx);

    const auto &cos_data = layer.cos_data();
    auto *const op_data = op.data();
    threading::parallel_for_ranges(
        layer.cos_span_count(),
        [op_data, &cos_data, cos_val](size_t begin, size_t end) {
            scale_cosine_span_range(op_data, cos_data, begin, end, cos_val);
        },
        threading::range_grain_size(layer.cos_span_count(), 1));
    apply_local_cycle_evolution(op, layer, cos_val, sin_val);
    synchronize_cross_rank_operator_impl(op, layer, cos_val, sin_val, comm);
}

auto evolve_step(VecD &op, const MPGraph &graph, double param, size_t layer_idx, MPI_Comm comm) -> void {
    evolve_step_impl(op, graph, param, layer_idx, comm);
}

auto evolve_step(VecD &op, const MPGraphView &graph, double param, size_t layer_idx, MPI_Comm comm) -> void {
    evolve_step_impl(op, graph, param, layer_idx, comm);
}

auto evolve_step(VecD &op, const MPExecutionPlan &graph, double param, size_t layer_idx, MPI_Comm comm) -> void {
    evolve_step_impl(op, graph, param, layer_idx, comm);
}

template <typename GraphType>
auto evolve_operator_impl(VecD coeffs, const GraphType &graph, const VecD &params, MPI_Comm comm) -> VecD {
    for (size_t i = 0; i < graph.layers(); ++i) {
        evolve_step_impl(coeffs, graph, params[i], i, comm);
    }
    return coeffs;
}

auto evolve_operator(const VecD &coeffs, const MPGraph &graph, const VecD &params, MPI_Comm comm) -> VecD {
    return evolve_operator_impl(VecD{coeffs}, graph, params, comm);
}

auto evolve_operator(VecD &&coeffs, const MPGraph &graph, const VecD &params, MPI_Comm comm) -> VecD {
    return evolve_operator_impl(std::move(coeffs), graph, params, comm);
}

auto evolve_operator(const VecD &coeffs, const MPGraphView &graph, const VecD &params, MPI_Comm comm) -> VecD {
    return evolve_operator_impl(VecD{coeffs}, graph, params, comm);
}

auto evolve_operator(VecD &&coeffs, const MPGraphView &graph, const VecD &params, MPI_Comm comm) -> VecD {
    return evolve_operator_impl(std::move(coeffs), graph, params, comm);
}

auto evolve_operator(const VecD &coeffs, const MPExecutionPlan &graph, const VecD &params, MPI_Comm comm) -> VecD {
    return evolve_operator_impl(VecD{coeffs}, graph, params, comm);
}

auto evolve_operator(VecD &&coeffs, const MPExecutionPlan &graph, const VecD &params, MPI_Comm comm) -> VecD {
    return evolve_operator_impl(std::move(coeffs), graph, params, comm);
}

} // namespace monoprop
