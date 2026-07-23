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

#include "BuildExecutionPlan.h"

#include "LayerFiltering.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop::masked_execution_plan_detail {

namespace {

struct BuilderExchangeBuffers final {
    VecI send_buffer;
    VecI recv_buffer;
};

enum class BuilderExchangeDirection {
    Outgoing,
    Incoming,
};

auto opposite_direction(BuilderExchangeDirection direction) -> BuilderExchangeDirection {
    return direction == BuilderExchangeDirection::Outgoing ? BuilderExchangeDirection::Incoming
                                                           : BuilderExchangeDirection::Outgoing;
}

auto cross_rank_size(const Layer &layer, size_t rank, BuilderExchangeDirection direction) -> size_t {
    return direction == BuilderExchangeDirection::Outgoing ? layer.cross_rank_out_size(rank)
                                                           : layer.cross_rank_in_size(rank);
}

template <typename Func>
auto for_each_remote_rank(const Layer &layer, size_t my_rank, Func &&func) -> void {
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        if (rank != my_rank) {
            func(rank);
        }
    }
}

auto has_remote_cross_rank_edges(const Layer &layer, size_t my_rank) -> bool {
    bool has_remote_edges = false;
    for_each_remote_rank(layer, my_rank, [&layer, &has_remote_edges](size_t rank) {
        if (cross_rank_size(layer, rank, BuilderExchangeDirection::Outgoing) != 0
            || cross_rank_size(layer, rank, BuilderExchangeDirection::Incoming) != 0) {
            has_remote_edges = true;
        }
    });
    return has_remote_edges;
}

auto build_builder_exchange_layout(const Layer &layer, size_t my_rank, BuilderExchangeDirection direction)
    -> BuilderExchangeLayout {
    BuilderExchangeLayout layout;
    layout.send_counts.resize(layer.cross_rank_rank_count(), 0);
    layout.send_displs.resize(layer.cross_rank_rank_count(), 0);
    layout.recv_counts.resize(layer.cross_rank_rank_count(), 0);
    layout.recv_displs.resize(layer.cross_rank_rank_count(), 0);

    size_t total_send = 0;
    size_t total_recv = 0;
    for_each_remote_rank(layer, my_rank, [&layer, &direction, &layout, &total_send, &total_recv](size_t rank) {
        const size_t send_count = cross_rank_size(layer, rank, direction);
        const size_t recv_count = cross_rank_size(layer, rank, opposite_direction(direction));
        layout.send_counts[rank] = detail::checked_mpi_int(send_count, "Mask builder send count");
        layout.recv_counts[rank] = detail::checked_mpi_int(recv_count, "Mask builder receive count");
        total_send += send_count;
        total_recv += recv_count;
    });

    size_t send_displacement = 0;
    size_t recv_displacement = 0;
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        layout.send_displs[rank] = detail::checked_mpi_int(send_displacement, "Mask builder send displacement");
        layout.recv_displs[rank] = detail::checked_mpi_int(recv_displacement, "Mask builder receive displacement");
        send_displacement += static_cast<size_t>(layout.send_counts[rank]);
        recv_displacement += static_cast<size_t>(layout.recv_counts[rank]);
    }

    layout.total_send = total_send;
    layout.total_recv = total_recv;
    return layout;
}

auto resize_builder_exchange_buffers(const BuilderExchangeLayout &layout, BuilderExchangeBuffers &buffers) -> void {
    buffers.send_buffer.resize(layout.total_send);
    buffers.recv_buffer.resize(layout.total_recv);
}

auto pack_source_keep_flags(const Layer &layer,
                            const std::vector<char> &nodes_to_keep,
                            const BuilderExchangeLayout &layout,
                            size_t my_rank,
                            VecI &send_buffer) -> void {
    for_each_remote_rank(layer, my_rank, [&layer, &layout, &send_buffer, &nodes_to_keep](size_t rank) {
        const size_t base = static_cast<size_t>(layout.send_displs[rank]);
        const size_t count = cross_rank_size(layer, rank, BuilderExchangeDirection::Outgoing);
        layer.for_each_cross_rank_out_range(
            rank,
            0,
            count,
            [&send_buffer, &base, &nodes_to_keep](size_t logical_idx, size_t src_idx, int) {
                send_buffer[base + logical_idx] = (src_idx < nodes_to_keep.size() && nodes_to_keep[src_idx]) ? 1 : 0;
            });
    });
}

auto merge_remote_selected_sources(std::vector<char> &nodes_to_keep,
                                   const Layer &layer,
                                   const BuilderExchangeLayout &layout,
                                   const VecI &recv_buffer,
                                   size_t my_rank) -> bool {
    bool changed = false;
    for_each_remote_rank(layer, my_rank, [&layer, &layout, &recv_buffer, &nodes_to_keep, &changed](size_t rank) {
        const size_t base = static_cast<size_t>(layout.recv_displs[rank]);
        const size_t count = cross_rank_size(layer, rank, BuilderExchangeDirection::Outgoing);
        layer.for_each_cross_rank_out_range(
            rank,
            0,
            count,
            [&recv_buffer, &base, &nodes_to_keep, &changed](size_t logical_idx, size_t src_idx, int) {
                if (recv_buffer[base + logical_idx] == 0) {
                    return;
                }

                if (src_idx < nodes_to_keep.size() && nodes_to_keep[src_idx] == 0) {
                    nodes_to_keep[src_idx] = 1;
                    changed = true;
                }
            });
    });
    return changed;
}

auto execute_builder_exchange(const BuilderExchangeLayout &layout, BuilderExchangeBuffers &buffers, MPI_Comm comm)
    -> void {
    if (layout.total_send == 0 && layout.total_recv == 0) {
        return;
    }

#ifdef monoprop_ENABLE_MPI
    MPI_Alltoallv(buffers.send_buffer.data(),
                  layout.send_counts.data(),
                  layout.send_displs.data(),
                  MPI_INT,
                  buffers.recv_buffer.data(),
                  layout.recv_counts.data(),
                  layout.recv_displs.data(),
                  MPI_INT,
                  comm);
#else
    (void)comm;
    buffers.recv_buffer = buffers.send_buffer;
#endif
}

auto reserve_execution_plan_storage(detail::ExecutionPlanStorage &storage, const MPGraph &graph) -> void {
    storage.cos_data_blocks.reserve(graph.layers());
    storage.local_cycle_position_blocks.reserve(graph.layers());
    storage.cross_rank_out_position_blocks.reserve(graph.layers());
    storage.cross_rank_in_position_blocks.reserve(graph.layers());
}

auto make_layer_execution_plan(const Layer &layer,
                               const std::shared_ptr<detail::ExecutionPlanStorage> &execution_storage,
                               LayerPlanFilterResult filtered) -> LayerExecutionPlan {
    if (filtered.preserves_cosine_data && filtered.preserves_local_cycles && filtered.preserves_cross_rank) {
        return LayerExecutionPlan{layer.shared_storage()};
    }

    size_t cos_data_index = 0;
    if (!filtered.preserves_cosine_data) {
        cos_data_index = execution_storage->cos_data_blocks.size();
        detail::shrink_compressed_cosine_data(filtered.masked_cos_data);
        execution_storage->cos_data_blocks.push_back(std::move(filtered.masked_cos_data));
    }

    size_t local_cycle_position_block_index = 0;
    if (!filtered.preserves_local_cycles) {
        local_cycle_position_block_index = execution_storage->local_cycle_position_blocks.size();
        detail::shrink_compressed_position_data(filtered.local_cycle_positions);
        execution_storage->local_cycle_position_blocks.push_back(std::move(filtered.local_cycle_positions));
    }

    size_t cross_rank_position_block_index = 0;
    if (!filtered.preserves_cross_rank) {
        cross_rank_position_block_index = execution_storage->cross_rank_out_position_blocks.size();
        detail::shrink_compressed_position_data(filtered.cross_rank_out_positions);
        detail::shrink_compressed_position_data(filtered.cross_rank_in_positions);
        execution_storage->cross_rank_out_position_blocks.push_back(std::move(filtered.cross_rank_out_positions));
        execution_storage->cross_rank_in_position_blocks.push_back(std::move(filtered.cross_rank_in_positions));
    }
    else {
        filtered.cross_rank_ranges.clear();
    }

    return LayerExecutionPlan{layer.shared_storage(),
                              execution_storage,
                              filtered.preserves_cosine_data,
                              cos_data_index,
                              filtered.preserves_local_cycles,
                              local_cycle_position_block_index,
                              filtered.preserves_cross_rank,
                              cross_rank_position_block_index,
                              std::move(filtered.cross_rank_ranges)};
}

} // namespace

auto build_masked_execution_plan(const VecZ &nonzero_inds,
                                 size_t local_index_count,
                                 const MPGraph &graph,
                                 bool schrodinger,
                                 MPI_Comm comm) -> MPExecutionPlan {
    const size_t num_layers = graph.layers();
    const int num_ranks = mpi::size(comm);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));

    std::vector<char> nodes_to_keep(local_index_count, 0);
    for (auto idx : nonzero_inds) {
        if (idx < nodes_to_keep.size()) {
            nodes_to_keep[idx] = 1;
        }
    }

    auto execution_storage = std::make_shared<detail::ExecutionPlanStorage>();
    reserve_execution_plan_storage(*execution_storage, graph);

    std::vector<LayerExecutionPlan> plan_layers(num_layers);
    BuilderExchangeBuffers source_keep_buffers;
    BuilderExchangeBuffers selection_buffers;

    for (size_t iter = 0; iter < num_layers; ++iter) {
        const size_t layer_idx = schrodinger ? iter : (num_layers - 1 - iter);
        const auto &layer = graph.get_layer(layer_idx);
        const bool has_remote_cross_rank = num_ranks > 1 && has_remote_cross_rank_edges(layer, my_rank);
        BuilderExchangeLayout source_keep_layout;
        BuilderExchangeLayout selection_layout;

        if (has_remote_cross_rank) {
            source_keep_layout = build_builder_exchange_layout(layer, my_rank, BuilderExchangeDirection::Outgoing);
            selection_layout = build_builder_exchange_layout(layer, my_rank, BuilderExchangeDirection::Incoming);
            resize_builder_exchange_buffers(source_keep_layout, source_keep_buffers);
            resize_builder_exchange_buffers(selection_layout, selection_buffers);
            pack_source_keep_flags(layer, nodes_to_keep, source_keep_layout, my_rank, source_keep_buffers.send_buffer);
            execute_builder_exchange(source_keep_layout, source_keep_buffers, comm);
        }
        else {
            source_keep_buffers.recv_buffer.clear();
            selection_buffers.send_buffer.clear();
            selection_buffers.recv_buffer.clear();
        }

        auto filtered = filter_layer_execution_plan(layer,
                                                    nodes_to_keep,
                                                    has_remote_cross_rank,
                                                    source_keep_layout,
                                                    source_keep_buffers.recv_buffer,
                                                    my_rank,
                                                    has_remote_cross_rank ? &selection_layout : nullptr,
                                                    has_remote_cross_rank ? &selection_buffers.send_buffer : nullptr);

        if (has_remote_cross_rank) {
            execute_builder_exchange(selection_layout, selection_buffers, comm);
            const bool changed = merge_remote_selected_sources(nodes_to_keep,
                                                               layer,
                                                               selection_layout,
                                                               selection_buffers.recv_buffer,
                                                               my_rank);
            if (changed) {
                filtered = filter_layer_execution_plan(layer,
                                                       nodes_to_keep,
                                                       has_remote_cross_rank,
                                                       source_keep_layout,
                                                       source_keep_buffers.recv_buffer,
                                                       my_rank,
                                                       &selection_layout,
                                                       &selection_buffers.send_buffer);
            }
        }

        plan_layers[layer_idx] = make_layer_execution_plan(layer, execution_storage, std::move(filtered));
    }

    execution_storage->cos_data_blocks.shrink_to_fit();
    execution_storage->local_cycle_position_blocks.shrink_to_fit();
    execution_storage->cross_rank_out_position_blocks.shrink_to_fit();
    execution_storage->cross_rank_in_position_blocks.shrink_to_fit();

    return MPExecutionPlan(graph.is_schrodinger(), std::move(plan_layers));
}

} // namespace monoprop::masked_execution_plan_detail

namespace monoprop {

auto get_masked_execution_plan(const VecD &state,
                               const VecD &op,
                               double threshold,
                               const MPGraph &graph,
                               bool schrodinger,
                               MPI_Comm comm) -> MPExecutionPlan {
    const auto &source = schrodinger ? op : state;
    VecZ nonzero_inds;
    nonzero_inds.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        if (std::abs(source[i]) > threshold) {
            nonzero_inds.push_back(i);
        }
    }

    return masked_execution_plan_detail::build_masked_execution_plan(nonzero_inds,
                                                                     source.size(),
                                                                     graph,
                                                                     schrodinger,
                                                                     comm);
}

} // namespace monoprop
