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

#include "LayerFiltering.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "monoprop/Threading.h"

namespace monoprop::masked_execution_plan_detail {

namespace {

enum class BuilderExchangeDirection {
    Outgoing,
    Incoming,
};

auto cross_rank_size(const Layer &layer, size_t rank, BuilderExchangeDirection direction) -> size_t {
    return direction == BuilderExchangeDirection::Outgoing ? layer.cross_rank_out_size(rank)
                                                           : layer.cross_rank_in_size(rank);
}

template <typename Func>
auto for_each_cross_rank_range(const Layer &layer, size_t rank, BuilderExchangeDirection direction, Func &&func)
    -> void {
    const size_t count = cross_rank_size(layer, rank, direction);
    if (direction == BuilderExchangeDirection::Outgoing) {
        layer.for_each_cross_rank_out_range(rank, 0, count, func);
        return;
    }

    layer.for_each_cross_rank_in_range(rank, 0, count, func);
}

struct CosineFilterBlock final {
    CompressedCosineData cos_data;
    bool preserves_original = true;
};

template <typename Func>
auto for_each_remote_rank(const Layer &layer, size_t my_rank, Func &&func) -> void {
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        if (rank != my_rank) {
            func(rank);
        }
    }
}

auto filter_block_size(size_t count) -> size_t {
    if (count == 0) {
        return threading::kDefaultGrainSize;
    }

    const size_t workers = threading::effective_parallelism();
    const size_t target_blocks = std::max<size_t>(1, workers * 8);
    const size_t block_size = (count + target_blocks - 1) / target_blocks;
    return std::clamp(block_size, size_t{256}, size_t{4096});
}

auto should_parallelize_cosine_filter(size_t span_count, size_t total_count) -> bool {
    return threading::effective_parallelism() > 1
           && (span_count >= (threading::kSmallLoopThreshold * 4) || total_count >= (size_t{1} << 15));
}

auto should_parallelize_local_cycle_filter(size_t cycle_count) -> bool {
    return threading::effective_parallelism() >= 8 && cycle_count >= (threading::kSmallLoopThreshold * 4);
}

auto cosine_span_is_fully_kept(const std::vector<char> &nodes_to_keep, size_t span_start, size_t span_count) -> bool {
    if (span_count == 0 || span_start >= nodes_to_keep.size()) {
        return false;
    }

    const size_t available = std::min(span_count, nodes_to_keep.size() - span_start);
    if (available != span_count) {
        return false;
    }

    const auto *keep_begin = nodes_to_keep.data() + span_start;
    return std::memchr(keep_begin, 0, available) == nullptr;
}

auto cosine_data_is_preserved(const Layer &layer, const std::vector<char> &nodes_to_keep) -> bool {
    const auto &cos_data = layer.cos_data();
    const size_t span_count = cos_data.span_count();
    bool preserved = true;
    detail::for_each_cosine_span_range(
        cos_data,
        0,
        span_count,
        [&preserved, &nodes_to_keep](size_t span_start, uint8_t span_count_local) {
            if (!preserved) {
                return;
            }

            preserved = cosine_span_is_fully_kept(nodes_to_keep, span_start, static_cast<size_t>(span_count_local));
        });
    return preserved;
}

auto append_kept_cosine_run(CompressedCosineData &filtered,
                            size_t run_start,
                            size_t run_length,
                            detail::PendingIndexRun &pending_run) -> void {
    if (run_length == 0) {
        return;
    }

    if (!pending_run.active) {
        pending_run.begin = run_start;
        pending_run.length = run_length;
        pending_run.active = true;
        return;
    }

    const bool contiguous = pending_run.length <= std::numeric_limits<size_t>::max() - pending_run.begin
                            && run_start == pending_run.begin + pending_run.length
                            && run_length <= std::numeric_limits<size_t>::max() - pending_run.length;
    if (contiguous) {
        pending_run.length += run_length;
        return;
    }

    detail::append_cosine_run(filtered, pending_run.begin, pending_run.length);
    pending_run.begin = run_start;
    pending_run.length = run_length;
    pending_run.active = true;
}

auto filter_cosine_span(CompressedCosineData &filtered,
                        bool &preserves_original,
                        const std::vector<char> &nodes_to_keep,
                        size_t span_start,
                        size_t span_count,
                        detail::PendingIndexRun &pending_run) -> void {
    if (span_count == 0 || span_start >= nodes_to_keep.size()) {
        preserves_original = false;
        return;
    }

    const size_t available = std::min(span_count, nodes_to_keep.size() - span_start);
    if (available != span_count) {
        preserves_original = false;
    }

    const auto *keep_begin = nodes_to_keep.data() + span_start;
    const auto *keep_end = keep_begin + available;
    const auto *first_kept = static_cast<const char *>(std::memchr(keep_begin, 1, available));
    if (first_kept == nullptr) {
        preserves_original = false;
        return;
    }

    const auto *first_dropped = static_cast<const char *>(std::memchr(keep_begin, 0, available));
    if (first_dropped == nullptr) {
        append_kept_cosine_run(filtered, span_start, available, pending_run);
        filtered.total_count += available;
        return;
    }

    preserves_original = false;
    const char *cursor = first_kept;
    while (cursor != nullptr && cursor < keep_end) {
        const auto *run_end = static_cast<const char *>(std::memchr(cursor, 0, static_cast<size_t>(keep_end - cursor)));
        if (run_end == nullptr) {
            run_end = keep_end;
        }

        const size_t run_start = span_start + static_cast<size_t>(cursor - keep_begin);
        const size_t run_length = static_cast<size_t>(run_end - cursor);
        append_kept_cosine_run(filtered, run_start, run_length, pending_run);
        filtered.total_count += run_length;

        if (run_end == keep_end) {
            break;
        }

        cursor = static_cast<const char *>(std::memchr(run_end, 1, static_cast<size_t>(keep_end - run_end)));
    }
}

auto filter_layer_cosine_data(const Layer &layer, const std::vector<char> &nodes_to_keep)
    -> std::pair<CompressedCosineData, bool> {
    const size_t span_count = layer.cos_span_count();
    if (span_count == 0) {
        return {{}, true};
    }
    const bool use_parallel = should_parallelize_cosine_filter(span_count, layer.num_cos_inds());
    if (!use_parallel && cosine_data_is_preserved(layer, nodes_to_keep)) {
        return {{}, true};
    }

    const auto &cos_data = layer.cos_data();
    auto scan_range = [&](CompressedCosineData &filtered, bool &preserves_original, size_t begin, size_t end) {
        detail::PendingIndexRun pending_run;
        detail::for_each_cosine_span_range(
            cos_data,
            begin,
            end,
            [&filtered, &preserves_original, &nodes_to_keep, &pending_run](size_t span_start,
                                                                           uint8_t span_count_local) {
                filter_cosine_span(filtered,
                                   preserves_original,
                                   nodes_to_keep,
                                   span_start,
                                   static_cast<size_t>(span_count_local),
                                   pending_run);
            });

        detail::finish_pending_cosine_run(filtered, pending_run);
    };

    if (!use_parallel) {
        CompressedCosineData filtered;
        detail::reserve_compressed_cosine_data(filtered, layer.num_cos_inds());
        bool preserves_original = true;
        scan_range(filtered, preserves_original, 0, span_count);
        if (preserves_original) {
            return {{}, true};
        }
        return {std::move(filtered), preserves_original};
    }

    const size_t block_size = filter_block_size(span_count);
    const size_t block_count = (span_count + block_size - 1) / block_size;
    std::vector<CosineFilterBlock> blocks(block_count);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, block_count, 1),
        [&blocks, block_size, span_count, &layer, &scan_range, block_count](const tbb::blocked_range<size_t> &range) {
            for (auto block_idx = range.begin(); block_idx < range.end(); ++block_idx) {
                auto &block = blocks[block_idx];
                const size_t begin = block_idx * block_size;
                const size_t end = std::min(span_count, begin + block_size);
                const size_t estimated_count =
                    std::max<size_t>(16, (layer.num_cos_inds() + block_count - 1) / block_count);
                detail::reserve_compressed_cosine_data(block.cos_data, estimated_count);
                scan_range(block.cos_data, block.preserves_original, begin, end);
            }
        });

    CompressedCosineData filtered;
    detail::reserve_compressed_cosine_data(filtered, layer.num_cos_inds());
    bool preserves_original = true;
    for (auto &block : blocks) {
        preserves_original = preserves_original && block.preserves_original;
        detail::append_compressed_cosine_data(filtered, block.cos_data);
    }

    if (preserves_original) {
        return {{}, true};
    }

    return {std::move(filtered), preserves_original};
}

auto filter_local_cycles(const Layer &layer, std::vector<char> &nodes_to_keep)
    -> std::pair<CompressedPositionData, bool> {
    const size_t cycle_count = layer.local_cycle_count();
    if (cycle_count == 0) {
        return {{}, true};
    }

    std::vector<unsigned char> keep_flags(cycle_count, 0);
    auto mark_cycle = [&](size_t cycle_idx, size_t src, size_t tgt) {
        const bool keep_src = src < nodes_to_keep.size() && nodes_to_keep[src];
        const bool keep_tgt = tgt < nodes_to_keep.size() && nodes_to_keep[tgt];

        if (!(keep_src || keep_tgt)) {
            return;
        }

        keep_flags[cycle_idx] = 1;
        if (!keep_src && src < nodes_to_keep.size()) {
            nodes_to_keep[src] = 1;
        }
        if (!keep_tgt && tgt < nodes_to_keep.size()) {
            nodes_to_keep[tgt] = 1;
        }
    };

    auto scan_cycle_range = [&](size_t begin, size_t end) {
        layer.for_each_local_cycle_range(begin, end, [&mark_cycle](size_t logical_idx, size_t src, size_t tgt, int) {
            mark_cycle(logical_idx, src, tgt);
        });
    };

    if (should_parallelize_local_cycle_filter(cycle_count)) {
        threading::parallel_for_ranges(cycle_count,
                                       [&scan_cycle_range](size_t begin, size_t end) { scan_cycle_range(begin, end); });
    }
    else {
        scan_cycle_range(0, cycle_count);
    }

    const bool preserves_original =
        std::all_of(keep_flags.begin(), keep_flags.end(), [](unsigned char keep_flag) { return keep_flag != 0; });

    if (preserves_original) {
        return {{}, true};
    }

    CompressedPositionData positions;
    detail::reserve_compressed_position_data(positions, cycle_count);
    detail::PendingIndexRun pending_run;
    for (size_t cycle_idx = 0; cycle_idx < cycle_count; ++cycle_idx) {
        if (keep_flags[cycle_idx] != 0) {
            detail::append_position_index(positions, cycle_idx, pending_run);
        }
    }

    detail::finish_pending_position_run(positions, pending_run);

    return {std::move(positions), preserves_original};
}

struct PositionSelectionAppender final {
    explicit PositionSelectionAppender(CompressedPositionData &positions) : positions_(&positions) {}

    auto append(size_t position) -> void { detail::append_position_index(*positions_, position, pending_run_); }

    auto finish() -> void { detail::finish_pending_position_run(*positions_, pending_run_); }

private:
    CompressedPositionData *positions_ = nullptr;
    detail::PendingIndexRun pending_run_;
};

struct CrossRankFilterInputs final {
    std::vector<char> &nodes_to_keep;
    const BuilderExchangeLayout &source_keep_layout;
    const VecI &remote_src_keep;
    size_t my_rank;
    const BuilderExchangeLayout *selection_layout = nullptr;
    VecI *selected_incoming_flags = nullptr;
};

auto reserve_cross_rank_filter_positions(LayerPlanFilterResult &result,
                                         const Layer &layer,
                                         const CrossRankFilterInputs &inputs) -> void {
    result.cross_rank_ranges.resize(layer.cross_rank_rank_count());

    size_t total_cross_rank_out = 0;
    size_t total_cross_rank_in = 0;
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        total_cross_rank_out += cross_rank_size(layer, rank, BuilderExchangeDirection::Outgoing);
        total_cross_rank_in += cross_rank_size(layer, rank, BuilderExchangeDirection::Incoming);
    }

    detail::reserve_compressed_position_data(result.cross_rank_out_positions, total_cross_rank_out);
    detail::reserve_compressed_position_data(result.cross_rank_in_positions, total_cross_rank_in);

    if (inputs.selected_incoming_flags != nullptr && inputs.selection_layout != nullptr) {
        inputs.selected_incoming_flags->assign(inputs.selection_layout->total_send, 0);
    }
}

auto filter_local_layer_components(LayerPlanFilterResult &result, const Layer &layer, std::vector<char> &nodes_to_keep)
    -> void {
    auto [masked_cos_data, preserves_cosine_data] = filter_layer_cosine_data(layer, nodes_to_keep);
    result.preserves_cosine_data = preserves_cosine_data;
    if (!preserves_cosine_data) {
        result.masked_cos_data = std::move(masked_cos_data);
    }

    auto [local_cycle_positions, preserves_local_cycles] = filter_local_cycles(layer, nodes_to_keep);
    result.preserves_local_cycles = preserves_local_cycles;
    if (!preserves_local_cycles) {
        result.local_cycle_positions = std::move(local_cycle_positions);
    }
}

auto filter_cross_rank_for_rank(LayerPlanFilterResult &result,
                                const Layer &layer,
                                size_t rank,
                                const CrossRankFilterInputs &inputs) -> void {
    auto &range = result.cross_rank_ranges[rank];
    range.out_offset = result.cross_rank_out_positions.total_count;
    range.in_offset = result.cross_rank_in_positions.total_count;

    const size_t remote_base = static_cast<size_t>(inputs.source_keep_layout.recv_displs[rank]);
    const bool should_notify_selection =
        inputs.selected_incoming_flags != nullptr && inputs.selection_layout != nullptr;
    const size_t notify_base =
        should_notify_selection ? static_cast<size_t>(inputs.selection_layout->send_displs[rank]) : 0;

    PositionSelectionAppender out_positions(result.cross_rank_out_positions);
    for_each_cross_rank_range(layer,
                              rank,
                              BuilderExchangeDirection::Outgoing,
                              [&inputs, &out_positions, &result](size_t logical_idx, size_t src_idx, int) {
                                  const bool keep_src =
                                      src_idx < inputs.nodes_to_keep.size() && inputs.nodes_to_keep[src_idx];
                                  if (keep_src) {
                                      out_positions.append(logical_idx);
                                  }
                                  else {
                                      result.preserves_cross_rank = false;
                                  }
                              });
    out_positions.finish();

    PositionSelectionAppender in_positions(result.cross_rank_in_positions);
    for_each_cross_rank_range(
        layer,
        rank,
        BuilderExchangeDirection::Incoming,
        [&inputs, &in_positions, &result, remote_base, should_notify_selection, notify_base](size_t logical_idx,
                                                                                             size_t tgt_idx,
                                                                                             int) {
            bool keep_tgt = tgt_idx < inputs.nodes_to_keep.size() && inputs.nodes_to_keep[tgt_idx];
            const bool keep_src = remote_base + logical_idx < inputs.remote_src_keep.size()
                                      ? inputs.remote_src_keep[remote_base + logical_idx] != 0
                                      : false;

            if (keep_src || keep_tgt) {
                in_positions.append(logical_idx);
                if (should_notify_selection) {
                    (*inputs.selected_incoming_flags)[notify_base + logical_idx] = 1;
                }

                if (!keep_tgt && tgt_idx < inputs.nodes_to_keep.size()) {
                    inputs.nodes_to_keep[tgt_idx] = 1;
                }
            }
            else {
                result.preserves_cross_rank = false;
            }
        });
    in_positions.finish();

    range.out_count = result.cross_rank_out_positions.total_count - range.out_offset;
    range.in_count = result.cross_rank_in_positions.total_count - range.in_offset;
}

auto filter_cross_rank_components(LayerPlanFilterResult &result,
                                  const Layer &layer,
                                  const CrossRankFilterInputs &inputs) -> void {
    for_each_remote_rank(layer, inputs.my_rank, [&result, &layer, &inputs](size_t rank) {
        filter_cross_rank_for_rank(result, layer, rank, inputs);
    });

    if (result.preserves_cross_rank) {
        result.cross_rank_out_positions.reset();
        result.cross_rank_in_positions.reset();
        result.cross_rank_ranges.clear();
    }
}

} // namespace

auto filter_layer_execution_plan(const Layer &layer,
                                 std::vector<char> &nodes_to_keep,
                                 bool has_remote_cross_rank,
                                 const BuilderExchangeLayout &source_keep_layout,
                                 const VecI &remote_src_keep,
                                 size_t my_rank,
                                 const BuilderExchangeLayout *selection_layout,
                                 VecI *selected_incoming_flags) -> LayerPlanFilterResult {
    LayerPlanFilterResult result;
    const CrossRankFilterInputs cross_rank_inputs{
        nodes_to_keep,
        source_keep_layout,
        remote_src_keep,
        my_rank,
        selection_layout,
        selected_incoming_flags,
    };

    filter_local_layer_components(result, layer, nodes_to_keep);
    if (has_remote_cross_rank) {
        reserve_cross_rank_filter_positions(result, layer, cross_rank_inputs);
        filter_cross_rank_components(result, layer, cross_rank_inputs);
    }

    return result;
}

} // namespace monoprop::masked_execution_plan_detail
