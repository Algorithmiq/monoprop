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
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"

namespace monoprop::detail {

inline constexpr size_t kMaxCosineSpanLength = static_cast<size_t>(std::numeric_limits<uint8_t>::max());
inline constexpr size_t kMaxPositionSpanLength = static_cast<size_t>(std::numeric_limits<uint16_t>::max());
inline constexpr size_t kEstimatedCosineRunLength = 8;
inline constexpr size_t kCosineChunkShift = 16;
inline constexpr size_t kCosineChunkSize = size_t{1} << kCosineChunkShift;

struct PendingIndexRun final {
    size_t begin = 0;
    size_t length = 0;
    bool active = false;
};

inline auto estimated_cosine_span_capacity(size_t cos_count) -> size_t {
    if (cos_count == 0) {
        return 0;
    }
    const size_t estimated = (cos_count + kEstimatedCosineRunLength - 1) / kEstimatedCosineRunLength;
    return std::min(cos_count, std::max<size_t>(estimated, 16));
}

inline auto checked_packed_index(size_t value, const char *what) -> uint32_t {
    if (value > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::overflow_error(std::format("{} {} exceeds the 32-bit packed index limit {}.",
                                              what,
                                              value,
                                              std::numeric_limits<uint32_t>::max()));
    }
    return static_cast<uint32_t>(value);
}

inline auto reserve_compressed_cosine_data(CompressedCosineData &data, size_t cos_count) -> void {
    const size_t span_capacity = estimated_cosine_span_capacity(cos_count);
    const size_t chunk_capacity =
        cos_count == 0 ? 0 : std::max<size_t>(16, 2 * ((cos_count + kCosineChunkSize - 1) / kCosineChunkSize));
    data.chunk_bases.reserve(chunk_capacity);
    data.chunk_span_starts.reserve(chunk_capacity);
    data.span_offsets.reserve(span_capacity);
    data.span_counts.reserve(span_capacity);
}

inline auto shrink_compressed_cosine_data(CompressedCosineData &data) -> void {
    data.chunk_bases.shrink_to_fit();
    data.chunk_span_starts.shrink_to_fit();
    data.span_offsets.shrink_to_fit();
    data.span_counts.shrink_to_fit();
}

inline auto reserve_compressed_position_data(CompressedPositionData &data, size_t position_count) -> void {
    data.spans.reserve(estimated_cosine_span_capacity(position_count));
}

inline auto shrink_compressed_position_data(CompressedPositionData &data) -> void {
    data.spans.shrink_to_fit();
}

inline auto cosine_chunk_base(size_t start) -> size_t {
    return start & ~(kCosineChunkSize - 1);
}

inline auto cosine_chunk_offset(size_t start) -> uint16_t {
    return static_cast<uint16_t>(start & (kCosineChunkSize - 1));
}

inline auto cosine_chunk_span_begin(const CompressedCosineData &data, size_t chunk_idx) -> size_t {
    return data.chunk_span_starts[chunk_idx];
}

inline auto cosine_chunk_span_end(const CompressedCosineData &data, size_t chunk_idx) -> size_t {
    return chunk_idx + 1 < data.chunk_count() ? data.chunk_span_starts[chunk_idx + 1] : data.span_offsets.size();
}

inline auto cosine_chunk_index_for_span(const CompressedCosineData &data, size_t span_idx) -> size_t {
    const auto it = std::upper_bound(data.chunk_span_starts.begin(), data.chunk_span_starts.end(), span_idx);
    return static_cast<size_t>((it - data.chunk_span_starts.begin()) - 1);
}

inline auto cosine_subspan_start(const CompressedCosineData &data, size_t chunk_idx, size_t span_idx) -> size_t {
    return data.chunk_bases[chunk_idx] | static_cast<size_t>(data.span_offsets[span_idx]);
}

inline auto append_cosine_subspan(CompressedCosineData &data,
                                  size_t chunk_base,
                                  uint16_t chunk_offset,
                                  uint8_t chunk_count) -> void {
    const size_t start = chunk_base | static_cast<size_t>(chunk_offset);
    data.has_wide_start_values =
        data.has_wide_start_values || start > static_cast<size_t>(std::numeric_limits<uint32_t>::max());

    if (!data.chunk_bases.empty() && data.chunk_bases.back() == chunk_base) {
        const size_t last_span_idx = data.span_offsets.size() - 1;
        const size_t last_end = static_cast<size_t>(data.span_offsets[last_span_idx])
                                + static_cast<size_t>(data.span_counts[last_span_idx]);
        const size_t merged_count = static_cast<size_t>(data.span_counts[last_span_idx]) + chunk_count;
        if (last_end == static_cast<size_t>(chunk_offset) && merged_count <= kMaxCosineSpanLength) {
            data.span_counts[last_span_idx] = static_cast<uint8_t>(merged_count);
            return;
        }
    }
    else {
        data.chunk_bases.push_back(chunk_base);
        data.chunk_span_starts.push_back(data.span_offsets.size());
    }

    data.span_offsets.push_back(chunk_offset);
    data.span_counts.push_back(chunk_count);
}

inline auto append_cosine_run(CompressedCosineData &data, size_t run_begin, size_t run_length) -> void {
    if (run_length != 0 && run_length <= kMaxCosineSpanLength) {
        const uint16_t chunk_offset = cosine_chunk_offset(run_begin);
        if (static_cast<size_t>(chunk_offset) + run_length <= kCosineChunkSize) {
            append_cosine_subspan(data, cosine_chunk_base(run_begin), chunk_offset, static_cast<uint8_t>(run_length));
            return;
        }
    }

    size_t span_begin = run_begin;
    size_t remaining = run_length;
    while (remaining != 0) {
        const size_t chunk_base = cosine_chunk_base(span_begin);
        const uint16_t chunk_offset = cosine_chunk_offset(span_begin);
        const size_t chunk_capacity = kCosineChunkSize - static_cast<size_t>(chunk_offset);
        const size_t chunk = std::min({remaining, kMaxCosineSpanLength, chunk_capacity});

        append_cosine_subspan(data, chunk_base, chunk_offset, static_cast<uint8_t>(chunk));

        if (span_begin > std::numeric_limits<size_t>::max() - chunk) {
            throw std::overflow_error(std::format("Cosine span end {} + {} exceeds size_t.", span_begin, chunk));
        }
        span_begin += chunk;
        remaining -= chunk;
    }
}

inline auto append_compressed_cosine_data(CompressedCosineData &target, const CompressedCosineData &source) -> void {
    target.total_count += source.total_count;
    target.has_wide_start_values = target.has_wide_start_values || source.has_wide_start_values;

    if (source.empty()) {
        return;
    }

    target.chunk_bases.reserve(target.chunk_bases.size() + source.chunk_bases.size());
    target.chunk_span_starts.reserve(target.chunk_span_starts.size() + source.chunk_span_starts.size());
    target.span_offsets.reserve(target.span_offsets.size() + source.span_offsets.size());
    target.span_counts.reserve(target.span_counts.size() + source.span_counts.size());

    for (size_t chunk_idx = 0; chunk_idx < source.chunk_count(); ++chunk_idx) {
        const size_t begin = cosine_chunk_span_begin(source, chunk_idx);
        const size_t end = cosine_chunk_span_end(source, chunk_idx);
        const size_t chunk_base = source.chunk_bases[chunk_idx];

        if (target.chunk_bases.empty() || target.chunk_bases.back() != chunk_base) {
            target.chunk_bases.push_back(chunk_base);
            target.chunk_span_starts.push_back(target.span_offsets.size());
            target.span_offsets.insert(target.span_offsets.end(),
                                       source.span_offsets.begin() + static_cast<std::ptrdiff_t>(begin),
                                       source.span_offsets.begin() + static_cast<std::ptrdiff_t>(end));
            target.span_counts.insert(target.span_counts.end(),
                                      source.span_counts.begin() + static_cast<std::ptrdiff_t>(begin),
                                      source.span_counts.begin() + static_cast<std::ptrdiff_t>(end));
            continue;
        }

        for (size_t span_idx = begin; span_idx < end; ++span_idx) {
            append_cosine_subspan(target, chunk_base, source.span_offsets[span_idx], source.span_counts[span_idx]);
        }
    }
}

inline auto append_cosine_index(CompressedCosineData &data, size_t idx, PendingIndexRun &pending_run) -> void {
    if (!pending_run.active) {
        pending_run.begin = idx;
        pending_run.length = 1;
        pending_run.active = true;
        return;
    }

    const bool contiguous = pending_run.length <= std::numeric_limits<size_t>::max() - pending_run.begin
                            && idx == pending_run.begin + pending_run.length;
    if (contiguous) {
        ++pending_run.length;
        return;
    }

    append_cosine_run(data, pending_run.begin, pending_run.length);
    pending_run.begin = idx;
    pending_run.length = 1;
}

inline auto append_cosine_indices(CompressedCosineData &data,
                                  std::span<const size_t> cos_inds,
                                  PendingIndexRun &pending_run) -> void {
    for (const size_t idx : cos_inds) {
        append_cosine_index(data, idx, pending_run);
    }
}

inline auto finish_pending_cosine_run(CompressedCosineData &data, PendingIndexRun &pending_run) -> void {
    if (!pending_run.active) {
        return;
    }

    append_cosine_run(data, pending_run.begin, pending_run.length);
    pending_run = {};
}

template <typename Func>
inline auto for_each_cosine_span_range(const CompressedCosineData &data, size_t begin, size_t end, Func &&func)
    -> void {
    if (begin == end) {
        return;
    }

    size_t chunk_idx = cosine_chunk_index_for_span(data, begin);
    size_t chunk_end = cosine_chunk_span_end(data, chunk_idx);
    size_t chunk_base = data.chunk_bases[chunk_idx];
    for (size_t span_idx = begin; span_idx < end; ++span_idx) {
        if (span_idx == chunk_end) {
            ++chunk_idx;
            chunk_end = cosine_chunk_span_end(data, chunk_idx);
            chunk_base = data.chunk_bases[chunk_idx];
        }

        func(chunk_base | static_cast<size_t>(data.span_offsets[span_idx]), data.span_counts[span_idx]);
    }
}

template <typename Func>
inline auto for_each_cosine_span(const CompressedCosineData &data, Func &&func) -> void {
    for_each_cosine_span_range(data, 0, data.span_count(), [&func](size_t start, uint8_t count) {
        func(CosineSpan{start, count});
    });
}

inline auto materialize_stored_cosine_spans(const CompressedCosineData &data) -> std::vector<CosineSpan> {
    std::vector<CosineSpan> spans;
    spans.reserve(data.span_count());
    for (size_t chunk_idx = 0; chunk_idx < data.chunk_count(); ++chunk_idx) {
        const size_t span_begin = cosine_chunk_span_begin(data, chunk_idx);
        const size_t span_end = cosine_chunk_span_end(data, chunk_idx);
        for (size_t span_idx = span_begin; span_idx < span_end; ++span_idx) {
            spans.push_back(CosineSpan{
                cosine_subspan_start(data, chunk_idx, span_idx),
                static_cast<uint16_t>(data.span_counts[span_idx]),
            });
        }
    }
    return spans;
}

inline auto build_compressed_cosine_data(const VecZ &cos_inds) -> CompressedCosineData {
    CompressedCosineData data;
    data.total_count = cos_inds.size();
    if (cos_inds.empty()) {
        return data;
    }

    reserve_compressed_cosine_data(data, cos_inds.size());

    PendingIndexRun pending_run;
    append_cosine_indices(data, std::span<const size_t>{cos_inds.data(), cos_inds.size()}, pending_run);
    finish_pending_cosine_run(data, pending_run);

    return data;
}

inline auto expand_compressed_cosine_data(const CompressedCosineData &data) -> VecZ {
    VecZ cos_inds;
    cos_inds.reserve(static_cast<size_t>(data.total_count));
    for_each_cosine_span(data, [&cos_inds](const CosineSpan &span) {
        size_t idx = span.start;
        const size_t end = idx + static_cast<size_t>(span.count);
        for (; idx < end; ++idx) {
            cos_inds.push_back(idx);
        }
    });
    return cos_inds;
}

template <typename Func>
inline auto for_each_compressed_position_range(const CompressedPositionData &data,
                                               size_t begin,
                                               size_t end,
                                               Func &&func) -> void {
    if (begin == end) {
        return;
    }

    auto it = std::upper_bound(
        data.spans.begin(),
        data.spans.end(),
        begin,
        [](size_t logical_idx, const StoredPositionSpan &span) { return logical_idx < span.logical_start; });
    if (it != data.spans.begin()) {
        --it;
    }

    for (; it != data.spans.end(); ++it) {
        const size_t span_begin = it->logical_start;
        const size_t span_end = span_begin + static_cast<size_t>(it->count);
        if (span_end <= begin) {
            continue;
        }
        if (span_begin >= end) {
            break;
        }

        const size_t overlap_begin = std::max(begin, span_begin);
        const size_t overlap_end = std::min(end, span_end);
        func(overlap_begin,
             static_cast<size_t>(it->position_start) + (overlap_begin - span_begin),
             overlap_end - overlap_begin);
    }
}

inline auto append_position_span(CompressedPositionData &data, uint32_t position_start, uint16_t count) -> void {
    if (count == 0) {
        return;
    }

    if (!data.spans.empty()) {
        auto &last = data.spans.back();
        const size_t expected_position = static_cast<size_t>(last.position_start) + static_cast<size_t>(last.count);
        const size_t merged_count = static_cast<size_t>(last.count) + static_cast<size_t>(count);
        if (expected_position == static_cast<size_t>(position_start) && merged_count <= kMaxPositionSpanLength) {
            last.count = static_cast<uint16_t>(merged_count);
            data.total_count += static_cast<size_t>(count);
            return;
        }
    }

    data.spans.push_back(StoredPositionSpan{data.total_count, position_start, count});
    data.total_count += static_cast<size_t>(count);
}

inline auto append_position_run(CompressedPositionData &data, size_t run_begin, size_t run_length) -> void {
    size_t position = run_begin;
    size_t remaining = run_length;
    while (remaining != 0) {
        const size_t chunk = std::min(remaining, kMaxPositionSpanLength);
        append_position_span(data,
                             checked_packed_index(position, "Masked execution plan position"),
                             static_cast<uint16_t>(chunk));
        position += chunk;
        remaining -= chunk;
    }
}

inline auto append_position_index(CompressedPositionData &data, size_t idx, PendingIndexRun &pending_run) -> void {
    if (!pending_run.active) {
        pending_run.begin = idx;
        pending_run.length = 1;
        pending_run.active = true;
        return;
    }

    const bool contiguous = pending_run.length <= std::numeric_limits<size_t>::max() - pending_run.begin
                            && idx == pending_run.begin + pending_run.length;
    if (contiguous) {
        ++pending_run.length;
        return;
    }

    append_position_run(data, pending_run.begin, pending_run.length);
    pending_run.begin = idx;
    pending_run.length = 1;
}

inline auto finish_pending_position_run(CompressedPositionData &data, PendingIndexRun &pending_run) -> void {
    if (!pending_run.active) {
        return;
    }

    append_position_run(data, pending_run.begin, pending_run.length);
    pending_run = {};
}

inline auto build_compressed_position_data(std::vector<uint32_t> positions) -> CompressedPositionData {
    CompressedPositionData data;
    if (positions.empty()) {
        return data;
    }

    reserve_compressed_position_data(data, positions.size());
    PendingIndexRun pending_run;
    for (const uint32_t position : positions) {
        append_position_index(data, static_cast<size_t>(position), pending_run);
    }
    finish_pending_position_run(data, pending_run);
    return data;
}

inline auto compressed_position_at(const CompressedPositionData &data, size_t idx) -> uint32_t {
    const auto it = std::upper_bound(
        data.spans.begin(),
        data.spans.end(),
        idx,
        [](size_t logical_idx, const StoredPositionSpan &span) { return logical_idx < span.logical_start; });
    if (it == data.spans.begin()) {
        throw std::out_of_range("Compressed position lookup is out of range.");
    }

    const auto &span = *std::prev(it);
    return span.position_start + static_cast<uint32_t>(idx - span.logical_start);
}

inline auto compressed_cosine_data_storage_bytes(const CompressedCosineData &data) -> size_t {
    return data.chunk_bases.capacity() * sizeof(size_t) + data.chunk_span_starts.capacity() * sizeof(size_t)
           + data.span_offsets.capacity() * sizeof(uint16_t) + data.span_counts.capacity() * sizeof(uint8_t);
}

inline auto compressed_position_data_storage_bytes(const CompressedPositionData &data) -> size_t {
    return data.spans.capacity() * sizeof(StoredPositionSpan);
}

} // namespace monoprop::detail
