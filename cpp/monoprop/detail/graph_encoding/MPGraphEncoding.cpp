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

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"

#include <algorithm>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace monoprop::detail {

auto checked_mpi_int(size_t value, const char *what) -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the MPI int limit {}.", what, value, std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

auto build_layer_exchange_layout(const std::vector<size_t> &send_counts, int scale, const char *what)
    -> LayerExchangeLayout {
    const std::string count_label = std::format("{} count", what);
    const std::string displacement_label = std::format("{} displacement", what);

    LayerExchangeLayout layout;
    layout.counts.resize(send_counts.size());
    layout.displs.resize(send_counts.size());
    size_t total = 0;
    for (size_t r = 0; r < send_counts.size(); ++r) {
        const size_t count = static_cast<size_t>(scale) * send_counts[r];
        layout.counts[r] = checked_mpi_int(count, count_label.c_str());
        layout.displs[r] = checked_mpi_int(total, displacement_label.c_str());
        total += count;
    }
    layout.total_count = total;
    return layout;
}

auto build_derivative_exchange_layout(const LayerExchangeLayout &evolution) -> LayerExchangeLayout {
    std::vector<size_t> send_counts;
    send_counts.reserve(evolution.counts.size());
    for (const int count : evolution.counts) {
        send_counts.push_back(static_cast<size_t>(count));
    }
    return build_layer_exchange_layout(send_counts, 2, "Layer derivative exchange");
}

auto checked_term_index(size_t value, const char *what) -> TermIndex {
    if (value > static_cast<size_t>(std::numeric_limits<TermIndex>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the TermIndex ceiling {}; rebuild with -Dmonoprop_WIDE_TERM_INDEX.",
                        what,
                        value,
                        std::numeric_limits<TermIndex>::max()));
    }
    return static_cast<TermIndex>(value);
}

auto checked_packed_phase(int value, const char *what) -> int8_t {
    if (value < static_cast<int>(std::numeric_limits<int8_t>::min())
        || value > static_cast<int>(std::numeric_limits<int8_t>::max())) {
        throw std::overflow_error(std::format("{} {} exceeds the 8-bit phase limit.", what, value));
    }
    return static_cast<int8_t>(value);
}

auto make_packed_phase_storage(size_t count, bool use_binary_phases) -> PackedPhaseStorage {
    PackedPhaseStorage storage;
    storage.uses_binary_phases = use_binary_phases;
    storage.total_count = count;
    if (use_binary_phases) {
        storage.phase_words.assign(packed_phase_word_count(count), 0);
    }
    else {
        storage.phase_values.resize(count);
    }
    return storage;
}

auto packed_phase_storage_bytes(const PackedPhaseStorage &storage) -> size_t {
    return storage.uses_binary_phases ? storage.phase_words.capacity() * sizeof(uint64_t)
                                      : storage.phase_values.capacity() * sizeof(int8_t);
}

auto build_packed_cross_rank_storage(const std::vector<CrossRankPartnerData> &data) -> PackedCrossRankStorage {
    PackedCrossRankStorage storage;
    const size_t num_ranks = data.size();
    storage.ranges.resize(num_ranks);

    size_t total_b = 0;
    size_t total_d = 0;
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        auto &range = storage.ranges[rank];
        range.sin_send_offset = total_b;
        range.sin_send_count = static_cast<TermIndex>(partner.sin_send_indices.size());
        range.sin_recv_offset = total_d;
        range.sin_recv_count = static_cast<TermIndex>(partner.sin_recv_entries.size());
        range.in_count = static_cast<TermIndex>(partner.in_count);
        total_b += partner.sin_send_indices.size();
        total_d += partner.sin_recv_entries.size();
    }

    bool uses_binary_phases = true;
    for (const auto &partner : data) {
        bool non_binary_phase = false;
        for (const auto &[recv_index, phase] : partner.sin_recv_entries) {
            non_binary_phase = non_binary_phase || !is_binary_phase(phase);
        }
        uses_binary_phases = uses_binary_phases && !non_binary_phase;
    }

    storage.sin_send_indices.resize(total_b);
    storage.sin_recv_phases = make_packed_phase_storage(total_d, uses_binary_phases);

    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        const size_t b_off = storage.ranges[rank].sin_send_offset;
        const size_t d_off = storage.ranges[rank].sin_recv_offset;

        for (size_t k = 0; k < partner.sin_send_indices.size(); ++k) {
            storage.sin_send_indices[b_off + k] = checked_term_index(partner.sin_send_indices[k], "Cross-rank B index");
        }

        for (size_t k = 0; k < partner.sin_recv_entries.size(); ++k) {
            const auto &[i, phi] = partner.sin_recv_entries[k];
            (void)i;
            store_packed_phase(storage.sin_recv_phases, d_off + k, phi, "Cross-rank D phase");
        }
    }

    return storage;
}

auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t {
    size_t bytes = cross_rank_slot_record_bytes(storage) + packed_phase_storage_bytes(storage.sin_recv_phases);
    bytes += storage.sin_send_indices.capacity() * sizeof(TermIndex);
    return bytes;
}

auto cross_rank_slot_record_bytes(const PackedCrossRankStorage &storage) -> size_t {
    return storage.ranges.capacity() * sizeof(CrossRankPartnerRange);
}

auto cross_rank_occupied_slots(const PackedCrossRankStorage &storage) -> size_t {
    // sin_send_count alone is the predicate. B and D hold the same endpoint set in two orders (see
    // cross_rank_sin_recv_index), so a slot cannot carry D entries while carrying no B entries, and
    // counting either gives the same answer.
    return static_cast<size_t>(std::ranges::count_if(
        storage.ranges, [](const CrossRankPartnerRange &range) { return range.sin_send_count != 0; }));
}

auto cross_rank_endpoint_count(const PackedCrossRankStorage &storage) -> size_t {
    size_t count = 0;
    for (const auto &range : storage.ranges) {
        count += range.sin_send_count;
    }
    return count;
}

auto layer_exchange_layout_storage_bytes(const LayerExchangeLayout &layout) -> size_t {
    return layout.counts.capacity() * sizeof(int) + layout.displs.capacity() * sizeof(int);
}

auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore> {
    auto storage = std::make_shared<LayerCore>();

    {
        std::vector<size_t> send_counts;
        send_counts.reserve(all_partners.size());
        for (size_t r = 0; r < all_partners.size(); ++r) {
            send_counts.push_back((r == my_rank) ? size_t{0} : all_partners[r].sin_send_indices.size());
        }
        storage->evolution_exchange_layout = build_layer_exchange_layout(send_counts, 1);

        // The derivative layout (2x) is allocated lazily on first gradient read, but validated here: an
        // overflow must throw during build_graph, not from inside the gradient collective window, where
        // peers are already blocked in mpi::resolve_recv's count round -> a distributed hang, not an error.
        static_cast<void>(build_derivative_exchange_layout(storage->evolution_exchange_layout));
    }

    storage->cross_rank = build_packed_cross_rank_storage(std::move(all_partners));

    // Both are indexed by the same rank space.
    if (storage->evolution_exchange_layout.counts.size() != storage->cross_rank.rank_count()) {
        throw ExchangeLayoutRankMismatch(
            std::format("Layer exchange layout covers {} ranks but cross-rank storage has {}.",
                        storage->evolution_exchange_layout.counts.size(),
                        storage->cross_rank.rank_count()));
    }
    return storage;
}

} // namespace monoprop::detail

namespace monoprop {

auto LayerCore::derivative_exchange_layout() const -> const LayerExchangeLayout & {
    if (!derivative_exchange_layout_cache_) {
        derivative_exchange_layout_cache_ = detail::build_derivative_exchange_layout(evolution_exchange_layout);
    }
    return *derivative_exchange_layout_cache_;
}

} // namespace monoprop
