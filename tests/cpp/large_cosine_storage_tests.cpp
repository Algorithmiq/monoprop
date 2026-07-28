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

#include <boost/test/unit_test.hpp>

#include <limits>
#include <memory>

#include "monoprop/Evolution.h"
#include "monoprop/MPGraph.h"
#include "monoprop/detail/evolution/CosineRecompute.h"

using namespace monoprop;

BOOST_AUTO_TEST_CASE(pruned_layer_supports_cos_counts_above_u32) {
    const size_t large_count = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 9;
    const size_t large_index = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 17;

    // B layout: in-block first (200..211), then out-block (100..107).
    std::vector<CrossRankPartnerData> cross_rank(2);
    auto &p = cross_rank[1];
    for (size_t idx = 0; idx < 12; ++idx) {
        p.sin_send_indices.push_back(200 + idx);
    }
    for (size_t idx = 0; idx < 8; ++idx) {
        p.sin_send_indices.push_back(100 + idx);
    }
    // Single phased D list: D- (out) block first with the phase negated, then D+ (in) block as-is.
    for (size_t idx = 0; idx < 8; ++idx) {
        p.sin_recv_entries.push_back({100 + idx, -(idx % 2 == 0 ? 1 : -1)});
    }
    for (size_t idx = 0; idx < 12; ++idx) {
        p.sin_recv_entries.push_back({200 + idx, idx % 2 == 0 ? -1 : 1});
    }
    p.in_count = 12; // in-block size P (D indices are derived from B via in_count)

    auto storage = detail::build_layer_storage_unified(std::move(cross_rank), /*my_rank=*/0);

    // An engaged pruned_cos stores its filtered cosine list explicitly, so num_cos_inds() reports its
    // total_count.
    CosMask pruned_cos;
    const size_t block_base = (large_index >> 6) << 6;
    pruned_cos.blocks.emplace_back(block_base, uint64_t{1} << (large_index & 63u));
    pruned_cos.total_count = large_count;

    Layer layer{storage, std::move(pruned_cos)};
    const auto lt = layer.traversal();

    BOOST_CHECK_EQUAL(lt.num_cos_inds(), large_count);

    // Cross-rank is read verbatim from the core (never masked): B[0] = in-block[0] = 200.
    size_t b_idx = static_cast<size_t>(-1);
    lt.for_each_cross_rank_sin_send_range(1, 0, 1, [&](size_t, size_t i) { b_idx = i; });
    BOOST_CHECK_EQUAL(b_idx, 200UL);

    // D[0] is derived from B: Q = sin_recv_count - in_count = 20 - 12 = 8, so D[0] = out-block[0] = 100,
    // stored phase = -(out_phases[0]) = -(+1) = -1.
    size_t d_idx = static_cast<size_t>(-1);
    int d_phi = 0;
    lt.for_each_cross_rank_sin_recv_range(1, 0, 1, [&](size_t, size_t i, int phi) {
        d_idx = i;
        d_phi = phi;
    });
    BOOST_CHECK_EQUAL(d_idx, 100UL);
    BOOST_CHECK_EQUAL(d_phi, -1);
}

// The per-rank cross-rank counts index into one layer's term set, so under the wide build they must
// be TermIndex-wide; uint32_t would silently cap a single partition/layer at ~2^32 terms.
BOOST_AUTO_TEST_CASE(cross_rank_partner_range_counts_track_term_index_width) {
    CrossRankPartnerRange r{};
    BOOST_CHECK_EQUAL(sizeof(r.sin_send_count), sizeof(TermIndex));
    BOOST_CHECK_EQUAL(sizeof(r.sin_recv_count), sizeof(TermIndex));
    BOOST_CHECK_EQUAL(sizeof(r.in_count), sizeof(TermIndex));
}

#if defined(monoprop_WIDE_TERM_INDEX)
// Under the wide build (TermIndex = u64), a cross-rank B (partner term) index above 2^32 must
// round-trip losslessly through the packed cross-rank storage rather than hit a UINT32_MAX cap.
BOOST_AUTO_TEST_CASE(cross_rank_sin_send_index_round_trips_above_u32) {
    const size_t big_in = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1000;
    const size_t big_out = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 5;

    std::vector<CrossRankPartnerData> cross_rank(2);
    auto &p = cross_rank[1];
    // B layout: in-block first, then out-block. One entry each.
    p.sin_send_indices.push_back(big_in);
    p.sin_send_indices.push_back(big_out);
    p.sin_recv_entries.push_back({big_out, -1}); // D- (out) block, stored -phase
    p.sin_recv_entries.push_back({big_in, 1});   // D+ (in) block
    p.in_count = 1;

    const auto storage = detail::build_packed_cross_rank_storage(std::move(cross_rank));

    BOOST_CHECK_EQUAL(detail::cross_rank_sin_send_index(storage, 1, 0), big_in);
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_send_index(storage, 1, 1), big_out);
    // D[0] is derived from B: Q = sin_recv_count - in_count = 1, so D[0] = out-block[0] = big_out.
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_index(storage, 1, 0), big_out);
}
#endif

// The cross-rank exchange uses MPI int counts/displacements, so a single per-rank exchange is capped
// at INT_MAX elements; checked_mpi_int must throw cleanly at that limit, never wrap silently.
BOOST_AUTO_TEST_CASE(checked_mpi_int_throws_cleanly_above_int_max) {
    const size_t at_limit = static_cast<size_t>(std::numeric_limits<int>::max());
    BOOST_CHECK_EQUAL(detail::checked_mpi_int(at_limit, "exchange count"), std::numeric_limits<int>::max());
    BOOST_CHECK_THROW(detail::checked_mpi_int(at_limit + 1, "exchange count"), std::overflow_error);
}

BOOST_AUTO_TEST_CASE(cosine_word_list_scale_and_accumulate) {
    using monoprop::CosMask;
    CosMask cos;
    cos.blocks = {{0, 0b1011ULL}, {64, 0b1ULL}, {192, (1ULL << 5)}};
    cos.total_count = 3 + 1 + 1;

    const std::vector<double> base_coeff(256, 2.0);
    std::vector<double> par = base_coeff;
    monoprop::detail::scale_cos_mask(par.data(), cos, 3.0);
    BOOST_TEST(par[0] == 6.0);
    BOOST_TEST(par[1] == 6.0);
    BOOST_TEST(par[3] == 6.0);
    BOOST_TEST(par[2] == 2.0);
    BOOST_TEST(par[64] == 6.0);
    BOOST_TEST(par[197] == 6.0);
    BOOST_TEST(par[100] == 2.0);

    std::vector<double> pp(256, 1.5), ph(256, 0.5);
    const double a_par = monoprop::detail::accumulate_cos_mask(pp.data(), ph.data(), cos, 0.7, 0.9);
    // Returns sum(state[i]*ham[i]) over the 5 set indices, taken before the scaling below.
    BOOST_TEST(a_par == 5.0 * 1.5 * 0.5, boost::test_tools::tolerance(1e-12));
    // state and ham at set indices scaled by cos_val and sec_val respectively
    BOOST_TEST(pp[0] == 1.5 * 0.7, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(ph[0] == 0.5 * 0.9, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(packed_cross_rank_storage_bit_packs_binary_phases) {
    std::vector<CrossRankPartnerData> binary_cross_rank(2);
    std::vector<CrossRankPartnerData> wide_phase_cross_rank(2);

    // B layout: in-block first (idx+1005), then out-block (idx+5).
    for (size_t idx = 0; idx < 128; ++idx) {
        binary_cross_rank[1].sin_send_indices.push_back(idx + 1005);
    }
    for (size_t idx = 0; idx < 128; ++idx) {
        binary_cross_rank[1].sin_send_indices.push_back(idx + 5);
    }
    // Single phased D list, as GraphSink::finalize() builds it: D- (out) block first with the phase
    // negated, then D+ (in) block with the phase stored as-is.
    for (size_t idx = 0; idx < 128; ++idx) {
        const int phase = idx % 2 == 0 ? 1 : -1;
        binary_cross_rank[1].sin_recv_entries.push_back({idx + 5, -phase});
    }
    for (size_t idx = 0; idx < 128; ++idx) {
        const int phase = idx % 2 == 0 ? 1 : -1;
        binary_cross_rank[1].sin_recv_entries.push_back({idx + 1005, phase});
    }
    binary_cross_rank[1].in_count = 128;
    wide_phase_cross_rank[1] = binary_cross_rank[1];
    // Make wide: set the first D- entry to a non-binary stored phase (-2).
    wide_phase_cross_rank[1].sin_recv_entries[0].second = -2;

    const auto binary_storage = detail::build_packed_cross_rank_storage(std::move(binary_cross_rank));
    const auto wide_phase_storage = detail::build_packed_cross_rank_storage(std::move(wide_phase_cross_rank));

    // All input phases are ±1 so the binary storage uses 1-bit packing.
    BOOST_CHECK(binary_storage.sin_recv_phases.uses_binary_phases);
    BOOST_CHECK(!wide_phase_storage.sin_recv_phases.uses_binary_phases);

    // D^-[1] = term idx+5=6, phase -1, stored negated: -(-1) = 1.
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_phase(binary_storage, 1, 1), 1);
    // D^+[0] = term idx+1005=1005, phase +1, stored as-is. (D^+ starts at flat index 128.)
    BOOST_CHECK_EQUAL(detail::cross_rank_sin_recv_phase(binary_storage, 1, 128), 1);

    BOOST_CHECK_LT(detail::cross_rank_storage_bytes(binary_storage),
                   detail::cross_rank_storage_bytes(wide_phase_storage));
}
