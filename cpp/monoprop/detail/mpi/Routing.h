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

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/EnvConfig.h"

// The single home for "which flat slot owns this monomial". Two call sites depend on agreeing exactly
// (Scan.h emits queries by it, MonomialPropagator seeds the operator by it), and a disagreement splits
// ownership silently rather than crashing -- so both go through Router::dest and nothing else.
//
// Two-level, because the levels cost differently: across MPI ranks the message COUNT is what hurts, so
// the rank index is GF(2)-linear in the support and a generator maps every query to one peer; within a
// rank partitions talk through shared memory, where fanout is free and only balance matters.
//
//     part = q % S                       q = monomial_hash(M) (splitmix, unchanged)
//     hi   = (q / S) % (R >> d)          the R>>d splitmix-chosen high rank bits
//     rank = (a & (2^d - 1)) | (hi << d) a = linear_hash(M); d = linear_bits
//     flat = rank * S + part
//
// The derivation, what d buys and what it costs: docs/content/docs/features/parallelism.mdx, under
// "Rank routing".
//
// Knobs, parsed and validated in EnvConfig.h:
//   monoprop_ROUTING            linear (default) | splitmix   -- linear defaults d to log2(R)
//   monoprop_ROUTE_LINEAR_BITS  explicit d, clamped to [0, log2(R)]; overrides monoprop_ROUTING
//   monoprop_ROUTE_SEED         uint64 seed for the linear basis (default kDefaultSeed)

namespace monoprop::routing {

inline constexpr uint64_t kDefaultSeed = 0x5DEE'CE66'D0C6'2517ULL;

inline constexpr auto mix64(uint64_t x) noexcept -> uint64_t {
    x += 0x9E37'79B9'7F4A'7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58'476D'1CE4'E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D0'49BB'1331'11EBULL;
    return x ^ (x >> 31);
}

inline auto seed_from_env() -> uint64_t {
    return config::get().route_seed.value_or(kDefaultSeed);
}

// One 64-bit vector per Majorana mode. Deterministic from the seed alone, so every rank builds the
// same table with no communication -- the property find_rank's contract rests on.
template <size_t NumBits>
inline auto linear_basis() -> const std::array<uint64_t, NumBits> & {
    static const auto table = [] {
        std::array<uint64_t, NumBits> v{};
        const uint64_t seed = seed_from_env();
        for (size_t i = 0; i < NumBits; ++i) {
            v[i] = mix64(mix64(seed) + (static_cast<uint64_t>(i) * 0x9E37'79B9'7F4A'7C15ULL));
        }
        return v;
    }();
    return table;
}

// The full 64-bit image, by walking the set bits. Router::dest does NOT use this -- it needs only the
// low d bits and takes the transposed form below -- but the map is defined here, and the GF(2)-linearity
// test and the plane build both read it as the definition.
template <size_t NumBits>
[[nodiscard]] inline auto linear_hash(const monoprop::Bitset<NumBits> &bits) noexcept -> uint64_t {
    const auto &v = linear_basis<NumBits>();
    uint64_t h = 0;
    for (size_t i = bits.find_first(); i < NumBits; i = bits.find_next(i)) {
        h ^= v[i];
    }
    return h;
}

inline constexpr size_t kLinearPlanes = 64; // one per output bit; d <= 63 of them are ever read

template <size_t NumBits>
inline constexpr size_t kPlaneWords = monoprop::Bitset<NumBits>::num_words();

// The basis transposed: plane j, bit i, is bit j of basis vector i. Then bit j of linear_hash(M) is
// popcount(M & plane_j) & 1 -- d * words branch-free AND/XOR/popcount ops instead of a gather whose
// length is the term's popcount (~20-28 under the production cutoff). Same map, bit for bit.
//
// Keyed on the seed and the width alone, never on the geometry, so one table serves every Router; the
// Router binds a pointer to it at construction and dest() never reaches the static-init guard.
template <size_t NumBits>
inline auto linear_planes() -> const std::array<uint64_t, kLinearPlanes * kPlaneWords<NumBits>> & {
    static const auto table = [] {
        std::array<uint64_t, kLinearPlanes * kPlaneWords<NumBits>> planes{};
        const auto &v = linear_basis<NumBits>();
        for (size_t i = 0; i < NumBits; ++i) {
            const size_t word = i / 64;
            const uint64_t bit = uint64_t{1} << (i % 64);
            for (size_t j = 0; j < kLinearPlanes; ++j) {
                if (((v[i] >> j) & 1U) != 0) {
                    planes[(j * kPlaneWords<NumBits>)+word] |= bit;
                }
            }
        }
        return planes;
    }();
    return table;
}

// GF(2) rank of a set of 64-bit vectors, by Gaussian elimination over the bit columns. The per-generator
// rank shifts must span at least log2(R) dimensions or the reachable destination ranks form a strict
// subspace and 2^d - 2^rank ranks stay empty. A balance failure, not a correctness one, so its caller
// (MonomialPropagator::report_routing_coverage_) warns on stderr rather than gating.
// Measured: rank 32 for the 60-site Hubbard's 416 distinct shifts, against the 7 bits R = 128 needs.
[[nodiscard]] inline auto gf2_rank(std::vector<uint64_t> vectors) noexcept -> size_t {
    size_t rank = 0;
    for (size_t bit = 0; bit < 64 && rank < vectors.size(); ++bit) {
        const uint64_t probe = uint64_t{1} << bit;
        size_t pivot = vectors.size();
        for (size_t i = rank; i < vectors.size(); ++i) {
            if ((vectors[i] & probe) != 0) {
                pivot = i;
                break;
            }
        }
        if (pivot == vectors.size()) {
            continue;
        }
        std::swap(vectors[rank], vectors[pivot]);
        for (size_t i = 0; i < vectors.size(); ++i) {
            if (i != rank && (vectors[i] & probe) != 0) {
                vectors[i] ^= vectors[rank];
            }
        }
        ++rank;
    }
    return rank;
}

// Trivially copyable and cheap to build; hold one per build_layer call rather than per term.
class Router final {
public:
    // Bound to a monomial width: dest() reads the transposed basis for THAT width, and binding the
    // pointer here is what keeps the static-init guard out of the per-term path. The only way to a
    // router with linear bits, so an unbound one cannot reach dest().
    template <size_t NumModes>
    [[nodiscard]] static auto for_modes(size_t ranks, size_t partitions, size_t linear_bits) -> Router {
        Router r{ranks, partitions, linear_bits};
        r.planes_ = linear_planes<2 * NumModes>().data();
        r.plane_words_ = kPlaneWords<2 * NumModes>;
        return r;
    }

    // Today's routing: one flat world, full avalanche, no linear bits. Also what d = 0 collapses to.
    // Width-free: with bits_ == 0 no plane is ever read.
    static constexpr auto splitmix(size_t flat_world) noexcept -> Router { return Router{flat_world, 1, 0}; }

    // The clamp on its own, for callers that need the resolved d without a monomial width.
    static constexpr auto clamp_bits(size_t ranks, size_t requested) noexcept -> size_t {
        if (!std::has_single_bit(ranks)) {
            return 0; // no XOR structure without a power-of-two rank count
        }
        const auto max_bits = static_cast<size_t>(std::countr_zero(ranks));
        return requested < max_bits ? requested : max_bits;
    }

    [[nodiscard]] constexpr auto ranks() const noexcept -> size_t { return ranks_; }
    [[nodiscard]] constexpr auto partitions() const noexcept -> size_t { return parts_; }
    [[nodiscard]] constexpr auto flat_world() const noexcept -> size_t { return flat_; }
    [[nodiscard]] constexpr auto linear_bits() const noexcept -> size_t { return bits_; }
    // Distinct destination RANKS one rank's queries for a single generator reach. 1 == pairwise.
    [[nodiscard]] constexpr auto fanout() const noexcept -> size_t { return ranks_ >> bits_; }

    // Flat destination slot in [0, flat_world). Branch is on a member, so it is perfectly predicted.
    template <size_t NumModes>
    [[nodiscard]] [[gnu::always_inline]] inline auto dest(const Monomial<NumModes> &mono) const noexcept -> size_t {
        const uint64_t q = monomial_hash<NumModes>(mono);
        if (bits_ == 0) {
            return static_cast<size_t>(q % flat_); // bit-for-bit today's `hash % P`
        }
        const uint64_t part = q % parts_;
        const uint64_t hi = (q / parts_) & hi_mask_; // ranks_>>bits_ is a power of two, so a mask
        return static_cast<size_t>(((linear_low_<NumModes>(mono) | (hi << bits_)) * parts_) + part);
    }

    // The rank-level shift a generator induces: rank(M^G) low bits == rank(M) low bits ^ shift(G).
    // Zero for every G iff bits_ == 0. This is what makes the destination predictable.
    template <size_t NumModes>
    [[nodiscard]] auto rank_shift(const Monomial<NumModes> &gen) const noexcept -> size_t {
        return static_cast<size_t>(linear_low_<NumModes>(gen));
    }

private:
    // ranks x partitions == the flat world the destinations index. linear_bits is clamped here, so a
    // caller may pass anything. Private: a router with bits_ > 0 must go through for_modes.
    constexpr Router(size_t ranks, size_t partitions, size_t linear_bits) noexcept
        : ranks_(ranks == 0 ? 1 : ranks),
          parts_(partitions == 0 ? 1 : partitions),
          flat_(ranks_ * parts_),
          bits_(clamp_bits(ranks_, linear_bits)),
          hi_mask_((ranks_ >> bits_) - 1) {}

    // linear_hash(M) & lin_mask, one output bit per plane: parity(popcount(M & plane_j)). Folding the
    // words with XOR before the popcount is the same parity (popcount(x)+popcount(y) == popcount(x^y)
    // mod 2) for one popcount per bit instead of one per word. bits_ == 0 reads no plane, so a
    // splitmix router needs none.
    template <size_t NumModes>
    [[nodiscard]] [[gnu::always_inline]] inline auto linear_low_(const Monomial<NumModes> &m) const noexcept
        -> uint64_t {
        constexpr size_t kW = kPlaneWords<2 * NumModes>;
        assert(bits_ == 0 || (planes_ != nullptr && plane_words_ == kW)); // bound at a different width
        uint64_t acc = 0;
        for (size_t j = 0; j < bits_; ++j) {
            const uint64_t *plane = planes_ + (j * kW);
            uint64_t fold = 0;
            for (size_t w = 0; w < kW; ++w) {
                fold ^= m.word(w) & plane[w];
            }
            acc |= static_cast<uint64_t>(std::popcount(fold) & 1) << j;
        }
        return acc;
    }

    size_t ranks_;
    size_t parts_;
    size_t flat_;
    size_t bits_;
    uint64_t hi_mask_;
    const uint64_t *planes_ = nullptr; // [kLinearPlanes x plane_words_], owned by linear_planes()
    size_t plane_words_ = 0;
};

// The requested linear-bit count, before clamping to a particular geometry. Parsed once.
inline auto requested_linear_bits() -> size_t {
    const auto &env = config::get();
    if (env.route_linear_bits.has_value()) {
        return static_cast<size_t>(*env.route_linear_bits);
    }
    if (env.routing_mode == config::RoutingMode::Splitmix) {
        return 0; // full avalanche across the flat world: every rank talks to every rank
    }
    // Default. "As many bits as this geometry allows" -- Router clamps to log2(R), and to 0 when R is
    // not a power of two, so a geometry without XOR structure keeps the dense path. Measured at the
    // production point: fanout 1 costs nothing on balance (rank occupancy max/mean 1.001 at R=128, all
    // ranks used) and takes messages per rank per layer from 362,712 to 1,397, i.e. from
    // proportional-to-R to flat.
    return ~size_t{0};
}

template <size_t NumModes>
inline auto make_router(size_t ranks, size_t partitions) -> Router {
    return Router::for_modes<NumModes>(ranks, partitions, requested_linear_bits());
}

// The resolved d for a geometry, without binding a width: the routing-agreement digest needs the
// number and nothing else.
inline auto linear_bits_for(size_t ranks) -> size_t {
    return Router::clamp_bits(ranks == 0 ? 1 : ranks, requested_linear_bits());
}

} // namespace monoprop::routing
