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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include "monoprop/core/Monomial.h"

// The single home for "which flat slot owns this monomial". Two call sites depend on agreeing exactly
// (Scan.h emits queries by it, MonomialPropagator seeds the operator by it), and a disagreement splits
// ownership silently rather than crashing -- so both go through Router::dest and nothing else.
//
// WHY there is a choice to make here at all
// ----------------------------------------
// A gate turns a term M into M^G (symmetric difference of Majorana support). With today's splitmix
// destination -- full avalanche -- the owner of M^G is unrelated to the owner of M, so a rank's queries
// for ONE generator spray across all R ranks and the exchange is a dense all-to-all whose message count
// grows as R*(R-1). If instead the RANK index is a GF(2)-LINEAR function of the support,
//
//     h(M) = XOR of v_i over i in support(M)   =>   h(M ^ G) = h(M) ^ h(G)
//
// then, since I own M, every query I emit for G goes to exactly one rank: my_rank ^ h(G). XOR is an
// involution, so that peer sends to me in the same round: the exchange becomes a pairwise Sendrecv.
//
// The routing is TWO-LEVEL, because the two levels have different costs: across MPI ranks the cost is
// the message COUNT (make it structured), within a rank partitions talk through shared memory where
// fanout is free and only balance matters (keep full avalanche).
//
//     part = q % S                       q = monomial_hash(M) (splitmix, unchanged)
//     hi   = (q / S) % (R >> d)          the R>>d splitmix-chosen high rank bits
//     rank = (a & (2^d - 1)) | (hi << d) a = linear_hash(M); d = linear_bits
//     flat = rank * S + part
//
// d is a dial, not a cliff: fanout is R >> d, so d = 0 is EXACTLY today's `q % (R*S)` (see dest()) and
// d = log2(R) is fanout 1. Non-power-of-two R has no XOR structure at all and falls back to d = 0.
//
// Knobs:
//   monoprop_ROUTING            splitmix (default) | linear   -- linear defaults d to log2(R)
//   monoprop_ROUTE_LINEAR_BITS  explicit d, clamped to [0, log2(R)]
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
    static const uint64_t seed = [] {
        const char *text = std::getenv("monoprop_ROUTE_SEED");
        return (text == nullptr || *text == '\0') ? kDefaultSeed : std::strtoull(text, nullptr, 10);
    }();
    return seed;
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

// Terms are sparse under a length cutoff (popcount <= ~2*cutoff), so iterating set bits beats a
// word-wise byte table. A CLMUL form would be popcount-independent; it is also a DIFFERENT linear map,
// so switching to it means re-measuring balance, not just re-benchmarking.
template <size_t NumBits>
[[nodiscard]] inline auto linear_hash(const monoprop::Bitset<NumBits> &bits) noexcept -> uint64_t {
    const auto &v = linear_basis<NumBits>();
    uint64_t h = 0;
    for (size_t i = bits.find_first(); i < NumBits; i = bits.find_next(i)) {
        h ^= v[i];
    }
    return h;
}

// GF(2) rank of a set of 64-bit vectors, by Gaussian elimination over the bit columns. The per-generator
// rank shifts must span at least log2(R) dimensions or the reachable destination ranks form a strict
// subspace and some ranks stay empty -- a load-balance failure, not a correctness one, which is why this
// is a diagnostic (measured: rank 32 for the 60-site Hubbard's 416 distinct shifts, against the 7 bits
// R = 128 needs) rather than a runtime gate.
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

enum class Mode : uint8_t { Splitmix, Linear };

// Trivially copyable and cheap to build; hold one per build_layer call rather than per term.
class Router final {
public:
    // ranks x partitions == the flat world the destinations index. linear_bits is clamped here, so a
    // caller may pass anything.
    constexpr Router(size_t ranks, size_t partitions, size_t linear_bits) noexcept
        : ranks_(ranks == 0 ? 1 : ranks),
          parts_(partitions == 0 ? 1 : partitions),
          flat_(ranks_ * parts_),
          bits_(clamp_bits_(ranks_, linear_bits)),
          lin_mask_((uint64_t{1} << bits_) - 1),
          hi_mask_((ranks_ >> bits_) - 1) {}

    // Today's routing: one flat world, full avalanche, no linear bits. Also what d = 0 collapses to.
    static constexpr auto splitmix(size_t flat_world) noexcept -> Router { return Router{flat_world, 1, 0}; }

    [[nodiscard]] constexpr auto ranks() const noexcept -> size_t { return ranks_; }
    [[nodiscard]] constexpr auto partitions() const noexcept -> size_t { return parts_; }
    [[nodiscard]] constexpr auto flat_world() const noexcept -> size_t { return flat_; }
    [[nodiscard]] constexpr auto linear_bits() const noexcept -> size_t { return bits_; }
    // Distinct destination RANKS one rank's queries for a single generator reach. 1 == pairwise.
    [[nodiscard]] constexpr auto fanout() const noexcept -> size_t { return ranks_ >> bits_; }
    [[nodiscard]] constexpr auto mode() const noexcept -> Mode { return bits_ == 0 ? Mode::Splitmix : Mode::Linear; }

    // Flat destination slot in [0, flat_world). Branch is on a member, so it is perfectly predicted.
    template <size_t NumModes>
    [[nodiscard]] [[gnu::always_inline]] inline auto dest(const Monomial<NumModes> &mono) const noexcept -> size_t {
        const uint64_t q = monomial_hash<NumModes>(mono);
        if (bits_ == 0) {
            return static_cast<size_t>(q % flat_); // bit-for-bit today's `hash % P`
        }
        const uint64_t part = q % parts_;
        const uint64_t hi = (q / parts_) & hi_mask_; // ranks_>>bits_ is a power of two, so a mask
        const uint64_t lin = linear_hash<2 * NumModes>(mono) & lin_mask_;
        return static_cast<size_t>(((lin | (hi << bits_)) * parts_) + part);
    }

    // The rank-level shift a generator induces: rank(M^G) low bits == rank(M) low bits ^ shift(G).
    // Zero for every G iff bits_ == 0. This is what makes the destination predictable.
    template <size_t NumModes>
    [[nodiscard]] auto rank_shift(const Monomial<NumModes> &gen) const noexcept -> size_t {
        return static_cast<size_t>(linear_hash<2 * NumModes>(gen) & lin_mask_);
    }

private:
    static constexpr auto clamp_bits_(size_t ranks, size_t requested) noexcept -> size_t {
        if (!std::has_single_bit(ranks)) {
            return 0; // no XOR structure without a power-of-two rank count
        }
        const size_t max_bits = static_cast<size_t>(std::countr_zero(ranks));
        return requested < max_bits ? requested : max_bits;
    }

    size_t ranks_;
    size_t parts_;
    size_t flat_;
    size_t bits_;
    uint64_t lin_mask_;
    uint64_t hi_mask_;
};

// The requested linear-bit count, before clamping to a particular geometry. Parsed once.
inline auto requested_linear_bits() -> size_t {
    static const size_t bits = [] {
        const char *explicit_bits = std::getenv("monoprop_ROUTE_LINEAR_BITS");
        if (explicit_bits != nullptr && *explicit_bits != '\0') {
            const long value = std::strtol(explicit_bits, nullptr, 10);
            return value > 0 ? static_cast<size_t>(value) : size_t{0};
        }
        const char *mode = std::getenv("monoprop_ROUTING");
        if (mode != nullptr && std::string_view{mode} == "splitmix") {
            return size_t{0}; // full avalanche across the flat world: every rank talks to every rank
        }
        // Default. "As many bits as this geometry allows" -- Router clamps to log2(R), and to 0
        // when R is not a power of two, so a geometry without XOR structure keeps the dense path.
        // Measured at the production point: fanout 1 costs nothing on balance (rank occupancy
        // max/mean 1.001 at R=128, all ranks used) and takes messages per rank per layer from
        // 362,712 to 1,397, i.e. from proportional-to-R to flat.
        return ~size_t{0};
    }();
    return bits;
}

inline auto make_router(size_t ranks, size_t partitions) -> Router {
    return Router{ranks, partitions, requested_linear_bits()};
}

} // namespace monoprop::routing
