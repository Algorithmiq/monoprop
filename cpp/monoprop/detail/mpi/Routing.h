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
#include <format>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/EnvConfig.h"

// The single home for "which flat slot owns this monomial". Two call sites depend on agreeing exactly
// (Scan.h emits queries by it, MonomialPropagator seeds the operator by it), and a disagreement splits
// ownership silently rather than crashing -- so both go through this Router and nothing else. Scan.h
// enters at dest_from_fingerprint with the partner's fingerprint fp(M) ^ fp(G).
//
// Two-level, because the levels cost differently: across MPI ranks the message COUNT is what hurts, so
// the rank index is GF(2)-linear in the support and a generator maps every query to one peer; within a
// rank partitions talk through shared memory, where fanout is free and only balance matters.
//
//     rank = fp & (R - 1)               fp = linear_hash(M); all log2(R) rank bits, so fanout is 1
//     part = (fp >> log2 R) & (S - 1)   S a power of two: linear too, so a slot has ONE peer slot
//          = mix64(fp) % S              otherwise: balance only
//     flat = rank * S + part
//
// Linear or not is a switch and not a dial: the rank takes every bit from the linear hash or none of
// them. None of them is the splitmix router, which is `q % (R*S)` bit for bit, and R == 1 is that case
// by construction. R > 1 must then be a power of two, or there is no XOR structure to route by and the
// geometry is rejected (UnroutableGeometry) rather than silently falling back. The same fingerprint is
// the engine's per-gate dedup key (layer_build/AntiTable.h), so one label table serves both.
//
// The derivation, what linear routing buys and what it costs: docs/content/docs/features/parallelism.mdx,
// under "Rank routing".
//
// Knobs, parsed and validated in EnvConfig.h:
//   monoprop_ROUTING     linear (default) | splitmix
//   monoprop_ROUTE_SEED  uint64 seed for the linear basis (default kDefaultSeed)

namespace monoprop::routing {

inline constexpr uint64_t kDefaultSeed = 0x5DEE'CE66'D0C6'2517ULL;

// A rank count linear routing cannot serve. Thrown at Router construction, not at the first term: the
// alternative is a silent fallback to splitmix on some ranks and a deadlocked exchange.
class UnroutableGeometry : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline constexpr auto mix64(uint64_t x) noexcept -> uint64_t {
    x += 0x9E37'79B9'7F4A'7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58'476D'1CE4'E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D0'49BB'1331'11EBULL;
    return x ^ (x >> 31);
}

inline auto seed_from_env() -> uint64_t {
    return config::get().route_seed.value_or(kDefaultSeed);
}

// One 64-bit label per bit position (2 per mode). Deterministic from the seed alone, so every rank builds
// the same table with no communication -- the property find_rank's contract rests on.
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

// The full 64-bit image of a dense monomial, by walking the set bits. fingerprint_positions below is the
// same map over a packed row; Router::dest reads this one for the dense call sites (seeding, find_rank).
template <size_t NumBits>
[[nodiscard]] inline auto linear_hash(const monoprop::Bitset<NumBits> &bits) noexcept -> uint64_t {
    const auto &v = linear_basis<NumBits>();
    uint64_t h = 0;
    for (size_t i = bits.find_first(); i < NumBits; i = bits.find_next(i)) {
        h ^= v[i];
    }
    return h;
}

// The fingerprint of a term given as ascending positions: the same map as linear_hash, one label XOR per
// position, so a packed row never has to be expanded into a bitset to be routed or matched. `labels` is
// linear_basis<NumBits>().data(), bound once per gate by the caller so the per-term path never reaches the
// static-init guard. Everything the engine derives from a term's identity -- its owning slot, the slot of
// its partner under a generator, and the per-gate dedup key -- is a function of this one number:
//   fp(M ^ G) == fp(M) ^ fp(G)                                (GF(2)-linear)
// It is NOT injective (2*NumModes bits into 64), so equality of fingerprints is a prefilter and every
// match must be confirmed against the positions themselves.
template <typename PosT>
[[nodiscard]] [[gnu::always_inline]] inline auto fingerprint_positions(const uint64_t *labels,
                                                                       const PosT *pos,
                                                                       size_t k) noexcept -> uint64_t {
    uint64_t h = 0;
    for (size_t j = 0; j < k; ++j) {
        h ^= labels[static_cast<size_t>(pos[j])];
    }
    return h;
}

// GF(2) rank of a set of 64-bit vectors, by Gaussian elimination over the bit columns. The per-generator
// shifts must span at least linear_bits() dimensions or the reachable destination slots form a strict
// subspace and 2^bits - 2^rank slots stay empty. A balance failure, not a correctness one, so its caller
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
//
// Under linear routing the flat slot is read straight off the fingerprint:
//     rank = fp & (R - 1)                          all log2(R) rank bits, so the rank fanout is 1
//     part = (fp >> log2 R) & (S - 1)              when S is 2^k: the next log2(S) bits, ALSO linear
//          = mix64(fp) % S                         otherwise: full avalanche, balance only
//     flat = rank * S + part
// With S a power of two flat is the bit-concatenation rank<<log2(S) | part, i.e. flat itself is
// GF(2)-linear in the term: flat(M ^ G) == flat(M) ^ flat_shift(G), and a generator pairs every slot with
// exactly one peer slot, inside a rank as well as across ranks. With S not a power of two only the rank
// level has that structure and a generator's queries from one slot land on the S slots of one peer rank.
class Router final {
public:
    // The only way to a linear router; throws if `linear` and ranks is not 2^k. Templated on the width for
    // symmetry with the call sites (find_rank, make_router); no per-width table is bound any more.
    template <size_t NumModes>
    [[nodiscard]] static auto for_modes(size_t ranks, size_t partitions, bool linear) -> Router {
        return Router{ranks, partitions, linear};
    }

    // Today's routing: one flat world, full avalanche, no linear bits.
    static constexpr auto splitmix(size_t flat_world) noexcept -> Router { return Router{flat_world, 1, false}; }

    [[nodiscard]] constexpr auto ranks() const noexcept -> size_t { return ranks_; }
    [[nodiscard]] constexpr auto partitions() const noexcept -> size_t { return parts_; }
    [[nodiscard]] constexpr auto flat_world() const noexcept -> size_t { return flat_; }
    // False for a splitmix router AND for R == 1, which has no rank bit to take: both route densely.
    [[nodiscard]] constexpr auto is_linear() const noexcept -> bool { return linear_; }
    // The whole flat slot is a GF(2)-linear function of the term: linear routing with S a power of two.
    // Then flat_shift(G) is defined and every slot has exactly one peer slot per generator.
    [[nodiscard]] constexpr auto is_flat_linear() const noexcept -> bool { return linear_ && parts_pow2_; }
    // Rank bits read off the fingerprint: log2(R), or 0 when not linear. The span a generator set must
    // cover for every rank to be reachable.
    [[nodiscard]] constexpr auto linear_bits() const noexcept -> size_t {
        return linear_ ? static_cast<size_t>(std::countr_zero(ranks_)) : 0;
    }
    // Every bit the slot is linear in: log2(R) + log2(S) when flat-linear, else linear_bits().
    [[nodiscard]] constexpr auto flat_linear_bits() const noexcept -> size_t {
        return is_flat_linear() ? linear_bits() + parts_log2_ : linear_bits();
    }

    // The same number for a geometry alone: it IS the constructor, so the resolution and the
    // non-power-of-two throw cannot drift from the router's.
    [[nodiscard]] static auto bits_for(size_t ranks, bool linear) -> size_t {
        return Router{ranks, 1, linear}.linear_bits();
    }

    // Flat destination slot in [0, flat_world) for a dense monomial. Branch is on a member, so it is
    // perfectly predicted. The splitmix arm is bit-for-bit today's `hash % P`.
    template <size_t NumModes>
    [[nodiscard]] [[gnu::always_inline]] inline auto dest(const Monomial<NumModes> &mono) const noexcept -> size_t {
        if (!linear_) {
            return static_cast<size_t>(monomial_hash<NumModes>(mono) % flat_);
        }
        return dest_from_fingerprint(linear_hash<2 * NumModes>(mono));
    }

    // The same slot from the term's fingerprint (linear routing only; asserted). This is the emit path:
    // the partner's fingerprint is fp(M) ^ fp(G), so no dense form and no per-term hash exist there.
    [[nodiscard]] [[gnu::always_inline]] inline auto dest_from_fingerprint(uint64_t fp) const noexcept -> size_t {
        assert(linear_ && "dest_from_fingerprint on a splitmix router: the fingerprint is not its map");
        const size_t rank = static_cast<size_t>(fp & static_cast<uint64_t>(ranks_ - 1));
        if (parts_ == 1) {
            return rank;
        }
        if (parts_pow2_) {
            return (rank << parts_log2_) | static_cast<size_t>((fp >> linear_bits()) & parts_mask_);
        }
        return (rank * parts_) + static_cast<size_t>(mix64(fp) % parts_);
    }

    // The rank-level shift a generator induces: rank(M^G) == rank(M) ^ rank_shift(G). Zero for every G
    // when the router is not linear.
    template <size_t NumModes>
    [[nodiscard]] auto rank_shift(const Monomial<NumModes> &gen) const noexcept -> size_t {
        if (!linear_) {
            return 0;
        }
        return static_cast<size_t>(linear_hash<2 * NumModes>(gen) & static_cast<uint64_t>(ranks_ - 1));
    }

    // The flat-slot shift a generator induces when the whole slot is linear: flat(M^G) == flat(M) ^
    // flat_shift(G). nullopt when it is not (splitmix, R == 1, or S not a power of two), so a caller
    // cannot XOR a shift that does not exist.
    template <size_t NumModes>
    [[nodiscard]] auto flat_shift(const Monomial<NumModes> &gen) const noexcept -> std::optional<size_t> {
        if (!is_flat_linear()) {
            return std::nullopt;
        }
        return dest_from_fingerprint(linear_hash<2 * NumModes>(gen));
    }

private:
    // ranks x partitions == the flat world the destinations index.
    constexpr Router(size_t ranks, size_t partitions, bool linear)
        : ranks_(ranks == 0 ? 1 : ranks),
          parts_(partitions == 0 ? 1 : partitions),
          flat_(ranks_ * parts_),
          linear_(linear && ranks_ > 1), // R == 1 takes no rank bit, so it IS the dense case
          parts_pow2_(std::has_single_bit(parts_)),
          parts_mask_(parts_ - 1),
          parts_log2_(static_cast<size_t>(std::countr_zero(parts_))) {
        if (linear && !std::has_single_bit(ranks_)) {
            throw UnroutableGeometry(
                std::format("linear routing needs a power-of-two rank count, got {}. Launch 2^k ranks, or set "
                            "monoprop_ROUTING=splitmix to keep the dense all-to-all.",
                            ranks_));
        }
    }

    size_t ranks_;
    size_t parts_;
    size_t flat_;
    bool linear_;
    bool parts_pow2_;   // S is 2^k, so `% S` is a mask and `/ S` a shift
    size_t parts_mask_; // S - 1, and parts_log2_ == log2(S); both meaningless unless parts_pow2_
    size_t parts_log2_;
};

// The mode, before any geometry. Linear unless asked otherwise: measured at the production point it
// costs nothing on balance (rank occupancy max/mean 1.001 at R=128, all ranks used) and takes messages
// per rank per layer from 362,712 to 1,397, i.e. from proportional-to-R to flat.
inline auto linear_requested() -> bool {
    return config::get().routing_mode.value_or(config::RoutingMode::Linear) == config::RoutingMode::Linear;
}

// Resolved rank bits for a geometry, without a router: the replay transport gates on the number.
inline auto linear_bits_for(size_t ranks) -> size_t {
    return Router::bits_for(ranks, linear_requested());
}

template <size_t NumModes>
inline auto make_router(size_t ranks, size_t partitions) -> Router {
    return Router::for_modes<NumModes>(ranks, partitions, linear_requested());
}

} // namespace monoprop::routing
