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
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/EnvConfig.h"

// The one owner function: Scan.h emits queries by it and MonomialPropagator seeds by it, so a second
// copy would split ownership silently.
//
//     part = monomial_hash(M) % S     splitmix: partition fanout is free, only balance matters
//     rank = linear_hash(M) & (R - 1) GF(2)-linear: rank(M ^ G) == rank(M) ^ rank(G), so fanout 1
//     flat = rank * S + part
//
// Linear takes all log2(R) rank bits or none; R == 1 and splitmix are `hash % P` bit for bit. Linear
// needs a power-of-two R (UnroutableGeometry). Derivation: docs/content/docs/features/parallelism.mdx.
//
//   monoprop_ROUTING     linear (default) | splitmix
//   monoprop_ROUTE_SEED  uint64 seed for the linear basis (default kDefaultSeed)

namespace monoprop::routing {

inline constexpr uint64_t kDefaultSeed = 0x5DEE'CE66'D0C6'2517ULL;

// Thrown at Router construction: a silent splitmix fallback on some ranks would deadlock the exchange.
class UnroutableGeometry : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Ranks resolved different routing configurations; left unchecked that hangs rather than misanswers.
class RoutingDisagreement : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

constexpr auto mix64(uint64_t x) noexcept -> uint64_t {
    x += 0x9E37'79B9'7F4A'7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58'476D'1CE4'E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D0'49BB'1331'11EBULL;
    return x ^ (x >> 31);
}

// Bit-columns linear_basis is drawn full rank over; a Router reads the low log2(R) of them.
inline constexpr size_t kLinearColumns = 64;

// GF(2) rank of a set of 64-bit vectors. Used for the basis draw and report_routing_coverage_.
[[nodiscard]] inline auto gf2_rank(std::vector<uint64_t> vectors) noexcept -> size_t {
    size_t rank = 0;
    for (size_t bit = 0; bit < 64; ++bit) {
        if (rank == vectors.size()) {
            break;
        }
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

inline auto seed_from_env() -> uint64_t {
    return config::get().route_seed.value_or(kDefaultSeed);
}

// One 64-bit vector per Majorana mode, deterministic from the seed so every rank builds the same table.
// Redrawn until the low kLinearColumns columns are independent: then every prefix is too, so every
// Router's rank fibres are equal-sized. A deficient draw leaves ranks unreachable.
template <size_t NumBits>
inline auto linear_basis() -> const std::array<uint64_t, NumBits> & {
    static const auto table = [] {
        constexpr size_t num_cols = std::min(kLinearColumns, NumBits);
        constexpr uint64_t col_mask = num_cols == 64 ? ~uint64_t{0} : (uint64_t{1} << num_cols) - 1;
        std::array<uint64_t, NumBits> v{};
        uint64_t draw = seed_from_env();
        // Bounded so a static initialiser cannot hang; each draw is full rank with probability >= 0.288.
        for (int attempt = 0; attempt < 1024; ++attempt, draw = mix64(draw)) {
            std::vector<uint64_t> cols(NumBits);
            for (size_t i = 0; i < NumBits; ++i) {
                v[i] = mix64(mix64(draw) + (static_cast<uint64_t>(i) * 0x9E37'79B9'7F4A'7C15ULL));
                cols[i] = v[i] & col_mask;
            }
            if (gf2_rank(std::move(cols)) == num_cols) {
                return v;
            }
        }
        throw UnroutableGeometry(std::format("no full-rank linear routing basis after 1024 draws from seed {}. "
                                             "Pick another monoprop_ROUTE_SEED, or set monoprop_ROUTING=splitmix.",
                                             seed_from_env()));
    }();
    return table;
}

// XOR of the basis vectors over the set bits: the definition Router's rank bits are the low end of.
template <size_t NumBits>
[[nodiscard]] inline auto linear_hash(const monoprop::Bitset<NumBits> &bits) noexcept -> uint64_t {
    const auto &v = linear_basis<NumBits>();
    uint64_t h = 0;
    for (size_t i = bits.find_first(); i < NumBits; i = bits.find_next(i)) {
        h ^= v[i];
    }
    return h;
}

// Trivially copyable and cheap to build; hold one per build_layer call rather than per term.
class Router final {
public:
    // The only way to a linear router. Binds the basis for this width, keeping its static-init guard off
    // dest(). Throws unless every rank is reachable: R == 2^d with d <= the full-rank column count.
    template <size_t NumModes>
    [[nodiscard]] static auto for_modes(size_t ranks, size_t partitions, bool linear) -> Router {
        constexpr size_t max_bits = std::min(kLinearColumns, 2 * NumModes);
        Router r{ranks, partitions, linear};
        if (r.linear_bits() > max_bits) {
            throw UnroutableGeometry(
                std::format("linear routing over {} modes reaches at most 2^{} ranks, got {}. Launch fewer "
                            "ranks, or set monoprop_ROUTING=splitmix to keep the dense all-to-all.",
                            NumModes,
                            max_bits,
                            ranks));
        }
        r.basis_ = linear_basis<2 * NumModes>().data();
        r.basis_bits_ = 2 * NumModes;
        return r;
    }

    // One flat world, full avalanche, no linear bits.
    static constexpr auto splitmix(size_t flat_world) -> Router { return Router{flat_world, 1, false}; }

    [[nodiscard]] constexpr auto ranks() const noexcept -> size_t { return ranks_; }
    [[nodiscard]] constexpr auto partitions() const noexcept -> size_t { return parts_; }
    [[nodiscard]] constexpr auto flat_world() const noexcept -> size_t { return flat_; }
    // False for a splitmix router AND for R == 1, which has no rank bit to take: both route densely.
    [[nodiscard]] constexpr auto is_linear() const noexcept -> bool { return linear_; }
    // log2(R), or 0 when not linear: the span a generator set needs to reach every rank.
    [[nodiscard]] constexpr auto linear_bits() const noexcept -> size_t {
        return linear_ ? static_cast<size_t>(std::countr_zero(ranks_)) : 0;
    }

    // Flat destination slot in [0, flat_world).
    template <size_t NumModes>
    [[nodiscard]] [[gnu::always_inline]] auto dest(const Monomial<NumModes> &mono) const noexcept -> size_t {
        if (!linear_) {
            return static_cast<size_t>(monomial_hash<NumModes>(mono) % flat_);
        }
        const auto rank = static_cast<size_t>(linear_low_<NumModes>(mono));
        if (parts_ == 1) {
            return rank; // S == 1 needs no hash
        }
        return (rank * parts_) + part_of_(monomial_hash<NumModes>(mono));
    }

    // dest(M ^ G) for a term M that `my_flat` owns, where `shift` == rank_shift(G). Wrong arguments move
    // ownership silently.
    template <size_t NumModes>
    [[nodiscard]] [[gnu::always_inline]] auto dest_from_shift(const Monomial<NumModes> &mono,
                                                              size_t my_flat,
                                                              size_t shift) const noexcept -> size_t {
        if (!linear_) {
            return dest<NumModes>(mono);
        }
        const size_t rank = rank_of_slot_(my_flat) ^ shift;
        if (parts_ == 1) {
            return rank;
        }
        return (rank * parts_) + part_of_(monomial_hash<NumModes>(mono));
    }

    // The rank-level shift a generator induces: rank(M^G) == rank(M) ^ shift(G). Zero when not linear.
    template <size_t NumModes>
    [[nodiscard]] auto rank_shift(const Monomial<NumModes> &gen) const noexcept -> size_t {
        return static_cast<size_t>(linear_low_<NumModes>(gen));
    }

private:
    // Private: a linear router must go through for_modes, which binds the basis.
    constexpr Router(size_t ranks, size_t partitions, bool linear)
        : ranks_(ranks == 0 ? 1 : ranks),
          parts_(partitions == 0 ? 1 : partitions),
          flat_(ranks_ * parts_),
          linear_(linear && ranks_ > 1), // R == 1 has no rank bit: dense by construction
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

    // parts_ is a runtime value, so strength-reduce the power-of-two case by hand.
    [[nodiscard]] [[gnu::always_inline]] auto part_of_(uint64_t q) const noexcept -> size_t {
        return static_cast<size_t>(parts_pow2_ ? (q & parts_mask_) : (q % parts_));
    }

    [[nodiscard]] [[gnu::always_inline]] auto rank_of_slot_(size_t flat_slot) const noexcept -> size_t {
        return parts_pow2_ ? (flat_slot >> parts_log2_) : (flat_slot / parts_);
    }

    // linear_hash(M) & (R - 1). Runs at seeding and once per generator, never per query.
    template <size_t NumModes>
    [[nodiscard]] auto linear_low_(const Monomial<NumModes> &m) const noexcept -> uint64_t {
        if (!linear_) {
            return 0;
        }
        assert(basis_ != nullptr && basis_bits_ == 2 * NumModes); // bound at a different width
        uint64_t h = 0;
        for (size_t i = m.find_first(); i < 2 * NumModes; i = m.find_next(i)) {
            h ^= basis_[i];
        }
        return h & (ranks_ - 1);
    }

    size_t ranks_;
    size_t parts_;
    size_t flat_;
    bool linear_;
    bool parts_pow2_;   // S is 2^k, so `% S` is a mask and `/ S` a shift
    size_t parts_mask_; // S - 1, and parts_log2_ == log2(S); both meaningless unless parts_pow2_
    size_t parts_log2_;
    const uint64_t *basis_ = nullptr; // linear_basis<basis_bits_>(), bound by for_modes
    size_t basis_bits_ = 0;
};

// The mode, before any geometry. Linear unless asked otherwise.
inline auto linear_requested() -> bool {
    return config::get().routing_mode.value_or(config::RoutingMode::Linear) == config::RoutingMode::Linear;
}

// What must match across ranks for the transports to pair up; raw values, so agreement is exact.
struct Config {
    uint64_t linear = 0;
    uint64_t partitions = 1;
    uint64_t seed = 0;

    static auto from_env(size_t partitions) -> Config {
        return {.linear = static_cast<uint64_t>(linear_requested()),
                .partitions = static_cast<uint64_t>(partitions),
                .seed = seed_from_env()};
    }

    [[nodiscard]] auto describe() const -> std::string {
        return std::format("linear={}, partitions={}, seed={}", linear, partitions, seed);
    }
};

// Whether an AGREED config routes point-to-point. Never throws, since the replay path asks from inside a
// collective; an unroutable geometry stays dense, and Router reports it where it can.
[[nodiscard]] inline auto routes_pairwise(const Config &agreed, size_t ranks) -> bool {
    return agreed.linear != 0 && ranks > 1 && std::has_single_bit(ranks);
}

template <size_t NumModes>
inline auto make_router(size_t ranks, size_t partitions) -> Router {
    return Router::for_modes<NumModes>(ranks, partitions, linear_requested());
}

} // namespace monoprop::routing
