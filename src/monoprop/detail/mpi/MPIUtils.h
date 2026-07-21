#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop::mpi_detail {

static_assert(sizeof(size_t) == sizeof(uint64_t), "MPI serialization assumes 64-bit size_t");

/// Number of size_t words per Monomial<NumModes>.
template <size_t NumModes>
inline constexpr size_t kWords = Monomial<NumModes>::num_words();

template <size_t NumModes>
inline auto append_majorana_words(const Monomial<NumModes> &maj, VecZ &buffer) -> void {
    const auto *src = maj.data();
    for (size_t i = 0; i < kWords<NumModes>; ++i)
        buffer.push_back(src[i]);
}

template <size_t NumModes>
inline auto read_majorana_from_words(const VecZ &buffer, size_t start) -> Monomial<NumModes> {
    Monomial<NumModes> maj;
    std::memcpy(maj.data(), &buffer[start], kWords<NumModes> * sizeof(uint64_t));
    return maj;
}

} // namespace monoprop::mpi_detail

namespace monoprop {

/**
 * @brief MPI utility functions for distributed MPOperator operations
 */
// Deterministic owner rank for a term: hash(maj) % n_ranks. Stateless and identical on every rank,
// so all ranks agree on which rank owns any given Majorana term without communication.
template <size_t NumModes>
auto find_rank(const Monomial<NumModes> &maj, const size_t n_ranks) -> size_t {
    if (n_ranks == 0) {
        return 0;
    }
    return monomial_hash<NumModes>(maj) % n_ranks;
}

} // namespace monoprop
