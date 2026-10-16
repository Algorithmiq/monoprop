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

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef monoprop_ENABLE_MPI
#include <mpi.h>
#else
// Fallback MPI types for non-MPI builds (single process).
using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr MPI_Comm MPI_COMM_SELF = 0;
#endif

namespace monoprop::mpi {

// The ordinary communicator handle: a real MPI communicator, or the MPI-off stub (one process). Trivially copyable,
// passed by value. The implicit MPI_Comm constructor is deliberate; read `.mpi` explicitly where a raw communicator is
// required. It carries no shard geometry: a sharded root derives its (P, T) owners from this communicator's size and
// its own captured budget.
struct Comm {
    MPI_Comm mpi = MPI_COMM_SELF;

    constexpr Comm() = default;
    constexpr Comm(MPI_Comm c) : mpi(c) {} // NOLINT(google-explicit-constructor): implicit on purpose (see above)
};

// A window-relative index, typed apart from a flat slot: mixing the two stays in bounds but addresses
// the wrong peer whenever the window does not start at 0.
struct WindowIndex {
    size_t value = 0;

    constexpr WindowIndex() = default;
    explicit constexpr WindowIndex(size_t v) noexcept : value(v) {}
};

// The contiguous run of flat slots a round can reach. Slots are rank-major, so a sparse plan's one peer
// is that rank's T contiguous slots and dense is the count == P * T case of the same run. See PeerPlan::window.
struct SlotWindow {
    size_t base = 0;  // first reachable flat slot
    size_t count = 0; // slots in the run

    constexpr auto operator==(const SlotWindow &) const -> bool = default;

    [[nodiscard]] constexpr auto stop() const -> size_t { return base + count; }
    [[nodiscard]] constexpr auto contains(size_t slot) const -> bool { return slot >= base && slot < stop(); }
    // Asserts membership, so an outside slot cannot alias another's entry.
    [[nodiscard]] constexpr auto index(size_t slot) const -> WindowIndex {
        assert(contains(slot) && "flat slot outside the window it is being re-based into");
        return WindowIndex{slot - base};
    }
    [[nodiscard]] constexpr auto slot(WindowIndex i) const -> size_t {
        assert(i.value < count);
        return base + i.value;
    }
    // Every WindowIndex of the run, in slot order.
    [[nodiscard]] constexpr auto indices() const {
        return std::views::iota(size_t{0}, count) | std::views::transform([](size_t k) { return WindowIndex{k}; });
    }
};

// A vector over a SlotWindow: at_slot() takes a flat slot, operator[] a WindowIndex, so a flat slot used
// as a raw index does not compile.
template <typename T>
class WindowVec {
public:
    using value_type = T;

    WindowVec() = default;
    explicit WindowVec(SlotWindow w) : win_(w), v_(w.count) {}

    auto reset(SlotWindow w) -> void {
        win_ = w;
        v_.assign(w.count, T{});
    }

    [[nodiscard]] auto window() const -> SlotWindow { return win_; }
    [[nodiscard]] auto size() const -> size_t { return v_.size(); }

    [[nodiscard]] auto operator[](WindowIndex i) -> T & { return v_[i.value]; }
    [[nodiscard]] auto operator[](WindowIndex i) const -> const T & { return v_[i.value]; }
    [[nodiscard]] auto at_slot(size_t slot) -> T & { return v_[win_.index(slot).value]; }
    [[nodiscard]] auto at_slot(size_t slot) const -> const T & { return v_[win_.index(slot).value]; }

    [[nodiscard]] auto begin() { return v_.begin(); }
    [[nodiscard]] auto end() { return v_.end(); }
    [[nodiscard]] auto begin() const { return v_.begin(); }
    [[nodiscard]] auto end() const { return v_.end(); }

private:
    SlotWindow win_{};
    std::vector<T> v_;
};

/*!
 * \brief Read-only span views of every block of `blocks`, over the same window.
 *
 * Copies no payload: view `wi` aliases `blocks[wi]`, so `blocks` must outlive the views and must not be resized
 * while they are read. An empty block yields an empty view, and a window with a nonzero base keeps its base.
 *
 * \param blocks One owning block per window slot.
 * \return One view per window slot, in slot order.
 */
template <typename T>
[[nodiscard]] auto views_of(const WindowVec<std::vector<T>> &blocks) -> WindowVec<std::span<const T>> {
    const SlotWindow w = blocks.window();
    WindowVec<std::span<const T>> views(w);
    for (const auto wi : w.indices()) {
        views[wi] = std::span<const T>(blocks[wi]);
    }
    return views;
}

// Which destination ranks a round can touch. Dense (the default) is every rank and the collective path;
// sparse is the single peer `me ^ shift` that linear routing implies. XOR is an involution, so every
// rank derives the same pairing without communicating.
//
// A wrong `shift` fails two ways: ranks that disagree deadlock, and ranks that agree on the same wrong
// value silently drop blocks outside the peer set.
struct PeerPlan {
    bool sparse = false;
    int shift = 0;

    [[nodiscard]] constexpr auto dense() const -> bool { return !sparse; }
    [[nodiscard]] constexpr auto count(int ranks) const -> int { return sparse ? 1 : ranks; }
    // `k` indexes the peer set, which is a singleton when sparse.
    [[nodiscard]] constexpr auto peer(int me, int k) const -> int { return sparse ? (me ^ shift) : k; }
    [[nodiscard]] constexpr auto contains(int me, int b) const -> bool { return !sparse || b == (me ^ shift); }
    // The flat slots reachable from `me_flat` in a `ranks` x `parts` world (`parts` owners per rank): the peer rank's,
    // or all.
    [[nodiscard]] constexpr auto window(size_t me_flat, size_t ranks, size_t parts) const -> SlotWindow {
        const size_t peer_rank = sparse ? ((me_flat / parts) ^ static_cast<size_t>(shift)) : 0;
        return SlotWindow{.base = peer_rank * parts,
                          .count = static_cast<size_t>(count(static_cast<int>(ranks))) * parts};
    }

    // `me ^ shift` is used as an index and an MPI rank unchecked. Power-of-two `ranks` is what makes
    // `shift < ranks` keep it in range.
    [[nodiscard]] constexpr auto routable(int ranks) const -> bool {
        return dense()
               || (ranks > 0 && std::has_single_bit(static_cast<unsigned>(ranks)) && shift >= 0 && shift < ranks);
    }
};

// Called by every sparse entry point before it indexes or posts; PeerPlan is an aggregate, so no ctor can.
inline auto require_routable(PeerPlan plan, int ranks) -> void {
    if (!plan.routable(ranks)) {
        throw std::invalid_argument(
            "sparse peer plan is not routable: it needs a power-of-two rank count and 0 <= shift < ranks");
    }
}

// The typed buffers and layout of one variable all-to-all over a communicator (post_flat_alltoallv). `send`,
// `send_counts` and `send_displs` are non-owning views into caller memory; MPI reads the send buffer, and writes the
// receive buffer, until the posted transfer completes, so neither may move, be freed or be modified before then.
// Counts and displacements are in elements of `T`, one per rank.
template <typename T>
struct FlatAlltoallvArgs {
    const T *send = nullptr;
    const int *send_counts = nullptr; // [P] elements sent to each destination
    const int *send_displs = nullptr; // [P] element offsets into `send`
    T *recv = nullptr;                // caller-owned, already sized for recv_counts
    const int *recv_counts = nullptr; // [P] input: the senders' transpose, same contract as MPI_Alltoallv
    const int *recv_displs = nullptr; // [P] element offsets into `recv`
};

} // namespace monoprop::mpi
