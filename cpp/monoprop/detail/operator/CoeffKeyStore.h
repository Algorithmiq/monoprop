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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "monoprop/DefaultInitAllocator.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

//! One coefficient out of a cell, by byte address. The cell stride makes the double 4-byte aligned.
[[nodiscard]] inline auto load_coeff(const std::byte *p) noexcept -> double {
    double v = 0.0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
//! The write side of load_coeff().
inline auto store_coeff(std::byte *p, double v) noexcept -> void {
    std::memcpy(p, &v, sizeof(v));
}

//! One join key out of a cell, by byte address.
[[nodiscard]] inline auto load_row_key(const std::byte *p) noexcept -> uint32_t {
    uint32_t k = 0;
    std::memcpy(&k, p, sizeof(k));
    return k;
}
//! The write side of load_row_key().
inline auto store_row_key(std::byte *p, uint32_t k) noexcept -> void {
    std::memcpy(p, &k, sizeof(k));
}

/*! @brief A dense per-row coefficient array as the hot loops address it: base, byte stride, length.
 *
 * Two layouts reach the scan through this one view -- a plain std::vector<double> (stride 8: the
 * Schrödinger live state and graph mode's private copy) and CoeffKeyStore's packed cells (stride 12).
 * The stride is a runtime value rather than a template parameter so fused_find_and_collect keeps one
 * instantiation per (NumModes, Algebra); the multiply sits on an address whose load already dominates.
 */
struct CoeffSpan {
    const std::byte *base = nullptr; //!< cell 0's coefficient
    size_t stride = sizeof(double);  //!< bytes between consecutive coefficients
    size_t n = 0;                    //!< addressable coefficients

    CoeffSpan() = default;
    CoeffSpan(const std::byte *base_, size_t stride_, size_t n_) noexcept : base(base_), stride(stride_), n(n_) {}
    //! The contiguous case: a coefficient vector with no key beside it.
    explicit CoeffSpan(const VecD &v) noexcept : base(reinterpret_cast<const std::byte *>(v.data())), n(v.size()) {}

    [[nodiscard]] auto size() const noexcept -> size_t { return n; }
    [[nodiscard]] auto empty() const noexcept -> bool { return n == 0; }
    [[nodiscard]] auto operator[](size_t i) const noexcept -> double { return load_coeff(base + (i * stride)); }
    //! Rows past the array read as 0: the scan runs over a store the picture's vector may not cover yet.
    [[nodiscard]] auto at_or_zero(size_t i) const noexcept -> double { return i < n ? (*this)[i] : 0.0; }
    [[nodiscard]] auto to_vector() const -> VecD {
        VecD out(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = (*this)[i];
        }
        return out;
    }
};

//! CoeffSpan's writable twin; the fused cos sweep and the contract apply both write through it.
struct MutCoeffSpan {
    std::byte *base = nullptr;
    size_t stride = sizeof(double);
    size_t n = 0;

    MutCoeffSpan() = default;
    MutCoeffSpan(std::byte *base_, size_t stride_, size_t n_) noexcept : base(base_), stride(stride_), n(n_) {}
    explicit MutCoeffSpan(VecD &v) noexcept : base(reinterpret_cast<std::byte *>(v.data())), n(v.size()) {}

    [[nodiscard]] auto size() const noexcept -> size_t { return n; }
    [[nodiscard]] auto empty() const noexcept -> bool { return n == 0; }
    [[nodiscard]] auto operator[](size_t i) const noexcept -> double { return load_coeff(base + (i * stride)); }
    auto set(size_t i, double v) const noexcept -> void { store_coeff(base + (i * stride), v); }
    //! Read-only view of the same cells.
    [[nodiscard]] auto as_const() const noexcept -> CoeffSpan { return CoeffSpan{base, stride, n}; }
};

/*! @brief The operator's per-row cells: each row's 4-byte join key, and its coefficient beside it.
 *
 * The join reads a row's key for every anticommuting row of every gate (BucketJoin::stage_rows) and the
 * scan reads that row's coefficient in the pass just before. Held as two arrays those are two cache
 * lines per row and two DRAM streams -- 28.1 M of the arm's 114.8 M LL read misses at 2.4 M terms were
 * the key stream alone (scratch/hashfree/diag/residual-2x-60254b20.md §6). Held in one cell they are one
 * line, which is the whole point of this container.
 *
 * Two layouts, because the coefficient is not always there: build_graph never materializes one, and
 * forcing an 8-byte hole on it would cost the graph path 8 B/term.
 *
 *   packed (has_coeffs)  [double coeff][uint32 key]   stride 12, key at +8
 *   keys only            [uint32 key]                 stride  4, key at +0
 *
 * Packing to 12 is deliberate: 8 + 4 is exactly what the two separate arrays cost, so B/term is
 * unchanged. The price is that the double is only 4-byte aligned (loaded and stored through memcpy,
 * one movsd on x86-64) and that one cell in eight straddles a cache line.
 *
 * enable_coeffs() promotes keys-only to packed and nothing demotes: a picture that has coefficients
 * keeps them for the operator's life.
 */
class CoeffKeyStore {
public:
    static constexpr size_t kKeyBytes = sizeof(uint32_t);
    static constexpr size_t kCoeffBytes = sizeof(double);
    static constexpr size_t kKeyOnlyStride = kKeyBytes;
    static constexpr size_t kPackedStride = kCoeffBytes + kKeyBytes;

    //! A hoisted key reader: the join's row-side pass reads this and nothing else per row.
    struct KeyReader {
        const std::byte *base = nullptr;
        size_t stride = kKeyOnlyStride;
        [[nodiscard]] auto operator()(size_t i) const noexcept -> uint32_t { return load_row_key(base + (i * stride)); }
    };

    [[nodiscard]] auto size() const noexcept -> size_t { return size_; }
    [[nodiscard]] auto capacity() const noexcept -> size_t { return buf_.capacity() / stride_; }
    [[nodiscard]] auto stride() const noexcept -> size_t { return stride_; }
    [[nodiscard]] auto has_coeffs() const noexcept -> bool { return has_coeffs_; }

    [[nodiscard]] auto key(size_t i) const noexcept -> uint32_t { return load_row_key(cell(i) + key_off_); }
    auto set_key(size_t i, uint32_t k) noexcept -> void { store_row_key(cell(i) + key_off_, k); }
    [[nodiscard]] auto key_reader() const noexcept -> KeyReader { return KeyReader{buf_.data() + key_off_, stride_}; }

    //! Valid only once enable_coeffs() has run; a keys-only cell has no coefficient field at all.
    [[nodiscard]] auto coeff(size_t i) const noexcept -> double {
        assert(has_coeffs_ && "coefficient read on a keys-only cell store");
        return load_coeff(cell(i));
    }
    auto set_coeff(size_t i, double v) noexcept -> void {
        assert(has_coeffs_ && "coefficient write on a keys-only cell store");
        store_coeff(cell(i), v);
    }

    [[nodiscard]] auto coeff_span() const noexcept -> CoeffSpan {
        return has_coeffs_ ? CoeffSpan{buf_.data(), stride_, size_} : CoeffSpan{};
    }
    [[nodiscard]] auto mut_coeff_span() noexcept -> MutCoeffSpan {
        return has_coeffs_ ? MutCoeffSpan{buf_.data(), stride_, size_} : MutCoeffSpan{};
    }

    auto reserve(size_t n) -> void { buf_.reserve(n * stride_); }

    // Grown cells hold an indeterminate key (every writer sets it before any read, like the rows) but a
    // ZERO coefficient: a minted row's slot is read by the contract apply's insert arm before anything
    // writes it, which is the guarantee op_coeffs.resize(size(), 0.0) used to give.
    auto resize(size_t n) -> void {
        const size_t old = size_;
        buf_.resize(n * stride_);
        if (has_coeffs_ && n > old) {
            std::memset(buf_.data() + (old * stride_), 0, (n - old) * stride_);
        }
        size_ = n;
    }

    auto shrink_to_fit() -> void { buf_.shrink_to_fit(); }

    // Adds the coefficient field, keeping every key. All coefficients start at 0.0, which is what the
    // lazy op_coeffs.resize(size(), 0.0) produced on its first materialization.
    auto enable_coeffs() -> void {
        if (has_coeffs_) {
            return;
        }
        const size_t want = std::max(capacity(), size_);
        DefaultInitVector<std::byte> next;
        next.reserve(want * kPackedStride);
        next.resize(size_ * kPackedStride);
        std::memset(next.data(), 0, next.size());
        for (size_t i = 0; i < size_; ++i) {
            store_row_key(next.data() + (i * kPackedStride) + kCoeffBytes,
                          load_row_key(buf_.data() + (i * kKeyOnlyStride)));
        }
        buf_ = std::move(next);
        stride_ = kPackedStride;
        key_off_ = kCoeffBytes;
        has_coeffs_ = true;
    }

    //! Overwrite every coefficient from a contiguous vector; rows past its end read back 0.0.
    auto assign_coeffs(std::span<const double> v) -> void {
        enable_coeffs();
        const size_t m = std::min(v.size(), size_);
        for (size_t i = 0; i < m; ++i) {
            store_coeff(cell(i), v[i]);
        }
        for (size_t i = m; i < size_; ++i) {
            store_coeff(cell(i), 0.0);
        }
    }

    //! The coefficients as their own contiguous array, for the callers that hand one to a caller of theirs.
    [[nodiscard]] auto coeffs_to_vector() const -> VecD {
        VecD out(size_, 0.0);
        if (has_coeffs_) {
            for (size_t i = 0; i < size_; ++i) {
                out[i] = load_coeff(cell(i));
            }
        }
        return out;
    }

    // Ledger. The two fields are priced separately so the memory breakdown keeps reporting a key line
    // and a coefficient line; together they are the buffer.
    [[nodiscard]] auto key_bytes() const noexcept -> size_t { return capacity() * kKeyBytes; }
    [[nodiscard]] auto coeff_bytes() const noexcept -> size_t { return has_coeffs_ ? capacity() * kCoeffBytes : 0; }
    //! The part of the buffer that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const noexcept -> size_t { return (capacity() - size_) * stride_; }

private:
    [[nodiscard]] auto cell(size_t i) const noexcept -> const std::byte * { return buf_.data() + (i * stride_); }
    [[nodiscard]] auto cell(size_t i) noexcept -> std::byte * { return buf_.data() + (i * stride_); }

    DefaultInitVector<std::byte> buf_ = {};
    size_t size_ = 0;
    size_t stride_ = kKeyOnlyStride;
    size_t key_off_ = 0;
    bool has_coeffs_ = false;
};

} // namespace monoprop::detail
