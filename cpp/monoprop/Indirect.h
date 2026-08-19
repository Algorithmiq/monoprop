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

#include <memory>
#include <utility>

namespace monoprop {

/// An allocator-free C++23 subset of C++26 `std::indirect`.
//
// The pointee has value semantics: copying copy-constructs a distinct T, while moving transfers the
// allocation and leaves the source valueless. T may be incomplete where indirect<T> is named, but must
// be complete wherever an operation constructs, copies, or destroys the pointee.
//
// This subset deliberately omits allocators, comparisons, hashing, value assignment, and constructors
// that adopt an existing pointer. Copy assignment replaces the allocation rather than preserving its
// address, so T need only be copy-constructible, not copy-assignable.
template <typename T>
class indirect {
public:
    /// The owned value type.
    using value_type = T;

    /// Construct a value-initialized `T`.
    explicit indirect() : ptr_(std::make_unique<T>()) {}

    /// Construct `T` directly from `args`.
    template <typename... Args>
    explicit indirect(std::in_place_t, Args &&...args) : ptr_(std::make_unique<T>(std::forward<Args>(args)...)) {}

    /// Copy-construct an independent `T`, or preserve a moved-from state.
    indirect(const indirect &other) : ptr_(other.ptr_ ? std::make_unique<T>(*other.ptr_) : nullptr) {}

    /// Replace the owned value with an independent copy of `other`.
    auto operator=(const indirect &other) -> indirect & {
        if (this != std::addressof(other)) {
            indirect replacement(other);
            ptr_.swap(replacement.ptr_);
        }
        return *this;
    }

    /// Transfer ownership and leave `other` valueless.
    indirect(indirect &&) noexcept = default;

    /// Transfer ownership and leave `other` valueless.
    auto operator=(indirect &&) noexcept -> indirect & = default;

    /// Destroy the owned value, if present.
    ~indirect() = default;

    /// Access the owned value; the object must not be valueless.
    auto operator*() noexcept -> T & { return *ptr_; }

    /// Access the owned value; the object must not be valueless.
    auto operator*() const noexcept -> const T & { return *ptr_; }

    /// Access the owned value; the object must not be valueless.
    auto operator->() noexcept -> T * { return ptr_.get(); }

    /// Access the owned value; the object must not be valueless.
    auto operator->() const noexcept -> const T * { return ptr_.get(); }

    /// Return whether ownership was transferred from this object.
    [[nodiscard]] auto valueless_after_move() const noexcept -> bool { return ptr_ == nullptr; }

private:
    std::unique_ptr<T> ptr_;
};

} // namespace monoprop
