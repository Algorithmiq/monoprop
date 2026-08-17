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

#include <concepts>
#include <memory>
#include <utility>

namespace monoprop {

/// A `unique_ptr` that copies its pointee instead of refusing to copy.
// Exclusive ownership with value semantics. It exists so that a class holding a heap member -- because
// the member is non-copyable, non-movable, or incomplete at that point -- needs no hand-written copy
// constructor or destructor, and so keeps its implicit move operations: the Rule of Zero. A hand-written
// copy constructor has to name every other member too, and silently default-initializes any member added
// later.
//
// Copying prefers `T::clone()` when the pointee declares one, and falls back to `T`'s copy constructor.
// A type that owns internal indices usually offers clone() precisely because it forbids copy construction.
//
// `T` may be incomplete where `value_ptr<T>` is named: as with `unique_ptr`, each member body that needs
// `T` complete is instantiated only at its own point of use.
template <typename T>
class value_ptr {
public:
    value_ptr() = default;

    explicit value_ptr(std::unique_ptr<T> owned) noexcept : ptr_(std::move(owned)) {}

    value_ptr(const value_ptr &other) : ptr_(clone_of_(other.ptr_)) {}

    auto operator=(const value_ptr &other) -> value_ptr & {
        // Clone and replace, never T's assignment: T need not be assignable, and taking the clone before
        // the old pointee is released makes self-assignment safe without a guard.
        ptr_ = clone_of_(other.ptr_);
        return *this;
    }

    value_ptr(value_ptr &&) noexcept = default;
    auto operator=(value_ptr &&) noexcept -> value_ptr & = default;
    ~value_ptr() = default;

    // const propagates to the pointee, unlike unique_ptr's: the pointee is a value member here, not a
    // pointer the object happens to hold.
    auto operator*() -> T & { return *ptr_; }
    auto operator*() const -> const T & { return *ptr_; }
    auto operator->() -> T * { return ptr_.get(); }
    auto operator->() const -> const T * { return ptr_.get(); }
    auto get() -> T * { return ptr_.get(); }
    auto get() const -> const T * { return ptr_.get(); }

    explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

private:
    static auto clone_of_(const std::unique_ptr<T> &src) -> std::unique_ptr<T> {
        if (!src) {
            return nullptr;
        }
        if constexpr (requires {
                          { src->clone() } -> std::convertible_to<std::unique_ptr<T>>;
                      }) {
            return src->clone();
        }
        else {
            return std::make_unique<T>(*src);
        }
    }

    std::unique_ptr<T> ptr_;
};

/// Construct a `value_ptr<T>` pointee in place, as `make_unique` does.
template <typename T, typename... Args>
auto make_value(Args &&...args) -> value_ptr<T> {
    return value_ptr<T>(std::make_unique<T>(std::forward<Args>(args)...));
}

} // namespace monoprop
