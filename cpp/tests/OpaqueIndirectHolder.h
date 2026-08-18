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

#include "monoprop/Indirect.h"

namespace test_utils {

struct OpaqueValue;

class OpaqueIndirectHolder {
public:
    /// Construct an opaque value.
    explicit OpaqueIndirectHolder(int value);

    /// Copy the opaque value.
    OpaqueIndirectHolder(const OpaqueIndirectHolder &);

    /// Replace the opaque value with a copy.
    auto operator=(const OpaqueIndirectHolder &) -> OpaqueIndirectHolder &;

    /// Move the opaque value.
    OpaqueIndirectHolder(OpaqueIndirectHolder &&) noexcept;

    /// Replace the opaque value by moving it.
    auto operator=(OpaqueIndirectHolder &&) noexcept -> OpaqueIndirectHolder &;

    /// Destroy the opaque value where its type is complete.
    ~OpaqueIndirectHolder();

    /// Return the stored value.
    auto value() const -> int;

private:
    monoprop::indirect<OpaqueValue> value_;
};

} // namespace test_utils
