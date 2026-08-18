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

#include "OpaqueIndirectHolder.h"

#include <utility>

namespace test_utils {

struct OpaqueValue {
    int value;
};

OpaqueIndirectHolder::OpaqueIndirectHolder(int value) : value_(std::in_place, value) {}
OpaqueIndirectHolder::OpaqueIndirectHolder(const OpaqueIndirectHolder &) = default;
auto OpaqueIndirectHolder::operator=(const OpaqueIndirectHolder &) -> OpaqueIndirectHolder & = default;
OpaqueIndirectHolder::OpaqueIndirectHolder(OpaqueIndirectHolder &&) noexcept = default;
auto OpaqueIndirectHolder::operator=(OpaqueIndirectHolder &&) noexcept -> OpaqueIndirectHolder & = default;
OpaqueIndirectHolder::~OpaqueIndirectHolder() = default;

auto OpaqueIndirectHolder::value() const -> int {
    return value_->value;
}

} // namespace test_utils
