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

#include <boost/test/unit_test.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "OpaqueIndirectHolder.h"
#include "monoprop/Indirect.h"

using monoprop::indirect;

namespace {

struct Value {
    int number = 3;
    std::string label = "default";

    Value() = default;
    Value(int number, std::string label) : number(number), label(std::move(label)) {}
};

struct ThrowOnCopy {
    static inline bool should_throw = false;
    int value;

    explicit ThrowOnCopy(int value) : value(value) {}
    ThrowOnCopy(const ThrowOnCopy &other) : value(other.value) {
        if (should_throw) {
            throw std::runtime_error("copy failed");
        }
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(indirect_default_and_in_place_construction_own_values) {
    indirect<Value> default_value;
    BOOST_CHECK_EQUAL(default_value->number, 3);
    BOOST_CHECK_EQUAL(default_value->label, "default");

    indirect<Value> initialized(std::in_place, 7, "in place");
    BOOST_CHECK_EQUAL(initialized->number, 7);
    BOOST_CHECK_EQUAL(initialized->label, "in place");
    BOOST_CHECK(!initialized.valueless_after_move());
}

BOOST_AUTO_TEST_CASE(indirect_copy_construction_is_deep) {
    indirect<Value> original(std::in_place, 5, "original");
    auto copy = original;

    BOOST_CHECK(std::addressof(*copy) != std::addressof(*original));
    copy->number = 9;
    BOOST_CHECK_EQUAL(original->number, 5);
}

BOOST_AUTO_TEST_CASE(indirect_copy_assignment_replaces_the_value_and_is_self_safe) {
    indirect<Value> source(std::in_place, 4, "source");
    indirect<Value> target(std::in_place, 8, "target");

    target = source;
    BOOST_CHECK_EQUAL(target->number, 4);
    BOOST_CHECK(std::addressof(*target) != std::addressof(*source));

    const auto *before = std::addressof(*target);
    const indirect<Value> &alias = target;
    target = alias;
    BOOST_CHECK(std::addressof(*target) == before);
}

BOOST_AUTO_TEST_CASE(indirect_copy_assignment_has_strong_exception_safety) {
    indirect<ThrowOnCopy> source(std::in_place, 2);
    indirect<ThrowOnCopy> target(std::in_place, 1);

    ThrowOnCopy::should_throw = true;
    BOOST_CHECK_THROW(target = source, std::runtime_error);
    ThrowOnCopy::should_throw = false;
    BOOST_CHECK_EQUAL(target->value, 1);
}

BOOST_AUTO_TEST_CASE(indirect_move_transfers_the_allocation) {
    indirect<Value> source(std::in_place, 6, "moved");
    const auto *address = std::addressof(*source);

    auto destination = std::move(source);
    BOOST_CHECK(source.valueless_after_move());
    BOOST_CHECK(std::addressof(*destination) == address);

    indirect<Value> assigned;
    assigned = std::move(destination);
    BOOST_CHECK(destination.valueless_after_move());
    BOOST_CHECK(std::addressof(*assigned) == address);
}

BOOST_AUTO_TEST_CASE(indirect_copy_preserves_a_valueless_state) {
    indirect<Value> source;
    auto owner = std::move(source);
    auto copy = source;

    BOOST_CHECK(source.valueless_after_move());
    BOOST_CHECK(copy.valueless_after_move());
    BOOST_CHECK(!owner.valueless_after_move());
}

BOOST_AUTO_TEST_CASE(indirect_propagates_const_to_the_value) {
    static_assert(std::is_same_v<decltype(*std::declval<indirect<Value> &>()), Value &>);
    static_assert(std::is_same_v<decltype(*std::declval<const indirect<Value> &>()), const Value &>);
    static_assert(std::is_same_v<decltype(std::declval<const indirect<Value> &>().operator->()), const Value *>);
}

BOOST_AUTO_TEST_CASE(indirect_supports_an_opaque_holder) {
    test_utils::OpaqueIndirectHolder original(11);
    auto copy = original;
    BOOST_CHECK_EQUAL(copy.value(), 11);

    test_utils::OpaqueIndirectHolder assigned(3);
    assigned = original;
    BOOST_CHECK_EQUAL(assigned.value(), 11);
}
