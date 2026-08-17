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

// value_ptr is the vocabulary type behind the Rule of Zero in MonomialPropagator and MPOperator, so its
// two copy routes (T::clone() and T's copy constructor) and its const propagation are pinned here rather
// than only through the propagator's deep-copy tests.

#include <boost/test/unit_test.hpp>

#include <memory>
#include <type_traits>

#include "monoprop/ValuePtr.h"

using monoprop::make_value;
using monoprop::value_ptr;

namespace {

// Copyable, no clone(): value_ptr must take the copy-constructor route.
struct Copyable {
    int value;
};

// Non-copyable but cloneable, as OperatorIndex is: only the clone() route can duplicate it.
struct Cloneable {
    int value;

    Cloneable(int v) : value(v) {}
    Cloneable(const Cloneable &) = delete;
    auto operator=(const Cloneable &) -> Cloneable & = delete;

    [[nodiscard]] auto clone() const -> std::unique_ptr<Cloneable> { return std::make_unique<Cloneable>(value); }
};

// A class holding one declares no special member of its own, and still gets all six.
struct Holder {
    value_ptr<Cloneable> held = make_value<Cloneable>(0);
};

static_assert(std::is_copy_constructible_v<Holder>);
static_assert(std::is_copy_assignable_v<Holder>);
static_assert(std::is_move_constructible_v<Holder>);
static_assert(std::is_move_assignable_v<Holder>);

} // namespace

BOOST_AUTO_TEST_CASE(value_ptr_copy_uses_clone_when_available) {
    auto original = make_value<Cloneable>(7);
    auto copy = original;

    BOOST_CHECK_EQUAL(copy->value, 7);
    BOOST_CHECK(copy.get() != original.get()); // a separate pointee, not a shared one

    copy->value = 9;
    BOOST_CHECK_EQUAL(original->value, 7);
}

BOOST_AUTO_TEST_CASE(value_ptr_copy_falls_back_to_copy_construction) {
    auto original = make_value<Copyable>(Copyable{.value = 3});
    auto copy = original;

    BOOST_CHECK_EQUAL(copy->value, 3);
    BOOST_CHECK(copy.get() != original.get());
}

BOOST_AUTO_TEST_CASE(value_ptr_assignment_is_self_safe_and_deep) {
    auto a = make_value<Cloneable>(1);
    auto b = make_value<Cloneable>(2);

    a = b;
    BOOST_CHECK_EQUAL(a->value, 2);
    BOOST_CHECK(a.get() != b.get());

    // Clone-before-release: assigning from itself must not read a freed pointee. Aliased so that
    // -Wself-assign-overloaded does not reject the very case under test.
    const value_ptr<Cloneable> &alias = a;
    const auto *before = a.get();
    a = alias;
    BOOST_CHECK_EQUAL(a->value, 2);
    BOOST_CHECK(a.get() != before); // the clone replaced the original, so the address moves
}

BOOST_AUTO_TEST_CASE(value_ptr_move_steals_the_pointee) {
    auto source = make_value<Cloneable>(5);
    const auto *owned = source.get();

    auto moved = std::move(source);
    BOOST_CHECK(moved.get() == owned); // no clone on the move path
    BOOST_CHECK(!source);
}

BOOST_AUTO_TEST_CASE(value_ptr_empty_copies_stay_empty) {
    value_ptr<Cloneable> empty;
    BOOST_CHECK(!empty);

    auto copy = empty;
    BOOST_CHECK(!copy);

    auto filled = make_value<Cloneable>(4);
    filled = empty;
    BOOST_CHECK(!filled);
}

BOOST_AUTO_TEST_CASE(value_ptr_propagates_const_to_the_pointee) {
    static_assert(std::is_same_v<decltype(*std::declval<value_ptr<Copyable> &>()), Copyable &>);
    static_assert(std::is_same_v<decltype(*std::declval<const value_ptr<Copyable> &>()), const Copyable &>);
    static_assert(std::is_same_v<decltype(std::declval<const value_ptr<Copyable> &>().get()), const Copyable *>);
}
