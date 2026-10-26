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

// A propagator talks on a private duplicate of the communicator it is given, so its fixed-tag
// point-to-point traffic cannot match the caller's own messages on that communicator.

#include <boost/test/unit_test.hpp>

#ifdef monoprop_ENABLE_MPI

#include <array>
#include <chrono>
#include <memory>
#include <vector>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = 8;

auto compare_comms(MPI_Comm lhs, MPI_Comm rhs) -> int {
    int result = MPI_UNEQUAL;
    MPI_Comm_compare(lhs, rhs, &result);
    return result;
}

BOOST_AUTO_TEST_CASE(propagator_runs_on_a_private_duplicate_of_its_communicator) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    for (const MPI_Comm user_comm : {MPI_COMM_SELF, MPI_COMM_WORLD}) {
        auto sim = build_simulator<kNumModes>(data, SimulatorConfig{.comm = user_comm});
        BOOST_TEST(sim.comm() != user_comm);
        BOOST_TEST(compare_comms(sim.comm(), user_comm) == MPI_CONGRUENT);
    }
}

// Counts the communicators an attribute was deleted from: MPI runs the delete callback when a
// communicator carrying it is freed, which is the only way to observe a free from outside.
int g_deleted_attributes = 0;

auto copy_attribute(MPI_Comm, int, void *, void *value_in, void *value_out, int *flag) -> int {
    *static_cast<void **>(value_out) = value_in;
    *flag = 1;
    return MPI_SUCCESS;
}

auto count_deleted_attribute(MPI_Comm, int, void *, void *) -> int {
    ++g_deleted_attributes;
    return MPI_SUCCESS;
}

BOOST_AUTO_TEST_CASE(duplicate_is_shared_by_copies_and_freed_with_the_last_owner) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    MPI_Comm user_comm = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &user_comm);
    int keyval = MPI_KEYVAL_INVALID;
    MPI_Comm_create_keyval(copy_attribute, count_deleted_attribute, &keyval, nullptr);
    MPI_Comm_set_attr(user_comm, keyval, nullptr);
    g_deleted_attributes = 0;

    auto original = std::make_unique<MonomialPropagator<kNumModes>>(
        build_simulator<kNumModes>(data, SimulatorConfig{.comm = user_comm}));
    auto copy = std::make_unique<MonomialPropagator<kNumModes>>(*original);
    BOOST_TEST(copy->comm() == original->comm());

    const MPI_Comm duplicate = copy->comm();
    original.reset();
    BOOST_TEST(g_deleted_attributes == 0); // the copy still owns it
    BOOST_TEST(MPI_Barrier(duplicate) == MPI_SUCCESS);

    copy.reset();
    BOOST_TEST(g_deleted_attributes == 1); // the duplicate, and only it, was freed

    MPI_Comm_free(&user_comm);
    BOOST_TEST(g_deleted_attributes == 2); // the caller's own communicator is still theirs to free
    MPI_Comm_free_keyval(&keyval);
}

// The caller's message sits unmatched on the very communicator monoprop is handed, one per internal tag.
// Without the duplicate, monoprop's own receives (same source, same tag) would consume it.
BOOST_AUTO_TEST_CASE(user_messages_on_internal_tags_survive_propagation) {
    const int rank = mpi::rank(MPI_COMM_WORLD);
    const int size = mpi::size(MPI_COMM_WORLD);
    if (size < 2) {
        BOOST_TEST_MESSAGE("Skipping user_messages_on_internal_tags_survive_propagation: world size = 1.");
        return;
    }
    const int partner = rank ^ 1;
    const bool has_partner = partner < size;
    constexpr std::array tags{mpi::kHybridCountTag,
                              mpi::kHybridPayloadTag,
                              mpi::kFlatPayloadTag,
                              mpi::kFlatCountTag,
                              mpi::kFlatReplayTag};

    std::array<int, tags.size()> outgoing{};
    std::vector<MPI_Request> sends;
    for (size_t i = 0; i < tags.size() && has_partner; ++i) {
        outgoing[i] = (1000 * rank) + tags[i];
        sends.emplace_back();
        MPI_Isend(&outgoing[i], 1, MPI_INT, partner, tags[i], MPI_COMM_WORLD, &sends.back());
    }

    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = build_simulator<kNumModes>(data, SimulatorConfig{.comm = MPI_COMM_WORLD});
    BOOST_CHECK_SMALL(evaluate_expval(sim, data, false) - data.actual_expval, 1e-9);

    // Polled against a deadline so a consumed message fails the test instead of hanging it.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (size_t i = 0; i < tags.size() && has_partner; ++i) {
        int incoming = -1;
        MPI_Request receive = MPI_REQUEST_NULL;
        MPI_Irecv(&incoming, 1, MPI_INT, partner, tags[i], MPI_COMM_WORLD, &receive);
        int done = 0;
        while (done == 0 && std::chrono::steady_clock::now() < deadline) {
            MPI_Test(&receive, &done, MPI_STATUS_IGNORE);
        }
        BOOST_REQUIRE_MESSAGE(done != 0, "the user's message with tag " << tags[i] << " was consumed");
        BOOST_TEST(incoming == (1000 * partner) + tags[i]);
    }
    MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
}

} // namespace

#endif // monoprop_ENABLE_MPI
