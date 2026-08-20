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

// Every case asserts a line COUNT: a silent instrument and an idle workload leave the same empty log.

#include <boost/test/unit_test.hpp>

#ifdef monoprop_ENABLE_PROFILE

#include <array>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "ThreadHarness.h"
#include "monoprop/detail/Profile.h"
#include "monoprop/detail/mpi/PartitionBarrier.h"
#include "monoprop/detail/mpi/ShmComm.h"

namespace profile = monoprop::detail::profile;
using monoprop::mpi::PartitionBarrier;
using monoprop::mpi::ShmComm;

namespace {

class CaptureFile {
public:
    CaptureFile() : f_(std::tmpfile()) { BOOST_REQUIRE(f_ != nullptr); }
    ~CaptureFile() {
        if (f_ != nullptr) {
            std::fclose(f_);
        }
    }
    CaptureFile(const CaptureFile &) = delete;
    auto operator=(const CaptureFile &) -> CaptureFile & = delete;

    auto stream() const -> std::FILE * { return f_; }

    auto text() const -> std::string {
        std::fflush(f_);
        std::rewind(f_);
        std::string out;
        std::array<char, 4096> buf{};
        while (const size_t n = std::fread(buf.data(), 1, buf.size(), f_)) {
            out.append(buf.data(), n);
        }
        return out;
    }

private:
    std::FILE *f_;
};

// Occurrences, not a contains check: firing the wrong number of times is the failure guarded against.
auto count(std::string_view haystack, std::string_view needle) -> size_t {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string_view::npos; at = haystack.find(needle, at + 1)) {
        ++n;
    }
    return n;
}

auto field(std::string_view line, std::string_view name) -> std::string {
    const size_t at = line.find(name);
    BOOST_REQUIRE(at != std::string_view::npos);
    const size_t start = at + name.size();
    const size_t end = line.find_first_of(" \n", start);
    return std::string(line.substr(start, end - start));
}

constexpr int kPartitions = 4;

} // namespace

BOOST_AUTO_TEST_CASE(comm_profile_enabled_emits_exactly_one_line) {
    const CaptureFile cap;
    {
        profile::CommRegistry reg(kPartitions, {.mpi_rank = 3, .enabled = true, .out = cap.stream()});
        for (int p = 0; p < kPartitions; ++p) {
            auto *slot = reg.slot(p);
            BOOST_REQUIRE(slot != nullptr);
            slot->n_barriers = 7;
            slot->barrier_ns = 1'000'000;
        }
    } // teardown emits

    const auto text = cap.text();
    BOOST_CHECK_EQUAL(count(text, "COMMPROF"), 1U);
    BOOST_CHECK_EQUAL(count(text, "\n"), 1U); // terminated, so a second line cannot merge into it
    BOOST_CHECK_EQUAL(field(text, "rank="), "3");
    BOOST_CHECK_EQUAL(field(text, "partitions="), "4");
}

BOOST_AUTO_TEST_CASE(comm_profile_disabled_emits_nothing_and_allocates_no_slot) {
    const CaptureFile cap;
    {
        profile::CommRegistry reg(kPartitions, {.mpi_rank = 0, .enabled = false, .out = cap.stream()});
        for (int p = 0; p < kPartitions; ++p) {
            BOOST_CHECK(reg.slot(p) == nullptr);
        }
        BOOST_CHECK_EQUAL(reg.emit(), 0);
    }
    BOOST_CHECK_EQUAL(cap.text().size(), 0U);
}

BOOST_AUTO_TEST_CASE(comm_profile_emit_is_one_shot) {
    const CaptureFile cap;
    {
        profile::CommRegistry reg(kPartitions, {.mpi_rank = 0, .enabled = true, .out = cap.stream()});
        BOOST_CHECK_EQUAL(reg.emit(), 1);
        BOOST_CHECK_EQUAL(reg.emit(), 0);
    }
    BOOST_CHECK_EQUAL(count(cap.text(), "COMMPROF"), 1U);
}

// `pinned=` is a SUM over partitions, not a flag, so a partial pin has to read as partial: three of four.
BOOST_AUTO_TEST_CASE(comm_profile_pinned_counts_partitions_not_transports) {
    const CaptureFile cap;
    {
        profile::CommRegistry reg(kPartitions, {.mpi_rank = 0, .enabled = true, .out = cap.stream()});
        for (int p = 0; p < kPartitions - 1; ++p) {
            reg.slot(p)->pinned = 1;
        }
    }
    BOOST_CHECK_EQUAL(field(cap.text(), "pinned="), "3");
}

// The seam a real run pins through: a master calls its transport, and the field is a per-partition flag.
BOOST_AUTO_TEST_CASE(comm_profile_note_pinned_reaches_the_printed_count) {
    const CaptureFile cap;
    {
        ShmComm shm(kPartitions, {.mpi_rank = 0, .enabled = true, .out = cap.stream()});
        shm.note_pinned(0);
        shm.note_pinned(0);
        shm.note_pinned(1);
    }
    const auto text = cap.text();
    BOOST_CHECK_EQUAL(count(text, "COMMPROF"), 1U);
    BOOST_CHECK_EQUAL(field(text, "partitions="), "4");
    BOOST_CHECK_EQUAL(field(text, "pinned="), "2");
    BOOST_CHECK(text.find("partitions=") < text.find("pinned="));
    BOOST_CHECK(text.find("pinned=") < text.find("verbs="));
}

BOOST_AUTO_TEST_CASE(comm_profile_note_pinned_is_a_no_op_when_the_region_is_off) {
    const CaptureFile cap;
    {
        ShmComm shm(kPartitions, {.mpi_rank = 0, .enabled = false, .out = cap.stream()});
        shm.note_pinned(0);
    }
    BOOST_CHECK_EQUAL(cap.text().size(), 0U);
}

// barrier_p0_s and barrier_peers_s compare only if every participant counts every sync, the last
// included -- hence a live barrier on real threads.
BOOST_AUTO_TEST_CASE(comm_profile_barrier_counts_every_sync_on_every_partition) {
    constexpr int kRounds = 5;
    const CaptureFile cap;
    profile::CommRegistry reg(kPartitions, {.mpi_rank = 0, .enabled = true, .out = cap.stream()});
    PartitionBarrier barrier(kPartitions);

    const auto errs = test_utils::run_comm_threads(barrier, kPartitions, [&](PartitionBarrier &b, int p) {
        for (int i = 0; i < kRounds; ++i) {
            b.sync(reg.slot(p));
        }
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
    for (int p = 0; p < kPartitions; ++p) {
        BOOST_CHECK_EQUAL(reg.slot(p)->n_barriers, static_cast<uint64_t>(kRounds));
    }
}

// A verb that forgot its counter, and one handed the wrong partition's slot, both look like a fast run.
BOOST_AUTO_TEST_CASE(comm_profile_shm_transport_records_its_verbs) {
    constexpr int kVerbsPerPartition = 2; // alltoall_counts + allreduce_sum, below
    const CaptureFile cap;
    {
        ShmComm shm(kPartitions, {.mpi_rank = 0, .enabled = true, .out = cap.stream()});
        const auto errs = test_utils::run_comm_threads(shm, kPartitions, [](ShmComm &sh, int p) {
            std::vector<int> send(static_cast<size_t>(kPartitions), 1);
            std::vector<int> recv(static_cast<size_t>(kPartitions), 0);
            sh.alltoall_counts(p, send.data(), recv.data());
            sh.allreduce_sum<double>(p, 1.0);
        });
        for (const auto &e : errs) {
            BOOST_CHECK(e == nullptr);
        }
    }

    const auto text = cap.text();
    BOOST_CHECK_EQUAL(count(text, "COMMPROF"), 1U);
    BOOST_CHECK_EQUAL(field(text, "verbs="), std::to_string(kVerbsPerPartition));
    // Two syncs per verb here, and the line reports partition 0's count.
    BOOST_CHECK_EQUAL(field(text, "barriers="), std::to_string(2 * kVerbsPerPartition));
    // No MPI exists on this path, so mpi_s is 0 by construction rather than by measurement.
    BOOST_CHECK_EQUAL(field(text, "mpi_s="), "0.0000");
}

BOOST_AUTO_TEST_CASE(layer_profile_line_count_follows_the_region) {
    const CaptureFile cap;
    profile::Slot s;
    s.n_gates = 3;
    s.total_ns = 7'000'000;

    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), false, 0, s), 0);
    BOOST_CHECK_EQUAL(cap.text().size(), 0U);

    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), true, 6, s), 1);
    const auto text = cap.text();
    BOOST_CHECK_EQUAL(count(text, "LAYERPROF"), 1U);
    BOOST_CHECK_EQUAL(count(text, "\n"), 1U);
    BOOST_CHECK_EQUAL(field(text, "row="), "6");
    BOOST_CHECK_EQUAL(field(text, "gates="), "3");
    BOOST_CHECK_EQUAL(field(text, "total_s="), "0.0070");
}

// A caches-only thread has n_gates == 0 and real work, and an absent row reads as one that did none.
BOOST_AUTO_TEST_CASE(layer_profile_skips_a_slot_that_did_nothing) {
    const CaptureFile cap;
    const profile::Slot idle;
    BOOST_CHECK(!idle.any_nonzero());
    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), true, 0, idle), 0);
    BOOST_CHECK_EQUAL(cap.text().size(), 0U);

    profile::Slot caches_only;
    caches_only.contract_ns = 1'000;
    BOOST_CHECK(caches_only.any_nonzero());
    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), true, 0, caches_only), 1);
}

// By NAME and by VALUE, values distinct: printed-but-zero and not-printed are how this looks alive.
BOOST_AUTO_TEST_CASE(layer_profile_carries_the_graph_fields) {
    const CaptureFile cap;
    profile::Slot s;
    s.encode_ns = 11'000'000;
    s.encode_layout_ns = 3'000'000;
    s.encode_pack_ns = 7'000'000;
    s.append_ns = 23'000'000;

    // Any one alone must make the slot non-empty, or a graph-store-only run prints nothing at all.
    for (const auto member : {&profile::Slot::encode_ns,
                              &profile::Slot::encode_layout_ns,
                              &profile::Slot::encode_pack_ns,
                              &profile::Slot::append_ns}) {
        profile::Slot one;
        one.*member = 1;
        BOOST_CHECK(one.any_nonzero());
    }

    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), true, 0, s), 1);
    const auto text = cap.text();
    BOOST_CHECK_EQUAL(field(text, "encode_s="), "0.0110");
    BOOST_CHECK_EQUAL(field(text, "encodelayout_s="), "0.0030");
    BOOST_CHECK_EQUAL(field(text, "encodepack_s="), "0.0070");
    BOOST_CHECK_EQUAL(field(text, "append_s="), "0.0230");

    // The line is flat and the timers nest, so position is the only thing that conveys the tree.
    BOOST_CHECK(text.find("encode_s=") < text.find("encodelayout_s="));
    BOOST_CHECK(text.find("encodelayout_s=") < text.find("encodepack_s="));
    BOOST_CHECK(text.find("encodepack_s=") < text.find("apply_s="));
    BOOST_CHECK(text.find("apply_s=") < text.find("append_s="));
    BOOST_CHECK(text.find("append_s=") < text.find("contract_s="));
}

// `exchange_s` is the SUM and printed first, so a parser that already sums it keeps its old meaning.
BOOST_AUTO_TEST_CASE(layer_profile_splits_exchange_into_its_two_legs) {
    const CaptureFile cap;
    profile::Slot s;
    s.exchange_ns = 5'000'000;
    s.exchange_resp_ns = 3'000'000;

    for (const auto member : {&profile::Slot::exchange_ns, &profile::Slot::exchange_resp_ns}) {
        profile::Slot one;
        one.*member = 1;
        BOOST_CHECK(one.any_nonzero());
    }

    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), true, 0, s), 1);
    const auto text = cap.text();
    BOOST_CHECK_EQUAL(field(text, "exchange_s="), "0.0080");
    BOOST_CHECK_EQUAL(field(text, "exchangeq_s="), "0.0050");
    BOOST_CHECK_EQUAL(field(text, "exchangeresp_s="), "0.0030");
    BOOST_CHECK_CLOSE(std::stod(field(text, "exchange_s=")),
                      std::stod(field(text, "exchangeq_s=")) + std::stod(field(text, "exchangeresp_s=")),
                      1e-9);
    BOOST_CHECK(text.find("sendbuf_s=") < text.find("exchange_s="));
    BOOST_CHECK(text.find("exchange_s=") < text.find("exchangeq_s="));
    BOOST_CHECK(text.find("exchangeq_s=") < text.find("exchangeresp_s="));
    BOOST_CHECK(text.find("exchangeresp_s=") < text.find("incoming_s="));
}

// Three harness parsers key on these names, and this change added fields to the line they read.
BOOST_AUTO_TEST_CASE(layer_profile_keeps_every_field_name_a_parser_keys_on) {
    const CaptureFile cap;
    profile::Slot s;
    s.n_gates = 1;
    BOOST_CHECK_EQUAL(profile::emit_layer_line(cap.stream(), true, 0, s), 1);
    const auto text = cap.text();
    for (const auto *name : {"row=",           "gates=",     "total_s=",    "gate_s=",      "layer_s=",
                             "fold_s=",        "scan_s=",    "emit_s=",     "index_s=",     "resolve_s=",
                             "insert_s=",      "sendbuf_s=", "exchange_s=", "exchangeq_s=", "exchangeresp_s=",
                             "incoming_s=",    "apply_s=",   "contract_s=", "cacheop_s=",   "cachestate_s=",
                             "cacheshrink_s=", "anti=",      "foll=",       "atol=",        "reject=",
                             "push=",          "hit=",       "miss=",       "qbytes=",      "words=",
                             "livewords=",     "rehash_s=",  "rehash=",     "rehashrows="}) {
        BOOST_CHECK_MESSAGE(text.find(name) != std::string::npos, "LAYERPROF lost the field " << name);
    }
}

BOOST_AUTO_TEST_CASE(replay_profile_line_count_follows_the_region) {
    const CaptureFile cap;
    profile::Slot s;
    s.n_ev = 2;
    s.ev_ns = 5'000'000;

    BOOST_CHECK_EQUAL(profile::emit_replay_line(cap.stream(), false, 0, s), 0);
    BOOST_CHECK_EQUAL(cap.text().size(), 0U);

    BOOST_CHECK_EQUAL(profile::emit_replay_line(cap.stream(), true, 4, s), 1);
    const auto text = cap.text();
    BOOST_CHECK_EQUAL(count(text, "REPLAYPROF"), 1U);
    BOOST_CHECK_EQUAL(count(text, "\n"), 1U);
    BOOST_CHECK_EQUAL(field(text, "row="), "4");
    BOOST_CHECK_EQUAL(field(text, "ev="), "2");
    BOOST_CHECK_EQUAL(field(text, "ev_s="), "0.0050");
}

// The two field sets share one Slot, so replay_nonzero() must be a separate predicate from any_nonzero.
BOOST_AUTO_TEST_CASE(replay_profile_skips_a_thread_that_only_ran_gates) {
    const CaptureFile cap;
    profile::Slot layer_only;
    layer_only.n_gates = 1'000;
    layer_only.layer_ns = 9'000'000;
    BOOST_REQUIRE(layer_only.any_nonzero());
    BOOST_CHECK(!layer_only.replay_nonzero());
    BOOST_CHECK_EQUAL(profile::emit_replay_line(cap.stream(), true, 0, layer_only), 0);

    profile::Slot replay_only;
    replay_only.n_pare = 1;
    replay_only.pare_ns = 3'000;
    BOOST_CHECK(replay_only.replay_nonzero());
    BOOST_CHECK_EQUAL(profile::emit_replay_line(cap.stream(), true, 0, replay_only), 1);

    BOOST_CHECK_EQUAL(count(cap.text(), "REPLAYPROF"), 1U);
}

BOOST_AUTO_TEST_CASE(rusage_profile_line_count_follows_the_region) {
    const CaptureFile cap;

    BOOST_CHECK_EQUAL(profile::emit_rusage_line(cap.stream(), false), 0);
    BOOST_CHECK_EQUAL(cap.text().size(), 0U);

    BOOST_CHECK_EQUAL(profile::emit_rusage_line(cap.stream(), true), 1);
    const auto text = cap.text();
    BOOST_CHECK_EQUAL(count(text, "LAYERRUSAGE"), 1U);
    BOOST_CHECK_EQUAL(count(text, "\n"), 1U);
    // A live reading, so only what a parser depends on: the fields exist, and maxrss_kb is a KiB figure.
    BOOST_CHECK(!field(text, "minflt=").empty());
    BOOST_CHECK(!field(text, "majflt=").empty());
    BOOST_CHECK(!field(text, "nvcsw=").empty());
    BOOST_CHECK(!field(text, "nivcsw=").empty());
    BOOST_CHECK_GT(std::stoull(field(text, "maxrss_kb=")), 0ULL);
}

// The default is declared twice -- CommOptions' member initialiser and `CommOptions prof = {}` per
// transport -- and a knob whose declarations disagree ships a regression enabled by default.
BOOST_AUTO_TEST_CASE(comm_profile_options_default_to_the_environment) {
    const profile::CommOptions declared;
    BOOST_CHECK_EQUAL(declared.enabled, monoprop::config::get().profile.comm);
    BOOST_CHECK(declared.out == stderr);
    BOOST_CHECK_EQUAL(declared.mpi_rank, 0);

    const CaptureFile cap;
    profile::CommRegistry defaulted(kPartitions, {.out = cap.stream()});
    BOOST_CHECK_EQUAL(defaulted.slot(0) != nullptr, monoprop::config::get().profile.comm);
    BOOST_CHECK_EQUAL(defaulted.emit(), monoprop::config::get().profile.comm ? 1 : 0);

    ShmComm shm(kPartitions);
    const auto errs = test_utils::run_comm_threads(shm, kPartitions, [](ShmComm &sh, int p) {
        std::vector<int> send(static_cast<size_t>(kPartitions), 1);
        std::vector<int> recv(static_cast<size_t>(kPartitions), 0);
        sh.alltoall_counts(p, send.data(), recv.data());
    });
    for (const auto &e : errs) {
        BOOST_CHECK(e == nullptr);
    }
}

#endif // monoprop_ENABLE_PROFILE
