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

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <numbers>
#include <optional>
#include <vector>

#include "KernelTestSupport.h"
#include "PropagatorTestAccess.h"
#include "ScanTestSupport.h"
#include "TestUtilities.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"

// The fused cos sweep's one deliberate FP deviation from the two-pass path (≤1 ulp per hit endpoint,
// from resolve's stored·(1/cos) recovery), against the build_graph()+replay evaluation as oracle.
//
// The threaded cases below run the one-store prototype (explicit partitions=1, budget captured from
// monoprop_NUM_THREADS at construction). Serial and threaded runs of the same algorithm must agree bitwise;
// fused-versus-replay comparisons keep the tolerances above.

namespace {

using namespace test_utils;
using namespace monoprop;

// The two paths accumulate FP differently even before the sweep, so demand 1e-12: tight enough that a
// wrong cos factor on any endpoint (relative error O(1)) fails loudly, loose enough for reordering.
constexpr double kAgreeAtol = 1e-12;
constexpr double kExactAtol = 1e-9;

template <size_t NumModes>
auto inplace_energy(const CaseData &data, const SimulatorConfig &cfg) -> double {
    auto sim = build_simulator<NumModes>(data, cfg);
    sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
    auto fn = sim.expectation_value_functional(std::nullopt);
    return fn(VecD{});
}

template <size_t NumModes>
auto graph_energy(const CaseData &data, const SimulatorConfig &cfg) -> double {
    auto sim = build_simulator<NumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto fn = sim.expectation_value_functional(std::nullopt);
    return fn(data.parameters);
}

template <size_t NumModes>
auto build_prototype(const CaseData &data, const SimulatorConfig &cfg, int budget) -> MonomialPropagator<NumModes> {
    const kernel_test::ScopedBudget scoped(budget);
    MonomialPropagator<NumModes> sim(data.hamiltonian,
                                     static_cast<unsigned int>(2 * NumModes),
                                     data.initial_state,
                                     cfg.schrodinger_cutoff,
                                     cfg.comm,
                                     cfg.atol,
                                     cfg.upper_atol,
                                     cfg.cutoff_type,
                                     cfg.basis_change,
                                     NumModes,
                                     Basis::Majorana,
                                     /*partitions=*/1);
    // Selection is checked, not assumed: only the prototype captures the environment budget.
    BOOST_TEST_REQUIRE(monoprop::detail::PropagatorTestAccess<NumModes>::options(sim).threads == budget);
    return sim;
}

void check_agreement(const CaseData &data, const SimulatorConfig &cfg, const char *label) {
    const double inplace = inplace_energy<ExampleDataFix::n_modes>(data, cfg);
    const double graph = graph_energy<ExampleDataFix::n_modes>(data, cfg);
    BOOST_TEST_CONTEXT(label << " inplace=" << inplace << " graph=" << graph) {
        BOOST_CHECK_SMALL(inplace - graph, kAgreeAtol);
        BOOST_CHECK_SMALL(inplace - data.actual_expval, kExactAtol);
    }
}

// check_agreement on threaded prototypes, with the same tolerances.
void check_prototype_agreement(const CaseData &data, const SimulatorConfig &cfg, int budget, const char *label) {
    constexpr size_t N = ExampleDataFix::n_modes;
    auto inplace_sim = build_prototype<N>(data, cfg, budget);
    inplace_sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
    const double inplace = inplace_sim.expectation_value_functional(std::nullopt)(VecD{});
    auto graph_sim = build_prototype<N>(data, cfg, budget);
    graph_sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double graph = graph_sim.expectation_value_functional(std::nullopt)(data.parameters);
    BOOST_TEST_CONTEXT(label << " budget=" << budget << " inplace=" << inplace << " graph=" << graph) {
        BOOST_CHECK_SMALL(inplace - graph, kAgreeAtol);
        BOOST_CHECK_SMALL(inplace - data.actual_expval, kExactAtol);
    }
}

// --- a workload large enough for the threaded kernels ---------------------------------------------------------

constexpr size_t kBig = 10; // 20 Majoranas
using BigAccess = monoprop::detail::PropagatorTestAccess<kBig>;

// A synthetic Hermitian operator of 16k even-weight terms and gates of weights 2-4 (odd included), large
// enough that fused records and replayed folds span several logical ranges. One gate sits at theta = pi/4,
// where the fused sweep recovers pre-cos values through 1/cos ~ 1.6e16.
auto big_case() -> CaseData {
    CaseData data;
    data.num_modes = kBig;
    kernel_test::SplitMix rng{2026};
    const auto draw_indices = [&rng](size_t weight) {
        VecZ idx;
        while (idx.size() < weight) {
            const size_t m = rng.below(2 * kBig);
            if (std::ranges::find(idx, m) == idx.end()) {
                idx.push_back(m);
            }
        }
        std::ranges::sort(idx);
        return idx;
    };
    while (data.hamiltonian.size() < 16000) {
        const auto idx = draw_indices(2 * (1 + rng.below(3)));
        const double x = static_cast<double>(rng.below(1U << 20)) / (1U << 20) - 0.5;
        data.hamiltonian[idx] = x * hermitian_coefficient<kBig>(indices_to_bitset<kBig>(idx));
    }
    data.initial_state = {0, 2, 4, 6, 8};
    constexpr size_t gates = 9;
    for (size_t g = 0; g < gates; ++g) {
        data.majoranas.push_back(draw_indices(2 + rng.below(3)));
        data.param_inds.push_back(g);
        data.gen_coeffs.push_back(1.0);
        data.parameters.push_back(g == 4 ? std::numbers::pi / 4 : 0.1 + static_cast<double>(rng.below(900)) / 1000.0);
    }
    // Repeated generators: rotating again by G targets the M*G terms the first rotation inserted, so both
    // pictures produce hit records as well as inserts.
    data.majoranas[6] = data.majoranas[2];
    data.majoranas[8] = data.majoranas[3];
    return data;
}

auto big_prototype(const CaseData &data, bool schrodinger, int budget, MPI_Comm comm) -> MonomialPropagator<kBig> {
    return build_prototype<kBig>(
        data,
        SimulatorConfig{.schrodinger_cutoff = schrodinger ? std::optional<unsigned int>(2 * kBig) : std::nullopt,
                        .comm = comm},
        budget);
}

auto bitwise_equal(const VecD &a, const VecD &b) -> bool {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

auto comm_size(MPI_Comm comm) -> int {
    return mpi::size(mpi::Comm(comm));
}

} // namespace

BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_heisenberg, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{}, "heisenberg");
}

// lower_atol active: the sin gate reads the pre-cos value, so the in-place store that follows must
// not change which terms are emitted.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_heisenberg_atol, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.atol = 1e-10}, "heisenberg atol=1e-10");
}

// Schrödinger: fresh inserts are born after the sweep with a nonzero coeff, so the apply's insert arm
// must fold the gate's cos into those slots itself.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_schrodinger, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes}, "schrodinger");
}

BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_schrodinger_atol, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes, .atol = 1e-10}, "schrodinger atol=1e-10");
}

// The two-pass apply (a length cap at the full width truncates nothing but disables the fused sweep) against
// graph replay, with the same tolerances: the insert snapshots must be gathered before the cosine phase.
BOOST_FIXTURE_TEST_CASE(two_pass_apply_matches_graph_replay, ExampleDataFix) {
    constexpr size_t cap = 2 * n_modes;
    for (const bool schrodinger : {false, true}) {
        for (const int budget : {1, 4}) {
            const SimulatorConfig cfg{.schrodinger_cutoff =
                                          schrodinger ? std::optional<unsigned int>(2 * n_modes) : std::nullopt};
            auto inplace_sim = build_prototype<n_modes>(data, cfg, budget);
            inplace_sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
            const double inplace = inplace_sim.expectation_value_functional(std::nullopt)(VecD{});
            auto graph_sim = build_prototype<n_modes>(data, cfg, budget);
            graph_sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, std::nullopt, cap);
            const double graph = graph_sim.expectation_value_functional(std::nullopt)(data.parameters);
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " budget=" << budget << " inplace=" << inplace
                                              << " graph=" << graph) {
                BOOST_CHECK_SMALL(inplace - graph, kAgreeAtol);
                BOOST_CHECK_SMALL(inplace - data.actual_expval, kExactAtol);
            }
        }
    }
}

// The same fused-versus-replay agreement on threaded one-store prototypes. This fixture is small, so the
// kernels take their serial paths here; the big-workload cases below cover the threaded ones.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_threaded_prototype, ExampleDataFix) {
    for (const int budget : {1, 4}) {
        check_prototype_agreement(data, SimulatorConfig{}, budget, "heisenberg");
        check_prototype_agreement(data, SimulatorConfig{.atol = 1e-10}, budget, "heisenberg atol=1e-10");
        check_prototype_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes}, budget, "schrodinger");
    }
}

// Records from the real ContractImmediately build path, gate by gate, in both pictures and in the fused-scale
// and two-pass (length cap) modes: every destination slot has exactly one add-owner across hits, inserts and
// cross-rank halves, and the apply at budgets 2-4 matches budget 1 bitwise. Under an MPI launch
// (MPI_COMM_WORLD with several ranks) the halves are real cross-rank records.
BOOST_AUTO_TEST_CASE(fused_real_records_have_one_add_owner_and_apply_exactly) {
    const auto data = big_case();
    const int ranks = comm_size(MPI_COMM_WORLD);
    const auto mapped = map_params(data.parameters, data.param_inds, data.gen_coeffs, 1.0);
    for (const bool schrodinger : {false, true}) {
        for (const bool length_cap : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " length_cap=" << length_cap << " ranks=" << ranks) {
                auto sim = big_prototype(data, schrodinger, 1, MPI_COMM_WORLD);
                // A cap at the full width truncates nothing but forces the two-pass apply.
                const std::optional<size_t> cap = length_cap ? std::optional<size_t>(2 * kBig) : std::nullopt;
                size_t duplicate_owners = 0;
                size_t hits = 0;
                size_t inserts = 0;
                size_t halves = 0;
                size_t max_apply_ranges = 0;
                size_t max_gather_ranges = 0;
                size_t max_gate_records = 0;
                size_t max_gate_inserts = 0;
                size_t nonzero_insert_snapshots = 0;
                bool fused_seen = false;
                bool two_pass_seen = false;
                bool mismatch = false;
                const size_t n = data.majoranas.size();
                for (size_t i = 0; i < n; ++i) {
                    // Production gate order and angles (evolve_mode_contract_immediately_ / gate_angle_).
                    const size_t idx = schrodinger ? i : n - 1 - i;
                    auto gate = BigAccess::contract_gate(sim, data.majoranas[idx], cap, mapped[idx]);
                    for (const auto &[slot, count] : kernel_test::add_owner_counts(gate.fc)) {
                        duplicate_owners += static_cast<size_t>(count != 1);
                    }
                    max_gate_records =
                        std::max(max_gate_records,
                                 gate.fc.hits.size() + gate.fc.inserts.size() + gate.fc.cross_half.size());
                    max_gate_inserts = std::max(max_gate_inserts, gate.fc.inserts.size());
                    hits += gate.fc.hits.size();
                    inserts += gate.fc.inserts.size();
                    halves += gate.fc.cross_half.size();
                    (gate.fused_scale ? fused_seen : two_pass_seen) = true;

                    const VecD before = *gate.coeffs;
                    for (const auto &r : gate.fc.inserts) {
                        nonzero_insert_snapshots += static_cast<size_t>(before[r.tgt] != 0.0);
                    }
                    const auto apply = [&](int threads, kernel_test::RangeLog *log) {
                        auto fc = gate.fc;
                        VecD out = before;
                        if (log != nullptr) {
                            monoprop::detail::apply_fused_contract(fc,
                                                                   out,
                                                                   gate.cos,
                                                                   gate.apply_angle,
                                                                   schrodinger,
                                                                   gate.fused_scale,
                                                                   {threads},
                                                                   kernel_test::RecordingObserver{log});
                        }
                        else {
                            monoprop::detail::apply_fused_contract(fc,
                                                                   out,
                                                                   gate.cos,
                                                                   gate.apply_angle,
                                                                   schrodinger,
                                                                   gate.fused_scale,
                                                                   {threads});
                        }
                        return out;
                    };
                    const VecD serial = apply(1, nullptr);
                    for (const int threads : {2, 3, 4}) {
                        kernel_test::RangeLog log;
                        const bool same = bitwise_equal(apply(threads, &log), serial);
                        mismatch = mismatch || !same;
                        max_apply_ranges =
                            std::max(max_apply_ranges, log[monoprop::detail::KernelRange::fused_apply].worker.size());
                        max_gather_ranges =
                            std::max(max_gather_ranges, log[monoprop::detail::KernelRange::fused_gather].worker.size());
                    }
                    *gate.coeffs = serial;
                }
                BOOST_TEST_MESSAGE("records: hits=" << hits << " inserts=" << inserts << " halves=" << halves
                                                    << " max apply ranges=" << max_apply_ranges
                                                    << " max gather ranges=" << max_gather_ranges
                                                    << " nonzero insert snapshots=" << nonzero_insert_snapshots);
                BOOST_TEST(duplicate_owners == 0u);
                BOOST_TEST(!mismatch);
                // One rank sees hits and inserts only; with several ranks the routing decides how many
                // rotations stay rank-local (linear routing can send all of them across), but halves appear.
                if (ranks == 1) {
                    BOOST_TEST(hits > 0u);
                    BOOST_TEST(inserts > 0u);
                    BOOST_TEST(halves == 0u);
                }
                else {
                    BOOST_TEST(halves > 0u);
                }
                // The kernel split the largest gate into the expected record ranges; on one rank this workload
                // is large enough for the threaded record path, not only its serial fallback. More ranks
                // divide the records and may leave a rank below two ranges.
                BOOST_TEST(max_apply_ranges == monoprop::detail::logical_ranges(max_gate_records, 1024));
                BOOST_TEST(max_gather_ranges
                           == (schrodinger ? monoprop::detail::logical_ranges(max_gate_inserts, 1024) : 0u));
                if (ranks == 1) {
                    BOOST_TEST(max_apply_ranges >= 2u);
                    BOOST_TEST(max_gather_ranges >= (schrodinger ? 2u : 0u));
                }
                BOOST_TEST(fused_seen == !length_cap);
                BOOST_TEST(two_pass_seen == length_cap);

                // The helper reproduced production: a serial prototype's own propagate gives the same vector.
                auto reference = big_prototype(data, schrodinger, 1, MPI_COMM_WORLD);
                reference.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
                BOOST_TEST(bitwise_equal(BigAccess::picture_coeffs(reference), BigAccess::picture_coeffs(sim)));
            }
        }
    }
}

// End to end through the public API: prototypes at budgets 1 and 4 propagate (fused apply, stored-mask and
// fused-scale paths) and replay a built graph (lazy folds) to bitwise-identical coefficients and energies.
BOOST_AUTO_TEST_CASE(fused_threaded_prototype_matches_serial_prototype_exactly) {
    const auto data = big_case();
    for (const bool schrodinger : {false, true}) {
        for (const bool length_cap : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " length_cap=" << length_cap) {
                const std::optional<size_t> cap = length_cap ? std::optional<size_t>(2 * kBig) : std::nullopt;
                auto serial = big_prototype(data, schrodinger, 1, MPI_COMM_WORLD);
                auto threaded = big_prototype(data, schrodinger, 4, MPI_COMM_WORLD);
                serial.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
                threaded.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
                BOOST_TEST(bitwise_equal(BigAccess::picture_coeffs(serial), BigAccess::picture_coeffs(threaded)));
                BOOST_TEST(serial.expectation_value(VecD{}) == threaded.expectation_value(VecD{}));
            }
        }
        BOOST_TEST_CONTEXT("graph replay, schrodinger=" << schrodinger) {
            auto serial = big_prototype(data, schrodinger, 1, MPI_COMM_WORLD);
            auto threaded = big_prototype(data, schrodinger, 4, MPI_COMM_WORLD);
            serial.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            threaded.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            BOOST_TEST_MESSAGE("graph-built operator terms on this rank: " << serial.mp_op().size());
            BOOST_TEST(bitwise_equal(serial.contract_partially(data.parameters, false),
                                     threaded.contract_partially(data.parameters, false)));
            BOOST_TEST(serial.expectation_value(data.parameters) == threaded.expectation_value(data.parameters));
        }
    }
}

// --- threaded bitmap scan: fused sweep and full construction -------------------------------------------------

// The fused cos sweep scales each anticommuting source exactly once, in whichever range owns it, after the
// scan captured its pre-scale value; every other coefficient stays untouched. Budgets 2-4 against the
// independently derived expectation, not only against budget 1.
BOOST_AUTO_TEST_CASE(openmp_scan_fused_sweep_scales_each_source_once) {
    const scan_test::Geometry solo{"one rank", 1, 1, false, 0};
    for (const auto &gen : {scan_test::mono_of_bits({14, 20, 33, 47}), scan_test::mono_of_bits({13, 29, 52})}) {
        const scan_test::Scenario plain{.label = "capture", .gen = gen, .capture = true};
        // Without a cap, the non-fused scan's cosine set is every anticommuting source.
        const auto reference = scan_test::run_scan(solo, plain, {.threads = 1});
        const auto anti = scan_test::concatenated_cos(reference.result);
        BOOST_TEST_REQUIRE(anti.total_count > 0U);
        const double cos_val = std::cos(2 * plain.param);
        VecD expect = reference.coeffs;
        for (const auto &[base, bits] : anti.blocks) {
            for (uint64_t m = bits; m != 0; m &= m - 1) {
                const size_t i = base + static_cast<size_t>(std::countr_zero(m));
                expect[i] = reference.coeffs[i] * cos_val;
            }
        }
        scan_test::Scenario fused = plain;
        fused.fused_scale = true;
        for (const int threads : {1, 2, 3, 4}) {
            BOOST_TEST_CONTEXT("budget " << threads) {
                const auto out = scan_test::run_scan(solo, fused, {.threads = threads});
                BOOST_TEST(scan_test::bitwise_equal(out.coeffs, expect));
                BOOST_TEST(out.result.cos_blocks.size() >= 1U);
                BOOST_TEST(scan_test::concatenated_cos(out.result).total_count == 0U); // the sweep replaces the set
                // Captured values are the pre-scale coefficients, and the same queries are emitted.
                const auto &a = reference.result.leader_val.at_slot(0);
                const auto &b = out.result.leader_val.at_slot(0);
                BOOST_TEST(scan_test::bitwise_equal(a, b));
                BOOST_TEST((reference.result.leader_src.at_slot(0) == out.result.leader_src.at_slot(0)));
                BOOST_TEST((reference.result.follower_src.at_slot(0) == out.result.follower_src.at_slot(0)));
            }
        }
    }
}

namespace {

constexpr size_t kWide = 16; // 32 Majoranas
using WideAccess = monoprop::detail::PropagatorTestAccess<kWide>;

// About 90k even-weight terms, so the scan's operator spans at least two fold blocks from the first gate and
// more as gates insert partners; gates of weights 2-4 with two repeats, and one at theta = pi/4.
auto wide_case() -> CaseData {
    CaseData data;
    data.num_modes = kWide;
    kernel_test::SplitMix rng{0x5CA2ULL};
    const auto draw_indices = [&rng](size_t weight) {
        VecZ idx;
        while (idx.size() < weight) {
            const size_t m = rng.below(2 * kWide);
            if (std::ranges::find(idx, m) == idx.end()) {
                idx.push_back(m);
            }
        }
        std::ranges::sort(idx);
        return idx;
    };
    while (data.hamiltonian.size() < 90000) {
        const auto idx = draw_indices(2 * (1 + rng.below(3)));
        const double x = static_cast<double>(rng.below(1U << 20)) / (1U << 20) - 0.5;
        data.hamiltonian[idx] = x * hermitian_coefficient<kWide>(indices_to_bitset<kWide>(idx));
    }
    data.initial_state = {0, 2, 4, 6, 8, 10, 12, 14};
    constexpr size_t gates = 6;
    for (size_t g = 0; g < gates; ++g) {
        data.majoranas.push_back(draw_indices(2 + rng.below(3)));
        data.param_inds.push_back(g);
        data.gen_coeffs.push_back(1.0);
        data.parameters.push_back(g == 2 ? std::numbers::pi / 4 : 0.1 + static_cast<double>(rng.below(900)) / 1000.0);
    }
    data.majoranas[4] = data.majoranas[1];
    return data;
}

auto wide_prototype(const CaseData &data, bool schrodinger, int budget, MPI_Comm comm) -> MonomialPropagator<kWide> {
    // The Schrodinger state starts from the whole paired basis, 2^16 rows (one fold block), and grows past it.
    const SimulatorConfig cfg{.schrodinger_cutoff = schrodinger ? std::optional<unsigned int>(2 * kWide) : std::nullopt,
                              .comm = comm,
                              .atol = 1e-12};
    const kernel_test::ScopedBudget scoped(budget);
    MonomialPropagator<kWide> sim(data.hamiltonian,
                                  /*cutoff=*/10,
                                  data.initial_state,
                                  cfg.schrodinger_cutoff,
                                  cfg.comm,
                                  cfg.atol,
                                  cfg.upper_atol,
                                  cfg.cutoff_type,
                                  cfg.basis_change,
                                  kWide,
                                  Basis::Majorana,
                                  /*partitions=*/1);
    BOOST_TEST_REQUIRE(WideAccess::options(sim).threads == budget);
    return sim;
}

// Rows in row-ID order: the store's IDs are its insertion order, so equal rows mean equal IDs.
auto same_rows(const MonomialPropagator<kWide> &a, const MonomialPropagator<kWide> &b) -> bool {
    const auto &sa = *a.mp_op().store;
    const auto &sb = *b.mp_op().store;
    if (sa.size() != sb.size()) {
        return false;
    }
    for (size_t i = 0; i < sa.size(); ++i) {
        if (materialize_row<kWide>(sa, i) != materialize_row<kWide>(sb, i)) {
            return false;
        }
    }
    return true;
}

auto same_core(const LayerCore &a, const LayerCore &b) -> bool {
    const auto &x = a.cross_rank;
    const auto &y = b.cross_rank;
    if (x.occupied.size() != y.occupied.size()) {
        return false;
    }
    for (size_t k = 0; k < x.occupied.size(); ++k) {
        if (x.occupied[k].slot != y.occupied[k].slot || x.occupied[k].sin_send_count != y.occupied[k].sin_send_count
            || x.occupied[k].in_count != y.occupied[k].in_count) {
            return false;
        }
    }
    return x.sin_send_indices == y.sin_send_indices
           && x.sin_recv_phases.uses_binary_phases == y.sin_recv_phases.uses_binary_phases
           && x.sin_recv_phases.total_count == y.sin_recv_phases.total_count
           && x.sin_recv_phases.phase_words == y.sin_recv_phases.phase_words
           && x.sin_recv_phases.phase_values == y.sin_recv_phases.phase_values && x.world_size == y.world_size
           && x.self_pos == y.self_pos && x.self_offset == y.self_offset && a.generator_words == b.generator_words
           && a.scaled_count == b.scaled_count && a.param_index == b.param_index
           && std::bit_cast<uint64_t>(a.gen_coeff) == std::bit_cast<uint64_t>(b.gen_coeff)
           && a.gate_index == b.gate_index;
}

auto same_graph(const MPGraph &a, const MPGraph &b) -> bool {
    if (a.layers() != b.layers()) {
        return false;
    }
    for (size_t l = 0; l < a.layers(); ++l) {
        const auto &la = a.get_layer(l);
        const auto &lb = b.get_layer(l);
        if (!same_core(la.core(), lb.core()) || (la.pruned_cos() == nullptr) != (lb.pruned_cos() == nullptr)) {
            return false;
        }
    }
    return true;
}

} // namespace

// Full construction at fixed geometry: prototypes at budgets 2-4 build the same graph (every layer's
// endpoints, phases, slots, generator and scaled_count), the same rows in the same row-ID order and the same
// coefficients as budget 1, and propagate (fused sweep and the two-pass length-cap path) to the same rows
// and coefficients, in both pictures. Under an MPI launch each rank compares its own store and graph.
BOOST_AUTO_TEST_CASE(openmp_scan_threaded_construction_matches_serial_exactly) {
    const auto data = wide_case();
    for (const bool schrodinger : {false, true}) {
        auto serial_graph = wide_prototype(data, schrodinger, 1, MPI_COMM_WORLD);
        serial_graph.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        BOOST_TEST_MESSAGE("schrodinger=" << schrodinger << " graph-built rows on this rank: " << serial_graph.size()
                                          << ", layers: " << serial_graph.graph().layers());
        std::vector<MonomialPropagator<kWide>> serial_prop;
        for (const bool cap : {false, true}) {
            serial_prop.push_back(wide_prototype(data, schrodinger, 1, MPI_COMM_WORLD));
            serial_prop.back().propagate(data.majoranas,
                                         data.param_inds,
                                         data.gen_coeffs,
                                         data.parameters,
                                         cap ? std::optional<size_t>(2 * kWide) : std::nullopt);
        }
        for (const int threads : {2, 3, 4}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " budget=" << threads) {
                auto threaded = wide_prototype(data, schrodinger, threads, MPI_COMM_WORLD);
                threaded.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
                BOOST_TEST(same_rows(serial_graph, threaded));
                BOOST_TEST(same_graph(serial_graph.graph(), threaded.graph()));
                BOOST_TEST(
                    bitwise_equal(WideAccess::picture_coeffs(serial_graph), WideAccess::picture_coeffs(threaded)));
                BOOST_TEST(bitwise_equal(serial_graph.contract_partially(data.parameters, false),
                                         threaded.contract_partially(data.parameters, false)));
                for (const bool cap : {false, true}) {
                    auto prop = wide_prototype(data, schrodinger, threads, MPI_COMM_WORLD);
                    prop.propagate(data.majoranas,
                                   data.param_inds,
                                   data.gen_coeffs,
                                   data.parameters,
                                   cap ? std::optional<size_t>(2 * kWide) : std::nullopt);
                    const auto &ref = serial_prop[cap ? 1 : 0];
                    BOOST_TEST(same_rows(ref, prop));
                    BOOST_TEST(bitwise_equal(WideAccess::picture_coeffs(ref), WideAccess::picture_coeffs(prop)));
                }
            }
        }
    }
}

// The real build_layer on a prototype's operator splits the scan into ranges that run inside its own region.
BOOST_AUTO_TEST_CASE(openmp_scan_construction_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    const auto data = wide_case();
    auto sim = wide_prototype(data, false, 4, MPI_COMM_SELF);
    const size_t words = sim.mp_op().inverted_index().words();
    const size_t expected =
        std::min<size_t>(4, monoprop::detail::logical_ranges(words, monoprop::detail::kColumnBlockWords));
    BOOST_TEST_REQUIRE(expected >= 2U);
    kernel_test::RangeLog log;
    (void)WideAccess::build_layer_observed(sim, data.majoranas[0], kernel_test::RecordingObserver{&log});
    kernel_test::check_participation(log[monoprop::detail::KernelRange::scan], expected, "build_layer scan");
    BOOST_TEST(!WideAccess::is_invalid(sim));
}

// A scan worker's failure inside the real build path is joined and rethrown on the caller; the propagator's
// operation guard then invalidates the object, and later operations and copies reject it.
BOOST_AUTO_TEST_CASE(openmp_scan_worker_failure_invalidates_the_propagator) {
    const auto data = wide_case();
    auto sim = wide_prototype(data, false, 3, MPI_COMM_SELF);
    kernel_test::RangeLog log;
    BOOST_CHECK_THROW((void)WideAccess::build_layer_observed(
                          sim,
                          data.majoranas[0],
                          kernel_test::RecordingObserver{&log, monoprop::detail::KernelRange::scan, 1}),
                      std::runtime_error);
    BOOST_TEST(log[monoprop::detail::KernelRange::scan].visits.at(1) == 1);
    BOOST_TEST(WideAccess::is_invalid(sim));
    BOOST_CHECK_THROW(sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs), InvalidPropagatorError);
    BOOST_CHECK_THROW((void)WideAccess::clone(sim), InvalidPropagatorError);
}
