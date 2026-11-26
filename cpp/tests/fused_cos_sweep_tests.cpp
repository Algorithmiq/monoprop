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
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

#include "monoprop/detail/sharded/RootObserver.h"

#include "KernelTestSupport.h"
#include "PropagatorTestAccess.h"
#include "ScanTestSupport.h"
#include "ShardedTestSupport.h"
#include "TestUtilities.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"

// The fused cos sweep's one deliberate FP deviation from the two-pass path (≤1 ulp per hit endpoint,
// from resolve's stored·(1/cos) recovery), against the build_graph()+replay evaluation as oracle.
//
// A budget is a shard count, which no case changes inside a process: the fused-versus-replay cases run at the launch's
// T (1 per case, 2 and 4 in fused_env_t*), the real fused records are read per shard from the propagation seam itself,
// and fixed-geometry runs must agree bitwise shard by shard. Across geometries only complete retained maps are
// comparable (tests/test_sharded_openmp.py).

namespace {

using namespace test_utils;
using namespace monoprop;
using scan_test::bitwise_equal;

// The two paths accumulate FP differently even before the sweep, so demand 1e-12: tight enough that a
// wrong cos factor on any endpoint (relative error O(1)) fails loudly, loose enough for reordering.
constexpr double agree_atol = 1e-12;

// Every root captures the launch's shard count T (sharded_test::team_size()), and no case changes it.
template <size_t NumModes>
auto build_root(const CaseData &data, const SimulatorConfig &cfg) -> MonomialPropagator<NumModes> {
    auto sim = build_simulator<NumModes>(data, cfg);
    BOOST_TEST_REQUIRE(static_cast<size_t>(monoprop::detail::PropagatorTestAccess<NumModes>::options(sim).threads)
                       == sharded_test::team_size());
    return sim;
}

// In-place propagation against graph replay, on roots at the launch's T; `cap` is only_rotate_len_k for both.
void check_agreement(const CaseData &data,
                     const SimulatorConfig &cfg,
                     const char *label,
                     std::optional<size_t> cap = std::nullopt) {
    constexpr size_t n_modes = ExampleDataFix::n_modes;
    auto inplace_sim = build_root<n_modes>(data, cfg);
    inplace_sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
    const double inplace = inplace_sim.expectation_value_functional(std::nullopt)(VecD{});
    auto graph_sim = build_root<n_modes>(data, cfg);
    graph_sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, std::nullopt, cap);
    const double graph = graph_sim.expectation_value_functional(std::nullopt)(data.parameters);
    BOOST_TEST_CONTEXT(label << " budget=" << sharded_test::team_size() << " inplace=" << inplace
                             << " graph=" << graph) {
        BOOST_CHECK_SMALL(inplace - graph, agree_atol);
        BOOST_CHECK_SMALL(inplace - data.actual_expval, sharded_test::exact_atol);
    }
}

// --- a workload large enough for the threaded kernels ---------------------------------------------------------

constexpr size_t big_modes = 10; // 20 Majoranas
using BigAccess = monoprop::detail::PropagatorTestAccess<big_modes>;

// A synthetic Hermitian operator of `terms` even-weight terms on the state filling M's even Majoranas, and `gates`
// gates of weights 2-4 (odd included). Gate `pi_gate` sits at theta = pi/4, where the fused sweep recovers pre-cos
// values through 1/cos ~ 1.6e16. Each (later, earlier) repeat gives gate `later` the generator of gate `earlier`.
template <size_t M>
auto synthetic_case(uint64_t seed,
                    size_t terms,
                    size_t gates,
                    size_t pi_gate,
                    std::initializer_list<std::pair<size_t, size_t>> repeats) -> CaseData {
    CaseData data;
    data.num_modes = M;
    kernel_test::SplitMix rng{seed};
    const auto draw_indices = [&rng](size_t weight) {
        VecZ idx;
        while (idx.size() < weight) {
            const size_t m = rng.below(2 * M);
            if (std::ranges::find(idx, m) == idx.end()) {
                idx.push_back(m);
            }
        }
        std::ranges::sort(idx);
        return idx;
    };
    while (data.hamiltonian.size() < terms) {
        const auto idx = draw_indices(2 * (1 + rng.below(3)));
        const double x = static_cast<double>(rng.below(1U << 20)) / (1U << 20) - 0.5;
        data.hamiltonian[idx] = x * hermitian_coefficient<M>(indices_to_bitset<M>(idx));
    }
    for (size_t m = 0; m < M / 2; ++m) {
        data.initial_state.push_back(2 * m);
    }
    for (size_t g = 0; g < gates; ++g) {
        data.majoranas.push_back(draw_indices(2 + rng.below(3)));
        data.param_inds.push_back(g);
        data.gen_coeffs.push_back(1.0);
        data.parameters.push_back(g == pi_gate ? std::numbers::pi / 4
                                               : 0.1 + static_cast<double>(rng.below(900)) / 1000.0);
    }
    for (const auto &[later, earlier] : repeats) {
        data.majoranas[later] = data.majoranas[earlier];
    }
    return data;
}

// 16k terms, so fused records and replayed folds span several logical ranges. Repeated generators: rotating again
// by G targets the M*G terms the first rotation inserted, so both pictures produce hit records as well as inserts.
auto big_case() -> CaseData {
    return synthetic_case<big_modes>(2026, 16000, 9, 4, {{6, 2}, {8, 3}});
}

auto big_root(const CaseData &data, bool schrodinger) -> MonomialPropagator<big_modes> {
    return build_root<big_modes>(
        data,
        SimulatorConfig{.schrodinger_cutoff = schrodinger ? std::optional<unsigned int>(2 * big_modes) : std::nullopt,
                        .comm = MPI_COMM_WORLD});
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
    check_agreement(data, SimulatorConfig{}, "two-pass heisenberg", cap);
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes}, "two-pass schrodinger", cap);
}

namespace {

namespace sharded = monoprop::detail::sharded;

// One shard's fused records for one gate, copied from the propagation seam just before production applied them.
struct CapturedGate {
    size_t step = 0;
    int worker = -1;
    monoprop::detail::FusedContract fc;
    CosMask cos;
    VecD before;
    double apply_angle = 0.0;
    bool schrodinger = false;
    bool fused_scale = false;
};

// Copies every owner's real records. Each owner appends to its own shard's list only; the test reads them after the
// propagation's team has joined.
class RecordCapture final : public sharded::RootObserver {
public:
    explicit RecordCapture(size_t shards) : gates_(shards) {}
    auto fused_records(size_t step, size_t shard, const sharded::FusedGateView &gate) const -> void override {
        gates_.at(shard).push_back(CapturedGate{.step = step,
                                                .worker = omp_get_thread_num(),
                                                .fc = gate.records,
                                                .cos = gate.cos,
                                                .before = VecD(gate.coeffs.begin(), gate.coeffs.end()),
                                                .apply_angle = gate.apply_angle,
                                                .schrodinger = gate.schrodinger,
                                                .fused_scale = gate.fused_scale});
    }
    [[nodiscard]] auto gates() const -> const std::vector<std::vector<CapturedGate>> & { return gates_; }

private:
    mutable std::vector<std::vector<CapturedGate>> gates_;
};

template <size_t N>
auto picture_of(const monoprop::detail::MPOperator<N> &op, bool schrodinger) -> const VecD & {
    return schrodinger ? op.state_coeffs : op.op_coeffs;
}

// Rows in row-ID order: a store's IDs are its insertion order, so equal rows mean equal IDs.
template <size_t N>
auto same_store_rows(const monoprop::detail::OperatorIndex<N> &sa, const monoprop::detail::OperatorIndex<N> &sb)
    -> bool {
    if (sa.size() != sb.size()) {
        return false;
    }
    for (size_t i = 0; i < sa.size(); ++i) {
        if (materialize_row<N>(sa, i) != materialize_row<N>(sb, i)) {
            return false;
        }
    }
    return true;
}

template <class Observer = monoprop::detail::NoRangeObserver>
auto apply_captured(const CapturedGate &gate, int threads, const Observer &observer = {}) -> VecD {
    return kernel_test::apply_fused_copy(gate.fc,
                                         gate.before,
                                         gate.cos,
                                         gate.apply_angle,
                                         gate.schrodinger,
                                         gate.fused_scale,
                                         threads,
                                         observer);
}

} // namespace

// Records from the root's real propagation, read per shard from the seam itself, in both pictures and in the
// fused-scale and two-pass (length cap) modes: every destination slot of a shard has exactly one add-owner across
// hits, inserts and cross-owner halves; each owner applies its records on its own worker; replaying a gate's records
// gives the next gate's starting coefficients and, after the last gate, the shard's picture, bitwise; and the apply
// kernel at budgets 2-4 (run here, outside the team) matches budget 1 bitwise. Halves appear exactly when there is
// another flat owner (P x T > 1). Under an MPI launch each rank audits its own shards.
BOOST_AUTO_TEST_CASE(fused_shard_records_have_one_add_owner_and_apply_exactly) {
    const auto data = big_case();
    const size_t launch = sharded_test::team_size();
    const auto flat_world = static_cast<size_t>(comm_size(MPI_COMM_WORLD)) * launch;
    for (const bool schrodinger : {false, true}) {
        for (const bool length_cap : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " length_cap=" << length_cap << " T=" << launch) {
                auto sim = big_root(data, schrodinger);
                const auto &shards = BigAccess::shards(sim);
                const RecordCapture capture(shards.size());
                BigAccess::observe(sim, &capture);
                const std::optional<size_t> cap = length_cap ? std::optional<size_t>(2 * big_modes) : std::nullopt;
                sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
                BigAccess::observe(sim, nullptr);

                size_t duplicate_owners = 0;
                size_t hits = 0;
                size_t inserts = 0;
                size_t halves = 0;
                size_t foreign_workers = 0;
                size_t replay_mismatches = 0;
                size_t budget_mismatches = 0;
                size_t max_gate_records = 0;
                size_t max_apply_ranges = 0;
                bool fused_seen = false;
                bool two_pass_seen = false;
                for (size_t t = 0; t < shards.size(); ++t) {
                    const auto &gates = capture.gates()[t];
                    BOOST_TEST_REQUIRE(gates.size() == data.majoranas.size());
                    for (size_t g = 0; g < gates.size(); ++g) {
                        const auto &gate = gates[g];
                        BOOST_TEST(gate.step == g);
                        foreign_workers += static_cast<size_t>(gate.worker != static_cast<int>(t));
                        for (const auto &[slot, count] : kernel_test::add_owner_counts(gate.fc)) {
                            duplicate_owners += static_cast<size_t>(count != 1);
                        }
                        hits += gate.fc.hits.size();
                        inserts += gate.fc.inserts.size();
                        halves += gate.fc.cross_half.size();
                        const size_t records = gate.fc.hits.size() + gate.fc.inserts.size() + gate.fc.cross_half.size();
                        (gate.fused_scale ? fused_seen : two_pass_seen) = true;
                        const VecD serial = apply_captured(gate, 1);
                        for (const int threads : {2, 3, 4}) {
                            kernel_test::RangeLog log;
                            const kernel_test::RecordingObserver observer{&log};
                            budget_mismatches +=
                                static_cast<size_t>(!bitwise_equal(apply_captured(gate, threads, observer), serial));
                            if (records > max_gate_records) {
                                max_gate_records = records;
                                max_apply_ranges = log[monoprop::detail::KernelRange::fused_apply].worker.size();
                            }
                        }
                        // What production applied: the last result is the shard's picture, and a two-pass gate
                        // starts from the previous result, extended by the rows it inserted (a fused sweep scales
                        // its sources during traversal first).
                        if (g + 1 == gates.size() || !gates[g + 1].fused_scale) {
                            const VecD &next =
                                g + 1 < gates.size() ? gates[g + 1].before : picture_of(shards[t]->op, schrodinger);
                            replay_mismatches += static_cast<size_t>(
                                next.size() < serial.size()
                                || std::memcmp(next.data(), serial.data(), serial.size() * sizeof(double)) != 0);
                        }
                    }
                }
                BOOST_TEST_MESSAGE("records: hits=" << hits << " inserts=" << inserts << " halves=" << halves);
                BOOST_TEST(duplicate_owners == 0U);
                BOOST_TEST(foreign_workers == 0U);
                BOOST_TEST(replay_mismatches == 0U);
                BOOST_TEST(budget_mismatches == 0U);
                BOOST_TEST(hits + inserts > 0U);
                BOOST_TEST((halves > 0U) == (flat_world > 1));
                BOOST_TEST(max_apply_ranges == monoprop::detail::logical_ranges(max_gate_records, 1024));
                BOOST_TEST(fused_seen == !length_cap);
                BOOST_TEST(two_pass_seen == length_cap);
            }
        }
    }
}

// At a fixed geometry the root is deterministic shard by shard: two roots propagate (fused and two-pass) and
// build and replay a graph to the same rows in the same row-ID order, bitwise coefficients and contraction blocks,
// and the same energies. Every row sits on the shard its flat owner names, and no key is on two shards.
BOOST_AUTO_TEST_CASE(fused_shards_are_deterministic_at_fixed_geometry) {
    const auto data = big_case();
    const size_t launch = sharded_test::team_size();
    const auto rank = static_cast<size_t>(mpi::rank(mpi::Comm(MPI_COMM_WORLD)));
    for (const bool schrodinger : {false, true}) {
        for (const bool length_cap : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " length_cap=" << length_cap << " T=" << launch) {
                const std::optional<size_t> cap = length_cap ? std::optional<size_t>(2 * big_modes) : std::nullopt;
                auto first = big_root(data, schrodinger);
                auto second = big_root(data, schrodinger);
                first.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
                second.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
                const auto &a = BigAccess::shards(first);
                const auto &b = BigAccess::shards(second);
                const auto &router = BigAccess::router(first);
                std::unordered_set<Monomial<big_modes>, MonomialHash<big_modes>> keys;
                size_t misrouted = 0;
                for (size_t t = 0; t < a.size(); ++t) {
                    BOOST_TEST(same_store_rows(*a[t]->op.store, *b[t]->op.store));
                    BOOST_TEST(bitwise_equal(picture_of(a[t]->op, schrodinger), picture_of(b[t]->op, schrodinger)));
                    a[t]->op.store->for_each([&](const auto &mono, size_t) {
                        misrouted += static_cast<size_t>(router.dest<big_modes>(mono) != rank * a.size() + t);
                        keys.insert(mono);
                    });
                }
                BOOST_TEST(misrouted == 0U);
                BOOST_TEST(keys.size() == first.size());
                BOOST_TEST(first.expectation_value(VecD{}) == second.expectation_value(VecD{}));
            }
        }
        BOOST_TEST_CONTEXT("graph replay, schrodinger=" << schrodinger << " T=" << launch) {
            auto first = big_root(data, schrodinger);
            auto second = big_root(data, schrodinger);
            first.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            second.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            BOOST_TEST(bitwise_equal(first.contract_partially(data.parameters, false),
                                     second.contract_partially(data.parameters, false)));
            BOOST_TEST(first.expectation_value(data.parameters) == second.expectation_value(data.parameters));
            const auto [value, gradient] = first.expectation_value_and_gradient(data.parameters);
            const auto [other, other_gradient] = second.expectation_value_and_gradient(data.parameters);
            BOOST_TEST(value == other);
            BOOST_TEST(bitwise_equal(gradient, other_gradient));
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

constexpr size_t wide_modes = 16; // 32 Majoranas
using WideAccess = monoprop::detail::PropagatorTestAccess<wide_modes>;

// About 90k even-weight terms, so the scan's operator spans at least two fold blocks from the first gate and
// more as gates insert partners.
auto wide_case() -> CaseData {
    return synthetic_case<wide_modes>(0x5CA2ULL, 90000, 6, 2, {{4, 1}});
}

// The root at the launch's T.
auto wide_root(const CaseData &data, bool schrodinger, MPI_Comm comm) -> MonomialPropagator<wide_modes> {
    return MonomialPropagator<wide_modes>(data.hamiltonian,
                                          /*cutoff=*/10,
                                          data.initial_state,
                                          schrodinger ? std::optional<unsigned int>(2 * wide_modes) : std::nullopt,
                                          comm,
                                          /*lower_atol=*/1e-12,
                                          std::nullopt,
                                          CutoffType::Length,
                                          std::nullopt,
                                          wide_modes,
                                          Basis::Majorana);
}

auto same_core(const LayerCore &a, const LayerCore &b) -> bool {
    const auto &x = a.cross_rank;
    const auto &y = b.cross_rank;
    const auto same_slot = [](const auto &p, const auto &q) {
        return p.slot == q.slot && p.sin_send_count == q.sin_send_count && p.in_count == q.in_count;
    };
    return std::ranges::equal(x.occupied, y.occupied, same_slot) && x.sin_send_indices == y.sin_send_indices
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

// Full construction at the launch's fixed geometry, shard by shard: two roots build the same graph on every shard
// (every layer's endpoints, phases, slots, generator and scaled_count), the same rows in the same row-ID order and the
// same coefficients and contraction blocks, and propagate (fused sweep and the two-pass length-cap path) to the same
// rows and coefficients, in both pictures. Under an MPI launch each rank compares its own shards. Changing T changes
// ownership, so other geometries are compared through complete retained maps (tests/test_sharded_openmp.py).
BOOST_AUTO_TEST_CASE(fused_wide_shards_match_at_fixed_geometry) {
    const auto data = wide_case();
    for (const bool schrodinger : {false, true}) {
        BOOST_TEST_CONTEXT("schrodinger=" << schrodinger) {
            auto first = wide_root(data, schrodinger, MPI_COMM_WORLD);
            auto second = wide_root(data, schrodinger, MPI_COMM_WORLD);
            first.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            second.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
            BOOST_TEST_MESSAGE("graph-built rows on this rank: " << first.size()
                                                                 << ", layers: " << first.graph_layers());
            const auto &a = WideAccess::shards(first);
            const auto &b = WideAccess::shards(second);
            BOOST_TEST_REQUIRE(a.size() == b.size());
            for (size_t t = 0; t < a.size(); ++t) {
                BOOST_TEST_CONTEXT("shard " << t) {
                    BOOST_TEST(same_store_rows(*a[t]->op.store, *b[t]->op.store));
                    BOOST_TEST(same_graph(a[t]->graph, b[t]->graph));
                    BOOST_TEST(bitwise_equal(picture_of(a[t]->op, schrodinger), picture_of(b[t]->op, schrodinger)));
                }
            }
            BOOST_TEST(bitwise_equal(first.contract_partially(data.parameters, false),
                                     second.contract_partially(data.parameters, false)));
            for (const bool cap : {false, true}) {
                auto x = wide_root(data, schrodinger, MPI_COMM_WORLD);
                auto y = wide_root(data, schrodinger, MPI_COMM_WORLD);
                const std::optional<size_t> k = cap ? std::optional<size_t>(2 * wide_modes) : std::nullopt;
                x.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, k);
                y.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, k);
                const auto &xs = WideAccess::shards(x);
                const auto &ys = WideAccess::shards(y);
                for (size_t t = 0; t < xs.size(); ++t) {
                    BOOST_TEST_CONTEXT("cap=" << cap << " shard " << t) {
                        BOOST_TEST(same_store_rows(*xs[t]->op.store, *ys[t]->op.store));
                        BOOST_TEST(
                            bitwise_equal(picture_of(xs[t]->op, schrodinger), picture_of(ys[t]->op, schrodinger)));
                    }
                }
            }
        }
    }
}
