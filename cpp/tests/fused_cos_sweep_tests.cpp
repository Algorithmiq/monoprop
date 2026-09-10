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

#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>

#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/evolution/layer_build/GateSinks.h"

#include "TestUtilities.h"

// The fused cos sweep's one deliberate FP deviation from the two-pass path (≤1 ulp per hit endpoint,
// from resolve's stored·(1/cos) recovery), against the build_graph()+replay evaluation as oracle.

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

void check_agreement(const CaseData &data, const SimulatorConfig &cfg, const char *label) {
    const double inplace = inplace_energy<ExampleDataFix::n_modes>(data, cfg);
    const double graph = graph_energy<ExampleDataFix::n_modes>(data, cfg);
    BOOST_TEST_CONTEXT(label << " inplace=" << inplace << " graph=" << graph) {
        BOOST_CHECK_SMALL(inplace - graph, kAgreeAtol);
        BOOST_CHECK_SMALL(inplace - data.actual_expval, kExactAtol);
    }
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

// Which endpoint of a pair reads a RECOVERED partner value, stored*(1/cos), and which reads the record's
// exact pre-cos double (GateSinks.h ContractSink::hit has the argument). The partner value below is a
// measured one on which the recovery is not the identity, so each check can actually fail.
BOOST_AUTO_TEST_CASE(fused_sink_recovers_only_the_found_endpoints_value) {
    constexpr double kParam = 0.01976005125;
    constexpr double kPartner = -0.9945421981876392;
    const double cos_build = std::cos(2 * kParam);
    const double inv_cos = 1.0 / cos_build;
    const double swept = cos_build * kPartner; // what the scan leaves in the partner's own slot
    const double recovered = swept * inv_cos;
    const auto bits = [](double x) { return std::bit_cast<uint64_t>(x); };
    BOOST_REQUIRE(bits(recovered) != bits(kPartner));

    monoprop::VecD coeffs{swept};
    monoprop::detail::FusedContract fc;
    fc.halves.reserve(4);
    monoprop::detail::ContractSink<ExampleDataFix::n_modes> sink{.fc = fc,
                                                                 .fused_scale = true,
                                                                 .op_coeffs = coeffs,
                                                                 .cos_build = cos_build,
                                                                 .inv_cos = inv_cos};
    sink.hit(0, 0, kPartner, 1, /*foll=*/false, /*own_rot=*/true);  // emitting leader: recovered
    sink.hit(0, 0, kPartner, 1, /*foll=*/true, /*own_rot=*/true);   // follower: the record's own double
    sink.hit(0, 0, kPartner, 1, /*foll=*/false, /*own_rot=*/false); // silent leader: its partner queried
    BOOST_REQUIRE_EQUAL(fc.halves.size(), 3u);
    BOOST_CHECK_EQUAL(bits(fc.halves[0].v_partner), bits(recovered));
    BOOST_CHECK_EQUAL(bits(fc.halves[1].v_partner), bits(kPartner));
    BOOST_CHECK_EQUAL(bits(fc.halves[2].v_partner), bits(kPartner));
    // A response's payload is always the found endpoint's own slot, recovered off the sweep.
    BOOST_CHECK_EQUAL(bits(sink.silent_value(0)), bits(recovered));

    // Without the sweep nothing is recovered: the array is pre-gate and the record already exact.
    monoprop::detail::FusedContract fc2;
    fc2.halves.reserve(2);
    monoprop::detail::ContractSink<ExampleDataFix::n_modes> two_pass{.fc = fc2,
                                                                     .fused_scale = false,
                                                                     .op_coeffs = coeffs,
                                                                     .cos_build = cos_build,
                                                                     .inv_cos = inv_cos};
    two_pass.hit(0, 0, kPartner, 1, /*foll=*/false, /*own_rot=*/true);
    BOOST_CHECK_EQUAL(bits(fc2.halves[0].v_partner), bits(kPartner));
    BOOST_CHECK_EQUAL(bits(two_pass.silent_value(0)), bits(swept));
}

// The apply's rounding shape is pinned, not inferred. `c += sin*phi*v` and `fma(sin*phi, v, c)` are the
// same expression and differ by 1 ULP, and which one a merged loop compiles to is the optimiser's choice;
// these two (swept coefficient, partner value) pairs are a measured first divergence of
// lih_fermionic_spin_exact under the Schrodinger cutoff, where the two shapes disagree.
BOOST_AUTO_TEST_CASE(fused_apply_rounds_each_arm_once) {
    // The layer's angle, from that fixture's first gate: cos(2p) = 0.9992191823830495.
    constexpr double kParam = 0.01976005125;
    const double cos_val = std::cos(2 * kParam);
    const double sin_val = std::sin(2 * kParam);

    struct Pair {
        double pre_cos; // the coefficient as the wire carries it, before the gate's cos sweep
        double partner; // v_partner
    };
    constexpr Pair kPairs[] = {
        {-0.0789579320133996, -0.9976581568252995},
        {0.0394789660066998, 0.9968779488844999},
    };

    for (const Pair &tc : kPairs) {
        const double swept = cos_val * tc.pre_cos; // what the scan leaves in a pre-gate slot
        const double sin_phase = sin_val * 1.0;    // phi = +1
        // volatile pins the intermediate roundings, so the reference for "rounds twice" cannot itself be
        // contracted into the fma it is meant to differ from.
        volatile const double sine_add = sin_phase * tc.partner;
        const double rounds_twice = swept + sine_add;
        const double rounds_once = std::fma(sin_phase, tc.partner, swept);

        monoprop::detail::FusedContract fc;
        fc.halves.push_back({.local_idx = 0, .phase_signed = 1, .is_insert = false, .v_partner = tc.partner});
        monoprop::VecD coeffs{swept};
        monoprop::detail::apply_fused_contract(fc, coeffs, monoprop::CosMask{}, kParam, true);

        BOOST_TEST_CONTEXT("plain arm pre_cos=" << tc.pre_cos << " partner=" << tc.partner) {
            // A pair on which the two shapes agree would make the check below vacuous.
            BOOST_CHECK(std::bit_cast<uint64_t>(rounds_once) != std::bit_cast<uint64_t>(rounds_twice));
            BOOST_CHECK_EQUAL(std::bit_cast<uint64_t>(coeffs[0]), std::bit_cast<uint64_t>(rounds_once));
        }

        // The insert arm folds the gate's cos in itself, with the sine product rounded before the sum.
        monoprop::detail::FusedContract fc_mint;
        fc_mint.halves.push_back({.local_idx = 0, .phase_signed = 1, .is_insert = true, .v_partner = tc.partner});
        monoprop::VecD mint{tc.pre_cos};
        monoprop::detail::apply_fused_contract(fc_mint, mint, monoprop::CosMask{}, kParam, true);
        const double mint_once = std::fma(cos_val, tc.pre_cos, sin_phase * tc.partner);
        BOOST_TEST_CONTEXT("insert arm pre_cos=" << tc.pre_cos) {
            BOOST_CHECK_EQUAL(std::bit_cast<uint64_t>(mint[0]), std::bit_cast<uint64_t>(mint_once));
        }
    }
}
