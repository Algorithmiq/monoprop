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

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "PauliTestOracle.h"
#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/PauliAlgebra.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;
using namespace pauli_oracle;

namespace {

// The JW basis-change table as a Python-facing index list (each basis vector -> its gamma indices),
// suitable for the MonomialPropagator basis_change parameter.
template <size_t NumModes>
auto jw_basis_indices(size_t n) -> std::vector<VecZ> {
    std::vector<VecZ> table(2 * NumModes);
    for (size_t i = 0; i < n; ++i) {
        VecZ z_str;
        for (size_t z = 0; z < 2 * i; ++z) {
            z_str.push_back(z);
        }
        VecZ even_vec = z_str;
        even_vec.push_back(2 * i);
        VecZ odd_vec = z_str;
        odd_vec.push_back(2 * i + 1);
        table[2 * i] = even_vec;
        table[2 * i + 1] = odd_vec;
    }
    // The inactive high modes map to themselves (identity), matching indices_to_bitset of a lone slot.
    for (size_t s = 2 * n; s < 2 * NumModes; ++s) {
        table[s] = VecZ{s};
    }
    return table;
}

// ── Native Pauli propagator drivers ──────────────────────────────────────────────────────────

// Build a native Pauli propagator over a string->real observable.
template <size_t N>
auto build_pauli_sim(const std::map<std::string, double> &obs,
                     unsigned int cutoff,
                     std::optional<unsigned int> schrodinger_cutoff = std::nullopt,
                     const VecZ &slater = {},
                     std::optional<double> lower_atol = std::nullopt) -> MonomialPropagator<N> {
    FermiOperatorMap init;
    for (const auto &[p, c] : obs) {
        init[slots_of_string(p)] = cd(c, 0.0);
    }
    return MonomialPropagator<N>(init,
                                 cutoff,
                                 slater,
                                 schrodinger_cutoff,
                                 MPI_COMM_SELF,
                                 lower_atol,
                                 std::nullopt,
                                 CutoffType::Support,
                                 std::nullopt,
                                 N,
                                 Basis::Pauli);
}

// Dense matrix of the propagator's current (Heisenberg) operator, decoded term-by-term.
template <size_t N>
auto dense_operator(MonomialPropagator<N> &mp) -> std::vector<cd> {
    const size_t d = size_t{1} << N;
    std::vector<cd> m(d * d, cd(0, 0));
    const auto &coeffs = mp.mp_op().get_operator();
    mp.indexing().for_each([&](const MajoranaSet<N> &maj, size_t idx) {
        if (idx >= coeffs.size()) {
            return;
        }
        const double c = coeffs[idx];
        if (c == 0.0) {
            return;
        }
        std::string s(N, 'I');
        for (size_t q = 0; q < N; ++q) {
            s[q] = letter_from_bitset<N>(maj, q);
        }
        const auto pm = matrix_from_string(s);
        for (size_t k = 0; k < d * d; ++k) {
            m[k] += c * pm[k];
        }
    });
    return m;
}

// Dense observable from string->real coefficients.
template <size_t N>
auto dense_observable(const std::map<std::string, double> &obs) -> std::vector<cd> {
    const size_t d = size_t{1} << N;
    std::vector<cd> m(d * d, cd(0, 0));
    for (const auto &[p, c] : obs) {
        const auto pm = matrix_from_string(p);
        for (size_t k = 0; k < d * d; ++k) {
            m[k] += c * pm[k];
        }
    }
    return m;
}

// T7 sub-case: apply one native Pauli gate exp(+i·g·θ·G) and compare the propagated operator to the
// dense ground truth U† O U (Heisenberg). Any single Pauli G has G² = I, so
// U = cos(gθ) I + i sin(gθ) G exactly (no general matrix-exp needed); U† = cos(gθ) I − i sin(gθ) G.
template <size_t N>
auto check_pauli_gate(const std::map<std::string, double> &obs, const std::string &gstr, double g, double theta)
    -> void {
    auto mp = build_pauli_sim<N>(obs, /*cutoff=*/2 * N);
    mp.propagate({slots_of_string(gstr)}, VecZ{0}, VecD{g}, VecD{theta});
    const auto engine = dense_operator<N>(mp);

    const size_t d = size_t{1} << N;
    const auto O = dense_observable<N>(obs);
    const auto G = matrix_from_string(gstr);
    const double param = g * theta;
    const double cs = std::cos(param);
    const double sn = std::sin(param);
    std::vector<cd> U(d * d, cd(0, 0));
    std::vector<cd> Ud(d * d, cd(0, 0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            const cd gij = G[i * d + j];
            const cd diag = (i == j) ? cd(cs, 0) : cd(0, 0);
            U[i * d + j] = diag + cd(0, sn) * gij;
            Ud[i * d + j] = diag - cd(0, sn) * gij; // U† = cos I − i sin G (G Hermitian)
        }
    }
    const auto ref = matmul(matmul(Ud, O, d), U, d);
    BOOST_TEST_CONTEXT("N=" << N << " G=" << gstr << " g=" << g << " theta=" << theta) {
        BOOST_TEST(approx_equal(engine, ref));
    }
}

// ── Jordan-Wigner Majorana arm (mirrors src/monoprop/{conversion_utils,circuit}.py) ──────────

// _pauli_to_fermi(pauli): JW-image gamma indices (ascending) plus the phase coeff, exactly as Python.
auto pauli_to_fermi_full(const std::string &pauli) -> std::pair<VecZ, cd> {
    std::vector<size_t> acc;
    bool flag_z = false;
    cd coeff(1.0, 0.0);
    for (int i = static_cast<int>(pauli.size()) - 1; i >= 0; --i) {
        const char p = pauli[static_cast<size_t>(i)];
        const auto ii = static_cast<size_t>(i);
        if ((p == 'Z' && !flag_z) || (p == 'I' && flag_z)) {
            acc.push_back(2 * ii + 1);
            acc.push_back(2 * ii);
            coeff *= cd(0, -1);
        }
        else if (p == 'X' && !flag_z) {
            acc.push_back(2 * ii);
            flag_z = true;
        }
        else if (p == 'X' && flag_z) {
            acc.push_back(2 * ii + 1);
            flag_z = false;
            coeff *= cd(0, -1);
        }
        else if (p == 'Y' && !flag_z) {
            acc.push_back(2 * ii + 1);
            flag_z = true;
        }
        else if (p == 'Y' && flag_z) {
            acc.push_back(2 * ii);
            flag_z = false;
            coeff *= cd(0, 1);
        }
    }
    return {VecZ(acc.rbegin(), acc.rend()), coeff};
}

// _antihermitian_gen_coeff(majorana, coeff): real structural generator coeff = Re(-coeff / i^{L(L-1)/2}).
auto antiherm_gen_coeff(size_t majorana_len, cd coeff) -> double {
    static const cd ipow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    const size_t e = (majorana_len * (majorana_len - 1) / 2) % 4;
    const cd gen = -coeff / ipow[e];
    return gen.real();
}

// A logical circuit of single Pauli-string gates exp(i·g·θ·G).
struct PauliCircuit {
    std::vector<std::string> gens; // per gate: the Pauli string generator
    VecD gs;                       // per gate: gen_coeff g
    VecZ param_map;                // per gate: parameter index
    VecD params;                   // parameter values
};

// Native gate arrays for the circuit: generator = native slots, gen_coeff = g.
auto native_gate_arrays(const PauliCircuit &c) -> std::pair<std::vector<VecZ>, VecD> {
    std::vector<VecZ> majs;
    VecD gcs;
    for (size_t k = 0; k < c.gens.size(); ++k) {
        majs.push_back(slots_of_string(c.gens[k]));
        gcs.push_back(c.gs[k]);
    }
    return {majs, gcs};
}

// JW gate arrays: generator = JW image, gen_coeff = antihermitian-normalized (Re(-g·jw / i^{L(L-1)/2})).
auto jw_gate_arrays(const PauliCircuit &c) -> std::pair<std::vector<VecZ>, VecD> {
    std::vector<VecZ> majs;
    VecD gcs;
    for (size_t k = 0; k < c.gens.size(); ++k) {
        const auto [idx, jw] = pauli_to_fermi_full(c.gens[k]);
        majs.push_back(idx);
        gcs.push_back(antiherm_gen_coeff(idx.size(), c.gs[k] * jw));
    }
    return {majs, gcs};
}

// Build the JW-image Majorana propagator representing the SAME physical observable, with the JW basis
// change so its Support cutoff measures Pauli weight (matching the native arm).
template <size_t N>
auto build_jw_sim(const std::map<std::string, double> &obs,
                  unsigned int cutoff,
                  std::optional<unsigned int> schrodinger_cutoff = std::nullopt,
                  const VecZ &slater = {},
                  std::optional<double> lower_atol = std::nullopt) -> MonomialPropagator<N> {
    FermiOperatorMap init;
    for (const auto &[p, c] : obs) {
        const auto [idx, jw] = pauli_to_fermi_full(p);
        init[idx] = jw * cd(c, 0.0);
    }
    return MonomialPropagator<N>(init,
                                 cutoff,
                                 slater,
                                 schrodinger_cutoff,
                                 MPI_COMM_SELF,
                                 lower_atol,
                                 std::nullopt,
                                 CutoffType::Support,
                                 jw_basis_indices<N>(N),
                                 N,
                                 Basis::Majorana);
}

} // namespace

// The repo's ctest discovery treats every --list_content line as a top-level test name, so cases use a
// flat shared prefix (pauli_build_layer_*) instead of a BOOST_AUTO_TEST_SUITE. Run with
// --run_test=pauli_build_layer_*.

// T7 (MANDATORY): dense-matrix ground truth — pins the emit sign (step A3 of the wiring).
BOOST_AUTO_TEST_CASE(pauli_build_layer_dense_matrix_ground_truth) {
    // n = 2: single- and two-qubit generators, Y-heavy observables, several angles.
    const std::map<std::string, double> o2{{"XY", 0.5}, {"ZZ", -0.3}, {"YX", 0.7}, {"IZ", 0.2}, {"YY", -0.15}};
    for (double th : {0.37, 0.8, 1.3, -0.6}) {
        check_pauli_gate<2>(o2, "XX", 1.0, th);
        check_pauli_gate<2>(o2, "ZZ", 0.9, th);
        check_pauli_gate<2>(o2, "XY", 1.1, th);
        check_pauli_gate<2>(o2, "YZ", 0.5, th);
        check_pauli_gate<2>(o2, "XI", 1.0, th);
        check_pauli_gate<2>(o2, "IY", 1.0, th);
        check_pauli_gate<2>(o2, "ZI", 0.7, th);
    }

    // n = 3: Y-heavy observable, various generators.
    const std::map<std::string, double> o3{{"XYZ", 0.4},
                                           {"YYY", -0.6},
                                           {"ZIZ", 0.25},
                                           {"IYX", 0.5},
                                           {"ZZI", -0.35},
                                           {"XXX", 0.2}};
    for (double th : {0.41, -0.9, 1.05}) {
        check_pauli_gate<3>(o3, "XZI", 1.0, th);
        check_pauli_gate<3>(o3, "YIY", 0.8, th);
        check_pauli_gate<3>(o3, "ZZZ", 0.6, th);
        check_pauli_gate<3>(o3, "IXY", 1.0, th);
        check_pauli_gate<3>(o3, "YYZ", 0.5, th);
    }

    // n = 4: random Hermitian observables and random generators.
    std::mt19937 rng(0xB0A710U);
    for (size_t trial = 0; trial < 40; ++trial) {
        std::map<std::string, double> o4;
        const size_t nterms = 3 + (rng() % 5);
        std::uniform_real_distribution<double> coeff(-1.0, 1.0);
        for (size_t t = 0; t < nterms; ++t) {
            std::string p = random_string(rng, 4);
            if (p == "IIII") {
                continue; // skip the identity (it lives in core_term, not the store)
            }
            o4[p] = coeff(rng);
        }
        if (o4.empty()) {
            continue;
        }
        std::string gstr = random_string(rng, 4);
        if (gstr == "IIII") {
            gstr = "XIII";
        }
        const double g = 0.5 + coeff(rng); // in (-0.5, 1.5)
        const double th = coeff(rng) * 1.5;
        check_pauli_gate<4>(o4, gstr, g, th);
    }
}

// Heisenberg ⟨HF|O_evolved|HF⟩ after a contract-immediately propagate: core + Σ state·op.
template <size_t N>
auto heisenberg_expval(MonomialPropagator<N> &sim) -> double {
    const auto &st = sim.mp_op().get_state();
    const auto &op = sim.mp_op().get_operator();
    double s = 0.0;
    for (size_t i = 0; i < op.size(); ++i) {
        s += st[i] * op[i];
    }
    return sim.core_term() + s;
}

// T6 (ARBITER): JW-vs-native isomorphism. The native Pauli propagator must match the JW-image Majorana
// propagator (same physical observable/gates, JW basis-change cutoff) on expectation value AND stored
// term count, across pictures / cutoffs / atol.
BOOST_AUTO_TEST_CASE(pauli_build_layer_jw_isomorphism) {
    constexpr size_t N = 3;
    // Kicked-Ising-like: single-qubit X rotations (incl. odd-popcount generators) + ZZ rotations.
    PauliCircuit circ;
    circ.gens = {"XII", "IXI", "IIX", "ZZI", "IZZ", "XIX"};
    circ.gs = {1.0, 0.8, 1.2, 0.9, 1.1, 0.7};
    circ.param_map = {0, 1, 2, 3, 4, 5}; // one distinct parameter per gate
    circ.params = {0.31, -0.5, 0.7, 0.42, -0.9, 0.25};
    const std::map<std::string, double> obs{{"ZII", 0.6},
                                            {"IZI", -0.4},
                                            {"YIY", 0.3},
                                            {"XZX", 0.25},
                                            {"IIZ", 0.5},
                                            {"ZZZ", -0.2}};
    const VecZ slater{0, 2}; // qubits 0 and 2 occupied

    const auto [nat_majs, nat_gcs] = native_gate_arrays(circ);
    const auto [jw_majs, jw_gcs] = jw_gate_arrays(circ);

    struct Cfg {
        std::optional<unsigned int> sch;
        unsigned int cutoff;
        std::optional<double> atol;
        const char *name;
    };
    const std::vector<Cfg> cfgs{
        {std::nullopt, 3, std::nullopt, "heisenberg-full-cutoff"},
        {std::nullopt, 2, std::nullopt, "heisenberg-cutoff-2"},
        {std::nullopt, 3, std::optional<double>(1e-6), "heisenberg-lower-atol"},
        {std::optional<unsigned int>(5), 3, std::nullopt, "schrodinger-full-cutoff"},
        {std::optional<unsigned int>(5), 3, std::optional<double>(1e-6), "schrodinger-lower-atol"},
    };
    for (const auto &cf : cfgs) {
        auto nat = build_pauli_sim<N>(obs, cf.cutoff, cf.sch, slater, cf.atol);
        auto jw = build_jw_sim<N>(obs, cf.cutoff, cf.sch, slater, cf.atol);
        nat.build_graph(nat_majs, circ.param_map, nat_gcs);
        jw.build_graph(jw_majs, circ.param_map, jw_gcs);
        const double en = nat.expectation_value(circ.params);
        const double ej = jw.expectation_value(circ.params);
        BOOST_TEST_CONTEXT(cf.name) {
            BOOST_TEST(en == ej, boost::test_tools::tolerance(1e-9));
            BOOST_TEST(nat.size() == jw.size());
        }
    }

    // Per-gate stored term count matches at every prefix (Heisenberg, atol off ⇒ purely structural).
    for (size_t k = 1; k <= circ.gens.size(); ++k) {
        PauliCircuit pre;
        pre.gens.assign(circ.gens.begin(), circ.gens.begin() + static_cast<std::ptrdiff_t>(k));
        pre.gs.assign(circ.gs.begin(), circ.gs.begin() + static_cast<std::ptrdiff_t>(k));
        pre.param_map.assign(circ.param_map.begin(), circ.param_map.begin() + static_cast<std::ptrdiff_t>(k));
        pre.params.assign(circ.params.begin(), circ.params.begin() + static_cast<std::ptrdiff_t>(k));
        const auto [nm, ng] = native_gate_arrays(pre);
        const auto [jm, jg] = jw_gate_arrays(pre);
        auto nat = build_pauli_sim<N>(obs, 3);
        auto jw = build_jw_sim<N>(obs, 3);
        nat.propagate(nm, pre.param_map, ng, pre.params);
        jw.propagate(jm, pre.param_map, jg, pre.params);
        BOOST_TEST_CONTEXT("prefix k=" << k) {
            BOOST_TEST(nat.size() == jw.size());
        }
    }
}

// T8 (replay/fold consumers + odd-popcount guard): for a native Pauli circuit that INCLUDES a
// single-qubit X layer (odd-popcount generator), the fused propagate path, the graph replay
// (expectation_value + contract_partially, which recompute the cos from the fold), and the JW-Majorana
// reference must all agree. Also directly checks the fold-recomputed per-layer cos set for the X layer.
BOOST_AUTO_TEST_CASE(pauli_build_layer_replay_fold_consumers) {
    constexpr size_t N = 3;
    const std::map<std::string, double> obs{{"ZII", 0.5}, {"IZI", -0.3}, {"YIY", 0.4}, {"XZI", 0.2}, {"IIZ", 0.6}};
    const VecZ slater{0}; // qubit 0 occupied

    // Direct fold guard: a single odd-popcount X gate. graph_data's fold-recomputed cos set must equal
    // the terms anticommuting with X (pauli sense), or the make_fold_* Pauli branch (step E) is wrong.
    {
        auto mp = build_pauli_sim<N>(obs, 3);
        mp.build_graph({slots_of_string("XII")}, VecZ{0}, VecD{1.0}); // structural single gate
        const auto layers = mp.graph_data();
        BOOST_TEST_REQUIRE(layers.size() == 1U);
        const VecZ &cos_inds = std::get<0>(layers[0]);
        std::set<size_t> got(cos_inds.begin(), cos_inds.end());
        const auto Gb = indices_to_bitset<N>(slots_of_string("XII"));
        std::set<size_t> expected;
        (void)mp.mp_op().get_operator(); // materialize the store size
        mp.indexing().for_each([&](const MajoranaSet<N> &maj, size_t idx) {
            if (pauli_anticommutes<N>(maj, Gb)) {
                expected.insert(idx);
            }
        });
        BOOST_TEST((got == expected));
    }

    // Self-consistency + JW arbiter over a multi-gate circuit including the odd-popcount X layer.
    PauliCircuit circ;
    circ.gens = {"XII", "ZZI", "IXI", "IZZ", "XIX"};
    circ.gs = {1.0, 0.9, 0.8, 1.1, 0.7};
    circ.param_map = {0, 1, 2, 3, 4};
    circ.params = {0.33, -0.7, 0.5, 0.9, -0.4};
    const auto [nat_majs, nat_gcs] = native_gate_arrays(circ);
    const auto [jw_majs, jw_gcs] = jw_gate_arrays(circ);

    // (a) fused contract-immediately propagate.
    auto prop = build_pauli_sim<N>(obs, 3, std::nullopt, slater);
    prop.propagate(nat_majs, circ.param_map, nat_gcs, circ.params);
    const double e_prop = heisenberg_expval<N>(prop);

    // (b) graph build + functional (expectation_value recomputes the cos from the fold).
    auto grp = build_pauli_sim<N>(obs, 3, std::nullopt, slater);
    grp.build_graph(nat_majs, circ.param_map, nat_gcs);
    const double e_graph = grp.expectation_value(circ.params);

    // (c) contract_partially (evolve_operator_with_recompute — the same fold path, non-inplace).
    auto ctr = build_pauli_sim<N>(obs, 3, std::nullopt, slater);
    ctr.build_graph(nat_majs, circ.param_map, nat_gcs);
    const auto evolved = ctr.contract_partially(circ.params, /*inplace=*/false);
    const auto &st = ctr.mp_op().get_state();
    double s = 0.0;
    for (size_t i = 0; i < evolved.size(); ++i) {
        s += st[i] * evolved[i];
    }
    const double e_contract = ctr.core_term() + s;

    // (d) JW-image Majorana reference.
    auto jw = build_jw_sim<N>(obs, 3, std::nullopt, slater);
    jw.build_graph(jw_majs, circ.param_map, jw_gcs);
    const double e_jw = jw.expectation_value(circ.params);

    BOOST_TEST(e_prop == e_graph, boost::test_tools::tolerance(1e-9));
    BOOST_TEST(e_contract == e_graph, boost::test_tools::tolerance(1e-9));
    BOOST_TEST(e_jw == e_graph, boost::test_tools::tolerance(1e-9));
}
