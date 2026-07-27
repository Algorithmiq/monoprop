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

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "TestData.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace detail {
template <typename T>
concept Printable = requires(std::ostream& os, const T& value) {
    { os << value } -> std::same_as<std::ostream&>;
};
} // namespace detail

// boost_test_print_type has to be in the same namespace as the printed type
namespace std {
template <typename T>
requires detail::Printable<T>
auto boost_test_print_type(std::ostream& os, const std::vector<T>& aVec) -> std::ostream& {
    os << "std::vector size " << aVec.size() << " [";
    for (const auto& i : aVec) {
        os << "\n    " << i;
    }
    os << "]";
    return os;
}
template <typename K, typename V>
requires detail::Printable<K> && detail::Printable<V>
auto boost_test_print_type(std::ostream& os, const std::pair<K, V>& aPair) -> std::ostream& {
    os << "[" << aPair.first << ", " << aPair.second << "]";
    return os;
}
} // namespace std

namespace test_utils {

namespace fs = std::filesystem;
using namespace monoprop;

inline auto resolve_test_data_path(int max_depth = 8) -> fs::path {
    if (const char* env_p = std::getenv("monoprop_REF_DATA_PATH")) {
        return fs::path(env_p);
    }

    fs::path cwd = fs::current_path();
    for (int i = 0; i < max_depth && !cwd.empty(); ++i) {
        fs::path cpath = cwd / "tests" / "data";
        if (fs::exists(cpath)) {
            return cpath;
        }
        if (!cwd.has_parent_path()) {
            break;
        }
        cwd = cwd.parent_path();
    }

    const fs::path source_dir = fs::path(__FILE__).parent_path().parent_path();
    const fs::path fallback = source_dir / "data";
    if (fs::exists(fallback)) {
        return fallback;
    }

    return fs::path("tests/data");
}

template <size_t NumModes>
inline auto load_case_data(const std::string& filename) -> CaseData {
    const fs::path data_path = resolve_test_data_path() / filename;
    BOOST_REQUIRE_MESSAGE(fs::exists(data_path), "Missing msgpack data file: " << data_path);
    return load_case(data_path);
}

struct SimulatorConfig {
    std::optional<unsigned int> schrodinger_cutoff = std::nullopt;
    MPI_Comm comm = MPI_COMM_SELF;
    std::optional<double> atol = std::nullopt;
    std::optional<double> upper_atol = std::nullopt;
    CutoffType cutoff_type = CutoffType::Length;
    std::optional<std::vector<VecZ>> basis_change = std::nullopt;
};

template <size_t NumModes>
inline auto build_simulator(const CaseData& data, const SimulatorConfig& cfg = {}) -> MonomialPropagator<NumModes> {
    const auto cutoff = static_cast<unsigned int>(2 * NumModes);
    return MonomialPropagator<NumModes>(data.hamiltonian,
                                        cutoff,
                                        data.initial_state,
                                        cfg.schrodinger_cutoff,
                                        cfg.comm,
                                        cfg.atol,
                                        cfg.upper_atol,
                                        cfg.cutoff_type,
                                        cfg.basis_change);
}

template <size_t NumModes>
inline auto evaluate_expval(MonomialPropagator<NumModes>& sim, const CaseData& data, bool pare) -> double {
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const std::optional<double> pare_threshold = pare ? std::optional<double>{1e-10} : std::nullopt;
    auto expval_fn = sim.expectation_value_functional(pare_threshold);
    return expval_fn(data.parameters);
}

inline auto check_expval_close(const char* label, double expval, double exact, double atol = 1e-9) -> void {
    BOOST_TEST_MESSAGE(std::string("[") + label + "] expval=" + std::format("{:.9f}", expval)
                       + ", exact=" + std::format("{:.9f}", exact));
    BOOST_CHECK_SMALL(expval - exact, atol);
}

// Mixed absolute/relative comparison; rtol absorbs the accumulation drift between n=1 and n>1 runs.
inline constexpr double kFpRtol = 1e-7;
inline auto near(double lhs, double rhs, double atol = 1e-9, double rtol = kFpRtol) -> bool {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= (atol + rtol * scale);
}

// Driven by build_graph_tests.cpp.
template <size_t n_modes>
inline auto test_evolve_build_graph(const CaseData& data, const SimulatorConfig& cfg, bool pare, double exact_expval)
    -> void {
    auto mp = build_simulator<n_modes>(data, cfg);
    mp.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    const std::optional<double> pare_threshold = pare ? std::optional<double>{1e-10} : std::nullopt;
    auto expval_fn = mp.expectation_value_functional(pare_threshold);
    double expval = expval_fn(data.parameters);
    BOOST_TEST_CONTEXT("n_modes=" << n_modes << " pare=" << pare << " sch_cutoff="
                                  << (cfg.schrodinger_cutoff ? std::to_string(*cfg.schrodinger_cutoff) : "none")) {
        check_expval_close("Expectation Value Build Graph", expval, exact_expval);
    }
}

template <size_t n_modes>
inline auto test_evolve_build_graph_with_coeffs(const CaseData& data,
                                                const SimulatorConfig& cfg,
                                                bool pare,
                                                double exact_expval) -> void {
    auto mp = build_simulator<n_modes>(data, cfg);

    // Coefficient-informed build: the seed is regenerated internally from the parameters.
    // gate_indices defaults (nullopt -> one gate per generator).
    mp.build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, data.parameters);

    const std::optional<double> pare_threshold = pare ? std::optional<double>{1e-10} : std::nullopt;
    auto expval_fn = mp.expectation_value_functional(pare_threshold);
    double expval = expval_fn(data.parameters);
    BOOST_TEST_CONTEXT("n_modes=" << n_modes << " pare=" << pare << " sch_cutoff="
                                  << (cfg.schrodinger_cutoff ? std::to_string(*cfg.schrodinger_cutoff) : "none")) {
        check_expval_close("Expectation Value Build Graph with coeffs", expval, exact_expval);
    }
}

struct ExampleDataFix {
    static constexpr size_t n_modes = 8;
    int cutoff = 2 * n_modes;
    CutoffType cutoff_type = CutoffType::Length;
    std::optional<std::vector<VecZ>> basis_change{std::nullopt};
    CaseData data;

    ExampleDataFix() {
        auto data_path = resolve_test_data_path();
        auto msgpack_file = data_path / "random_exact.msgpack";
        data = load_case(msgpack_file);
    }
};

struct LihFixture {
    static constexpr size_t n_modes = 12;
    CaseData data;
    LihFixture() : data(load_case_data<n_modes>("lih_fermionic_spin_exact.msgpack")) {}
};

inline constexpr std::array<bool, 2> ds_pare_values{false, true};
inline constexpr std::array<bool, 2> ds_schrodinger_enabled{false, true};

inline auto make_schrodinger_cutoff(bool enabled, int cutoff, int offset = 2) -> std::optional<int> {
    if (!enabled) {
        return std::nullopt;
    }
    return cutoff + offset;
}

} // namespace test_utils

using test_utils::resolve_test_data_path;
