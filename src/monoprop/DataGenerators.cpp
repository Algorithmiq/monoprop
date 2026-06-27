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

#include "monoprop/DataGenerators.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <msgpack.hpp>

#include "monoprop/Info.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/TypeAliases.h"

namespace monoprop {

namespace {

auto count_paired(const VecZ& indices) -> size_t {
    VecZ modes;
    modes.reserve(indices.size());
    for (const auto& index : indices) {
        modes.push_back(index / 2);
    }
    std::unordered_set<size_t> unique_modes(modes.begin(), modes.end());
    return modes.size() - unique_modes.size();
}

} // namespace

auto load_from_msgpack(const std::filesystem::path& filename) -> MPData {
    // Read the MsgPack file
    auto file = std::ifstream(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open MsgPack file: {}", filename.string()));
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error(std::format("Error reading file: {}", filename.string()));
    }
    file.close();

    // Parse MsgPack data
    msgpack::object_handle oh;
    try {
        oh = msgpack::unpack(buffer.data(), buffer.size());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::format("Failed to parse MsgPack file: {} - {}", filename.string(), e.what()));
    }

    msgpack::object obj = oh.get();

    if (obj.type != msgpack::type::MAP) {
        throw std::runtime_error("Expected MsgPack map at root level");
    }

    // Convert to std::map for easier access
    std::map<std::string, msgpack::object> data;
    obj.convert(data);

    auto raw_fermionic_hamiltonian = data.at("fermionic_hamiltonian").as<std::map<std::string, msgpack::object>>();
    // extract keys
    auto ks = raw_fermionic_hamiltonian.at("keys").as<std::vector<VecZ>>();
    // extract values
    auto vs = raw_fermionic_hamiltonian.at("vals").as<std::map<std::string, msgpack::object>>();
    auto real = vs["real"].as<VecD>();
    auto imag = vs["imag"].as<VecD>();

    std::map<VecZ, std::complex<double>> fermionic_hamiltonian;
    for (size_t i = 0; i < ks.size(); ++i) {
        fermionic_hamiltonian[ks[i]] = std::complex{real[i], imag[i]};
    }

    auto majoranas = data["majoranas"].as<std::vector<VecZ>>();

    auto slater_determinant = data["hartree_fock"].as<VecZ>();

    auto gen_coeffs = data["gen_coeffs"].as<VecD>();

    auto param_inds = data["param_inds"].as<VecZ>();

    auto parameters = data["parameters"].as<VecD>();

    auto actual_expval = data.at("actual_energy").as<double>();

    auto circuit_expval = data.at("ansatz_energy").as<double>();

    const auto num_modes = data.at("num_modes").as<size_t>();

    return {actual_expval,
            circuit_expval,
            slater_determinant,
            param_inds,
            gen_coeffs,
            parameters,
            majoranas,
            fermionic_hamiltonian,
            num_modes};
}
} // namespace monoprop
