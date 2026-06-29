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

#include "TestData.h"

#include <complex>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include "monoprop/TypeAliases.h"

namespace test_utils {

using namespace monoprop;

auto load_case(const std::filesystem::path& filename) -> CaseData {
    auto file = std::ifstream(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open msgpack file: {}", filename.string()));
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error(std::format("Error reading file: {}", filename.string()));
    }
    file.close();

    msgpack::object_handle oh;
    try {
        oh = msgpack::unpack(buffer.data(), buffer.size());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::format("Failed to parse msgpack file: {} - {}", filename.string(), e.what()));
    }

    msgpack::object obj = oh.get();
    if (obj.type != msgpack::type::MAP) {
        throw std::runtime_error("Expected msgpack map at root level");
    }

    std::map<std::string, msgpack::object> data;
    obj.convert(data);

    // Hamiltonian: parallel keys / real / imag arrays.
    auto raw_hamiltonian = data.at("hamiltonian").as<std::map<std::string, msgpack::object>>();
    auto keys = raw_hamiltonian.at("keys").as<std::vector<VecZ>>();
    auto real = raw_hamiltonian.at("real").as<VecD>();
    auto imag = raw_hamiltonian.at("imag").as<VecD>();

    FermiOperatorMap hamiltonian;
    for (size_t i = 0; i < keys.size(); ++i) {
        hamiltonian[keys[i]] = std::complex{real[i], imag[i]};
    }

    return {data.at("actual_energy").as<double>(),
            data.at("hartree_fock").as<VecZ>(),
            data.at("param_inds").as<VecZ>(),
            data.at("gen_coeffs").as<VecD>(),
            data.at("parameters").as<VecD>(),
            data.at("majoranas").as<std::vector<VecZ>>(),
            hamiltonian,
            data.at("num_modes").as<size_t>()};
}

} // namespace test_utils
