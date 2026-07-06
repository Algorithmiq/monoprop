# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using PauliPropagation
using JSON

########################### SETTINGS ##########################
h = 1.0
j = 1.5 * h

dt = 0.002 / h
tot_time = 1.0 / h
num_steps = trunc(Int, tot_time / dt)

qubit_range = 5:5:120
max_pauli_weight = 8
lower_atol = 1e-8

label = "PauliProp (Julia)"
results_file = joinpath(@__DIR__, "trotter_ising_results.json")
###############################################################

theta_x = dt * h
theta_zz = dt * j

runtimes = Float64[]

for nq in qubit_range
    topology = staircasetopology(nq)
    circuit = tfitrottercircuit(nq, num_steps; topology=topology, start_with_ZZ=false)

    parameters = Float64[]
    for _ in 1:num_steps
        append!(parameters, fill(theta_x, nq))
        append!(parameters, fill(theta_zz, nq - 1))
    end

    observable = PauliSum(nq)
    for i in 1:nq
        add!(observable, :Z, i, 1.0)
    end

    t1 = time_ns()
    pauli_sum = propagate(
        circuit, observable, parameters;
        max_weight=max_pauli_weight, min_abs_coeff=lower_atol,
    )
    expval = overlapwithzero(pauli_sum)
    t2 = time_ns()

    push!(runtimes, (t2 - t1) / 1e9)
end

data = JSON.parsefile(results_file)
data["runtimes"][label] = runtimes
open(results_file, "w") do file
    JSON.print(file, data, 4)
end
