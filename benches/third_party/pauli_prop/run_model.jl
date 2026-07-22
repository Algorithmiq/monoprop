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
using ProgressMeter

settings = JSON.parsefile(joinpath(@__DIR__, "settings.json"))

nx, ny = settings["nx"], settings["ny"]
nq = nx * ny
hx = settings["hx"]
hz = settings["hz"]
j = settings["j"]
dt = settings["dt"]

step_range = settings["step_min"]:settings["step_max"]:settings["step_size"]
lower_atol = settings["lower_atol"]
max_pauli_weight = something(settings["cutoff"], Inf)
obs_qubits = Tuple(q + 1 for q in settings["obs_qubits"])

theta_x = dt * hx
theta_z = dt * hz
theta_zz = dt * j
topology = rectangletopology(nx, ny)

step_circuit = tiltedtfitrottercircuit(nq, 1; topology=topology)
step_parameters = Float64[]
append!(step_parameters, fill(theta_zz, length(topology)))
append!(step_parameters, fill(theta_z, nq))
append!(step_parameters, fill(theta_x, nq))

pauli_sum = PauliSum(nq)
add!(pauli_sum, [:Z, :Z], collect(obs_qubits), 1.0)

num_terms = Int[]
runtime = Float64[]
memory = Float64[]
expvals = Float64[]

@showprogress for (step_idx, num_steps) in enumerate(step_range)
    t1 = time_ns()
    global pauli_sum = propagate(
        step_circuit, pauli_sum, step_parameters;
        min_abs_coeff=lower_atol, max_weight=max_pauli_weight,
    )
    expval = overlapwithzero(pauli_sum)
    t2 = time_ns()

    if step_idx > 1
        push!(runtime, (t2 - t1) / 1e9)
    end
    push!(num_terms, length(pauli_sum))
    push!(memory, Base.summarysize(pauli_sum) / 1024^2)
    push!(expvals, expval)
end

results_file = joinpath(@__DIR__, "results.json")
data = JSON.parsefile(results_file)
data["num_terms"]["PauliPropagation.jl"] = num_terms
data["runtime"]["PauliPropagation.jl"] = runtime
data["memory"]["PauliPropagation.jl"] = memory
data["expvals"]["PauliPropagation.jl"] = expvals

open(results_file, "w") do file
    JSON.print(file, data, 4)
end
