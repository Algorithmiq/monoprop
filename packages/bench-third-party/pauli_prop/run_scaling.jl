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

# PauliPropagation.jl at ONE lattice size, appending one totals record to a JSONL file.
# The Julia counterpart of run_one.py — same model, same record schema, so run_scaling.py
# can drive it as just another backend and plot_scaling.py needs no special case.

using PauliPropagation
using JSON

include(joinpath(@__DIR__, "..", "bench_common.jl"))

function parse_args(args)
    opts = Dict{String,Any}(
        "nx" => nothing, "ny" => nothing, "step-max" => nothing,
        "settings" => joinpath(@__DIR__, "settings.json"), "output" => nothing,
    )
    i = 1
    while i <= length(args)
        key = lstrip(args[i], '-')
        haskey(opts, key) || error("unknown argument $(args[i])")
        opts[key] = args[i+1]
        i += 2
    end
    opts["output"] === nothing && error("--output is required")
    return opts
end

opts = parse_args(ARGS)
settings = JSON.parsefile(opts["settings"])

nx = opts["nx"] === nothing ? settings["nx"] : parse(Int, opts["nx"])
ny = opts["ny"] === nothing ? settings["ny"] : parse(Int, opts["ny"])
step_max = opts["step-max"] === nothing ? settings["step_max"] : parse(Int, opts["step-max"])
nq = nx * ny
dt = settings["dt"]
lower_atol = settings["lower_atol"]
max_pauli_weight = something(settings["cutoff"], Inf)

obs_row = div(ny, 2)
obs_col = max(div(nx, 2) - 1, 0)
obs_qubits = (obs_row * nx + obs_col + 1, obs_row * nx + obs_col + 2)

theta_zz = dt * settings["j"]
theta_z = dt * settings["hz"]
theta_x = dt * settings["hx"]

topology = rectangletopology(nx, ny)
step_circuit = tiltedtfitrottercircuit(nq, 1; topology=topology)
step_parameters = Float64[]
append!(step_parameters, fill(theta_zz, length(topology)))
append!(step_parameters, fill(theta_z, nq))
append!(step_parameters, fill(theta_x, nq))

isdefined(PauliPropagation, :Performance) ||
    error("needs the PauliPropagation dev branch (0.8.0+) for the Performance submodule; " *
          "v$(pkgversion(PauliPropagation)) has none")

pauli_sum = VectorPauliSum(nq)
add!(pauli_sum, [:Z, :Z], collect(obs_qubits), 1.0)

advance!(psum) = PauliPropagation.Performance.propagate!(
    step_circuit, psum, step_parameters;
    min_abs_coeff=lower_atol, max_weight=max_pauli_weight)

step_range = settings["step_min"]:settings["step_size"]:step_max
runtimes = Float64[]
num_terms = Int[]
memory = Float64[]
operator_memory = Float64[]
expvals = Float64[]

println("[PauliPropagation.jl] $(nx)x$(ny) ($nq qubits), dt=$dt, atol=$lower_atol, " *
        "obs=ZZ$(collect(obs_qubits)), steps $(first(step_range))..$(last(step_range)), " *
        "threads=$(Threads.nthreads()), v$(pkgversion(PauliPropagation))")
flush(stdout)

# Same instrument as the Python arm (backends._run_steps), so the two are comparable.
window = HighWaterMark()

# Taken before the first window opens: start! resets the kernel's mm->hiwater_rss, which is
# what Sys.maxrss() reports, so it stops being a whole-run ceiling from that point on.
setup_peak_mb = Sys.maxrss() / 1024^2

for _ in step_range
    start!(window)
    t1 = time_ns()
    global pauli_sum = advance!(pauli_sum)
    expval = overlapwithzero(pauli_sum)
    t2 = time_ns()
    stop!(window)
    push!(runtimes, (t2 - t1) / 1e9)
    push!(num_terms, length(pauli_sum))
    push!(memory, peak_mb(window))
    push!(operator_memory, Base.summarysize(pauli_sum) / 1024^2)
    push!(expvals, expval)
end

record = Dict(
    "backend" => "juliapp",
    "label" => "PauliPropagation.jl",
    "nx" => nx, "ny" => ny, "num_qubits" => nq,
    "num_steps" => length(runtimes),
    "threads" => Threads.nthreads(),
    "library_version" => string(pkgversion(PauliPropagation)),
    "host" => gethostname(),
    "status" => "ok",
    "total_runtime_s" => sum(runtimes),
    # The first step carries Julia's JIT warm-up, as in the fixed-size benchmark.
    "total_runtime_excl_first_s" => sum(runtimes[2:end]),
    "final_step_s" => runtimes[end],
    "final_memory_MB" => memory[end],
    "memory_metric" => "peak process RSS over the step (kernel VmHWM)",
    "operator_memory_MB" => operator_memory[end],
    "operator_memory_metric" => "Base.summarysize of the Pauli sum",
    "peak_rss_MB" => max(setup_peak_mb, maximum(memory)),
    "final_num_terms" => num_terms[end],
    "final_expval" => expvals[end],
    "max_terms_budget" => nothing,
    # Mirrors run_one.py: the truncation as fields, with a cutoff that cannot bind
    # (Inf here, num_qubits there) recorded as absent rather than as a number.
    "lower_atol" => lower_atol,
    "weight_cutoff" => (isfinite(max_pauli_weight) && max_pauli_weight < nq ?
                        max_pauli_weight : nothing),
    "settings" => "$(nx)x$(ny) ($nq qubits), dt=$dt, atol=$lower_atol, " *
                  "weight cutoff=$max_pauli_weight, obs=ZZ$(collect(obs_qubits))",
)

mkpath(dirname(opts["output"]))
open(opts["output"], "a") do io
    JSON.print(io, record)
    write(io, "\n")
end

println("[PauliPropagation.jl] $nq qubits: $(round(sum(runtimes), digits=3)) s total, " *
        "$(round(memory[end], digits=1)) MB, $(num_terms[end]) terms, expval $(expvals[end])")
