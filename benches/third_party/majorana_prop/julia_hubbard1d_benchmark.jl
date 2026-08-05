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

using MajoranaPropagation
using PauliPropagation
using BenchmarkTools
using ArgParse
using JSON


"""CPU seconds (user + system) consumed by this process so far, summed over all threads.

Fields 14 and 15 of /proc/self/stat are utime/stime in clock ticks; _SC_CLK_TCK is 2.
"""
function process_cpu_seconds()
    fields = split(read("/proc/self/stat", String))
    ticks = ccall(:sysconf, Clong, (Cint,), 2)
    return (parse(Int, fields[14]) + parse(Int, fields[15])) / ticks
end


"""Peak resident set size in MB, from VmHWM in /proc/self/status."""
function peak_rss_mb()
    for line in eachline("/proc/self/status")
        if startswith(line, "VmHWM:")
            return parse(Int, split(line)[2]) / 1024
        end
    end
    return NaN
end


function experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    site_index = N_spinful_sites ÷ 2

    min_abs_coeff = 1.e-8
    max_unpaired = 10

    obs = VectorMajoranaSum(MajoranaSum(N_spinful_sites, :nup, site_index))

    res = zeros(n_layers + 1)
    res[1] = overlapwithfock(obs, fock_state)


    cpu_start = process_cpu_seconds()
    stats = @timed for k = 1:n_layers
        propagate!(circ_single, obs, thetas_single, min_abs_coeff=min_abs_coeff, max_unpaired=max_unpaired)
        res[k+1] = overlapwithfock(obs, fock_state)
    end
    cpu_seconds = process_cpu_seconds() - cpu_start
    memory_size = Base.summarysize(obs) / 1024^2
    return (
        res=res,
        num_terms=length(obs),
        runtime_seconds=stats.time,
        gc_seconds=stats.gctime,
        cpu_seconds=cpu_seconds,
        memory_MB=memory_size,
        # Recorded, not hardcoded: only the Vector container dispatches into the
        # AcceleratedKernels path, so this is what proves the run was threaded at all.
        container=string(nameof(typeof(obs))),
    )


end
function save_result(output_path, N_spinful_sites, n_layers, result)
    """Append one benchmark result as a JSON line, creating the parent directory if needed."""
    record = Dict(
        "n_spinful_sites" => N_spinful_sites,
        "n_layers" => n_layers,
        "num_terms" => result.num_terms,
        "final_overlap" => result.res[end],
        "runtime_seconds" => result.runtime_seconds,
        "cpu_seconds" => result.cpu_seconds,
        "busy_cores" => result.cpu_seconds / result.runtime_seconds,
        "gc_seconds" => result.gc_seconds,
        "num_threads" => Threads.nthreads(),
        "container" => result.container,
        "memory_MB" => result.memory_MB,
        "peak_rss_MB" => peak_rss_mb(),
        "library_version" => string(pkgversion(MajoranaPropagation)),
        "pauliprop_version" => string(pkgversion(PauliPropagation)),
        "host" => gethostname(),
    )

    open(output_path, "a") do io
        JSON.print(io, record)
        println(io)
    end
end


function main(args)
    s = ArgParseSettings(description="Arguments for the 1D Hubbard model benchmark.")

    @add_arg_table! s begin
        "--case", "-c"
        help = "Case pair to run."
        arg_type = Int
        default = 1
        "--output", "-o"
        help = "Path to the JSONL file results are appended to."
        arg_type = String
        default = joinpath(@__DIR__, "julia_hubbard1d_benchmark_results.jsonl")

    end

    parsed_args = parse_args(s)

    spin_layers_pairs = []
    for i in [20, 40, 60]
        for j in 10:2:18
            push!(spin_layers_pairs, (i, j))
        end
    end

    case_pair = parsed_args["case"]
    N_spinful_sites, n_layers = spin_layers_pairs[case_pair]

    t = 1.
    U = 1.5

    dt = 0.07

    topo = bricklayertopology(N_spinful_sites)

    circ_single = []
    thetas_single = []

    for (i, j) in topo
        push!(circ_single, FermionicRotation(:hopup, [i, j]))
        push!(thetas_single, -t * dt)
    end

    for (i, j) in topo
        push!(circ_single, FermionicRotation(:hopdn, [i, j]))
        push!(thetas_single, -t * dt)
    end

    for i = 1:N_spinful_sites
        push!(circ_single, FermionicRotation(:nupndn, i))
        push!(thetas_single, U * dt)
    end

    initial_state_label = "Checkerboard"
    fock_state = FockState(N_spinful_sites, :checkerboard, true)

    # Warm-up run to trigger JIT compilation before timing
    experiment(N_spinful_sites, fock_state, circ_single, thetas_single, 1)

    println("Number of threads: $(Threads.nthreads())")

    result = experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    busy_cores = result.cpu_seconds / result.runtime_seconds
    println("$N_spinful_sites n_spin $n_layers layers $(result.num_terms) num_terms $(result.res[end]) final overlap $(result.runtime_seconds) seconds")
    println("container $(result.container)  cpu $(round(result.cpu_seconds, digits=1)) s  busy_cores $(round(busy_cores, digits=2))  gc $(round(result.gc_seconds, digits=1)) s")

    save_result(parsed_args["output"], N_spinful_sites, n_layers, result)

end

main(ARGS)
