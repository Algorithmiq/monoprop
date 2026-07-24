using MajoranaPropagation
using BenchmarkTools
using ArgParse
using JSON

include(joinpath(@__DIR__, "..", "bench_common.jl"))


function experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    site_index = N_spinful_sites ÷ 2

    min_abs_coeff = 1.e-8
    max_unpaired = 10

    obs = VectorMajoranaSum(MajoranaSum(N_spinful_sites, :nup, site_index))

    values = zeros(n_layers + 1)
    term_counts = zeros(Int, n_layers + 1)
    cumulative_runtimes = zeros(n_layers + 1)
    memory_size = zeros(n_layers + 1)
    native_memory_size = zeros(n_layers + 1)

    sampler = start_sampler()

    reset!(sampler)
    cumulative_runtimes[1] = @elapsed (values[1] = overlapwithfock(obs, fock_state))
    term_counts[1] = length(obs)
    memory_size[1] = peak_mb(sampler)
    native_memory_size[1] = Base.summarysize(obs) / 1024^2
    for k = 1:n_layers
        reset!(sampler)
        step_runtime = @elapsed propagate!(circ_single, obs, thetas_single, min_abs_coeff=min_abs_coeff, max_unpaired=max_unpaired)
        values[k+1] = overlapwithfock(obs, fock_state)
        term_counts[k+1] = length(obs)
        cumulative_runtimes[k+1] = cumulative_runtimes[k] + step_runtime
        memory_size[k+1] = peak_mb(sampler)
        native_memory_size[k+1] = Base.summarysize(obs) / 1024^2
    end

    stop!(sampler)

    return values, term_counts, cumulative_runtimes, memory_size, native_memory_size


end
function save_result(output_path, source, N_spinful_sites, n_layers, term_counts, values, cumulative_runtimes, memory_size, native_memory_size, num_threads)
    """Merge this run's per-step data into the shared results JSON file, keyed by source label."""
    data = if isfile(output_path)
        JSON.parsefile(output_path)
    else
        Dict(
            "n_spinful_sites" => N_spinful_sites,
            "n_layers" => n_layers,
            "step_range" => collect(0:n_layers),
            "num_threads" => Dict(),
            "runtime_seconds" => Dict(),
            "expectation_value" => Dict(),
            "num_terms" => Dict(),
            "memory_MB" => Dict(),
            "native_memory_MB" => Dict(),
        )
    end
    data["num_threads"] = get(data, "num_threads", Dict())
    data["num_threads"][source] = num_threads
    data["runtime_seconds"][source] = cumulative_runtimes
    data["expectation_value"][source] = values
    data["num_terms"][source] = term_counts
    data["memory_MB"][source] = memory_size
    data["native_memory_MB"] = get(data, "native_memory_MB", Dict())
    data["native_memory_MB"][source] = native_memory_size

    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON.print(io, data, 4)
    end
end


function main(args)
    # initialize the settings (the description is for the help screen)
    s = ArgParseSettings(description="Arguments for the 1D Hubbard model benchmark.")

    @add_arg_table! s begin
        "--n-spins", "-n"
        help = "Number of spinful sites."
        arg_type = Int
        default = 60
        dest_name = "n_spins"
        "--max-layers", "-l"
        help = "Number of Trotter layers."
        arg_type = Int
        default = 20
        dest_name = "max_layers"
        "--output", "-o"
        help = "Path to the shared JSON file results are merged into."
        arg_type = String
        default = joinpath(@__DIR__, "results.json")

    end

    parsed_args = parse_args(s) # the result is a Dict{String,Any}

    N_spinful_sites = parsed_args["n_spins"]
    n_layers = parsed_args["max_layers"]

    t = 1.
    U = 1.5

    dt = 0.07

    topo = bricklayertopology(N_spinful_sites)

    circ_single = []
    thetas_single = []

    #up hoppings
    for (i, j) in topo
        push!(circ_single, FermionicRotation(:hopup, [i, j]))
        push!(thetas_single, -t * dt)
    end

    #down hoppings
    for (i, j) in topo
        push!(circ_single, FermionicRotation(:hopdn, [i, j]))
        push!(thetas_single, -t * dt)
    end

    #on-site repulsion
    for i = 1:N_spinful_sites
        push!(circ_single, FermionicRotation(:nupndn, i))
        push!(thetas_single, U * dt)
    end

    initial_state_label = "Checkerboard"
    fock_state = FockState(N_spinful_sites, :checkerboard, true)

    # Warm-up run to trigger JIT compilation before timing
    experiment(N_spinful_sites, fock_state, circ_single, thetas_single, 1)

    println("Number of threads: $(Threads.nthreads())")

    values, term_counts, cumulative_runtimes, memory_size, native_memory_size = experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    println("$N_spinful_sites n_spin $n_layers layers $(term_counts[end]) num_terms $(values[end]) final overlap $(cumulative_runtimes[end]) seconds")

    save_result(parsed_args["output"], "MajoranaPropagation.jl", N_spinful_sites, n_layers, term_counts, values, cumulative_runtimes, memory_size, native_memory_size, Threads.nthreads())

end

main(ARGS)
