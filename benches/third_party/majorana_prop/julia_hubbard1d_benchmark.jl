using MajoranaPropagation
using BenchmarkTools
using ArgParse
using JSON


function experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    site_index = N_spinful_sites ÷ 2

    min_abs_coeff = 1.e-8
    max_unpaired = 10

    obs = VectorMajoranaSum(MajoranaSum(N_spinful_sites, :nup, site_index))

    res = zeros(n_layers + 1)
    res[1] = overlapwithfock(obs, fock_state)


    loop_elapsed = @elapsed for k = 1:n_layers
        propagate!(circ_single, obs, thetas_single, min_abs_coeff=min_abs_coeff, max_unpaired=max_unpaired)
        res[k+1] = overlapwithfock(obs, fock_state)
    end
    memory_size = Base.summarysize(obs) / 1024^2
    return res, length(obs), loop_elapsed, memory_size


end
function save_result(output_path, N_spinful_sites, n_layers, obs_length, final_res, loop_elapsed, memory_size)
    """Append one benchmark result as a JSON line, creating the parent directory if needed."""
    record = Dict(
        "n_spinful_sites" => N_spinful_sites,
        "n_layers" => n_layers,
        "num_terms" => obs_length,
        "final_overlap" => final_res,
        "runtime_seconds" => loop_elapsed,
        "memory_MB" => memory_size,
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
        for j in range(10, 18, 2)
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

    res, obs_length, loop_elapsed, memory_size = experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    final_res = res[end]
    println("$N_spinful_sites n_spin $n_layers layers $obs_length num_terms $final_res final overlap $loop_elapsed seconds")

    save_result(parsed_args["output"], N_spinful_sites, n_layers, obs_length, final_res, loop_elapsed, memory_size)

end

main(ARGS)
