using MajoranaPropagation
using BenchmarkTools
using ArgParse


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
    return res, length(obs), loop_elapsed


end


function main(args)
    # initialize the settings (the description is for the help screen)
    s = ArgParseSettings(description="Arguments for the 1D Hubbard model benchmark.")

    @add_arg_table! s begin
        "--case", "-c"
        help = "Case pair to run."
        arg_type = Int
        default = 1
        # "--num-sites", "-n"
        # help = "Number of spinful sites in the 1D Hubbard model"
        # arg_type = Int
        # default = 30
        # "--num-layers", "-l"
        # help = "Number of layers in the bricklayer circuit"
        # arg_type = Int
        # default = 10
    end

    parsed_args = parse_args(s) # the result is a Dict{String,Any}

    spin_layers_pairs = []
    for i in [20, 40, 60]
        for j in range(10,20)
            push!(spin_layers_pairs, (i, j))
        end
    end

    case_pair = parsed_args["case"]
    N_spinful_sites, n_layers = spin_layers_pairs[case_pair]
    # println("Running benchmark for $N_spinful_sites spinful sites and $n_layers layers.")

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

    res, obs_length, loop_elapsed = experiment(N_spinful_sites, fock_state, circ_single, thetas_single, n_layers)
    final_res = res[end]
    println("$N_spinful_sites n_spin $n_layers layers $obs_length num_terms $final_res final overlap $loop_elapsed seconds")

end

main(ARGS)
