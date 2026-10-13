using ArgParse
using JSON
using MajoranaPropagation
using PauliPropagation

const PB = PauliPropagation.PropagationBase

const REQUIRED_VERSION = v"0.4.1"
if pkgversion(MajoranaPropagation) != REQUIRED_VERSION
    error("this driver targets MajoranaPropagation $REQUIRED_VERSION exactly; the active " *
          "project has $(pkgversion(MajoranaPropagation)). Use --project=scripts/julia-majorana.")
end

# The circuit of build_hubbard2d in monoprop_single_layer.py, gate for gate. monoprop's ExpGate
# applies exp(+i dt H) with H = -t hop + U n_up n_dn; FermionicRotation applies exp(-i theta O),
# so hopping takes theta = +t dt and the interaction theta = -U dt.
# _checkerboard_topology in monoprop_single_layer.py, 1-based: horizontal bonds from even x,
# from odd x, then vertical bonds from even y, from odd y; each class a matching.
function checkerboard_topology(lx::Int, ly::Int)
    site(x, y) = y * lx + x + 1
    bonds = Tuple{Int,Int}[]
    for p in (0, 1), y in 0:(ly - 1), x in p:2:(lx - 2)
        push!(bonds, (site(x, y), site(x + 1, y)))
    end
    for p in (0, 1), y in p:2:(ly - 2), x in 0:(lx - 1)
        push!(bonds, (site(x, y), site(x, y + 1)))
    end
    return bonds
end

# One layer's nine commuting gate groups (hop-up per bond class, hop-down per bond class, the
# interaction) as index ranges into the layer circuit, in the order they propagate: the last
# group first, as the Heisenberg picture applies a circuit. build_hubbard2d's `steps`.
function layer_steps(lx::Int, ly::Int)
    classes = [ly * length(0:2:(lx - 2)), ly * length(1:2:(lx - 2)),
               lx * length(0:2:(ly - 2)), lx * length(1:2:(ly - 2))]
    sizes = [classes; classes; lx * ly]
    bounds = cumsum([0; sizes])
    return [(bounds[i] + 1):bounds[i + 1] for i in reverse(eachindex(sizes)) if sizes[i] > 0]
end

function build_circuit(lx::Int, ly::Int, layers::Int; t = 1.0, U = 1.5, dt = 0.07)
    topology = checkerboard_topology(lx, ly)
    circuit = FermionicRotation[]
    thetas = Float64[]
    for _ in 1:layers
        for symbol in (:hopup, :hopdn), (i, j) in topology
            push!(circuit, FermionicRotation(symbol, [i, j]))
            push!(thetas, t * dt)
        end
        for site in 1:(lx * ly)
            push!(circuit, FermionicRotation(:nupndn, site))
            push!(thetas, -U * dt)
        end
    end
    return circuit, thetas
end

# Staggered magnetisation sum_s (-1)^(x+y) (n_up - n_dn), as monoprop builds it. The constant
# parts of n_up and n_dn cancel site by site; the zero-coefficient identity they leave is
# dropped so that both engines start from the same term set.
function build_observable(lx::Int, ly::Int)
    nsites = lx * ly
    msum = MajoranaSum(Float64, nsites, true)
    for site in 1:nsites
        y, x = divrem(site - 1, lx)
        sign = iseven(x + y) ? 1.0 : -1.0
        add!(msum, :nup, site, sign)
        add!(msum, :ndn, site, -sign)
    end
    clean = MajoranaSum(Float64, nsites, true)
    for (ms, c) in msum
        iszero(c) || set!(clean, ms, c)
    end
    return VectorMajoranaSum(clean)
end

# monoprop's cutoff_type="support": drop a string that touches more than `cutoff` modes unless
# it is fully paired. Mode k holds bits 2k and 2k+1, and `mask` has the odd bit of every pair.
function support_truncation(::Type{TT}, nmodes::Int, cutoff::Int) where {TT}
    mask = create_unpaired_mask(TT, nmodes)
    return function (mstr, coeff)
        support = count_ones((mstr | (mstr << 1)) & mask)
        unpaired = count_ones((mstr ⊻ (mstr << 1)) & mask)
        return support > cutoff && unpaired > 0
    end
end

# monoprop's cutoff_type="length": drop a string of more than `cutoff` Majoranas unless it is
# fully paired. MajoranaPropagation's own max_weight has no paired exception, so it is passed as
# a custom function too.
function length_truncation(::Type{TT}, nmodes::Int, cutoff::Int) where {TT}
    mask = create_unpaired_mask(TT, nmodes)
    return function (mstr, coeff)
        unpaired = count_ones((mstr ⊻ (mstr << 1)) & mask)
        return count_ones(mstr) > cutoff && unpaired > 0
    end
end

# The round rule of monoprop_single_layer.keep_timing: a second round whenever it fits (a
# fresh process's first round can be far slower), more while under ROUND_BUDGET_S, and none
# that would be expected to end past the point's time limit (negative: no limit).
const ROUND_BUDGET_S = 60.0

function keep_timing(rounds_done::Int, timed::Float64, last::Float64, time_limit::Float64)
    time_limit >= 0 && timed + last > time_limit && return false
    return rounds_done < 2 || timed < ROUND_BUDGET_S
end

function save_result(path::String, record::Dict)
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, JSON.json(record))
    end
end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--lx"; arg_type = Int; required = true
        "--ly"; arg_type = Int; required = true
        "--cutoff"; arg_type = Int; required = true
        "--cutoff-type"; arg_type = String; default = "support"
        "--layers"; arg_type = Int; default = 1
        "--rounds"; arg_type = Int; default = 3
        "--time-limit"; arg_type = Float64; default = -1.0
        "--per-layer"; action = :store_true
        "--out", "-o"; arg_type = String; required = true
    end
    args = parse_args(s)
    lx, ly = args["lx"], args["ly"]
    nsites = lx * ly
    nmodes = 2 * nsites
    cutoff, layers = args["cutoff"], args["layers"]
    # No coefficient truncation: the support cutoff alone bounds the operator.
    atol = 0.0
    rounds = max(1, args["rounds"])

    circuit, thetas = build_circuit(lx, ly, layers)
    layer_circuit, layer_thetas = build_circuit(lx, ly, 1)
    steps = layer_steps(lx, ly)
    sum(length, steps) == length(layer_circuit) || error("steps do not tile the layer")
    TT = typeof(first(terms(build_observable(lx, ly))))
    cutoff_type = args["cutoff-type"]
    cutoff_type in ("support", "length") || error("--cutoff-type must be support or length")
    truncfunc = (cutoff_type == "support" ? support_truncation : length_truncation)(TT, nmodes, cutoff)
    fock = FockState(nsites, :checkerboard, true; nx = lx)
    # truncate_each_mr: truncate after every Majorana rotation, as monoprop does. By default a
    # hopping gate (two commuting rotations) is truncated only once, after both, which keeps
    # terms whose intermediate exceeded the cutoff; with it the term sets agree exactly.
    kw = (min_abs_coeff = atol, customtruncfunc = truncfunc, truncate_each_mr = true,
          heisenberg = true, thread = false)

    # JIT warm-up of both paths, so no timed layer compiles.
    propagate(layer_circuit, build_observable(lx, ly), layer_thetas; kw...)
    let wobs = build_observable(lx, ly), wcache = PB.PropagationCache(wobs)
        for r in steps
            propagate!(layer_circuit[r], wcache, layer_thetas[r]; kw...)
        end
        length(wcache)
        overlapwithfock(PB.extractsum!(wcache, wobs), fock)
    end

    best = Inf
    local result
    step_seconds = fill(Inf, layers * length(steps))
    step_terms = Int[]
    initial_terms = length(build_observable(lx, ly))
    rounds_done, timed = 0, 0.0
    while rounds_done < rounds
        obs = build_observable(lx, ly)
        if args["per-layer"]
            cache = PB.PropagationCache(obs)
            terms_now = Int[]
            t0 = time_ns()
            for _ in 1:layers, r in steps
                tl = time_ns()
                propagate!(layer_circuit[r], cache, layer_thetas[r]; kw...)
                i = length(terms_now) + 1
                step_seconds[i] = min(step_seconds[i], (time_ns() - tl) / 1e9)
                push!(terms_now, length(cache))
            end
            dt = (time_ns() - t0) / 1e9
            result = PB.extractsum!(cache, obs)
            isempty(step_terms) || step_terms == terms_now ||
                error("per-step term counts differ between rounds")
            step_terms = terms_now
        else
            t0 = time_ns()
            result = propagate(circuit, obs, thetas; kw...)
            dt = (time_ns() - t0) / 1e9
        end
        best = min(best, dt)
        rounds_done += 1
        timed += dt
        keep_timing(rounds_done, timed, dt, args["time-limit"]) || break
    end

    result isa VectorMajoranaSum ||
        error("expected the VectorMajoranaSum backend, got $(typeof(result))")
    num_terms = length(result)
    memory_bytes = Base.summarysize(result)
    expectation = real(overlapwithfock(result, fock))

    record = Dict(
        "engine" => "julia_majorana",
        "basis" => "majorana",
        "model" => "hubbard2d",
        "lattice" => [lx, ly],
        "num_qubits" => nmodes,
        "cutoff" => cutoff,
        "cutoff_type" => cutoff_type,
        "observable" => "extensive",
        "layers" => layers,
        "lower_atol" => atol,
        "num_threads" => Threads.nthreads(),
        "num_terms" => num_terms,
        "memory_bytes" => memory_bytes,
        "bytes_per_term" => num_terms > 0 ? memory_bytes / num_terms : 0.0,
        "seconds" => best,
        "rounds" => rounds_done,
        "max_rounds" => rounds,
        "time_limit" => args["time-limit"] < 0 ? nothing : args["time-limit"],
        "expectation" => expectation,
        "active_window" => nothing,
        "gates" => length(circuit),
        "gates_per_layer" => length(layer_circuit),
        "initial_terms" => initial_terms,
        "per_layer" => args["per-layer"],
        "layer_terms" => args["per-layer"] ? step_terms[length(steps):length(steps):end] : nothing,
        # A layer's time is the sum of its steps' minima.
        "layer_seconds" => args["per-layer"] ?
                           [sum(c) for c in Iterators.partition(step_seconds, length(steps))] :
                           nothing,
        "step_gates" => args["per-layer"] ? repeat(length.(steps), layers) : nothing,
        "step_terms" => args["per-layer"] ? step_terms : nothing,
        "step_seconds" => args["per-layer"] ? step_seconds : nothing,
        "host" => gethostname(),
        "library_version" => string(pkgversion(MajoranaPropagation)),
        "pauli_propagation_version" => string(pkgversion(PauliPropagation)),
        "julia_version" => string(VERSION),
        "backend" => "vector",
        "truncation" => "customtruncfunc: $(cutoff_type) > cutoff and not fully paired; " *
                        "truncate_each_mr = true",
        "result_type" => string(typeof(result)),
        "library_threading" => false,
    )
    println("[julia/majorana] $(lx)x$(ly) modes=$nmodes cutoff=$cutoff terms=$num_terms " *
            "t=$(round(best, digits=4))s exp=$(round(expectation, digits=8))")
    save_result(args["out"], record)
end

main()
