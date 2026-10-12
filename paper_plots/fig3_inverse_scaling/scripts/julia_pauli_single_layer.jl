using ArgParse
using JSON
using PauliPropagation

const REQUIRED_VERSION = v"0.9.0"
if pkgversion(PauliPropagation) != REQUIRED_VERSION
    error("this driver targets PauliPropagation $REQUIRED_VERSION exactly; the active " *
          "project has $(pkgversion(PauliPropagation)). Use --project=scripts/julia-pauli.")
end

# `active` confines every gate to qubits 1..active while the register stays `nqubits`
# wide; the rest are idle spectators. See build_pauli in monoprop_single_layer.py for why
# that is the sweep that isolates per-term cost in N.
function build_circuit(nqubits::Int, layers::Int, theta::Float64, coupling::Float64,
                       active::Int = nqubits)
    circuit = Gate[]
    thetas = Float64[]
    for _ in 1:layers
        for q in 1:active                        # Rx layer on every active qubit
            push!(circuit, PauliRotation(:X, q))
            push!(thetas, theta)
        end
        for i in 1:(active - 1)                  # Rzz layer on the 1D chain
            push!(circuit, PauliRotation([:Z, :Z], [i, i + 1]))
            push!(thetas, coupling)
        end
    end
    return circuit, thetas
end

function build_observable(nqubits::Int, active::Int = nqubits)
    psum = VectorPauliSum(nqubits)                # extensive: sum_i Z_i over the active window
    for q in 1:active
        add!(psum, :Z, q, 1.0)
    end
    return psum
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
        "--num-qubits"; arg_type = Int; required = true
        "--cutoff"; arg_type = Int; required = true
        "--layers"; arg_type = Int; default = 5
        "--lower-atol"; arg_type = Float64; default = 1e-8
        "--active-window"; arg_type = Int; default = -1
        "--rounds"; arg_type = Int; default = 3
        "--time-limit"; arg_type = Float64; default = -1.0
        "--per-layer"; action = :store_true
        "--out", "-o"; arg_type = String; required = true
    end
    args = parse_args(s)

    nq = args["num-qubits"]
    cutoff = args["cutoff"]
    layers = args["layers"]
    atol = args["lower-atol"]
    rounds = max(1, args["rounds"])

    active = args["active-window"] < 0 ? nq : args["active-window"]
    if !(1 <= active <= nq)
        error("--active-window must be in 1..num-qubits, got $active with N=$nq")
    end

    theta = pi / 4
    coupling = pi / 4
    circuit, thetas = build_circuit(nq, layers, theta, coupling, active)

    # JIT warm-up
    let wc, wt
        wc, wt = build_circuit(nq, 1, theta, coupling, active)
        propagate(wc, build_observable(nq, active), wt;
                  min_abs_coeff = atol, max_weight = cutoff, heisenberg = true,
                  thread = false)
        # The per-layer path compiles different methods; layer 1 must not time them.
        wobs = build_observable(nq, active)
        wcache = PropagationCache(wobs)
        propagate!(wc, wcache, wt; min_abs_coeff = atol, max_weight = cutoff,
                   heisenberg = true, thread = false)
        length(wcache)
        extractsum!(wcache, wobs)
    end

    best = Inf
    local result
    # --per-layer: one propagate! per layer on a propagation cache built once before the clock
    # starts, as monoprop's propagator is, so the cache keeps its buffers across layers and the
    # per-layer split adds no allocation that a whole-circuit call would not pay.
    layer_circuit, layer_thetas = build_circuit(nq, 1, theta, coupling, active)
    layer_seconds = fill(Inf, layers)
    layer_terms = Int[]
    rounds_done, timed = 0, 0.0
    while rounds_done < rounds
        obs = build_observable(nq, active)  # propagate deepcopies; rebuild anyway
        if args["per-layer"]
            cache = PropagationCache(obs)
            terms_now = Int[]
            t0 = time_ns()
            for l in 1:layers
                tl = time_ns()
                propagate!(layer_circuit, cache, layer_thetas;
                           min_abs_coeff = atol, max_weight = cutoff,
                           heisenberg = true, thread = false)
                layer_seconds[l] = min(layer_seconds[l], (time_ns() - tl) / 1e9)
                push!(terms_now, length(cache))
            end
            dt = (time_ns() - t0) / 1e9
            result = extractsum!(cache, obs)
            isempty(layer_terms) || layer_terms == terms_now ||
                error("per-layer term counts differ between rounds")
            layer_terms = terms_now
        else
            t0 = time_ns()
            # Single-threaded
            result = propagate(circuit, obs, thetas;
                               min_abs_coeff = atol, max_weight = cutoff,
                               heisenberg = true, thread = false)
            dt = (time_ns() - t0) / 1e9
        end
        best = min(best, dt)
        rounds_done += 1
        timed += dt
        keep_timing(rounds_done, timed, dt, args["time-limit"]) || break
    end

    result isa VectorPauliSum || error("expected the VectorPauliSum backend, got $(typeof(result))")
    num_terms = length(result)
    memory_bytes = Base.summarysize(result)
    expectation = real(overlapwithzero(result))

    record = Dict(
        "engine" => "julia_pauli",
        "basis" => "pauli",
        "model" => "kicked_ising",
        "per_layer" => args["per-layer"],
        "gates_per_layer" => length(layer_circuit),
        "initial_terms" => active,
        "layer_terms" => args["per-layer"] ? layer_terms : nothing,
        "layer_seconds" => args["per-layer"] ? layer_seconds : nothing,
        # One step per layer here; the Hubbard drivers split a layer into gate groups.
        "step_gates" => args["per-layer"] ? fill(length(layer_circuit), layers) : nothing,
        "step_terms" => args["per-layer"] ? layer_terms : nothing,
        "step_seconds" => args["per-layer"] ? layer_seconds : nothing,
        "num_qubits" => nq,
        "cutoff" => cutoff,
        "observable" => "extensive",
        "layers" => layers,
        "lower_atol" => atol,
        "num_threads" => Threads.nthreads(),
        "num_terms" => num_terms,
        "memory_bytes" => memory_bytes,
        "bytes_per_term" => num_terms > 0 ? memory_bytes / num_terms : 0.0,
        "seconds" => best,
        # The number of timed repetitions the reported minimum was taken over. Recorded
        # because a min-of-1 and a min-of-10 are different measurements, and the shipped
        # data used to carry no way to tell them apart.
        "rounds" => rounds_done,
        "max_rounds" => rounds,
        "time_limit" => args["time-limit"] < 0 ? nothing : args["time-limit"],
        "expectation" => expectation,
        # Provenance, matching monoprop_single_layer.py: the shipped Leonardo data carries
        # none of this, so a record from it cannot be audited or placed.
        "active_window" => args["active-window"] < 0 ? nothing : active,
        "gates" => length(circuit),
        "host" => gethostname(),
        "library_version" => string(pkgversion(PauliPropagation)),
        "julia_version" => string(VERSION),
        # The propagated representation (checked above), versus the dictionary-backed PauliSum.
        "backend" => "vector",
        "result_type" => string(typeof(result)),
        "library_threading" => false,
    )
    println("[julia/pauli] N=$nq cutoff=$cutoff terms=$num_terms " *
            "mem=$(round(memory_bytes/1024^2, digits=2))MB " *
            "b/term=$(round(record["bytes_per_term"], digits=1)) " *
            "t=$(round(best, digits=4))s exp=$(round(expectation, digits=6))")
    save_result(args["out"], record)
end

main()
