using ArgParse
using JSON
using PauliPropagation

const REQUIRED_VERSION = v"0.8.2"
if pkgversion(PauliPropagation) != REQUIRED_VERSION
    error("this driver targets PauliPropagation $REQUIRED_VERSION exactly; the active " *
          "project has $(pkgversion(PauliPropagation)). Use --project=scripts.")
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
    end

    best = Inf
    local result
    for _ in 1:rounds
        obs = build_observable(nq, active)  # propagate deepcopies; rebuild anyway
        t0 = time_ns()
        # Single-threaded
        result = propagate(circuit, obs, thetas;
                           min_abs_coeff = atol, max_weight = cutoff,
                           heisenberg = true, thread = false)
        dt = (time_ns() - t0) / 1e9
        best = min(best, dt)
    end

    num_terms = length(result)
    memory_bytes = Base.summarysize(result)
    expectation = real(overlapwithzero(result))

    record = Dict(
        "engine" => "julia_pauli",
        "basis" => "pauli",
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
        "rounds" => rounds,
        "expectation" => expectation,
        # Provenance, matching monoprop_single_layer.py: the shipped Leonardo data carries
        # none of this, so a record from it cannot be audited or placed.
        "active_window" => args["active-window"] < 0 ? nothing : active,
        "gates" => length(circuit),
        "host" => gethostname(),
        "library_version" => string(pkgversion(PauliPropagation)),
        # Which of the library's two operator representations was propagated. Nothing in
        # the figure scripts reads it; it exists so a row can be told apart from the
        # dictionary-backed PauliSum rows earlier versions of this driver could produce.
        "backend" => "vector",
        "library_threading" => false,
    )
    println("[julia/pauli] N=$nq cutoff=$cutoff terms=$num_terms " *
            "mem=$(round(memory_bytes/1024^2, digits=2))MB " *
            "b/term=$(round(record["bytes_per_term"], digits=1)) " *
            "t=$(round(best, digits=4))s exp=$(round(expectation, digits=6))")
    save_result(args["out"], record)
end

main()
