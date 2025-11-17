# might wanna Pkg.dev these
using ComputableDAGs
using QEDFeynmanDiagrams

using QEDcore, QEDprocesses
using Distributed
using BenchmarkTools
using Logging
using JLD2

using ComputableDAGs: task

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20.0

ComputableDAGs.init(@__MODULE__)

global_logger(NullLogger())

function time_compilation(expr; setup = nothing)
    ps = addprocs(1)
    remotecall_fetch(only(ps)) do
        @eval begin
            using QEDprocesses, QEDcore, ComputableDAGs, QEDFeynmanDiagrams
        end
    end

    (; compile_time) = remotecall_fetch(only(ps)) do
        @eval begin
            $setup
            @timed $expr
        end
    end
    rmprocs(ps)
    return compile_time
end

function bench_compilation(expr; setup = nothing, n = 5)
    times = Float64[]
    for _ in 1:n
        push!(times, time_compilation(expr; setup = setup))
    end

    return times
end

# ------------------

function make_nphoton_compton(n::Int)
    return ScatteringProcess(
        (Electron(), Photon()),
        (Electron(), ntuple(_ -> Photon(), n)...),
    )
end

MODEL = PerturbativeQED()
IN_PSL = ComptonRestSystem()
PSL = FlatPhaseSpaceLayout(IN_PSL)

return 0

SCATTERING_PROCESSES = [
    (make_nphoton_compton(1), "ke->ke"),
    (make_nphoton_compton(2), "ke->kke"),
    (make_nphoton_compton(3), "ke->kkke"),
    (make_nphoton_compton(4), "ke->kkkke"),
    (make_nphoton_compton(5), "ke->kkkkke"),
    (
        ScatteringProcess(
            (Electron(), Positron()),
            (Electron(), Positron()),
        ), "ep->ep",
    ),
    (
        ScatteringProcess(
            (Electron(), Positron()),
            (Electron(), Positron(), Electron(), Positron()),
        ), "ep->epep",
    ),
    (
        ScatteringProcess(
            (Electron(), Positron()),
            (Electron(), Positron(), Photon()),
        ), "ep->epk",
    ),
    (
        ScatteringProcess(
            (Electron(), Positron()),
            (Electron(), Positron(), Photon(), Photon()),
        ), "ep->epkk",
    ),

    #(make_nphoton_compton(6), "ke->kkkkkke"),
]

SUITE = BenchmarkGroup()
SUITE["graph_gen"] = BenchmarkGroup()

graph_props = Dict{String, GraphProperties}()
comp_times = Dict{String, Vector{Float64}}()
comp_times_closures = Dict{String, Vector{Float64}}()
node_dicts = Dict{String, Dict{Type, Int64}}()

for (INSTANCE, INSTANCE_STR) in SCATTERING_PROCESSES
    println("Processing: $INSTANCE_STR")
    graph(INSTANCE)
    SUITE["graph_gen"][INSTANCE_STR] = @benchmarkable graph(proc) setup = (proc = $INSTANCE; GC.gc())

    g = graph(INSTANCE)
    graph_props[INSTANCE_STR] = properties(g)

    node_dicts[INSTANCE_STR] = Dict{Type, Int64}()
    for node in values(g.nodes)
        if haskey(node_dicts[INSTANCE_STR], typeof(task(node)))
            node_dicts[INSTANCE_STR][typeof(task(node))] = node_dicts[INSTANCE_STR][typeof(task(node))] + 1
        else
            node_dicts[INSTANCE_STR][typeof(task(node))] = 1
        end
    end

    psp = PhaseSpacePoint(
        INSTANCE,
        MODEL,
        PSL,
        tuple((rand(SFourMomentum) for _ in 1:number_incoming_particles(INSTANCE))...),
        tuple((rand(SFourMomentum) for _ in 1:number_outgoing_particles(INSTANCE))...),
    )

    func = compute_function(g, INSTANCE, cpu_st(), @__MODULE__)

    SUITE["f_gen"][INSTANCE_STR] = @benchmarkable compute_function(g_, proc, machine, @__MODULE__) setup = (
        g_ = $g; proc = $INSTANCE; machine = cpu_st(); GC.gc()
    )

    comp_times[INSTANCE_STR] = bench_compilation(
        :(f(p));
        setup = quote
            using QEDcore, QEDprocesses, QEDFeynmanDiagrams, ComputableDAGs
            ComputableDAGs.init(@__MODULE__)
            p = PhaseSpacePoint(
                $INSTANCE,
                $MODEL,
                FlatPhaseSpaceLayout(ComptonRestSystem()),
                tuple((rand(SFourMomentum) for _ in 1:number_incoming_particles($INSTANCE))...),
                tuple((rand(SFourMomentum) for _ in 1:number_outgoing_particles($INSTANCE))...),
            )
            f = compute_function($g, $INSTANCE, cpu_st(), @__MODULE__)
        end,
    )
    println("collected $(length(comp_times[INSTANCE_STR])) compile time samples")
    #=
    comp_times_closures[INSTANCE_STR] = bench_compilation(
        :(f(p));
        setup = quote
            using QEDcore, QEDprocesses, QEDFeynmanDiagrams, ComputableDAGs
            ComputableDAGs.init(@__MODULE__)
            p = PhaseSpacePoint(
                $INSTANCE,
                $MODEL,
                FlatPhaseSpaceLayout(ComptonRestSystem()),
                tuple((rand(SFourMomentum) for _ in 1:number_incoming_particles($INSTANCE))...),
                tuple((rand(SFourMomentum) for _ in 1:number_outgoing_particles($INSTANCE))...),
            )
            f = compute_function($g, $INSTANCE, cpu_st(), @__MODULE__; closures_size = 1000, concrete_input_type = typeof(p))
        end,
    )
    println("collected $(length(comp_times_closures[INSTANCE_STR])) compile time samples")=#
end

#result = run(SUITE; verbose = true)
#BenchmarkTools.save("$(@__DIR__)/data/bench.json", result)

@save "$(@__DIR__)/data/bench.jld2" graph_props node_dicts comp_times comp_times_closures
