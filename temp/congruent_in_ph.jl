ENV["UCX_ERROR_SIGNALS"] = "SIGILL,SIGBUS,SIGFPE"

using ComputableDAGs
using QEDbase
using QEDcore
using QEDprocesses
using QEDFeynmanDiagrams
using Random
using UUIDs

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using NamedDims
using CSV
using JLD2
using FlexiMaps

RNG = Random.MersenneTwister(123)

# theta ∈ [0, 2π] and phi ∈ [0, 2π]
function congruent_input_momenta(
    processDescription::ScatteringProcess, omega::Number, theta::Number, phi::Number
)
    # -------------
    # same as above

    # generate an input sample for given e + nk -> e' + k' process, where the nk are equal
    inputMasses = Vector{Float64}()
    for particle in incoming_particles(processDescription)
        push!(inputMasses, mass(particle))
    end
    outputMasses = Vector{Float64}()
    for particle in outgoing_particles(processDescription)
        push!(outputMasses, mass(particle))
    end

    initial_momenta = [
        if i == length(inputMasses)
            SFourMomentum(1, 0, 0, 0)
        else
            SFourMomentum(omega, 0, 0, omega)
        end for i in 1:length(inputMasses)
    ]

    ss = sqrt(sum(initial_momenta) * sum(initial_momenta))

    # up to here
    # ----------

    # now calculate the final_momenta from omega, cos_theta and phi
    n = number_particles(processDescription, Incoming(), Photon())

    cos_theta = cos(theta)
    omega_prime = (n * omega) / (1 + n * omega * (1 - cos_theta))

    k_prime =
        omega_prime * SFourMomentum(
            1, sqrt(1 - cos_theta^2) * cos(phi), sqrt(1 - cos_theta^2) * sin(phi), cos_theta
        )
    p_prime = sum(initial_momenta) - k_prime

    final_momenta = (k_prime, p_prime)

    return (tuple(initial_momenta...), tuple(final_momenta...))
end

function build_psp(processDescription::ScatteringProcess, momenta)
    return PhaseSpacePoint(
        processDescription,
        PerturbativeQED(),
        PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
        momenta[1],
        momenta[2],
    )
end

println("Running with $(Threads.nthreads()) threads")

# hack to fix stacksize for threading
with_stacksize(f, n) = fetch(schedule(Task(f, n)))

# scenario 2
N = 64     # thetas
M = 64     # phis
K = 64     # omegas

thetas = collect(LinRange(0, 2π, N))
phis = collect(LinRange(0, 2π, M))
omegas = collect(maprange(log, 2e2, 2e-7, K))

println("Gridsize:")
println("$N thetas from $(thetas[1]) to $(thetas[end])")
println("$M phis from $(phis[1]) to $(phis[end])")
println("$K omegas from $(omegas[1]) to $(omegas[end])")

for photons in 1:3
    # temp process to generate momenta
    println("Generating $(K*N*M) inputs for $photons photons")

    temp_process = ScatteringProcess(
        (Electron(), ntuple(_ -> Photon(), photons)...),  # incoming particles
        (Electron(), Photon()),                           # outgoing particles
        (AllSpin(), ntuple(_ -> AllPol(), photons)...),   # incoming particle spin/pols
        (AllSpin(), AllPol()),                            # outgoing particle spin/pols
    )

    GC.gc()

    input_momenta = Array{
        typeof(congruent_input_momenta(temp_process, omegas[1], thetas[1], phis[1]))
    }(
        undef, (K, N, M)
    )

    Threads.@threads for k in 1:K
        Threads.@threads for i in 1:N
            Threads.@threads for j in 1:M
                input_momenta[k, i, j] = congruent_input_momenta(
                    temp_process, omegas[k], thetas[i], phis[j]
                )
            end
        end
    end

    results = zeros((K, N, M))
    #=
    results = CuArray{Float64}(undef, size(input_momenta))
    fill!(cu_results, 0.0)
    =#

    i = 1
    for in_pol in [PolX(), PolY()]
        println("[$i/2] Calculating for in-pol $in_pol... Preparing inputs... ")

        process = ScatteringProcess(
            (Electron(), ntuple(_ -> Photon(), photons)...),  # incoming particles
            (Electron(), Photon()),                           # outgoing particles
            (AllSpin(), ntuple(_ -> in_pol, photons)...),     # incoming particle spin/pols
            (AllSpin(), AllPol()),                            # outgoing particle spin/pols
        )

        #=
        inputs = Array{typeof(build_psp(process, input_momenta[1, 1, 1]))}(undef, (K, N, M))
        #println("input_momenta: $input_momenta")
        Threads.@threads for k in 1:K
            Threads.@threads for i in 1:N
                Threads.@threads for j in 1:M
                    inputs[k, i, j] = 
                end
            end
        end
        #cu_inputs = CuArray(inputs)
        =#

        println("Preparing graph... ")
        @time graph = generate_DAG(process)
        println("  Graph has $(length(graph.nodes)) nodes")
        #optimize_to_fixpoint!(ReductionOptimizer(), graph)
        println("Preparing function... ")
        #kernel! = get_cuda_kernel(graph, process, cpu_st())
        @time func = get_compute_function(graph, process, cpu_st(), @__MODULE__)
        println("Initial function call for compilation... ")
        initial_input = build_psp(process, input_momenta[1, 1, 1])
        @time func(initial_input)

        println("Calculating... ")
        @time Threads.@threads for k in 1:K
            with_stacksize(64 * 1024 * 1024) do
                for i in 1:N
                    for j in 1:M
                        results[k, i, j] += func(build_psp(process, input_momenta[k, i, j]))
                    end
                end
            end
        end
        #=ts = 32
        bs = Int64(length(cu_inputs) / 32)

        outputs = CuArray{ComplexF64}(undef, size(cu_inputs))

        @cuda threads = ts blocks = bs always_inline = true kernel!(
            cu_inputs, outputs, length(cu_inputs)
        )
        CUDA.device_synchronize()
        cu_results += abs2.(outputs)
        =#

        println("Done.")
        i += 1
    end

    println("Writing results")

    out_ph_moms = getindex.(getindex.(input_momenta, 2), 1)
    out_el_moms = getindex.(getindex.(input_momenta, 2), 2)

    results = NamedDimsArray{(:omegas, :thetas, :phis)}(results)
    println("Named results array: $(typeof(results))")

    @save "$(photons)_congruent_photons_grid.jld2" omegas thetas phis results
end
