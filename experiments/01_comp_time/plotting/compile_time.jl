# == Function Compile Time ==
@load "data/bench.jld2"

l = length(comp_times)
data = getindex.(Ref(comp_times), SCATTERING_PROCESSES[1:l])
data_closures = getindex.(Ref(comp_times_closures), SCATTERING_PROCESSES[1:l])
for i in eachindex(data)
    data[i] = data[i] *= 1.0e9 # convert to nanoseconds
    data_closures[i] = data_closures[i] *= 1.0e9
end
data = median.(data)
data_closures = median.(data_closures)

f = Figure()
ax = Axis(
    f[1, 1];
    xlabel = "number of incoming photons",
    ylabel = "function compile time",
    limits = (nothing, _find_y_lims([data, data_closures])),
    yminorgridvisible = true,
    yminorticksvisible = true,
    yminorticks = IntervalsBetween(10),
    yscale = log10,
    yticks = (yticks1, yticks2),
    xticks = ([(1:l)...], proc_str.(SCATTERING_PROCESSES[1:l])),
)

data_sc = scatter!(ax, [(1:l)...], data)
data_closures_sc = scatter!(ax, [(1:l)...], data_closures)

Legend(
    f[2, 1],
    [data_sc, data_closures_sc],
    ["compile time", "compile time with closures"];
    tellheight = true,
    tellwidth = false,
    margin = (10, 10, 10, 10),
    #halign = :center,
    valign = :bottom,
    orientation = :horizontal,
)

save(joinpath(plotpath, "f_compile_compton.pdf"), f)
