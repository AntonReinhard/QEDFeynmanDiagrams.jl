# == Function Compile Time ==
@load "data/bench.jld2"

PROCS = keys(comp_times)

data = getindex.(Ref(comp_times), PROCS)
props_vec = getindex.(Ref(graph_props), PROCS)
@show no_nodes = getfield.(props_vec, :number_of_nodes)
for i in eachindex(data)
    data[i] = data[i] *= 1.0e9 # convert to nanoseconds
end
@show data = median.(data)

f = Figure()
ax = Axis(
    f[1, 1];
    xlabel = "lines of code",
    ylabel = "function compile time",
    limits = (nothing, _find_y_lims(data)),
    yminorgridvisible = true,
    yminorticksvisible = true,
    yminorticks = IntervalsBetween(10),
    yscale = log10,
    xscale = log10,
    yticks = (yticks1, yticks2),
    #xticks = ()
)

data_sc = scatter!(ax, no_nodes, data)

Legend(
    f[2, 1],
    [data_sc],
    ["compile time"];
    tellheight = true,
    tellwidth = false,
    margin = (10, 10, 10, 10),
    #halign = :center,
    valign = :bottom,
    orientation = :horizontal,
)

save(joinpath(plotpath, "f_compile_time_to_lines.pdf"), f)
