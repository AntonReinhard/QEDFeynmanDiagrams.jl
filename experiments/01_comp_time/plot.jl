using ComputableDAGs
using QEDFeynmanDiagrams
using QEDcore, QEDprocesses

using BenchmarkTools
using JLD2

using CairoMakie
using BenchmarkPlots
using LaTeXStrings

jsonfile = "$(@__DIR__)/data/bench.json"

include("$(@__DIR__)/../utils.jl")

plotpath = "$(@__DIR__)/plots"
if !isdir(plotpath)
    mkdir(plotpath)
end

_to_vec(s) = [s]

function proc_str(s::String)
    s = replace(s, "e" => "e^-")
    s = replace(s, "k" => "γ")

    (prefix, suffix) = split(s, "->")
    k_count = count(c -> c == 'γ', suffix)

    if k_count > 1
        suffix = replace(suffix, r"γ+" => "γ^$k_count")
    end

    return L"%$(k_count)"
end

SCATTERING_PROCESSES = [
    "ke->ke", "ke->kke", "ke->kkke", "ke->kkkke", "ke->kkkkke",
]

with_theme(theme_latexfonts()) do
    #include("plotting/compile_time.jl")
    include("plotting/compile_time_to_lines.jl")
end
