using Unitful, Latexify, UnitfulLatexify

"""
    _find_y_lims(data)
Return the limits which encloses the data, i.e. the supremum and infimum of the data in powers of 10.
This is useful for fining limit of logarithmic data.
"""
function _find_y_lims(data::BenchmarkTools.BenchmarkGroup)
    t_min = time(minimum(data))
    data_min = minimum(values(t_min))
    ymin = 10^floor(log10(data_min))

    t_max = time(maximum(data))
    data_max = maximum(values(t_max))
    ymax = 10^ceil(log10(data_max))

    return (ymin, ymax)
end

function _find_y_lims(data::AbstractVector{<:Number})
    y_min = minimum(data)
    y_min = 10^floor(log10(y_min))

    y_max = maximum(data)
    y_max = 10^ceil(log10(y_max))

    return (y_min, y_max)
end

function _find_y_lims(data::AbstractVector{T}) where {T}
    local y_min = Inf64
    local y_max = -Inf64
    for t in data
        (y_min_n, y_max_n) = _find_y_lims(t)
        y_min = min(y_min_n, y_min)
        y_max = max(y_max_n, y_max)
    end

    return (y_min, y_max)
end

yticks1 = [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13]
yticks2 = [
    latexify(1u"ns"),
    latexify(10u"ns"),
    latexify(100u"ns"),
    latexify(1u"μs"),
    latexify(10u"μs"),
    latexify(100u"μs"),
    latexify(1u"ms"),
    latexify(10u"ms"),
    latexify(100u"ms"),
    latexify(1u"s"),
    latexify(10u"s"),
    latexify(100u"s"),
    latexify(1u"ks"),
    latexify(10u"ks"),
]
