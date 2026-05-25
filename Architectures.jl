struct NodewiseWrapper{M}
    model::M
end

function (w::NodewiseWrapper)(g::GNNGraph)
    return w.model(g)   # 3 × n
end

# function (w::NodewiseWrapper)(Gs::AbstractVector{<:GNNGraph})
#     θhats = map(g -> w.model(g), Gs)
#     return cat(θhats...; dims = 3)  # 3 × n × K
# end
function (w::NodewiseWrapper)(Gs::AbstractVector{<:GNNGraph})
    return map(g -> w.model(g), Gs)
end