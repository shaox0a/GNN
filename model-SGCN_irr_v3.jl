using Flux
using Statistics: mean
using GraphNeuralNetworks

# ==========================================================
# model-SGCN_irr_fixed_v4.jl
# ----------------------------------------------------------
# Nodewise SGCN-style estimator for irregular + fixed spatial
# locations.
#
# Changes relative to v3:
#   1) the SGCN edge weighting now follows Danel et al. (2020)
#      more faithfully:
#         ReLU(U' (p_j - p_i) + b) ⊙ h_j
#      with multi-filter concatenation;
#   2) the spatial weighting function is exactly one affine map
#      plus ReLU (implemented as Dense(2 => in*K) and relu.);
#   3) remove the previous outer-product / vec message expansion;
#   4) remove neighbourhood normalisation of edge weights.
#
# We still keep the GraphNeuralNetworks message-passing API
# (apply_edges / aggregate_neighbors / edge_index), and keep
# the existing nodewise output head for the sitewise regression
# task.
# ==========================================================

# function constrain_nodewise_sgcn(Yraw::AbstractMatrix)
#     rho1  = sigmoid.(Yraw[1:1, :])
#     rho2  = sigmoid.(Yraw[2:2, :])
#     delta = tanh.(Yraw[3:3, :])
#     return vcat(rho1, rho2, delta)
# end

# Same reshaping convention as in Graphs.txt / spatialgraph(g, Z)
reshape_nodewise_Z(Z::AbstractVector) = reshape_nodewise_Z(reshape(Z, length(Z), 1))
reshape_nodewise_Z(Z::AbstractMatrix) = reshape_nodewise_Z(reshape(Z, 1, size(Z)...))
function reshape_nodewise_Z(Z::AbstractArray{T,3}) where {T}
    return permutedims(Float32.(Z), (1, 3, 2))
end

# ------------------------------------------------------------------
# SGCN convolution.
# ------------------------------------------------------------------
struct SGCNConv{W1 <: AbstractMatrix, W2 <: AbstractMatrix, B, A, G} <: GNNLayer
    Γ1::W1          # self map (task adaptation; not in the original paper formula)
    Γ2::W2          # map after concatenated spatial filters
    b::B            # layer bias
    w::A            # one affine map: Dense(2 => in*nfilters)
    g::G            # output nonlinearity
    nfilters::Int   # number of SGCN filters (Eq. 4 in Danel et al.)
end

Flux.@functor SGCNConv

function SGCNConv(
    ch::Pair{Int, Int},
    g = relu;
    init = glorot_uniform,
    bias::Bool = true,
    w = nothing,
    nfilters::Integer = 16,
)
    in, out = ch

    # Original SGCN weighting block:
    #   ReLU(U' Δp + b)
    # with K filters concatenated together.
    # Implemented as one Dense layer returning in*K outputs.
    if isnothing(w)
        w = Dense(2, in * nfilters; init = init, bias = true)
    end

    Γ1 = init(out, in)
    Γ2 = init(out, in * nfilters)
    b  = bias ? Flux.create_bias(Γ1, true, out) : false

    return SGCNConv(Γ1, Γ2, b, w, g, Int(nfilters))
end

function (l::SGCNConv)(g::GNNGraph)
    h = l(g, g.ndata.Z)
    return GNNGraph(g, ndata = (Z = h, g.ndata.S))
end

function (l::SGCNConv)(g::GNNGraph, x::AbstractMatrix)
    return l(g, reshape(x, size(x, 1), 1, size(x, 2)))
end

function (l::SGCNConv)(g::GNNGraph, x::AbstractArray{T,3}) where {T}
    check_num_nodes(g, x)

    # x: c × m × n
    c, m, _ = size(x)
    K = l.nfilters

    # Edge features: either displacement directly, or full
    # [x_i, y_i, x_j, y_j, Δx, Δy, d] layout.
    e = :e ∈ keys(g.edata) ? g.edata.e : permutedims(g.graph[3])
    if isa(e, AbstractVector)
        e = permutedims(e)
    end
    
    Δp = size(e, 1) >= 6 ? e[5:6, :] : e
    α = relu.(l.w(Δp))
    # if size(e, 1) >= 6
    #     p_i = e[1:2, :]      # center node position
    #     Δp  = e[5:6, :]      # relative displacement
    # else
    #     s, _ = edge_index(g)
    #     p_i = g.ndata.S[:, s]
    #     Δp  = e[1:2, :]
    # end
    # spatial_input = vcat(p_i, Δp)   # 4 × E
    # α = relu.(l.w(spatial_input))

    # Spatially weighted neighbour messages:
    # for each filter k, multiply α_k(i,j) elementwise with h_j,
    # then concatenate over filters.
    msg = apply_edges((xi, xj, α) -> begin
        E = size(α, 2)
        α4 = reshape(α, c, K, 1, E)         # c × K × 1 × E
        xj4 = reshape(xj, c, 1, m, E)       # c × 1 × m × E
        z = α4 .* xj4                       # c × K × m × E
        reshape(z, c * K, m, E)            # (cK) × m × E
    end, g, x, x, α)

    # Sum messages over incoming neighbours
    h̄ = aggregate_neighbors(g, +, msg)      # (cK) × m × n

    # Pointwise linear maps over nodes / replicates
    xself = l.Γ1 ⊠ x
    xagg  = l.Γ2 ⊠ h̄
    # xself = reshape(l.Γ1 * reshape(x,  c,        :), size(l.Γ1, 1), size(x, 2),  size(x, 3))
    # xagg  = reshape(l.Γ2 * reshape(h̄, c * K,    :), size(l.Γ2, 1), size(h̄, 2), size(h̄, 3))

    return l.g.(xself .+ xagg .+ l.b)
end

function Base.show(io::IO, l::SGCNConv)
    in_channel  = size(l.Γ1, 2)
    out_channel = size(l.Γ1, 1)
    print(io, "SGCNConv(", in_channel, " => ", out_channel)
    print(io, ", nfilters=", l.nfilters)
    l.g == identity || print(io, ", ", l.g)
    print(io, ", w=", l.w)
    print(io, ")")
end


# ------------------------------------------------------------------
# Nodewise model: fully separate heads (no explicit S to head)
# ------------------------------------------------------------------
struct NodewiseSGCN{P, H1, H2, H3}
    propagation::P
    head_rho1::H1
    head_rho2::H2
    head_delta::H3
end

Flux.@functor NodewiseSGCN

function (m::NodewiseSGCN)(g::GNNGraph)
    h = m.propagation(g)
    H = h.ndata.Z

    # pool over replicate dimension only, keep nodewise outputs
    # Hsummary = dropdims(mean(H, dims = 2), dims = 2)   # hidden × n
    Hsummary = dropdims(sum(H, dims = 2), dims = 2)

    rho1_raw  = m.head_rho1(Hsummary)                  # 1 × n
    rho2_raw  = m.head_rho2(Hsummary)                  # 1 × n
    delta_raw = m.head_delta(Hsummary)                 # 1 × n

    return vcat(rho1_raw, rho2_raw, delta_raw)      # 3 × n
end

function make_nodewise_sgcn(
    ;
    hidden1::Int = 64,
    hidden2::Int = 64,
    hidden3::Int = 64,
    K1::Int = 16,
    K2::Int = 16,
    K3::Int = 16,
    head_hidden1::Int = 32,
    head_hidden2::Int = 32,
    activation = relu,
)
    layer1 = SGCNConv(1       => hidden1, activation; nfilters = K1)
    layer2 = SGCNConv(hidden1 => hidden2, activation; nfilters = K2)
    layer3 = SGCNConv(hidden2 => hidden3, activation; nfilters = K3)

    propagation = GNNChain(layer1, layer2, layer3)

    head_in = hidden3

    head_rho1 = Chain(
        Dense(head_in,       head_hidden1, activation),
        Dense(head_hidden1,  head_hidden2, activation),
        Dense(head_hidden2,  1, sigmoid),
    )

    head_rho2 = Chain(
        Dense(head_in,       head_hidden1, activation),
        Dense(head_hidden1,  head_hidden2, activation),
        Dense(head_hidden2,  1, sigmoid),
    )

    head_delta = Chain(
        Dense(head_in,       head_hidden1, activation),
        Dense(head_hidden1,  head_hidden2, activation),
        Dense(head_hidden2,  1, tanh),
    )

    return NodewiseSGCN(
        propagation,
        head_rho1,
        head_rho2,
        head_delta,
    )
end

# ------------------------------------------------------------------
# Losses / prediction helpers
# ------------------------------------------------------------------
# function graph_smoothness_sgcn(g::GNNGraph, θ::AbstractMatrix)
#     s, t = edge_index(g)
#     return mean(abs, θ[:, s] .- θ[:, t])
# end

# function node_loss_sgcn(model, g::GNNGraph, y; λ_smooth::Real = 0f0)
#     θhat = model(g)
#     mae = mean(abs, θhat .- y)
#     return λ_smooth > 0 ? mae + λ_smooth * graph_smoothness_sgcn(g, θhat) : mae
# end

# function node_dataset_loss_sgcn(model, X_list, Y_list; λ_smooth::Real = 0f0)
#     vals = map(eachindex(X_list)) do i
#         node_loss_sgcn(model, X_list[i], Y_list[i]; λ_smooth = λ_smooth)
#     end
#     return mean(vals)
# end

# function predict_nodewise_sgcn(model, g::GNNGraph)
#     θhat = model(g)   # q × n
#     return Array(θhat)
# end

# function predict_nodewise_sgcn(model, g::GNNGraph, Z)
#     gZ = GNNGraph(g, ndata = (g.ndata..., Z = reshape_nodewise_Z(Z)))
#     return predict_nodewise_sgcn(model, gZ)
# end


# # ------------------------------------------------------------------
# # Losses / prediction helpers 2
# # ------------------------------------------------------------------

# as_channel_first(y::AbstractMatrix) = size(y, 1) == 3 ? y : permutedims(y)

# function estimate_channel_scale(Y_list; eps::Float32 = 1f-3)
#     Ys = map(as_channel_first, Y_list)
#     Ycat = reduce(hcat, Ys)                      # 3 × total_nodes

#     μ = vec(mean(Ycat, dims = 2))                # 3
#     mad = vec(mean(abs.(Ycat .- μ), dims = 2))   # 3

#     return Float32.(max.(mad, eps))
# end

# function channelwise_mae(
#     θhat::AbstractMatrix,
#     y::AbstractMatrix;
#     channel_scale::AbstractVector = Float32[1, 1, 1],
#     channel_weight::AbstractVector = Float32[1, 1, 1],
# )
#     ycf = as_channel_first(y)                    # 3 × n
#     err = abs.(θhat .- ycf)                      # 3 × n
#     mae_ch = vec(mean(err, dims = 2))            # 3
#     loss_ch = mae_ch ./ channel_scale            # 3

#     return sum(channel_weight .* loss_ch) / sum(channel_weight)
# end

# function graph_smoothness_sgcn(
#     g::GNNGraph,
#     θ::AbstractMatrix;
#     channel_scale::AbstractVector = Float32[1, 1, 1],
#     channel_weight::AbstractVector = Float32[1, 1, 1],
# )
#     s, t = edge_index(g)

#     diff = abs.(θ[:, s] .- θ[:, t])              # 3 × E
#     sm_ch = vec(mean(diff, dims = 2))            # 3
#     loss_ch = sm_ch ./ channel_scale             # 3

#     return sum(channel_weight .* loss_ch) / sum(channel_weight)
# end

# function node_loss_sgcn(
#     model,
#     g::GNNGraph,
#     y;
#     λ_smooth::Real = 0f0,
#     channel_scale::AbstractVector = Float32[1, 1, 1],
#     channel_weight::AbstractVector = Float32[1, 1, 1],
# )
#     θhat = model(g)

#     fit_loss = channelwise_mae(
#         θhat, y;
#         channel_scale = channel_scale,
#         channel_weight = channel_weight,
#     )

#     if λ_smooth > 0
#         smooth_loss = graph_smoothness_sgcn(
#             g, θhat;
#             channel_scale = channel_scale,
#             channel_weight = channel_weight,
#         )
#         return fit_loss + λ_smooth * smooth_loss
#     else
#         return fit_loss
#     end
# end

# function node_dataset_loss_sgcn(
#     model,
#     X_list,
#     Y_list;
#     λ_smooth::Real = 0f0,
#     channel_scale::AbstractVector = Float32[1, 1, 1],
#     channel_weight::AbstractVector = Float32[1, 1, 1],
# )
#     vals = map(eachindex(X_list)) do i
#         node_loss_sgcn(
#             model,
#             X_list[i],
#             Y_list[i];
#             λ_smooth = λ_smooth,
#             channel_scale = channel_scale,
#             channel_weight = channel_weight,
#         )
#     end
#     return mean(vals)
# end

# function predict_nodewise_sgcn(model, g::GNNGraph)
#     θhat = model(g)
#     return Array(vec(θhat[1, :])), Array(vec(θhat[2, :])), Array(vec(θhat[3, :]))
# end

# function predict_nodewise_sgcn(model, g::GNNGraph, Z)
#     gZ = GNNGraph(g, ndata = (g.ndata..., Z = reshape_nodewise_Z(Z)))
#     return predict_nodewise_sgcn(model, gZ)
# end