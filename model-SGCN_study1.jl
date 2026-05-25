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
# Study1 Nodewise model:
#   only estimate one sitewise rho surface
#
# output:
#   rho_hat : 1 × n
#
# Study1 covariance interpretation:
#   rho1 = rho
#   rho2 = rho
#   delta = 0
# ------------------------------------------------------------------

struct NodewiseSGCNStudy1{P, H}
    propagation::P
    head_rho::H
end

Flux.@functor NodewiseSGCNStudy1

function (m::NodewiseSGCNStudy1)(g::GNNGraph)
    h = m.propagation(g)
    H = h.ndata.Z

    # H: hidden × m × n
    # pool over replicate dimension only, keep nodewise outputs
    Hsummary = dropdims(mean(H, dims = 2), dims = 2)   # hidden × n

    rho_raw = m.head_rho(Hsummary)                     # 1 × n

    return sigmoid.(rho_raw)                           # 1 × n
end

function make_nodewise_sgcn_study1(
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

    head_rho = Chain(
        Dense(head_in,       head_hidden1, activation),
        Dense(head_hidden1,  head_hidden2, activation),
        Dense(head_hidden2,  1),
    )

    return NodewiseSGCNStudy1(
        propagation,
        head_rho,
    )
end







using NNlib: softplus

# ------------------------------------------------------------------
# Study1 Nodewise SGCN interval model:
#   only estimate one sitewise rho surface interval
#
# output:
#   2 × n
#   row 1 = rho_lower(s)
#   row 2 = rho_upper(s)
#
# construction:
#   η_lower = head_lower(Hsummary)
#   η_upper = η_lower + softplus(head_width(Hsummary))
#
#   rho_lower = sigmoid(η_lower)
#   rho_upper = sigmoid(η_upper)
#
# guarantees:
#   0 < rho_lower < rho_upper < 1
# ------------------------------------------------------------------

struct NodewiseSGCNStudy1Interval{P, HL, HW}
    propagation::P
    head_lower::HL
    head_width::HW
end

Flux.@functor NodewiseSGCNStudy1Interval

function (m::NodewiseSGCNStudy1Interval)(g::GNNGraph)
    h = m.propagation(g)
    H = h.ndata.Z

    # H: hidden × m × n
    # pool over replicate dimension only
    Hsummary = dropdims(mean(H, dims = 2), dims = 2)   # hidden × n

    η_lower = m.head_lower(Hsummary)                   # 1 × n
    w_raw   = m.head_width(Hsummary)                   # 1 × n

    η_upper = η_lower .+ softplus.(w_raw)

    rho_lower = sigmoid.(η_lower)                      # 1 × n
    rho_upper = sigmoid.(η_upper)                      # 1 × n

    return vcat(rho_lower, rho_upper)                  # 2 × n
end

function make_nodewise_sgcn_study1_interval(
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

    head_lower = Chain(
        Dense(head_in,       head_hidden1, activation),
        Dense(head_hidden1,  head_hidden2, activation),
        Dense(head_hidden2,  1),
    )

    head_width = Chain(
        Dense(head_in,       head_hidden1, activation),
        Dense(head_hidden1,  head_hidden2, activation),
        Dense(head_hidden2,  1),
    )

    return NodewiseSGCNStudy1Interval(
        propagation,
        head_lower,
        head_width,
    )
end