using NeuralEstimators
using GraphNeuralNetworks
using Flux
using SparseArrays
using Statistics: mean

f32(x) = Float32.(x)

# ----------------------------------------------------------
# edge feature layout for nodewise nonstationary anisotropy
# e = [x_i, y_i, x_j, y_j, Δx, Δy, d]'
# where i is the center node, j is the neighbour
# ----------------------------------------------------------
function nodegraph(
    S::AbstractMatrix;
    k::Integer = 12,
    r::AbstractFloat = 0.20f0,
    random::Bool = false,
)
    S = f32(S)

    # use package function directly
    A = adjacencymatrix(S; k = k, r = r, random = random)

    # In this adjacency convention:
    # row index I = neighbour j
    # col index J = center i
    I, J, V = findnz(A)

    s_i = permutedims(S[J, :])         # 2 × E  center locations
    s_j = permutedims(S[I, :])         # 2 × E  neighbour locations
    Δs  = s_j .- s_i                   # 2 × E  displacement
    d   = reshape(f32(V), 1, :)        # 1 × E  distance

    e = vcat(s_i, s_j, Δs, d)          # 7 × E

    GNNGraph(A, ndata = (S = permutedims(S),), edata = (e = e,))
end

# attach data to an already constructed graph
function reshapeZ(Z::AbstractVector)
    reshapeZ(reshape(Z, length(Z), 1))
end

function reshapeZ(Z::AbstractMatrix)
    # input: n × m
    # output required by SpatialGraphConv: q × m × n, here q = 1
    permutedims(reshape(f32(Z), 1, size(Z, 1), size(Z, 2)), (1, 3, 2))
end

function nodegraph(g::GNNGraph, Z)
    GNNGraph(g, ndata = (g.ndata..., Z = reshapeZ(Z)))
end

function nodegraph(S, Z; kwargs...)
    g = nodegraph(S; kwargs...)
    nodegraph(g, Z)
end



# ----------------------------------------------------------
# helper bins
# ----------------------------------------------------------
function interval_bins(lo::Real, hi::Real, n_bins::Integer)
    @assert hi > lo
    cut = collect(range(lo, stop = hi, length = n_bins + 1))
    mu = [(cut[i] + cut[i+1]) / 2 for i in 1:n_bins]
    sigma = [(cut[i+1] - cut[i]) / 4 for i in 1:n_bins]
    return f32(mu), f32(sigma)
end

function radial_bins(d_max::Real, n_bins::Integer)
    interval_bins(0, d_max, n_bins)
end

function angular_bins(n_bins::Integer)
    interval_bins(-π, π, n_bins)
end

wrap_angle(x) = Base.mod2pi(x + π) - π

# ----------------------------------------------------------
# local anisotropy weights: polar coordinates of displacement
# input can be 2×E (Δx,Δy) or 4×E (x_i,y_i,x_j,y_j)
# output: (Kd + Kp) × E
# ----------------------------------------------------------
struct KernelWeightsPolar{T1,T2}
    mud::T1
    sigd::T2
    mup::T1
    sigp::T2
end

function KernelWeightsPolar(d_max::Real, Kd::Integer, Kp::Integer)
    mud, sigd = radial_bins(d_max, Kd)
    mup, sigp = angular_bins(Kp)
    KernelWeightsPolar(mud, sigd, mup, sigp)
end

function (l::KernelWeightsPolar)(s::AbstractMatrix{T}) where {T}
    @assert size(s,1) == 2 || size(s,1) == 4

    if size(s,1) == 4
        xi, yi = @views s[1:1, :], s[2:2, :]
        xj, yj = @views s[3:3, :], s[4:4, :]
        Δx, Δy = xj .- xi, yj .- yi
    else
        Δx, Δy = @views s[1:1, :], s[2:2, :]
    end

    d  = sqrt.(Δx.^2 .+ Δy.^2)
    φ  = atan.(Δy, Δx)

    mud   = reshape(l.mud, :, 1)
    sigd2 = reshape(l.sigd .^ 2, :, 1)
    mup   = reshape(l.mup, :, 1)
    sigp2 = reshape(l.sigp .^ 2, :, 1)

    Nd = exp.(- (d .- mud).^2 ./ (2f0 .* sigd2))
    δφ = wrap_angle.(φ .- mup)
    Np = exp.(- (δφ).^2 ./ (2f0 .* sigp2))

    f32(vcat(Nd, Np))
end

Flux.trainable(::KernelWeightsPolar) = NamedTuple()

# ----------------------------------------------------------
# global nonstationarity weights: midpoint location
# input must be 4×E = (x_i,y_i,x_j,y_j)
# output: (Kx + Ky) × E
# ----------------------------------------------------------
struct KernelWeightsGlobal{T1,T2}
    mux::T1
    sigx::T2
    muy::T1
    sigy::T2
end

function KernelWeightsGlobal(
    domx::Tuple{Real,Real},
    domy::Tuple{Real,Real},
    Kx::Integer,
    Ky::Integer,
)
    mux, sigx = interval_bins(domx[1], domx[2], Kx)
    muy, sigy = interval_bins(domy[1], domy[2], Ky)
    KernelWeightsGlobal(mux, sigx, muy, sigy)
end

function (l::KernelWeightsGlobal)(s::AbstractMatrix{T}) where {T}
    @assert size(s,1) == 4

    xi, yi = @views s[1:1, :], s[2:2, :]
    xj, yj = @views s[3:3, :], s[4:4, :]

    mx, my = (xi .+ xj) ./ 2, (yi .+ yj) ./ 2

    Mx = [exp.(- (mx .- l.mux[i:i]).^2 ./ (2f0 * l.sigx[i:i].^2)) for i in eachindex(l.mux)]
    My = [exp.(- (my .- l.muy[i:i]).^2 ./ (2f0 * l.sigy[i:i].^2)) for i in eachindex(l.muy)]

    f32(vcat(reduce(vcat, Mx), reduce(vcat, My)))
end

Flux.trainable(::KernelWeightsGlobal) = NamedTuple()

# ----------------------------------------------------------
# combine local + global edge weights
# input e must be 7×E
# output: (Kd + Kp + Kx + Ky) × E
# ----------------------------------------------------------
struct CombinedEdgeWeights{A,B}
    local_w::A
    global_w::B
end

function (w::CombinedEdgeWeights)(e::AbstractMatrix{T}) where {T}
    @assert size(e,1) == 7
    # e[1:4,:] -> absolute endpoints
    # e[5:6,:] -> displacement
    vcat(
        w.local_w(e[5:6, :]),
        w.global_w(e[1:4, :]),
    )
end

Flux.trainable(::CombinedEdgeWeights) = NamedTuple()