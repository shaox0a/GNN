using Random
using Distributions
using Folds
using LinearAlgebra
using SpecialFunctions
using StaticArrays

# ---------------------------------------------------------
# 依赖：
# 1) 先 include prior_experiment3.jl
# 2) 再 include Surface_types.jl
# 3) 再 include Graph_nodewise.jl
# ---------------------------------------------------------

# =========================================================
# helpers
# =========================================================

function expandgrid(x::AbstractVector, y::AbstractVector)
    n = length(x) * length(y)
    S = Matrix{Float64}(undef, n, 2)

    idx = 1
    for yy in y, xx in x
        S[idx, 1] = xx
        S[idx, 2] = yy
        idx += 1
    end
    return S
end

# =========================================================
# covariance construction
# =========================================================

@inline function matern_core(d::Real, ν::Real)
    if d <= 0
        return 1.0
    else
        return (2^(1 - ν) / gamma(ν)) * (d^ν) * besselk(ν, d)
    end
end

@inline Omega_general(ρ1::Real, ρ2::Real, δ::Real) =
    @SMatrix [ρ1^2  ρ1 * ρ2 * δ;
              ρ1 * ρ2 * δ  ρ2^2]

function covariancematrix(
    coords::AbstractMatrix,
    sig::AbstractVector,
    r1::AbstractVector,
    r2::AbstractVector,
    del::AbstractVector,
    ν::Real;
    nugget::Real = 0.0,
    jitter::Real = 1e-6,
)
    n = size(coords, 1)
    @assert size(coords, 2) == 2 "coords must be n×2"
    @assert length(sig) == n == length(r1) == length(r2) == length(del)

    δ = clamp.(del, -1 + 1e-8, 1 - 1e-8)

    Omegas  = Vector{Matrix{Float64}}(undef, n)
    logdetO = Vector{Float64}(undef, n)

    for i in 1:n
        Om = Omega_general(r1[i], r2[i], δ[i])
        Omegas[i]  = Om
        logdetO[i] = logdet(Om)
    end

    C = Matrix{Float64}(undef, n, n)

    @inbounds for i in 1:n
        C[i, i] = sig[i]^2
        Mi = Omegas[i]

        for j in (i + 1):n
            Mj = Omegas[j]
            Mhalf = 0.5 .* (Mi .+ Mj)

            logA = 0.25 * (logdetO[i] + logdetO[j]) - 0.5 * logdet(Mhalf)
            Aij  = exp(logA)

            @views h = coords[i, :] .- coords[j, :]
            Dij = sqrt(dot(h, Mhalf \ h))
            Rij = matern_core(Dij, ν)

            Cij = sig[i] * sig[j] * Aij * Rij
            C[i, j] = Cij
            C[j, i] = Cij
        end
    end

    for i in 1:n
        C[i, i] += nugget + jitter
    end

    return C
end

# =========================================================
# parameters container
# =========================================================

struct Parameters <: ParameterConfigurations
    θ
    L
    S
    g
end

get_sites(parameters::Parameters, k::Integer) =
    parameters.S isa AbstractVector ? parameters.S[k] : parameters.S

get_graph(parameters::Parameters, k::Integer) =
    parameters.g isa AbstractVector ? parameters.g[k] : parameters.g

get_surface(parameters::Parameters, k::Integer) =
    parameters.θ[k]

# =========================================================
# prior()
# =========================================================

# function prior(
#     K::Integer,
#     fixed_locs::Bool,
#     surface_sampler::Function;
#     gridded::Bool = true,
#     grid_side::Integer = 16,
#     n_range::UnitRange{Int} = 200:300,
#     λ_range::Tuple{<:Real,<:Real} = (10.0, 90.0),
#     loc_sampler::Symbol = :materncluster,
#     graph_k::Integer = 12,
#     graph_r::Float32 = 0.20f0,
#     S_fixed = nothing,
#     g_fixed = nothing,
# )
#     K >= 1 || throw(ArgumentError("K must be >= 1."))
#     isempty(n_range) && throw(ArgumentError("n_range cannot be empty."))

#     λlow, λhigh = float(λ_range[1]), float(λ_range[2])
#     λlow < λhigh || throw(ArgumentError("λ_range must satisfy low < high."))

#     # --------------------------------------------------
#     # 1) spatial locations
#     # --------------------------------------------------
#     if fixed_locs
#         if S_fixed !== nothing
#             S = S_fixed
#         else
#             if gridded
#                 pts = collect(range(0, 1, length = grid_side))
#                 S = expandgrid(pts, pts)
#             else
#                 if loc_sampler == :materncluster
#                     n_fixed = rand(n_range)
#                     λ_fixed = rand(Uniform(λlow, λhigh))
#                     S = maternclusterprocess(λ = λ_fixed, μ = n_fixed / λ_fixed)
#                 else
#                     error("Unsupported loc_sampler = $loc_sampler.")
#                 end
#             end
#         end
#     else
#         gridded && throw(ArgumentError("gridded=true currently requires fixed_locs=true."))

#         nvec = rand(n_range, K)
#         λvec = rand(Uniform(λlow, λhigh), K)

#         if loc_sampler == :materncluster
#             S = [maternclusterprocess(λ = λvec[k], μ = nvec[k] / λvec[k]) for k in 1:K]
#         else
#             error("Unsupported loc_sampler = $loc_sampler.")
#         end
#     end

#     # --------------------------------------------------
#     # 2) graphs
#     # --------------------------------------------------
#     if fixed_locs
#         g = (g_fixed === nothing) ? nodegraph(S; k = graph_k, r = graph_r, random = false) : g_fixed
#     else
#         g = [nodegraph(S[k]; k = graph_k, r = graph_r, random = false) for k in 1:K]
#     end

#     # --------------------------------------------------
#     # 3) sample θ and attach graph
#     # --------------------------------------------------
#     θ = Vector{SurfaceBundle{Float32}}(undef, K)

#     for k in 1:K
#         S_k = fixed_locs ? S : S[k]
#         g_k = fixed_locs ? g : g[k]

#         surf_k = surface_sampler(S_k)

#         θ[k] = SurfaceBundle(
#             Float32.(surf_k.sigma),
#             Float32.(surf_k.rho1),
#             Float32.(surf_k.rho2),
#             Float32.(surf_k.delta);
#             g = g_k,
#         )
#     end

#     # --------------------------------------------------
#     # 4) cholesky factors
#     # --------------------------------------------------
#     L = Folds.map(1:K) do k
#         S_k  = fixed_locs ? S : S[k]
#         surf = θ[k]

#         Σ = Symmetric(covariancematrix(
#             S_k,
#             Float64.(surf.sigma),
#             Float64.(surf.rho1),
#             Float64.(surf.rho2),
#             Float64.(surf.delta),
#             1.5,
#         ))

#         cholesky(Σ).L
#     end

#     return Parameters(θ, L, S, g)
# end
function prior(
    K::Integer,
    fixed_locs::Bool,
    surface_sampler::Function;
    gridded::Bool = true,
    grid_side::Integer = 16,
    n_range::UnitRange{Int} = 200:300,
    λ_range::Tuple{<:Real,<:Real} = (10.0, 90.0),
    loc_sampler::Symbol = :materncluster,
    graph_k::Integer = 12,
    graph_r::Float32 = 0.20f0,
    S_fixed = nothing,
    g_fixed = nothing,
    model_type::Symbol = :nonstationary,
)
    K >= 1 || throw(ArgumentError("K must be >= 1."))
    isempty(n_range) && throw(ArgumentError("n_range cannot be empty."))

    λlow, λhigh = float(λ_range[1]), float(λ_range[2])
    λlow < λhigh || throw(ArgumentError("λ_range must satisfy low < high."))

    # --------------------------------------------------
    # 1) spatial locations
    # --------------------------------------------------
    if fixed_locs
        if S_fixed !== nothing
            S = S_fixed
        else
            if gridded
                pts = collect(range(0, 1, length = grid_side))
                S = expandgrid(pts, pts)
            else
                if loc_sampler == :materncluster
                    n_fixed = rand(n_range)
                    λ_fixed = rand(Uniform(λlow, λhigh))
                    S = maternclusterprocess(λ = λ_fixed, μ = n_fixed / λ_fixed)
                else
                    error("Unsupported loc_sampler = $loc_sampler.")
                end
            end
        end
    else
        gridded && throw(ArgumentError("gridded=true currently requires fixed_locs=true."))

        nvec = rand(n_range, K)
        λvec = rand(Uniform(λlow, λhigh), K)

        if loc_sampler == :materncluster
            S = [maternclusterprocess(λ = λvec[k], μ = nvec[k] / λvec[k]) for k in 1:K]
        else
            error("Unsupported loc_sampler = $loc_sampler.")
        end
    end

    # --------------------------------------------------
    # 2) graphs
    # --------------------------------------------------
    if fixed_locs
        g = (g_fixed === nothing) ? nodegraph(S; k = graph_k, r = graph_r, random = false) : g_fixed
    else
        g = [nodegraph(S[k]; k = graph_k, r = graph_r, random = false) for k in 1:K]
    end

    # --------------------------------------------------
    # 3) sample θ and attach graph
    # --------------------------------------------------
    θ = Vector{SurfaceBundle{Float32}}(undef, K)

    for k in 1:K
        S_k = fixed_locs ? S : S[k]
        g_k = fixed_locs ? g : g[k]

        # sampler 本身也支持 model_type
        surf_k = surface_sampler(S_k; model_type = model_type)

        sigma = Float32.(surf_k.sigma)
        rho1  = Float32.(surf_k.rho1)

        if model_type == :local_isotropy
            rho2  = copy(rho1)
            delta = fill(0f0, length(rho1))
        else
            rho2  = Float32.(surf_k.rho2)
            delta = Float32.(surf_k.delta)
        end

        θ[k] = SurfaceBundle(
            sigma,
            rho1,
            rho2,
            delta;
            nu = surf_k.nu,
            g = g_k,
        )
    end

    # --------------------------------------------------
    # 4) cholesky factors
    # --------------------------------------------------
    L = Folds.map(1:K) do k
        S_k  = fixed_locs ? S : S[k]
        surf = θ[k]

        Σ = Symmetric(covariancematrix(
            S_k,
            Float64.(surf.sigma),
            Float64.(surf.rho1),
            Float64.(surf.rho2),
            Float64.(surf.delta),
            Float64(surf.nu),
        ))

        cholesky(Σ).L
    end

    return Parameters(θ, L, S, g)
end


# # =========================================================
# # simulate
# # =========================================================

# function simulate(
#     parameters::Parameters,
#     m::Integer = 100;
#     gridded::Bool = true,
#     grid_shape::Tuple{Int,Int} = (16, 16),
# )
#     K = length(parameters.L)
#     mvec = fill(Int(m), K)
#     return _simulate_vec(parameters, mvec; gridded = gridded, grid_shape = grid_shape)
# end

# function simulate(
#     parameters::Parameters,
#     mvec::AbstractVector{<:Integer};
#     gridded::Bool = true,
#     grid_shape::Tuple{Int,Int} = (16, 16),
# )
#     K = length(parameters.L)
#     length(mvec) == K || throw(ArgumentError("length(mvec) must be $K"))
#     return _simulate_vec(parameters, mvec; gridded = gridded, grid_shape = grid_shape)
# end

# function _simulate_vec(
#     parameters::Parameters,
#     mvec::AbstractVector{<:Integer};
#     gridded::Bool = true,
#     grid_shape::Tuple{Int,Int} = (16, 16),
# )
#     K = length(parameters.L)

#     return Folds.map(1:K) do k
#         L  = parameters.L[k]
#         zₖ = simulategaussianprocess(L, Int(mvec[k]))   # n × m_k

#         if gridded
#             H, W = grid_shape
#             size(zₖ, 1) == H * W || error("Grid reshape failed: size(z,1) != H*W.")
#             Float32.(reshape(zₖ, H, W, 1, size(zₖ, 2)))
#         else
#             Float32.(Array(zₖ))   # n × m
#         end
#     end
# end





# =========================================================
# simulate
# =========================================================
#
# simulate_grided:
#   gridded CNN 路径，返回 H × W × 1 × m
#
# simulate_irregular:
#   irregular GNN 路径，返回 GNNGraph
#   每个 graph 已经带有：
#       ndata.S : 2 × n
#       ndata.Z : 1 × m × n
#       edata.e : 7 × E
# =========================================================

# ---------------------------------------------------------
# 1) gridded simulation
# ---------------------------------------------------------

function simulate_grided(
    parameters::Parameters,
    m::Integer = 100;
    grid_shape::Tuple{Int,Int} = (16, 16),
)
    K = length(parameters.L)
    mvec = fill(Int(m), K)
    return _simulate_grided_vec(parameters, mvec; grid_shape = grid_shape)
end

function simulate_grided(
    parameters::Parameters,
    mvec::AbstractVector{<:Integer};
    grid_shape::Tuple{Int,Int} = (16, 16),
)
    K = length(parameters.L)
    length(mvec) == K || throw(ArgumentError("length(mvec) must be $K"))
    return _simulate_grided_vec(parameters, mvec; grid_shape = grid_shape)
end

function _simulate_grided_vec(
    parameters::Parameters,
    mvec::AbstractVector{<:Integer};
    grid_shape::Tuple{Int,Int} = (16, 16),
)
    K = length(parameters.L)
    H, W = grid_shape

    return Folds.map(1:K) do k
        L = parameters.L[k]
        zₖ = L * randn(size(L, 1), mvec[k])
        # zₖ = simulategaussianprocess(L, Int(mvec[k]))   # n × m_k

        size(zₖ, 1) == H * W || error(
            "Grid reshape failed for k=$k: size(z,1)=$(size(zₖ,1)) but H*W=$(H*W)."
        )

        Float32.(reshape(zₖ, H, W, 1, size(zₖ, 2)))
    end
end


# ---------------------------------------------------------
# 2) irregular simulation + graph construction
# ---------------------------------------------------------
# zₖ = L * randn(size(L, 1), mvec[k])
# zₖ = simulategaussian(L, Int(mvec[k]))   # n_k × m_k

function simulate_irregular(
    parameters::Parameters,
    m::Integer = 100,
    process::Symbol = :GP,
)
    K = length(parameters.L)
    mvec = fill(Int(m), K)
    return _simulate_irregular_vec(parameters, mvec, process)
end

function simulate_irregular(
    parameters::Parameters,
    mvec::AbstractVector{<:Integer},
    process::Symbol = :GP,
)
    K = length(parameters.L)
    length(mvec) == K || throw(ArgumentError("length(mvec) must be $K"))
    return _simulate_irregular_vec(parameters, mvec, process)
end

function _simulate_irregular_vec(
    parameters::Parameters,
    mvec::AbstractVector{<:Integer},
    process::Symbol = :GP,
)
    K = length(parameters.L)

    return Folds.map(1:K) do k
        L  = parameters.L[k]
        gₖ = get_graph(parameters, k)

        if process in (:GP, :Gaussian)
            zₖ = L * randn(size(L, 1), mvec[k])

        elseif process in (:MSP, :BR_MSP)
            # marginal variance of Y when Y = L * ε:
            # diag(L * L') = row-wise sum of squares of L
            σ² = vec(sum(abs2, Matrix(L); dims = 2))
            zₖ = simulatebrownresnick(L, Int(mvec[k]); σ² = σ²)

        else
            throw(ArgumentError("process must be :GP or :BR_MSP, got $process"))
        end

        nodegraph(gₖ, Float32.(Array(zₖ)))
    end
end

function simulate_irregular_wrapper(
    parameters::Parameters,
    m_candidates::AbstractVector{<:Integer},
    process::Symbol = :GP,
)
    K = length(parameters.L)
    mvec = rand(m_candidates, K)
    return simulate_irregular(parameters, mvec, process)
end


# # ---------------------------------------------------------
# # 3) optional backward-compatible wrapper
# # ---------------------------------------------------------
# # 这样旧代码 simulate(...; gridded=true/false) 还可以跑。
# # 但建议在 irregular_data.jl 里显式用 simulate_irregular。

# function simulate(
#     parameters::Parameters,
#     m::Integer = 100;
#     gridded::Bool = true,
#     grid_shape::Tuple{Int,Int} = (16, 16),
# )
#     if gridded
#         return simulate_grided(parameters, m; grid_shape = grid_shape)
#     else
#         return simulate_irregular(parameters, m)
#     end
# end

# function simulate(
#     parameters::Parameters,
#     mvec::AbstractVector{<:Integer};
#     gridded::Bool = true,
#     grid_shape::Tuple{Int,Int} = (16, 16),
# )
#     if gridded
#         return simulate_grided(parameters, mvec; grid_shape = grid_shape)
#     else
#         return simulate_irregular(parameters, mvec)
#     end
# end