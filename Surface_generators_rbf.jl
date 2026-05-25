# include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_types.jl")
# include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/prior_experiment3.jl")

using Random
using Distributions
using StaticArrays

# ---------------------------------------------------------
# 假设你已经先 include 了 prior_experiment3.jl
# 这里只依赖其中的 sample_parameters_repulsive(...)
# ---------------------------------------------------------

# -------------------------
# RBF prior config
# -------------------------
const Π_base = (
    c     = Uniform(0.0, 1.0),
    beta  = Uniform(0.5, 8.0),
    gamma = Uniform(-0.9, 0.9),
    a0    = Uniform(-1.0, 1.0),
)

function make_priors_for_map(N_rbf::Int, Πb = Π_base)
    bound = 1.0
    ρk = Uniform(-2bound, 2bound)
    δk = Uniform(-2bound, 2bound)

    return vcat(
        fill(Πb.c,     N_rbf),   # c1
        fill(Πb.c,     N_rbf),   # c2
        fill(Πb.beta,  N_rbf),   # beta1
        fill(Πb.beta,  N_rbf),   # beta2
        fill(Πb.gamma, N_rbf),   # gamma
        [Πb.a0],                 # rho1_0
        fill(ρk,      N_rbf),    # rho1_k
        [Πb.a0],                 # rho2_0
        fill(ρk,      N_rbf),    # rho2_k
        [Πb.a0],                 # delta_0
        fill(δk,      N_rbf),    # delta_k
    )
end

function make_parameter_names_rbf(N_rbf::Integer)
    vcat(
        ["c1_$j"     for j in 1:N_rbf],
        ["c2_$j"     for j in 1:N_rbf],
        ["beta1_$j"  for j in 1:N_rbf],
        ["beta2_$j"  for j in 1:N_rbf],
        ["gamma_$j"  for j in 1:N_rbf],
        ["rho1_0"],
        ["rho1_$j"   for j in 1:N_rbf],
        ["rho2_0"],
        ["rho2_$j"   for j in 1:N_rbf],
        ["delta_0"],
        ["delta_$j"  for j in 1:N_rbf],
    )
end

# ---------------------------------------------------------
# theta layout:
# (c1_1..c1_K, c2_1..c2_K,
#  beta1_1..beta1_K, beta2_1..beta2_K,
#  gamma_1..gamma_K,
#  rho1_0, rho1_1..rho1_K,
#  rho2_0, rho2_1..rho2_K,
#  delta_0, delta_1..delta_K)
# ---------------------------------------------------------
function unpack_rbf_theta(θ::AbstractVector, N_rbf::Integer)
    K = N_rbf
    @assert length(θ) == 8K + 3 "length(θ) must be 8*N_rbf + 3"

    c1 = @view θ[1:K]
    c2 = @view θ[K+1:2K]

    β1 = @view θ[2K+1:3K]
    β2 = @view θ[3K+1:4K]
    γ  = @view θ[4K+1:5K]

    ρ1_0 = θ[5K+1]
    ρ1_k = @view θ[5K+2:6K+1]

    ρ2_0 = θ[6K+2]
    ρ2_k = @view θ[6K+3:7K+2]

    δ0   = θ[7K+3]
    δk   = @view θ[7K+4:8K+3]

    centers = hcat(c1, c2)
    return centers, β1, β2, γ, ρ1_0, ρ1_k, ρ2_0, ρ2_k, δ0, δk
end

@inline function B_from_params(β1::Real, β2::Real, γ::Real)
    s = sqrt(β1 * β2)
    return @SMatrix [β1  γ*s;
                     γ*s β2]
end

function rbf_combination(
    S::AbstractMatrix,
    centers::AbstractMatrix,
    β1::AbstractVector,
    β2::AbstractVector,
    γ::AbstractVector,
    a0::Real,
    ak::AbstractVector,
)
    n = size(S, 1)
    K = size(centers, 1)
    @assert length(β1) == K == length(β2) == length(γ) == length(ak)

    η = fill(float(a0), n)

    @inbounds for k in 1:K
        B = B_from_params(β1[k], β2[k], γ[k])
        cx = centers[k, 1]
        cy = centers[k, 2]
        b11 = B[1, 1]
        b12 = B[1, 2]
        b22 = B[2, 2]

        for i in 1:n
            dx = S[i, 1] - cx
            dy = S[i, 2] - cy
            q  = b11 * dx * dx + 2 * b12 * dx * dy + b22 * dy * dy
            η[i] += ak[k] * exp(-q)
        end
    end

    return η
end

@inline function logis(x::Real)
    if x >= 0
        return inv(1 + exp(-x))
    else
        ex = exp(x)
        return ex / (1 + ex)
    end
end


# function surface_bundle_from_theta(
#     S::AbstractMatrix,
#     θ::AbstractVector,
#     N_rbf::Integer;
#     vary_rho1::Bool = true,
#     vary_rho2::Bool = true,
#     vary_delta::Bool = true,
#     rho1_const::Real = 0.5,
#     rho2_const::Real = 0.5,
#     delta_const::Real = 0.0,
# )
#     centers, β1, β2, γ, ρ1_0, ρ1_k, ρ2_0, ρ2_k, δ0, δk =
#         unpack_rbf_theta(θ, N_rbf)

#     n = size(S, 1)
#     sigma = ones(Float32, n)

#     rho1 = if vary_rho1
#         η1 = rbf_combination(S, centers, β1, β2, γ, ρ1_0, ρ1_k)
#         Float32.(clamp.(logis.(η1), 1f-3, 1f0 - 1f-3))
#     else
#         fill(clamp(Float32(rho1_const), 1f-3, 1f0 - 1f-3), n)
#     end

#     rho2 = if vary_rho2
#         η2 = rbf_combination(S, centers, β1, β2, γ, ρ2_0, ρ2_k)
#         Float32.(clamp.(logis.(η2), 1f-3, 1f0 - 1f-3))
#     else
#         fill(clamp(Float32(rho2_const), 1f-3, 1f0 - 1f-3), n)
#     end

#     delta = if vary_delta
#         ηδ = rbf_combination(S, centers, β1, β2, γ, δ0, δk)
#         Float32.(clamp.(-1f0 .+ 2f0 .* logis.(ηδ), -0.999f0, 0.999f0))
#     else
#         fill(clamp(Float32(delta_const), -0.999f0, 0.999f0), n)
#     end

#     return SurfaceBundle(sigma, rho1, rho2, delta)
# end
function surface_bundle_from_theta(
    S::AbstractMatrix,
    θ::AbstractVector,
    N_rbf::Integer;
    model_type::Symbol = :nonstationary,

    vary_sigma::Bool = false,
    vary_rho1::Bool = true,
    vary_rho2::Bool = true,
    vary_delta::Bool = true,

    sigma_const::Real = 1.0,
    rho1_const::Real = 0.5,
    rho2_const::Real = 0.5,
    delta_const::Real = 0.0,

    sigma_range::Tuple{<:Real,<:Real} = (0.5, 2.0),
    sigma_0::Real = 0.0,
    sigma_k = nothing,

    nu::Real = 1.5,
)
    centers, β1, β2, γ, ρ1_0, ρ1_k, ρ2_0, ρ2_k, δ0, δk =
        unpack_rbf_theta(θ, N_rbf)

    n = size(S, 1)

    σlo = Float32(sigma_range[1])
    σhi = Float32(sigma_range[2])
    σlo > 0f0 || error("sigma_range lower bound must be positive.")
    σhi > σlo || error("sigma_range must satisfy lower < upper.")

    sigma = if vary_sigma
        sigma_k === nothing && error("sigma_k must be provided when vary_sigma=true.")
        length(sigma_k) == N_rbf || error("length(sigma_k) must be N_rbf.")

        ησ = rbf_combination(S, centers, β1, β2, γ, sigma_0, sigma_k)
        Float32.(σlo .+ (σhi - σlo) .* logis.(ησ))
    else
        fill(clamp(Float32(sigma_const), σlo, σhi), n)
    end

    rho1 = if vary_rho1
        η1 = rbf_combination(S, centers, β1, β2, γ, ρ1_0, ρ1_k)
        Float32.(clamp.(logis.(η1), 1f-3, 1f0 - 1f-3))
    else
        fill(clamp(Float32(rho1_const), 1f-3, 1f0 - 1f-3), n)
    end

    if model_type == :local_isotropy
        rho2  = copy(rho1)
        delta = fill(0f0, n)
    else
        rho2 = if vary_rho2
            η2 = rbf_combination(S, centers, β1, β2, γ, ρ2_0, ρ2_k)
            Float32.(clamp.(logis.(η2), 1f-3, 1f0 - 1f-3))
        else
            fill(clamp(Float32(rho2_const), 1f-3, 1f0 - 1f-3), n)
        end

        delta = if vary_delta
            ηδ = rbf_combination(S, centers, β1, β2, γ, δ0, δk)
            Float32.(clamp.(-1f0 .+ 2f0 .* logis.(ηδ), -0.999f0, 0.999f0))
        else
            fill(clamp(Float32(delta_const), -0.999f0, 0.999f0), n)
        end
    end

    return SurfaceBundle(sigma, rho1, rho2, delta; nu = nu)
end

# ---------------------------------------------------------
# 对外暴露的接口：
# 给我一个 S，我返回一个 SurfaceBundle
# 外部不需要知道 theta / N_rbf
# ---------------------------------------------------------
# function make_rbf_surface_sampler(;
#     N_rbf::Int,
#     Πb = Π_base,
#     τ::Float64 = 0.2,
#     ν::Int = 2,
#     nsteps::Int = 30,
#     σ_rw::Float64 = 0.05,
#     vary_rho1::Bool = true,
#     vary_rho2::Bool = true,
#     vary_delta::Bool = true,
#     rho1_const::Real = 0.2,
#     rho2_const::Real = 0.2,
#     delta_const::Real = 0.0,
# )
#     Π = make_priors_for_map(N_rbf, Πb)

#     function sampler(S::AbstractMatrix)
#         θ = sample_parameters_repulsive(
#             N_rbf, Π;
#             τ = τ,
#             ν = ν,
#             nsteps = nsteps,
#             σ_rw = σ_rw,
#             canonicalize = true,
#             rng = Random.default_rng(),
#         )

#         return surface_bundle_from_theta(
#             S, θ, N_rbf;
#             vary_rho1 = vary_rho1,
#             vary_rho2 = vary_rho2,
#             vary_delta = vary_delta,
#             rho1_const = rho1_const,
#             rho2_const = rho2_const,
#             delta_const = delta_const,
#         )
#     end

#     return sampler
# end
function make_rbf_surface_sampler(;
    N_rbf::Int,
    Πb = Π_base,
    τ::Float64 = 0.2,
    ν::Int = 2,                 # repulsive-center prior exponent, not Matérn ν
    nsteps::Int = 30,
    σ_rw::Float64 = 0.05,

    model_type::Symbol = :nonstationary,

    vary_sigma::Bool = false,
    vary_rho1::Bool = true,
    vary_rho2::Bool = true,
    vary_delta::Bool = true,

    sigma_const::Real = 1.0,
    rho1_const::Real = 0.2,
    rho2_const::Real = 0.2,
    delta_const::Real = 0.0,

    sigma_range::Tuple{<:Real,<:Real} = (0.5, 2.0),
    sigma_a0_dist = Uniform(-1.0, 1.0),
    sigma_k_dist  = Uniform(-1.5, 1.5),

    sample_nu::Bool = false,
    nu_const::Real = 1.5,
    nu_range::Tuple{<:Real,<:Real} = (0.5, 2.5),
)
    Π = make_priors_for_map(N_rbf, Πb)

    σlo = Float32(sigma_range[1])
    σhi = Float32(sigma_range[2])
    σlo > 0f0 || error("sigma_range lower bound must be positive.")
    σhi > σlo || error("sigma_range must satisfy lower < upper.")

    νlo = Float32(nu_range[1])
    νhi = Float32(nu_range[2])
    νlo > 0f0 || error("nu_range lower bound must be positive.")
    νhi > νlo || error("nu_range must satisfy lower < upper.")

    function sampler(S::AbstractMatrix; model_type::Symbol = model_type)
        rng = Random.default_rng()

        θ = sample_parameters_repulsive(
            N_rbf, Π;
            τ = τ,
            ν = ν,
            nsteps = nsteps,
            σ_rw = σ_rw,
            canonicalize = true,
            rng = rng,
        )

        sigma_0 = Float64(rand(rng, sigma_a0_dist))
        sigma_k = Float64.(rand(rng, sigma_k_dist, N_rbf))

        nu = sample_nu ?
            Float32(rand(rng, Uniform(Float64(νlo), Float64(νhi)))) :
            clamp(Float32(nu_const), νlo, νhi)

        return surface_bundle_from_theta(
            S, θ, N_rbf;
            model_type = model_type,
            vary_sigma = vary_sigma,
            vary_rho1 = vary_rho1,
            vary_rho2 = vary_rho2,
            vary_delta = vary_delta,
            sigma_const = sigma_const,
            rho1_const = rho1_const,
            rho2_const = rho2_const,
            delta_const = delta_const,
            sigma_range = sigma_range,
            sigma_0 = sigma_0,
            sigma_k = sigma_k,
            nu = nu,
        )
    end

    return sampler
end