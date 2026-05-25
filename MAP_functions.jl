# =========================================================
# ---- MAP for local-isotropy nodewise rho ----
#
# rho = [rho_1,...,rho_n]
#
# local isotropy:
#   rho1_i = rho_i
#   rho2_i = rho_i
#   delta_i = 0
#
# support prior:
#   rho_i ~ Uniform(0, 1)
#
# implementation:
#   optimize unconstrained eta ∈ R^n
#   rho_i = scaledlogistic(eta_i, Uniform(0, 1))
#
# optional eta-ridge:
#   loss = -loglik + λ_ridge * ||eta - eta0||^2
#
# optional eta-smoothness:
#   loss = -loglik + λ_smoo * weighted_mean_{(i,j) in E} (eta_i - eta_j)^2
#
# smoo_weight options:
#   :inv_distance  -> w_ij = 1 / distance(s_i, s_j), default
#   :unit      -> w_ij = 1
#
# eta0 is estimated from a stationary model if use_stationary_init=true.
# =========================================================

# 0) Prior construction
function make_nodewise_prior(S::AbstractMatrix)
    n = size(S, 1)
    return fill(Uniform(0.0, 1.0), n)
end

function make_nodewise_prior(S::AbstractVector{<:AbstractMatrix})
    return [make_nodewise_prior(S[k]) for k in eachindex(S)]
end

# 1) scaled logistic transform
#    unconstrained eta ∈ R -> constrained rho ∈ prior support
@inline function scaledlogistic(x::Real, Π::Uniform)
    a = Float64(minimum(Π))
    b = Float64(maximum(Π))

    p = if x >= 0
        inv(1 + exp(-x))
    else
        ex = exp(x)
        ex / (1 + ex)
    end

    return a + (b - a) * p
end

# 2) Extract n × m data matrix from GNNGraph
function _extract_Z_from_graph(g::GNNGraph)
    Z = Array(g.ndata.Z)

    ndims(Z) == 3 || error("Expected g.ndata.Z to be 3D, got size $(size(Z)).")
    size(Z, 1) == 1 || error("Expected g.ndata.Z first dimension to be 1, got size $(size(Z)).")

    # 1 × m × n -> n × m
    return Float64.(dropdims(permutedims(Z, (3, 2, 1)), dims = 3))
end

# 3) Utilities for local-isotropy rho
function _split_local_iso_theta(θ::AbstractVector, S::AbstractMatrix)
    n = size(S, 1)
    length(θ) == n || error("length(θ) must be n = $(n), got $(length(θ)).")

    rho = Float64.(θ[1:n])
    rho = clamp.(rho, 1e-4, 1 - 1e-4)

    sigma = ones(Float64, n)
    rho1  = rho
    rho2  = copy(rho)
    delta = zeros(Float64, n)

    return sigma, rho1, rho2, delta
end

function _theta_matrix_from_vector(θ::AbstractVector, S::AbstractMatrix)
    n = size(S, 1)
    length(θ) == n || error("length(θ) must be n = $(n), got $(length(θ)).")

    # return 1 × n
    return Float32.(reshape(θ[1:n], 1, :))
end

function _prior_for_k(Π, k::Integer)
    if Π isa AbstractVector && !isempty(Π) && Π[1] isa AbstractVector
        return Π[k]
    else
        return Π
    end
end

# 4) Graph smoothness penalty on eta
#
# smoo_weight = :unit:
#     (1 / |E|) * sum_{(i,j) in E} (eta_i - eta_j)^2
#
# smoo_weight = :inv_distance:
#     weighted average with w_ij = 1 / distance(s_i, s_j)
#     sum w_ij * (eta_i - eta_j)^2 / sum w_ij
function smoothness_eta(
    eta,
    S::AbstractMatrix,
    g::GNNGraph;
    smoo_weight::Symbol = :inv_distance,
    eps_dist::Real = 1e-8,
)
    s, t = edge_index(g)
    diff2 = abs2.(eta[s] .- eta[t])

    if smoo_weight == :unit
        return mean(diff2)
    elseif smoo_weight == :inv_distance || smoo_weight == :distance
        dist = sqrt.(vec(sum(abs2, S[s, :] .- S[t, :]; dims = 2)))
        w = 1.0 ./ max.(dist, eps_dist)
        return sum(w .* diff2) / sum(w)
    else
        error("smoo_weight must be :inv_distance or :unit, got $(smoo_weight).")
    end
end

# 5) MAP
function MAP(
    Z::AbstractVector,
    ξ;
    λ_ridge::Real = 0.0,
    λ_smoo::Real = 0.0,
    smoo_weight::Symbol = :inv_distance,
    use_stationary_init::Bool = true,
)
    S = ξ.S
    Π = ξ.Π
    g = ξ.g

    θ = Folds.map(eachindex(Z)) do k
        Sₖ = S isa AbstractVector ? S[k] : S
        Πₖ = _prior_for_k(Π, k)
        gₖ = g isa AbstractVector ? g[k] : g

        MAP(
            Z[k],
            Sₖ,
            Πₖ,
            gₖ;
            λ_ridge = λ_ridge,
            λ_smoo = λ_smoo,
            smoo_weight = smoo_weight,
            use_stationary_init = use_stationary_init,
        )
    end

    # Return Vector{Matrix}; each θ[k] is 1 × n_k
    return θ
end

function MAP(
    Z::GNNGraph,
    S,
    Π,
    g;
    λ_ridge::Real = 0.0,
    λ_smoo::Real = 0.0,
    smoo_weight::Symbol = :inv_distance,
    use_stationary_init::Bool = true,
)
    Z = _extract_Z_from_graph(Z)

    return MAP(
        Z,
        S,
        Π,
        g;
        λ_ridge = λ_ridge,
        λ_smoo = λ_smoo,
        smoo_weight = smoo_weight,
        use_stationary_init = use_stationary_init,
    )
end

function MAP(
    Z::A,
    S,
    Π,
    g;
    λ_ridge::Real = 0.0,
    λ_smoo::Real = 0.0,
    smoo_weight::Symbol = :inv_distance,
    use_stationary_init::Bool = true,
) where {T, N, A <: AbstractArray{T, N}}

    Z = flatten(Z)
    Z = Float64.(Z)

    Π = [Π...]

    n = size(S, 1)
    length(Π) == n || error("length(Π) must be n = $(n), got $(length(Π)).")

    # -------------------------------------------------
    # eta0:
    #   1) initial value for nonstationary eta
    #   2) center of eta-ridge penalty
    # -------------------------------------------------
    eta0 = if use_stationary_init
        eta_stat = fit_stationary_eta(Z, S, Π)

        println("Stationary MAP init:")
        println("  eta_stat = ", eta_stat)
        println("  rho_stat = ", scaledlogistic(eta_stat, Π[1]))

        fill(eta_stat, n)
    else
        zeros(Float64, n)
    end

    loss(eta) = nll(
        eta,
        Z,
        S,
        Π,
        g;
        eta0 = eta0,
        λ_ridge = λ_ridge,
        λ_smoo = λ_smoo,
        smoo_weight = smoo_weight,
    )

    eta_hat = optimize(loss, eta0, LBFGS()) |> Optim.minimizer

    rho_hat = scaledlogistic.(eta_hat, Π)

    return _theta_matrix_from_vector(rho_hat, S)
end

# 6) Stationary model for eta initialization
function fit_stationary_eta(Z, S, Π; eta_init::Real = 0.0)
    Π = [Π...]

    n = size(S, 1)
    length(Π) == n || error("length(Π) must be n = $(n), got $(length(Π)).")

    loss_stat(eta_vec) = nll_stationary_eta(eta_vec, Z, S, Π)

    eta_hat = optimize(loss_stat, [Float64(eta_init)], LBFGS()) |> Optim.minimizer

    return only(eta_hat)
end

function nll_stationary_eta(eta_vec, Z, S, Π)
    eta = only(eta_vec)

    # one global eta -> one global rho repeated over all sites
    rho = scaledlogistic.(fill(eta, length(Π)), Π)

    val = -logmvnorm(rho, Z, S)

    return isfinite(val) ? val : Inf
end

# 7) Nonstationary objective with eta-ridge and eta-smoothness
function nll(
    eta,
    Z,
    S,
    Π,
    g;
    eta0 = nothing,
    λ_ridge::Real = 0.0,
    λ_smoo::Real = 0.0,
    smoo_weight::Symbol = :inv_distance,
)
    rho = scaledlogistic.(eta, Π)

    negloglik = -logmvnorm(rho, Z, S)
    isfinite(negloglik) || return Inf

    ridge = if λ_ridge > 0
        eta0 === nothing && error("eta0 must be supplied when λ_ridge > 0.")
        λ_ridge * sum(abs2, eta .- eta0)
    else
        0.0
    end

    smoo = if λ_smoo > 0
        λ_smoo * smoothness_eta(
            eta,
            S,
            g;
            smoo_weight = smoo_weight,
        )
    else
        0.0
    end

    val = negloglik + ridge + smoo

    return isfinite(val) ? val : Inf
end

function logmvnorm(rho, Z, S; kwargs...)
    σv, ρ1v, ρ2v, δv = _split_local_iso_theta(rho, S)

    Σ = covariancematrix(
        S,
        σv,
        ρ1v,
        ρ2v,
        δv,
        1.5;
        jitter = 1e-6,
    )

    gaussiandensity(Z, Σ; logdensity = true)
end
