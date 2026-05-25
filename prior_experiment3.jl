using Random
using Distributions
using LinearAlgebra

"""
    sample_centers_min_dist(K, d_min; xdist=Uniform(0,1), ydist=Uniform(0,1),
                            max_tries=1_000_000, rng=Random.default_rng())

Sample K centers in R^2 with pairwise Euclidean distance >= d_min.
Each coordinate is sampled from xdist and ydist (defaults to Uniform(0,1)).
Returns a K×2 Matrix{Float64}.
"""
function sample_centers_min_dist(K::Integer, d_min::Real;
    xdist::Distribution = Uniform(0.0, 1.0),
    ydist::Distribution = Uniform(0.0, 1.0),
    max_tries::Int = 1_000_000,
    rng::AbstractRNG = Random.default_rng()
)
    (K >= 1) || throw(ArgumentError("K must be >= 1"))
    (d_min >= 0) || throw(ArgumentError("d_min must be >= 0"))

    d2 = float(d_min)^2
    centers = Matrix{Float64}(undef, K, 2)

    k = 0
    tries = 0
    while k < K
        tries += 1
        if tries > max_tries
            error("Failed after $max_tries tries. Try lowering d_min or increasing max_tries.")
        end

        candx = Float64(rand(rng, xdist))
        candy = Float64(rand(rng, ydist))

        if k == 0
            k = 1
            centers[k, 1] = candx
            centers[k, 2] = candy
        else
            ok = true
            @inbounds for i in 1:k
                dx = centers[i, 1] - candx
                dy = centers[i, 2] - candy
                if dx*dx + dy*dy < d2
                    ok = false
                    break
                end
            end
            if ok
                k += 1
                centers[k, 1] = candx
                centers[k, 2] = candy
            end
        end
    end

    return centers
end



# ------------------------------------------------------------
# ex3: θ = [c1; c2; β1; β2; γ; ρ1_0; ρ1_k; ρ2_0; ρ2_k; δ0; δk]
# length = 8N_rbf + 3
# Π 的布局需与 make_priors_for_map(N_rbf, Πb) 一致
# ------------------------------------------------------------
function sample_parameters_min_dist_ex3(N_rbf::Integer, Π::AbstractVector;
    d_min::Real = 0.0,
    canonicalize::Bool = false,
    max_tries::Int = 1_000_000,
    rng::AbstractRNG = Random.default_rng()
)
    length(Π) == 8N_rbf + 3 || throw(ArgumentError("length(Π) must be 8N_rbf + 3"))

    K = N_rbf

    xdist   = Π[1]
    ydist   = Π[K + 1]
    b1dist  = Π[2K + 1]
    b2dist  = Π[3K + 1]
    gdist   = Π[4K + 1]

    ρ1_0dist = Π[5K + 1]
    ρ1_kdist = Π[5K + 2]

    ρ2_0dist = Π[6K + 2]
    ρ2_kdist = Π[6K + 3]

    δ0dist   = Π[7K + 3]
    δkdist   = Π[7K + 4]

    if d_min > 0
        centers = sample_centers_min_dist(K, d_min;
            xdist = xdist, ydist = ydist,
            max_tries = max_tries, rng = rng
        )
        c1 = centers[:, 1]
        c2 = centers[:, 2]
    else
        c1 = Float64.(rand(rng, xdist, K))
        c2 = Float64.(rand(rng, ydist, K))
        centers = hcat(c1, c2)
    end

    β1 = Float64.(rand(rng, b1dist, K))
    β2 = Float64.(rand(rng, b2dist, K))
    γ  = Float64.(rand(rng, gdist,  K))

    ρ1_0 = Float64(rand(rng, ρ1_0dist))
    ρ1_k = Float64.(rand(rng, ρ1_kdist, K))

    ρ2_0 = Float64(rand(rng, ρ2_0dist))
    ρ2_k = Float64.(rand(rng, ρ2_kdist, K))

    δ0   = Float64(rand(rng, δ0dist))
    δk   = Float64.(rand(rng, δkdist, K))

    if canonicalize && K > 1
        p = sortperm(1:K; by = i -> (centers[i, 1], centers[i, 2]))
        centers = centers[p, :]
        c1 = c1[p]
        c2 = c2[p]
        β1 = β1[p]
        β2 = β2[p]
        γ  = γ[p]
        ρ1_k = ρ1_k[p]
        ρ2_k = ρ2_k[p]
        δk   = δk[p]
    end

    return vcat(c1, c2, β1, β2, γ, ρ1_0, ρ1_k, ρ2_0, ρ2_k, δ0, δk)
end



#################################################################################
#################################################################################
#################################################################################


# ----------------------------
# Helpers: reflect proposal back to [0,1]
# ----------------------------
@inline function reflect01(x::Float64)
    while x < 0.0 || x > 1.0
        x = x < 0.0 ? -x : 2.0 - x
    end
    return x
end

# log g(d) where g(d)=exp(-τ d^{-ν})
@inline function log_g_repulsion(d::Float64, τ::Float64, ν::Int; d_eps::Float64 = 1e-12)
    if d <= d_eps
        return -Inf  # numerically treat "almost collision" as impossible
    end
    return -τ * d^(-ν)
end

# base log-density g0(x,y)=xdist(x)*ydist(y)
@inline function log_base_center(x::Float64, y::Float64,
                                xdist::Distribution, ydist::Distribution)
    return logpdf(xdist, x) + logpdf(ydist, y)
end

# ----------------------------
# Repulsive-prior MCMC sampler for centers
# ----------------------------
function sample_centers_repulsive_mcmc(K::Integer;
    xdist::Distribution = Uniform(0.0, 1.0),
    ydist::Distribution = Uniform(0.0, 1.0),
    τ::Real = 5.0,
    ν::Integer = 2,
    nsteps::Int = 40,          # number of full sweeps
    σ_rw::Real = 0.05,         # RW proposal scale
    canonicalize::Bool = true, # sort centers to fix labels
    rng::AbstractRNG = Random.default_rng()
)
    (K >= 1) || throw(ArgumentError("K must be >= 1"))
    τ = Float64(τ)
    ν = Int(ν)
    σ_rw = Float64(σ_rw)

    # init from base
    C = Matrix{Float64}(undef, K, 2)
    @inbounds for i in 1:K
        C[i,1] = Float64(rand(rng, xdist))
        C[i,2] = Float64(rand(rng, ydist))
    end

    @inbounds for step in 1:nsteps          # 外层：做 nsteps 次“整轮扫描”(sweep)
        for i in 1:K                        # 内层：逐个更新每一个 center c_i（Gibbs 外壳）
            x_old, y_old = C[i,1], C[i,2]   # 当前状态：第 i 个 center 的旧位置 c_i = (x_old, y_old)

            # ------------------------------------------------------------
            # 1) 提案：Random-walk proposal（对称提案）
            #    c_i' = c_i + σ_rw * ε,  ε ~ N(0, I_2)
            #    并通过 reflect01 把结果映射回 [0,1]，保证仍在 unit square 内
            # ------------------------------------------------------------
            x_new = reflect01(x_old + σ_rw * randn(rng))
            y_new = reflect01(y_old + σ_rw * randn(rng))
            # 得到提案点 c_i' = (x_new, y_new)

            # ------------------------------------------------------------
            # 2) base prior（独立先验）部分的 log-density
            #    g0(c_i) = p_x(x_i) * p_y(y_i)
            #    例如 Uniform(0,1) 时 logpdf 恒为常数（在区间内）
            # ------------------------------------------------------------
            logbase_old = log_base_center(x_old, y_old, xdist, ydist)  # log g0(c_i)
            logbase_new = log_base_center(x_new, y_new, xdist, ydist)  # log g0(c_i')

            # 若提案点落在支持集外（理论上 reflect 后不会，但做防御性检查）
            if !isfinite(logbase_new)
                continue  # 直接拒绝这次提案，进入下一个 i
            end

            # ------------------------------------------------------------
            # 3) repulsion 部分的变化量：只需要计算“涉及 i 的 pair”
            #
            #    目标中 repulsion 是  ∏_{j<s} g(||c_s - c_j||)
            #    当只更新 c_i 时，不含 i 的那些 pair 在 MH 比值里会完全抵消：
            #
            #    π(C')/π(C) = [g0(c_i')/g0(c_i)] *
            #                ∏_{j≠i} g(||c_i' - c_j||) / g(||c_i - c_j||)
            #
            #    所以只需要计算 j≠i 的 K-1 个距离变化，而不是全对 K(K-1)/2 个。
            # ------------------------------------------------------------
            Δlogrep = 0.0     # 记录 repulsion 的 log 比值增量：Σ (log g_new - log g_old)
            ok = true         # 标记提案是否可接受（例如避免数值上“碰撞”导致 -Inf）

            for j in 1:K
                j == i && continue  # 跳过自己与自己的距离（为 0 没意义）

                # 旧距离 d_old = ||c_i - c_j||，新距离 d_new = ||c_i' - c_j||
                d_old = hypot(x_old - C[j,1], y_old - C[j,2])
                d_new = hypot(x_new - C[j,1], y_new - C[j,2])

                # log g(d)；对于 g(d)=exp(-τ d^{-ν})，log g(d) = -τ d^{-ν}
                lg_old = log_g_repulsion(d_old, τ, ν)
                lg_new = log_g_repulsion(d_new, τ, ν)

                # 若提案导致某个距离过小，则 log_g_repulsion 返回 -Inf
                # 这等价于：repulsive prior 几乎不允许 center“撞在一起”
                if !isfinite(lg_new)
                    ok = false
                    break            # 无需继续算其它 j，这个提案必拒绝
                end

                # 累积 log 比值增量：
                # Δlogrep = Σ_{j≠i} [log g(d_new) - log g(d_old)]
                Δlogrep += (lg_new - lg_old)
            end

            ok || continue  # 如果提案“无效”（太近），直接拒绝，进入下一个 i

            # ------------------------------------------------------------
            # 4) 计算 MH 接受率（对数形式）
            #
            #    logα = log π(C') - log π(C)
            #         = [log g0(c_i') - log g0(c_i)] + Δlogrep
            #
            #    注：这里默认 proposal 是对称的随机游走（q(a|b)=q(b|a)），
            #        因此 proposal 比值项约为 0（被省略）。
            # ------------------------------------------------------------
            logα = (logbase_new - logbase_old) + Δlogrep

            # ------------------------------------------------------------
            # 5) 接受/拒绝
            #
            #    α = min(1, exp(logα))
            #    用 log(u) < min(0, logα) 的形式做数值稳定的判断
            # ------------------------------------------------------------
            if log(rand(rng)) < min(0.0, logα)
                # 接受：更新第 i 个 center
                C[i,1] = x_new
                C[i,2] = y_new
            end
            # 否则拒绝：保持 C[i,:] 不变，进入下一个 i
        end
    end


    # deterministic labeling to reduce label switching
    if canonicalize && K > 1
        p = sortperm(1:K; by = i -> (C[i,1], C[i,2]))
        C = C[p, :]
    end

    return C
end

# ------------------------------------------------------------
# repulsive version: 只对 centers 做 repulsion
# 其余参数独立采样；与 prior_experiment2 的思路一致
# ------------------------------------------------------------
function sample_parameters_repulsive(N_rbf::Integer, Π::AbstractVector;
    τ::Real = 0.2,
    ν::Integer = 2,
    nsteps::Int = 40,
    σ_rw::Real = 0.05,
    tol_x::Real = 0.05,
    canonicalize::Bool = true,
    rng::AbstractRNG = Random.default_rng()
)
    length(Π) == 8N_rbf + 3 || throw(ArgumentError("length(Π) must be 8N_rbf + 3"))

    K = N_rbf

    xdist   = Π[1]
    ydist   = Π[K + 1]
    b1dist  = Π[2K + 1]
    b2dist  = Π[3K + 1]
    gdist   = Π[4K + 1]

    ρ1_0dist = Π[5K + 1]
    ρ1_kdist = Π[5K + 2]

    ρ2_0dist = Π[6K + 2]
    ρ2_kdist = Π[6K + 3]

    δ0dist   = Π[7K + 3]
    δkdist   = Π[7K + 4]

    centers = sample_centers_repulsive_mcmc(K;
        xdist = xdist,
        ydist = ydist,
        τ = τ,
        ν = ν,
        nsteps = nsteps,
        σ_rw = σ_rw,
        canonicalize = false,
        rng = rng
    )

    β1 = Float64.(rand(rng, b1dist, K))
    β2 = Float64.(rand(rng, b2dist, K))
    γ  = Float64.(rand(rng, gdist,  K))

    ρ1_0 = Float64(rand(rng, ρ1_0dist))
    ρ1_k = Float64.(rand(rng, ρ1_kdist, K))

    ρ2_0 = Float64(rand(rng, ρ2_0dist))
    ρ2_k = Float64.(rand(rng, ρ2_kdist, K))

    δ0   = Float64(rand(rng, δ0dist))
    δk   = Float64.(rand(rng, δkdist, K))

    if canonicalize && K > 1
        p = sortperm(1:K; by = i -> (
            floor(Int, centers[i, 1] / tol_x),
            centers[i, 2],
            centers[i, 1]
        ))
        centers = centers[p, :]
        β1 = β1[p]
        β2 = β2[p]
        γ  = γ[p]
        ρ1_k = ρ1_k[p]
        ρ2_k = ρ2_k[p]
        δk   = δk[p]
    end

    c1 = centers[:, 1]
    c2 = centers[:, 2]

    return vcat(c1, c2, β1, β2, γ, ρ1_0, ρ1_k, ρ2_0, ρ2_k, δ0, δk)
end