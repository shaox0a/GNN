# =========================================================
# Data_functions.jl
# surface-only version
#
# require: include sim_experiment3.jl
#   - Parameters
#   - SurfaceBundle
#   - get_sites / get_graph / get_surface
#   - nodegraph(...)
# =========================================================

using GraphNeuralNetworks

# =========================================================
# 0) helpers
# =========================================================

function surface_from_sites(S::AbstractMatrix, v::AbstractVector)
    x = sort(unique(S[:, 1]))
    y = sort(unique(S[:, 2]))
    Z = fill(Float32(NaN), length(x), length(y))

    @inbounds for idx in eachindex(v)
        i = searchsortedfirst(x, S[idx, 1])
        j = searchsortedfirst(y, S[idx, 2])
        Z[i, j] = Float32(v[idx])
    end
    return Z
end

function coord_surfaces(S::AbstractMatrix)
    xsurf = surface_from_sites(S, vec(S[:, 1]))
    ysurf = surface_from_sites(S, vec(S[:, 2]))
    return xsurf, ysurf
end

logitf(x) = log(x / (1f0 - x))

# =========================================================
# 1) targets from SurfaceBundle
# =========================================================

# function extract_true_surfaces(S::AbstractMatrix, surf::SurfaceBundle)
#     ρ1surf = surface_from_sites(S, surf.rho1)
#     ρ2surf = surface_from_sites(S, surf.rho2)
#     δsurf  = surface_from_sites(S, surf.delta)
#     return ρ1surf, ρ2surf, δsurf
# end
function extract_true_surfaces(
    S::AbstractMatrix,
    surf::SurfaceBundle;
    return_rho1::Bool = true,
    return_rho2::Bool = true,
    return_delta::Bool = true,
)
    Zs = Matrix{Float32}[]

    return_rho1 && push!(Zs, surface_from_sites(S, surf.rho1))
    return_rho2 && push!(Zs, surface_from_sites(S, surf.rho2))
    return_delta && push!(Zs, surface_from_sites(S, surf.delta))

    isempty(Zs) && error("At least one of return_rho1/return_rho2/return_delta must be true.")
    return Zs
end

# function extract_true_nodewise(surf::SurfaceBundle)
#     n = length(surf.rho1)
#     Y = Array{Float32}(undef, 3, n)
#     Y[1, :] = surf.rho1
#     Y[2, :] = surf.rho2
#     Y[3, :] = surf.delta
#     return Y
# end
function extract_true_nodewise(
    surf::SurfaceBundle;
    return_rho1::Bool = true,
    return_rho2::Bool = true,
    return_delta::Bool = true,
)
    rows = Vector{Vector{Float32}}()

    return_rho1 && push!(rows, Float32.(surf.rho1))
    return_rho2 && push!(rows, Float32.(surf.rho2))
    return_delta && push!(rows, Float32.(surf.delta))

    isempty(rows) && error("At least one of return_rho1/return_rho2/return_delta must be true.")

    n = length(surf.rho1)
    q = length(rows)
    Y = Array{Float32}(undef, q, n)
    for j in 1:q
        Y[j, :] = rows[j]
    end
    return Y
end

# function extract_true_nodewise_latent(surf::SurfaceBundle)
#     ρ1v = clamp.(surf.rho1, 1f-4, 1f0 - 1f-4)
#     ρ2v = clamp.(surf.rho2, 1f-4, 1f0 - 1f-4)
#     δv  = clamp.(surf.delta, -0.999f0, 0.999f0)

#     η1 = logitf.(ρ1v)
#     η2 = logitf.(ρ2v)
#     ηδ = atanh.(δv)

#     n = length(ρ1v)
#     Y = Array{Float32}(undef, 3, n)
#     Y[1, :] = η1
#     Y[2, :] = η2
#     Y[3, :] = ηδ
#     return Y
# end
function extract_true_nodewise_latent(
    surf::SurfaceBundle;
    return_rho1::Bool = true,
    return_rho2::Bool = true,
    return_delta::Bool = true,
)
    ρ1v = clamp.(surf.rho1, 1f-4, 1f0 - 1f-4)
    ρ2v = clamp.(surf.rho2, 1f-4, 1f0 - 1f-4)
    δv  = clamp.(surf.delta, -0.999f0, 0.999f0)

    η1 = Float32.(logitf.(ρ1v))
    η2 = Float32.(logitf.(ρ2v))
    ηδ = Float32.(atanh.(δv))

    rows = Vector{Vector{Float32}}()

    return_rho1 && push!(rows, η1)
    return_rho2 && push!(rows, η2)
    return_delta && push!(rows, ηδ)

    isempty(rows) && error("At least one of return_rho1/return_rho2/return_delta must be true.")

    n = length(ρ1v)
    q = length(rows)
    Y = Array{Float32}(undef, q, n)
    for j in 1:q
        Y[j, :] = rows[j]
    end
    return Y
end


"""
If gridded=true:
    return H×W×3×1

If gridded=false and latent=false:
    return 3×n   (bounded target)

If gridded=false and latent=true:
    return 3×n   (latent target)
"""
function target_tensor(
    S::AbstractMatrix,
    surf::SurfaceBundle;
    gridded::Bool = true,
    latent::Bool = false,
    return_rho1::Bool = true,
    return_rho2::Bool = true,
    return_delta::Bool = true,
)
    if gridded
        Ys = extract_true_surfaces(
            S, surf;
            return_rho1 = return_rho1,
            return_rho2 = return_rho2,
            return_delta = return_delta,
        )

        q = length(Ys)
        H, W = size(Ys[1])
        Y = Array{Float32}(undef, H, W, q, 1)

        for j in 1:q
            Y[:, :, j, 1] = Ys[j]
        end
        return Y

    else
        if latent
            return extract_true_nodewise_latent(
                surf;
                return_rho1 = return_rho1,
                return_rho2 = return_rho2,
                return_delta = return_delta,
            )
        else
            return extract_true_nodewise(
                surf;
                return_rho1 = return_rho1,
                return_rho2 = return_rho2,
                return_delta = return_delta,
            )
        end
    end
end
# function target_tensor(
#     S::AbstractMatrix,
#     surf::SurfaceBundle;
#     gridded::Bool = true,
#     latent::Bool = false,
# )
#     if gridded
#         ρ1surf, ρ2surf, δsurf = extract_true_surfaces(S, surf)
#         Y = Array{Float32}(undef, size(ρ1surf, 1), size(ρ1surf, 2), 3, 1)
#         Y[:, :, 1, 1] = ρ1surf
#         Y[:, :, 2, 1] = ρ2surf
#         Y[:, :, 3, 1] = δsurf
#         return Y
#     else
#         if latent
#             return extract_true_nodewise_latent(surf)
#         else
#             return extract_true_nodewise(surf)
#         end
#     end
# end

# =========================================================
# 2) input preparation
# =========================================================

"""
Grid input path
X: H×W×1×m
S: n×2
return: H×W×3×m

channel 1 = field replicate
channel 2 = x-coordinate
channel 3 = y-coordinate
"""
function _prepare_input_grid(X::AbstractArray{<:Real,4}, S::AbstractMatrix)
    # H, W, _, m = size(X)
    # xsurf, ysurf = coord_surfaces(S)

    # inp = Array{Float32}(undef, H, W, 3, m)
    # raw = Float32.(Array(X[:, :, 1, :]))

    # inp[:, :, 1, :] = raw
    # for r in 1:m
    #     inp[:, :, 2, r] = xsurf
    #     inp[:, :, 3, r] = ysurf
    # end
    # return inp
    return Float32.(Array(X))
end

"""
gridded=true:
    X must be H×W×1×m
    return H×W×3×m

gridded=false:
    irregular 路径不再单独走 prepare_input；
    直接在 build_supervised_dataset 里构造 GNNGraph
"""
function prepare_input(
    X,
    S::AbstractMatrix;
    gridded::Bool = true,
)
    if gridded
        X isa AbstractArray{<:Real,4} || error("When gridded=true, X must be H×W×1×m.")
        return _prepare_input_grid(X, S)
    else
        error("prepare_input(...; gridded=false) is no longer used. Use nodegraph(g, Z) inside build_supervised_dataset.")
    end
end

# =========================================================
# 3) build supervised dataset
# =========================================================

"""
parameters: output of prior(...)
Z: output of simulate(parameters, m)

If gridded=true:
    X_list[k] = H×W×3×m
    Y_list[k] = H×W×3×1

If gridded=false:
    X_list[k] = GNNGraph
    Y_list[k] = 3×n
"""

# function build_supervised_dataset(
#     parameters::Parameters,
#     Z;
#     gridded::Bool = true,
#     latent_targets::Bool = false,
#     return_rho1::Bool = true,
#     return_rho2::Bool = true,
#     return_delta::Bool = true,
# )
#     K = length(Z)

#     if gridded
#         X_list = Vector{Array{Float32,4}}(undef, K)
#         Y_list = Vector{Array{Float32,4}}(undef, K)

#         for k in 1:K
#             S_k    = get_sites(parameters, k)
#             surf_k = get_surface(parameters, k)
#             X_k    = Z[k]

#             X_list[k] = prepare_input(X_k, S_k; gridded = true)
#             Y_list[k] = target_tensor(
#                 S_k, surf_k;
#                 gridded = true,
#                 latent = false,
#                 return_rho1 = return_rho1,
#                 return_rho2 = return_rho2,
#                 return_delta = return_delta,
#             )
#         end

#         return X_list, Y_list

#     else
#         X_list = Vector{GNNGraph}(undef, K)
#         Y_list = Vector{Matrix{Float32}}(undef, K)

#         for k in 1:K
#             S_k    = get_sites(parameters, k)
#             g_k    = get_graph(parameters, k)
#             surf_k = get_surface(parameters, k)
#             Z_k    = Z[k]   # n × m

#             X_list[k] = nodegraph(g_k, Z_k)

#             Y_list[k] = target_tensor(
#                 S_k, surf_k;
#                 gridded = false,
#                 latent = latent_targets,
#                 return_rho1 = return_rho1,
#                 return_rho2 = return_rho2,
#                 return_delta = return_delta,
#             )
#         end

#         return X_list, Y_list
#     end
# end



function build_supervised_dataset(
    parameters::Parameters,
    Z;
    gridded::Bool = true,
    latent_targets::Bool = false,
    return_rho1::Bool = true,
    return_rho2::Bool = true,
    return_delta::Bool = true,
)
    K = length(Z)

    if gridded
        X_list = Vector{Array{Float32,4}}(undef, K)
        Y_list = Vector{Array{Float32,4}}(undef, K)

        for k in 1:K
            S_k    = get_sites(parameters, k)
            surf_k = get_surface(parameters, k)
            X_k    = Z[k]   # H × W × 1 × m

            X_list[k] = prepare_input(X_k, S_k; gridded = true)

            Y_list[k] = target_tensor(
                S_k, surf_k;
                gridded = true,
                latent = false,
                return_rho1 = return_rho1,
                return_rho2 = return_rho2,
                return_delta = return_delta,
            )
        end

        return X_list, Y_list

    else
        X_list = Vector{GNNGraph}(undef, K)
        Y_list = Vector{Matrix{Float32}}(undef, K)

        for k in 1:K
            S_k    = get_sites(parameters, k)
            surf_k = get_surface(parameters, k)
            gZ_k   = Z[k]   # already GNNGraph from simulate_irregular

            gZ_k isa GNNGraph || throw(ArgumentError(
                "For gridded=false, Z[$k] must be a GNNGraph returned by simulate_irregular."
            ))

            X_list[k] = gZ_k

            Y_list[k] = target_tensor(
                S_k, surf_k;
                gridded = false,
                latent = latent_targets,
                return_rho1 = return_rho1,
                return_rho2 = return_rho2,
                return_delta = return_delta,
            )
        end

        return X_list, Y_list
    end
end
