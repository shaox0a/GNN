using Statistics
using GraphNeuralNetworks: edge_index

function nodewise_mae_nbe_study1(
    θhat::AbstractVector{<:AbstractMatrix},
    y::AbstractVector{<:SurfaceBundle};
    λ_smooth::Real = 1f-1,
    smooth_weight::Symbol = :inv_distance,
)
    @assert length(θhat) == length(y)

    vals = map(eachindex(y)) do k
        surf = y[k]

        yk = Float32.(reshape(surf.rho1, 1, :))
        yk = convert(typeof(θhat[k]), yk)

        mae = mean(abs, θhat[k] .- yk)

        surf.g === nothing && error(
            "surf.g is nothing. Check that graph has been attached to each SurfaceBundle in prior(...)."
        )

        s, t = edge_index(surf.g)

        if smooth_weight == :constant
            w = ones(Float32, length(s))
        elseif smooth_weight == :inv_distance
            e = :e ∈ keys(surf.g.edata) ? surf.g.edata.e : permutedims(surf.g.graph[3])
            e = e isa AbstractVector ? permutedims(e) : e

            d = if size(e, 1) >= 7
                vec(e[7, :])
            elseif size(e, 1) >= 6
                vec(sqrt.(sum(abs2, e[5:6, :], dims = 1)))
            else
                vec(sqrt.(sum(abs2, e[1:2, :], dims = 1)))
            end

            w = Float32.(1.0 ./ max.(d, eps(Float32)))
        else
            error("smooth_weight must be :inv_distance or :constant.")
        end

        w = convert(typeof(vec(θhat[k][1, s])), w)
        smooth = mean(reshape(w ./ mean(w), 1, :) .* abs.(θhat[k][:, s] .- θhat[k][:, t]))

        return mae + λ_smooth * smooth
    end

    return mean(vals)
end


function quantile_loss(q, y, τ)
    ind = Float32.(y .< q)
    ind = convert(typeof(q), ind)
    return mean((τ .- ind) .* (y .- q))
end

function nodewise_interval_loss_study1(
    θhat::AbstractVector{<:AbstractMatrix},
    y::AbstractVector{<:SurfaceBundle};
    probs = (0.025f0, 0.975f0),
    λ_smooth::Real = 1f-1,
    smooth_weight::Symbol = :inv_distance,
)
    @assert length(θhat) == length(y)

    τl, τu = probs

    vals = map(eachindex(y)) do k
        surf = y[k]

        yk = Float32.(reshape(surf.rho1, 1, :))

        ql = θhat[k][1:1, :]
        qu = θhat[k][2:2, :]

        yk = convert(typeof(ql), yk)

        loss_l = quantile_loss(ql, yk, τl)
        loss_u = quantile_loss(qu, yk, τu)

        surf.g === nothing && error(
            "surf.g is nothing. Check that graph has been attached to each SurfaceBundle in prior(...)."
        )

        s, t = edge_index(surf.g)

        if smooth_weight == :constant
            w = ones(Float32, length(s))
        elseif smooth_weight == :inv_distance
            e = :e ∈ keys(surf.g.edata) ? surf.g.edata.e : permutedims(surf.g.graph[3])
            e = e isa AbstractVector ? permutedims(e) : e

            d = if size(e, 1) >= 7
                vec(e[7, :])
            elseif size(e, 1) >= 6
                vec(sqrt.(sum(abs2, e[5:6, :], dims = 1)))
            else
                vec(sqrt.(sum(abs2, e[1:2, :], dims = 1)))
            end

            w = Float32.(1.0 ./ max.(d, eps(Float32)))
        else
            error("smooth_weight must be :inv_distance or :constant.")
        end

        w = convert(typeof(vec(θhat[k][1, s])), w)
        smooth = mean(reshape(w ./ mean(w), 1, :) .* abs.(θhat[k][:, s] .- θhat[k][:, t]))

        return loss_l + loss_u + λ_smooth * smooth
    end

    return mean(vals)
end