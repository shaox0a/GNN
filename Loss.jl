using Statistics
using GraphNeuralNetworks: edge_index


function nodewise_mae_nbe(
    θhat::AbstractVector{<:AbstractMatrix},
    y::AbstractVector{<:SurfaceBundle};
    λ_smooth::Real = 1f-1,
    smooth_weight::Symbol = :inv_distance,
)
    @assert length(θhat) == length(y)

    vals = map(eachindex(y)) do k
        surf = y[k]

        yk = Float32.(vcat(
            reshape(surf.rho1,  1, :),
            reshape(surf.rho2,  1, :),
            reshape(surf.delta, 1, :),
        ))

        # Move yk to the same array/device type as θhat[k],
        # without explicitly referencing CUDA/CuArray.
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
            w = w ./ mean(w)
        else
            error("smooth_weight must be :inv_distance or :constant.")
        end

        w = convert(typeof(vec(θhat[k][1, s])), w)
        smooth = mean(reshape(w, 1, :) .* abs.(θhat[k][:, s] .- θhat[k][:, t]))

        return mae + λ_smooth * smooth
    end

    return mean(vals)
end




function nodewise_mae_nbe_v31(
    θhat::AbstractVector,
    y::AbstractVector{<:SurfaceBundle};
    λ_smooth::Real = 1f-1,
    smooth_weight::Symbol = :inv_distance,
)
    @assert length(θhat) == length(y)

    site_scale_vec = Float32[1.5, 1.0, 1.0, 2.0]  # sigma/rho1/rho2/delta
    nu_scale = 2f0                                # nu range roughly 0.5--2.5

    vals = map(eachindex(y)) do k
        surf = y[k]

        site_hat, nu_hat = θhat[k]     # site_hat: 4×n, nu_hat: 1×1
        n = length(surf.rho1)

        y_site = Float32.(vcat(
            reshape(surf.sigma, 1, :),
            reshape(surf.rho1,  1, :),
            reshape(surf.rho2,  1, :),
            reshape(surf.delta, 1, :),
        ))

        y_site = convert(typeof(site_hat), y_site)

        site_scale = convert(
            typeof(site_hat),
            reshape(site_scale_vec, :, 1),
        )

        site_mae = mean(abs.(site_hat .- y_site) ./ site_scale)

        # nu_hat is 1×1, surf.nu is scalar
        y_nu = fill(Float32(surf.nu), size(nu_hat))
        y_nu = convert(typeof(nu_hat), y_nu)

        nu_mae = mean(abs.(nu_hat .- y_nu)) / nu_scale

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
            w = w ./ mean(w)
        else
            error("smooth_weight must be :inv_distance or :constant.")
        end

        w = convert(typeof(vec(site_hat[1, s])), w)
        smooth = mean(reshape(w, 1, :) .* abs.(site_hat[:, s] .- site_hat[:, t]) ./ site_scale)

        return site_mae + nu_mae + λ_smooth * smooth
    end

    return mean(vals)
end