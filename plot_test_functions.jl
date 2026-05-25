

# =========================================================
# 6a) prediction for gridded-data CNN
# =========================================================

function predict_surfaces(model, X, S)
    x = prepare_input(X, S; gridded = true)
    ŷ = model(x)

    rho1_surface_hat  = Array(ŷ[:, :, 1, 1])
    rho2_surface_hat  = Array(ŷ[:, :, 2, 1])
    delta_surface_hat = Array(ŷ[:, :, 3, 1])

    return rho1_surface_hat, rho2_surface_hat, delta_surface_hat
end

# =========================================================
# 6b) nodewise prediction for irregular-data GNN
# =========================================================

function predict_nodewise(model, X, S, g)
    x = prepare_input(X, S; gridded=false)   # 3×n×m
    ŷ = model(g, x)                          # 3×n

    rho1_hat  = Array(vec(ŷ[1, :]))
    rho2_hat  = Array(vec(ŷ[2, :]))
    delta_hat = Array(vec(ŷ[3, :]))

    return rho1_hat, rho2_hat, delta_hat
end

function predict_nodewise(
    model,
    X,
    S;
    k_graph::Int = 8,
)
    x = prepare_input(X, S; gridded = false)
    batch = make_graph_batch(x, S; k_graph = k_graph)
    return predict_nodewise(model, batch)
end

# =========================================================
# 7) quick plotting helper
# =========================================================

function plot_three_surfaces(A, B, C; titles=("rho1", "rho2", "delta"), resolution=(1100, 350))
    fig = Figure(resolution=resolution)

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title=titles[1])
    hm1 = heatmap!(ax1, A)
    Colorbar(fig[1,2], hm1)

    ax2 = Axis(fig[1,3], aspect=DataAspect(), title=titles[2])
    hm2 = heatmap!(ax2, B)
    Colorbar(fig[1,4], hm2)

    ax3 = Axis(fig[1,5], aspect=DataAspect(), title=titles[3])
    hm3 = heatmap!(ax3, C)
    Colorbar(fig[1,6], hm3)

    return fig
end


using CairoMakie

function _expand_to_n(x, n::Int)
    if x isa Tuple || x isa AbstractVector
        length(x) == n || error("Expected length $n, got $(length(x)).")
        return collect(x)
    else
        return fill(x, n)
    end
end

function _scatter_with_optional_crange!(
    ax,
    xs,
    ys,
    vals;
    colormap = :viridis,
    marker = :circle,
    markersize = 10,
    colorrange = nothing,
)
    if isnothing(colorrange)
        return scatter!(
            ax, xs, ys;
            color = vals,
            colormap = colormap,
            marker = marker,
            markersize = markersize,
        )
    else
        return scatter!(
            ax, xs, ys;
            color = vals,
            colormap = colormap,
            marker = marker,
            markersize = markersize,
            colorrange = colorrange,
        )
    end
end

function plot_three_node_fields(
    S::AbstractMatrix,
    A::AbstractVector,
    B::AbstractVector,
    C::AbstractVector;
    titles = ("rho1", "rho2", "delta"),
    resolution = (1100, 350),
    cmaps = (:viridis, :viridis, :balance),
    cranges = (nothing, nothing, (-1, 1)),
    marker = :circle,
    markersize = 10,
)
    fig = Figure(resolution = resolution)

    vals_list = (A, B, C)
    markers = _expand_to_n(marker, 3)
    markersizes = _expand_to_n(markersize, 3)

    for j in 1:3
        ax = Axis(fig[1, 2j - 1], aspect = DataAspect(), title = titles[j])
        sc = _scatter_with_optional_crange!(
            ax, S[:,1], S[:,2], vals_list[j];
            colormap = cmaps[j],
            marker = markers[j],
            markersize = markersizes[j],
            colorrange = cranges[j],
        )
        Colorbar(fig[1, 2j], sc)
    end

    return fig
end

function plot_true_hat_node_fields(
    S::AbstractMatrix,
    A_true::AbstractVector,
    B_true::AbstractVector,
    C_true::AbstractVector,
    A_hat::AbstractVector,
    B_hat::AbstractVector,
    C_hat::AbstractVector;
    titles = (
        "rho1 true", "rho2 true", "delta true",
        "rho1 hat",  "rho2 hat",  "delta hat"
    ),
    resolution = (1600, 650),
    cmaps = (:viridis, :viridis, :balance, :viridis, :viridis, :balance),
    cranges = ((0, 1), (0, 1), (-1, 1), (0, 1), (0, 1), (-1, 1)),
    marker = :circle,
    markersize = 10,
)
    fig = Figure(resolution = resolution)

    vals_list = (A_true, B_true, C_true, A_hat, B_hat, C_hat)
    markers = _expand_to_n(marker, 6)
    markersizes = _expand_to_n(markersize, 6)

    for j in 1:6
        row = j <= 3 ? 1 : 2
        col0 = j <= 3 ? 2j - 1 : 2(j - 3) - 1

        ax = Axis(fig[row, col0], aspect = DataAspect(), title = titles[j])
        sc = _scatter_with_optional_crange!(
            ax, S[:,1], S[:,2], vals_list[j];
            colormap = cmaps[j],
            marker = markers[j],
            markersize = markersizes[j],
            colorrange = cranges[j],
        )
        Colorbar(fig[row, col0 + 1], sc)
    end

    return fig
end




function _safe_colorrange(lo::Real, hi::Real; pad_frac::Real = 0.02, pad_min::Real = 1e-6)
    lo = Float64(lo)
    hi = Float64(hi)

    if !isfinite(lo) || !isfinite(hi)
        return (0.0, 1.0)
    end

    if hi < lo
        lo, hi = hi, lo
    end

    if isapprox(lo, hi; atol = 1e-12, rtol = 1e-12)
        pad = max(abs(lo) * pad_frac, pad_min)
        return (lo - pad, hi + pad)
    end

    return (lo, hi)
end


function plot_true_hat_surface_interp(
    S::AbstractMatrix,
    surf_true::SurfaceBundle,
    surf_hat::SurfaceBundle;
    which::Tuple = (:sigma, :rho1, :rho2, :delta),
    irregular_mode::Symbol = :interp,
    add_contour::Bool = true,
    interp_nx::Integer = 140,
    interp_ny::Integer = 140,
    interp_power::Real = 2.0,
    point_markersize::Real = 10,
    interp_site_markersize::Real = 2.5,
    interp_site_color = :black,
    interp_site_alpha::Real = 0.45,
    colorrange_mode::Symbol = :shared,   # :shared or :fixed
    sigma_range::Tuple{<:Real,<:Real} = (0.5, 2.0),
    resolution = (1200, 1400),
)
    n = size(S, 1)

    length(surf_true.sigma) == n || error("surf_true.sigma length mismatch with S")
    length(surf_true.rho1)  == n || error("surf_true.rho1 length mismatch with S")
    length(surf_true.rho2)  == n || error("surf_true.rho2 length mismatch with S")
    length(surf_true.delta) == n || error("surf_true.delta length mismatch with S")

    length(surf_hat.sigma) == n || error("surf_hat.sigma length mismatch with S")
    length(surf_hat.rho1)  == n || error("surf_hat.rho1 length mismatch with S")
    length(surf_hat.rho2)  == n || error("surf_hat.rho2 length mismatch with S")
    length(surf_hat.delta) == n || error("surf_hat.delta length mismatch with S")

    σ_true  = vec(Array(surf_true.sigma))
    ρ1_true = vec(Array(surf_true.rho1))
    ρ2_true = vec(Array(surf_true.rho2))
    δ_true  = vec(Array(surf_true.delta))

    σ_hat  = vec(Array(surf_hat.sigma))
    ρ1_hat = vec(Array(surf_hat.rho1))
    ρ2_hat = vec(Array(surf_hat.rho2))
    δ_hat  = vec(Array(surf_hat.delta))

    function pick_range(vtrue, vhat, sym)
        if colorrange_mode == :fixed
            if sym === :sigma
                return (Float64(sigma_range[1]), Float64(sigma_range[2]))
            elseif sym === :delta
                return (-1.0, 1.0)
            elseif sym === :rho1 || sym === :rho2
                return (0.0, 1.0)
            else
                error("Unknown surface symbol: $sym")
            end
        elseif colorrange_mode == :shared
            return _safe_colorrange(
                min(minimum(vtrue), minimum(vhat)),
                max(maximum(vtrue), maximum(vhat)),
            )
        else
            error("colorrange_mode must be :shared or :fixed.")
        end
    end

    all_rows = Dict(
        :sigma => (
            σ_true, σ_hat,
            "True σ(s)", "Estimated σ(s)",
            :viridis,
            pick_range(σ_true, σ_hat, :sigma),
        ),
        :rho1 => (
            ρ1_true, ρ1_hat,
            "True ρ₁(s)", "Estimated ρ₁(s)",
            :viridis,
            pick_range(ρ1_true, ρ1_hat, :rho1),
        ),
        :rho2 => (
            ρ2_true, ρ2_hat,
            "True ρ₂(s)", "Estimated ρ₂(s)",
            :viridis,
            pick_range(ρ2_true, ρ2_hat, :rho2),
        ),
        :delta => (
            δ_true, δ_hat,
            "True δ(s)", "Estimated δ(s)",
            :balance,
            pick_range(δ_true, δ_hat, :delta),
        ),
    )

    for k in which
        haskey(all_rows, k) || error(
            "Unknown surface: $k. Use one or more of (:sigma, :rho1, :rho2, :delta)."
        )
    end

    rows = [all_rows[k] for k in which]

    fig = Figure(size = resolution)

    function draw_panel!(ax, values, ttl, cmap, cr)
        if irregular_mode == :interp
            xg, yg, Zg = _idw_surface(
                S, values;
                nx = interp_nx,
                ny = interp_ny,
                power = interp_power,
            )

            plt = _surface_plot!(
                ax, xg, yg, Zg;
                title = ttl,
                colorrange = cr,
                colormap = cmap,
                add_contour = add_contour,
            )

            scatter!(
                ax, S[:, 1], S[:, 2];
                markersize = interp_site_markersize,
                color = interp_site_color,
                alpha = interp_site_alpha,
            )

            return plt

        elseif irregular_mode == :points
            return _points_plot!(
                ax, S, values;
                title = ttl,
                colorrange = cr,
                colormap = cmap,
                markersize = point_markersize,
            )
        else
            error("irregular_mode must be :interp or :points.")
        end
    end

    for (i, (vtrue, vhat, ttl_true, ttl_hat, cmap, cr)) in enumerate(rows)
        ax_true = Axis(fig[i, 1], aspect = DataAspect())
        ax_hat  = Axis(fig[i, 2], aspect = DataAspect())

        plt_true = draw_panel!(ax_true, vtrue, ttl_true, cmap, cr)
        _        = draw_panel!(ax_hat,  vhat,  ttl_hat,  cmap, cr)

        Colorbar(fig[i, 3], plt_true)
    end

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 12)

    return fig
end