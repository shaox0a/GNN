using CairoMakie
using LinearAlgebra
using Statistics

# =========================================================
# plot_functions.jl
# surface-only version
#
# 要求：
#   1) 先 include Surface_types.jl
#   2) 再 include sim_experiment3.jl
#      因为这里会用到 Omega_general(...)
# =========================================================

# =========================================================
# helpers
# =========================================================

function _resolve_sites(S::AbstractMatrix, sample_idx::Int=1)
    return S
end

function _resolve_sites(S::AbstractVector{<:AbstractMatrix}, sample_idx::Int=1)
    1 <= sample_idx <= length(S) || error("sample_idx out of bounds.")
    return S[sample_idx]
end

function _resolve_surface(surf::SurfaceBundle, sample_idx::Int=1)
    return surf
end

function _resolve_surface(surf::AbstractVector{<:SurfaceBundle}, sample_idx::Int=1)
    1 <= sample_idx <= length(surf) || error("sample_idx out of bounds.")
    return surf[sample_idx]
end

# =========================================================
# gridded helper
# =========================================================

function gridify(S::AbstractMatrix, v::AbstractVector)
    x = sort(unique(S[:, 1]))
    y = sort(unique(S[:, 2]))
    Z = fill(NaN, length(x), length(y))

    @inbounds for idx in eachindex(v)
        i = searchsortedfirst(x, S[idx, 1])
        j = searchsortedfirst(y, S[idx, 2])
        Z[i, j] = v[idx]
    end

    return collect(x), collect(y), Z
end

# =========================================================
# replicate extraction
# =========================================================

function _extract_sitewise_replicate(X::AbstractMatrix, nsites::Integer, rep::Integer)
    nr, nc = size(X)

    if nr == nsites
        1 <= rep <= nc || error("rep=$rep out of bounds: X has $nc replicates in columns.")
        return vec(Array(X[:, rep]))
    elseif nc == nsites
        1 <= rep <= nr || error("rep=$rep out of bounds: X has $nr replicates in rows.")
        return vec(Array(X[rep, :]))
    else
        error(
            "For irregular plotting, X must have one dimension equal to number of sites.\n" *
            "size(S,1) = $nsites, but size(X) = ($(nr), $(nc))."
        )
    end
end

function _extract_plot_replicate(X, nsites::Integer; rep::Integer=1, gridded::Bool=true)
    if gridded
        X isa AbstractArray{<:Real,4} || error("When gridded=true, X must be H×W×1×m.")
        _, _, _, m = size(X)
        1 <= rep <= m || error("rep=$rep out of bounds: X has $m replicates.")
        return Array(X[:, :, 1, rep])  # H×W
    else
        if X isa AbstractVector
            length(X) == nsites || error(
                "For irregular plotting with vector input, length(X) must equal size(S,1).\n" *
                "length(X) = $(length(X)), size(S,1) = $nsites."
            )
            return vec(Array(X))
        elseif X isa AbstractMatrix
            return _extract_sitewise_replicate(X, nsites, rep)
        elseif X isa AbstractArray{<:Real,4}
            H, W, _, m = size(X)
            H * W == nsites || error(
                "For irregular plotting with 4D input, H*W must equal size(S,1).\n" *
                "H*W = $(H*W), size(S,1) = $nsites."
            )
            1 <= rep <= m || error("rep=$rep out of bounds: X has $m replicates.")
            return vec(Array(X[:, :, 1, rep]))
        else
            error("Unsupported X type for plotting.")
        end
    end
end

# =========================================================
# parameter fields from SurfaceBundle
# =========================================================

function _surface_fields(S::AbstractMatrix, surf::SurfaceBundle)
    n = size(S, 1)
    length(surf.sigma) == n || error("surf.sigma length mismatch with S")
    length(surf.rho1)  == n || error("surf.rho1 length mismatch with S")
    length(surf.rho2)  == n || error("surf.rho2 length mismatch with S")
    length(surf.delta) == n || error("surf.delta length mismatch with S")

    return (
        Float64.(surf.sigma),
        Float64.(surf.rho1),
        Float64.(surf.rho2),
        Float64.(surf.delta),
    )
end

# =========================================================
# spacing / ellipses
# =========================================================

function _characteristic_spacing(S::AbstractMatrix)
    n = size(S, 1)
    n <= 1 && return 1.0

    nn = fill(Inf, n)

    @inbounds for i in 1:n-1
        xi, yi = S[i, 1], S[i, 2]
        for j in i+1:n
            dx = xi - S[j, 1]
            dy = yi - S[j, 2]
            d = sqrt(dx * dx + dy * dy)
            if d < nn[i]
                nn[i] = d
            end
            if d < nn[j]
                nn[j] = d
            end
        end
    end

    vals = nn[isfinite.(nn) .& (nn .> 0)]
    isempty(vals) && return 1.0
    return median(vals)
end

function ellipse_points(
    S::AbstractMatrix,
    ρ1::AbstractVector,
    ρ2::AbstractVector,
    δ::AbstractVector;
    scale::Real = 0.60,
    npts::Integer = 80,
    every::Integer = 1,
)
    cell = _characteristic_spacing(S)
    s = scale * cell

    idxs = collect(1:every:size(S, 1))
    ell = Vector{Matrix{Float64}}(undef, length(idxs))

    for (k, i) in enumerate(idxs)
        δi = clamp(δ[i], -1 + 1e-8, 1 - 1e-8)
        Ω = Matrix(Omega_general(ρ1[i], ρ2[i], δi))
        ee = eigen(Symmetric(Ω))
        λ = max.(ee.values, 1e-12)
        Q = ee.vectors
        B = Q * Diagonal(s .* sqrt.(λ))

        t = range(0, 2π, length=npts)
        P = B * hcat(cos.(t), sin.(t))'

        x = S[i, 1] .+ P[1, :]
        y = S[i, 2] .+ P[2, :]
        ell[k] = hcat(x, y)
    end

    return ell, idxs
end

function _draw_ellipses!(
    ax,
    S::AbstractMatrix,
    ρ1::AbstractVector,
    ρ2::AbstractVector,
    δ::AbstractVector;
    scale::Real = 0.60,
    npts::Integer = 80,
    every::Integer = 1,
    color = :white,
    linewidth::Real = 1.0,
    alpha::Real = 0.9,
)
    ell, _ = ellipse_points(S, ρ1, ρ2, δ; scale=scale, npts=npts, every=every)
    for E in ell
        lines!(ax, E[:, 1], E[:, 2]; color=color, linewidth=linewidth, alpha=alpha)
    end
    return ax
end

# =========================================================
# irregular interpolation
# =========================================================

function _regular_plot_grid(
    S::AbstractMatrix;
    nx::Integer = 140,
    ny::Integer = 140,
    pad_frac::Real = 0.02,
)
    xmin, xmax = extrema(S[:, 1])
    ymin, ymax = extrema(S[:, 2])

    dx = xmax - xmin
    dy = ymax - ymin

    padx = max(pad_frac * dx, 1e-6)
    pady = max(pad_frac * dy, 1e-6)

    xg = collect(range(xmin - padx, xmax + padx, length=nx))
    yg = collect(range(ymin - pady, ymax + pady, length=ny))
    return xg, yg
end

function _idw_surface(
    S::AbstractMatrix,
    v::AbstractVector;
    nx::Integer = 140,
    ny::Integer = 140,
    power::Real = 2.0,
    eps::Real = 1e-10,
    pad_frac::Real = 0.02,
)
    length(v) == size(S, 1) || error("length(v) must match number of rows in S.")

    xg, yg = _regular_plot_grid(S; nx=nx, ny=ny, pad_frac=pad_frac)
    Zg = Matrix{Float64}(undef, length(xg), length(yg))

    xs = Float64.(vec(S[:, 1]))
    ys = Float64.(vec(S[:, 2]))
    vv = Float64.(vec(v))

    @inbounds for ix in eachindex(xg)
        x0 = xg[ix]
        for iy in eachindex(yg)
            y0 = yg[iy]

            d2 = (xs .- x0).^2 .+ (ys .- y0).^2
            j = argmin(d2)

            if d2[j] <= eps
                Zg[ix, iy] = vv[j]
            else
                w = (d2 .+ eps).^(-power / 2)
                Zg[ix, iy] = sum(w .* vv) / sum(w)
            end
        end
    end

    return xg, yg, Zg
end

# =========================================================
# low-level plotting
# =========================================================

function _surface_plot!(
    ax,
    x::AbstractVector,
    y::AbstractVector,
    Z::AbstractMatrix;
    title::AbstractString = "",
    colorrange = nothing,
    colormap = :viridis,
    add_contour::Bool = false,
    contour_color = :white,
    contour_linewidth::Real = 1.0,
)
    hm = isnothing(colorrange) ?
        heatmap!(ax, x, y, Z; colormap=colormap) :
        heatmap!(ax, x, y, Z; colormap=colormap, colorrange=colorrange)

    if add_contour
        contour!(ax, x, y, Z; color=contour_color, linewidth=contour_linewidth)
    end

    ax.title = title
    ax.xlabel = "x"
    ax.ylabel = "y"
    return hm
end

function _points_plot!(
    ax,
    S::AbstractMatrix,
    v::AbstractVector;
    title::AbstractString = "",
    colorrange = nothing,
    colormap = :viridis,
    markersize::Real = 10,
)
    plt = isnothing(colorrange) ?
        scatter!(ax, S[:, 1], S[:, 2]; color=v, colormap=colormap, markersize=markersize) :
        scatter!(ax, S[:, 1], S[:, 2]; color=v, colormap=colormap, colorrange=colorrange, markersize=markersize)

    ax.title = title
    ax.xlabel = "x"
    ax.ylabel = "y"
    return plt
end

# =========================================================
# internal implementations
# =========================================================

function _plot_replicate_impl(
    S::AbstractMatrix,
    X,
    surf::Union{Nothing,SurfaceBundle}=nothing;
    rep::Integer=1,
    gridded::Bool=true,
    irregular_mode::Symbol=:interp,
    add_contour::Bool=true,
    add_ellipses::Bool=false,
    ellipse_every::Integer=1,
    ellipse_scale::Real=0.60,
    ellipse_npts::Integer=80,
    ellipse_color=:white,
    ellipse_linewidth::Real=1.0,
    title::Union{Nothing,String}=nothing,
    colorrange=nothing,
    colormap=:viridis,
    interp_nx::Integer=140,
    interp_ny::Integer=140,
    interp_power::Real=2.0,
    point_markersize::Real=10,
    interp_site_markersize::Real=2.5,
    interp_site_color=:black,
    interp_site_alpha::Real=0.45,
    resolution=(700, 600),
)
    nsites = size(S, 1)
    Zrep = _extract_plot_replicate(X, nsites; rep=rep, gridded=gridded)

    fig = Figure(size=resolution)
    ax = Axis(fig[1, 1], aspect=DataAspect())

    if gridded
        x, y, Z = gridify(S, vec(Zrep))
        plt = _surface_plot!(
            ax, x, y, Z;
            title = isnothing(title) ? "Replicate $rep" : title,
            colorrange = colorrange,
            colormap = colormap,
            add_contour = add_contour,
        )
    else
        if irregular_mode == :interp
            xg, yg, Zg = _idw_surface(S, vec(Zrep); nx=interp_nx, ny=interp_ny, power=interp_power)
            plt = _surface_plot!(
                ax, xg, yg, Zg;
                title = isnothing(title) ? "Replicate $rep" : title,
                colorrange = colorrange,
                colormap = colormap,
                add_contour = add_contour,
            )

            scatter!(
                ax, S[:, 1], S[:, 2];
                markersize = interp_site_markersize,
                color = interp_site_color,
                alpha = interp_site_alpha,
            )

        elseif irregular_mode == :points
            plt = _points_plot!(
                ax, S, vec(Zrep);
                title = isnothing(title) ? "Replicate $rep" : title,
                colorrange = colorrange,
                colormap = colormap,
                markersize = point_markersize,
            )
        else
            error("irregular_mode must be :interp or :points.")
        end
    end

    Colorbar(fig[1, 2], plt)

    if add_ellipses
        isnothing(surf) && error("surf must be provided when add_ellipses=true.")
        _, ρ1v, ρ2v, δv = _surface_fields(S, surf)

        _draw_ellipses!(
            ax, S, ρ1v, ρ2v, δv;
            every = ellipse_every,
            scale = ellipse_scale,
            npts = ellipse_npts,
            color = ellipse_color,
            linewidth = ellipse_linewidth,
            alpha = 0.9,
        )
    end

    return fig
end

function _plot_surface_impl(
    S::AbstractMatrix,
    surf::SurfaceBundle;
    param::Symbol=:all,
    gridded::Bool=true,
    irregular_mode::Symbol=:interp,
    add_contour::Bool=false,
    interp_nx::Integer=140,
    interp_ny::Integer=140,
    interp_power::Real=2.0,
    point_markersize::Real=10,
    interp_site_markersize::Real=2.5,
    interp_site_color=:black,
    interp_site_alpha::Real=0.45,
    resolution=(1000, 800),
    title::Union{Nothing,String}=nothing,
    colorrange=nothing,
    colormap=nothing,
)
    σv, ρ1v, ρ2v, δv = _surface_fields(S, surf)

    function make_panel!(ax, values; title="", colormap=:viridis, colorrange=nothing)
        if gridded
            x, y, Z = gridify(S, values)
            plt = _surface_plot!(
                ax, x, y, Z;
                title = title,
                colorrange = colorrange,
                colormap = colormap,
                add_contour = add_contour,
            )
        else
            if irregular_mode == :interp
                xg, yg, Zg = _idw_surface(S, values; nx=interp_nx, ny=interp_ny, power=interp_power)
                plt = _surface_plot!(
                    ax, xg, yg, Zg;
                    title = title,
                    colorrange = colorrange,
                    colormap = colormap,
                    add_contour = add_contour,
                )

                scatter!(
                    ax, S[:, 1], S[:, 2];
                    markersize = interp_site_markersize,
                    color = interp_site_color,
                    alpha = interp_site_alpha,
                )

            elseif irregular_mode == :points
                plt = _points_plot!(
                    ax, S, values;
                    title = title,
                    colorrange = colorrange,
                    colormap = colormap,
                    markersize = point_markersize,
                )
            else
                error("irregular_mode must be :interp or :points.")
            end
        end
        return plt
    end

    if param == :sigma
        fig = Figure(size=(700, 600))
        ax = Axis(fig[1, 1], aspect=DataAspect())
        ttl = isnothing(title) ? "σ(s)" : title
        cmap = isnothing(colormap) ? :viridis : colormap
        plt = make_panel!(ax, σv; title=ttl, colormap=cmap, colorrange=colorrange)
        Colorbar(fig[1, 2], plt)
        return fig

    elseif param == :rho1
        fig = Figure(size=(700, 600))
        ax = Axis(fig[1, 1], aspect=DataAspect())
        ttl = isnothing(title) ? "ρ₁(s)" : title
        cmap = isnothing(colormap) ? :viridis : colormap
        plt = make_panel!(ax, ρ1v; title=ttl, colormap=cmap, colorrange=colorrange)
        Colorbar(fig[1, 2], plt)
        return fig

    elseif param == :rho2
        fig = Figure(size=(700, 600))
        ax = Axis(fig[1, 1], aspect=DataAspect())
        ttl = isnothing(title) ? "ρ₂(s)" : title
        cmap = isnothing(colormap) ? :viridis : colormap
        plt = make_panel!(ax, ρ2v; title=ttl, colormap=cmap, colorrange=colorrange)
        Colorbar(fig[1, 2], plt)
        return fig

    elseif param == :delta
        fig = Figure(size=(700, 600))
        ax = Axis(fig[1, 1], aspect=DataAspect())
        ttl = isnothing(title) ? "δ(s)" : title
        cmap = isnothing(colormap) ? :balance : colormap
        cr = isnothing(colorrange) ? (-1, 1) : colorrange
        plt = make_panel!(ax, δv; title=ttl, colormap=cmap, colorrange=cr)
        Colorbar(fig[1, 2], plt)
        return fig

    elseif param == :all
        fig = Figure(size=resolution)

        ax11 = Axis(fig[1, 1], aspect=DataAspect())
        plt11 = make_panel!(ax11, σv; title="σ(s)", colormap=:viridis)
        Colorbar(fig[1, 2], plt11)

        ax12 = Axis(fig[1, 3], aspect=DataAspect())
        plt12 = make_panel!(ax12, ρ1v; title="ρ₁(s)", colormap=:viridis)
        Colorbar(fig[1, 4], plt12)

        ax21 = Axis(fig[2, 1], aspect=DataAspect())
        plt21 = make_panel!(ax21, ρ2v; title="ρ₂(s)", colormap=:viridis)
        Colorbar(fig[2, 2], plt21)

        ax22 = Axis(fig[2, 3], aspect=DataAspect())
        plt22 = make_panel!(ax22, δv; title="δ(s)", colormap=:balance, colorrange=(-1, 1))
        Colorbar(fig[2, 4], plt22)

        return fig
    else
        error("param must be one of :sigma, :rho1, :rho2, :delta, :all")
    end
end

function _plot_ellipses_impl(
    S::AbstractMatrix,
    surf::SurfaceBundle;
    every::Integer=1,
    scale::Real=0.60,
    npts::Integer=80,
    color=:black,
    linewidth::Real=1.0,
    title::String="Local anisotropy ellipses",
    resolution=(700, 700),
)
    _, ρ1v, ρ2v, δv = _surface_fields(S, surf)

    fig = Figure(size=resolution)
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title, xlabel="x", ylabel="y")

    _draw_ellipses!(
        ax, S, ρ1v, ρ2v, δv;
        every = every,
        scale = scale,
        npts = npts,
        color = color,
        linewidth = linewidth,
        alpha = 0.9,
    )

    scatter!(ax, S[1:every:end, 1], S[1:every:end, 2]; markersize=3, color=:gray40)
    return fig
end

# =========================================================
# public methods
# =========================================================

function plot_replicate(
    S::AbstractMatrix,
    X,
    surf::Union{Nothing,SurfaceBundle}=nothing;
    rep::Integer=1,
    gridded::Bool=true,
    irregular_mode::Symbol=:interp,
    add_contour::Bool=true,
    add_ellipses::Bool=false,
    ellipse_every::Integer=1,
    ellipse_scale::Real=0.60,
    ellipse_npts::Integer=80,
    ellipse_color=:white,
    ellipse_linewidth::Real=1.0,
    title::Union{Nothing,String}=nothing,
    colorrange=nothing,
    colormap=:viridis,
    interp_nx::Integer=140,
    interp_ny::Integer=140,
    interp_power::Real=2.0,
    point_markersize::Real=10,
    interp_site_markersize::Real=2.5,
    interp_site_color=:black,
    interp_site_alpha::Real=0.45,
    resolution=(700, 600),
)
    return _plot_replicate_impl(
        S, X, surf;
        rep=rep,
        gridded=gridded,
        irregular_mode=irregular_mode,
        add_contour=add_contour,
        add_ellipses=add_ellipses,
        ellipse_every=ellipse_every,
        ellipse_scale=ellipse_scale,
        ellipse_npts=ellipse_npts,
        ellipse_color=ellipse_color,
        ellipse_linewidth=ellipse_linewidth,
        title=title,
        colorrange=colorrange,
        colormap=colormap,
        interp_nx=interp_nx,
        interp_ny=interp_ny,
        interp_power=interp_power,
        point_markersize=point_markersize,
        interp_site_markersize=interp_site_markersize,
        interp_site_color=interp_site_color,
        interp_site_alpha=interp_site_alpha,
        resolution=resolution,
    )
end

function plot_replicate(
    S::AbstractVector{<:AbstractMatrix},
    X,
    surf::Union{Nothing,SurfaceBundle,AbstractVector{<:SurfaceBundle}}=nothing;
    rep::Integer=1,
    sample_idx::Integer=1,
    gridded::Bool=true,
    irregular_mode::Symbol=:interp,
    add_contour::Bool=true,
    add_ellipses::Bool=false,
    ellipse_every::Integer=1,
    ellipse_scale::Real=0.60,
    ellipse_npts::Integer=80,
    ellipse_color=:white,
    ellipse_linewidth::Real=1.0,
    title::Union{Nothing,String}=nothing,
    colorrange=nothing,
    colormap=:viridis,
    interp_nx::Integer=140,
    interp_ny::Integer=140,
    interp_power::Real=2.0,
    point_markersize::Real=10,
    interp_site_markersize::Real=2.5,
    interp_site_color=:black,
    interp_site_alpha::Real=0.45,
    resolution=(700, 600),
)
    S_use = _resolve_sites(S, sample_idx)
    surf_use = isnothing(surf) ? nothing : _resolve_surface(surf, sample_idx)

    return _plot_replicate_impl(
        S_use, X, surf_use;
        rep=rep,
        gridded=gridded,
        irregular_mode=irregular_mode,
        add_contour=add_contour,
        add_ellipses=add_ellipses,
        ellipse_every=ellipse_every,
        ellipse_scale=ellipse_scale,
        ellipse_npts=ellipse_npts,
        ellipse_color=ellipse_color,
        ellipse_linewidth=ellipse_linewidth,
        title=title,
        colorrange=colorrange,
        colormap=colormap,
        interp_nx=interp_nx,
        interp_ny=interp_ny,
        interp_power=interp_power,
        point_markersize=point_markersize,
        interp_site_markersize=interp_site_markersize,
        interp_site_color=interp_site_color,
        interp_site_alpha=interp_site_alpha,
        resolution=resolution,
    )
end

function plot_surface(
    S::AbstractMatrix,
    surf::SurfaceBundle;
    param::Symbol=:all,
    gridded::Bool=true,
    irregular_mode::Symbol=:interp,
    add_contour::Bool=false,
    interp_nx::Integer=140,
    interp_ny::Integer=140,
    interp_power::Real=2.0,
    point_markersize::Real=10,
    interp_site_markersize::Real=2.5,
    interp_site_color=:black,
    interp_site_alpha::Real=0.45,
    resolution=(1000, 800),
    title::Union{Nothing,String}=nothing,
    colorrange=nothing,
    colormap=nothing,
)
    return _plot_surface_impl(
        S, surf;
        param=param,
        gridded=gridded,
        irregular_mode=irregular_mode,
        add_contour=add_contour,
        interp_nx=interp_nx,
        interp_ny=interp_ny,
        interp_power=interp_power,
        point_markersize=point_markersize,
        interp_site_markersize=interp_site_markersize,
        interp_site_color=interp_site_color,
        interp_site_alpha=interp_site_alpha,
        resolution=resolution,
        title=title,
        colorrange=colorrange,
        colormap=colormap,
    )
end


function plot_surface(
    S::AbstractVector{<:AbstractMatrix},
    surf::Union{SurfaceBundle,AbstractVector{<:SurfaceBundle}};
    sample_idx::Integer=1,
    param::Symbol=:all,
    gridded::Bool=true,
    irregular_mode::Symbol=:interp,
    add_contour::Bool=false,
    interp_nx::Integer=140,
    interp_ny::Integer=140,
    interp_power::Real=2.0,
    point_markersize::Real=10,
    interp_site_markersize::Real=2.5,
    interp_site_color=:black,
    interp_site_alpha::Real=0.45,
    resolution=(1000, 800),
    title::Union{Nothing,String}=nothing,
    colorrange=nothing,
    colormap=nothing,
)
    S_use = _resolve_sites(S, sample_idx)
    surf_use = _resolve_surface(surf, sample_idx)

    return _plot_surface_impl(
        S_use, surf_use;
        param=param,
        gridded=gridded,
        irregular_mode=irregular_mode,
        add_contour=add_contour,
        interp_nx=interp_nx,
        interp_ny=interp_ny,
        interp_power=interp_power,
        point_markersize=point_markersize,
        interp_site_markersize=interp_site_markersize,
        interp_site_color=interp_site_color,
        interp_site_alpha=interp_site_alpha,
        resolution=resolution,
        title=title,
        colorrange=colorrange,
        colormap=colormap,
    )
end

function plot_ellipses(
    S::AbstractMatrix,
    surf::SurfaceBundle;
    every::Integer=1,
    scale::Real=0.60,
    npts::Integer=80,
    color=:black,
    linewidth::Real=1.0,
    title::String="Local anisotropy ellipses",
    resolution=(700, 700),
)
    return _plot_ellipses_impl(
        S, surf;
        every=every,
        scale=scale,
        npts=npts,
        color=color,
        linewidth=linewidth,
        title=title,
        resolution=resolution,
    )
end

function plot_ellipses(
    S::AbstractVector{<:AbstractMatrix},
    surf::Union{SurfaceBundle,AbstractVector{<:SurfaceBundle}};
    sample_idx::Integer=1,
    every::Integer=1,
    scale::Real=0.60,
    npts::Integer=80,
    color=:black,
    linewidth::Real=1.0,
    title::String="Local anisotropy ellipses",
    resolution=(700, 700),
)
    S_use = _resolve_sites(S, sample_idx)
    surf_use = _resolve_surface(surf, sample_idx)

    return _plot_ellipses_impl(
        S_use, surf_use;
        every=every,
        scale=scale,
        npts=npts,
        color=color,
        linewidth=linewidth,
        title=title,
        resolution=resolution,
    )
end