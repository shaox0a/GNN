# struct SurfaceBundle{T}
#     sigma::Vector{T}
#     rho1::Vector{T}
#     rho2::Vector{T}
#     delta::Vector{T}
# end

# function SurfaceBundle(
#     sigma::AbstractVector,
#     rho1::AbstractVector,
#     rho2::AbstractVector,
#     delta::AbstractVector,
# )
#     n = length(sigma)
#     length(rho1) == n || error("rho1 length mismatch")
#     length(rho2) == n || error("rho2 length mismatch")
#     length(delta) == n || error("delta length mismatch")

#     return SurfaceBundle{Float32}(
#         Float32.(sigma),
#         Float32.(rho1),
#         Float32.(rho2),
#         Float32.(delta),
#     )
# end



# function Base.convert(::Type{SurfaceBundle{Float32}}, surf::SurfaceBundle)
#     return SurfaceBundle{Float32}(
#         Float32.(surf.sigma),
#         Float32.(surf.rho1),
#         Float32.(surf.rho2),
#         Float32.(surf.delta),
#     )
# end


# ------------------------------------------------------------------------

# struct SurfaceBundle{T}
#     sigma::AbstractVector{T}
#     rho1::AbstractVector{T}
#     rho2::AbstractVector{T}
#     delta::AbstractVector{T}
#     g::Any
# end

# function SurfaceBundle(
#     sigma::AbstractVector,
#     rho1::AbstractVector,
#     rho2::AbstractVector,
#     delta::AbstractVector,
#     g,
# )
#     n = length(sigma)
#     length(rho1) == n || error("rho1 length mismatch")
#     length(rho2) == n || error("rho2 length mismatch")
#     length(delta) == n || error("delta length mismatch")

#     return SurfaceBundle{Float32}(
#         Float32.(sigma),
#         Float32.(rho1),
#         Float32.(rho2),
#         Float32.(delta),
#         g,
#     )
# end

# function SurfaceBundle(
#     sigma::AbstractVector,
#     rho1::AbstractVector,
#     rho2::AbstractVector,
#     delta::AbstractVector;
#     g = nothing,
# )
#     return SurfaceBundle(sigma, rho1, rho2, delta, g)
# end

# function Base.convert(::Type{SurfaceBundle{Float32}}, surf::SurfaceBundle)
#     return SurfaceBundle(
#         surf.sigma,
#         surf.rho1,
#         surf.rho2,
#         surf.delta,
#         surf.g,
#     )
# end

# -------------------------------------------------------------------------

struct SurfaceBundle{T}
    sigma::AbstractVector{T}
    rho1::AbstractVector{T}
    rho2::AbstractVector{T}
    delta::AbstractVector{T}
    nu::T          # Matérn smoothness, constant over the domain
    g::Any
end

function SurfaceBundle(
    sigma::AbstractVector,
    rho1::AbstractVector,
    rho2::AbstractVector,
    delta::AbstractVector,
    g;
    nu::Real = 1.5,
)
    n = length(sigma)
    length(rho1) == n || error("rho1 length mismatch")
    length(rho2) == n || error("rho2 length mismatch")
    length(delta) == n || error("delta length mismatch")

    return SurfaceBundle{Float32}(
        Float32.(sigma),
        Float32.(rho1),
        Float32.(rho2),
        Float32.(delta),
        Float32(nu),
        g,
    )
end

function SurfaceBundle(
    sigma::AbstractVector,
    rho1::AbstractVector,
    rho2::AbstractVector,
    delta::AbstractVector;
    g = nothing,
    nu::Real = 1.5,
)
    return SurfaceBundle(sigma, rho1, rho2, delta, g; nu = nu)
end

function Base.convert(::Type{SurfaceBundle{Float32}}, surf::SurfaceBundle)
    return SurfaceBundle(
        surf.sigma,
        surf.rho1,
        surf.rho2,
        surf.delta,
        surf.g;
        nu = surf.nu,
    )
end