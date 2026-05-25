using Flux
using Flux: Conv, Chain
using NNlib: softplus

# =========================================================
# CNN for gridded local-isotropy sitewise rho estimation
#
# Input:
#   X[k] : H × W × 1 × m
#
# Output:
#   θhat[k] : 1 × n
#             row 1 = rho(s)
# =========================================================

struct CNNStudy1{E, H}
    encoder::E
    head::H
    pool::Symbol
end

Flux.@functor CNNStudy1 (encoder, head)

function (m::CNNStudy1)(x)
    h = m.encoder(x)                                           # H × W × hidden × m
    h = m.pool == :sum ? sum(h, dims = 4) : mean(h, dims = 4)  # H × W × hidden × 1
    ρ = sigmoid.(m.head(h))                                    # H × W × 1 × 1

    return reshape(ρ, 1, :)
end

function (m::CNNStudy1)(X::AbstractVector)
    return map(x -> m(x), X)
end

function make_cnn_study1(;
    hidden::Int = 64,
    pool::Symbol = :sum,
)
    encoder = Chain(
        Conv((3, 3), 1 => 32, pad = 1),
        x -> relu.(x),
        Conv((3, 3), 32 => hidden, pad = 1),
        x -> relu.(x),
        Conv((3, 3), hidden => hidden, pad = 1),
        x -> relu.(x),
    )

    head = Chain(
        Conv((3, 3), hidden => hidden, pad = 1),
        x -> relu.(x),
        Conv((1, 1), hidden => 32),
        x -> relu.(x),
        Conv((1, 1), 32 => 1),
    )

    return CNNStudy1(encoder, head, pool)
end





# =========================================================
# CNN interval estimator for gridded local-isotropy sitewise rho
#
# Input:
#   x : H × W × 1 × m
#
# Output:
#   (
#       lower = 1 × n,
#       upper = 1 × n,
#   )
#
# Here rho(s) ∈ (0,1), so we construct interval on latent scale:
#
#   η_lower = head_lower(h)
#   η_upper = η_lower + softplus(head_width(h))
#
# then map both through sigmoid:
#
#   rho_lower = sigmoid(η_lower)
#   rho_upper = sigmoid(η_upper)
#
# This guarantees:
#   0 < rho_lower < rho_upper < 1
# =========================================================
using Flux
using Flux: Conv, Chain

struct CNNStudy1Interval{E, HL, HW}
    encoder::E
    head_lower::HL
    head_width::HW
    pool::Symbol
end

Flux.@functor CNNStudy1Interval (encoder, head_lower, head_width)

function (m::CNNStudy1Interval)(x)
    h = m.encoder(x)                                           # H × W × hidden × m
    h = m.pool == :sum ? sum(h, dims = 4) : mean(h, dims = 4)  # H × W × hidden × 1

    η_lower = m.head_lower(h)                                  # H × W × 1 × 1
    w_raw   = m.head_width(h)                                  # H × W × 1 × 1

    η_upper = η_lower .+ softplus.(w_raw)

    q_lower = sigmoid.(η_lower)
    q_upper = sigmoid.(η_upper)

    q_lower = reshape(q_lower, 1, :)                            # 1 × n
    q_upper = reshape(q_upper, 1, :)                            # 1 × n

    return vcat(q_lower, q_upper)                               # 2 × n
end

function (m::CNNStudy1Interval)(X::AbstractVector)
    return map(x -> m(x), X)                                    # Vector{Matrix}
end

function make_cnn_study1_interval(;
    hidden::Int = 64,
    pool::Symbol = :sum,
)
    encoder = Chain(
        Conv((3, 3), 1 => 32, pad = 1),
        x -> relu.(x),
        Conv((3, 3), 32 => hidden, pad = 1),
        x -> relu.(x),
        Conv((3, 3), hidden => hidden, pad = 1),
        x -> relu.(x),
    )

    head_lower = Chain(
        Conv((3, 3), hidden => hidden, pad = 1),
        x -> relu.(x),
        Conv((1, 1), hidden => 32),
        x -> relu.(x),
        Conv((1, 1), 32 => 1),
    )

    head_width = Chain(
        Conv((3, 3), hidden => hidden, pad = 1),
        x -> relu.(x),
        Conv((1, 1), hidden => 32),
        x -> relu.(x),
        Conv((1, 1), 32 => 1),
    )

    return CNNStudy1Interval(
        encoder,
        head_lower,
        head_width,
        pool,
    )
end