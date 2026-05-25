using Flux
using Flux: Conv, Chain, Adam
using Statistics
using LinearAlgebra
using Random
using CairoMakie


# =========================================================
# 3) model: replicate-invariant CNN
# =========================================================

# output constraints:
# channel 1,2 -> sigmoid for rho1/rho2 in (0,1)
# channel 3   -> tanh for delta in (-1,1)
function constrain_surfaces(Yraw)
    rho1 = sigmoid.(Yraw[:, :, 1:1, :])
    rho2 = sigmoid.(Yraw[:, :, 2:2, :])
    delta = tanh.(Yraw[:, :, 3:3, :])
    return cat(rho1, rho2, delta; dims=3)
end

struct SurfaceNet
    encoder
    head
end

Flux.@functor SurfaceNet

function (m::SurfaceNet)(x)
    # x: H×W×3×R
    h = m.encoder(x)         # H×W×C×R
    hbar = mean(h, dims=4)   # H×W×C×1
    yraw = m.head(hbar)      # H×W×3×1
    return constrain_surfaces(yraw)
end

function make_surface_net(; in_ch=3, hidden=64)
    encoder = Chain(
        Conv((3,3), in_ch => 32, pad=1),
        x -> relu.(x),
        Conv((3,3), 32 => hidden, pad=1),
        x -> relu.(x),
        Conv((3,3), hidden => hidden, pad=1),
        x -> relu.(x),
    )

    head = Chain(
        Conv((3,3), hidden => hidden, pad=1),
        x -> relu.(x),
        Conv((1,1), hidden => 32),
        x -> relu.(x),
        Conv((1,1), 32 => 3),
    )

    return SurfaceNet(encoder, head)
end

# =========================================================
# 4) loss
# =========================================================

function smoothness_penalty(Y)
    p1 = mean(abs2, Y[2:end, :, :, :] .- Y[1:end-1, :, :, :])
    p2 = mean(abs2, Y[:, 2:end, :, :] .- Y[:, 1:end-1, :, :])
    return p1 + p2
end

function surface_loss(model, x, y; λ_smooth=1f-4)
    ŷ = model(x)
    mse = mean(abs2, ŷ .- y)
    smooth = smoothness_penalty(ŷ)
    return mse + λ_smooth * smooth
end

function dataset_loss(model, X_list, Y_list; λ_smooth=0f0)
    vals = map(eachindex(X_list)) do i
        surface_loss(model, X_list[i], Y_list[i]; λ_smooth=λ_smooth)
    end
    return mean(vals)
end




