using NeuralEstimators
using Flux
using NNlib: softplus, sigmoid

struct IntervalEstimator1{M, H} <: BayesEstimator
    network::M
    probs::H
end

Flux.@functor IntervalEstimator1 (network,)

function IntervalEstimator1(
    network;
    probs = [0.025f0, 0.975f0],
)
    if !isa(probs, AbstractArray)
        probs = [probs]
    end
    @assert all(0 .< probs .< 1)

    return IntervalEstimator1(network, probs)
end

function (est::IntervalEstimator1)(Z::AbstractVector)
    return est.network(Z)
end

function (est::IntervalEstimator1)(Z)
    return est.network(Z)
end