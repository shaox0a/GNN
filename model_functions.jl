function simulatebrownresnick(obj::M, m::Integer; kwargs...) where {M <: AbstractMatrix{T}} where {T <: Number}
    y = [simulatebrownresnick(obj; kwargs...) for _ ∈ 1:m]
    y = stack(y)
    return y
end


function simulatebrownresnick(
    obj::M;
    C = 20.0,
    Gumbel::Bool = false,
    σ² = nothing,
) where {M <: AbstractMatrix{T}} where {T <: Number}

    n = size(obj, 1)

    Z = fill(zero(T), n)

    ζ⁻¹ = randexp(T)
    ζ = one(T) / ζ⁻¹

    # Brown--Resnick spectral process:
    #
    #     W(s) = exp{Y(s) - Var(Y(s))/2}
    #
    # This ensures E[W(s)] = 1, which is required for unit Fréchet margins.
    #
    # If simulategaussian(obj) returns a Gaussian field with marginal variance 1,
    # then Var(Y(s)) = 1 for all s, and the correction is simply -1/2.
    #
    # More generally, σ² can be supplied as a length-n vector containing the
    # marginal variances of Y at the observed sites.

    if isnothing(σ²)
        σ²_vec = fill(one(T), n)
    elseif σ² isa Number
        σ²_vec = fill(T(σ²), n)
    else
        length(σ²) == n || error("σ² must have length n = $n")
        σ²_vec = T.(σ²)
    end

    while (ζ * T(C)) > minimum(Z)
        Y = simulategaussian(obj)

        W = exp.(Y .- T(0.5) .* σ²_vec)

        Z = max.(Z, ζ .* W)

        E = randexp(T)
        ζ⁻¹ += E
        ζ = one(T) / ζ⁻¹
    end

    # Log transform from unit Fréchet scale to Gumbel scale
    if Gumbel
        Z = log.(Z)
    end

    return Z
end