module BouncyParticleSamplerDemo

using Distributions, LinearAlgebra
using ForwardDiff: gradient

# A demo is built to sample from N(μ,Σ) where μ = [0,0] and Σ = [1 2;2 5]
# Constraints is defined to be |x| < 1
Σ = [1 2; 2 5]
μ = [0,0]

# The constrained region will be {x ∈ R^2: C(x) < 0} 
C(x) = norm(x) - 1.0

# define the constraned log-density of the normalized/unnormalized target 
function logpi(x)
    if C(x) < 0
        return logpdf(MultivariateNormal(μ,Σ),x)
    else
        return -Inf
    end
end

# Define the energy function within the constrained region
U(x) = -logpdf(MultivariateNormal(μ,Σ),x)


"""
φ1(x0::Vector{Float64},u0::Vector{Float64},δ::Float64)
    
Involution performing the position update proposal. It returns φ1(x,u) := (x + δu, -u) 

# Arguments
- `x0::Vector{Float64}`: the current position of the target. 
- `u0::Vector{Float64}`: the current velocity the BPS has
- `δ::Float64`: step size of the proposal

"""
function φ1(x0::Vector{Float64},u0::Vector{Float64},δ::Float64)
    return (x0 .+ δ*u0, -u0)
end
export φ1

function b(x0::Vector{Float64},u0::Vector{Float64};U)
    n = normalize(gradient(U,x0))
    return u0 .- 2.0 * dot(u0,n) * n
end
export b
function b_BC(x0::Vector{Float64},u0::Vector{Float64};C)
    n = normalize(gradient(C,x0))
    return u0 .- 2.0 * dot(u0,n) * n
end
export b_BC
end

