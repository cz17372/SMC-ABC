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


function b(x0::Vector{Float64},u0::Vector{Float64};gradFunc)
    n = normalize(gradient(gradFunc,x0))
    return u0 .- 2.0 * dot(u0,n) * n
end


function σ(x0,u0)
    return (x0,-u0)
end


function φ2(x0::Vector{Float64},u0::Vector{Float64},δ::Float64;gradFunc)
    x1,u1 = φ1(x0,u0,δ)
    xflip, uflip = σ(x1,u1)
    xb = xflip; ub = b(xflip,uflip,gradFunc=gradFunc)
    x2, u2 = φ1(xb,ub,δ)
    return x2,u2
end

function α1(x0,u0,δ)
    x1,u1 = φ1(x0,u0,δ)
    return min(0,logpi(x1)-logpi(x0))
end

function α2(x0,u0,δ)
    x2,u2 = φ2(x0,u0,δ,gradFunc=U)
    firstproprejectratio = log(1 - exp(α1(x2,u2,δ))) - log(1 - exp(α1(x0,u0,δ)))
    llkratio = logpi(x2) - logpi(x0)
    return min(0,firstproprejectratio + llkratio)
end


function DirectionRefresh(u0,κ)
    ind = rand(Bernoulli(κ))
    if ind == 0
        return u0
    else
        return normalize(rand(Normal(0,1),2))
    end
end

function BPS(N,x0,δ,κ)
    X = zeros(N,2)
    X[1,:] = x0
    u0 = normalize(rand(Normal(0,1),2))
    acc = 0 
    for n = 2:N
        x1,u1 = φ1(X[n-1,:],u0,δ)
        if C(x1) >= 0
            x2,u2 = φ2(X[n-1,:],u0,δ,gradFunc = C)
            if log(rand(Uniform(0,1))) < min(0,logpi(x2)-logpi(X[n-1,:]))
                xhat = x2
                uhat = u2
            else
                xhat = X[n-1,:]
                uhat = u0
            end
        else
            if log(rand(Uniform(0,1))) < α1(X[n-1,:],u0,δ)
                xhat = x1
                uhat = u1
            else
                x2,u2 = φ2(X[n-1,:],u0,δ,gradFunc=U)
                if log(rand(Uniform(0,1))) < α2(X[n-1,:],u0,δ)
                    xhat = x2
                    uhat = u2
                else
                    xhat = X[n-1,:]
                    uhat = u0
                end
            end
        end
        xhat,uhat = σ(xhat,uhat)
        u0 = DirectionRefresh(uhat,κ)
        X[n,:] = xhat
    end
    return X
end

end