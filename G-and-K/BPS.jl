using LinearAlgebra, Distributions, StatsPlots, Plots, Roots
using ForwardDiff:gradient
using Random
C(x) = norm(x) - 2.0
u0 = normalize(rand(Normal(0,1),2))
x0 = [0.2,0.3]
δ  = 3.0

function logpi(x)
    if C(x) < 0
        return logpdf(MultivariateNormal(μ,Σ),x)
    else
        return -Inf
    end
end

function BoundaryBounce(x0,δ,u0)
    output = x0
    working_delta = δ
    intermediate_x = copy(x0)
    intermediate_u = copy(u0)
    while C(intermediate_x .+ working_delta*intermediate_u) >= 0
        obj(k) = C(intermediate_x .+ k*intermediate_u) 
        k = find_zero(obj,working_delta)
        intermediate_x = intermediate_x .+ k*intermediate_u
        output = hcat(output,intermediate_x)
        working_delta -= k
        n = normalize(gradient(C,intermediate_x))
        intermediate_u = intermediate_u .- 2*dot(intermediate_u,n)*n
    end
    output = hcat(output,intermediate_x .+ working_delta*intermediate_u)
    return output,intermediate_u
end


function BoundaryBounce(x0,δ,u0)
    output = x0
    working_delta = δ
    intermediate_x = copy(x0)
    intermediate_u = copy(u0)
    while C(intermediate_x .+ working_delta*intermediate_u) >= 0
        obj(k) = C(intermediate_x .+ k*intermediate_u) 
        k = find_zero(obj,working_delta)
        intermediate_x = intermediate_x .+ k*intermediate_u
        output = hcat(output,intermediate_x)
        working_delta -= k
        n = normalize(gradient(C,intermediate_x))
        intermediate_u = intermediate_u .- 2*dot(intermediate_u,n)*n
    end
    output = hcat(output,intermediate_x .+ working_delta*intermediate_u)
    return output[:,end],intermediate_u
end

Σ = [1 2; 2 5]
μ = [0,0]
U(x) = -logpdf(MultivariateNormal(μ,Σ),x)

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

function DirectionRefresh(u0,δ,κ)
    p = exp(-κ*δ)
    ind = rand(Bernoulli(p))
    if ind == 1
        return u0
    else
        return normalize(rand(Normal(0,1),length(u0)))
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
            x2,u2 = BoundaryBounce2(X[n-1,:],δ,u0)
            if log(rand(Uniform(0,1))) < min(0,logpi(x2)-logpi(X[n-1,:]))
                acc += 1
                xhat = x2
                uhat = u2
            else
                xhat = X[n-1,:]
                uhat = u0
            end
        else
            if log(rand(Uniform(0,1))) < α1(X[n-1,:],u0,δ)
                acc += 1
                xhat = x1
                uhat = u1
            else
                x2,u2 = φ2(X[n-1,:],u0,δ,gradFunc=U)
                if log(rand(Uniform(0,1))) < α2(X[n-1,:],u0,δ)
                    acc += 1
                    xhat = x2
                    uhat = u2
                else
                    xhat = X[n-1,:]
                    uhat = u0
                end
            end
        end
        xhat,uhat = σ(xhat,uhat)
        u0 = DirectionRefresh(uhat,δ,κ)
        X[n,:] = xhat
    end
    return X,acc/(N-1)
end


X,acc = BPS(100000,x0,0.5,2.0)
plot(X[:,1],X[:,2])

density(X[50001:end,2])