using Distributions,LinearAlgebra,ProgressMeter, Random
using ForwardDiff: gradient

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

function dist(ξ)
    return sum((f.(ξ[5:end],θ=ξ[1:4]) .- ystar).^2)
end

C(ξ;ϵ) = ϵ - dist(ξ)


function logpi(ξ;ϵ)
    if C(ξ,ϵ=ϵ) > 0
        logpdf_θ = sum(logpdf.(Uniform(0,10),ξ[1:4]))
        logpdf_z = sum(logpdf.(Normal(0,1),ξ[5:end]))
        return logpdf_θ + logpdf_z
    else
        return -Inf
    end
end

function RWMH(N,x0;ϵ,Σ,δ)
    D = length(x0)
    X = zeros(N,length(x0))
    X[1,:] = x0
    for n = 2:N
        xcand = rand(MultivariateNormal(X[n-1,:],δ^2*Σ))
        α = min(0,logpi(xcand,ϵ=ϵ)-logpi(X[n-1,:],ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
        else
            X[n,:] = X[n-1,:]
        end
    end
    return X[end,:]
end