module gku
using Distributions, LinearAlgebra

f(z,θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

function ConSimulator(N,θ)
    z = randn(N)
    return f.(z,Ref(θ))
end

function Simulator(N)
    θ = 10*rand(4)
    z = randn(N)
    return f.(z,Ref(θ))
end


function ϕ(u)
    θ = 10 * u[1:4]
    return f.(quantile(Normal(0,1),u[5:end]),Ref(θ))
end

function U(u)
    if all(0.0 .< u .< 1.0)
        return 0
    else
        return -Inf
    end
end

function genseed(N)
    return rand(N)
end

function g(u,θ)
    z = quantile(Normal(0,1),u)
    return f.(z,Ref(θ))
end

end