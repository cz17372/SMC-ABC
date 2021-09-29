module gkn
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
    θ = 10 * cdf(Normal(0,1),u[1:4])
    return f.(u[5:end],Ref(θ))
end

U(u) = sum(logpdf(Normal(0,1),u))

function genseed(N)
    return randn(N)
end

function g(u,θ)
    return f.(u,Ref(θ))
end

end