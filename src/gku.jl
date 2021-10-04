module gku
using Distributions, LinearAlgebra
pardim = 4
f(z,θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

function ptheta(θ)
    return sum(logpdf.(Uniform(0,10),θ))
end
function U(u)
    return sum(logpdf(Uniform(0,1),u))
end

function GenPar()
    return rand(Uniform(0,10),4)
end

function Simulator(N)
    θ = 10*rand(4)
    z = randn(N)
    return f.(z,Ref(θ))
end
function ConSimulator(N,θ)
    z = randn(N)
    return f.(z,Ref(θ))
end



function genseed(N)
    return rand(N)
end
function ϕ(u)
    θ = 10 * u[1:4]
    return f.(quantile(Normal(0,1),u[5:end]),Ref(θ))
end
function g(u,θ)
    z = quantile(Normal(0,1),u)
    return f.(z,Ref(θ))
end


end