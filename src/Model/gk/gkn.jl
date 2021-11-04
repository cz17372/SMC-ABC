module gkn
using Distributions, LinearAlgebra
NoParam = 4
f(z,θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;


function ptheta(θ)
    return sum(logpdf.(Uniform(0,10),θ))
end
U(u) = sum(logpdf(Normal(0,1),u))



function GenPar()
    return rand(Uniform(0,10),4)
end

function GenSeed(N)
    return randn(N)
end
# Simulators
function Simulator(N)
    θ = 10*rand(4)
    z = randn(N)
    return f.(z,Ref(θ))
end

function ConSimulator(N,θ)
    z = randn(N)
    return f.(z,Ref(θ))
end




function Ψ(u)
    θ = 10 * cdf(Normal(0,1),u[1:4])
    return f.(u[5:end],Ref(θ))
end
function ϕ(u,θ)
    return f.(u,Ref(θ))
end

function transform(u)
    return 10*cdf(Normal(0,1),u)
end
end