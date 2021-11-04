module gku
using Distributions, LinearAlgebra
NoParam = 4

f(z,θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

"""
    ptheta(θ)
Calculate the log-prior of the static parameters under the model

# Arguments
- `θ::Vector{Float}`: The parameter vectors whose log-prior will be calculated
"""
function ptheta(θ)
    return sum(logpdf.(Uniform(0,10),θ))
end

"""
    U(u)
Calculate the log-prior of the latents variables

# Arguments
- `u::Vector{Float}`: The latents whose joint log-prior will be calculated
"""
function U(u)
    return sum(logpdf(Uniform(0,1),u))
end



function GenPar()
    return rand(Uniform(0,10),4)
end

function GenSeed(N)
    return rand(N)
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

function Ψ(u)
    θ = 10 * u[1:4]
    return f.(quantile(Normal(0,1),u[5:end]),Ref(θ))
end


function ϕ(u,θ)
    z = quantile(Normal(0,1),u)
    return f.(z,Ref(θ))
end

function transform(x)
    return 10*x
end

end