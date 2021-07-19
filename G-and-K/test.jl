using Plots: reset_defaults
using ForwardDiff: derivative
using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra
theme(:ggplot2)
function ϕ(u)
    θ = 10.0*u[1:4]
    z = quantile(Normal(0,1),u[5:end])
    return f.(z,θ=θ)
end
function f(u;θ)
    z = quantile(Normal(0,1),u)
    return θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^(θ[4])*z
end
Random.seed!(123)
θ0 = [3.0,1.0,2.0,0.5];
u0 = rand(100)
ystar = f.(u0,θ=θ0)
include("RandomWalk/RW2.jl")
R2 = RW.SMC(5000,ystar,η=0.95,TerminalTol=1.0)
Epsilon = 1.0
T = findfirst(R2.EPSILON .< Epsilon)
Index = findall(R2.WEIGHT[:,T] .> 0)
X = R2.U[T][:,Index]
include("MCMC/MCMC.jl")

R,alpha = RWM(10000,1.0*I,0.2)
Σ = cov(R)
R,alpha = RWM(100000,Σ,0.2)

plot(R[:,4])

density(R[50001:end,1])
density!(10*X[1,:])

@load "try.jld2"

R = Results 
X = R.U[1]

density(10*X[4,:])
