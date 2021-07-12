using ForwardDiff: derivative
using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra
theme(:wong2)
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
Random.seed!(123)
z0 = rand(Normal(0,1),100)
θ0 = [3.0,1.0,2.0,0.5]
ystar = f.(z0,θ=θ0)
include("RandomWalk/RWABCSMC.jl")

@load "data/1000data_RW_2000Particles1.jld2"
R = Results

t = 4
density(R.U[1][t,:],label="")
for n = 2:length(R.U)
    density!(R.U[n][t,:],label="")
end
current()

a = rand(Normal(0,1),10)
f(x) = norm(cdf(Normal(0,1),x))

gradient(f,a)

plot(log.(R.EPSILON[1]),label="")
for n = 2:length(R.U)
    plot!(log.(R.EPSILON[n]))
end
current()