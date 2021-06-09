using Base: task_result
using Random, Distributions, JLD2, LinearAlgebra
using Plots,StatsPlots
# Transformation of standard normal RV's to g-and-k
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
# Set the true static parameters
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);
include("Langevin/Langevin-SMC-ABC.jl")
include("BPS/BPS-SMC-ABC.jl")
include("RandomWalk/RW-SMC-ABC.jl")


@load "20data_RW_Results.jld2" Results

U = Results.Theta

density(U[1][4,:],label="",linewidth=0.5,color=:grey)
for n = 2:30
    density!(U[n][4,:],label="",linewidth=0.5,color=:grey)
end
current()

plot(log.(Results.EPSILON[1]),label="")
for n = 2:30
    plot!(log.(Results.EPSILON[n]),label="")
end
current()
plot(Results.K[1],label="")
for n = 2:30
    plot!(Results.K[n],label="")
end
current()

