using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra
theme(:wong2)
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
Random.seed!(123)
z0 = rand(Normal(0,1),250)
θ0 = [3.0,1.0,2.0,0.5]
ystar = f.(z0,θ=θ0)
cd("G-and-K"); 
include("RandomWalk/RWABCSMC.jl")

R = RWABCSMC.SMC(1000,ystar)

Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
density(X[2,:])
