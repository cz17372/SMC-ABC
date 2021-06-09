# This is an experiment on 20 data with RW-SMC-ABC method

using Distributions, Random

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
θ0 = [3.0,1.0,2.0,0.5];
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);

include("../RandomWalk/RW-SMC-ABC.jl")

EPSILON = Array{Any,1}(undef,30)
K       = Array{Any,1}(undef,30)
α       = Array{Any,1}(undef,30)
Theta   = Array{Any,1}(undef,30)

for i = 1:30
    R = RandomWalk.SMC(1000,250,dat20,Threshold=0.8,δ = 0.3,K0 = 5)
    EPSILON[i] = R.EPSILON
    K[i]       = R.K 
    α[i]       = R.AcceptanceProb
    Index      = findall(R.WEIGHT[:,end] .> 0)
    Theta[i]   = R.U[1:4,index,end]
end