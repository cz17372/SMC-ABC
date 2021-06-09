using Distributions
using Distributions, Plots, StatsPlots
using ForwardDiff: gradient
using Random

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);

include("BPS/ExactBPS-SMC-ABC.jl")

R = ExactBPS.SMC(1000,250,dat20,Threshold=0.8,δ=0.1,κ=2.0,K0=2,MH=ExactBPS.BPS1,Dist=ExactBPS.Dist2)

index = findall(R.WEIGHT[:,100] .> 0)
density(R.U[1,index,100])


include("RandomWalk/RW-SMC-ABC.jl")

R = RandomWalk.SMC(1000,250,dat20,Threshold=0.8,δ=0.3,K0=5)

index = findall(R.WEIGHT[:,end] .> 0)
density(R.U[2,index,end])