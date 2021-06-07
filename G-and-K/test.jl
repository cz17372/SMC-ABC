using Distributions: include
using Distributions, Plots, StatsPlots
using ForwardDiff: gradient

θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);

include("BPS/ExactBPS-SMC-ABC.jl")

R = ExactBPS.SMC(1000,100,dat20,Threshold=0.8,δ=0.3,κ=2.0,K0=2,MH=ExactBPS.BPS1,Dist=Dist2)

index = findall(R.WEIGHT[:,end] .>0)
density(R.U[1,index,end])