using Distributions: include
using Distributions, Plots, StatsPlots
using ForwardDiff: gradient

θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);

include("BPS/ExactBPS-SMC-ABC.jl")