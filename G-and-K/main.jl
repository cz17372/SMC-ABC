using Random, Distributions, JLD2, LinearAlgebra
# Transformation of standard normal RV's to g-and-k
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

# Set the true static parameters
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);
#=
# Random the MCMC sampler
include("G-and-K/MCMC/MCMC.jl")
@load "G-and-K/MCMC/MCMC_COV.jld2"
R_MCMC,α_MCMC = RWM(30000,Σ,0.2,y = dat20,θ0 = rand(Uniform(0,10),4))
plot(R_MCMC[:,4],linewidth=2.0,color=:grey)
density(R_MCMC[15001:end,3],linewidth=2.0,color=:darkgreen,label="")

=#

include("Langevin/Langevin-SMC-ABC.jl")
include("BPS/BPS-SMC-ABC.jl")
include("RandomWalk/RW-SMC-ABC.jl")
#include("MCMC/MCMC.jl")
#@load "MCMC/MCMC_COV.jld2"