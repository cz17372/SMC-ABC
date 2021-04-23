using Plots, StatsPlots, Random, Distributions, JLD2
theme(:ggplot2)
# Transformation of standard normal RV's to g-and-k
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

# Set the true static parameters
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123)
dat20 = f.(rand(Normal(0,1),20),θ=θ0)

# Random the MCMC sampler
include("G-and-K/MCMC/MCMC.jl")
@load "G-and-K/MCMC/MCMC_COV.jld2"
R_MCMC,α_MCMC = RWM(30000,Σ,0.2,y = dat20,θ0 = rand(Uniform(0,10),4))
plot(R_MCMC[:,4],linewidth=2.0,color=:grey)
density(R_MCMC[15001:end,3],linewidth=2.0,color=:darkgreen,label="")


# Random-walk SMC-ABC
ystar = dat20
include("G-and-K/RandomWalk/RW-SMC-ABC.jl")
R_RW  = RW_SMC_ABC(1000,100,20,Threshold=0.8,δ=0.1,K=50)
ESS(x) = length(findall(x .> 0))

plot(mapslices(ESS, R_RW.WEIGHT, dims = 1)[1,:])
plot!(mapslices(uni,R_RW.DISTANCE,dims=1)[1,:])
plot(log.(R_RW.EPSILON))
t0 = 301; n= 3
index = findall(R_RW.DISTANCE[:,t0] .> 0)
density(R_RW.U[n,index,t0])
density!(R_MCMC[150001:end,n])

Σ = [1 2;2 5]
L = cholesky(Σ).L
X = transpose(L *rand(Normal(0,1),2,10000))
X2 = transpose(rand(MultivariateNormal([0,0],Σ),10000))