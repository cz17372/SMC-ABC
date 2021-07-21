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
include("src/RW.jl")
include("src/MCMC.jl")

R,alpha = RWM(10000,1.0*I,0.2)
Σ = cov(R)
R,alpha = RWM(100000,Σ,0.2)


@load "data/100data_RW5000Particles1.jld2"
R = Results
@load "data/100Data_RW5000Particles2.jld2"
R2 = Results
@load "data/100data_RW5000Particles3.jld2"
R3 = Results


i = 4
density(10*R.U[1][i,:],label="",color=:darkolivegreen,size=(500,500));
for n = 2:length(R.U)
    density!(10*R.U[n][i,:],label="",color=:darkolivegreen);
end
for n = 1:length(R2.U)
    density!(10*R2.U[n][i,:],label="",color=:red);
end
for n = 1:length(R3.U)
    density!(10*R3.U[n][i,:],label="",color=:blue);
end
current()

@load "data/100data_RW2000Particles1.jld2"

R20data = reset_defaults
i = 3
density(10*R20data.U[1][i,:],label="",color=:darkolivegreen,size=(500,500));
for n = 2:length(R20data.U)
    density!(10*R20data.U[n][i,:],label="",color=:darkolivegreen);
end
current()