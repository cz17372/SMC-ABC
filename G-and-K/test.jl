using ForwardDiff: derivative
using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra
theme(:wong2)
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
density(ystar)
include("RandomWalk/RW2.jl")

@load "RWCompCost.jld2"
plot(Information.tolerance_vec,Information.CompCostVec)
scatter!(Information.tolerance_vec,Information.CompCostVec,markershape=:circle)