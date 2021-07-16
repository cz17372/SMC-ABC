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
include("RandomWalk/RW2.jl")


@load "RWCompCost.jld2"
plot(Information.tolerance_vec,Information.CompCostVec,label="RW-ABC-SMC",xlabel="Terminal Tolerance",ylabel="Log(NoMCMC/ESS)");
scatter!(Information.tolerance_vec,Information.CompCostVec,markershape=:circle,markersize=5.0,color=:darkolivegreen,label="");
@load "StdCompCost.jld2"
plot!(Information.tolerance_vec,Information.CompCostVec[1:10],label="Std-ABC-SMC",xlabel="Terminal Tolerance",ylabel="Log(NoMCMC/ESS)");
scatter!(Information.tolerance_vec,Information.CompCostVec[1:10],markershape=:circle,markersize=5.0,color=:red,label="")



@load "RWCompCost.jld2"
plot(Information.tolerance_vec, Information.SMCTimeVec,xlabel="Terminal Tolerance",ylabel="No SMC Steps",label="RW-ABC-SMC")
R = RW.SMC(2000,ystar,η = 0.8)
Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
density(10*X[4,:])
