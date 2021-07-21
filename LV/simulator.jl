using Distributions: size
using Distributions, Plots, StatsPlots
using Random
using LinearAlgebra
using JLD2, Plots
using ForwardDiff:gradient
theme(:ggplot2)
include("DelMoralABCSMC.jl")
include("RW2.jl")
function f(u;θ)
    θ = exp.(θ)
    N = length(u) ÷ 2
    r0 = 100.0; f0 = 100.0; dt = 1.0; σr = 1.0; σf = 1.0; 
    rvec = zeros(N+1)
    fvec = zeros(N+1)
    rvec[1] = r0; fvec[1] = f0
    for n = 1:N
        rvec[n+1] = max(rvec[n] + dt*(θ[1]*rvec[n]-θ[2]*rvec[n]*fvec[n]) + sqrt(dt)*σr*u[2*n-1],0)
        fvec[n+1] = max(fvec[n] + dt*(θ[4]*rvec[n]*fvec[n]-θ[3]*fvec[n]) + sqrt(dt)*σf*u[2*n],0)
    end
    return [rvec[2:end];fvec[2:end]]
end
function SimulateOne(N)
    u = rand(4+N)
    return ϕ(u)
end

function ϕ(u)
    θ = -2.0 .+ 3.0*quantile(Normal(0,1),u[1:4])
    z = quantile(Normal(0,1),u[5:end])
    return f(z,θ=θ)
end

Eucliean(u;y) = norm(ϕ(u) .- y)
Random.seed!(17372);
θstar = log.([0.4,0.005,0.05,0.001]);
uθ = cdf(Normal(-2,3.0),θstar)
uz = rand(100)
ustar = [uθ;uz]
ystar = ϕ(ustar)

grad(u) = gradient(u->Eucliean(u,y=ystar),u)
R = RW.SMC(3000,ystar,InitStep= 0.3,MinStep=0.1,η =0.8,TerminalTol=1.0)
R2 = RW.SMC(5000,ystar,InitStep= 0.3,MinStep=0.1,η =0.8,TerminalTol=1.0)
Index = findall(R.WEIGHT[:,end].>0)
X = R.U[end][:,Index]
R2 = DelMoral.SMC(1000,ystar,InitStep=0.3,MinStep=0.2,MinProb=0.2,IterScheme="Fixed",InitIter=2,PropParMoved=0.99,TolScheme="unique",η=0.95,TerminalTol=0.1,TerminalProb=0.01)

Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
Index2 = findall(R2.WEIGHT[:,end] .> 0)
X2 = R2.U[end][:,Index2]
density(X[4,:]*3 .- 2)
density!(X2[4,:]*3 .- 2)
density(quantile(Normal(-2,3.0),X[1,:]))
log(0.4)

density(X[1,:])

@load "try.jld2"

X = Results.U[1]

data = zeros(size(X)[2],100)
for i = 1:size(X)[2]
    data[i,:] = ϕ(X[:,i])
end

plot(data[1,1:50],label="",color=:grey,linewidth=0.1)
for n = 2:size(data)[1]
    plot!(data[n,1:50],label="",color=:grey,linewidth=0.1)
end
for n = 1:size(data)[1]
    plot!(data[n,51:end],label="",color=:grey,linewidth=0.1)
end
current()