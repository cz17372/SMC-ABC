using Distributions, Random, LinearAlgebra, Plots, StatsPlots
include("src/RW.jl")
include("src/RESMC.jl")
# Lotka-Volterra Model 
Dist(x,y) = norm(x .- y)
# Define the simulator
function f(u,θ)
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
function ϕ(u::Vector{Float64})
    θ = -2.0 .+ 3.0*u[1:4]
    return f(u[5:end],θ)
end

Random.seed!(1550)
θstar = log.([0.4,0.005,0.05,0.001]);
ustar = randn(100)
ystar = f(ustar,θstar)

R = RW.SMC(1000,ystar,length(ystar)+4,ϕ,Dist;TerminalTol=10.0,η = 0.8)

R = RESMC.SMC(10000,ystar,θstar=log.([0.4,0.005,0.05,0.001]),g=f,Dist=Dist,TerminalTol=20.0,gc=false,PrintRes=true,MT = false)



f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
function ϕ(u)
    θ = 10.0*cdf(Normal(0,1),u[1:4])
    return f.(u[5:end],θ=θ)
end
Random.seed!(123)
θstar = [3.0,1.0,2.0,0.5]
u = randn(20)
ystar = f.(u,θ=θstar)
R = RW.SMC(5000,ystar,length(ystar)+4,ϕ,Dist;TerminalTol=5.0,η = 0.8)
Σ = cov(R.U[end][1:4,findall(R.WEIGHT[:,end] .>0)],dims=2)
function g(x,θ)
    z = quantile(Normal(0,1),x)
    return f.(z,θ=θ)
end
R = RESMC.SMC(1000,θstar,y=ystar,g=g,Dist=Dist,TerminalTol=1.0,PrintRes=true,MT = false)

include("G-and-K/src/RE-SMC.jl")
R = RESMC.PMMH(θstar,1000,5000,y=ystar,g=g,Dist=Dist,ϵ=5.0,Σ=Σ,PR=false)

plot(R.theta[:,1])