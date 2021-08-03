using ForwardDiff: derivative, gradient
using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra

theme(:ggplot2)
function ϕ(u)
    θ = 10.0*u[1:4]
    z = quantile(Normal(0,1),u[5:end])
    return f.(z,θ=θ)
end
function f(z;θ)
    return θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^(θ[4])*z
end
Euclidean(u;y) = norm(ϕ(u) .- y)
Random.seed!(123)
θ0 = [3.0,1.0,2.0,0.5];
u0 = rand(20) 
z0 = quantile(Normal(0,1),u0)
ystar = f.(z0,θ=θ0)



grad(u) = normalize(gradient(u->Euclidean(u,y=ystar),u))

include("src/RW.jl")
include("src/MCMC.jl")

R,alpha = RWM(10000,1.0*I,0.2)
Σ = cov(R)
R,alpha = RWM(100000,Σ,0.2)


R = RW.SMC(10000,ystar,η = 0.8,InitStep=0.3,MinStep=0.1,MinProb=0.2,TerminalTol=1.0)
Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
theta= 10*cdf(Normal(0,1),X[1:4,:])

density(theta[1,:])
density(10*X[2,:])
Σ = cov(theta,dims=2)
density(10*cdf(Normal(0,1),X[4,:]))

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

density!(10*cdf(Normal(0,1),X[i,:]),label="",linewidth=5.0,color=:yellow)

@load "data/100data_RW2000Particles1.jld2"

R20data = reset_defaults
i = 3
density(10*R20data.U[1][i,:],label="",color=:darkolivegreen,size=(500,500));
for n = 2:length(R20data.U)
    density!(10*R20data.U[n][i,:],label="",color=:darkolivegreen);
end
current()

plot(log.(R.EPSILON))

plot(R.StepSize)

R = load("data/S100data_RW5000ParticlesNew.jld2","Results")
function PlotBatchRes(R,i;color=:grey,linewidth=0.1,newplot=true,size=(400,400),xlabel="",ylabel="",label="")
    if newplot
        density(R.U[1][i,:],color=color,linewidth=linewidth,size=size,xlabel=xlabel,ylabel=ylabel,label=label)
        for n = 2:length(R.U)
            density!(R.U[n][i,:],color=color,linewidth=linewidth,label="")
        end
        current()
    else
        density!(R.U[1][i,:],color=color,linewidth=linewidth,label=label)
        for n = 2:length(R.U)
            density!(R.U[n][i,:],color=color,linewidth=linewidth,label="")
        end
        current()
    end
end

PlotBatchRes(R,1,color=:black)