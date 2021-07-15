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

R=  RW.SMC(2000,ystar,η = 0.95,TerminalTol=0.2,MinStep=0.1,MaxStep=5.0)
Index = findall(R.WEIGHT[:,end] .> 0)
X2 = R.U[end][:,Index]
density(10*X2[3,:])
@load "timevec.jld2"
@load "deltimevec.jld2"

tol_vec = [1.0,3.0,5.0,10.0,15.0,20.0,25.0]
tol_delvec = [16.5,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0]
tol_delvec_20data = collect(15:25)

DelNoMCMC_20data = zeros(length(tol_delvec_20data))
for i = 1:length(DelNoMCMC_20data)
    R = DelMoralSMCABC.SMC(1000,ystar,TerminalProb=0.00,TerminalTol=tol_delvec_20data[i],η = 0.8)
    DelNoMCMC_20data[i] = log.(sum(R.K .* R.ESS[2:end])/R.ESS[end])
end

RWNoMCMC_20data = zeros(length(tol_vec))
for i = 1:length(timevec)
    R = RW.SMC(1000,ystar,TerminalProb=0.00,TerminalTol=tol_vec[i],η = 0.8)
    RWNoMCMC_20data[i] = log.(sum(R.K .* R.ESS[2:end])/R.ESS[end])
end


plot(tol_vec,RWNoMCMC_100data,label="RW-ABC-SMC",xlabel="Terminal Tolerance",ylabel="Log(NoMCMCSteps/ESS)")
scatter!(tol_vec,RWNoMCMC_100data,label="",markershape=:circle,color=:darkolivegreen,markersize=5.0)
plot!(tol_delvec,DelNoMCMC_100data,label="Std-ABC-SMC")
scatter!(tol_delvec,DelNoMCMC_100data,markershape=:circle,markersize=5.0,color=:red,label="")
@load "RWNoMCMC_100data.jld2" RWNoMCMC_100data
@load "DelNoMCMC_100data.jld2" DelNoMCMC_100data


plot(tol_vec,RWNoMCMC_20data,label="RW-ABC-SMC",xlabel="Terminal Tolerance",ylabel="Log(NoMCMCSteps/ESS)");
scatter!(tol_vec,RWNoMCMC_20data,label="",markershape=:circle,color=:darkolivegreen,markersize=5.0);
plot!(tol_delvec_20data,DelNoMCMC_20data,label="Std-ABC-SMC");
scatter!(tol_delvec_20data,DelNoMCMC_20data,markershape=:circle,markersize=5.0,color=:red,label="")


plot(tol_vec,RWNoMCMC_100data,label="RW-ABC-SMC",xlabel="Terminal Tolerance",ylabel="Log(NoMCMCSteps/ESS)");
scatter!(tol_vec,RWNoMCMC_100data,label="",markershape=:circle,color=:darkolivegreen,markersize=5.0);
plot!(tol_delvec,DelNoMCMC_100data,label="Std-ABC-SMC");
scatter!(tol_delvec,DelNoMCMC_100data,markershape=:circle,markersize=5.0,color=:red,label="")


plot(tol_vec,RWNoMCMC_100data,label="100 observations",xlabel="Terminal Tolerance",ylabel="Log(NoMCMCSteps/ESS)");
scatter!(tol_vec,RWNoMCMC_100data,label="",markershape=:circle,color=:darkolivegreen,markersize=5.0);
plot!(tol_vec,RWNoMCMC_20data,label="25 observations",xlabel="Terminal Tolerance",ylabel="Log(NoMCMCSteps/ESS)");
scatter!(tol_vec,RWNoMCMC_20data,label="",markershape=:circle,color=:darkolivegreen,markersize=5.0)

SMC_Length = zeros(length(tol_vec))
for i = 1:length(SMC_Length)
    R = RW.SMC(1000,ystar,TerminalProb=0.00,TerminalTol=tol_vec[i],η = 0.8)
    RWNoMCMC_20data[i] = log.(sum(R.K .* R.ESS[2:end])/R.ESS[end])
    SMC_Length[i] = length(R.ESS)-1
    GC.gc()
end

plot(tol_vec,SMC_Length)

