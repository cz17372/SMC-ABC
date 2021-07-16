using Distributions, LinearAlgebra, Random, JLD2

# Sample the "true" observations with seed "123"
# No. of observations = 100
function f(u;θ)
    z = quantile(Normal(0,1),u)
    return θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^(θ[4])*z
end
Random.seed!(123);
θ0 = [3.0,1.0,2.0,0.5];
u0 = rand(100);
ystar = f.(u0,θ=θ0)


include("SMC-ABC(Del Morel)/DelMoralABCSMC.jl")

tolerance_vec = collect(16:25)
CompCostVec = zeros(10)
SMCTimeVec  = zeros(10)



for i = 1:length(SMCTimeVec)
    R = DelMoralSMCABC.SMC(2000,ystar,η = 0.9,TerminalTol=tolerance_vec[i],MinStep=0.1,TerminalProb=0.0)
    CompCostVec[i] = log.(sum(R.K .* R.ESS[2:end])/R.ESS[end])
    SMCTimeVec[i]  = length(R.ESS) - 1
end

Information = (NoParticles = 2000, η = 0.9, ystar = ystar, tolerance_vec=tolerance_vec, MinStep = 0.1,CompCostVec=CompCostVec,SMCTimeVec=SMCTimeVec)
