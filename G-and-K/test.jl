include("main.jl")
using Plots,StatsPlots, JLD2, LinearAlgebra
R = BPS.SMC(1000,100,dat20,Threshold=0.8,δ=0.5,κ=0.7,K0=10,MH=BPS.BPS2)
index = findall(R.DISTANCE[:,end] .> 0)
density(R.U[1,index,end])
plot(log.(R.EPSILON))


@load "C:/Users/chang/OneDrive/Documents/BPS2_Thres080_Stepsize045_Refresh060.jld2"
R = BPS2_Thres080_Stepsize045_Refresh060

ESS(x) = length(findall(x .> 0))

plot(log.(R.EPSILON))
index = findall(R.DISTANCE[:,end] .> 0)
density(R.U[2,index,end])
plot(R.BoundaryBounceProposed ./(mapslices(ESS,R.WEIGHT,dims=1)[1,2:end] .* R.K[1:end-1]),label="")

@load "MCMC/MCMC_COV.jld2"
MCMC,α = RWM(200000,Σ,0.2,y=dat20,θ0=rand(Uniform(0,10),4))


density(MCMC[50001:end,1])
density!(R.U[1,index,end])

anim = @animate for i = 1:301
    index = findall(R.WEIGHT[:,i] .> 0)
    density(R.U[2,index,i],label="")
    density!(MCMC[50001:end,2],label="")
end

gif(anim,fps=4)