include("main.jl")
using Plots,StatsPlots, JLD2, LinearAlgebra
R = BPS.SMC(1000,200,dat20,Threshold=0.8,δ=0.5,κ=0.7,K0=10,MH=BPS.BPS2)
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


samp() = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
TrueX = zeros(10000,24)
n = 0
while n < 10000
    x = samp()
    if norm(f.(x[5:end],θ=x[1:4]) .- dat20) < 8.0
        n += 1
        println(n)
        TrueX[n,:] = x
    end
end

Σ = cov(TrueX)
d = mean(mapslices(norm,rand(MultivariateNormal(zeros(24),Σ),10000),dims=1))

R = BPS.BPS1(50000,TrueX[1,:],0.5,0.5,y=dat20,ϵ=8.0,Σ=1/d^2*Σ)