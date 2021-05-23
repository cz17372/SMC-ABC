using Plots,StatsPlots, JLD2, LinearAlgebra
include("main.jl")
@load "C:/Users/chang/OneDrive/Documents/BPS2_Thres080_Stepsize045_Refresh060.jld2"
@load "C:/Users/chang/OneDrive/Documents/BPS2_Thres080_Stepsize050_Refresh060_Adaptive.jld2"
R = BPS2_Thres080_Stepsize045_Refresh060
R2 = BPS2_Thres080_Stepsize050_Refresh060_Adaptive

ESS(x) = length(findall(x .> 0))

plot(log.(R.EPSILON))
index = findall(R.DISTANCE[:,end] .> 0)
density(R.U[2,index,end])
plot(R.BoundaryBounceProposed ./(mapslices(ESS,R.WEIGHT,dims=1)[1,2:end] .* R.K[1:end-1]),label="")

@load "MCMC/MCMC_COV.jld2"
include("MCMC/MCMC.jl")
MCMC,α = RWM(200000,Σ,0.2,y=dat20,θ0=rand(Uniform(0,10),4))
Σ = cov(MCMC[150001:end,:])

density(MCMC[150001:end,1])
density!(R2.U[1,index,end])

anim = @animate for i = 1:301
    index = findall(R2.WEIGHT[:,i] .> 0)
    p1 = density(R2.U[1,index,i],label="")
    density!(MCMC[50001:end,1],label="")
    p2 = density(R2.U[2,index,i],label="")
    density!(MCMC[50001:end,2],label="")
    p3 = density(R2.U[3,index,i],label="")
    density!(MCMC[50001:end,3],label="")
    p4 = density(R2.U[4,index,i],label="")
    density!(MCMC[50001:end,4],label="")
    plot(p1,p2,p3,p4,layout=(2,2),title="Iteration $(i)",size=(600,600))
end

gif(anim,fps=4)
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
index = findall(R.WEIGHT[:,end] .> 0)
Σ = cov(R.U[:,index,end],dims=2)
d = mean(mapslices(norm,rand(MultivariateNormal(zeros(24),Σ),10000),dims=1))
x = R.U[:,1,end]

R3 = BPS.BPS1(20000,R.U[:,1,end],0.03,exp(-2*0.03),y=dat20,ϵ=0.2)

R4 = BPS.SMC(1000,250,dat20,Threshold=0.8,δ=0.5,κ=3.0,K0=10,MH=BPS.BPS1)
R5 = BPS.SMC(1000,300,dat20,Threshold=0.8,δ=0.5,κ=3.0,K0=10,MH=BPS.BPS1)
index = findall(R4.WEIGHT[:,end] .> 0)
density(R4.U[1,index,end])
density!(MCMC[150001:end,3])
histogram(R4.U[3,index,end],bins=50,normalize=true) 
plot(R5.K)

dist(x) = norm(sort(f.(x[5:end],θ=x[1:4])) .- sort(dat20))

x = samp()

using ForwardDiff:gradient

dist(x)
x = x .- 0.2*normalize(gradient(dist,x))