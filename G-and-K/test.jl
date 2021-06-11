using Distributions
using Distributions, Plots, StatsPlots
using ForwardDiff: gradient
using Random

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);

include("BPS/ExactBPS-SMC-ABC.jl")
include("BPS/BPS-SMC-ABC.jl")



R = BPS.SMC(2000,250,dat20,Threshold=0.8,δ=0.3,κ=2.0,K0=2,MH=BPS.BPS1,Dist=BPS.Dist2)

index = findall(R.WEIGHT[:,end] .> 0)
density(R.U[4,index,end])


include("RandomWalk/RW-SMC-ABC.jl")

R = RandomWalk.SMC(1000,250,dat20,Threshold=0.8,δ=0.3,K0=5)

index = findall(R.WEIGHT[:,end] .> 0)
density(R.U[2,index,end])

x0 = R.U[:,1,end]

ExactBPS.Dist2(x0,y=dat20)

X,acc,b = ExactBPS.BPS(50000,x0,0.5,2.0,y=dat20,ϵ=0.4,Dist=ExactBPS.Dist2)
plot(X[:,2])    

X2,acc,b = ExactBPS.BPS(100000,x0,0.1,2.0,y=dat20,ϵ=0.4,Dist=ExactBPS.Dist2)
plot(X[:,1])

plot(X2[:,4])
density(X2[:,1])
density!(R[400001:end,1],color=:red,label="",linewidth=2)

X,_,_,_ = BPS.FullBPS(200000,x0,0.05,2.0,y=dat20,ϵ=0.4,Dist=BPS.Dist2)
X2,_,_,_ = BPS.FullBPS(20000,x0,0.05,2.0,y=dat20,ϵ=0.4,Dist=BPS.Dist2,MaxBounce=500)
plot(X[:,1])
density(X[:,4])

u0  = rand(Normal(0,1),24)
x0 .+ 0.1

v = rand(Normal(0,1),24,100000)
density((mapslices(norm,v,dims=1)[1,:]))
