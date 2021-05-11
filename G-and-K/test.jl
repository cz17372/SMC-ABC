include("main.jl")

R = BPS.BPS_SMC_ABC(1000,50,dat20,Threshold=0.8,δ=0.5,κ=0.5,K0=10)
using Plots,StatsPlots, JLD2, LinearAlgebra

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-UnorderNormMetric/100obs/RW_Thres075_Step015_100.jld2"
RW_Thres080_Step015_100 = R_RW

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-OrderNormMetric/100Particles/RW_Thres070_Step01_100.jld2"
RW_Thres080_Step010_100_Ordered = R_RW

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/100Data/RandomWalk/Summary/RW_Thres090_Step010.jld2"
RW_Summary_Thres090_Step010 = R_RW
RandomWalk.Epsilon(RW_Thres080_Step010_100_Ordered,color=:darkolivegreen,label="Metric=Ordered-Euclidean")
RandomWalk.Epsilon(RW_Thres080_Step015_100,color=:red,label="Metric = Standard Euclidean",new=false)

n = 4
RandomWalk.epdf(RW_Summary_Thres090_Step010,n,100,color=:darkolivegreen,xlabel="a",label="metric=summary")
RandomWalk.epdf(RW_Thres080_Step010_100_Ordered,n,100,color=:red,label="metric=ordered euclidean",new=false)
RandomWalk.epdf(RW_Thres080_Step015_100,n,100,color=:purple,label="metric=standard euclidean",new=false)
density!(MCMCRES[80001:end,n],color=:darkgrey,label="MCMC")

include("MCMC/MCMC.jl")
MCMCRES,alpha = RWM(100000,Σ,0.5,y=dat20,θ0 = rand(Uniform(0,10),4))
plot(MCMCRES[:,4],color=:darkgrey)
Σ = cov(MCMCRES[80001:end,:])

Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);
R = BPS.BPS_SMC_ABC(1000,100,dat20,Threshold=0.8,δ=0.5,κ=0.5,K0=10)

plot(R.RejectedBoundary,xlabel="iteration",title="Boundary Reflection",ylabel="average log-density ratio of velocity",size=(600,600))
plot(R.RejectedEnergy,xlabel="iteration",title="Energy Reflection",ylabel="average log-density ratio of velocity",size=(600,600))
plot(R.EnergyBounceProposed,label="proposed",title="Energy Reflection");plot!(R.EnergyBounceAccepted,label="accepted",size=(600,400))
plot(R.BoundaryBounceProposed,label="proposed",title="Boundary Reflection");plot!(R.BoundaryBounceAccepted,label="accepted",size=(600,400))

plot(R.BoundaryBounceAccepted./R.BoundaryBounceProposed)
using Plots