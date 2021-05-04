include("main.jl")
i
R = BPS.BPS_SMC_ABC(1000,50,dat20,Threshold=0.8,δ=0.5,κ=0.5,K0=10)

using Plots,StatsPlots, JLD2

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-UnorderNormMetric/RW_Thres075_Step01.jld2"
RW_Thres075_Step01 = R_RW
@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-UnorderNormMetric/RW_Thres080_Step01.jld2"
RW_Thres080_Step01 = R_RW

RandomWalk.Epsilon(RW_Thres075_Step01,color=:red,label="Threshold=0.75")
RandomWalk.Epsilon(RW_Thres080_Step01,color=:darkolivegreen,new=false)
RandomWalk.UniqueParticle(R_RW)

n = 4
RandomWalk.epdf(RW_Thres080_Step01,n,200,color=:darkolivegreen,figsize=(400,400))
density!(GroundTruth[50001:end,n])

GroundTruth,α = RWM(100000,Σ,0.2,y=dat20,θ0=rand(Uniform(0,10),4))

plot(GroundTruth[:,4])